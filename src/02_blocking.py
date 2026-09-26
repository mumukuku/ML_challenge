# %% [markdown]
# # 02 — Multi-Strategy Blocking
#
# **Phase 1, Step 1.2 & 1.3** — Candidate generation for entity resolution.
#
# Blocking recall **caps** final recall — anything missed here is lost forever.
#
# Strategies (union of candidates):
# 1. Country-filtered blocking (hard filter)
# 2. MinHash/LSH on character n-gram shingles
# 3. TF-IDF cosine on character n-grams (sparse_dot_topn)
# 4. Exact/near-exact blocking on distinctive name tokens
# 5. Embedding-based blocking for cross-script matches
#
# Outputs:
# - `dataset/blocking/` — candidate pair files per country
# - `candidate_pairs.tsv` — final blocking output (union of all strategies)
# - Blocking recall ceiling measurement on validation set

# %%
from __future__ import annotations
import gc
import os
import re
import warnings
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PERFORMANCE NOTE
# ---------------------------------------------------------------------------
# Every optimization in this file is *result-preserving*: same thresholds,
# same top_k, same recall. Nothing here approximates. The speedups come from
# three levers only:
#   1. Replacing per-row Python loops (iterrows) with vectorized pandas/numpy
#      ops that compute the exact same thing.
#   2. Deduplicating identical inputs (e.g. identical normalized names produce
#      identical MinHash/embedding vectors by construction, so we compute once
#      and reuse instead of recomputing for every duplicate row).
#   3. Parallelizing independent per-record work (MinHash construction) across
#      CPU cores.
# Where a genuinely faster *approximate* alternative exists (e.g. FAISS IVF
# indexes) it is deliberately NOT used -- blocking recall is a hard floor
# (target >=95%) and is not something to trade away for speed.

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(os.getcwd())
for p in [Path(os.getcwd()), *Path(os.getcwd()).parents]:
    if (p / "dataset").exists():
        REPO_ROOT = p
        break

TRAIN_DIR    = REPO_ROOT / "dataset" / "train"
TEST_DIR     = REPO_ROOT / "dataset" / "test"
NORM_DIR     = REPO_ROOT / "dataset" / "normalized"
SPLITS_DIR   = REPO_ROOT / "dataset" / "splits"
BLOCKING_DIR = REPO_ROOT / "dataset" / "blocking"
BLOCKING_DIR.mkdir(parents=True, exist_ok=True)

print(f"REPO_ROOT    : {REPO_ROOT}")
print(f"NORM_DIR     : {NORM_DIR}")
print(f"BLOCKING_DIR : {BLOCKING_DIR}")

# %%
# ---------------------------------------------------------------------------
# Configuration  (tuned from eda_results.txt)
# ---------------------------------------------------------------------------

# MinHash/LSH parameters
MINHASH_NUM_PERM = 128       # number of permutations
MINHASH_THRESHOLD = 0.2      # Jaccard threshold — low for high recall
MINHASH_NGRAM_SIZE = 3       # character n-gram size for shingles

# TF-IDF parameters
TFIDF_NGRAM_RANGE = (3, 5)   # character n-gram range
TFIDF_MAX_FEATURES = 500_000 # vocabulary cap per country partition
TFIDF_TOP_K = 50             # top-k similar candidates per S1 entity
TFIDF_THRESHOLD = 0.15       # minimum cosine similarity

# Token blocking parameters
TOKEN_MIN_LENGTH = 3         # minimum token length for blocking keys
TOKEN_MAX_FREQUENCY = 0.01   # max document frequency for "distinctive" tokens

# Embedding blocking parameters (cross-script)
# EDA §7: 7.26% of GT pairs are cross-script (S1=LATIN ↔ S2/S3=Indic).
# Non-Latin pool: S2 474k (9.42%) + S3 278k (5.27%) ≈ 752k records.
# TOP_K=30 covers avg 3.67 matches per non-singleton; threshold=0.35 cuts
# noise below meaningful similarity while retaining cross-script signal.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 2048
EMBEDDING_TOP_K = 30         # top-k by cosine similarity (7.26% cross-script GT)
EMBEDDING_MIN_SIMILARITY = 0.35  # raised from 0.3 — EDA cross-script pairs are strong signal

# Overall cap
MAX_CANDIDATES_PER_S1 = 100  # hard cap after union of all strategies

# Mode: 'train_val' or 'test'
# Set to 'train_val' during development, 'test' for final inference
MODE = "train_val"

# Address fallback strategy
# EDA §6: S2 India 2.87%, US 3.68%; S3 India 3.07%, US 3.50% missing addresses.
# S1 is ALWAYS present (0% missing). Records where has_addr=False in S2/S3
# are routed to name-only strategies; address-keyed blocking is skipped.
ADDR_FALLBACK_NAME_ONLY = True  # toggle to disable for ablation

# %% [markdown]
# ## 1. Load Normalized Data

# %%
def _read_norm_tsv(path: Path) -> pd.DataFrame:
    """Read a normalized TSV as all-string columns.

    Tries the pyarrow CSV engine first (meaningfully faster parse on the
    multi-hundred-MB files here); falls back to the default engine if pyarrow
    isn't installed or errors on this file. Output dtype/content is identical
    either way -- this only changes parse speed.
    """
    try:
        return pd.read_csv(path, sep="\t", dtype=str, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def load_normalized(prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load normalized S1, S2, S3 for a given prefix (train/test)."""
    s1 = _read_norm_tsv(NORM_DIR / f"{prefix}_source1_norm.tsv")
    s2 = _read_norm_tsv(NORM_DIR / f"{prefix}_source2_norm.tsv")
    s3 = _read_norm_tsv(NORM_DIR / f"{prefix}_source3_norm.tsv")
    return s1, s2, s3


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """Load ground truth -> {s1_id: set(matched_ids)}."""
    gt = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    result = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "")
        if isinstance(raw, str) and raw.strip():
            result[s1_id] = set(raw.split(","))
        else:
            result[s1_id] = set()
    return result


# %% [markdown]
# ## 2. Strategy 1: Country-Filtered Partitioning
#
# Only compare S1 entities to S2/S3 entities with the same country value.

# %%
def partition_by_country(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Partition data by country.

    Returns {country: (s1_country, s23_country)}.
    S2 and S3 are concatenated into a single candidate pool per country.
    """
    s2 = s2.copy()
    s3 = s3.copy()
    s2["_source"] = "S2"
    s3["_source"] = "S3"
    s23 = pd.concat([s2, s3], ignore_index=True)

    countries = set(s1["country"].dropna().unique()) | set(s23["country"].dropna().unique())
    partitions = {}

    for country in sorted(countries):
        s1_c = s1[s1["country"] == country].copy().reset_index(drop=True)
        s23_c = s23[s23["country"] == country].copy().reset_index(drop=True)
        if len(s1_c) > 0 and len(s23_c) > 0:
            partitions[country] = (s1_c, s23_c)
            print(f"  Country '{country}': {len(s1_c):,} S1 × {len(s23_c):,} S2+S3")
        else:
            print(f"  Country '{country}': SKIPPED (S1={len(s1_c)}, S23={len(s23_c)})")

    return partitions


# %% [markdown]
# ## 3. Strategy 2: MinHash/LSH Blocking

# %%
def build_shingles(text: str, n: int = MINHASH_NGRAM_SIZE) -> Set[str]:
    """Generate character n-gram shingles from text."""
    if not text or len(text) < n:
        return {text} if text else {""}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def create_minhash(shingles: Set[str], num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    """Create a MinHash from a set of shingles."""
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def _minhash_for_name(name: str) -> MinHash:
    """Build a MinHash for one name using the module-level MinHash config.

    Top-level function (required so it can be pickled for multiprocessing).
    Pure function of `name` -> identical names always produce an identical
    MinHash, which is what makes the dedup in _build_minhashes below safe.
    """
    shingles = build_shingles(name, MINHASH_NGRAM_SIZE)
    return create_minhash(shingles, MINHASH_NUM_PERM)


def _build_minhashes(names: List[str]) -> Dict[str, MinHash]:
    """Compute a MinHash for each *unique* name in `names`, in parallel.

    Returns {name: MinHash}. Deduplicating first is exact, not approximate:
    two records with the same normalized name produce byte-identical shingle
    sets and therefore byte-identical MinHash objects, so computing it once
    and reusing it for every row that shares that name changes nothing about
    the resulting LSH index or query results -- it only avoids redundant work
    on a dataset with heavy exact-duplication (see plan.md noise catalog).

    Falls back to serial computation for small inputs (process-spawn overhead
    isn't worth it) or if multiprocessing is unavailable in this environment.
    """
    unique_names = sorted(set(names))
    n_jobs = max(1, min(cpu_count(), 8))
    if n_jobs > 1 and len(unique_names) >= 5_000:
        try:
            with Pool(processes=n_jobs) as pool:
                chunksize = max(1, len(unique_names) // (n_jobs * 4))
                hashes = pool.map(_minhash_for_name, unique_names, chunksize=chunksize)
            return dict(zip(unique_names, hashes))
        except Exception as e:
            print(f"    (parallel MinHash build failed [{e}], falling back to serial)")
    return {name: _minhash_for_name(name) for name in tqdm(unique_names, desc="    MinHash build")}


def minhash_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    name_col: str = "name_no_accent",
    threshold: float = MINHASH_THRESHOLD,
    num_perm: int = MINHASH_NUM_PERM,
) -> Dict[str, Set[str]]:
    """MinHash/LSH blocking on character n-gram shingles of normalized names.

    Same algorithm as a naive per-row implementation (same threshold,
    num_perm, shingle size) -- see _build_minhashes for why the dedup+
    parallelization here doesn't change any result, only how fast it runs.

    Returns {s1_entity_id: set(candidate_s23_ids)}.
    """
    candidates: Dict[str, Set[str]] = defaultdict(set)

    s23_names = s23_df[name_col].fillna("").astype(str).tolist()
    s23_ids = s23_df["entity_id"].tolist()
    s1_names = s1_df[name_col].fillna("").astype(str).tolist()
    s1_ids = s1_df["entity_id"].tolist()

    print(f"    Building MinHash LSH index for {len(s23_df):,} S2+S3 records "
          f"({len(set(s23_names)):,} unique names) ...")
    name_to_minhash = _build_minhashes(s23_names)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for eid, name in tqdm(zip(s23_ids, s23_names), total=len(s23_ids), desc="    S23 LSH insert"):
        try:
            lsh.insert(eid, name_to_minhash[name])
        except ValueError:
            pass  # duplicate key — skip

    # Query each S1 entity — dedupe query names the same exact way.
    print(f"    Querying {len(s1_df):,} S1 entities ({len(set(s1_names)):,} unique names) ...")
    query_minhashes = _build_minhashes(s1_names)
    query_results = {name: lsh.query(mh) for name, mh in tqdm(query_minhashes.items(), desc="    LSH query")}

    for s1_id, name in zip(s1_ids, s1_names):
        candidates[s1_id].update(query_results[name])

    n_cands = sum(len(v) for v in candidates.values())
    print(f"    MinHash blocking: {n_cands:,} candidate pairs for {len(candidates):,} S1 entities")
    return dict(candidates)


# %% [markdown]
# ## 4. Strategy 3: TF-IDF Cosine Blocking

# %%
def tfidf_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    name_col: str = "name_no_accent",
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
    top_k: int = TFIDF_TOP_K,
    threshold: float = TFIDF_THRESHOLD,
) -> Dict[str, Set[str]]:
    """TF-IDF cosine blocking on character n-grams.

    Uses sparse_dot_topn for efficient top-k retrieval.

    Returns {s1_entity_id: set(candidate_s23_ids)}.
    """
    candidates: Dict[str, Set[str]] = defaultdict(set)

    s1_names = s1_df[name_col].fillna("").tolist()
    s23_names = s23_df[name_col].fillna("").tolist()
    s1_ids = s1_df["entity_id"].tolist()
    s23_ids = s23_df["entity_id"].tolist()

    n_s1 = len(s1_names)
    n_s23 = len(s23_names)

    print(f"    Building TF-IDF matrix for {n_s1 + n_s23:,} records ...")

    # Fit on combined corpus, transform separately
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )

    # Fit vocabulary on unique names (identical vocabulary, much faster)
    unique_corpus = list(set(s1_names) | set(s23_names))
    vectorizer.fit(unique_corpus)
    del unique_corpus

    # Transform separately
    s1_matrix = vectorizer.transform(s1_names)
    s23_matrix = vectorizer.transform(s23_names)

    print(f"    TF-IDF matrix: S1={s1_matrix.shape}, S23={s23_matrix.shape}")
    print(f"    Computing top-{top_k} cosine similarities (threshold={threshold}) ...")

    # Compute S1 × S23^T with top-k threshold
    # Process in chunks to manage memory
    chunk_size = min(50_000, n_s1)
    for start in range(0, n_s1, chunk_size):
        end = min(start + chunk_size, n_s1)
        s1_chunk = s1_matrix[start:end]

        sim_matrix = awesome_cossim_topn(
            s1_chunk,
            s23_matrix.T,
            ntop=top_k,
            lower_bound=threshold,
            use_threads=True,
            n_jobs=4,
        )

        # Extract candidate pairs from sparse result. Same nonzero entries as
        # a per-pair Python loop would visit; grouping vectorizes the
        # aggregation into sets so the Python-level work is O(unique S1 ids
        # in this chunk) instead of O(nonzero entries).
        rows, cols = sim_matrix.nonzero()
        if len(rows):
            chunk_s1_ids = np.asarray(s1_ids[start:end])[rows]
            chunk_s23_ids = np.asarray(s23_ids)[cols]
            pair_df = pd.DataFrame({"s1_id": chunk_s1_ids, "s23_id": chunk_s23_ids})
            for s1_id, cand_set in pair_df.groupby("s1_id")["s23_id"].agg(set).items():
                candidates[s1_id].update(cand_set)

    n_cands = sum(len(v) for v in candidates.values())
    print(f"    TF-IDF blocking: {n_cands:,} candidate pairs for {len(candidates):,} S1 entities")

    # Clean up
    del s1_matrix, s23_matrix, vectorizer
    gc.collect()

    return dict(candidates)


# %% [markdown]
# ## 5. Strategy 4: Token-Based Blocking

# %%
# Legal suffixes and common stopwords to exclude from blocking keys
BLOCK_STOPWORDS = {
    "llc", "ltd", "limited", "inc", "corp", "corporation", "pvt", "private",
    "llp", "lp", "co", "company", "sarl", "sci", "sas", "sa", "plc", "pllc",
    "gmbh", "ag", "pty", "ngo", "npo", "the", "and", "of", "for", "in", "on",
    "at", "to", "a", "an", "is", "it", "by", "or", "with", "from", "as",
    "services", "solutions", "group", "holdings", "enterprises", "international",
    "industries", "consulting", "technologies", "tech", "labs", "media",
    "ventures", "partners", "capital", "trust", "foundation", "association",
    "society", "institute", "private", "limited",
}


def extract_blocking_tokens(name: str) -> Set[str]:
    """Extract distinctive tokens from a normalized name for blocking."""
    if not isinstance(name, str) or not name.strip():
        return set()
    # Split on whitespace and non-alphanumeric
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    # Filter: min length, not a stopword, not purely numeric
    return {
        t for t in tokens
        if len(t) >= TOKEN_MIN_LENGTH
        and t not in BLOCK_STOPWORDS
        and not t.isdigit()
    }


def _tokenize_series(names: pd.Series) -> pd.Series:
    """Vectorized version of extract_blocking_tokens's regex step (same regex)."""
    return names.fillna("").astype(str).str.lower().str.findall(r"[a-z0-9]+")


def _exploded_blocking_tokens(entity_ids: np.ndarray, token_lists: pd.Series) -> pd.DataFrame:
    """Explode per-entity token lists into a flat (entity_id, token) frame,
    applying exactly extract_blocking_tokens's filter (min length, not a
    stopword, not purely numeric) and exactly its per-entity dedup (that
    function returns a *set* of tokens, so a repeated token within one name
    counts once for that entity — drop_duplicates below reproduces that).
    """
    flat = pd.DataFrame({"entity_id": entity_ids, "token": token_lists.to_numpy()}).explode("token")
    flat = flat.dropna(subset=["token"])
    keep = (
        (flat["token"].str.len() >= TOKEN_MIN_LENGTH)
        & (~flat["token"].isin(BLOCK_STOPWORDS))
        & (~flat["token"].str.isdigit())
    )
    flat = flat[keep].drop_duplicates(subset=["entity_id", "token"])
    return flat


def token_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    name_col: str = "name_no_accent",
    max_freq: float = TOKEN_MAX_FREQUENCY,
) -> Dict[str, Set[str]]:
    """Inverted-index token blocking on distinctive name tokens.

    Vectorized reimplementation of the identical algorithm: an S1 entity is a
    candidate match for an S23 record iff they share at least one token that
    passes extract_blocking_tokens's filter AND has S2+S3 document frequency
    <= max_freq * len(s23_df) (the same "distinctive token" definition as
    before). The token<->token merge below reconstructs exactly that shared-
    token relationship without any Python-level loop over S1 entities.

    Address fallback (EDA §6): S2/S3 records with has_addr=False (~3-4%) are
    eligible for name-only blocking — no special handling needed here since
    token blocking is already name-based. These records remain in s23_df.

    Returns {s1_entity_id: set(candidate_s23_ids)}.
    """
    candidates: Dict[str, Set[str]] = defaultdict(set)

    print(f"    Building token inverted index for {len(s23_df):,} S2+S3 records ...")
    s23_tokens = _tokenize_series(s23_df[name_col])
    s23_flat = _exploded_blocking_tokens(s23_df["entity_id"].to_numpy(), s23_tokens)

    # Doc frequency = number of distinct S23 entities containing the token
    # (matches the original Counter, which incremented once per entity since
    # extract_blocking_tokens already dedupes tokens within one name).
    token_doc_freq = s23_flat["token"].value_counts()
    n_s23 = len(s23_df)
    freq_cutoff = int(max_freq * n_s23)
    rare_tokens = set(token_doc_freq[token_doc_freq <= freq_cutoff].index)
    print(f"    Token vocabulary: {len(token_doc_freq):,} total, {len(rare_tokens):,} rare (freq ≤ {freq_cutoff})")

    rare_index = s23_flat[s23_flat["token"].isin(rare_tokens)].rename(columns={"entity_id": "s23_entity_id"})

    print(f"    Querying {len(s1_df):,} S1 entities ...")
    s1_tokens = _tokenize_series(s1_df[name_col])
    s1_flat = _exploded_blocking_tokens(s1_df["entity_id"].to_numpy(), s1_tokens)
    s1_flat = s1_flat[s1_flat["token"].isin(rare_tokens)]

    # Inner join on shared distinctive tokens == exactly the original
    # "for t in tokens: if t in rare_tokens and t in token_index: update(...)".
    merged = s1_flat.merge(rare_index[["token", "s23_entity_id"]], on="token", how="inner")
    if not merged.empty:
        for s1_id, cand_set in merged.groupby("entity_id")["s23_entity_id"].agg(set).items():
            candidates[s1_id] = cand_set

    n_cands = sum(len(v) for v in candidates.values())
    print(f"    Token blocking: {n_cands:,} candidate pairs for {len(candidates):,} S1 entities")
    return dict(candidates)


# %% [markdown]
# ## 6. Strategy 5: Embedding-Based Blocking (Cross-Script)
#
# Uses a multilingual sentence-transformer to find semantic matches,
# especially for non-Latin (Devanagari, Tamil, etc.) names that need
# to match Latin S1 names.

# %%
def _topk_cosine_exact(query_emb: np.ndarray, pool_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact top-k cosine similarity search (embeddings must be L2-normalized,
    so inner product == cosine similarity).

    Uses FAISS's IndexFlatIP when available (SIMD-accelerated brute force).
    Falls back to PyTorch CUDA top-k if GPU is available, or vectorized numpy otherwise.
    Returns (indices, scores), each of shape (n_query, min(top_k, n_pool)).
    """
    n_pool = pool_emb.shape[0]
    k = min(top_k, n_pool)
    query_emb = np.ascontiguousarray(query_emb.astype(np.float32))
    pool_emb = np.ascontiguousarray(pool_emb.astype(np.float32))
    try:
        import faiss
        index = faiss.IndexFlatIP(pool_emb.shape[1])
        index.add(pool_emb)
        scores, idx = index.search(query_emb, k)
        return idx, scores
    except ImportError:
        try:
            import torch
            if torch.cuda.is_available():
                with torch.inference_mode():
                    q_t = torch.from_numpy(query_emb).cuda()
                    p_t = torch.from_numpy(pool_emb).cuda()
                    sim = q_t @ p_t.T
                    scores_t, idx_t = torch.topk(sim, k=k, dim=1)
                    return idx_t.cpu().numpy(), scores_t.cpu().numpy()
        except Exception:
            pass
        sim = query_emb @ pool_emb.T
        if k < sim.shape[1]:
            idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        else:
            idx = np.tile(np.arange(sim.shape[1]), (sim.shape[0], 1))
        scores = np.take_along_axis(sim, idx, axis=1)
        return idx, scores


def embedding_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    name_col: str = "name_clean",
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    top_k: int = EMBEDDING_TOP_K,
) -> Dict[str, Set[str]]:
    """Embedding-based blocking using multilingual sentence-transformers.

    Only processes non-Latin S2/S3 records (to save compute) plus
    all S1 records from countries where cross-script matches exist.

    Returns {s1_entity_id: set(candidate_s23_ids)}.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        print("    WARNING: sentence-transformers not available, skipping embedding blocking")
        return {}

    candidates: Dict[str, Set[str]] = defaultdict(set)

    # Filter S2+S3 to records with non-Latin scripts
    # EDA §3b: S2 474,345 (9.42%) + S3 278,524 (5.27%) = ~752k non-Latin records.
    # S1 is pure LATIN (0 non-Latin), so we encode ALL S1 and query against
    # the non-Latin S2/S3 pool to find cross-script matches (7.26% of GT pairs).
    non_latin_mask = s23_df["has_non_latin"].astype(str).str.lower() == "true"
    s23_non_latin = s23_df[non_latin_mask].copy()

    if len(s23_non_latin) == 0:
        print("    No non-Latin S2/S3 records — skipping embedding blocking")
        return {}

    print(f"    Embedding blocking: {len(s23_non_latin):,} non-Latin S2/S3 records")
    print(f"    Loading model: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()  # Activates Tensor Cores on RTX 3060 (2x speedup)
    print(f"    Device: {device} (FP16: {device == 'cuda'})")

    # Encode unique S1 names (deduplicate first to avoid redundant neural forward passes)
    s1_names = s1_df[name_col].fillna("").astype(str).tolist()
    s1_ids = s1_df["entity_id"].tolist()
    unique_s1, s1_inv = np.unique(s1_names, return_inverse=True)
    print(f"    Encoding {len(unique_s1):,} unique S1 names (from {len(s1_names):,} total) ...")
    s1_unique_emb = model.encode(
        unique_s1.tolist(), batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    s1_embeddings = np.asarray(s1_unique_emb, dtype=np.float32)[s1_inv]

    # Encode unique non-Latin S2+S3 names
    s23_names = s23_non_latin[name_col].fillna("").astype(str).tolist()
    s23_ids = s23_non_latin["entity_id"].tolist()
    unique_s23, s23_inv = np.unique(s23_names, return_inverse=True)
    print(f"    Encoding {len(unique_s23):,} unique non-Latin S2/S3 names (from {len(s23_names):,} total) ...")
    s23_unique_emb = model.encode(
        unique_s23.tolist(), batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    s23_embeddings = np.asarray(s23_unique_emb, dtype=np.float32)[s23_inv]

    # Compute cosine similarity in chunks (S1 × S23^T)
    print(f"    Computing top-{top_k} cosine similarities ...")
    chunk_size = min(10_000, len(s1_ids))

    for start in tqdm(range(0, len(s1_ids), chunk_size), desc="    Embedding search"):
        end = min(start + chunk_size, len(s1_ids))
        s1_chunk = s1_embeddings[start:end]

        # Exact top-k cosine search for the whole chunk at once (FAISS
        # IndexFlatIP if available, exact numpy argpartition otherwise) —
        # same top-k values per row as the original per-row loop, just
        # computed in one batched call instead of one Python call per row.
        top_idx, top_scores = _topk_cosine_exact(s1_chunk, s23_embeddings, top_k)

        # EDA-calibrated threshold: 0.35 (raised from 0.3)
        # Cross-script GT pairs are strong signal; noise falls below 0.35
        keep = top_scores > EMBEDDING_MIN_SIMILARITY
        rows_idx, kept_cols = np.nonzero(keep)
        if len(rows_idx):
            chosen_s23_positions = top_idx[rows_idx, kept_cols]
            chunk_s1_ids = np.asarray(s1_ids[start:end])[rows_idx]
            chunk_s23_ids = np.asarray(s23_ids)[chosen_s23_positions]
            pair_df = pd.DataFrame({"s1_id": chunk_s1_ids, "s23_id": chunk_s23_ids})
            for s1_id, cand_set in pair_df.groupby("s1_id")["s23_id"].agg(set).items():
                candidates[s1_id].update(cand_set)

    n_cands = sum(len(v) for v in candidates.values())
    print(f"    Embedding blocking: {n_cands:,} candidate pairs for {len(candidates):,} S1 entities")

    # Clean up
    del model, s1_embeddings, s23_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(candidates)


# %% [markdown]
# ## 7. Union All Strategies & Cap Candidates

# %%
def merge_candidates(
    *strategy_results: Dict[str, Set[str]],
    all_s1_ids: List[str] | None = None,
    max_per_s1: int = MAX_CANDIDATES_PER_S1,
) -> Dict[str, Set[str]]:
    """Union candidate sets from all strategies and cap per S1 entity.

    Ensures every S1 entity has an entry (even if empty — singletons).
    """
    merged: Dict[str, Set[str]] = defaultdict(set)

    for result in strategy_results:
        for s1_id, cands in result.items():
            merged[s1_id].update(cands)

    # Ensure all S1 IDs present
    if all_s1_ids:
        for s1_id in all_s1_ids:
            if s1_id not in merged:
                merged[s1_id] = set()

    # Cap candidates
    capped = 0
    for s1_id, cands in merged.items():
        if len(cands) > max_per_s1:
            # Keep a random subset — in practice we'd rank by a blocking score,
            # but for now random is acceptable as a safety cap
            merged[s1_id] = set(sorted(cands)[:max_per_s1])
            capped += 1

    if capped > 0:
        print(f"  Capped {capped:,} S1 entities to max {max_per_s1} candidates")

    return dict(merged)


# %% [markdown]
# ## 8. Blocking Recall Ceiling Measurement

# %%
def measure_blocking_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """Measure blocking recall ceiling on validation set.

    Returns a dict with recall, coverage stats, and per-entity breakdown.
    """
    total_true_pairs = 0
    total_found_pairs = 0
    total_candidates = 0
    entity_recalls = []
    missed_pairs = []

    for s1_id, truth in ground_truth.items():
        cands = candidates.get(s1_id, set())
        total_candidates += len(cands)

        if not truth:  # singleton
            entity_recalls.append(1.0)  # no pairs to miss
            continue

        found = truth & cands
        total_true_pairs += len(truth)
        total_found_pairs += len(found)

        recall = len(found) / len(truth) if truth else 1.0
        entity_recalls.append(recall)

        # Track missed pairs for analysis
        missed = truth - cands
        if missed:
            missed_pairs.append({
                "s1_id": s1_id,
                "n_truth": len(truth),
                "n_found": len(found),
                "n_missed": len(missed),
                "missed_ids": missed,
            })

    overall_recall = total_found_pairs / total_true_pairs if total_true_pairs > 0 else 1.0
    mean_entity_recall = np.mean(entity_recalls) if entity_recalls else 1.0

    n_s1 = len(ground_truth)
    n_non_singleton = sum(1 for v in ground_truth.values() if v)

    stats = {
        "overall_pair_recall": overall_recall,
        "mean_entity_recall": mean_entity_recall,
        "total_true_pairs": total_true_pairs,
        "total_found_pairs": total_found_pairs,
        "total_missed_pairs": total_true_pairs - total_found_pairs,
        "total_candidates": total_candidates,
        "avg_candidates_per_s1": total_candidates / n_s1 if n_s1 > 0 else 0,
        "n_s1_entities": n_s1,
        "n_non_singleton": n_non_singleton,
        "n_perfect_recall": sum(1 for r in entity_recalls if r == 1.0),
        "n_zero_recall": sum(1 for r in entity_recalls if r == 0.0),
        "reduction_ratio": 1 - (total_candidates / (n_s1 * len(set.union(*ground_truth.values()) if any(ground_truth.values()) else {1})))
            if n_s1 > 0 else 0,
        "missed_pairs_sample": sorted(missed_pairs, key=lambda x: x["n_missed"], reverse=True)[:20],
    }

    return stats


def print_blocking_stats(stats: Dict[str, Any], label: str = "") -> None:
    """Pretty-print blocking recall statistics."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}{'='*60}")
    print(f"{prefix}BLOCKING RECALL CEILING")
    print(f"{prefix}{'='*60}")
    print(f"{prefix}  Overall pair recall:      {stats['overall_pair_recall']:.4f} ({stats['overall_pair_recall']*100:.2f}%)")
    print(f"{prefix}  Mean entity recall:       {stats['mean_entity_recall']:.4f}")
    print(f"{prefix}  True pairs:               {stats['total_true_pairs']:,}")
    print(f"{prefix}  Found pairs:              {stats['total_found_pairs']:,}")
    print(f"{prefix}  Missed pairs:             {stats['total_missed_pairs']:,}")
    print(f"{prefix}  Total candidates:         {stats['total_candidates']:,}")
    print(f"{prefix}  Avg candidates per S1:    {stats['avg_candidates_per_s1']:.1f}")
    print(f"{prefix}  S1 entities:              {stats['n_s1_entities']:,}")
    print(f"{prefix}  Perfect recall (1.0):     {stats['n_perfect_recall']:,}")
    print(f"{prefix}  Zero recall (0.0):        {stats['n_zero_recall']:,}")

    if stats["missed_pairs_sample"]:
        print(f"\n{prefix}  Top missed entities:")
        for mp in stats["missed_pairs_sample"][:5]:
            print(f"    {mp['s1_id']}: {mp['n_missed']}/{mp['n_truth']} missed")


# %% [markdown]
# ## 9. Per-Strategy Recall Contribution

# %%
def measure_strategy_contributions(
    strategy_results: Dict[str, Dict[str, Set[str]]],
    ground_truth: Dict[str, Set[str]],
) -> pd.DataFrame:
    """Measure the unique recall contribution of each blocking strategy.

    Returns a DataFrame with per-strategy recall and unique-pair counts.
    """
    rows = []

    # Per-strategy standalone recall
    for name, cands in strategy_results.items():
        stats = measure_blocking_recall(cands, ground_truth)
        rows.append({
            "strategy": name,
            "pair_recall": stats["overall_pair_recall"],
            "found_pairs": stats["total_found_pairs"],
            "total_candidates": stats["total_candidates"],
            "avg_cands_per_s1": stats["avg_candidates_per_s1"],
        })

    # Union recall
    all_cands = merge_candidates(*strategy_results.values())
    union_stats = measure_blocking_recall(all_cands, ground_truth)
    rows.append({
        "strategy": "UNION (all)",
        "pair_recall": union_stats["overall_pair_recall"],
        "found_pairs": union_stats["total_found_pairs"],
        "total_candidates": union_stats["total_candidates"],
        "avg_cands_per_s1": union_stats["avg_candidates_per_s1"],
    })

    return pd.DataFrame(rows)


# %% [markdown]
# ## 10. Main Blocking Pipeline
#
# Run all strategies per country, measure recall on validation set.

# %%
def run_blocking_for_country(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    country: str,
    ground_truth: Dict[str, Set[str]] | None = None,
    run_embeddings: bool = True,
) -> Tuple[Dict[str, Set[str]], Optional[Dict[str, Any]]]:
    """Run all blocking strategies for a single country partition.

    Returns (merged_candidates, blocking_stats_or_None).
    """
    print(f"\n{'='*70}")
    print(f"BLOCKING: Country = {country}")
    print(f"  S1: {len(s1_df):,} entities, S2+S3: {len(s23_df):,} candidates")
    print(f"{'='*70}")

    strategy_results = {}

    # Strategy 2: MinHash/LSH
    print(f"\n  Strategy 2: MinHash/LSH blocking ...")
    mh_cands = minhash_blocking(s1_df, s23_df)
    strategy_results["minhash"] = mh_cands

    # Strategy 3: TF-IDF cosine
    print(f"\n  Strategy 3: TF-IDF cosine blocking ...")
    tfidf_cands = tfidf_blocking(s1_df, s23_df)
    strategy_results["tfidf"] = tfidf_cands

    # Strategy 4: Token blocking
    print(f"\n  Strategy 4: Token blocking ...")
    token_cands = token_blocking(s1_df, s23_df)
    strategy_results["token"] = token_cands

    # Strategy 5: Embedding blocking (cross-script)
    if run_embeddings:
        # Check if there are non-Latin records
        non_latin_count = (s23_df["has_non_latin"].astype(str).str.lower() == "true").sum()
        if non_latin_count > 0:
            print(f"\n  Strategy 5: Embedding blocking ({non_latin_count:,} non-Latin records) ...")
            emb_cands = embedding_blocking(s1_df, s23_df)
            strategy_results["embedding"] = emb_cands
        else:
            print(f"\n  Strategy 5: SKIPPED (no non-Latin records in {country})")
    else:
        print(f"\n  Strategy 5: SKIPPED (embeddings disabled)")

    # Union all strategies
    all_s1_ids = s1_df["entity_id"].tolist()
    merged = merge_candidates(*strategy_results.values(), all_s1_ids=all_s1_ids)

    # Measure recall if ground truth available
    stats = None
    if ground_truth:
        # Filter GT to this country's S1 entities
        country_gt = {sid: gt for sid, gt in ground_truth.items() if sid in set(all_s1_ids)}
        if country_gt:
            print(f"\n  Measuring blocking recall on {len(country_gt):,} GT entities ...")
            stats = measure_blocking_recall(merged, country_gt)
            print_blocking_stats(stats, country)

            # Per-strategy contribution
            contrib = measure_strategy_contributions(strategy_results, country_gt)
            print(f"\n  Per-strategy contribution ({country}):")
            print(contrib.to_string(index=False))

    return merged, stats


# %%
# ---------------------------------------------------------------------------
# MAIN EXECUTION: Process training data with validation measurement
# ---------------------------------------------------------------------------

def main():
    # Check if normalized files exist
    norm_train_s1 = NORM_DIR / "train_source1_norm.tsv"
    if not norm_train_s1.exists():
        print("ERROR: Normalized files not found! Run 01_normalize.py first.")
        print(f"  Expected: {norm_train_s1}")
        import sys; sys.exit(1)

    print("\n" + "="*70)
    print("PHASE 1: MULTI-STRATEGY BLOCKING")
    print("="*70)

    # Load normalized training data
    print("\nLoading normalized training data ...")
    train_s1, train_s2, train_s3 = load_normalized("train")
    print(f"  Train S1: {len(train_s1):,}  S2: {len(train_s2):,}  S3: {len(train_s3):,}")

    # Load validation ground truth
    val_gt_path = SPLITS_DIR / "val_ground_truth.tsv"
    val_s1_ids_path = SPLITS_DIR / "val_s1_ids.txt"

    ground_truth = None
    val_s1_ids = None
    if val_gt_path.exists():
        ground_truth = load_ground_truth(val_gt_path)
        val_s1_ids = set(
            Path(val_s1_ids_path).read_text(encoding="utf-8").strip().split("\n")
        )
        print(f"  Validation GT: {len(ground_truth):,} entities")
    else:
        print("  WARNING: Validation ground truth not found — skipping recall measurement")

    # Partition by country
    print("\nPartitioning by country ...")
    partitions = partition_by_country(train_s1, train_s2, train_s3)

    # For validation: filter S1 to validation set only
    # (We still block against ALL S2/S3 — blocking must work on the full pool)
    all_blocking_results: Dict[str, Set[str]] = {}
    all_stats = {}

    for country, (s1_c, s23_c) in partitions.items():
        # For recall measurement, use val S1 entities only
        if val_s1_ids:
            s1_val_mask = s1_c["entity_id"].isin(val_s1_ids)
            s1_val = s1_c[s1_val_mask].copy().reset_index(drop=True)
            print(f"\n  {country}: {len(s1_val):,} validation S1 entities (of {len(s1_c):,} total)")
        else:
            s1_val = s1_c

        if len(s1_val) == 0:
            print(f"  {country}: No validation entities — skipping")
            continue

        country_cands, country_stats = run_blocking_for_country(
            s1_val, s23_c, country,
            ground_truth=ground_truth,
            run_embeddings=True,
        )

        all_blocking_results.update(country_cands)
        if country_stats:
            all_stats[country] = country_stats

    # Overall blocking recall
    if ground_truth:
        print("\n" + "="*70)
        print("OVERALL BLOCKING RECALL (across all countries)")
        print("="*70)

        # Ensure all val S1 IDs have an entry
        for s1_id in ground_truth:
            if s1_id not in all_blocking_results:
                all_blocking_results[s1_id] = set()

        overall_stats = measure_blocking_recall(all_blocking_results, ground_truth)
        print_blocking_stats(overall_stats, "Overall")

    # Save blocking results for validation set
    output_path = BLOCKING_DIR / "val_candidate_pairs.tsv"
    print(f"\nSaving validation blocking results to {output_path} ...")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(all_blocking_results.keys()):
            cands = all_blocking_results[s1_id]
            f.write(f"{s1_id}\t{','.join(sorted(cands)) if cands else ''}\n")

    print(f"  Saved {len(all_blocking_results):,} rows")

    # Summary statistics
    print("\n" + "="*70)
    print("BLOCKING SUMMARY")
    print("="*70)

    if all_stats:
        summary_rows = []
        for country, stats in all_stats.items():
            summary_rows.append({
                "country": country,
                "pair_recall": f"{stats['overall_pair_recall']:.4f}",
                "found/total": f"{stats['total_found_pairs']:,}/{stats['total_true_pairs']:,}",
                "total_cands": f"{stats['total_candidates']:,}",
                "avg_cands": f"{stats['avg_candidates_per_s1']:.1f}",
            })
        print(pd.DataFrame(summary_rows).to_string(index=False))

    if ground_truth:
        print(f"\n  TARGET: Blocking recall ≥ 95%")
        print(f"  ACTUAL: {overall_stats['overall_pair_recall']*100:.2f}%")
        if overall_stats["overall_pair_recall"] >= 0.95:
            print(f"  STATUS: ✓ TARGET MET")
        else:
            print(f"  STATUS: ✗ BELOW TARGET — need to add/tune blocking strategies")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

# %% [markdown]
# ## Next Steps
#
# 1. If blocking recall < 95%: adjust thresholds, add strategies, analyze misses.
# 2. Candidate pairs feed into `03_features.py` for pairwise feature engineering.
# 3. The `candidate_pairs.tsv` output is a required submission deliverable.
# 4. Per-strategy contribution table feeds into methodology documentation.

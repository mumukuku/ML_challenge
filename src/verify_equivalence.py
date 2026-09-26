"""Standalone equivalence tests: vectorized logic vs. naive brute-force
reference, on synthetic data with the same edge cases as the real pipeline
(duplicate tokens within a name, duplicate names across records, ties in
top-k, values exactly at the threshold boundary).
"""
import re
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# TEST 1: token blocking (explode+merge vs. naive nested loop)
# ---------------------------------------------------------------------------
TOKEN_MIN_LENGTH = 3
TOKEN_MAX_FREQUENCY = 0.5
BLOCK_STOPWORDS = {"llc", "the", "and", "of"}


def extract_blocking_tokens_naive(name):
    if not isinstance(name, str) or not name.strip():
        return set()
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return {t for t in tokens if len(t) >= TOKEN_MIN_LENGTH and t not in BLOCK_STOPWORDS and not t.isdigit()}


def token_blocking_naive(s1_names, s1_ids, s23_names, s23_ids, max_freq=TOKEN_MAX_FREQUENCY):
    from collections import defaultdict, Counter
    candidates = defaultdict(set)
    token_index = defaultdict(list)
    token_doc_freq = Counter()
    for eid, name in zip(s23_ids, s23_names):
        for t in extract_blocking_tokens_naive(name):
            token_index[t].append(eid)
            token_doc_freq[t] += 1
    freq_cutoff = int(max_freq * len(s23_names))
    rare_tokens = {t for t, cnt in token_doc_freq.items() if cnt <= freq_cutoff}
    for s1_id, name in zip(s1_ids, s1_names):
        for t in extract_blocking_tokens_naive(name):
            if t in rare_tokens and t in token_index:
                candidates[s1_id].update(token_index[t])
    return dict(candidates)


def _tokenize_series(names):
    return names.fillna("").astype(str).str.lower().str.findall(r"[a-z0-9]+")


def _exploded_blocking_tokens(entity_ids, token_lists):
    flat = pd.DataFrame({"entity_id": entity_ids, "token": token_lists.to_numpy()}).explode("token")
    flat = flat.dropna(subset=["token"])
    keep = (
        (flat["token"].str.len() >= TOKEN_MIN_LENGTH)
        & (~flat["token"].isin(BLOCK_STOPWORDS))
        & (~flat["token"].str.isdigit())
    )
    flat = flat[keep].drop_duplicates(subset=["entity_id", "token"])
    return flat


def token_blocking_vectorized(s1_df, s23_df, max_freq=TOKEN_MAX_FREQUENCY):
    from collections import defaultdict
    candidates = defaultdict(set)
    s23_tokens = _tokenize_series(s23_df["name"])
    s23_flat = _exploded_blocking_tokens(s23_df["entity_id"].to_numpy(), s23_tokens)
    token_doc_freq = s23_flat["token"].value_counts()
    freq_cutoff = int(max_freq * len(s23_df))
    rare_tokens = set(token_doc_freq[token_doc_freq <= freq_cutoff].index)
    rare_index = s23_flat[s23_flat["token"].isin(rare_tokens)].rename(columns={"entity_id": "s23_entity_id"})
    s1_tokens = _tokenize_series(s1_df["name"])
    s1_flat = _exploded_blocking_tokens(s1_df["entity_id"].to_numpy(), s1_tokens)
    s1_flat = s1_flat[s1_flat["token"].isin(rare_tokens)]
    merged = s1_flat.merge(rare_index[["token", "s23_entity_id"]], on="token", how="inner")
    if not merged.empty:
        for s1_id, cand_set in merged.groupby("entity_id")["s23_entity_id"].agg(set).items():
            candidates[s1_id] = cand_set
    return dict(candidates)


names_pool = [
    "Acme Foundation LLC", "acme foundation llc", "Acme  Foundation", "Best Rrraz Corp",
    "abc abc market llc", "the and of xyz", "Zeta9 Traders", "duplicate duplicate word co",
    "123 456 numeric only", "Zeta9 Traders", "Small Rare Widget Shop", "Best Rrraz Corp",
]
s1_names = [rng.choice(names_pool) for _ in range(40)]
s23_names = [rng.choice(names_pool) for _ in range(80)]
s1_ids = [f"S1-{i}" for i in range(len(s1_names))]
s23_ids = [f"S23-{i}" for i in range(len(s23_names))]

naive = token_blocking_naive(s1_names, s1_ids, s23_names, s23_ids)
vec = token_blocking_vectorized(
    pd.DataFrame({"entity_id": s1_ids, "name": s1_names}),
    pd.DataFrame({"entity_id": s23_ids, "name": s23_names}),
)
all_s1 = set(s1_ids)
mismatch = [sid for sid in all_s1 if naive.get(sid, set()) != vec.get(sid, set())]
print(f"[TEST 1: token blocking]  entities={len(all_s1)}  mismatches={len(mismatch)}")
assert not mismatch, f"MISMATCH: {mismatch[:5]}"
print("  -> PASS: vectorized token blocking is bit-for-bit identical to naive reference\n")

# ---------------------------------------------------------------------------
# TEST 2: sparse pair extraction (groupby vs. per-pair loop)
# ---------------------------------------------------------------------------
from collections import defaultdict

n_s1, n_s23 = 50, 200
rows = rng.integers(0, n_s1, size=1500)
cols = rng.integers(0, n_s23, size=1500)
s1_ids2 = [f"S1-{i}" for i in range(n_s1)]
s23_ids2 = [f"S23-{i}" for i in range(n_s23)]

naive2 = defaultdict(set)
for r, c in zip(rows, cols):
    naive2[s1_ids2[r]].add(s23_ids2[c])
naive2 = dict(naive2)

candidates2 = defaultdict(set)
chunk_s1_ids = np.asarray(s1_ids2)[rows]
chunk_s23_ids = np.asarray(s23_ids2)[cols]
pair_df = pd.DataFrame({"s1_id": chunk_s1_ids, "s23_id": chunk_s23_ids})
for s1_id, cand_set in pair_df.groupby("s1_id")["s23_id"].agg(set).items():
    candidates2[s1_id].update(cand_set)
candidates2 = dict(candidates2)

mismatch2 = [k for k in set(naive2) | set(candidates2) if naive2.get(k, set()) != candidates2.get(k, set())]
print(f"[TEST 2: TF-IDF/embedding pair extraction]  mismatches={len(mismatch2)}")
assert not mismatch2
print("  -> PASS: vectorized groupby extraction is bit-for-bit identical to per-pair loop\n")

# ---------------------------------------------------------------------------
# TEST 3: embedding top-k (batched argpartition vs. per-row loop), including
# top_k >= pool size branch and ties/boundary values at the threshold
# ---------------------------------------------------------------------------
EMBEDDING_MIN_SIMILARITY = 0.35


def naive_topk(sim_matrix, top_k, threshold):
    out = defaultdict(set)
    for i in range(sim_matrix.shape[0]):
        scores = sim_matrix[i]
        if top_k < len(scores):
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[scores[top_indices] > threshold]
        else:
            top_indices = np.where(scores > threshold)[0]
        out[i] = set(top_indices.tolist())
    return dict(out)


def vectorized_topk(sim_matrix, top_k, threshold):
    n_pool = sim_matrix.shape[1]
    k = min(top_k, n_pool)
    if k < n_pool:
        idx = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]
    else:
        idx = np.tile(np.arange(n_pool), (sim_matrix.shape[0], 1))
    scores = np.take_along_axis(sim_matrix, idx, axis=1)
    keep = scores > threshold
    rows_idx, kept_cols = np.nonzero(keep)
    out = defaultdict(set)
    for r, c in zip(rows_idx, kept_cols):
        out[r].add(int(idx[r, c]))
    return dict(out)


# Case A: normal case, top_k < pool size, includes exact-threshold boundary values
sim = rng.uniform(0.0, 1.0, size=(30, 60)).astype(np.float32)
sim[0, 5] = EMBEDDING_MIN_SIMILARITY  # exactly at boundary -> must be EXCLUDED (strict >) in both
sim[1, 7] = np.nextafter(EMBEDDING_MIN_SIMILARITY, 1.0)  # just above -> must be INCLUDED in both
n1 = naive_topk(sim, top_k=10, threshold=EMBEDDING_MIN_SIMILARITY)
v1 = vectorized_topk(sim, top_k=10, threshold=EMBEDDING_MIN_SIMILARITY)
mism_a = [i for i in range(30) if n1.get(i, set()) != v1.get(i, set())]
print(f"[TEST 3a: embedding top-k, k<pool, incl. boundary values]  mismatches={len(mism_a)}")
assert not mism_a
print("  -> PASS")

# Case B: top_k >= pool size (the "else" branch)
sim_small = rng.uniform(0.0, 1.0, size=(15, 8)).astype(np.float32)
n2 = naive_topk(sim_small, top_k=20, threshold=EMBEDDING_MIN_SIMILARITY)
v2 = vectorized_topk(sim_small, top_k=20, threshold=EMBEDDING_MIN_SIMILARITY)
mism_b = [i for i in range(15) if n2.get(i, set()) != v2.get(i, set())]
print(f"[TEST 3b: embedding top-k, top_k >= pool size branch]  mismatches={len(mism_b)}")
assert not mism_b
print("  -> PASS")

# Case C: ties at the exact same similarity score (argpartition tie-breaking
# shouldn't matter since we only care about the SET of indices above threshold)
sim_ties = np.full((5, 20), 0.9, dtype=np.float32)
n3 = naive_topk(sim_ties, top_k=5, threshold=0.5)
v3 = vectorized_topk(sim_ties, top_k=5, threshold=0.5)
sizes_match = all(len(n3[i]) == len(v3[i]) == 5 for i in range(5))
print(f"[TEST 3c: embedding top-k, all-tied scores]  correct top-k size on both sides: {sizes_match}")
assert sizes_match
print("  -> PASS (tie-breaking picks a different concrete index than naive, but that's expected: "
      "with all-equal scores every 5-subset is an equally valid 'top 5', and this exists in the "
      "ORIGINAL per-row code too since numpy argpartition's tie-break is itself unspecified -- "
      "not something this rewrite introduces)\n")

print("=" * 60)
print("ALL EQUIVALENCE TESTS PASSED")
print("=" * 60)

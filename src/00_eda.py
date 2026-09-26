# %% [markdown]
# # 00 — Exploratory Data Analysis & Noise Discovery
#
# **Phase 0, Steps 0.3 & 0.4** — Systematic EDA with programmatic profiling.
#
# Sections:
# 1. Setup & data loading (sample-based for speed)
# 2. Schema inspection — column names, dtypes, missing rates
# 3. Country distribution
# 4. Character-set profiling — **fully dynamic** Unicode script discovery
# 5. Token frequency analysis
# 6. Regex noise scans with counts
# 7. Missing field rates by country
# 8. Cross-script match analysis (ground truth pairs)
# 9. Noise catalog — prioritised summary

# %%
from __future__ import annotations
import os, re, warnings, unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import regex  # used for tokenization in Section 5

warnings.filterwarnings("ignore")

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(os.getcwd())
for p in [Path(os.getcwd()), *Path(os.getcwd()).parents]:
    if (p / "dataset").exists():
        REPO_ROOT = p
        break

TRAIN_DIR = REPO_ROOT / "dataset" / "train"
TEST_DIR  = REPO_ROOT / "dataset" / "test"

print(f"REPO_ROOT  : {REPO_ROOT}")
print(f"TRAIN_DIR  : {TRAIN_DIR}")
print(f"TEST_DIR   : {TEST_DIR}")
assert TRAIN_DIR.exists(), f"Train dir not found: {TRAIN_DIR}"
assert TEST_DIR.exists(),  f"Test dir not found:  {TEST_DIR}"

# %%
# ---------------------------------------------------------------------------
# Sampling configuration
# Full files are 200-500 MB each. We sample for interactive EDA.
# ---------------------------------------------------------------------------
SAMPLE_N    = 200_000   # rows per source for interactive analysis
RANDOM_SEED = 42

def load_tsv_sample(path, n=SAMPLE_N, seed=RANDOM_SEED):
    """Load a random sample of rows from a TSV file efficiently."""
    total = sum(1 for _ in open(path, "r", encoding="utf-8")) - 1  # minus header
    print(f"  {path.name}: {total:,} rows  ->  sampling {min(n, total):,}")

    if total <= n:
        return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)

    rng = np.random.default_rng(seed)
    skip_idx = rng.choice(total, total - n, replace=False)
    skip_set = set(skip_idx + 1)  # +1 because header is row 0
    return pd.read_csv(
        path, sep="\t", dtype=str, low_memory=False,
        skiprows=lambda i: i in skip_set,
    )

print("Loading samples ...")
s1_train = load_tsv_sample(TRAIN_DIR / "train_source1.tsv")
s2_train = load_tsv_sample(TRAIN_DIR / "train_source2.tsv")
s3_train = load_tsv_sample(TRAIN_DIR / "train_source3.tsv")
print("Done.")

# %% [markdown]
# ## 1. Schema inspection

# %%
for name, df in [("S1 (train)", s1_train), ("S2 (train)", s2_train), ("S3 (train)", s3_train)]:
    print(f"\n--- {name} ({len(df):,} sampled rows) ---")
    print(f"Columns: {list(df.columns)}")
    print(df.dtypes)
    print("\nMissing rates:")
    print((df.isnull() | (df == "")).mean().to_string())

# %% [markdown]
# ## 2. Country distribution

# %%
for name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    col = "country" if "country" in df.columns else None
    if col:
        print(f"\n{name} country distribution:")
        print(df[col].value_counts(dropna=False).to_string())

# %% [markdown]
# ## 3. Character-set profiling — fully dynamic Unicode script discovery
#
# ### 3a. Character-level inventory
#
# We make **no assumptions** about which scripts exist. We scan every unique
# non-ASCII character and use `unicodedata.name()` to extract the script/block
# family automatically. No predefined list.

# %%
NAME_COL = "business_name"
ADDR_COL = "business_address"


def build_script_inventory(df, col):
    """
    Dynamically discover every Unicode script/block present in `col`.

    1. Collect all unique non-ASCII characters across the column.
    2. unicodedata.name(ch) -> e.g. "DEVANAGARI LETTER RA"
    3. First word = script/block family.  No hardcoded list.

    Returns
    -------
    script_counts : Counter  {script_name: distinct_char_count}
    sample_chars  : dict     {script_name: [up to 8 sample chars]}
    """
    all_text  = "".join(df[col].dropna().astype(str))
    non_ascii = {ch for ch in all_text if ord(ch) >= 128}

    script_counts = Counter()
    sample_chars  = {}
    unknown_chars = []

    for ch in non_ascii:
        uname  = unicodedata.name(ch, "UNKNOWN")
        script = uname.split()[0] if uname != "UNKNOWN" else "UNKNOWN"

        script_counts[script] += 1
        if len(sample_chars.get(script, [])) < 8:
            sample_chars.setdefault(script, []).append(ch)

        if uname == "UNKNOWN":
            unknown_chars.append(f"U+{ord(ch):04X}")

    if unknown_chars:
        print(f"  WARNING: {len(unknown_chars)} chars had no Unicode name: {unknown_chars[:10]}")

    return script_counts, sample_chars


# --- Run per source ----------------------------------------------------------
SCRIPT_INVENTORIES = {}

for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if NAME_COL not in df.columns:
        print(f"{src_name}: no '{NAME_COL}' column"); continue

    counts, samples = build_script_inventory(df, NAME_COL)
    SCRIPT_INVENTORIES[src_name] = (counts, samples)

    print(f"\n{src_name} -- Unicode script/block inventory ({len(counts)} families found):")
    rows = []
    for script, n_chars in counts.most_common():
        chars_display = " ".join(samples.get(script, []))
        rows.append({"script/block": script, "distinct_chars": n_chars, "samples": chars_display})
    inv_df = pd.DataFrame(rows)
    print(inv_df.to_string(index=False))

# %% [markdown]
# ### 3b. Record-level script classifier (fully dynamic)
#
# Uses `unicodedata.category()` (Unicode standard) to identify **letters/marks**
# (script-bearing) vs punctuation/symbols/digits (structural).
# Then `unicodedata.name()` extracts the script.
#
# Only assumption: `LATIN` is the baseline. Everything else is discovered.
# **No hardcoded script list whatsoever.**

# %%
# Unicode General Categories that carry script identity:
#   L* = Letters,  M* = Marks (diacritics/combining, tied to a script)
# Everything else (P*, S*, N*, Z*, C*) is structural.
SCRIPT_BEARING_CATEGORIES = {"L", "M"}


def get_char_script(ch):
    """Return the script family for a letter/mark character, or None for structural."""
    if ord(ch) < 128:
        return None  # ASCII — skip
    cat = unicodedata.category(ch)
    if cat[0] not in SCRIPT_BEARING_CATEGORIES:
        return None  # punctuation, symbol, digit, separator, control
    uname = unicodedata.name(ch, "")
    if not uname:
        return None
    return uname.split()[0]


def detect_scripts_dynamic(text):
    """
    Return the set of all script families for letter/mark characters in text.
    Purely data-driven. No predefined script list.
    """
    if not isinstance(text, str) or not text.strip():
        return set()
    scripts = set()
    for ch in text:
        s = get_char_script(ch)
        if s is not None:
            scripts.add(s)
    return scripts


def is_non_latin_name(text):
    """True if the name contains letter/mark characters from any non-LATIN script."""
    scripts = detect_scripts_dynamic(text)
    scripts.discard("LATIN")
    return len(scripts) > 0


# --- Record-level stats per source -------------------------------------------
for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if NAME_COL not in df.columns:
        continue

    non_latin_mask = df[NAME_COL].apply(is_non_latin_name)
    pct = non_latin_mask.mean() * 100
    print(f"\n{src_name}: {non_latin_mask.sum():,} / {len(df):,} ({pct:.2f}%) records contain non-Latin script chars")

    script_record_counter = Counter()
    mixed_script_count = 0
    for text in df.loc[non_latin_mask, NAME_COL]:
        scripts = detect_scripts_dynamic(text)
        scripts.discard("LATIN")
        for s in scripts:
            script_record_counter[s] += 1
        if len(scripts) > 1:
            mixed_script_count += 1

    print("  Discovered non-Latin scripts (records containing each):")
    for script, cnt in script_record_counter.most_common():
        print(f"    {script:<20}: {cnt:>7,}")
    print(f"  Mixed non-Latin script records (2+ scripts in one name): {mixed_script_count:,}")

# %% [markdown]
# ## 4. Token frequency analysis — top tokens & legal suffix inventory

# %%
def tokenize_latin(text):
    if not isinstance(text, str):
        return []
    return regex.findall(r"[a-zA-Z0-9]+", text.lower())

for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if NAME_COL not in df.columns:
        continue
    counter = Counter()
    for text in df[NAME_COL].dropna():
        counter.update(tokenize_latin(text))
    print(f"\n{src_name} -- Top 30 business_name tokens:")
    print(counter.most_common(30))
    print(f"  Total unique tokens: {len(counter):,}")

# %%
# Legal suffix inventory
LEGAL_RE = regex.compile(
    r"\b(llc|ltd|limited|inc|corp|corporation|pvt|private|llp|lp|co|company|"
    r"sarl|sci|sas|plc|pllc|gmbh|bv|ag|sa|sl|srl|pty|ngo|npo|trust|"
    r"foundation|association|society|institute|university|college|school|"
    r"hospital|clinic|academy|services|solutions|group|holdings|enterprises|"
    r"international|industries|consulting|technologies|tech|labs|media|"
    r"ventures|partners|capital)$",
    flags=regex.IGNORECASE,
)

def get_legal_suffix(text):
    if not isinstance(text, str):
        return None
    m = LEGAL_RE.search(text.strip())
    return m.group(0).lower() if m else None

for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if NAME_COL not in df.columns:
        continue
    suffixes = df[NAME_COL].apply(get_legal_suffix)
    print(f"\n{src_name} -- Legal suffix distribution:")
    print(suffixes.value_counts(dropna=False).head(20).to_string())

# %% [markdown]
# ## 5. Regex noise scans with counts

# %%
NOISE_PATTERNS = {
    "hashtag_prefix":     re.compile(r"^[#<\s]+"),
    "appended_digits":    re.compile(r"[-\s]\d{6,}\s*$|#\d{5,}\s*$"),
    "url_in_name":        re.compile(r"www\.|\.(com|net|org|in|co\.in)|https?://|@\w+"),
    "duplicate_adj_word": re.compile(r"\b(\w{3,})\s+\1\b", re.IGNORECASE),
    "extra_whitespace":   re.compile(r"\s{2,}"),
    "null_literal":       re.compile(r"\bnull\b", re.IGNORECASE),
    "angle_brackets":     re.compile(r"<<|>>"),
    "all_caps_word":      re.compile(r"\b[A-Z]{4,}\b"),
    "dba_alias":          re.compile(r"\bdba\b", re.IGNORECASE),
    "parenthetical_sfx":  re.compile(r"\((?:LLC|LLP|Inc|Corp|Co|Ltd|Limited|Pvt|Private)\)", re.IGNORECASE),
    "accent_anomaly":     re.compile(r"[\u00c0-\u00ff]"),
    "pipe_separator":     re.compile(r"\|"),
    "leetspeak_digit":    re.compile(r"\d[a-zA-Z]|[a-zA-Z]\d"),   # e.g. "5uperior"
}

for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if NAME_COL not in df.columns:
        continue
    col = df[NAME_COL].fillna("")
    print(f"\n{src_name} -- Noise pattern scan on business_name (n={len(df):,}):")
    for pat_name, pat in NOISE_PATTERNS.items():
        mask = col.str.contains(pat, na=False)
        n = mask.sum()
        pct = n / len(df) * 100
        if n > 0:
            example = col[mask].iloc[0]
            print(f"  {pat_name:<22}: {n:>7,} ({pct:5.2f}%)  e.g. {repr(example[:80])}")

# %%
# Address-specific noise
ADDR_PATTERNS = {
    "null_literal":    re.compile(r"\bnull\b", re.IGNORECASE),
    "po_box":          re.compile(r"\bP\.?O\.?\s*Box\b", re.IGNORECASE),
    "all_caps":        re.compile(r"^[A-Z0-9\s,\.#-]+$"),
    "reordered_addr":  re.compile(r"^[A-Z]{2},"),  # state before street
    "empty":           re.compile(r"^\s*$"),
    "non_latin_chars": re.compile(r"[^\x00-\x7f]"),  # any non-ASCII in address
}

for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if ADDR_COL not in df.columns:
        continue
    col = df[ADDR_COL].fillna("")
    print(f"\n{src_name} -- Address noise scan (n={len(df):,}):")
    for pat_name, pat in ADDR_PATTERNS.items():
        if pat_name in ("all_caps", "reordered_addr", "empty"):
            mask = col.str.match(pat)
        else:
            mask = col.str.contains(pat, na=False)
        n = mask.sum()
        pct = n / len(df) * 100
        print(f"  {pat_name:<20}: {n:>7,} ({pct:5.2f}%)")

# %% [markdown]
# ## 6. Missing field rates by country

# %%
for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if "country" not in df.columns or ADDR_COL not in df.columns:
        continue
    missing = (
        df.assign(_empty=(df[ADDR_COL].isnull() | (df[ADDR_COL].str.strip() == "")))
          .groupby("country")["_empty"]
          .agg(["sum", "count", "mean"])
          .rename(columns={"sum": "empty", "count": "total", "mean": "pct_empty"})
    )
    print(f"\n{src_name} -- address missing rate by country:")
    print(missing.to_string())

# %% [markdown]
# ## 7. Cross-script match analysis (ground truth pairs)
#
# Quantifies how many ground-truth (S1, S2/S3) pairs involve a script mismatch.
# Uses `detect_scripts_dynamic()` from Section 3b — fully data-driven.

# %%
gt_path = TRAIN_DIR / "train_ground_truth.tsv"
if not gt_path.exists():
    print("Ground truth file not found -- skipping cross-script analysis")
else:
    print("Loading ground truth sample ...")
    gt_sample = load_tsv_sample(gt_path, n=50_000)
    print(f"  Loaded {len(gt_sample):,} GT rows")
    print(f"  Columns: {list(gt_sample.columns)}")

    # Singletons
    has_match = gt_sample["matched_entity_ids"].notna() & (gt_sample["matched_entity_ids"].str.strip() != "")
    print(f"\n  Singletons (no match): {(~has_match).sum():,} / {len(gt_sample):,} ({(~has_match).mean()*100:.2f}%)")

    # Match count distribution
    match_counts = gt_sample.loc[has_match, "matched_entity_ids"].str.split(",").str.len()
    print(f"\n  Match count distribution (non-singletons):")
    print(match_counts.value_counts().sort_index().head(15).to_string())
    print(f"  Avg matches per non-singleton: {match_counts.mean():.2f}")
    print(f"  Max matches: {match_counts.max()}")

# %%
# Cross-script: uses detect_scripts_dynamic() -- no hardcoded script list.
if gt_path.exists() and "gt_sample" in dir():
    # Build name lookups from samples
    id_col = "entity_id"

    s1_name_map = s1_train.set_index(id_col)[NAME_COL].to_dict() if NAME_COL in s1_train.columns else {}
    s2_name_map = s2_train.set_index(id_col)[NAME_COL].to_dict() if NAME_COL in s2_train.columns else {}
    s3_name_map = s3_train.set_index(id_col)[NAME_COL].to_dict() if NAME_COL in s3_train.columns else {}

    cross_script        = 0
    total_pairs         = 0
    script_pair_counter = Counter()  # (s1_scripts, s23_scripts) -> count

    for _, row in gt_sample.iterrows():
        s1_id      = row.get("source1_entity_id", "")
        s1_name    = s1_name_map.get(s1_id, "")
        s1_scripts = frozenset(detect_scripts_dynamic(s1_name)) or frozenset(["LATIN"])

        raw = row.get("matched_entity_ids", "")
        if not isinstance(raw, str) or not raw.strip():
            continue

        for mid in raw.split(","):
            mid = mid.strip()
            if   mid.startswith("S2-"): matched_name = s2_name_map.get(mid, "")
            elif mid.startswith("S3-"): matched_name = s3_name_map.get(mid, "")
            else: continue

            total_pairs  += 1
            s23_scripts   = frozenset(detect_scripts_dynamic(matched_name)) or frozenset(["LATIN"])

            if s1_scripts != s23_scripts:
                cross_script += 1
                pair_key = (tuple(sorted(s1_scripts)), tuple(sorted(s23_scripts)))
                script_pair_counter[pair_key] += 1

    if total_pairs > 0:
        print(f"Cross-script pairs: {cross_script:,} / {total_pairs:,} ({cross_script/total_pairs*100:.2f}%)")
        print("  (note: limited by sampled lookups -- not all matched IDs may be in sample)")
        print("\n  Top cross-script pair types:")
        for (s1_s, s23_s), cnt in script_pair_counter.most_common(10):
            print(f"    S1:{list(s1_s)} <-> S2/S3:{list(s23_s)}  -- {cnt:,} pairs")
    else:
        print("No matching pairs found in sample overlap -- increase SAMPLE_N or run on full data")
# %% [markdown]
# ## 7b. Catch-All Unknown Pattern Discovery
#
# Dynamically discover high-frequency anomalous tokens not covered by predefined patterns.
# This catches unexpected noise patterns specific to this dataset.

# %%
def find_anomalous_tokens(df, col):
    """"""Find suspiciously high-frequency non-word tokens and mixed alphanumeric patterns.""""""
    tokens = []
    for text in df[col].dropna():
        if not isinstance(text, str):
            continue
        # Extract non-standard tokens:
        # - Pure symbol sequences (e.g., "###", "|||")
        # - Mixed digit-letter patterns (e.g., "5uperior", "H3llo")
        # - Repeated punctuation
        tokens.extend(regex.findall(r"[^\w\s]{2,}|\d+[a-z]+|[a-z]+\d+|\w*[^\w\s]+\w+", text.lower()))
    
    counter = Counter(tokens)
    # Flag tokens appearing in >0.5% of records
    threshold = len(df) * 0.005
    return {tok: cnt for tok, cnt in counter.items() if cnt > threshold}

for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if NAME_COL not in df.columns:
        continue
    print(f"
{src_name} -- High-frequency anomalous tokens (candidates for new noise rules):")
    anomalies = find_anomalous_tokens(df, NAME_COL)
    if anomalies:
        for tok, cnt in sorted(anomalies.items(), key=lambda x: -x[1])[:20]:
            pct = cnt / len(df) * 100
            # Find an example
            example_mask = df[NAME_COL].str.contains(regex.escape(tok), na=False, case=False)
            example = df.loc[example_mask, NAME_COL].iloc[0] if example_mask.any() else ""
            print(f"  {repr(tok):<25}: {cnt:>7,} ({pct:5.2f}%)  e.g. {repr(example[:60])}")
    else:
        print("  No high-frequency anomalous tokens found (threshold: >0.5% of records)")

# %% [markdown]
# ## 7c. Verification Pass � Full Data Frequency Check
#
# **Purpose**: Verify that pattern frequencies observed in the 200k sample are
# representative of the full dataset. Critical patterns that will drive normalization
# rules are re-counted on the complete data.
#
# **Sampling strategy**: Initial EDA used stratified random samples of 200k records
# per source (~4-9% of training data). This verification pass confirms that
# high-frequency patterns (>1%) have stable frequency estimates before building
# normalization rules.

# %%
import csv

def verify_pattern_on_full_data(path, patterns, col_name="business_name"):
    """"""
    Stream full TSV file and count pattern matches without loading into memory.
    
    Parameters
    ----------
    path : Path
        Path to TSV file
    patterns : dict
        {pattern_name: compiled regex}
    col_name : str
        Column name to scan
    
    Returns
    -------
    dict : {pattern_name: (count, percentage)}
    """"""
    results = {name: 0 for name in patterns.keys()}
    total_rows = 0
    
    print(f"  Streaming {path.name} ...")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            text = row.get(col_name, "")
            for pat_name, pat in patterns.items():
                if pat.search(text):
                    results[pat_name] += 1
            
            # Progress indicator every 500k rows
            if total_rows % 500_000 == 0:
                print(f"    ... processed {total_rows:,} rows")
    
    return {name: (count, count/total_rows*100 if total_rows > 0 else 0.0) 
            for name, count in results.items()}, total_rows

# Select patterns with >1% frequency in sample to verify on full data
VERIFY_PATTERNS = {
    name: pat for name, pat in NOISE_PATTERNS.items()
    # Add patterns here that showed >1% in sample - will be determined after first EDA run
}

# By default, verify the most impactful patterns
CRITICAL_PATTERNS = {
    "url_in_name":        NOISE_PATTERNS["url_in_name"],
    "duplicate_adj_word": NOISE_PATTERNS["duplicate_adj_word"],
    "all_caps_word":      NOISE_PATTERNS["all_caps_word"],
    "extra_whitespace":   NOISE_PATTERNS["extra_whitespace"],
    "dba_alias":          NOISE_PATTERNS["dba_alias"],
}

print("
" + "="*80)
print("VERIFICATION PASS: Pattern frequencies on FULL training data")
print("="*80)
print("
This confirms that sample-based frequencies are representative.")
print("Only run this after initial EDA to verify patterns you plan to normalize.
")

# Verify on Source 2 (noisiest source)
s2_path = TRAIN_DIR / "train_source2.tsv"
if s2_path.exists():
    print(f"
Verifying on {s2_path.name} (full file)...")
    full_results, total = verify_pattern_on_full_data(s2_path, CRITICAL_PATTERNS, NAME_COL)
    
    print(f"
Full data pattern frequencies (n={total:,} rows):")
    print(f"{'Pattern':<22}  {'Count':>10}  {'Percentage':>10}  {'Sample vs Full'}")
    print("-" * 75)
    for pat_name, (count, pct) in sorted(full_results.items(), key=lambda x: -x[1][1]):
        # Compare to sample if available
        comparison = ""
        if NAME_COL in s2_train.columns:
            sample_mask = s2_train[NAME_COL].fillna("").str.contains(
                CRITICAL_PATTERNS[pat_name], na=False
            )
            sample_pct = sample_mask.mean() * 100
            diff = pct - sample_pct
            comparison = f"{diff:+.2f}% diff"
        
        print(f"{pat_name:<22}: {count:>10,}  {pct:>9.2f}%  {comparison}")
    
    print(f"
? Verification complete. Update noise_catalog with full-data frequencies.")
else:
    print(f"
? {s2_path} not found - skipping full-data verification.")
    print("  Run this cell after initial EDA to confirm pattern frequencies.")


# %% [markdown]
# ## 8. Automated Noise Catalog ? Prioritized Summary
#
# This section automatically collects pattern frequencies from all scans above
# and builds a prioritized catalog for normalization planning.
#
# **Priority levels:**
# - **HIGH**: >10% prevalence or critical for matching (cross-script, legal suffixes)
# - **MEDIUM**: 1-10% prevalence, consistent noise across sources
# - **LOW**: <1% but systematic, worth tracking

# %%
# Collect pattern frequencies from all scans
catalog_data = []

# 1. Collect noise pattern frequencies from Section 5
print("="*80)
print("AUTOMATED NOISE CATALOG")
print("="*80)
print("\nCollecting pattern frequencies from scans...\n")

for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if NAME_COL not in df.columns:
        continue
    col = df[NAME_COL].fillna("")
    
    for pat_name, pat in NOISE_PATTERNS.items():
        mask = col.str.contains(pat, na=False)
        n = mask.sum()
        pct = n / len(df) * 100
        
        # Determine priority
        if pct >= 10:
            priority = "HIGH"
        elif pct >= 1:
            priority = "MEDIUM"
        elif pct >= 0.1:
            priority = "LOW"
        else:
            priority = "TRACE"
        
        # Map pattern to normalization action
        action_map = {
            "hashtag_prefix": "strip prefix",
            "appended_digits": "strip suffix",
            "url_in_name": "remove URL fragments",
            "duplicate_adj_word": "collapse duplicates",
            "extra_whitespace": "normalize whitespace",
            "null_literal": "remove 'null' strings",
            "angle_brackets": "strip brackets",
            "all_caps_word": "case normalization",
            "dba_alias": "split into legal_name + alias_name",
            "parenthetical_sfx": "strip parentheses, normalize suffix",
            "accent_anomaly": "selective accent stripping",
            "pipe_separator": "remove or split on pipe",
            "leetspeak_digit": "flag for manual review",
        }
        
        if n > 0:  # Only include patterns that were found
            catalog_data.append({
                "pattern": pat_name,
                "source": src_name,
                "count": n,
                "pct": pct,
                "action": action_map.get(pat_name, "TBD"),
                "priority": priority,
                "sample_size": len(df)
            })

# 2. Add address pattern frequencies
for src_name, df in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
    if ADDR_COL not in df.columns:
        continue
    col = df[ADDR_COL].fillna("")
    
    for pat_name, pat in ADDR_PATTERNS.items():
        if pat_name in ("all_caps", "reordered_addr", "empty"):
            mask = col.str.match(pat)
        else:
            mask = col.str.contains(pat, na=False)
        n = mask.sum()
        pct = n / len(df) * 100
        
        if pct >= 10:
            priority = "HIGH"
        elif pct >= 1:
            priority = "MEDIUM"
        elif pct >= 0.1:
            priority = "LOW"
        else:
            priority = "TRACE"
        
        action_map_addr = {
            "null_literal": "remove 'null' strings",
            "po_box": "extract PO Box, keep separate field",
            "all_caps": "case normalization",
            "reordered_addr": "parse and reorder components",
            "empty": "flag for name-only matching",
            "non_latin_chars": "preserve for embedding, transliterate for token matching",
        }
        
        if n > 0:
            catalog_data.append({
                "pattern": f"addr:{pat_name}",
                "source": src_name,
                "count": n,
                "pct": pct,
                "action": action_map_addr.get(pat_name, "TBD"),
                "priority": priority,
                "sample_size": len(df)
            })

# 3. Add cross-script stats if available
if 'cross_script' in dir() and 'total_pairs' in dir() and total_pairs > 0:
    cross_script_pct = (cross_script / total_pairs) * 100
    priority = "HIGH" if cross_script_pct >= 5 else "MEDIUM"
    catalog_data.append({
        "pattern": "cross_script_pairs",
        "source": "GT",
        "count": cross_script,
        "pct": cross_script_pct,
        "action": "embedding-based blocking + matching",
        "priority": priority,
        "sample_size": total_pairs
    })

# Build and display catalog
if catalog_data:
    catalog_df = pd.DataFrame(catalog_data)
    
    # Sort by priority then percentage
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "TRACE": 3}
    catalog_df['_priority_sort'] = catalog_df['priority'].map(priority_order)
    catalog_df = catalog_df.sort_values(['_priority_sort', 'pct'], ascending=[True, False])
    catalog_df = catalog_df.drop('_priority_sort', axis=1)
    
    # Display by priority level
    print("\n" + "="*80)
    print("HIGH PRIORITY PATTERNS (>10% or critical for matching)")
    print("="*80)
    high_priority = catalog_df[catalog_df['priority'] == 'HIGH']
    if len(high_priority) > 0:
        for _, row in high_priority.iterrows():
            print(f"{row['pattern']:<30} | {row['source']:<4} | {row['count']:>8,} ({row['pct']:>6.2f}%) | {row['action']}")
    else:
        print("  No high-priority patterns found")
    
    print("\n" + "="*80)
    print("MEDIUM PRIORITY PATTERNS (1-10% prevalence)")
    print("="*80)
    med_priority = catalog_df[catalog_df['priority'] == 'MEDIUM']
    if len(med_priority) > 0:
        for _, row in med_priority.iterrows():
            print(f"{row['pattern']:<30} | {row['source']:<4} | {row['count']:>8,} ({row['pct']:>6.2f}%) | {row['action']}")
    else:
        print("  No medium-priority patterns found")
    
    print("\n" + "="*80)
    print("LOW PRIORITY PATTERNS (0.1-1% prevalence)")
    print("="*80)
    low_priority = catalog_df[catalog_df['priority'] == 'LOW']
    if len(low_priority) > 0:
        for _, row in low_priority.iterrows():
            print(f"{row['pattern']:<30} | {row['source']:<4} | {row['count']:>8,} ({row['pct']:>6.2f}%) | {row['action']}")
    else:
        print("  No low-priority patterns found")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total patterns detected: {len(catalog_df)}")
    print(f"  HIGH priority:   {len(high_priority)}")
    print(f"  MEDIUM priority: {len(med_priority)}")
    print(f"  LOW priority:    {len(low_priority)}")
    print(f"\n? Catalog complete. Use this to prioritize normalization rules in 01_normalize.ipynb")
    
    # Export catalog for reference
    catalog_export = catalog_df[['pattern', 'source', 'pct', 'action', 'priority']].copy()
    print(f"\nMarkdown table export:\n")
    # Try to use tabulate if available, otherwise use manual formatting
    try:
        print(catalog_export.to_markdown(index=False))
    except ImportError:
        # Manual markdown table formatting (tabulate not installed)
        print('| pattern | source | pct | action | priority |')
        print('|---------|--------|-----|--------|----------|')
        for _, row in catalog_export.iterrows():
            print(f'| {row["pattern"]} | {row["source"]} | {row["pct"]:.2f} | {row["action"]} | {row["priority"]} |')
        print('\n(Note: Install tabulate package for better table formatting: pip install tabulate)')
else:
    print("No patterns collected. Run the scan cells above first.")

# %%

# %% [markdown]
# ## Next steps
#
# 1. Review counts above — mark HIGH / MEDIUM / LOW priority for normalization.
# 2. Hand off frequency table to `01_normalize.ipynb`.
# 3. Cross-script % -> informs embedding blocking budget in `02_blocking.ipynb`.
# 4. Address missing rate -> informs address fallback strategy.
# 5. Update `Documentation_template.md` Problem Analysis section with these numbers.
# 6. Note which patterns to instrument as toggleable rules (Step 1.1 rule-level instrumentation).

# Business Entity Resolution Challenge — Execution Plan

## Objective
Build a complete entity resolution pipeline matching Source 2/3 records to Source 1
reference entities, optimized for **macro-averaged F_0.5** (precision-weighted 2×
over recall). Produce two output files: `matching_results.tsv` (scored) and
`candidate_pairs.tsv` (blocking audit), plus a filled-in methodology document and
runnable code package.

---

## Confirmed Data Landscape

### Scale (from actual files)

| File | Records | Size |
|---|---|---|
| train_source1.tsv | ~2,206,822 | 210 MB |
| train_source2.tsv | ~5,034,617 | 489 MB |
| train_source3.tsv | ~5,285,604 | 504 MB |
| train_ground_truth.tsv | 2,206,821 rows (1 per S1 entity) | 127 MB |
| test_source1.tsv | ~1,732,545 | 175 MB |
| test_source2.tsv | ~4,887,274 | 509 MB |
| test_source3.tsv | ~5,082,317 | 506 MB |

**Total test entities to produce matches for: ~1.73M Source 1 entities.**
**Total candidate pool (S2+S3 test): ~9.97M records.**

Full cross-product is **1.73M × 9.97M ≈ 17.2 trillion** comparisons — blocking
is absolutely mandatory and is the single highest-leverage stage.

### Ground Truth Statistics (from full training data)

| Metric | Value |
|---|---|
| Total S1 entities | 2,206,821 |
| Singletons (no matches) | 123,247 (5.58%) |
| Non-singletons | 2,083,574 (94.42%) |
| Avg matches per non-singleton | ~3.65 |
| Max matches per entity | 11 |
| Match count distribution peak | 3 matches (most common) |

**Key implication:** Singletons are ~5.6% of entities. Under F_0.5 macro-averaging,
each singleton scores 1.0 when correctly predicted empty and 0.0 for any false merge.
This means singletons are *easy precision wins* if classified correctly, but a naïve
"match everything" approach incurs a 5.6% precision drag. A high-quality singleton
detector is worth building.

### Country Distribution

- **Training**: `US` and `India` only
- **Test**: `US`, `India`, and **`France`** (unseen in training)
- France appears in test with French-style names (SARL, SCI, SAS suffixes),
  French addresses (Rue, Boulevard, Allée), and French regions (Nouvelle-Aquitaine,
  Hauts-de-France, Loire-Atlantique, Nord)
- **Critical constraint**: Pipeline must be country-agnostic. No hardcoded
  US/India-only logic (no hardcoded ZIP/PIN regex, no country whitelists)

### Observed Noise Patterns (preliminary — from a small manual sample)

> **These are hints spotted from ~30 rows per source, NOT systematic findings.**
> The real discovery with frequency counts across the full dataset happens via
> `00_eda.py` (Phase 0, Step 0.3), outputting comprehensive statistics and frequency
> counts (with an optional stats notebook only if visual distributions/plots are needed).
> These examples tell us *what to scan for*; the programmatic profiling results tell us
> *how common each pattern actually is* and whether it warrants a normalization rule.

**Source 1 (reference, deduplicated):**
- Clean, mixed-case names: "Orelee's Barbershop", "Custom Wealth Services LLC"
- Structured US addresses: "1795 Westchester Drive, High Point, NC"
- Structured India addresses: "797, Lake Town Block A, Kolkata, Howrah, West Bengal"
- Entity IDs: `S1-XXXXXXXXX` format

**Source 2 (noisiest):**
- **ALL-CAPS addresses**: "914 PIERPONT AVE, CLEVELAND, OH"
- **Devanagari script business names**: "राम मार्केटिंग प्राइवेट लिमिटेड" (must match Latin S1 names)
- **Tamil script names**: "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்"
- **URL/website leakage in names**: "heassociates.com", "SHIVSHAKTI VIDYALAYA ... | www.shivshakti.com"
- **Duplicated words in names**: "Crestline Crestline Clean LP", "Keystone Odyssey Odyssey LLC"
- **Parenthetical legal suffixes**: "COBALT (LLC)", "Sweet Barbershop (Co)"
- **Typos**: "Tetlecommunication" (Telecommunication), "FTT MITCHELL" (Ft Mitchell)
- **Extra spaces**: "FOUNDATION EXCEL AGENCY PRIVATE  LIMITED"
- **Accent artifacts**: "Nétwork" (Network)
- **Address reordering**: "NC, CANLER, 110 MEADOWBROOK ACRES" (city/state before street)
- **Abbreviations**: AVE, ST, DR, LN, RD in addresses

**Source 3 (also noisy, different patterns):**
- **Mixed script + Latin hybrid names**: "அரிஹந்த் Foundation Private Limited"
- **Hashtag prefixes**: "#centraleducation"
- **Leetspeak/digit substitution**: "5uperior" (Superior)
- **Appended numeric IDs**: "MW Management Private Limited - 2067865001"
- **Appended random numbers in names**: "Cardiology Heartland Cáre Associates #98825"
- **Double spaces**: "The Rapid  Learning Alliance LLC"
- **Typos**: "Hospirlg" (Hospital?), "Hurricanne" (Hurricane), "Connmre" (Conner?)
- **Angle bracket artifacts**: "<< Team Ecole"
- **Accent in wrong places**: "Léarning" (Learning), "Cáre" (Care), "Béque"
- **Empty address fields**: "International South Consultants Private Ltd" with blank address
- **Kannada script**: "ಕರ್ನಾಟಕ" appearing in address state fields
- **Hindi/Marathi state names**: "महाराष्ट्र" (Maharashtra), "दिल्ली" (Delhi)
- **PO Box in address**: "PO Box 6009"
- **"null" string in addresses**: "G.t. Karnal Road, Industrial Area, New Delhi, null, A-68"
- **Word "Pvt." with period**: "Pvt. EFS Print Ventures Ltd."
- **Duplicated words**: "FERRERO FERRERO DUKE"
- **dba aliases**: "Ectolumdrex dba X+ Madison Inc"

**France-specific patterns (test only):**
- Legal suffixes: SARL, SCI, SAS, S.A.S
- Address format: "5 bis Rue Pierre Dignac", "63 R. DE DIEPPE"
- Regions as state: "Nouvelle-Aquitaine", "Hauts-de-France"
- Département names: "Gironde", "Nord", "Loire-Atlantique"
- Accented characters: "à" in names

---

## Architecture

```
┌──────────────┐     ┌─────────────────────────┐     ┌──────────────────┐
│ Normalization │ ──→ │ Multi-Strategy Blocking  │ ──→ │ Feature Eng.     │
│ (all sources) │     │ (union of candidate sets)│     │ (pairwise feats) │
└──────────────┘     └─────────────────────────┘     └──────────────────┘
                                                              │
                                                              ▼
                      ┌──────────────────┐     ┌──────────────────────────┐
                      │ Output Generation │ ←── │ Classifier + Threshold   │
                      │ (TSV files)       │     │ Tuning (F_0.5 optimized) │
                      └──────────────────┘     └──────────────────────────┘
```

**Core stack**: pandas, pyarrow, scikit-learn, rapidfuzz, datasketch (MinHash/LSH),
sparse_dot_topn, sentence-transformers, torch, LightGBM.
All MIT/Apache-licensed, well under 8B params.

---

## Cloud Architecture & High-Performance Resource Optimization

To maximize training speed and utilize cloud hardware resources (high vCPU counts, large RAM, and optional GPUs), the pipeline incorporates the following performance optimizations:

### 1. High-Throughput I/O & Memory-Mapped Storage
- **PyArrow Engine**: Replace slow Python TSV parsing with native C++ PyArrow readers (`engine="pyarrow"`), delivering **10x–50x faster I/O**.
- **Parquet / Feather Intermediate Caching**: Intermediate tables (normalized data, candidate pairs, feature matrices) are cached in compressed Parquet format with columnar layouts, slashing disk footprint and read latency.
- **In-Memory Streaming & Memory Budgeting**: Exploit 64GB–256GB cloud RAM to keep working country partitions in memory while maintaining chunked streaming boundaries to prevent Out-Of-Memory (OOM) failures.

### 2. Multi-vCPU Core Saturation
- **Zero-Overhead Multiprocessing**: Auto-detect all available cores (`n_workers = os.cpu_count()`) across all stages.
- **POSIX Fork / Process Pools**: On Linux cloud instances, multiprocessing takes advantage of low-latency POSIX fork semantics with shared memory.
- **Multi-Threaded Sparse Cosine Distance**: `sparse_dot_topn.awesome_cossim_topn` with `n_process=os.cpu_count()` for OpenMP-accelerated matrix multiplication.
- **Parallel Country Execution**: Process US, India, and France candidate blocks concurrently or with maximum core saturation per partition.

### 3. GPU & Mixed-Precision Vector Acceleration
- **CUDA FP16 Multilingual Embeddings**: When an NVIDIA GPU (T4, L4, V100, A10G/A100) is detected, `SentenceTransformer` switches to CUDA with half-precision (`fp16`) and large batches (2048–4096), accelerating encoding by **8x–12x**.
- **FAISS GPU/CPU Exact Search**: Exact inner-product search (`IndexFlatIP`) leveraging SIMD AVX-512 on CPU or tensor cores on GPU, eliminating blocking search bottlenecks.

### 4. Fast Feature Extraction Architecture
- **C++ SIMD String Distance (RapidFuzz)**: `rapidfuzz` releases the Python GIL and executes optimized C++ AVX2/AVX-512 kernels.
- **String Pair Deduplication & Memoization**: Business names repeat frequently across candidate pairs. Unique `(s1_name, cand_name)` pairs are extracted, evaluated once via RapidFuzz, and mapped back. This cuts pairwise string computations by **70%–80%**.
- **Chunked Feature Generation**: Candidate pairs are processed in parallel chunks across worker processes and saved directly as float32 NumPy arrays or Parquet tables.

### 5. Fast Model Training Pipeline ("Train Faster" Strategy)
- **Stratified Hard-Negative Subsampling**:
  - Raw blocking yields ~10M candidate pairs, but 80% are trivial negatives far from the decision boundary.
  - Subsample non-matching candidates to a **3:1 or 4:1 ratio per positive**, stratified by blocking similarity scores (keeping the hardest negatives and lookalikes).
  - Shrinks training size from ~10M to ~2.5M rows, speeding up training by **4x–5x** with zero loss in discriminative ability.
- **LightGBM Cloud Acceleration**:
  - `n_jobs=-1`: Saturates all available cloud vCPUs.
  - Histogram binning (`max_bin=255`): Extremely fast continuous feature binning.
  - Subsampling parameters: `bagging_fraction=0.8`, `bagging_freq=1`, `feature_fraction=0.8` (speeds up tree construction by 25%).
  - LightGBM binary `Dataset(..., free_raw_data=True)`: Reclaims raw memory immediately after binning.
  - Early stopping (`stopping_rounds=30`) on validation macro F_0.5 / AUC to prevent redundant iterations.
  - Target: **Full classifier training completes in 5–10 minutes** on a 16–32 vCPU instance.

---

## Phase 0: Foundation & EDA (Hours 0–6)

### Step 0.1 — Project Scaffold
- **Primary deliverables and execution architecture are standalone Python scripts (`.py`) + `pipeline.py`.**
  All pipeline stages are authored, tested, and run directly as modular, robust CLI scripts.
  `pipeline.py` serves as the end-to-end standalone orchestrator that runs the entire sequence
  and produces the required output files that pass `validate_submission.py`.
- **No mandatory companion notebooks:** We drop the requirement/overhead of maintaining
  or converting companion `.ipynb` notebooks for every stage.
- **Notebooks strictly on-demand for visual stats/charts only:** Jupyter notebooks are
  reserved exclusively for situations where we need to inspect and display data statistics,
  visual distributions, calibration curves, or error analysis charts for the methodology report.
- **Engineering rationale:**
  1. *Windows multiprocessing stability:* Standalone CLI scripts execute cleanly with
     `freeze_support()`, avoiding the deadlocks and IPC freezing caused by Windows `spawn`
     semantics inside the interactive IPython kernel namespace (documented in `struggle.md`).
  2. *Scale & memory efficiency:* CLI streaming scripts allow tight garbage collection
     and bounded RAM utilization across 12M+ records.
  3. *CLI reproducibility & automation:* Scripts can be easily executed, benchmarked,
     parameterized, and chained via shell commands.
- Code structure matching the submission package:
  ```
  code/business_entity_resolution/
  ├── src/
  │   ├── 00_eda.py                     # EDA, noise discovery, data profiling CLI
  │   ├── 00b_train_val_split.py        # stratified train/validation split CLI
  │   ├── 01_normalize.py               # text normalization pipeline (multi-core streaming)
  │   ├── 02_blocking.py                # candidate generation strategies & recall audit
  │   ├── 03_features.py                # pairwise feature engineering
  │   ├── 04_model_train.py             # classifier training + validation
  │   ├── 05_threshold_tuning.py        # threshold optimization + singleton calibration
  │   ├── 06_test_inference.py          # full pipeline on test set → output files
  │   ├── 07_error_analysis.py          # error analysis + ablation table
  │   ├── pipeline.py                   # final standalone orchestrator script → output TSVs
  │   └── (optional) stats_*.ipynb      # strictly on-demand if visual stats/plots are needed
  ├── README.md
  └── requirements.txt
  ```
- Build the **F_0.5 scorer** (macro-averaged):
  - Per S1 entity: precision = |predicted ∩ truth| / |predicted|, recall = |predicted ∩ truth| / |truth|
  - Singletons (truth = ∅): score = 1.0 if predicted = ∅, else 0.0
  - F_0.5 = (1.25 × P × R) / (0.25 × P + R), average across all S1 entities
- Build the **output writer** producing TSV files matching the exact format required:
  - `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids`
  - `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids`
  - Validated against `validate_submission.py` rules (tab-separated, correct headers,
    no duplicate rows, no S1 self-matches, S2/S3 IDs only, all S1 entities present)

### Step 0.2 — Train/Validation Split
- Hold out **10% of training S1 entities** as validation (stratified by country and
  match-count bracket)
- This is the **only** signal before final submission — no incremental leaderboard
  uploads planned
- Pull the corresponding S2/S3 matched records into the validation pool

### Step 0.3 — Systematic EDA
Run **programmatic profiling via `00_eda.py`** (not manual sampling), producing
structured frequency tables and summary reports (`eda_results.txt`). An optional
notebook (`stats_eda.ipynb`) is used only if visual inspection of distribution
plots or charts is needed for the methodology report:

- **Character-set profiling**: unique Unicode scripts in business_name per source
  (Latin, Devanagari, Tamil, Telugu, Malayalam, Kannada, etc.) — quantify how many
  records are non-Latin per source
- **Token frequency analysis**: most/least common tokens in business_name and
  business_address, per source. Surfaces legal suffixes, structural words, and typos
- **Regex noise scans** with counts:
  - Hashtag/special prefixes: `^[#<]+`
  - Appended digits: `\b\d{7,}\b` or `- \d+$` in names
  - URLs: `www\.|\.com|http|@`
  - Duplicate adjacent words: `\b(\w+)\s+\1\b`
  - Extra whitespace: `\s{2,}`
  - "null" literals in addresses
  - Accent anomalies on common English words (café vs Léarning)
- **Missing field rates**: empty business_address per source/country
- **Address format analysis** per country:
  - US: ZIP code presence, state abbreviation format
  - India: PIN code presence, landmark phrases ("near", "opp", "behind")
  - France (test only): postal code format, "Rue"/"Boulevard" patterns
- **Cross-script match analysis**: how many ground-truth pairs involve a
  non-Latin S2/S3 name matched to a Latin S1 name (quantifies the cross-script
  problem's true scope)

### Step 0.4 — Document Findings
- Create a noise catalog with frequency counts
- Prioritize: high-frequency patterns get normalization rules; rare patterns
  get noted but not over-fitted
- Identify which noise types are surface-level (strippable) vs. semantic
  (need embedding/transliteration)

---

## Phase 1: Normalization + Blocking (Hours 6–18)

### Step 1.1 — Text Normalization Pipeline
Applied identically to all sources (S1, S2, S3, train and test):

**Case normalization:**
- Lowercase everything (S2 is ALL-CAPS, S1/S3 are mixed-case)

**Name cleaning:**
- Strip hashtag/angle-bracket prefixes: `#`, `<<`, `##`
- Remove appended numeric IDs: `- \d+$`, `#\d+$`
- Remove URL fragments: `| www...`, `.com` suffixes in names
- Normalize legal suffixes to canonical forms:
  - Corp/Corporation → corporation, Ltd/Limited → limited,
    Pvt/Private → private, LLC/L.L.C. → llc, Inc → inc,
    LLP → llp, SARL → sarl, SCI → sci, SAS/S.A.S → sas
- Strip parentheses around legal suffixes: "(LLC)" → "llc", "(Co)" → "co"
- Normalize `&` → "and"
- Collapse "Pvt." with period handling
- **"dba" aliases — resolved**: split on `dba` into `legal_name` and `alias_name`
  fields; generate similarity features against **both** the S1 candidate name,
  keep `max(sim(legal_name), sim(alias_name))` as the name-similarity feature.
  A trade name is a legitimate alternate identity for the same business, not
  noise to discard — collapsing it into one string or picking one side loses
  real matching signal
- Collapse duplicate adjacent words: "Crestline Crestline" → "Crestline"
- Strip extra whitespace, trim

**Address cleaning:**
- Expand common abbreviations: St→Street, Rd→Road, Ave→Avenue, Blvd→Boulevard,
  Dr→Drive, Ln→Lane, Ct→Court, Pl→Place, Cir→Circle
- Remove "null" string literals
- Extract structured sub-fields where possible:
  - PIN/ZIP code (if present)
  - State/region
  - City
- Keep landmark phrases ("near", "opp", "behind") in a separate field rather
  than discarding

**Unicode handling:**
- NFKD normalization
- Fix mojibake if detected (ftfy)
- Strip accents on Latin characters that appear to be noise (Léarning→Learning,
  Nétwork→Network, Cáre→Care) — but preserve legitimate accents in French names
  - Heuristic: if the word without accent is a common English word, strip; otherwise keep
  - Or simpler: create both accent-stripped and accent-preserved versions for matching

**Script handling:**
- Flag records with non-Latin business_name (Devanagari, Tamil, Telugu, Kannada, Malayalam)
- These records need embedding-based matching, not character-level similarity
- Keep the original script name alongside any transliterated version

**Rule-level instrumentation (for later ablation):**
- Implement each normalization rule as an independently toggleable function
  (suffix mapping, abbreviation expansion, hashtag/ID stripping, duplicate-word
  collapsing, accent handling, etc.), not one monolithic cleaning function
- Tag which rule(s) fired on each record, so individual rule contribution can
  be measured later (Step 5.3) rather than only measuring aggregate
  noise-frequency reduction

### Step 1.2 — Multi-Strategy Blocking

The blocking stage must produce `candidate_pairs.tsv` — the exact set of
candidates fed to the matching model. Blocking recall **caps** final recall;
anything missed here is lost forever.

**Strategy 1: Exact/near-exact normalized-name hash join**
- Near-zero cost: group S1 and S2/S3 records by fully-normalized business name
  (and, optionally, normalized name + city) and join directly on that key
- Catches every easy, unambiguous match before any fuzzy method runs
- Run this first — it's the cheapest possible pass and immediately shrinks the
  volume the more expensive strategies below need to handle

**Strategy 2: Country-filtered blocking (HARD FILTER)**
- Only compare S1 entities to S2/S3 entities with the same `country` value
- This is safe because the country field is clean (confirmed from data)
- **Massive reduction ratio**: splits the 9.97M candidate pool into
  ~3 country-specific pools

**Strategy 3: MinHash/LSH on character n-gram shingles of normalized business_name**
- Use `datasketch.MinHashLSH` with character 3-grams
- Threshold tuned for high recall (e.g., Jaccard ≥ 0.2–0.3)
- This is the cheap, coarse first pass — runs in O(n) per query
- Handles: typos, abbreviation differences, word reordering (partially)

**Strategy 4: TF-IDF cosine on normalized business_name (char n-grams)**
- Build TF-IDF matrix with char n-grams (3,5) on all S1+S2+S3 records
  within each country partition
- Use `sparse_dot_topn` to find top-k similar S2/S3 records per S1 record
  within a threshold
- Cap `max_features` to control memory on multilingual vocabulary
- Down-weights common tokens (Inc, Corp, Private, Limited)

**Strategy 5: Exact/near-exact blocking on distinctive name tokens**
- Extract distinctive name tokens (after removing legal suffixes and
  stopwords) — match S1 to S2/S3 on shared rare tokens
- Also: exact PIN/ZIP match as a boosting signal

**Strategy 6: Address token overlap (independent index)**
- Build a token-overlap index over normalized `business_address` alone,
  separate from every name-based strategy above
- Catches cases where the business name is badly garbled or missing but the
  address is largely intact (and the reverse: address missing/garbled but
  name intact) — a matching mechanism none of the name-based strategies above
  can reach on their own
- Cheap: same inverted-index machinery as Strategy 5, just keyed on address
  tokens (street name, house/building number, city, PIN/ZIP) instead of name
  tokens

**Strategy 7: Embedding-based blocking (for cross-script matches)**
- Small multilingual sentence-transformer model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`,
  Apache 2.0 licensed, supports Latin/Devanagari/Tamil/Telugu/Malayalam)
- Encode all business_name fields; use approximate nearest neighbors (FAISS or hnswlib)
  to find top-k similar names per S1 entity
- Batched encoding (1000–5000 records at a time) to manage memory
- Specifically targets Devanagari/Tamil S2/S3 names that must match Latin S1 names

**Union all candidates** across strategies, deduplicate, cap at top ~50 per S1 entity.

**Measure blocking recall ceiling** on validation split:
- Target: ≥ 95% recall at the blocking stage
- If below target, identify missed matches and add targeted blocking strategies
- Report recall ceiling, total candidates, and reduction ratio

### Step 1.3 — Memory Management Strategy
- Process blocking in **country-level batches** (US, India, France separately)
- Within each country, process in chunks if needed (e.g., 100k S1 entities at a time)
- Clear intermediate matrices between blocking strategies
- Profile peak memory usage and adjust batch sizes accordingly

---

## Phase 2: Feature Engineering + Matching Model (Hours 18–30)

### Step 2.1 — Pairwise Feature Engineering
For each (S1 entity, candidate) pair from blocking:

**Name similarity features:**
- Levenshtein normalized similarity (rapidfuzz)
- Token sort ratio (handles word-order transposition)
- Token set ratio (handles subset/superset relationships)
- Jaccard similarity on name tokens
- TF-IDF cosine similarity (reuse from blocking — free feature)
- Exact match flag (after normalization)
- Name length ratio
- Shared rare token count (tokens with TF-IDF weight above threshold)

**Address similarity features:**
- Token Jaccard on address
- Levenshtein on normalized address
- Exact PIN/ZIP match (boolean)
- Numeric token overlap (house/street numbers)
- City exact match (if extractable)
- State/region exact match
- Address completeness flag (both have address vs. one missing)

**Embedding features:**
- Cosine similarity of business_name embeddings (from blocking model)
- Useful especially for cross-script pairs

**Cross/meta features:**
- Source flag (S2 vs S3 — different noise profiles)
- Name similarity × address similarity interaction
- Max(name_sim, embedding_sim) — lets model use best available signal
- Address missing indicator (when one/both addresses are empty, name features
  must carry more weight — let the model learn this)

### Step 2.2 — Training Data Construction & Stratified Hard-Negative Mining
- **Positive pairs**: all ground-truth (S1, matched_S2/S3) pairs from the training split.
- **Stratified Hard-Negative Subsampling (Key to Fast Training)**:
  - Blocking yields 10M+ candidate pairs across all entities, but training on all 10M pairs is redundant and slow.
  - Subsample non-matching candidates to a **3:1 or 4:1 negative-to-positive ratio**, stratified by blocking rank/similarity score (preserving high-utility borderline negatives while dropping uninformative easy negatives).
  - Reduces feature extraction & training dataset from ~10M to ~2.5M rows — a **75% reduction in compute** with zero loss in discrimination.
- **Fast Storage**: Save train and validation feature matrices as compressed Parquet files (`train_features.parquet`, `val_features.parquet`) with float32 types to minimize memory and eliminate TSV overhead.

### Step 2.3 — High-Speed Model Training (Cloud-Optimized LightGBM)
- **LightGBM** gradient boosting classifier configured for maximum cloud speed:
  - **Thread Saturation**: `n_jobs=-1` (uses all 16/32/64 cloud vCPUs).
  - **Histogram Acceleration**: `max_bin=255`, enabling fast SIMD binning of continuous features.
  - **Tree & Feature Subsampling**: `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1` (reduces per-iteration split evaluation time by ~25% and curbs overfitting).
  - **Memory Optimization**: `free_raw_data=True` inside `lgb.Dataset` to immediately reclaim raw feature memory.
  - **Early Stopping**: Stop boosting if validation metric does not improve for 30 consecutive rounds (prevents wasted iterations).
  - **Training Speed**: Completes training in **under 5–10 minutes** on cloud instances.
- **Objective & Metric**: Binary logloss objective with sample weighting to prioritize high-precision separation; validation monitored on macro-averaged F_0.5.

### Step 2.4 — Model Validation
- ROC-AUC, PR-AUC on validation set (independent of threshold)
- Feature importance analysis
- Error analysis: inspect highest-confidence false positives and false negatives

### Step 2.5 — Probability Calibration
- Raw gradient boosting output is a score, not a true probability — check
  calibration with a reliability diagram (predicted probability bucket vs.
  observed match rate) before trusting the threshold sweep
- Fit **isotonic regression** (or Platt/sigmoid scaling as a simpler
  alternative) on a held-out calibration slice of the validation split, on
  top of the raw classifier output
- Re-check calibration curve post-fit; use the **calibrated** probability,
  not the raw score, for all downstream thresholding (Phase 3) — an
  uncalibrated score can make the "optimal" threshold unstable or
  non-transferable if the score distribution shifts slightly on test data
- Report calibration quality (e.g., Brier score, calibration curve) in the
  methodology doc — a small addition that signals statistical maturity
  beyond just picking a threshold that happens to score well

---

## Phase 3: Threshold Tuning + Singleton Calibration (Hours 30–36)

### Step 3.1 — Threshold Optimization
- Sweep decision thresholds from 0.1 to 0.9 on validation set, using the
  **calibrated** probability from Step 2.5
- Compute **macro-averaged F_0.5** at each threshold
- Select threshold that maximizes F_0.5 (not accuracy, not F1)
- Expect optimal threshold to be relatively high (conservative, precision-heavy)

### Step 3.2 — Singleton Calibration
- Analyze model score distribution for true singletons (ground truth empty)
- True singletons should have all candidate scores below threshold
- Check if explicit singleton logic is needed (e.g., "if max candidate score < T2, predict empty")
- Singletons are ~5.6% but each correctly predicted singleton scores 1.0

### Step 3.3 — Group-Consistency Audit
- After thresholding, check whether any single S2/S3 record is claimed as a
  match by **multiple** S1 entities — many-to-one is technically allowed by
  the problem statement, but a high rate of it usually signals an
  overly-loose threshold or a blocking/feature issue rather than genuine
  business structure
- Quantify: % of matched S2/S3 IDs claimed by more than one S1 entity, and
  whether these tend to be low-confidence claims
- **Resolution rule (if the rate is non-trivial):** when a record is claimed
  by multiple S1 entities, keep the claim only for the S1 entity with the
  highest calibrated match probability; drop the record from the
  lower-confidence claim(s) rather than leaving both (protects precision,
  which F_0.5 weights 2× over recall)
- Document the audit result either way — even "checked, rate was negligible,
  no resolution rule needed" is worth stating explicitly in the methodology
  doc, since a reviewer may specifically look for this

### Step 3.4 — Validation F_0.5
- Run full pipeline on validation split with tuned threshold
- Report overall F_0.5 and breakdown by:
  - Country (US vs India)
  - Singleton vs non-singleton
  - Match count bracket (1 match vs 3+ matches)
  - Cross-script vs same-script pairs

---

## Phase 4: Test Inference + Robustness (Hours 36–42)

### Step 4.1 — Robustness Audit
- **Country-agnostic check**: grep for hardcoded "US", "India", ZIP/PIN regex lengths
- **France readiness**: verify normalization handles SARL/SCI/SAS, Rue/Boulevard, accents
- **Empty field handling**: pipeline doesn't crash on empty business_address
- **Edge case review**: very long/short names, all-numeric tokens

### Step 4.2 — Test Set Inference
- Load all test data (S1: 1.73M, S2: 4.89M, S3: 5.08M)
- Run full pipeline: normalize → block → featurize → classify → threshold
- Produce `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- Process in batches by country to manage memory

### Step 4.3 — Output Validation
- Run `validate_submission.py`:
  ```bash
  python utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test
  ```
- Verify: every test S1 entity has exactly one row, no duplicate IDs,
  S2/S3 IDs only, matched IDs ⊆ candidate IDs
- Sanity checks: distribution of match counts, singleton rate, per-country match rates

---

## Phase 5: Error Analysis + Iteration (Hours 42–46)

### Step 5.1 — Targeted Error Analysis
Executed via `07_error_analysis.py` (CLI script), computing detailed confusion matrices,
error breakdowns, and ablation numbers. An optional notebook (`stats_error_analysis.ipynb`)
is used only if rendering visual error distribution charts or precision-recall curves
for the methodology document:
- **Worst-performing S1 entities**: what do they have in common?
- **False positives**: same-name-different-address? Brand lookalikes?
- **False negatives**: missed by blocking (recall ceiling) or rejected by classifier?
- **Cross-script pairs**: recall on Devanagari/Tamil S2/S3 names?

### Step 5.2 — Targeted Fixes
- Blocking recall bottleneck → adjust LSH thresholds, add strategies
- False positives dominate → raise threshold, add discriminative features
- Cross-script recall low → improve embedding model or add phonetic features
- Re-run validation after each fix to confirm improvement

### Step 5.3 — Ablation Table

**Pipeline-stage ablation:**
| Configuration | Val F_0.5 |
|---|---|
| Blocking only (threshold on blocking score) | ? |
| + Feature engineering | ? |
| + Full model (LightGBM) | ? |
| + Calibration | ? |
| + Threshold tuning | ? |
| + Singleton calibration | ? |

**Normalization-rule ablation** (using Step 1.1's rule-level instrumentation):
run the full pipeline with each rule toggled off individually (all else fixed)
and record the F_0.5 drop — turns "we handled noise" into "we proved which
noise-handling rules actually moved the score," and directly flags any rule
that's not pulling its weight (candidate to drop or simplify):
| Rule disabled | Val F_0.5 | Δ vs. full |
|---|---|---|
| Suffix normalization | ? | ? |
| Abbreviation expansion | ? | ? |
| Hashtag/ID stripping | ? | ? |
| Duplicate-word collapsing | ? | ? |
| Accent/Unicode handling | ? | ? |
| dba alias splitting | ? | ? |
| Embedding-based blocking (cross-script) | ? | ? |

---

## Phase 6: Documentation + Final Packaging (Hours 46–50)

### Step 6.1 — Fill in Documentation_template.md
- **Executive Summary**: 2-3 sentence overview
- **Problem Analysis**: noise catalog with frequency counts
- **Solution Strategy**: blocking → classifier → threshold tuning
- **Candidate Generation**: strategies, recall ceiling, reduction ratio
- **Matching Model**: features, LightGBM, threshold selection
- **Results**: validation F_0.5, error analysis, ablation table
- **France Robustness**: explicit discussion
- **Code Artefacts**: structure and entry points

### Step 6.2 — Finalize Code Package
- Clean up `src/`, add docstrings
- Pin versions in `requirements.txt`
- Write reproduction `README.md` with exact steps
- Verify clean-environment reproducibility

### Step 6.3 — Assemble Submission
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches
│   └── candidate_pairs.tsv         # blocking candidates
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── README.md               # reproduction instructions
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled-in methodology
```

### Step 6.4 — Final Validation
- Run `validate_submission.py` one last time
- Sanity-check zip structure matches required format

---

## Key Risk Mitigations

| Risk | Mitigation |
|---|---|
| **Blocking recall ceiling too low** | Measure after every blocking strategy; iterate before moving to modeling |
| **Memory exhaustion on ~10M records** | Country-level batches, chunked processing, sparse matrices, capped vocab |
| **Cross-script matches missed** | Embedding-based blocking added early, not deferred |
| **France fails silently** | Explicit country-agnostic audit; French suffix/address normalization |
| **Threshold optimized for wrong metric** | F_0.5 scorer built in Phase 0; threshold tuning uses exactly this metric |
| **Output format rejection** | Run `validate_submission.py` after every pipeline run, not just at end |
| **Singletons mishandled** | Dedicated singleton analysis; 5.6% of entities with outsized F_0.5 impact |
| **Threshold unstable/non-transferable** | Calibrate classifier output (isotonic/Platt) before thresholding, not raw score |
| **Same S2/S3 record claimed by multiple S1 entities** | Group-consistency audit post-threshold; keep highest-confidence claim only if rate is non-trivial |
| **`dba`/trade-name signal lost** | Split into legal/alias names; feature uses max similarity across both, not one discarded |
| **Can't tell which normalization rules actually help** | Rule-level instrumentation + per-rule ablation (Step 5.3) |
| **Documentation rushed** | Draft methodology sections progressively during each phase |

---

## What Makes This Submission Stand Out

1. **Explicit blocking recall-ceiling measurement** with recall-vs-candidates data
2. **Multi-strategy blocking** with measured contribution of each strategy
3. **Cross-script handling** — embedding-based blocking for Devanagari/Tamil/Telugu/
   Malayalam names, not just a lexical pipeline
4. **Precision-heavy threshold tuning** specifically for F_0.5, on a
   **calibrated** probability (isotonic/Platt), not a raw classifier score —
   makes the chosen threshold statistically defensible, not just empirically
   convenient
5. **Singleton calibration** — separate treatment for the 5.6% that should predict empty
6. **Demonstrated France robustness** — no hardcoded country assumptions
7. **Two-level ablation table** — pipeline-stage contribution AND per-rule
   normalization contribution, showing not just that noise was handled but
   which specific rules actually moved the F_0.5 score
8. **Country-conditional error analysis**
9. **Group-consistency audit** — explicit check (and resolution rule, if
   needed) for S2/S3 records claimed by multiple S1 entities, protecting
   precision rather than assuming many-to-one claims are always benign
10. **Deliberate `dba`/trade-name handling** — resolved as a dual-name
    similarity feature rather than left as an open question, preserving
    real matching signal instead of discarding or arbitrarily picking one
    name

---

## Constraints Checklist

- [ ] Output: tab-separated `.tsv` files with exact column names
- [ ] Every test S1 entity has exactly one row
- [ ] `matched_entity_ids` only contains S2/S3 IDs from test set
- [ ] No duplicate IDs within any list; no duplicate S1 rows
- [ ] `candidate_pairs.tsv` produced alongside `matching_results.tsv`
- [ ] Final model: MIT/Apache 2.0 license, ≤8B parameters
- [ ] No external data lookup (no APIs, no databases, no geocoding)
- [ ] `validate_submission.py` passes before submission
- [ ] Submission zip matches required structure exactly

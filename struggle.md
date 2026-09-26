# Engineering Log: Challenges, Struggles & Solutions

This document tracks all architectural, algorithmic, operational, and platform challenges encountered during the development of the Business Entity Resolution pipeline, along with the root-cause analyses and exact solutions implemented.

---

## 1. Data Scale & Execution Bottlenecks

### The Challenge
- The dataset is massive: **Train-S1 (2.2M rows)**, **Train-S2 (5.0M rows)**, and **Train-S3 (5.3M rows)**, plus test sets.
- Initial single-threaded, row-by-row string normalization via Pandas `apply()` or sequential Python loops was taking **45–60 minutes per source file**, stalling development and feedback loops.
- GPU acceleration was considered, but text normalization (regex substitution, Unicode NFKD decomposition, character filtering) consists of CPU-bound string manipulation in Python/C extensions, where transferring small strings to GPU VRAM introduces excessive PCIe overhead rather than speedups.

### The Struggle
- Attempted to scale across CPU cores using Python `multiprocessing.Pool` and `concurrent.futures.ProcessPoolExecutor`.
- However, when run inside Jupyter Notebook on Windows, execution completely froze at chunk 1:
  ```text
  [Train-S1 chunk 1] Normalizing 500,000 business names across 15 workers ...
  [Train-S1 chunk 1] Names:   0%|          | 0/500000 [00:00<?, ?it/s]
  ```

### Root Cause
1. **Windows Process Spawning Semantics (`spawn` vs `fork`)**: Unlike Linux which uses `fork()`, Windows uses `spawn` to start child processes. Child processes re-import the calling script/module from the top.
2. **Jupyter Interactive Namespace**: In Jupyter (`ipykernel`), there is no standard `__main__` module. Spawned worker processes attempted to bootstrap inside the IPython interactive environment, creating recursion locks and deadlocking the IPC pipes.

### How We Overcame It
1. **Chunked Streaming Pipeline**: Partitioned files into manageable chunks of `500,000` rows using `pd.read_csv(..., chunksize=CHUNK_SIZE)` to keep RAM utilization bounded and predictable.
2. **Terminal Script Execution**: Shifted the heavy multiprocessing workload from the interactive Jupyter kernel to a standalone CLI script (`01_normalize.py`).
3. **Execution Guard & `freeze_support`**: Wrapped all top-level pipeline code inside `def main():` and guarded with `if __name__ == '__main__': multiprocessing.freeze_support(); main()`.
4. **Result**: Full 15-core CPU saturation achieved, normalizing **500,000 records in ~1.7 seconds (>250,000 records/sec)**.

---

## 2. Windows Inter-Process Communication (IPC) & Memory Bloat

### The Challenge
- In `_parallel_map()`, calling `ex.submit(fn, item)` in a loop over 500,000 records created:
  - 500,000 individual `concurrent.futures.Future` objects in memory.
  - 500,000 separate IPC serialization events across Windows process boundaries.
- This caused significant latency spikes before processing even started and risked memory exhaustion.

### How We Overcame It
- Refactored `_parallel_map` to use chunked `ex.map(fn, items, chunksize=chunksize)` (e.g. `chunksize=2000`):
  ```python
  with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
      return list(tqdm(ex.map(fn, items, chunksize=chunksize), desc=desc, total=n))
  ```
- This reduced IPC transactions from **500,000 individual calls to ~250 batched calls per chunk**, while preserving exact ordering and enabling low-overhead real-time `tqdm` progress tracking.

---

## 3. Offline / Air-Gapped Environment Constraints

### The Challenge
- Attempts to install helper libraries like `joblib` (for its robust `loky` multiprocessing backend) failed due to network isolation:
  ```text
  pip install joblib
  WARNING: Retrying ... Failed to establish a new connection: [Errno 11001] getaddrinfo failed
  ```

### How We Overcame It
- Enforced a **zero-external-dependency policy for concurrency**:
  - Replaced third-party parallel libraries with standard library modules: `concurrent.futures`, `multiprocessing`, `functools`.
  - Avoided introducing dependencies that require internet access or C compilation not present in the local virtual environment.

---

## 4. EDA Calibration vs. Real-World Rule Fire Rates

### The Challenge
- Sanity checks calibrated directly against raw exploratory data analysis (EDA) noise scans initially raised false warning alarms:
  ```text
  ✗ lowercase          : 0.8519  (expected 0.150–0.250)
  ✗ collapse_whitespace: 0.1454  (expected 0.080–0.140)
  ✗ strip_accent_noise : 0.1507  (expected 0.040–0.080)
  WARNING: Some fire rates outside expected range — review normalization rules.
  ```

### Root Cause
- **`lowercase`**: EDA counted words with 4+ consecutive capital letters (`all_caps_word` = 19.89%). But in normalization, `rule_lowercase` fires if *any* uppercase character exists (e.g., standard Title Case names like "Tata Motors"). Almost all business names (~85%) contain uppercase letters.
- **`collapse_whitespace`**: EDA detected 2+ consecutive spaces (11.01%), but `rule_collapse_whitespace` also strips leading/trailing spaces and tab characters.
- **`strip_accent_noise`**: EDA scanned precomposed Unicode characters (U+00C0–00FF = 5.77%). However, because normalization applies NFKD decomposition first, decomposed combining diacritical marks are exposed and stripped, legitimately increasing the fire rate to ~15%.

### How We Overcame It
- Recalibrated `EXPECTED_FIRE_RATES_S2` to reflect true pipeline mechanics rather than raw scan heuristics:
  - `lowercase`: updated to `(0.50, 0.95)`
  - `collapse_whitespace`: updated to `(0.08, 0.18)`
  - `strip_accent_noise`: updated to `(0.08, 0.20)`
- Documented the exact semantic rationale in comments so future ablations and rule evaluations remain transparent.

---

## 5. Domain-Specific Noise Patterns in Business Names & Addresses

### The Challenges
1. **Email and URL Bleed**: Names frequently contained full or embedded URLs (`shivshakti.com`) and contact emails (`info@corp.com`).
2. **Indian Entity Artifacts**: Indian entity names often had honorific/business prefixes like `M/s` (Mesdames/Messrs) or location suffixes in parentheses like `(India)` or `(Ahmedabad)`.
3. **DBA (Doing Business As) Aliases**: Records contained dual names concatenated with `DBA`, `D/B/A`, `T/A`, or `F/K/A`.
4. **Missing Addresses**: 3.3% of S2 and S3 records lacked addresses entirely, whereas S1 had 0% missing addresses.
5. **Cross-Script Entities**: 7.26% of ground-truth matches were cross-script (e.g., Latin name in S1 vs. Devanagari/Tamil in S2/S3).

### How We Overcame It
1. **Rule Isolation & Tagging**:
   - Implemented `rule_strip_email` and `rule_strip_urls` to strip embedded web references while preserving real entity tokens.
   - Built `rule_strip_ms_prefix` to strip `M/s`, `M/S`, `Messrs`.
   - Extracted parenthetical geography via `rule_strip_geo_parenthetical`.
2. **Dual-Entity Preservation**:
   - `rule_split_dba` splits legal and trade names into `name_clean` and `alias_clean`, preventing loss of match signal.
3. **Routing Flags for Downstream Blocking**:
   - Added `has_addr` flag: allows blocking to fall back to name-only indexing for records without addresses.
   - Added `has_non_latin` and `scripts_detected` flags: routes cross-script entities to multilingual dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) in Phase 2.
   - Tagged `name_rules_fired` and `addr_rules_fired` on every record to enable granular ablation studies (Phase 5).

---

## 6. Development Workflow: Terminal Performance vs. Notebook Visibility

### The Challenge
- Running inside Jupyter Notebook caused freezing on Windows, but running in the terminal lacked the interactive exploration, dataframe inspection, and persisted cell outputs convenient for notebooks.

### How We Overcame It
Provided a clean three-tiered execution strategy:
1. **In-Notebook Subprocess (`!python -u ...`)**: Runs the standalone script from a notebook cell as an external OS process, bypassing the Windows interactive kernel freeze while streaming live progress directly into the notebook.
2. **PowerShell Pipeline with `Tee-Object`**:
   ```powershell
   python -u code\business_entity_resolution\src\01_normalize.py | Tee-Object -FilePath "normalize_run.log"
   ```
   Outputs live to the terminal and simultaneously writes a persistent log that the notebook can read and inspect.
3. **Separation of Compute and Analysis**: Heavy data generation runs headlessly to parquet/tsv files; notebooks are reserved for inspecting distributions, validating edge cases, and tuning parameters.

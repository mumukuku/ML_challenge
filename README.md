# Business Entity Resolution — Code Package

## Quick Start

```bash
# 1. Install dependencies (from repo root)
pip install -r code/business_entity_resolution/requirements.txt

# 2. (Phase 0 sanity check) Run the stub pipeline — writes empty but valid output files
python code/business_entity_resolution/src/pipeline.py \
    --data-dir dataset \
    --out-dir  code/business_entity_resolution/output

# 3. Validate the output format
python utils/validate_submission.py \
    --matching code/business_entity_resolution/output/matching_results.tsv \
    --candidate code/business_entity_resolution/output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline Execution Order (CLI Scripts)

Run the standalone Python scripts **in order** via terminal, or execute `pipeline.py`
for end-to-end orchestration. Notebooks are not required for pipeline execution and are
reserved strictly for on-demand visual inspection of stats and diagnostic plots.

| Script | Phase | Purpose |
|---|---|---|
| `00b_train_val_split.py` | 0.2 | Stratified 90/10 train/val split |
| `00_eda.py` | 0.3–0.4 | EDA, noise discovery, frequency counts |
| `01_normalize.py` | 1.1 | Text normalization pipeline (multi-core streaming) |
| `02_blocking.py` | 1.2 | Multi-strategy candidate generation & recall audit |
| `03_features.py` | 2.1 | Pairwise feature engineering |
| `04_model_train.py` | 2.2–2.4 | LightGBM classifier training & validation |
| `05_threshold_tuning.py` | 3.1–3.3 | Probability calibration & F_0.5 threshold selection |
| `06_test_inference.py` | 4.1–4.3 | Full test inference + output validation |
| `07_error_analysis.py` | 5.1–5.3 | Error analysis & ablation benchmarks |

## Directory Layout

```
code/business_entity_resolution/
├── src/
│   ├── 00_eda.py                 # EDA, noise discovery, data profiling
│   ├── 00b_train_val_split.py    # Stratified 90/10 train/val split
│   ├── 01_normalize.py           # Text normalization pipeline
│   ├── 02_blocking.py            # Candidate generation strategies
│   ├── 03_features.py            # Pairwise feature engineering
│   ├── 04_model_train.py         # Classifier training + validation
│   ├── 05_threshold_tuning.py    # Threshold optimization + singleton calibration
│   ├── 06_test_inference.py      # Full pipeline on test set → output files
│   ├── 07_error_analysis.py      # Error analysis + ablation table
│   ├── pipeline.py               # Standalone end-to-end orchestrator script
│   └── (optional) stats_*.ipynb  # Visual statistics / diagnostic plots only
├── output/
│   ├── matching_results.tsv      # (generated) final matches
│   └── candidate_pairs.tsv       # (generated) blocking candidates
├── README.md                     # this file
└── requirements.txt              # pinned dependencies
```

## Metric

**Macro-averaged F_0.5** (precision-weighted 2× over recall):

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)   per S1 entity, averaged across all.
Singletons (truth = ∅): score = 1.0 if predicted = ∅, else 0.0.
```

The `pipeline.py` module exports `f05_per_entity()`, `macro_f05()`,
and `score_from_tsv()` for use across scripts and evaluation routines.

## Design Constraints

- **Country-agnostic**: no hardcoded ZIP/PIN regex lengths, no country whitelists.
  France (unseen in training) must be handled transparently.
- **License**: MIT/Apache 2.0 only. No external API calls.
- **Memory**: country-level batches; chunked processing for the full ~10M record pool.
- **Output**: tab-separated UTF-8 TSV, validated by `utils/validate_submission.py`.

## Cloud Execution & Hardware Acceleration

The pipeline is engineered to scale across cloud infrastructure (AWS/GCP/Azure/Lambda):

1. **Multi-vCPU Scaling**: Every script automatically leverages `os.cpu_count()` (tested up to 96 vCPUs).
2. **GPU Acceleration**: Multilingual embedding extraction in `02_blocking.py` automatically uses PyTorch CUDA with FP16 and large batching (2048+) when an NVIDIA GPU is present.
3. **Fast Training Strategy**:
   - Stratified hard-negative subsampling (3:1 to 4:1 ratio) reduces candidate pair training volume by 75% while preserving boundary discrimination.
   - LightGBM runs with `n_jobs=-1`, histogram binning (`max_bin=255`), and early stopping (30 rounds), completing model training in **under 5–10 minutes**.
4. **Accelerated I/O**: `pyarrow` engine is enabled for high-speed TSV and columnar Parquet operations.

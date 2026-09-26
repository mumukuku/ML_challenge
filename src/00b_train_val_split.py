# %% [markdown]
# # 00b — Train / Validation Split
#
# **Phase 0, Step 0.2** — Hold out 10% of training S1 entities for validation.
#
# Stratification: by `country` × `match_count_bracket` (0, 1, 2, 3-5, 6+).
#
# Outputs saved to `dataset/splits/`:
# - `val_s1_ids.txt`         — validation S1 entity IDs (one per line)
# - `train_s1_ids.txt`       — training S1 entity IDs
# - `val_ground_truth.tsv`   — GT rows for validation entities
# - `train_ground_truth.tsv` — GT rows for training entities

# %%
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(os.getcwd())
for p in [Path(os.getcwd()), *Path(os.getcwd()).parents]:
    if (p / "dataset").exists():
        REPO_ROOT = p
        break

TRAIN_DIR  = REPO_ROOT / "dataset" / "train"
SPLITS_DIR = REPO_ROOT / "dataset" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
VAL_FRACTION = 0.10

print(f"REPO_ROOT  : {REPO_ROOT}")
print(f"SPLITS_DIR : {SPLITS_DIR}")

# %% [markdown]
# ## 1. Load Source 1 and Ground Truth

# %%
print("Loading train_source1.tsv ...")
s1 = pd.read_csv(
    TRAIN_DIR / "train_source1.tsv",
    sep="\t", dtype=str, low_memory=False,
    usecols=["entity_id", "country"],
)
s1 = s1.rename(columns={"entity_id": "source1_entity_id"})
print(f"  S1 rows: {len(s1):,}")

print("Loading train_ground_truth.tsv ...")
gt = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t", dtype=str, low_memory=False,
)
print(f"  GT rows: {len(gt):,}")
print(f"  GT columns: {list(gt.columns)}")

# %% [markdown]
# ## 2. Compute match_count_bracket for stratification

# %%
def match_count_bracket(raw):
    if not isinstance(raw, str) or not raw.strip():
        return "0"       # singleton
    n = len(raw.split(","))
    if n == 1:  return "1"
    if n == 2:  return "2"
    if n <= 5:  return "3-5"
    return "6+"

gt["bracket"] = gt["matched_entity_ids"].apply(match_count_bracket)

# Merge country into GT
gt = gt.merge(s1[["source1_entity_id", "country"]], on="source1_entity_id", how="left")

# Stratification key
gt["strat_key"] = gt["country"].fillna("UNKNOWN") + "__" + gt["bracket"]

print("\nStratification key distribution:")
print(gt["strat_key"].value_counts().to_string())

# %% [markdown]
# ## 3. Stratified split

# %%
X = gt.index.to_numpy().reshape(-1, 1)
y = gt["strat_key"].to_numpy()

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
train_idx, val_idx = next(sss.split(X, y))

gt_train = gt.iloc[train_idx].copy()
gt_val   = gt.iloc[val_idx].copy()

print(f"Training set  : {len(gt_train):,} S1 entities")
print(f"Validation set: {len(gt_val):,} S1 entities ({len(gt_val)/len(gt)*100:.1f}%)")

# Verify stratification quality
train_dist = gt_train["strat_key"].value_counts(normalize=True).rename("train")
val_dist   = gt_val["strat_key"].value_counts(normalize=True).rename("val")
strat_check = pd.concat([train_dist, val_dist], axis=1).round(4)
print("\nStratification check (proportions):")
print(strat_check.to_string())

# %% [markdown]
# ## 4. Save split files

# %%
out_cols = ["source1_entity_id", "matched_entity_ids"]

# Ground truth subsets
gt_train[out_cols].to_csv(SPLITS_DIR / "train_ground_truth.tsv", sep="\t", index=False)
gt_val[out_cols].to_csv(SPLITS_DIR / "val_ground_truth.tsv", sep="\t", index=False)

# ID lists
(SPLITS_DIR / "train_s1_ids.txt").write_text(
    "\n".join(gt_train["source1_entity_id"].tolist()), encoding="utf-8"
)
(SPLITS_DIR / "val_s1_ids.txt").write_text(
    "\n".join(gt_val["source1_entity_id"].tolist()), encoding="utf-8"
)

# Country composition summary
summary = {
    "train": gt_train["country"].value_counts().to_dict(),
    "val":   gt_val["country"].value_counts().to_dict(),
}
print("\nSaved to", SPLITS_DIR)
print("Split files:")
for f in sorted(SPLITS_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

print("\nCountry composition:")
for split_name, counts in summary.items():
    print(f"  {split_name}: {counts}")

# %% [markdown]
# ## 5. Sanity check: no overlap between train and val

# %%
train_ids = set(gt_train["source1_entity_id"])
val_ids   = set(gt_val["source1_entity_id"])
overlap   = train_ids & val_ids

assert len(overlap) == 0, f"OVERLAP DETECTED: {len(overlap)} shared IDs"
assert len(train_ids) + len(val_ids) == len(gt), "Partition size mismatch"

print("Sanity check PASSED:")
print(f"  No overlap between train and val")
print(f"  train + val = {len(train_ids):,} + {len(val_ids):,} = {len(gt):,} total")

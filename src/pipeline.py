#!/usr/bin/env python3
"""
Business Entity Resolution — Final Pipeline
============================================
Standalone script that runs the full pipeline on the test set and writes:
  output/matching_results.tsv  — final matches (scored file)
  output/candidate_pairs.tsv   — blocking candidates (audit file)

Usage (from repo root):
    python code/business_entity_resolution/src/pipeline.py \
        --data-dir dataset \
        --out-dir  code/business_entity_resolution/output

The script is intentionally country-agnostic: no hardcoded ZIP/PIN regex
lengths, no country whitelists, no France-specific branches.

Metric: macro-averaged F_0.5 (precision-weighted 2x over recall).
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)   per entity, then averaged.
    Singletons (truth = empty): score = 1.0 if predicted = empty, else 0.0.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# F_0.5 scorer  (macro-averaged, singletons handled explicitly)
# ---------------------------------------------------------------------------

def f05_per_entity(
    predicted: Set[str],
    truth: Set[str],
) -> float:
    """Compute F_0.5 for a single Source-1 entity.

    Parameters
    ----------
    predicted : set of matched S2/S3 IDs from the pipeline
    truth     : set of ground-truth matched IDs (empty => singleton)

    Returns
    -------
    float in [0, 1]
    """
    if not truth:                        # singleton entity
        return 1.0 if not predicted else 0.0
    if not predicted:                    # we predicted empty for a non-singleton
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall    = tp / len(truth)
    # F_beta with beta=0.5: (1+0.25)*P*R / (0.25*P + R)
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """Compute macro-averaged F_0.5 over all Source-1 entities.

    Parameters
    ----------
    predictions  : {s1_id: set(matched_ids)}  — pipeline output
    ground_truth : {s1_id: set(matched_ids)}  — from ground-truth TSV

    Returns
    -------
    float in [0, 1]
    """
    scores = []
    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        scores.append(f05_per_entity(pred, truth))
    return float(np.mean(scores)) if scores else 0.0


def score_from_tsv(
    matching_path: str | Path,
    ground_truth_path: str | Path,
) -> float:
    """Convenience wrapper: load both TSVs and return macro-averaged F_0.5."""
    preds = _load_id_list_tsv(matching_path, id_col="source1_entity_id", list_col="matched_entity_ids")
    truth = _load_ground_truth_tsv(ground_truth_path)
    return macro_f05(preds, truth)


# ---------------------------------------------------------------------------
# TSV I/O helpers
# ---------------------------------------------------------------------------

def _load_id_list_tsv(
    path: str | Path,
    id_col: str,
    list_col: str,
) -> Dict[str, Set[str]]:
    """Load a two-column TSV where the second column is a comma-separated list.

    Returns {id_col_value: set(list items)}. An empty list cell yields an empty set.
    """
    result: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row[id_col].strip()
            raw = row[list_col].strip()
            result[s1] = set(raw.split(",")) if raw else set()
    return result


def _load_ground_truth_tsv(path: str | Path) -> Dict[str, Set[str]]:
    """Load train_ground_truth.tsv -> {s1_id: set(matched_ids)}.

    Expected columns: source1_entity_id, matched_entity_ids
    The matched_entity_ids cell is a comma-separated list; empty => singleton.
    """
    return _load_id_list_tsv(path, "source1_entity_id", "matched_entity_ids")


def write_matching_results(
    predictions: Dict[str, Set[str]],
    out_path: str | Path,
) -> None:
    """Write matching_results.tsv.

    Format (tab-separated, UTF-8):
        source1_entity_id<TAB>matched_entity_ids
    where matched_entity_ids is a comma-separated list or empty string.

    Guarantees:
    - Exactly one row per S1 entity.
    - No duplicate S1 rows.
    - matched_entity_ids contains only S2-/S3- IDs (caller responsibility).
    - No duplicate IDs within a list.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in sorted(predictions.keys()):
            ids = predictions[s1_id]
            writer.writerow([s1_id, ",".join(sorted(ids)) if ids else ""])
    log.info("Wrote %d rows to %s", len(predictions), out_path)


def write_candidate_pairs(
    candidates: Dict[str, Set[str]],
    out_path: str | Path,
) -> None:
    """Write candidate_pairs.tsv.

    Format (tab-separated, UTF-8):
        source1_entity_id<TAB>candidate_entity_ids
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in sorted(candidates.keys()):
            ids = candidates[s1_id]
            writer.writerow([s1_id, ",".join(sorted(ids)) if ids else ""])
    log.info("Wrote %d rows to %s", len(candidates), out_path)


# ---------------------------------------------------------------------------
# Placeholder pipeline (to be fleshed out in Phase 1-4 notebooks then brought
# back here as the final standalone script)
# ---------------------------------------------------------------------------

def run_pipeline(data_dir: Path, out_dir: Path) -> None:
    """Run the full entity resolution pipeline on the test set.

    Phases (each delegated to a helper that mirrors the corresponding notebook):
      1. Load & normalise  (mirrors 01_normalize.ipynb)
      2. Block             (mirrors 02_blocking.ipynb)
      3. Feature engineer  (mirrors 03_features.ipynb)
      4. Classify          (mirrors 04_model_train.ipynb)
      5. Threshold & write (mirrors 05_threshold_tuning.ipynb + 06_test_inference.ipynb)
    """
    test_dir  = data_dir / "test"
    train_dir = data_dir / "train"

    log.info("=== Business Entity Resolution Pipeline ===")
    log.info("Data dir : %s", data_dir)
    log.info("Output   : %s", out_dir)

    # ------------------------------------------------------------------ #
    # PHASE 0 sanity-check: load test S1 entities and write empty outputs #
    # This stub ensures validate_submission.py passes immediately after   #
    # Phase 0 (all entities present, no matches yet).                     #
    # ------------------------------------------------------------------ #
    log.info("Loading test_source1.tsv ...")
    s1_path = test_dir / "test_source1.tsv"
    if not s1_path.exists():
        log.error("test_source1.tsv not found at %s", s1_path)
        sys.exit(1)

    s1_df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id"], dtype=str)
    s1_ids: List[str] = s1_df["entity_id"].dropna().tolist()
    log.info("  %d test S1 entities loaded", len(s1_ids))

    # Stub: no matches yet — Phase 1-4 will fill these in
    predictions: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}
    candidates:  Dict[str, Set[str]] = {sid: set() for sid in s1_ids}

    # Write outputs
    write_matching_results(predictions, out_dir / "matching_results.tsv")
    write_candidate_pairs(candidates,   out_dir / "candidate_pairs.tsv")

    log.info("Pipeline complete.  Run validate_submission.py to verify outputs.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution — standalone pipeline (Phase 0+ stub)"
    )
    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Root data directory containing train/ and test/ subfolders (default: dataset)",
    )
    parser.add_argument(
        "--out-dir",
        default="code/business_entity_resolution/output",
        help="Output directory for matching_results.tsv and candidate_pairs.tsv",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="After running, score the output against train ground truth (for sanity)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    t0 = time.perf_counter()
    run_pipeline(data_dir, out_dir)
    elapsed = time.perf_counter() - t0
    log.info("Total runtime: %.1f s", elapsed)

    if args.score:
        gt_path = data_dir / "train" / "train_ground_truth.tsv"
        if gt_path.exists():
            f05 = score_from_tsv(out_dir / "matching_results.tsv", gt_path)
            log.info("Macro-averaged F_0.5 on train ground truth: %.4f", f05)
        else:
            log.warning("Ground truth not found at %s — skipping scoring", gt_path)


if __name__ == "__main__":
    main()

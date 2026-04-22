#!/usr/bin/env python3
"""
Locked coverage-aware XGBoost — **test evaluation** + **outcome observation window** scoring only.

Parameters (fixed; no tuning, no changes):
  scale_pos_weight=20, max_depth=5, min_child_weight=1, gamma=1, threshold=0.12

A. Test set: binary metrics + row-level confusion + store-level confusion summary.
B. Outcome window (2025-12-01 ≤ obs_month_start < 2026-02-01): probabilities, threshold flag,
   ranked table, store max-P table only — **no** accuracy/precision/recall/F1/maps vs truth.

Writes (under a timestamped run directory):
  test_set_predictions.csv
  test_set_metrics_summary.csv
  test_set_confusion_matrix_summary.csv
  outcome_window_predictions.csv
  outcome_window_store_level_ranked.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train_lr_rf_baselines import FEATURE_COLS, SPLIT_RANGES, TARGET, temporal_mask  # noqa: E402
from xgb_joint_validation_grid import make_pipeline  # noqa: E402

LOCKED_SCALE_POS_WEIGHT = 20.0
LOCKED_MAX_DEPTH = 5
LOCKED_MIN_CHILD_WEIGHT = 1.0
LOCKED_GAMMA = 1.0
THRESHOLD = 0.12

# Outcome observation window (must match temporal_split_modeling_panel.py)
OUTCOME_LO = "2025-12-01"
OUTCOME_HI = "2026-02-01"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_test_split(path: Path, lo: str, hi: str) -> pd.DataFrame:
    need = FEATURE_COLS + [
        TARGET,
        "obs_month_start",
        "persistent_id_store",
        "obs_month",
        "LOCATION_NAME",
        "close_date",
        "LATITUDE",
        "LONGITUDE",
    ]
    df = pd.read_csv(path.expanduser().resolve(), usecols=need, low_memory=False)
    m = temporal_mask(df["obs_month_start"], lo, hi)
    sub = df.loc[m & df[TARGET].notna()].copy()
    sub[TARGET] = sub[TARGET].astype(int)
    for c in FEATURE_COLS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub


def load_outcome_window_rows(path: Path, lo: str, hi: str) -> pd.DataFrame:
    """Outcome window: do **not** drop rows for missing target (labels not yet valid)."""
    need = FEATURE_COLS + [
        TARGET,
        "obs_month_start",
        "persistent_id_store",
        "obs_month",
        "LOCATION_NAME",
        "close_date",
        "LATITUDE",
        "LONGITUDE",
    ]
    df = pd.read_csv(path.expanduser().resolve(), usecols=need, low_memory=False)
    m = temporal_mask(df["obs_month_start"], lo, hi)
    sub = df.loc[m].copy()
    for c in FEATURE_COLS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub


def store_level_binary_from_panel(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """One row per store: y_store = any positive label; pred_store = any row predicted positive."""
    g = df.groupby("persistent_id_store", sort=False).agg(
        y_store=(TARGET, "max"),
        pred_store=(pred_col, "max"),
    )
    g["y_store"] = g["y_store"].astype(int)
    g["pred_store"] = g["pred_store"].astype(int)
    return g.reset_index()


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=project_root() / "advan-backbone" / "advan_monthly_modeling_panel.csv")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=project_root() / "processed" / "experiments" / "xgb_locked_test_outcome_eval",
    )
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root.expanduser().resolve() / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lo, train_hi = SPLIT_RANGES[0][1], SPLIT_RANGES[0][2]
    test_lo, test_hi = SPLIT_RANGES[2][1], SPLIT_RANGES[2][2]

    panel = args.panel.expanduser().resolve()
    train_df = load_test_split(panel, train_lo, train_hi)
    test_df = load_test_split(panel, test_lo, test_hi)

    pipe = make_pipeline(
        scale_pos_weight=LOCKED_SCALE_POS_WEIGHT,
        max_depth=LOCKED_MAX_DEPTH,
        min_child_weight=LOCKED_MIN_CHILD_WEIGHT,
        gamma=LOCKED_GAMMA,
    )
    pipe.fit(train_df[FEATURE_COLS], train_df[TARGET])

    # ----- A. Test -----
    Xt = test_df[FEATURE_COLS]
    y_test = test_df[TARGET].to_numpy(dtype=int)
    proba_test = pipe.predict_proba(Xt)[:, 1]
    pred_test = (proba_test >= THRESHOLD).astype(int)

    test_out = test_df.copy()
    test_out["predicted_proba_closure"] = proba_test
    test_out["predicted_positive_thr012"] = pred_test

    pred_cols = [
        "persistent_id_store",
        "obs_month_start",
        "obs_month",
        "LOCATION_NAME",
        "LATITUDE",
        "LONGITUDE",
        TARGET,
        "predicted_proba_closure",
        "predicted_positive_thr012",
    ]
    test_out[pred_cols].to_csv(out_dir / "test_set_predictions.csv", index=False)

    cm_row = confusion_counts(y_test, pred_test)
    n_pred_pos = int(pred_test.sum())
    n_actual_pos = int(y_test.sum())
    n_tp_rows = cm_row["TP"]

    metrics = {
        "split": "test",
        "obs_month_start_lo": test_lo,
        "obs_month_start_hi": test_hi,
        "threshold": THRESHOLD,
        "scale_pos_weight": LOCKED_SCALE_POS_WEIGHT,
        "max_depth": LOCKED_MAX_DEPTH,
        "min_child_weight": LOCKED_MIN_CHILD_WEIGHT,
        "gamma": LOCKED_GAMMA,
        "n_observations": len(test_df),
        "n_predicted_positive_rows": n_pred_pos,
        "n_actual_positive_observations": n_actual_pos,
        "n_true_positive_rows_TP": n_tp_rows,
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba_test),
        "pr_auc": average_precision_score(y_test, proba_test),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "test_set_metrics_summary.csv", index=False)

    cm_df = pd.DataFrame(
        [
            {"level": "observation", **cm_row},
        ]
    )
    test_for_store = test_out.copy()
    store_tbl = store_level_binary_from_panel(test_for_store, "predicted_positive_thr012")
    y_s = store_tbl["y_store"].to_numpy(dtype=int)
    p_s = store_tbl["pred_store"].to_numpy(dtype=int)
    cm_store = confusion_counts(y_s, p_s)
    cm_df = pd.concat(
        [cm_df, pd.DataFrame([{"level": "distinct_store", **cm_store}])],
        ignore_index=True,
    )
    cm_df.to_csv(out_dir / "test_set_confusion_matrix_summary.csv", index=False)

    # ----- B. Outcome window (no truth metrics) -----
    ow = load_outcome_window_rows(panel, OUTCOME_LO, OUTCOME_HI)
    Xo = ow[FEATURE_COLS]
    proba_o = pipe.predict_proba(Xo)[:, 1]
    pred_o = (proba_o >= THRESHOLD).astype(int)

    ow_out = ow.copy()
    ow_out["predicted_proba_closure"] = proba_o
    ow_out["predicted_positive_thr012"] = pred_o

    order = np.argsort(-proba_o)
    ranks = np.empty(len(ow_out), dtype=int)
    ranks[order] = np.arange(1, len(ow_out) + 1)
    ow_out["risk_rank_desc"] = ranks
    ow_sorted = ow_out.sort_values("predicted_proba_closure", ascending=False).reset_index(drop=True)

    ow_export_cols = [
        "risk_rank_desc",
        "persistent_id_store",
        TARGET,
        "obs_month_start",
        "obs_month",
        "LOCATION_NAME",
        "LATITUDE",
        "LONGITUDE",
        "predicted_proba_closure",
        "predicted_positive_thr012",
    ]
    ow_sorted[[c for c in ow_export_cols if c in ow_sorted.columns]].to_csv(
        out_dir / "outcome_window_predictions.csv",
        index=False,
    )

    idx = ow_out.groupby("persistent_id_store")["predicted_proba_closure"].idxmax()
    store_rep = ow_out.loc[idx].copy()
    store_rep["store_max_predicted_proba"] = store_rep["predicted_proba_closure"].astype(float)
    store_rep["store_predicted_positive_thr012"] = (store_rep["store_max_predicted_proba"] >= THRESHOLD).astype(int)
    store_rep = store_rep.sort_values("store_max_predicted_proba", ascending=False).reset_index(drop=True)
    store_rep.insert(0, "store_risk_rank_desc", range(1, len(store_rep) + 1))
    store_export_cols = [
        "store_risk_rank_desc",
        "persistent_id_store",
        "store_max_predicted_proba",
        "store_predicted_positive_thr012",
        "obs_month_start",
        "obs_month",
        "LOCATION_NAME",
        "LATITUDE",
        "LONGITUDE",
    ]
    store_rep[store_export_cols].to_csv(out_dir / "outcome_window_store_level_ranked.csv", index=False)

    print(f"Wrote: {out_dir}")
    print(
        f"  test: n_pred+={n_pred_pos}, n_actual+={n_actual_pos}, TP={n_tp_rows}, "
        f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
    )
    print(f"  outcome rows: {len(ow_out)} (no label-based metrics)")


if __name__ == "__main__":
    main()

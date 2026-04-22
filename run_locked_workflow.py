#!/usr/bin/env python3
"""
Self-contained reproducible workflow for the locked XGBoost model.

What this script does:
1) Load the locked model from ./model/xgb_locked_pipeline.joblib
2) Score validation/test/outcome-window CSVs in ./data
3) Compute binary metrics for validation/test at threshold=0.12
4) Compute confusion summaries (observation + distinct-store) for validation/test
5) Export prospective-only ranking outputs for outcome window (no truth metrics)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

TARGET = "closure_within_next_2_months"
THRESHOLD = 0.12

FEATURE_COLS = [
    "average_foot_traffic",
    "foot_traffic_trend",
    "ct_storefront_median_rent_psf",
    "distance_nearest_subway_station_m",
    "distance_nearest_bus_stop_m",
    "corner_adjacent",
    "semantic_competitor_count",
    "total_restaurants_500m",
    "recent_nearby_closures_1000m_3mo",
    "Brand scale proxy",
    "inspection_violation_count",
    "inspection_matched",
]


def read_scoring_df(path: Path, *, keep_na_target: bool) -> pd.DataFrame:
    need = FEATURE_COLS + [
        TARGET,
        "persistent_id_store",
        "obs_month_start",
        "obs_month",
        "LOCATION_NAME",
        "LATITUDE",
        "LONGITUDE",
    ]
    df = pd.read_csv(path, usecols=need, low_memory=False)
    if not keep_na_target:
        df = df.loc[df[TARGET].notna()].copy()
        df[TARGET] = df[TARGET].astype(int)
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    return {
        "TP": int(((y == 1) & (pred == 1)).sum()),
        "FP": int(((y == 0) & (pred == 1)).sum()),
        "TN": int(((y == 0) & (pred == 0)).sum()),
        "FN": int(((y == 1) & (pred == 0)).sum()),
    }


def score_labeled_split(df: pd.DataFrame, pipe, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].to_numpy(dtype=int)
    p = pipe.predict_proba(df[FEATURE_COLS])[:, 1]
    pred = (p >= THRESHOLD).astype(int)

    out = df.copy()
    out["predicted_proba_closure"] = p
    out["predicted_positive_thr012"] = pred

    cm_obs = confusion_counts(y, pred)
    metrics = {
        "split": split_name,
        "threshold": THRESHOLD,
        "n_observations": len(df),
        "n_predicted_positive_rows": int(pred.sum()),
        "n_actual_positive_observations": int(y.sum()),
        "n_true_positive_rows_TP": cm_obs["TP"],
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
    }

    store_tbl = out.groupby("persistent_id_store", sort=False).agg(
        y_store=(TARGET, "max"),
        pred_store=("predicted_positive_thr012", "max"),
    )
    cm_store = confusion_counts(store_tbl["y_store"].to_numpy(dtype=int), store_tbl["pred_store"].to_numpy(dtype=int))
    cm_df = pd.DataFrame(
        [
            {"split": split_name, "level": "observation", **cm_obs},
            {"split": split_name, "level": "distinct_store", **cm_store},
        ]
    )
    return out, pd.DataFrame([metrics]), cm_df


def score_outcome_window(df: pd.DataFrame, pipe) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pipe.predict_proba(df[FEATURE_COLS])[:, 1]
    pred = (p >= THRESHOLD).astype(int)
    out = df.copy()
    out["predicted_proba_closure"] = p
    out["predicted_positive_thr012"] = pred

    order = np.argsort(-p)
    ranks = np.empty(len(out), dtype=int)
    ranks[order] = np.arange(1, len(out) + 1)
    out["risk_rank_desc"] = ranks
    out = out.sort_values("predicted_proba_closure", ascending=False).reset_index(drop=True)

    idx = out.groupby("persistent_id_store")["predicted_proba_closure"].idxmax()
    store = out.loc[idx].copy()
    store["store_max_predicted_proba"] = store["predicted_proba_closure"]
    store["store_predicted_positive_thr012"] = (store["store_max_predicted_proba"] >= THRESHOLD).astype(int)
    store = store.sort_values("store_max_predicted_proba", ascending=False).reset_index(drop=True)
    store.insert(0, "store_risk_rank_desc", np.arange(1, len(store) + 1))
    return out, store


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=here / "data")
    ap.add_argument("--model-path", type=Path, default=here / "model" / "xgb_locked_pipeline.joblib")
    ap.add_argument("--out-dir", type=Path, default=here / "outputs")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    model_path = args.model_path.resolve()
    out_root = args.out_dir.resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(model_path)

    val_df = read_scoring_df(data_dir / "modeling_panel_validation.csv", keep_na_target=False)
    test_df = read_scoring_df(data_dir / "modeling_panel_test.csv", keep_na_target=False)
    outcome_df = read_scoring_df(data_dir / "modeling_panel_outcome_observation_window.csv", keep_na_target=True)

    val_pred, val_metrics, val_cm = score_labeled_split(val_df, pipe, "validation")
    test_pred, test_metrics, test_cm = score_labeled_split(test_df, pipe, "test")
    outcome_pred, outcome_store = score_outcome_window(outcome_df, pipe)

    keep_cols = [
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
    val_pred[keep_cols].to_csv(out_dir / "validation_set_predictions.csv", index=False)
    test_pred[keep_cols].to_csv(out_dir / "test_set_predictions.csv", index=False)
    pd.concat([val_metrics, test_metrics], ignore_index=True).to_csv(out_dir / "validation_test_metrics_summary.csv", index=False)
    pd.concat([val_cm, test_cm], ignore_index=True).to_csv(out_dir / "validation_test_confusion_summary.csv", index=False)

    outcome_cols = [
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
    outcome_pred[[c for c in outcome_cols if c in outcome_pred.columns]].to_csv(
        out_dir / "outcome_window_predictions.csv", index=False
    )

    outcome_store_cols = [
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
    outcome_store[outcome_store_cols].to_csv(out_dir / "outcome_window_store_level_ranked.csv", index=False)

    # Also save the exact test-only files with names used in the project workflow.
    test_metrics.to_csv(out_dir / "test_set_metrics_summary.csv", index=False)
    test_cm.loc[test_cm["split"] == "test", ["level", "TP", "FP", "TN", "FN"]].to_csv(
        out_dir / "test_set_confusion_matrix_summary.csv",
        index=False,
    )

    print(f"Done. Outputs written to: {out_dir}")
    print(f"  test precision={float(test_metrics['precision'].iloc[0]):.4f}, recall={float(test_metrics['recall'].iloc[0]):.4f}")
    print("  outcome window is prospective scoring only (no truth-based metrics).")


if __name__ == "__main__":
    main()

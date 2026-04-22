#!/usr/bin/env python3
"""
Test set only: **top-100 store-level risk ranking** for the locked coverage-aware XGBoost.

Per store: risk score = **max predicted P(closure)** across that store's test rows.
Rank stores by this score (descending), take top 100.

Outputs: CSV (full + map-ready), metrics summary, Folium HTML (color = actual closure in test window).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_closure_maps import build_store_topn_closure_status_map  # noqa: E402
from train_lr_rf_baselines import FEATURE_COLS, SPLIT_RANGES, TARGET  # noqa: E402
from xgb_joint_validation_grid import make_pipeline  # noqa: E402
from xgb_locked_topk_evaluation import load_split_df, project_root  # noqa: E402

LOCKED_SCALE_POS_WEIGHT = 20.0
LOCKED_MAX_DEPTH = 5
LOCKED_MIN_CHILD_WEIGHT = 1.0
LOCKED_GAMMA = 1.0

TOP_N = 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=project_root() / "advan-backbone" / "advan_monthly_modeling_panel.csv")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=project_root() / "processed" / "experiments" / "xgb_store_top100_test",
        help="Creates a timestamped subdirectory",
    )
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root.expanduser().resolve() / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    train_lo, train_hi = SPLIT_RANGES[0][1], SPLIT_RANGES[0][2]
    test_lo, test_hi = SPLIT_RANGES[2][1], SPLIT_RANGES[2][2]

    panel = args.panel.expanduser().resolve()
    train_df = load_split_df(panel, train_lo, train_hi)
    test_df = load_split_df(panel, test_lo, test_hi)

    pipe = make_pipeline(
        scale_pos_weight=LOCKED_SCALE_POS_WEIGHT,
        max_depth=LOCKED_MAX_DEPTH,
        min_child_weight=LOCKED_MIN_CHILD_WEIGHT,
        gamma=LOCKED_GAMMA,
    )
    pipe.fit(train_df[FEATURE_COLS], train_df[TARGET])

    Xt = test_df[FEATURE_COLS]
    test_df = test_df.copy()
    test_df["predicted_proba_closure"] = pipe.predict_proba(Xt)[:, 1]

    idx = test_df.groupby("persistent_id_store")["predicted_proba_closure"].idxmax()
    rep = test_df.loc[idx].copy()

    store_any_positive = (
        test_df.groupby("persistent_id_store")[TARGET].max().astype(int).rename("store_true_positive_in_test").reset_index()
    )
    rep = rep.merge(store_any_positive, on="persistent_id_store", how="left")

    rep["store_risk_score"] = rep["predicted_proba_closure"].astype(float)
    rep = rep.sort_values("store_risk_score", ascending=False).reset_index(drop=True)

    k = min(TOP_N, len(rep))
    top = rep.iloc[:k].copy()
    top.insert(0, "store_risk_rank", range(1, len(top) + 1))

    n_tp = int(top["store_true_positive_in_test"].sum())
    precision_100 = n_tp / len(top) if len(top) else float("nan")

    out_cols = [
        "store_risk_rank",
        "persistent_id_store",
        "store_risk_score",
        "store_true_positive_in_test",
        "LOCATION_NAME",
        "LATITUDE",
        "LONGITUDE",
        "obs_month_start",
        "obs_month",
    ]
    csv_path = run_dir / "test_top100_stores_ranked.csv"
    top[out_cols].to_csv(csv_path, index=False)

    map_ready = top[
        [
            "store_risk_rank",
            "persistent_id_store",
            "store_risk_score",
            "store_true_positive_in_test",
            "LATITUDE",
            "LONGITUDE",
            "LOCATION_NAME",
            "obs_month",
        ]
    ].copy()
    map_ready.to_csv(run_dir / "test_top100_stores_map_ready.csv", index=False)

    cfg = {
        "model": "xgboost_hist_locked_coverage_aware",
        "scale_pos_weight": LOCKED_SCALE_POS_WEIGHT,
        "max_depth": LOCKED_MAX_DEPTH,
        "min_child_weight": LOCKED_MIN_CHILD_WEIGHT,
        "gamma": LOCKED_GAMMA,
        "test_window": [test_lo, test_hi],
        "top_n": TOP_N,
        "n_selected": int(len(top)),
        "n_true_positive_stores_in_top_n": n_tp,
        "store_precision_at_n": precision_100,
    }
    (run_dir / "metrics.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    joblib.dump(pipe, run_dir / "xgb_locked_pipeline.joblib")

    overlay = (
        "Locked XGBoost — store-level top-100 risk (test)\n"
        f"scale_pos_weight={LOCKED_SCALE_POS_WEIGHT:g}, max_depth={LOCKED_MAX_DEPTH}, "
        f"min_child_weight={LOCKED_MIN_CHILD_WEIGHT:g}, gamma={LOCKED_GAMMA:g}\n"
        f"Risk score = max P(closure) across store rows · test: {test_lo} ≤ obs_month_start < {test_hi}\n"
        f"Selected stores: {len(top)} · True-positive stores (closure in window): {n_tp} · precision@{len(top)} = {precision_100:.6f}"
    )

    maps_dir = run_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    plot_df = top.copy()
    build_store_topn_closure_status_map(
        "(Test) Top store-level risk scores — closure status",
        plot_df,
        prob_col="store_risk_score",
        closure_in_test_col="store_true_positive_in_test",
        rank_col="store_risk_rank",
        out_path=maps_dir / "map_test_top100_stores_closure_status.html",
        split_label=f"test window: obs_month_start in [{test_lo}, {test_hi})",
        parameter_overlay=overlay,
    )

    summary = f"""# Store-level top-{len(top)} risk ranking (test set)

## Metrics

- **Selected stores:** {len(top)}
- **True-positive stores** (≥ one `closure_within_next_2_months` = 1 in test): **{n_tp}**
- **Store-level precision@{len(top)}:** **{precision_100:.6f}**

## Method

1. Score each store by **maximum** predicted closure probability across that store's rows in the test split.
2. Sort descending; keep top **{TOP_N}** (or fewer if universe smaller).

## Outputs (`{run_dir.relative_to(project_root())}`)

| File | Description |
| --- | --- |
| `test_top100_stores_ranked.csv` | Ranked table with coords and labels |
| `test_top100_stores_map_ready.csv` | Slim columns for GIS / overlay |
| `maps/map_test_top100_stores_closure_status.html` | Folium map · red = closure observed in test · slate = not |
| `metrics.json` | Numeric summary |
| `xgb_locked_pipeline.joblib` | Fitted pipeline (train → test scoring) |

"""
    (run_dir / "store_top100_summary.md").write_text(summary, encoding="utf-8")

    print(f"Wrote {run_dir}")
    print(f"  precision@{len(top)} = {precision_100:.6f} (TP stores {n_tp}/{len(top)})")


if __name__ == "__main__":
    main()

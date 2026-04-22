#!/usr/bin/env python3
"""
Locked coverage-aware XGBoost: **single fit** on the training window, then **risk ranking**
metrics on validation (comparison) and **test** (primary).

Locked hyperparameters (fixed; no alternative tuning in this script):
  scale_pos_weight=20, max_depth=5, min_child_weight=1, gamma=1
  (reference threshold from selection = 0.12 — reported for context only; ranking uses probabilities.)

Outputs (timestamped run dir):
  - Full observation-level probability tables (ranked)
  - Store-level ranked table (max P per store)
  - Observation- and store-level top-k metrics (1%, 5%, 10%)
  - Top-k slice CSVs for mapping / external viz
  - Folium **probability** maps (color depth = P(closure); store-deduped for readability)

Does **not** refit alternative parameter combinations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_closure_maps import build_probability_poi_map  # noqa: E402
from train_lr_rf_baselines import (  # noqa: E402
    FEATURE_COLS,
    SPLIT_RANGES,
    TARGET,
    df_md,
    temporal_mask,
)
from xgb_joint_validation_grid import make_pipeline  # noqa: E402

# Locked configuration (coverage-aware winner)
LOCKED_SCALE_POS_WEIGHT = 20.0
LOCKED_MAX_DEPTH = 5
LOCKED_MIN_CHILD_WEIGHT = 1.0
LOCKED_GAMMA = 1.0
REFERENCE_THRESHOLD = 0.12  # selection rule only; top-k uses continuous scores

TOP_FRACS = (0.01, 0.05, 0.10)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split_df(path: Path, lo: str, hi: str) -> pd.DataFrame:
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


def observation_topk_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    fracs: tuple[float, ...],
    *,
    split_label: str,
) -> pd.DataFrame:
    """Row-level top-k: sort all observations by predicted probability (desc)."""
    n = len(y_true)
    pos_total = int(np.asarray(y_true).sum())
    order = np.argsort(-proba)
    rows = []
    for frac in fracs:
        k = max(1, int(np.ceil(n * frac)))
        idx = order[:k]
        tp = int(np.asarray(y_true)[idx].sum())
        rows.append(
            {
                "split": split_label,
                "top_frac": frac,
                "top_frac_label": f"{frac:.0%}",
                "n_observations_total": n,
                "n_positive_observations_total": pos_total,
                "n_in_top_k": k,
                "n_true_positives_in_top_k": tp,
                "precision_at_k": tp / k if k else np.nan,
                "recall_at_k": tp / pos_total if pos_total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def store_topk_metrics(
    tbl: pd.DataFrame,
    *,
    store_col: str,
    y_col: str,
    proba_col: str,
    fracs: tuple[float, ...],
    split_label: str,
) -> pd.DataFrame:
    """
    Store-level top-k: each store scored by **max** predicted probability across its rows in the split.
    Positive store = at least one observation with y=1 in that split.
    """
    pos_stores = tbl.loc[tbl[y_col] == 1, store_col].unique()
    n_pos_stores = len(pos_stores)

    agg = tbl.groupby(store_col, sort=False).agg(
        score=(proba_col, "max"),
        positive_store=(y_col, "max"),
    ).reset_index()
    n_stores = len(agg)
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)

    rows = []
    for frac in fracs:
        k = max(1, int(np.ceil(n_stores * frac)))
        top = agg.iloc[:k]
        tp_stores = int((top["positive_store"] == 1).sum())
        rows.append(
            {
                "split": split_label,
                "top_frac": frac,
                "top_frac_label": f"{frac:.0%}",
                "n_distinct_stores_total": n_stores,
                "n_distinct_positive_stores_total": n_pos_stores,
                "n_stores_in_top_k": k,
                "n_distinct_true_positive_stores_in_top_k": tp_stores,
                "store_precision_at_k": tp_stores / k if k else np.nan,
                "store_recall_at_k": tp_stores / n_pos_stores if n_pos_stores else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_observation_ranks(tbl: pd.DataFrame, proba_col: str) -> pd.DataFrame:
    out = tbl.copy()
    order = np.argsort(-out[proba_col].to_numpy())
    ranks = np.empty(len(out), dtype=int)
    ranks[order] = np.arange(1, len(out) + 1)
    out["probability_rank_desc"] = ranks
    return out.sort_values(proba_col, ascending=False).reset_index(drop=True)


def store_representation_table(tbl: pd.DataFrame, proba_col: str) -> pd.DataFrame:
    """One row per store: row attaining max P; columns for mapping."""
    idx = tbl.groupby("persistent_id_store")[proba_col].idxmax()
    one = tbl.loc[idx].copy()
    one = one.sort_values(proba_col, ascending=False).reset_index(drop=True)
    one.insert(0, "store_rank_by_max_proba", np.arange(1, len(one) + 1))
    any_pos = tbl.groupby("persistent_id_store")[TARGET].max().reset_index()
    any_pos = any_pos.rename(columns={TARGET: "store_any_positive_observation"})
    one = one.merge(any_pos, on="persistent_id_store", how="left")
    one = one.rename(columns={proba_col: "predicted_proba_store_max"})
    return one


def slice_observations_top_frac(tbl: pd.DataFrame, proba_col: str, frac: float) -> pd.DataFrame:
    n = len(tbl)
    k = max(1, int(np.ceil(n * frac)))
    return tbl.sort_values(proba_col, ascending=False).head(k).copy()


def overlay_text() -> str:
    return (
        "Locked XGBoost (coverage-aware)\n"
        f"scale_pos_weight={LOCKED_SCALE_POS_WEIGHT:g}, max_depth={LOCKED_MAX_DEPTH}, "
        f"min_child_weight={LOCKED_MIN_CHILD_WEIGHT:g}, gamma={LOCKED_GAMMA:g}\n"
        f"reference threshold (selection context): P ≥ {REFERENCE_THRESHOLD:.4f}\n"
        "Ranking: observation- and store-level top-k use predicted probabilities only.\n"
        f"train: {SPLIT_RANGES[0][1]} ≤ obs_month_start < {SPLIT_RANGES[0][2]}\n"
        f"val:   {SPLIT_RANGES[1][1]} ≤ obs_month_start < {SPLIT_RANGES[1][2]}\n"
        f"test:  {SPLIT_RANGES[2][1]} ≤ obs_month_start < {SPLIT_RANGES[2][2]}\n"
        "Maps: one marker per store at coordinates of the obs row with max P(store)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=project_root() / "advan-backbone" / "advan_monthly_modeling_panel.csv")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=project_root() / "processed" / "experiments" / "xgb_locked_topk_eval",
        help="Creates a timestamped subdirectory here",
    )
    ap.add_argument("--no-maps", action="store_true", help="Skip Folium HTML (still writes CSVs)")
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root.expanduser().resolve() / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = run_dir / "maps"
    slices_dir = run_dir / "topk_slices"

    panel = args.panel.expanduser().resolve()

    train_lo, train_hi = SPLIT_RANGES[0][1], SPLIT_RANGES[0][2]
    val_lo, val_hi = SPLIT_RANGES[1][1], SPLIT_RANGES[1][2]
    test_lo, test_hi = SPLIT_RANGES[2][1], SPLIT_RANGES[2][2]

    train_df = load_split_df(panel, train_lo, train_hi)
    val_df = load_split_df(panel, val_lo, val_hi)
    test_df = load_split_df(panel, test_lo, test_hi)

    pipe = make_pipeline(
        scale_pos_weight=LOCKED_SCALE_POS_WEIGHT,
        max_depth=LOCKED_MAX_DEPTH,
        min_child_weight=LOCKED_MIN_CHILD_WEIGHT,
        gamma=LOCKED_GAMMA,
    )
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET]
    pipe.fit(X_tr, y_tr)

    cfg = {
        "model": "xgboost_hist",
        "selection": "locked_coverage_aware_config",
        "scale_pos_weight": LOCKED_SCALE_POS_WEIGHT,
        "max_depth": LOCKED_MAX_DEPTH,
        "min_child_weight": LOCKED_MIN_CHILD_WEIGHT,
        "gamma": LOCKED_GAMMA,
        "reference_threshold_validation_selection": REFERENCE_THRESHOLD,
        "train_window": [train_lo, train_hi],
        "validation_window": [val_lo, val_hi],
        "test_window": [test_lo, test_hi],
        "top_fracs": list(TOP_FRACS),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    joblib.dump(pipe, run_dir / "xgb_locked_pipeline.joblib")

    def score_split(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        X = df[FEATURE_COLS]
        p = pipe.predict_proba(X)[:, 1]
        out = df.copy()
        out["predicted_proba_closure"] = p
        out = add_observation_ranks(out, "predicted_proba_closure")
        y = out[TARGET].to_numpy(dtype=int)
        return out, y, p

    val_scored, y_val, p_val = score_split(val_df, "validation")
    test_scored, y_test, p_test = score_split(test_df, "test")

    obs_val = observation_topk_metrics(y_val, p_val, TOP_FRACS, split_label="validation")
    obs_test = observation_topk_metrics(y_test, p_test, TOP_FRACS, split_label="test")
    obs_combo = pd.concat([obs_test, obs_val], ignore_index=True)
    obs_combo.to_csv(run_dir / "topk_observation_level_metrics.csv", index=False)

    store_val = store_topk_metrics(
        val_scored,
        store_col="persistent_id_store",
        y_col=TARGET,
        proba_col="predicted_proba_closure",
        fracs=TOP_FRACS,
        split_label="validation",
    )
    store_test = store_topk_metrics(
        test_scored,
        store_col="persistent_id_store",
        y_col=TARGET,
        proba_col="predicted_proba_closure",
        fracs=TOP_FRACS,
        split_label="test",
    )
    store_combo = pd.concat([store_test, store_val], ignore_index=True)
    store_combo.to_csv(run_dir / "topk_store_level_metrics.csv", index=False)

    # Full ranked exports
    obs_cols = [
        "probability_rank_desc",
        "predicted_proba_closure",
        TARGET,
        "persistent_id_store",
        "obs_month_start",
        "obs_month",
        "LOCATION_NAME",
        "LATITUDE",
        "LONGITUDE",
    ]
    test_scored[obs_cols].to_csv(run_dir / "test_observations_ranked_by_probability.csv", index=False)
    val_scored[obs_cols].to_csv(run_dir / "validation_observations_ranked_by_probability.csv", index=False)

    test_stores = store_representation_table(test_scored, "predicted_proba_closure")
    val_stores = store_representation_table(val_scored, "predicted_proba_closure")
    test_stores.to_csv(run_dir / "test_stores_ranked_by_max_probability.csv", index=False)
    val_stores.to_csv(run_dir / "validation_stores_ranked_by_max_probability.csv", index=False)

    slices_dir.mkdir(parents=True, exist_ok=True)
    for frac in TOP_FRACS:
        lab = f"{int(frac * 100):02d}pct"
        slice_observations_top_frac(test_scored, "predicted_proba_closure", frac)[obs_cols].to_csv(
            slices_dir / f"test_top_{lab}_observations.csv",
            index=False,
        )
        slice_observations_top_frac(val_scored, "predicted_proba_closure", frac)[obs_cols].to_csv(
            slices_dir / f"validation_top_{lab}_observations.csv",
            index=False,
        )

    # Map-ready compact exports: store table with geo + prob (test primary)
    map_cols = [
        "store_rank_by_max_proba",
        "predicted_proba_store_max",
        "store_any_positive_observation",
        "persistent_id_store",
        "LATITUDE",
        "LONGITUDE",
        "LOCATION_NAME",
        "obs_month",
    ]
    test_stores[map_cols].to_csv(run_dir / "map_ready_test_stores.csv", index=False)
    val_stores[map_cols].to_csv(run_dir / "map_ready_validation_stores.csv", index=False)

    ov = overlay_text()
    if not args.no_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)
        plot_df = test_stores.rename(columns={"predicted_proba_store_max": "_p_map"}).copy()
        build_probability_poi_map(
            "(Test) XGBoost — closure probability (store max P)",
            plot_df,
            prob_col="_p_map",
            out_path=maps_dir / "map_test_probability_stores.html",
            split_label=f"test: {test_lo} ≤ obs_month_start < {test_hi}",
            parameter_overlay=ov,
        )
        plot_v = val_stores.rename(columns={"predicted_proba_store_max": "_p_map"}).copy()
        build_probability_poi_map(
            "(Validation) XGBoost — closure probability (store max P)",
            plot_v,
            prob_col="_p_map",
            out_path=maps_dir / "map_validation_probability_stores.html",
            split_label=f"validation: {val_lo} ≤ obs_month_start < {val_hi}",
            parameter_overlay=ov,
        )

    # Summary markdown
    lines = [
        "# Locked XGBoost — top-k risk ranking evaluation",
        "",
        "## Configuration (fixed)",
        "",
        json.dumps(cfg, indent=2),
        "",
        "## Observation-level top-k (primary: **test**)",
        "",
        "### Test",
        "",
        df_md(obs_test.round(6)),
        "",
        "### Validation (comparison)",
        "",
        df_md(obs_val.round(6)),
        "",
        "## Store-level top-k (store score = max P across rows in split)",
        "",
        "### Test",
        "",
        df_md(store_test.round(6)),
        "",
        "### Validation (comparison)",
        "",
        df_md(store_val.round(6)),
        "",
        "## Outputs",
        "",
        f"- Run directory: `{run_dir.relative_to(project_root())}`",
        "- `test_observations_ranked_by_probability.csv` — full test set, sorted by P (desc)",
        "- `validation_observations_ranked_by_probability.csv`",
        "- `test_stores_ranked_by_max_probability.csv` — one row per store (max-P row)",
        "- `topk_observation_level_metrics.csv`, `topk_store_level_metrics.csv`",
        f"- `topk_slices/` — observation subsets for top 1%, 5%, 10%",
        "- `map_ready_test_stores.csv`, `map_ready_validation_stores.csv`",
        "- `maps/map_test_probability_stores.html` — probability-colored markers (inferno scale)",
        "- `xgb_locked_pipeline.joblib` — fitted pipeline",
        "",
    ]
    (run_dir / "topk_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {run_dir}")


if __name__ == "__main__":
    main()

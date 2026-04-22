# Try It Out: Locked XGBoost Repro Bundle

This folder is a minimal reproducible package so others can run the final locked-model workflow without navigating the full project structure.

## 1) Contents

- `model/`
  - `xgb_locked_pipeline.joblib`: final locked model (already trained)
  - `locked_model_config.json`: locked parameter configuration
- `data/`
  - `modeling_panel_training.csv`
  - `modeling_panel_validation.csv`
  - `modeling_panel_test.csv`
  - `modeling_panel_outcome_observation_window.csv`
  - `split_summary.csv`
- `scripts/`
  - `run_locked_workflow.py`: **recommended entry point** (standalone and minimal)
  - `xgb_locked_test_outcome_scoring.py`: original project script (kept for reference)
  - `xgb_locked_topk_evaluation.py`: original project script (kept for reference)
  - `xgb_store_top100_test.py`: original project script (kept for reference)
- `requirements.txt`

---

## 2) Locked Model Parameters (fixed)

- Model: XGBoost
- `scale_pos_weight = 20`
- `max_depth = 5`
- `min_child_weight = 1`
- `gamma = 1`
- `threshold = 0.12`

---

## 3) Quick Start

Open terminal and go to this folder:

```bash
cd "/Volumes/T7/URP530/Final Project/try it out"
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the workflow:

```bash
python3 scripts/run_locked_workflow.py
```

---

## 4) Output Files

The script writes results to `outputs/<timestamp>/`.

Key files:

- `validation_set_predictions.csv`
- `test_set_predictions.csv`
- `validation_test_metrics_summary.csv`
- `validation_test_confusion_summary.csv`
- `test_set_metrics_summary.csv`
- `test_set_confusion_matrix_summary.csv`
- `outcome_window_predictions.csv`
- `outcome_window_store_level_ranked.csv`

Notes:

- `test_*` files are final evaluated outputs (ground truth available).
- `outcome_window_*` files are prospective risk scoring only (no future truth labels available yet).

---

## 5) Workflow in One Line

Load locked model -> score validation/test/outcome datasets -> report standard classification metrics on test -> export prospective risk rankings for the outcome window.

---

## 6) FAQ

- **Q: Why are there no precision/recall metrics for the outcome window?**  
  A: Future two-month truth labels are not available for that window, so only prospective scoring is valid.

- **Q: I only need test + outcome outputs. What should I run?**  
  A: Run `python3 scripts/run_locked_workflow.py`; it already generates those files.


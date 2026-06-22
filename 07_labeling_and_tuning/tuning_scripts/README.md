# STEPScore Threshold Tuning Pack

This folder gives you everything needed to tune STEPScore thresholds from real run data.

## Files
- `build_pairs_from_runs.py`
  - Extracts one row per run from `.cad42_runs/*/result.json`.
  - Produces a labeling CSV + `metrics_meta.json` (metric directions + default thresholds).
- `auto_tune_thresholds.py`
  - Tunes thresholds automatically from labeled rows.
  - Outputs `threshold_overrides.json` usable in STEPScore.
- `pairs_labeled_template.csv`
  - Empty template with all 44 metric columns.
- `labeling_guide.md`
  - Human labeling rules (`positive` / `negative` / `review`).
- `run_commands.md`
  - Copy-paste commands for the full workflow.

## Workflow (Simple)
1. Build the labeling dataset from existing runs.
2. Label only the `label` column (`positive`, `negative`, or `review`).
3. Run automatic tuner.
4. Apply generated `threshold_overrides.json` in STEPScore.

## Expected Inputs
- STEPScore run outputs in:
  - `.cad42_runs/<run_id>/result.json`
  - `.cad42_runs/<run_id>/inputs.json` (optional but helpful metadata)

## Expected Outputs
- `pairs_for_labeling.csv`
- `metrics_meta.json`
- `output/metric_tuning_report.csv`
- `output/tuning_summary.md`
- `output/thresholds_recommended.json`
- `output/threshold_overrides.json`

## Notes
- The tuner is per-metric and data-driven.
- It uses your labeled outcomes to choose thresholds that best match human judgments.
- It supports cost-sensitive tuning (false pass penalty vs false fail penalty).


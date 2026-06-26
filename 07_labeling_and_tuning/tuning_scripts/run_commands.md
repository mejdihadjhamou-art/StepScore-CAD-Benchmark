# Threshold Tuning Commands

Run from:

```bash
cd "./cad42_platform"
conda activate stepscore
```

## 1) Build labeling CSV from existing runs

```bash
python threshold_tuning/build_pairs_from_runs.py \
  --runs-root .cad42_runs \
  --output-csv threshold_tuning/pairs_for_labeling.csv \
  --output-metrics-meta threshold_tuning/metrics_meta.json \
  --default-label review
```

## 2) Label rows manually

Edit:
- `threshold_tuning/pairs_for_labeling.csv`

Set `label` as:
- `positive`
- `negative`
- `review`

## 3) Automatic threshold tuning

```bash
python threshold_tuning/auto_tune_thresholds.py \
  --pairs-csv threshold_tuning/pairs_for_labeling.csv \
  --metrics-meta-json threshold_tuning/metrics_meta.json \
  --label-column label \
  --positive-labels positive,pass \
  --negative-labels negative,fail,rework \
  --objective balanced \
  --fp-cost 5 \
  --fn-cost 1 \
  --nan-policy fail \
  --output-dir threshold_tuning/output
```

## 4) Apply tuned thresholds in STEPScore UI

Open:
- `threshold_tuning/output/threshold_overrides.json`

Copy JSON into STEPScore:
- `Threshold overrides JSON (optional)`

## 5) Optional: tune only specific metrics

```bash
python threshold_tuning/auto_tune_thresholds.py \
  --pairs-csv threshold_tuning/pairs_for_labeling.csv \
  --metrics-meta-json threshold_tuning/metrics_meta.json \
  --tune-metrics chamfer_distance_mm,hausdorff_95p_mm,emd_distance \
  --objective f1 \
  --output-dir threshold_tuning/output_subset
```


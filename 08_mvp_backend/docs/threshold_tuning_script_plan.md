# Threshold Tuning Script Plan

## Goal
Implement a script that recommends metric thresholds from labeled calibration pairs.

## Script name
`tools/tune_thresholds.py`

## Inputs
- `--pairs-csv` path to labeled pairs CSV
- `--objective` one of: `precision`, `recall`, `f1`, `balanced`
- `--tier-column` optional column name for task tier (`easy|medium|hard`)
- `--min-fitness` optional alignment fitness floor

## Expected CSV columns
- `pair_id`
- `label` (`positive` or `negative`)
- `tier` (optional)
- `chamfer_mm`
- `haus95_mm`
- `haus99_mm`
- `volume_diff_pct`
- `bbox_max_diff_mm`
- `watertight_pass` (0/1)
- `single_component_pass` (0/1)
- `alignment_failed` (0/1)
- `icp_fitness`

## Outputs
1. `thresholds_recommended.json`
2. `threshold_report.md`
3. optional plots in `artifacts/`:
- per-metric histograms by class
- ROC/PR-like sweeps for scalar thresholds
- confusion matrix under chosen policy

## Tuning strategy

### Step 1: Filter invalid rows
- drop rows with missing required metric values
- optionally drop rows below `min_fitness`

### Step 2: Candidate grids
- Chamfer grid: percentile sweep over positives (e.g., p70..p99)
- Haus95 grid: percentile sweep over positives
- Volume grid: percentile sweep over positives
- BBox grid: percentile sweep over positives

### Step 3: Evaluate policy per grid point
Policy under test:
- hard gates must pass
- metric thresholds must pass

Compute:
- TP, TN, FP, FN
- precision, recall, f1, accuracy, specificity

### Step 4: Select best thresholds
By objective:
- `precision`: maximize precision, tie-break by recall
- `recall`: maximize recall, tie-break by precision
- `f1`: maximize f1
- `balanced`: maximize `(specificity + recall)/2`

### Step 5: Tiered thresholds
If `tier` is present:
- tune per tier
- also produce global fallback thresholds

## JSON output shape (example)
```json
{
  "version": "1.0",
  "objective": "f1",
  "global": {
    "chamfer_threshold_mm": 0.95,
    "hausdorff95_threshold_mm": 1.25,
    "volume_threshold_percent": 2.2,
    "bbox_threshold_mm": 1.1,
    "min_icp_fitness": 0.35
  },
  "tier_overrides": {
    "easy": {"chamfer_threshold_mm": 0.8, "hausdorff95_threshold_mm": 1.0, "volume_threshold_percent": 1.8, "bbox_threshold_mm": 0.9},
    "medium": {"chamfer_threshold_mm": 1.0, "hausdorff95_threshold_mm": 1.3, "volume_threshold_percent": 2.2, "bbox_threshold_mm": 1.2},
    "hard": {"chamfer_threshold_mm": 1.3, "hausdorff95_threshold_mm": 1.8, "volume_threshold_percent": 3.0, "bbox_threshold_mm": 1.6}
  },
  "selected_metrics": {
    "precision": 0.93,
    "recall": 0.88,
    "f1": 0.90,
    "specificity": 0.94,
    "accuracy": 0.91
  }
}
```

## Integration steps
1. Add calibration dataset generator/export script.
2. Run `tune_thresholds.py` on calibration labels.
3. Write selected thresholds into benchmark config.
4. Re-run validation set and confirm target FP/FN rates.

## Validation checklist
- identical pairs pass
- rigid transforms pass
- known bad pairs fail
- chosen thresholds stable across bootstrap resamples

## Nice-to-have extensions
- bootstrap confidence intervals for recommended thresholds
- robust optimization under class imbalance
- separate policies for deterministic vs inference tracks

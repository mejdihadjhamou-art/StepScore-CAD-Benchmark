#!/usr/bin/env python3
"""
StepScore Threshold Tuning - Tune all 44 metrics using user-provided labels.

Finds the optimal threshold for each metric that maximizes F1 score
for distinguishing positive from negative CAD generations.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

EXCEL = Path(".stepscore_harness_runs/final73_anthropic_run_02/labeled_pairs_for_review.xlsx")
OUTPUT_DIR = Path(".stepscore_harness_runs/final73_anthropic_run_02/tuning_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("🎯 StepScore Threshold Tuning")
print("="*70)

# Load data
df = pd.read_excel(EXCEL)
print(f"\n📊 Loaded {len(df)} labeled pairs")
print(f"   Positive: {(df['label'] == 'positive').sum()}")
print(f"   Negative: {(df['label'] == 'negative').sum()}")

# Identify metric columns (exclude metadata)
SKIP_COLS = {'label', 'pair_id', 'task_id', 'model_name', 'prompt_level',
             'family', 'model', 'overall_pass'}
metric_cols = [c for c in df.columns if c not in SKIP_COLS and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

print(f"\n🔍 Found {len(metric_cols)} numeric metrics to tune\n")

# Convert labels to binary: positive=1, negative=0
y_true = (df['label'] == 'positive').astype(int).values


def compute_f1(y_true, y_pred):
    """Compute precision, recall, F1"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)

    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'accuracy': round(accuracy, 4)
    }


def tune_metric(values, y_true, metric_name):
    """
    Find optimal threshold for a single metric.
    Tests both directions: >= threshold (higher is better) and <= threshold (lower is better).
    """
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 10:
        return None

    valid_values = values[valid_mask]
    valid_labels = y_true[valid_mask]

    # Generate candidate thresholds
    percentiles = np.percentile(valid_values, np.arange(1, 100, 0.5))
    unique_vals = np.unique(valid_values)
    
    # Combine unique midpoints and percentiles
    if len(unique_vals) > 1:
        midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2
        candidates = np.unique(np.concatenate([percentiles, midpoints]))
    else:
        candidates = percentiles

    best = {'f1': -1}

    for thresh in candidates:
        # Test: predict positive if value >= threshold (higher is better)
        y_pred_ge = (valid_values >= thresh).astype(int)
        stats_ge = compute_f1(valid_labels, y_pred_ge)

        if stats_ge['f1'] > best['f1']:
            best = {**stats_ge, 'threshold': round(float(thresh), 6),
                    'direction': 'gte', 'description': f'{metric_name} >= {thresh:.6f}'}

        # Test: predict positive if value <= threshold (lower is better)
        y_pred_le = (valid_values <= thresh).astype(int)
        stats_le = compute_f1(valid_labels, y_pred_le)

        if stats_le['f1'] > best['f1']:
            best = {**stats_le, 'threshold': round(float(thresh), 6),
                    'direction': 'lte', 'description': f'{metric_name} <= {thresh:.6f}'}

    best['metric'] = metric_name
    best['n_samples'] = int(valid_mask.sum())
    best['value_range'] = [round(float(valid_values.min()), 6), round(float(valid_values.max()), 6)]
    best['mean_positive'] = round(float(valid_values[valid_labels == 1].mean()), 6) if (valid_labels == 1).sum() > 0 else None
    best['mean_negative'] = round(float(valid_values[valid_labels == 0].mean()), 6) if (valid_labels == 0).sum() > 0 else None

    return best


# Run tuning on all metrics
results = []
print(f"{'Metric':<45} {'F1':>6} {'Acc':>6} {'Direction':>5} {'Threshold':>12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
print("-" * 100)

for metric in sorted(metric_cols):
    values = df[metric].values.astype(float)
    result = tune_metric(values, y_true, metric)

    if result is None:
        print(f"  {metric:<43} SKIPPED (insufficient data)")
        continue

    results.append(result)

    dir_symbol = ">=" if result['direction'] == 'gte' else "<="
    print(f"  {metric:<43} {result['f1']:>6.3f} {result['accuracy']:>6.3f} "
          f"  {dir_symbol}  {result['threshold']:>10.4f}  "
          f"{result['tp']:>4} {result['fp']:>4} {result['fn']:>4} {result['tn']:>4}")

# Sort by F1 score
results.sort(key=lambda x: x['f1'], reverse=True)

# Print summary
print("\n" + "="*70)
print("📊 TOP 10 METRICS BY F1 SCORE")
print("="*70)
for i, r in enumerate(results[:10]):
    dir_symbol = ">=" if r['direction'] == 'gte' else "<="
    print(f"  {i+1:>2}. {r['metric']:<40} F1={r['f1']:.4f}  "
          f"({dir_symbol} {r['threshold']:.4f})")

print(f"\n📊 BOTTOM 5 METRICS (least discriminative)")
for r in results[-5:]:
    dir_symbol = ">=" if r['direction'] == 'gte' else "<="
    print(f"      {r['metric']:<40} F1={r['f1']:.4f}")

# Save results
# 1. Full results JSON
with open(OUTPUT_DIR / "all_thresholds.json", 'w') as f:
    json.dump(results, f, indent=2)

# 2. Threshold overrides (for the harness)
overrides = {}
for r in results:
    overrides[r['metric']] = {
        'threshold': r['threshold'],
        'direction': r['direction'],
        'f1': r['f1'],
        'precision': r['precision'],
        'recall': r['recall']
    }
with open(OUTPUT_DIR / "threshold_overrides.json", 'w') as f:
    json.dump(overrides, f, indent=2)

# 3. Summary CSV
summary_df = pd.DataFrame(results)
summary_df.to_csv(OUTPUT_DIR / "tuning_summary.csv", index=False)

# 4. Summary stats
total_metrics = len(results)
perfect_f1 = sum(1 for r in results if r['f1'] >= 0.99)
good_f1 = sum(1 for r in results if r['f1'] >= 0.8)
avg_f1 = np.mean([r['f1'] for r in results])

summary = {
    'total_pairs': len(df),
    'positive_count': int((df['label'] == 'positive').sum()),
    'negative_count': int((df['label'] == 'negative').sum()),
    'total_metrics_tuned': total_metrics,
    'metrics_with_perfect_f1': perfect_f1,
    'metrics_with_good_f1': good_f1,
    'average_f1': round(avg_f1, 4),
    'top_10_metrics': [r['metric'] for r in results[:10]],
    'results': results
}
with open(OUTPUT_DIR / "calibration_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n" + "="*70)
print(f"✅ TUNING COMPLETE")
print(f"="*70)
print(f"\n  Total metrics tuned:        {total_metrics}")
print(f"  Metrics with F1 >= 0.99:    {perfect_f1}")
print(f"  Metrics with F1 >= 0.80:    {good_f1}")
print(f"  Average F1 across all:      {avg_f1:.4f}")
print(f"\n  Results saved to: {OUTPUT_DIR}/")
print(f"    - all_thresholds.json")
print(f"    - threshold_overrides.json")
print(f"    - tuning_summary.csv")
print(f"    - calibration_summary.json")
print(f"\n" + "="*70)

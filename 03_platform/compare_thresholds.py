#!/usr/bin/env python3
"""
Threshold Comparison Tool - Compare default vs tuned thresholds
and see performance difference on labeled data.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

HARNESS = Path(".stepscore_harness_runs/final73_anthropic_run_02")
EXCEL = HARNESS / "labeled_pairs_for_review.xlsx"
TUNED = HARNESS / "tuning_results" / "threshold_overrides.json"

# ── Original defaults from metric_engine.py ──
DEFAULTS = {
    "valid_cad_rate": {"threshold": 1.0, "direction": "gte"},
    "alignment_quality_icp_fitness": {"threshold": 0.30, "direction": "gte"},
    "alignment_inlier_rmse_mm": {"threshold": 1.00, "direction": "lte"},
    "chamfer_distance_mm": {"threshold": 1.00, "direction": "lte"},
    "edge_chamfer_mm": {"threshold": 0.80, "direction": "lte"},
    "hausdorff_95p_mm": {"threshold": 1.50, "direction": "lte"},
    "hausdorff_99p_mm": {"threshold": 2.50, "direction": "lte"},
    "point_to_surface_mean_mm": {"threshold": 0.80, "direction": "lte"},
    "point_to_surface_max_mm": {"threshold": 4.00, "direction": "lte"},
    "volume_diff_percent": {"threshold": 2.00, "direction": "lte"},
    "signed_volume_diff_percent": {"threshold": 2.00, "direction": "lte"},
    "surface_area_diff_percent": {"threshold": 3.00, "direction": "lte"},
    "bbox_error_max_mm": {"threshold": 1.00, "direction": "lte"},
    "bbox_error_axis_x_mm": {"threshold": 1.00, "direction": "lte"},
    "bbox_error_axis_y_mm": {"threshold": 1.00, "direction": "lte"},
    "bbox_error_axis_z_mm": {"threshold": 1.00, "direction": "lte"},
    "obb_error_max_mm": {"threshold": 1.50, "direction": "lte"},
    "centroid_offset_mm": {"threshold": 1.00, "direction": "lte"},
    "inertia_tensor_error": {"threshold": 0.10, "direction": "lte"},
    "mass_properties_error": {"threshold": 0.10, "direction": "lte"},
    "component_count_match": {"threshold": 1.0, "direction": "gte"},
    "watertight_manifold_pass": {"threshold": 1.0, "direction": "gte"},
    "self_intersection_count": {"threshold": 0.0, "direction": "lte"},
    "euler_genus_match": {"threshold": 1.0, "direction": "gte"},
    "void_hole_count_match": {"threshold": 1.0, "direction": "gte"},
    "feature_count_match": {"threshold": 0.90, "direction": "gte"},
    "critical_dimension_error_mm": {"threshold": 0.50, "direction": "lte"},
    "tolerance_band_pass_rate": {"threshold": 0.90, "direction": "gte"},
    "feature_edge_distance_mm": {"threshold": 0.80, "direction": "lte"},
    "normal_consistency": {"threshold": 0.95, "direction": "gte"},
    "normal_angle_error_deg": {"threshold": 12.0, "direction": "lte"},
    "curvature_distribution_error": {"threshold": 0.15, "direction": "lte"},
    "cross_section_iou": {"threshold": 0.90, "direction": "gte"},
    "slice_contour_distance_mm": {"threshold": 0.80, "direction": "lte"},
    "voxel_iou": {"threshold": 0.88, "direction": "gte"},
    "occupancy_precision": {"threshold": 0.90, "direction": "gte"},
    "occupancy_recall": {"threshold": 0.90, "direction": "gte"},
    "occupancy_f1": {"threshold": 0.90, "direction": "gte"},
    "emd_distance": {"threshold": 0.02, "direction": "lte"},
    "silhouette_iou": {"threshold": 0.92, "direction": "gte"},
    "render_ssim": {"threshold": 0.90, "direction": "gte"},
    "render_lpips": {"threshold": 0.15, "direction": "lte"},
    "registration_failure_rate": {"threshold": 0.05, "direction": "lte"},
    "composite_weighted_score": {"threshold": 0.85, "direction": "gte"},
    "quality_score_0_100": {"threshold": 70.0, "direction": "gte"},
}

def _main():
    # Load data
    df = pd.read_excel(EXCEL)
    y_true = (df['label'] == 'positive').astype(int).values
    tuned = json.loads(TUNED.read_text())

def evaluate(thresholds, df, y_true):
    """Evaluate a threshold set against labeled data."""
    results = {}
    for metric, config in thresholds.items():
        if metric not in df.columns:
            continue
        values = df[metric].values.astype(float)
        valid = ~np.isnan(values)
        if valid.sum() < 10:
            continue

        t = config['threshold']
        d = config['direction']
        if d == 'gte':
            pred = (values[valid] >= t).astype(int)
        else:
            pred = (values[valid] <= t).astype(int)

        yt = y_true[valid]
        tp = int(np.sum((yt == 1) & (pred == 1)))
        fp = int(np.sum((yt == 0) & (pred == 1)))
        fn = int(np.sum((yt == 1) & (pred == 0)))
        tn = int(np.sum((yt == 0) & (pred == 0)))
        prec = tp / (tp+fp) if (tp+fp) else 0
        rec  = tp / (tp+fn) if (tp+fn) else 0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0
        acc  = (tp+tn) / len(yt)

        results[metric] = {
            'threshold': t, 'direction': d,
            'f1': round(f1, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'accuracy': round(acc, 4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    return results

    # Evaluate both sets
    print("="*90)
    print("THRESHOLD COMPARISON: Default (Set 1) vs Tuned (Set 2)")
    print("="*90)
    print(f"\n  Data: {len(df)} pairs ({(y_true==1).sum()} positive, {(y_true==0).sum()} negative)\n")

    res_default = evaluate(DEFAULTS, df, y_true)
    res_tuned   = evaluate(tuned, df, y_true)

    # Header
    print(f"{'Metric':<35} | {'SET 1 (Default)':^24} | {'SET 2 (Tuned)':^24} | {'F1':>6}")
    print(f"{'':35} | {'Thresh':>8} {'F1':>6} {'Acc':>6}   | {'Thresh':>8} {'F1':>6} {'Acc':>6}   |")
    print("-"*100)

    all_metrics = sorted(set(list(res_default.keys()) | set(res_tuned.keys())))

    total_f1_default = []
    total_f1_tuned = []
    improved = 0
    degraded = 0
    unchanged = 0

    for m in all_metrics:
        d = res_default.get(m)
        t = res_tuned.get(m)

        d_thresh = f"{d['threshold']:>8.4f}" if d else "    N/A "
        d_f1     = f"{d['f1']:>6.3f}"       if d else "   N/A"
        d_acc    = f"{d['accuracy']:>6.3f}"  if d else "   N/A"
        t_thresh = f"{t['threshold']:>8.4f}" if t else "    N/A "
        t_f1     = f"{t['f1']:>6.3f}"       if t else "   N/A"
        t_acc    = f"{t['accuracy']:>6.3f}"  if t else "   N/A"

        if d and t:
            delta = t['f1'] - d['f1']
            if delta > 0.001:
                arrow = f"  +{delta:>+.3f}"
                improved += 1
            elif delta < -0.001:
                arrow = f"  {delta:>+.3f}"
                degraded += 1
            else:
                arrow = "     ="
                unchanged += 1
            total_f1_default.append(d['f1'])
            total_f1_tuned.append(t['f1'])
        elif t:
            arrow = "   NEW"
            total_f1_tuned.append(t['f1'])
        else:
            arrow = ""

        print(f"  {m:<33} | {d_thresh} {d_f1} {d_acc}   | {t_thresh} {t_f1} {t_acc}   | {arrow}")

    # Summary
    avg_default = np.mean(total_f1_default) if total_f1_default else 0
    avg_tuned   = np.mean(total_f1_tuned)   if total_f1_tuned   else 0

    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"\n  {'':35}   SET 1 (Default)   SET 2 (Tuned)")
    print(f"  {'Average F1 Score':<35}   {avg_default:>10.4f}       {avg_tuned:>10.4f}")
    print(f"  {'Total metrics compared':<35}   {len(total_f1_default):>10}       {len(total_f1_tuned):>10}")
    print(f"\n  Improved  (Tuned > Default):  {improved} metrics")
    print(f"  Unchanged (Tuned = Default):  {unchanged} metrics")
    print(f"  Degraded  (Tuned < Default):  {degraded} metrics")
    print(f"\n  Overall F1 improvement:       {avg_tuned - avg_default:>+.4f} ({(avg_tuned-avg_default)/avg_default*100:>+.1f}%)")
    print("="*90)

    # Save comparison
    comparison = {
        'set1_name': 'Default Thresholds (metric_engine.py)',
        'set2_name': 'Tuned Thresholds (user-labeled)',
        'set1_avg_f1': round(avg_default, 4),
        'set2_avg_f1': round(avg_tuned, 4),
        'improvement': round(avg_tuned - avg_default, 4),
        'improved_count': improved,
        'unchanged_count': unchanged,
        'degraded_count': degraded,
        'per_metric': {}
    }
    for m in all_metrics:
        comparison['per_metric'][m] = {
            'default': res_default.get(m),
            'tuned': res_tuned.get(m)
        }

    out = HARNESS / "tuning_results" / "comparison.json"
    with open(out, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved to: {out}\n")


if __name__ == "__main__":
    _main()

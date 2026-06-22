#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DIRECTION_FALLBACK: Dict[str, str] = {
    "valid_cad_rate": "higher_better",
    "alignment_quality_icp_fitness": "higher_better",
    "alignment_inlier_rmse_mm": "lower_better",
    "chamfer_distance_mm": "lower_better",
    "edge_chamfer_mm": "lower_better",
    "hausdorff_95p_mm": "lower_better",
    "hausdorff_99p_mm": "lower_better",
    "point_to_surface_mean_mm": "lower_better",
    "point_to_surface_max_mm": "lower_better",
    "volume_diff_percent": "lower_better",
    "signed_volume_diff_percent": "lower_better",
    "surface_area_diff_percent": "lower_better",
    "bbox_error_max_mm": "lower_better",
    "bbox_error_axis_x_mm": "lower_better",
    "bbox_error_axis_y_mm": "lower_better",
    "bbox_error_axis_z_mm": "lower_better",
    "obb_error_max_mm": "lower_better",
    "centroid_offset_mm": "lower_better",
    "inertia_tensor_error": "lower_better",
    "mass_properties_error": "lower_better",
    "component_count_match": "higher_better",
    "watertight_manifold_pass": "higher_better",
    "self_intersection_count": "lower_better",
    "euler_genus_match": "higher_better",
    "void_hole_count_match": "higher_better",
    "feature_count_match": "higher_better",
    "critical_dimension_error_mm": "lower_better",
    "tolerance_band_pass_rate": "higher_better",
    "feature_edge_distance_mm": "lower_better",
    "normal_consistency": "higher_better",
    "normal_angle_error_deg": "lower_better",
    "curvature_distribution_error": "lower_better",
    "cross_section_iou": "higher_better",
    "slice_contour_distance_mm": "lower_better",
    "voxel_iou": "higher_better",
    "occupancy_precision": "higher_better",
    "occupancy_recall": "higher_better",
    "occupancy_f1": "higher_better",
    "emd_distance": "lower_better",
    "silhouette_iou": "higher_better",
    "render_ssim": "higher_better",
    "render_lpips": "lower_better",
    "registration_failure_rate": "lower_better",
    "composite_weighted_score": "higher_better",
}

DEFAULT_PERCENTILES = [60, 65, 70, 75, 80, 85, 90, 92, 95, 97, 99]
OBJECTIVES = {"f1", "balanced", "precision", "recall", "cost"}
NAN_POLICIES = {"fail", "skip"}


@dataclass
class Confusion:
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass
class Scores:
    precision: float
    recall: float
    specificity: float
    accuracy: float
    f1: float
    balanced: float
    cost: float


@dataclass
class MetricTuneResult:
    metric: str
    direction: str
    status: str
    reason: str
    rows_used: int
    positives: int
    negatives: int
    nan_rate: float
    old_threshold: Optional[float]
    new_threshold: Optional[float]
    objective_value: Optional[float]
    scores: Optional[Scores]
    confusion: Optional[Confusion]


def _safe_float(v: Any) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _parse_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("percentile() with empty values")
    vals = sorted(values)
    if p <= 0:
        return float(vals[0])
    if p >= 100:
        return float(vals[-1])
    k = (len(vals) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(vals[lo])
    w = k - lo
    return float(vals[lo] * (1.0 - w) + vals[hi] * w)


def _split_tokens(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _load_metric_meta(path: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, float]]:
    if path is None or not path.exists():
        return {}, {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    metrics = obj.get("metrics", {})
    directions: Dict[str, str] = {}
    thresholds: Dict[str, float] = {}
    if isinstance(metrics, dict):
        for name, meta in metrics.items():
            if not isinstance(meta, dict):
                continue
            d = str(meta.get("direction", "")).strip().lower()
            if d in {"lower_better", "higher_better", "equal"}:
                directions[name] = d
            t = _safe_float(meta.get("threshold_default"))
            if t is not None:
                thresholds[name] = t
    return directions, thresholds


def _infer_metric_columns(rows: List[Dict[str, str]], label_col: str) -> List[str]:
    if not rows:
        return []
    ignore_prefixes = {
        "pair_id",
        "run_id",
        "label",
        "tier",
        "notes",
        "task_mode",
        "provider",
        "model",
        "prompt_level",
        "part_id",
        "family",
        "reference_path",
        "generated_path",
        "reference_mesh_path",
        "generated_mesh_path",
        "fast_mode",
        "sample_points",
        "voxel_pitch_mm",
        "grading_profile",
        "pass_rate",
        "quality_score_0_100",
        "overall_pass",
    }
    cols: List[str] = []
    for c in rows[0].keys():
        lc = c.strip().lower()
        if lc == label_col.strip().lower():
            continue
        if lc in ignore_prefixes:
            continue
        cols.append(c)
    return cols


def _compute_scores(
    confusion: Confusion,
    fp_cost: float,
    fn_cost: float,
) -> Scores:
    tp, tn, fp, fn = confusion.tp, confusion.tn, confusion.fp, confusion.fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced = (recall + specificity) / 2.0
    cost = fp_cost * fp + fn_cost * fn
    return Scores(
        precision=precision,
        recall=recall,
        specificity=specificity,
        accuracy=accuracy,
        f1=f1,
        balanced=balanced,
        cost=cost,
    )


def _objective_value(scores: Scores, objective: str) -> float:
    if objective == "precision":
        return scores.precision
    if objective == "recall":
        return scores.recall
    if objective == "f1":
        return scores.f1
    if objective == "balanced":
        return scores.balanced
    if objective == "cost":
        return -scores.cost
    raise ValueError(f"Unsupported objective: {objective}")


def _compare_key(scores: Scores, objective: str) -> Tuple[float, float, float, float, float, float]:
    if objective == "cost":
        return (-scores.cost, scores.f1, scores.balanced, scores.precision, scores.recall, scores.accuracy)
    return (
        _objective_value(scores, objective),
        -scores.cost,
        scores.f1,
        scores.balanced,
        scores.precision,
        scores.recall,
    )


def _predict_pass(value: Optional[float], threshold: float, direction: str, nan_policy: str) -> Optional[bool]:
    if value is None:
        if nan_policy == "skip":
            return None
        return False
    if direction == "lower_better":
        return bool(value <= threshold)
    if direction == "higher_better":
        return bool(value >= threshold)
    if direction == "equal":
        return bool(abs(value - threshold) < 1e-9)
    return None


def _evaluate_metric(
    rows: List[Dict[str, str]],
    metric: str,
    direction: str,
    threshold: float,
    label_col: str,
    pos_labels: set[str],
    neg_labels: set[str],
    nan_policy: str,
    fp_cost: float,
    fn_cost: float,
) -> Tuple[Confusion, Scores, int, int, int]:
    tp = tn = fp = fn = 0
    used = 0
    nan_count = 0

    for row in rows:
        label = row.get(label_col, "").strip().lower()
        if label not in pos_labels and label not in neg_labels:
            continue
        is_positive = label in pos_labels

        value = _safe_float(row.get(metric, ""))
        if value is None:
            nan_count += 1

        pred = _predict_pass(value, threshold, direction, nan_policy)
        if pred is None:
            continue

        used += 1
        if pred and is_positive:
            tp += 1
        elif pred and not is_positive:
            fp += 1
        elif (not pred) and is_positive:
            fn += 1
        else:
            tn += 1

    confusion = Confusion(tp=tp, tn=tn, fp=fp, fn=fn)
    scores = _compute_scores(confusion, fp_cost=fp_cost, fn_cost=fn_cost)
    return confusion, scores, used, nan_count, tp + fn


def _candidate_thresholds(
    positives: List[float],
    negatives: List[float],
    percentiles: Sequence[float],
) -> List[float]:
    source = positives if positives else (positives + negatives)
    if not source:
        return []
    all_vals = positives + negatives
    cands = set()
    for p in percentiles:
        cands.add(round(_percentile(source, p), 9))
    cands.add(round(min(source), 9))
    cands.add(round(max(source), 9))
    cands.add(round(sum(source) / max(len(source), 1), 9))
    cands.add(round(_percentile(source, 50), 9))
    if all_vals:
        cands.add(round(_percentile(all_vals, 50), 9))
        cands.add(round(_percentile(all_vals, 10), 9))
        cands.add(round(_percentile(all_vals, 90), 9))
    return sorted(cands)


def tune_metric(
    rows: List[Dict[str, str]],
    metric: str,
    direction: str,
    label_col: str,
    pos_labels: set[str],
    neg_labels: set[str],
    percentiles: Sequence[float],
    objective: str,
    nan_policy: str,
    fp_cost: float,
    fn_cost: float,
    min_pos: int,
    min_neg: int,
    old_threshold: Optional[float],
) -> MetricTuneResult:
    positives: List[float] = []
    negatives: List[float] = []
    considered = 0
    nan_count = 0

    for row in rows:
        label = row.get(label_col, "").strip().lower()
        if label not in pos_labels and label not in neg_labels:
            continue
        considered += 1
        v = _safe_float(row.get(metric, ""))
        if v is None:
            nan_count += 1
            continue
        if label in pos_labels:
            positives.append(v)
        else:
            negatives.append(v)

    if len(positives) < min_pos or len(negatives) < min_neg:
        return MetricTuneResult(
            metric=metric,
            direction=direction,
            status="skipped",
            reason=f"insufficient_support(pos={len(positives)},neg={len(negatives)})",
            rows_used=0,
            positives=len(positives),
            negatives=len(negatives),
            nan_rate=(nan_count / considered) if considered else 0.0,
            old_threshold=old_threshold,
            new_threshold=None,
            objective_value=None,
            scores=None,
            confusion=None,
        )

    cands = _candidate_thresholds(positives=positives, negatives=negatives, percentiles=percentiles)
    if not cands:
        return MetricTuneResult(
            metric=metric,
            direction=direction,
            status="skipped",
            reason="no_candidate_thresholds",
            rows_used=0,
            positives=len(positives),
            negatives=len(negatives),
            nan_rate=(nan_count / considered) if considered else 0.0,
            old_threshold=old_threshold,
            new_threshold=None,
            objective_value=None,
            scores=None,
            confusion=None,
        )

    best_threshold: Optional[float] = None
    best_conf: Optional[Confusion] = None
    best_scores: Optional[Scores] = None
    best_key: Optional[Tuple[float, float, float, float, float, float]] = None
    best_rows_used = 0

    for thr in cands:
        conf, scores, rows_used, _, _ = _evaluate_metric(
            rows=rows,
            metric=metric,
            direction=direction,
            threshold=thr,
            label_col=label_col,
            pos_labels=pos_labels,
            neg_labels=neg_labels,
            nan_policy=nan_policy,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        if rows_used == 0:
            continue
        key = _compare_key(scores, objective=objective)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = thr
            best_conf = conf
            best_scores = scores
            best_rows_used = rows_used

    if best_threshold is None or best_conf is None or best_scores is None:
        return MetricTuneResult(
            metric=metric,
            direction=direction,
            status="skipped",
            reason="no_valid_candidate_after_eval",
            rows_used=0,
            positives=len(positives),
            negatives=len(negatives),
            nan_rate=(nan_count / considered) if considered else 0.0,
            old_threshold=old_threshold,
            new_threshold=None,
            objective_value=None,
            scores=None,
            confusion=None,
        )

    return MetricTuneResult(
        metric=metric,
        direction=direction,
        status="ok",
        reason="",
        rows_used=best_rows_used,
        positives=len(positives),
        negatives=len(negatives),
        nan_rate=(nan_count / considered) if considered else 0.0,
        old_threshold=old_threshold,
        new_threshold=best_threshold,
        objective_value=_objective_value(best_scores, objective=objective),
        scores=best_scores,
        confusion=best_conf,
    )


def _write_report_csv(path: Path, results: List[MetricTuneResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "direction",
        "status",
        "reason",
        "rows_used",
        "positives",
        "negatives",
        "nan_rate",
        "old_threshold",
        "new_threshold",
        "objective_value",
        "precision",
        "recall",
        "f1",
        "specificity",
        "accuracy",
        "balanced",
        "cost",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "metric": r.metric,
                "direction": r.direction,
                "status": r.status,
                "reason": r.reason,
                "rows_used": r.rows_used,
                "positives": r.positives,
                "negatives": r.negatives,
                "nan_rate": f"{r.nan_rate:.6f}",
                "old_threshold": "" if r.old_threshold is None else f"{r.old_threshold:.9g}",
                "new_threshold": "" if r.new_threshold is None else f"{r.new_threshold:.9g}",
                "objective_value": "" if r.objective_value is None else f"{r.objective_value:.9g}",
                "precision": "",
                "recall": "",
                "f1": "",
                "specificity": "",
                "accuracy": "",
                "balanced": "",
                "cost": "",
                "tp": "",
                "tn": "",
                "fp": "",
                "fn": "",
            }
            if r.scores is not None and r.confusion is not None:
                row.update(
                    {
                        "precision": f"{r.scores.precision:.9g}",
                        "recall": f"{r.scores.recall:.9g}",
                        "f1": f"{r.scores.f1:.9g}",
                        "specificity": f"{r.scores.specificity:.9g}",
                        "accuracy": f"{r.scores.accuracy:.9g}",
                        "balanced": f"{r.scores.balanced:.9g}",
                        "cost": f"{r.scores.cost:.9g}",
                        "tp": r.confusion.tp,
                        "tn": r.confusion.tn,
                        "fp": r.confusion.fp,
                        "fn": r.confusion.fn,
                    }
                )
            w.writerow(row)


def _write_summary_md(
    path: Path,
    args: argparse.Namespace,
    total_rows: int,
    labeled_rows: int,
    results: List[MetricTuneResult],
) -> None:
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status != "ok"]
    lines: List[str] = []
    lines.append("# Threshold Tuning Summary")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Input CSV: `{Path(args.pairs_csv).resolve()}`")
    lines.append(f"- Objective: `{args.objective}`")
    lines.append(f"- FP cost: `{args.fp_cost}`")
    lines.append(f"- FN cost: `{args.fn_cost}`")
    lines.append(f"- Label column: `{args.label_column}`")
    lines.append(f"- Total rows in CSV: `{total_rows}`")
    lines.append(f"- Labeled rows used (positive/negative): `{labeled_rows}`")
    lines.append(f"- Metrics tuned successfully: `{len(ok)}`")
    lines.append(f"- Metrics skipped: `{len(skipped)}`")
    lines.append("")
    lines.append("## Tuned Metrics")
    lines.append("")
    for r in sorted(ok, key=lambda x: x.metric):
        assert r.scores is not None
        lines.append(
            f"- `{r.metric}`: threshold `{r.old_threshold}` -> `{r.new_threshold}`, "
            f"F1 `{r.scores.f1:.3f}`, balanced `{r.scores.balanced:.3f}`, cost `{r.scores.cost:.1f}`"
        )
    lines.append("")
    if skipped:
        lines.append("## Skipped Metrics")
        lines.append("")
        for r in sorted(skipped, key=lambda x: x.metric):
            lines.append(f"- `{r.metric}`: {r.reason}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_percentiles(raw: str) -> List[float]:
    out: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = float(tok)
        if val < 0 or val > 100:
            raise ValueError(f"Percentile out of range: {val}")
        out.append(val)
    if not out:
        raise ValueError("No valid percentiles provided.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic per-metric threshold tuning for STEPScore from labeled pairs CSV."
    )
    parser.add_argument("--pairs-csv", required=True, help="Labeled wide CSV (one row per pair).")
    parser.add_argument(
        "--metrics-meta-json",
        default=None,
        help="Optional metrics metadata from build_pairs_from_runs.py (direction + old thresholds).",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Label column name in CSV.",
    )
    parser.add_argument(
        "--positive-labels",
        default="positive,pass",
        help="Comma-separated labels treated as positive class.",
    )
    parser.add_argument(
        "--negative-labels",
        default="negative,fail,rework",
        help="Comma-separated labels treated as negative class.",
    )
    parser.add_argument(
        "--objective",
        choices=sorted(OBJECTIVES),
        default="balanced",
        help="Optimization target.",
    )
    parser.add_argument(
        "--fp-cost",
        type=float,
        default=5.0,
        help="False-pass penalty for cost-sensitive tuning.",
    )
    parser.add_argument(
        "--fn-cost",
        type=float,
        default=1.0,
        help="False-fail penalty for cost-sensitive tuning.",
    )
    parser.add_argument(
        "--nan-policy",
        choices=sorted(NAN_POLICIES),
        default="fail",
        help="How to handle missing/NaN metric values during tuning.",
    )
    parser.add_argument(
        "--candidate-percentiles",
        default=",".join(str(x) for x in DEFAULT_PERCENTILES),
        help="Comma-separated percentile candidates used to generate threshold search grid.",
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=10,
        help="Minimum positive samples required per metric.",
    )
    parser.add_argument(
        "--min-negative",
        type=int,
        default=10,
        help="Minimum negative samples required per metric.",
    )
    parser.add_argument(
        "--tune-metrics",
        default="",
        help="Optional comma-separated metric subset to tune. Empty = tune all found metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="threshold_tuning/output",
        help="Output directory for JSON/CSV/MD artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs_csv = Path(args.pairs_csv).expanduser().resolve()
    if not pairs_csv.exists():
        raise FileNotFoundError(f"CSV not found: {pairs_csv}")

    rows = _parse_csv(pairs_csv)
    if not rows:
        raise ValueError("Input CSV has no rows.")

    pos_labels = set(_split_tokens(args.positive_labels))
    neg_labels = set(_split_tokens(args.negative_labels))
    if not pos_labels or not neg_labels:
        raise ValueError("Positive/negative labels must not be empty.")
    overlap = pos_labels.intersection(neg_labels)
    if overlap:
        raise ValueError(f"Label overlap between positive and negative sets: {sorted(overlap)}")

    labeled_rows = 0
    for r in rows:
        lab = r.get(args.label_column, "").strip().lower()
        if lab in pos_labels or lab in neg_labels:
            labeled_rows += 1
    if labeled_rows == 0:
        raise ValueError("No labeled rows found (positive/negative).")

    percentiles = _parse_percentiles(args.candidate_percentiles)
    directions_from_meta: Dict[str, str] = {}
    thresholds_from_meta: Dict[str, float] = {}
    meta_path = Path(args.metrics_meta_json).expanduser().resolve() if args.metrics_meta_json else None
    if meta_path:
        directions_from_meta, thresholds_from_meta = _load_metric_meta(meta_path)

    metric_columns = _infer_metric_columns(rows, label_col=args.label_column)
    tune_subset = set(_split_tokens(args.tune_metrics))
    if tune_subset:
        metric_columns = [m for m in metric_columns if m.strip().lower() in tune_subset]

    results: List[MetricTuneResult] = []
    recommended_thresholds: Dict[str, float] = {}

    for metric in metric_columns:
        direction = (
            directions_from_meta.get(metric)
            or DEFAULT_DIRECTION_FALLBACK.get(metric)
            or "unknown"
        )
        if direction not in {"lower_better", "higher_better", "equal"}:
            results.append(
                MetricTuneResult(
                    metric=metric,
                    direction=direction,
                    status="skipped",
                    reason="unknown_direction",
                    rows_used=0,
                    positives=0,
                    negatives=0,
                    nan_rate=0.0,
                    old_threshold=thresholds_from_meta.get(metric),
                    new_threshold=None,
                    objective_value=None,
                    scores=None,
                    confusion=None,
                )
            )
            continue

        result = tune_metric(
            rows=rows,
            metric=metric,
            direction=direction,
            label_col=args.label_column,
            pos_labels=pos_labels,
            neg_labels=neg_labels,
            percentiles=percentiles,
            objective=args.objective,
            nan_policy=args.nan_policy,
            fp_cost=float(args.fp_cost),
            fn_cost=float(args.fn_cost),
            min_pos=int(args.min_positive),
            min_neg=int(args.min_negative),
            old_threshold=thresholds_from_meta.get(metric),
        )
        results.append(result)
        if result.status == "ok" and result.new_threshold is not None:
            recommended_thresholds[metric] = float(result.new_threshold)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_csv = output_dir / "metric_tuning_report.csv"
    summary_md = output_dir / "tuning_summary.md"
    thresholds_full_json = output_dir / "thresholds_recommended.json"
    thresholds_overrides_json = output_dir / "threshold_overrides.json"

    _write_report_csv(report_csv, results)
    _write_summary_md(
        path=summary_md,
        args=args,
        total_rows=len(rows),
        labeled_rows=labeled_rows,
        results=results,
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(pairs_csv),
        "input_meta_json": str(meta_path) if meta_path else None,
        "objective": args.objective,
        "fp_cost": float(args.fp_cost),
        "fn_cost": float(args.fn_cost),
        "nan_policy": args.nan_policy,
        "candidate_percentiles": percentiles,
        "rows_total": len(rows),
        "rows_labeled": labeled_rows,
        "recommended_thresholds": recommended_thresholds,
        "results_count_ok": sum(1 for r in results if r.status == "ok"),
        "results_count_skipped": sum(1 for r in results if r.status != "ok"),
    }
    thresholds_full_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    thresholds_overrides_json.write_text(json.dumps(recommended_thresholds, indent=2), encoding="utf-8")

    print(f"input_csv={pairs_csv}")
    print(f"rows_total={len(rows)}")
    print(f"rows_labeled={labeled_rows}")
    print(f"metrics_considered={len(metric_columns)}")
    print(f"metrics_tuned={sum(1 for r in results if r.status == 'ok')}")
    print(f"wrote={report_csv}")
    print(f"wrote={summary_md}")
    print(f"wrote={thresholds_full_json}")
    print(f"wrote={thresholds_overrides_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


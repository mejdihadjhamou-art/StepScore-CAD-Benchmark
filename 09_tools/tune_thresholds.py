#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_COLUMNS = (
    "pair_id",
    "label",
    "chamfer_mm",
    "haus95_mm",
    "haus99_mm",
    "volume_diff_pct",
    "bbox_max_diff_mm",
    "watertight_pass",
    "single_component_pass",
    "alignment_failed",
    "icp_fitness",
)

OBJECTIVES = ("precision", "recall", "f1", "balanced")

PERCENTILE_GRID = (70, 75, 80, 85, 90, 92, 95, 97, 99)


@dataclass
class Row:
    pair_id: str
    label: str
    tier: Optional[str]
    chamfer_mm: float
    haus95_mm: float
    haus99_mm: float
    volume_diff_pct: float
    bbox_max_diff_mm: float
    watertight_pass: int
    single_component_pass: int
    alignment_failed: int
    icp_fitness: float


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
    f1: float
    specificity: float
    accuracy: float
    balanced: float


@dataclass
class Thresholds:
    chamfer_threshold_mm: float
    hausdorff95_threshold_mm: float
    volume_threshold_percent: float
    bbox_threshold_mm: float
    min_icp_fitness: float


@dataclass
class CandidateResult:
    thresholds: Thresholds
    confusion: Confusion
    scores: Scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune CAD benchmark thresholds from labeled pair metrics.")
    parser.add_argument("--pairs-csv", required=True, help="Path to labeled pairs CSV")
    parser.add_argument("--objective", choices=OBJECTIVES, default="f1", help="Optimization objective")
    parser.add_argument("--tier-column", default=None, help="Optional tier column name (e.g. tier)")
    parser.add_argument("--min-fitness", type=float, default=None, help="Optional minimum ICP fitness filter")
    parser.add_argument("--output-dir", default=None, help="Output directory for artifacts")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    return parser.parse_args()


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile on empty sequence")
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ordered[int(k)])
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return float(d0 + d1)


def parse_bool_int(raw: str, field: str) -> int:
    value = raw.strip().lower()
    if value in ("1", "true", "yes"):
        return 1
    if value in ("0", "false", "no"):
        return 0
    raise ValueError(f"Invalid boolean/integer value for {field}: {raw}")


def read_rows(path: str, tier_column: Optional[str]) -> List[Row]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        if tier_column and tier_column not in reader.fieldnames:
            raise ValueError(f"Tier column '{tier_column}' not found in CSV")

        rows: List[Row] = []
        for idx, r in enumerate(reader, start=2):
            try:
                label = r["label"].strip().lower()
                if label not in ("positive", "negative"):
                    raise ValueError("label must be positive or negative")

                row = Row(
                    pair_id=r["pair_id"].strip(),
                    label=label,
                    tier=(r[tier_column].strip().lower() if tier_column else None),
                    chamfer_mm=float(r["chamfer_mm"]),
                    haus95_mm=float(r["haus95_mm"]),
                    haus99_mm=float(r["haus99_mm"]),
                    volume_diff_pct=float(r["volume_diff_pct"]),
                    bbox_max_diff_mm=float(r["bbox_max_diff_mm"]),
                    watertight_pass=parse_bool_int(r["watertight_pass"], "watertight_pass"),
                    single_component_pass=parse_bool_int(r["single_component_pass"], "single_component_pass"),
                    alignment_failed=parse_bool_int(r["alignment_failed"], "alignment_failed"),
                    icp_fitness=float(r["icp_fitness"]),
                )
                rows.append(row)
            except Exception as exc:
                raise ValueError(f"Row {idx} invalid: {exc}") from exc

    return rows


def filter_rows(rows: Iterable[Row], min_fitness: Optional[float]) -> List[Row]:
    out: List[Row] = []
    for row in rows:
        if min_fitness is not None and row.icp_fitness < min_fitness:
            continue
        out.append(row)
    return out


def bool_predict_pass(row: Row, th: Thresholds) -> bool:
    hard_gates_ok = (
        row.watertight_pass == 1
        and row.single_component_pass == 1
        and row.alignment_failed == 0
        and row.icp_fitness >= th.min_icp_fitness
    )
    if not hard_gates_ok:
        return False

    return (
        row.chamfer_mm <= th.chamfer_threshold_mm
        and row.haus95_mm <= th.hausdorff95_threshold_mm
        and row.volume_diff_pct <= th.volume_threshold_percent
        and row.bbox_max_diff_mm <= th.bbox_threshold_mm
    )


def evaluate(rows: Sequence[Row], th: Thresholds) -> CandidateResult:
    tp = tn = fp = fn = 0
    for row in rows:
        pred_pass = bool_predict_pass(row, th)
        actual_positive = row.label == "positive"
        if pred_pass and actual_positive:
            tp += 1
        elif pred_pass and not actual_positive:
            fp += 1
        elif not pred_pass and actual_positive:
            fn += 1
        else:
            tn += 1

    confusion = Confusion(tp=tp, tn=tn, fp=fp, fn=fn)
    scores = compute_scores(confusion)
    return CandidateResult(thresholds=th, confusion=confusion, scores=scores)


def compute_scores(c: Confusion) -> Scores:
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    specificity = c.tn / (c.tn + c.fp) if (c.tn + c.fp) else 0.0
    accuracy = (c.tp + c.tn) / (c.tp + c.tn + c.fp + c.fn) if (c.tp + c.tn + c.fp + c.fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced = (specificity + recall) / 2.0
    return Scores(
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        accuracy=accuracy,
        balanced=balanced,
    )


def objective_value(scores: Scores, objective: str) -> float:
    if objective == "precision":
        return scores.precision
    if objective == "recall":
        return scores.recall
    if objective == "f1":
        return scores.f1
    if objective == "balanced":
        return scores.balanced
    raise ValueError(f"Unsupported objective: {objective}")


def tie_break_key(scores: Scores, objective: str) -> Tuple[float, float, float, float, float]:
    if objective == "precision":
        return (scores.precision, scores.recall, scores.f1, scores.specificity, scores.accuracy)
    if objective == "recall":
        return (scores.recall, scores.precision, scores.f1, scores.specificity, scores.accuracy)
    if objective == "f1":
        return (scores.f1, scores.precision, scores.recall, scores.specificity, scores.accuracy)
    if objective == "balanced":
        return (scores.balanced, scores.f1, scores.precision, scores.recall, scores.accuracy)
    raise ValueError(f"Unsupported objective: {objective}")


def candidate_values_from_positives(rows: Sequence[Row], attr: str) -> List[float]:
    positives = [getattr(r, attr) for r in rows if r.label == "positive"]
    if not positives:
        raise ValueError("No positive rows found; cannot tune thresholds")

    vals = {round(percentile(positives, p), 6) for p in PERCENTILE_GRID}
    vals.add(round(max(positives), 6))
    vals.add(round(min(positives), 6))
    return sorted(vals)


def find_best_thresholds(rows: Sequence[Row], objective: str, min_fitness: float) -> CandidateResult:
    chamfer_candidates = candidate_values_from_positives(rows, "chamfer_mm")
    haus95_candidates = candidate_values_from_positives(rows, "haus95_mm")
    volume_candidates = candidate_values_from_positives(rows, "volume_diff_pct")
    bbox_candidates = candidate_values_from_positives(rows, "bbox_max_diff_mm")

    best: Optional[CandidateResult] = None
    best_key: Optional[Tuple[float, float, float, float, float]] = None

    for c_thr, h_thr, v_thr, b_thr in itertools.product(
        chamfer_candidates, haus95_candidates, volume_candidates, bbox_candidates
    ):
        th = Thresholds(
            chamfer_threshold_mm=float(c_thr),
            hausdorff95_threshold_mm=float(h_thr),
            volume_threshold_percent=float(v_thr),
            bbox_threshold_mm=float(b_thr),
            min_icp_fitness=float(min_fitness),
        )
        result = evaluate(rows, th)
        key = tie_break_key(result.scores, objective)
        if best is None or key > best_key:  # max tuple lexicographically
            best = result
            best_key = key

    assert best is not None
    return best


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown_report(
    path: str,
    objective: str,
    input_csv: str,
    filtered_count: int,
    raw_count: int,
    global_result: CandidateResult,
    tier_results: Dict[str, CandidateResult],
    min_fitness: float,
) -> None:
    lines: List[str] = []
    lines.append("# Threshold Tuning Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Input CSV: `{input_csv}`")
    lines.append(f"- Objective: `{objective}`")
    lines.append(f"- Rows used: `{filtered_count}` / `{raw_count}`")
    lines.append(f"- Minimum ICP fitness gate: `{min_fitness}`")
    lines.append("")

    def append_block(title: str, result: CandidateResult) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("### Recommended Thresholds")
        lines.append("")
        lines.append(f"- Chamfer (mm): `{result.thresholds.chamfer_threshold_mm:.6f}`")
        lines.append(f"- Hausdorff95 (mm): `{result.thresholds.hausdorff95_threshold_mm:.6f}`")
        lines.append(f"- Volume diff (%): `{result.thresholds.volume_threshold_percent:.6f}`")
        lines.append(f"- BBox max diff (mm): `{result.thresholds.bbox_threshold_mm:.6f}`")
        lines.append(f"- Min ICP fitness: `{result.thresholds.min_icp_fitness:.6f}`")
        lines.append("")
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append(f"- TP: `{result.confusion.tp}`")
        lines.append(f"- TN: `{result.confusion.tn}`")
        lines.append(f"- FP: `{result.confusion.fp}`")
        lines.append(f"- FN: `{result.confusion.fn}`")
        lines.append("")
        lines.append("### Metrics")
        lines.append("")
        lines.append(f"- Precision: `{result.scores.precision:.4f}`")
        lines.append(f"- Recall: `{result.scores.recall:.4f}`")
        lines.append(f"- F1: `{result.scores.f1:.4f}`")
        lines.append(f"- Specificity: `{result.scores.specificity:.4f}`")
        lines.append(f"- Accuracy: `{result.scores.accuracy:.4f}`")
        lines.append(f"- Balanced: `{result.scores.balanced:.4f}`")
        lines.append("")

    append_block("Global", global_result)
    for tier in sorted(tier_results.keys()):
        append_block(f"Tier: {tier}", tier_results[tier])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def maybe_generate_plots(rows: Sequence[Row], output_dir: str, disabled: bool) -> List[str]:
    if disabled:
        return []

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    created: List[str] = []
    metrics = [
        ("chamfer_mm", "Chamfer (mm)"),
        ("haus95_mm", "Hausdorff 95p (mm)"),
        ("volume_diff_pct", "Volume diff (%)"),
        ("bbox_max_diff_mm", "BBox max diff (mm)"),
    ]

    positives = [r for r in rows if r.label == "positive"]
    negatives = [r for r in rows if r.label == "negative"]

    for attr, title in metrics:
        p_vals = [getattr(r, attr) for r in positives]
        n_vals = [getattr(r, attr) for r in negatives]
        if not p_vals and not n_vals:
            continue

        plt.figure(figsize=(7, 4))
        bins = 30
        if n_vals:
            plt.hist(n_vals, bins=bins, alpha=0.6, label="negative", color="#d95f02")
        if p_vals:
            plt.hist(p_vals, bins=bins, alpha=0.6, label="positive", color="#1b9e77")
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"hist_{attr}.png")
        plt.savefig(out_path)
        plt.close()
        created.append(out_path)

    return created


def to_dict_thresholds(th: Thresholds) -> Dict[str, float]:
    return {
        "chamfer_threshold_mm": round(th.chamfer_threshold_mm, 6),
        "hausdorff95_threshold_mm": round(th.hausdorff95_threshold_mm, 6),
        "volume_threshold_percent": round(th.volume_threshold_percent, 6),
        "bbox_threshold_mm": round(th.bbox_threshold_mm, 6),
        "min_icp_fitness": round(th.min_icp_fitness, 6),
    }


def to_dict_scores(sc: Scores) -> Dict[str, float]:
    return {
        "precision": round(sc.precision, 6),
        "recall": round(sc.recall, 6),
        "f1": round(sc.f1, 6),
        "specificity": round(sc.specificity, 6),
        "accuracy": round(sc.accuracy, 6),
        "balanced": round(sc.balanced, 6),
    }


def run(args: argparse.Namespace) -> int:
    input_csv = os.path.abspath(args.pairs_csv)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_csv), "threshold_tuning_output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    raw_rows = read_rows(input_csv, tier_column=args.tier_column)
    min_fitness = args.min_fitness if args.min_fitness is not None else 0.0
    rows = filter_rows(raw_rows, min_fitness=min_fitness)

    if not rows:
        raise ValueError("No rows available after filtering")

    global_best = find_best_thresholds(rows, objective=args.objective, min_fitness=min_fitness)

    tier_results: Dict[str, CandidateResult] = {}
    if args.tier_column:
        grouped: Dict[str, List[Row]] = {}
        for r in rows:
            tier = (r.tier or "unknown").strip().lower() or "unknown"
            grouped.setdefault(tier, []).append(r)

        for tier, group_rows in grouped.items():
            if len(group_rows) < 4:
                continue
            positives = sum(1 for r in group_rows if r.label == "positive")
            negatives = sum(1 for r in group_rows if r.label == "negative")
            if positives == 0 or negatives == 0:
                continue
            tier_results[tier] = find_best_thresholds(
                group_rows,
                objective=args.objective,
                min_fitness=min_fitness,
            )

    plots = maybe_generate_plots(rows, output_dir, disabled=args.no_plots)

    payload = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": args.objective,
        "input_csv": input_csv,
        "rows_used": len(rows),
        "rows_total": len(raw_rows),
        "global": to_dict_thresholds(global_best.thresholds),
        "tier_overrides": {tier: to_dict_thresholds(result.thresholds) for tier, result in sorted(tier_results.items())},
        "selected_metrics": to_dict_scores(global_best.scores),
        "confusion": {
            "tp": global_best.confusion.tp,
            "tn": global_best.confusion.tn,
            "fp": global_best.confusion.fp,
            "fn": global_best.confusion.fn,
        },
        "artifacts": {
            "plots": [os.path.abspath(p) for p in plots],
        },
    }

    json_path = os.path.join(output_dir, "thresholds_recommended.json")
    report_path = os.path.join(output_dir, "threshold_report.md")

    write_json(json_path, payload)
    write_markdown_report(
        path=report_path,
        objective=args.objective,
        input_csv=input_csv,
        filtered_count=len(rows),
        raw_count=len(raw_rows),
        global_result=global_best,
        tier_results=tier_results,
        min_fitness=min_fitness,
    )

    print(f"Wrote: {json_path}")
    print(f"Wrote: {report_path}")
    if plots:
        print(f"Wrote {len(plots)} plot(s) to: {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

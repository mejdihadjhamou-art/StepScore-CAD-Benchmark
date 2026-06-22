#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_COLUMNS = [
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
]


def _safe_float(v: Any) -> float | None:
    try:
        out = float(v)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


@dataclass
class MetricMeta:
    direction: str
    thresholds_seen: List[float]
    count: int

    def add(self, direction: str, threshold: float | None) -> None:
        self.count += 1
        if direction and direction != self.direction:
            self.direction = "mixed"
        if threshold is not None:
            self.thresholds_seen.append(float(threshold))

    def threshold_default(self) -> float | None:
        if not self.thresholds_seen:
            return None
        vals = sorted(self.thresholds_seen)
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return float(vals[mid])
        return float((vals[mid - 1] + vals[mid]) / 2.0)


def build_rows(
    runs_root: Path,
    label_default: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, MetricMeta], int]:
    rows: List[Dict[str, Any]] = []
    metric_order: List[str] = []
    metric_meta: Dict[str, MetricMeta] = {}
    total_dirs = 0

    for run_dir in _iter_run_dirs(runs_root):
        total_dirs += 1
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue

        try:
            result = _json_load(result_path)
        except Exception:
            continue

        metrics = result.get("metrics") or []
        if not isinstance(metrics, list) or not metrics:
            continue

        inputs_path = run_dir / "inputs.json"
        inputs: Dict[str, Any] = {}
        if inputs_path.exists():
            try:
                inputs = _json_load(inputs_path)
            except Exception:
                inputs = {}

        generation = inputs.get("generation") if isinstance(inputs, dict) else {}
        if not isinstance(generation, dict):
            generation = {}

        prompt_input = generation.get("prompt_input")
        if not isinstance(prompt_input, dict):
            prompt_input = {}

        summary = result.get("summary")
        if not isinstance(summary, dict):
            summary = {}

        row: Dict[str, Any] = {
            "pair_id": run_dir.name,
            "run_id": run_dir.name,
            "label": label_default,
            "tier": "",
            "notes": "",
            "task_mode": _safe_str(inputs.get("task_mode")),
            "provider": _safe_str(generation.get("provider")),
            "model": _safe_str(generation.get("model")),
            "prompt_level": _safe_str(prompt_input.get("mode")),
            "part_id": "",
            "family": "",
            "reference_path": _safe_str(inputs.get("reference_path")),
            "generated_path": _safe_str(inputs.get("generated_path")),
            "reference_mesh_path": _safe_str(inputs.get("reference_mesh_path")),
            "generated_mesh_path": _safe_str(inputs.get("generated_mesh_path")),
            "fast_mode": _safe_str(inputs.get("fast_mode")),
            "sample_points": _safe_str(inputs.get("sample_points")),
            "voxel_pitch_mm": _safe_str(inputs.get("voxel_pitch_mm")),
            "grading_profile": _safe_str(inputs.get("grading_profile") or summary.get("grading_profile")),
            "pass_rate": _safe_str(summary.get("pass_rate")),
            "quality_score_0_100": _safe_str(summary.get("quality_score_0_100")),
            "overall_pass": _safe_str(summary.get("overall_pass")),
        }

        for m in metrics:
            if not isinstance(m, dict):
                continue
            name = _safe_str(m.get("name")).strip()
            if not name:
                continue

            if name not in metric_meta:
                metric_meta[name] = MetricMeta(
                    direction=_safe_str(m.get("direction")) or "unknown",
                    thresholds_seen=[],
                    count=0,
                )
                metric_order.append(name)

            metric_meta[name].add(
                direction=_safe_str(m.get("direction")) or "unknown",
                threshold=_safe_float(m.get("threshold")),
            )

            value = _safe_float(m.get("value"))
            row[name] = "" if value is None else value

        rows.append(row)

    return rows, metric_order, metric_meta, total_dirs


def write_csv(path: Path, rows: List[Dict[str, Any]], metric_order: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = BASE_COLUMNS + metric_order
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)


def write_metric_meta(
    path: Path,
    runs_root: Path,
    metric_order: List[str],
    metric_meta: Dict[str, MetricMeta],
    rows_written: int,
    run_dirs_seen: int,
) -> None:
    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root.resolve()),
        "run_dirs_seen": run_dirs_seen,
        "rows_written": rows_written,
        "metric_names": metric_order,
        "metrics": {},
    }
    for name in metric_order:
        meta = metric_meta[name]
        payload["metrics"][name] = {
            "direction": meta.direction,
            "threshold_default": meta.threshold_default(),
            "count_seen": meta.count,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build threshold-tuning labeling CSV from STEPScore .cad42_runs outputs."
    )
    parser.add_argument(
        "--runs-root",
        default=".cad42_runs",
        help="Path to STEPScore runs root directory (default: .cad42_runs).",
    )
    parser.add_argument(
        "--output-csv",
        default="threshold_tuning/pairs_for_labeling.csv",
        help="Output CSV path for labeling.",
    )
    parser.add_argument(
        "--output-metrics-meta",
        default="threshold_tuning/metrics_meta.json",
        help="Output JSON path for metric directions/default thresholds.",
    )
    parser.add_argument(
        "--default-label",
        default="review",
        help="Default label assigned to each row (recommended: review).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    rows, metric_order, metric_meta, total_dirs = build_rows(
        runs_root=runs_root,
        label_default=args.default_label.strip(),
    )

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_meta = Path(args.output_metrics_meta).expanduser().resolve()

    write_csv(output_csv, rows, metric_order)
    write_metric_meta(
        output_meta,
        runs_root=runs_root,
        metric_order=metric_order,
        metric_meta=metric_meta,
        rows_written=len(rows),
        run_dirs_seen=total_dirs,
    )

    print(f"runs_root={runs_root}")
    print(f"run_dirs_seen={total_dirs}")
    print(f"rows_written={len(rows)}")
    print(f"metrics_found={len(metric_order)}")
    print(f"wrote_csv={output_csv}")
    print(f"wrote_metrics_meta={output_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


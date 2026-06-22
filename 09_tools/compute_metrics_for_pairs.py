#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metric_engine import compare_models


def _ref_stl_from_step(step_path: Path) -> Path:
    # Mirror labeling_app.py behavior:
    # ref_stl = ref_step.parent.parent / "references_parametric_stl" / ref_step.with_suffix(".stl").name
    return step_path.parent.parent / "references_parametric_stl" / step_path.with_suffix(".stl").name


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    p = argparse.ArgumentParser(description="Compute STEPScore metrics for pairs CSV.")
    p.add_argument("--pairs-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--sample-points", type=int, default=10000)
    p.add_argument("--voxel-pitch-mm", type=float, default=2.0)
    p.add_argument("--fast-mode", choices=["true", "false"], default="true")
    p.add_argument("--grading-profile", default="full_44")
    p.add_argument("--alignment-method", default="pca_icp")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", help="Skip rows that already have metrics.")
    args = p.parse_args()

    pairs_csv = Path(args.pairs_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    rows = load_rows(pairs_csv)
    if not rows:
        print("no rows")
        return 1

    metric_cols = set()
    updated = 0
    processed = 0

    for row in rows:
        gen_stl = row.get("generated_mesh_path", "")
        ref_path = row.get("reference_path", "")
        if not gen_stl or not ref_path:
            continue

        # skip if metrics already present
        if args.skip_existing:
            existing = _safe_float(row.get("chamfer_distance_mm"))
            if existing is not None:
                continue

        ref_step = Path(ref_path)
        gen_stl_path = Path(gen_stl)
        if not ref_step.exists() or not gen_stl_path.exists():
            continue

        ref_stl = _ref_stl_from_step(ref_step)
        if not ref_stl.exists():
            # if missing, fall back to reference STEP (compare_models expects mesh, but try)
            ref_stl = ref_step

        print(f"[metrics] {row.get('part_id')} provider={row.get('provider')}")
        res = compare_models(
            reference_path=str(ref_stl),
            generated_path=str(gen_stl_path),
            sample_points=args.sample_points,
            voxel_pitch_mm=args.voxel_pitch_mm,
            fast_mode=(args.fast_mode == "true"),
            grading_profile=args.grading_profile,
            alignment_method=args.alignment_method,
        )

        if not res.get("ok", True):
            row["metrics_error"] = str(res.get("error", "unknown"))
            continue

        metrics = res.get("metrics", [])
        for m in metrics:
            name = m.get("name")
            val = m.get("value")
            if name:
                metric_cols.add(name)
                row[name] = val

        summary = res.get("summary", {})
        for key in ("pass_rate", "quality_score_0_100", "overall_pass"):
            if key in summary:
                row[key] = summary[key]

        updated += 1
        processed += 1
        if args.limit and processed >= args.limit:
            break

    # Build output fieldnames: existing + new metric cols
    fieldnames = list(rows[0].keys())
    for m in sorted(metric_cols):
        if m not in fieldnames:
            fieldnames.append(m)
    for k in ("pass_rate", "quality_score_0_100", "overall_pass", "metrics_error"):
        if k not in fieldnames:
            fieldnames.append(k)

    write_rows(output_csv, rows, fieldnames)
    print(f"rows={len(rows)} updated={updated} output={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

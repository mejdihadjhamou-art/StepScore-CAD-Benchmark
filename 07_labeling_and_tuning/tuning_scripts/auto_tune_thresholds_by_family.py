#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Tune thresholds per family by wrapping auto_tune_thresholds.py")
    p.add_argument("--pairs-csv", required=True, help="Labeled pairs CSV (must include 'family' column)")
    p.add_argument("--metrics-meta-json", required=True, help="metrics_meta.json path")
    p.add_argument("--output-dir", default="threshold_tuning/output_by_family")
    p.add_argument("--min-family-rows", type=int, default=30)
    p.add_argument("--objective", default="f1")
    p.add_argument("--label-col", default="label")
    p.add_argument("--pos-labels", default="usable,pass,yes,1,ok,good")
    p.add_argument("--neg-labels", default="unusable,fail,no,0,bad")
    p.add_argument("--nan-policy", default="skip")
    p.add_argument("--fp-cost", type=float, default=2.0)
    p.add_argument("--fn-cost", type=float, default=1.0)
    args = p.parse_args()

    pairs_csv = Path(args.pairs_csv).expanduser().resolve()
    metrics_meta = Path(args.metrics_meta_json).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = read_rows(pairs_csv)
    if not rows:
        print("No rows found in pairs CSV", file=sys.stderr)
        return 2

    if "family" not in rows[0].keys():
        print("pairs CSV must include a 'family' column", file=sys.stderr)
        return 2

    by_family: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        fam = (r.get("family") or "unknown").strip().lower() or "unknown"
        by_family[fam].append(r)

    combined: Dict[str, Dict[str, float]] = {}
    skipped: Dict[str, str] = {}

    auto_script = Path(__file__).parent / "auto_tune_thresholds.py"

    for fam, fam_rows in sorted(by_family.items()):
        if len(fam_rows) < args.min_family_rows:
            skipped[fam] = f"too_few_rows({len(fam_rows)}<{args.min_family_rows})"
            continue

        fam_dir = out_root / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        fam_pairs = fam_dir / "pairs.csv"
        write_rows(fam_pairs, fam_rows)

        cmd = [
            sys.executable,
            str(auto_script),
            "--pairs-csv",
            str(fam_pairs),
            "--metrics-meta-json",
            str(metrics_meta),
            "--output-dir",
            str(fam_dir),
            "--objective",
            args.objective,
            "--label-column",
            args.label_col,
            "--positive-labels",
            args.pos_labels,
            "--negative-labels",
            args.neg_labels,
            "--nan-policy",
            args.nan_policy,
            "--fp-cost",
            str(args.fp_cost),
            "--fn-cost",
            str(args.fn_cost),
        ]
        subprocess.run(cmd, check=True)

        overrides_path = fam_dir / "threshold_overrides.json"
        if overrides_path.exists():
            combined[fam] = json.loads(overrides_path.read_text(encoding="utf-8"))
        else:
            skipped[fam] = "no_overrides_generated"

    combined_path = out_root / "thresholds_by_family.json"
    combined_path.write_text(json.dumps({"by_family": combined, "skipped": skipped}, indent=2), encoding="utf-8")
    print(f"Wrote {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

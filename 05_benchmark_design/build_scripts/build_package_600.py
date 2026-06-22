from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class PromptRow:
    part_id: str
    family: str
    step_path: str
    prompt_l2: str
    prompt_l3: str


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a balanced 600-pair STEPScore package from existing golden prompt CSVs."
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        type=Path,
        default=[
            Path("/Users/mejdi/Desktop/reference_step files/golden_prompts/golden_prompts_l2_l3.csv"),
            Path("/Users/mejdi/Desktop/reference_step files/golden_prompts_advanced/golden_prompts_l2_l3.csv"),
        ],
        help="Input golden prompt CSV(s), each with part_id,family,step_path,prompt_l2,prompt_l3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "packages" / "package_600",
        help="Output directory for merged prompts/manifests/splits.",
    )
    parser.add_argument(
        "--selected-step-dir",
        type=Path,
        default=Path("/Users/mejdi/Desktop/reference_step files/package_600/reference_steps_150"),
        help="Destination folder for selected STEP references.",
    )
    parser.add_argument(
        "--target-parts",
        type=int,
        default=150,
        help="Number of reference parts to select (150 gives 600 rows with 2 levels x 2 models x 1 replicate).",
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        type=str,
        default=["openai:gpt-5.2", "anthropic:claude-opus-4-1-20250805"],
        help="Model spec provider:model_id. Can be provided multiple times.",
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--prompt-level", type=str, default="both", choices=["L2", "L3", "both"])
    parser.add_argument("--grading-profile", type=str, default="full_44")
    parser.add_argument("--sample-points", type=int, default=10000)
    parser.add_argument("--voxel-pitch-mm", type=float, default=2.0)
    parser.add_argument("--fast-mode", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_prompt_rows(paths: Sequence[Path]) -> List[PromptRow]:
    rows: List[PromptRow] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"input CSV not found: {p}")
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                part_id = (raw.get("part_id") or "").strip()
                family = (raw.get("family") or "").strip()
                step_path = (raw.get("step_path") or "").strip()
                prompt_l2 = (raw.get("prompt_l2") or "").strip()
                prompt_l3 = (raw.get("prompt_l3") or "").strip()
                if not part_id or not step_path:
                    continue
                if not prompt_l2 and not prompt_l3:
                    continue
                rows.append(
                    PromptRow(
                        part_id=part_id,
                        family=family,
                        step_path=step_path,
                        prompt_l2=prompt_l2,
                        prompt_l3=prompt_l3,
                    )
                )
    return rows


def _dedupe_by_part(rows: Iterable[PromptRow]) -> List[PromptRow]:
    out: Dict[str, PromptRow] = {}
    for r in rows:
        if r.part_id not in out:
            out[r.part_id] = r
    return sorted(out.values(), key=lambda x: x.part_id)


def _filter_existing_step(rows: Iterable[PromptRow]) -> List[PromptRow]:
    out = []
    for r in rows:
        if Path(r.step_path).exists():
            out.append(r)
    return out


def _balanced_select(rows: Sequence[PromptRow], target_parts: int, seed: int) -> List[PromptRow]:
    by_family: Dict[str, List[PromptRow]] = defaultdict(list)
    for r in rows:
        by_family[r.family].append(r)
    rng = random.Random(seed)
    families = sorted(by_family.keys())
    for fam in families:
        by_family[fam].sort(key=lambda x: x.part_id)
        rng.shuffle(by_family[fam])

    selected: List[PromptRow] = []
    while len(selected) < target_parts:
        added = False
        for fam in families:
            q = by_family[fam]
            if q:
                selected.append(q.pop())
                added = True
                if len(selected) >= target_parts:
                    break
        if not added:
            break
    return sorted(selected, key=lambda x: x.part_id)


def _write_prompt_csv(path: Path, rows: Sequence[PromptRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["part_id", "family", "step_path", "prompt_l2", "prompt_l3"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "part_id": r.part_id,
                    "family": r.family,
                    "step_path": r.step_path,
                    "prompt_l2": r.prompt_l2,
                    "prompt_l3": r.prompt_l3,
                }
            )


def _build_manifest(args: argparse.Namespace, selected_csv: Path, out_manifest: Path) -> None:
    script = Path(__file__).resolve().parent / "build_harness_manifest.py"
    cmd: List[str] = [
        sys.executable,
        str(script),
        "--golden-prompts-csv",
        str(selected_csv),
        "--prompt-level",
        args.prompt_level,
        "--replicates",
        str(args.replicates),
        "--grading-profile",
        args.grading_profile,
        "--sample-points",
        str(args.sample_points),
        "--voxel-pitch-mm",
        str(args.voxel_pitch_mm),
        "--fast-mode",
        args.fast_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output",
        str(out_manifest),
    ]
    for spec in dict.fromkeys(args.model_spec):
        cmd.extend(["--model-spec", str(spec)])

    subprocess.run(cmd, check=True)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dedupe_manifest_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        key = r.get("job_id", "")
        if key and key not in out:
            out[key] = r
    return list(out.values())


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _split_part_ids(part_ids: List[str], seed: int) -> Tuple[set, set, set]:
    ids = sorted(part_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_cal = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    n_blind = n - n_cal - n_val
    cal = set(ids[:n_cal])
    val = set(ids[n_cal : n_cal + n_val])
    blind = set(ids[n_cal + n_val : n_cal + n_val + n_blind])
    return cal, val, blind


def _copy_selected_steps(rows: Sequence[PromptRow], dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for r in rows:
        src = Path(r.step_path)
        if not src.exists():
            continue
        shutil.copy2(src, dest_dir / src.name)
        count += 1
    return count


def _family_counts(rows: Sequence[PromptRow]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for r in rows:
        out[r.family] += 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def main() -> None:
    args = parse_args()
    if args.target_parts < 1:
        raise ValueError("--target-parts must be >= 1")
    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _read_prompt_rows(args.input_csv)
    deduped_rows = _dedupe_by_part(raw_rows)
    existing_rows = _filter_existing_step(deduped_rows)
    if len(existing_rows) < args.target_parts:
        raise RuntimeError(
            f"Not enough references with existing STEP files: need {args.target_parts}, found {len(existing_rows)}."
        )

    selected_rows = _balanced_select(existing_rows, args.target_parts, args.seed)
    combined_csv = out_dir / "golden_prompts_combined_all.csv"
    selected_csv = out_dir / "golden_prompts_selected_150.csv"
    _write_prompt_csv(combined_csv, existing_rows)
    _write_prompt_csv(selected_csv, selected_rows)

    manifest_600 = out_dir / "harness_manifest_600.csv"
    _build_manifest(args, selected_csv, manifest_600)
    all_manifest_rows = _dedupe_manifest_rows(_read_csv(manifest_600))
    _write_csv(manifest_600, all_manifest_rows)

    selected_part_ids = sorted({r.part_id for r in selected_rows})
    cal_ids, val_ids, blind_ids = _split_part_ids(selected_part_ids, args.seed)

    cal_rows = [r for r in all_manifest_rows if r["part_id"] in cal_ids]
    val_rows = [r for r in all_manifest_rows if r["part_id"] in val_ids]
    blind_rows = [r for r in all_manifest_rows if r["part_id"] in blind_ids]
    _write_csv(out_dir / "harness_manifest_600_calibration.csv", cal_rows)
    _write_csv(out_dir / "harness_manifest_600_validation.csv", val_rows)
    _write_csv(out_dir / "harness_manifest_600_blind.csv", blind_rows)

    copied_count = _copy_selected_steps(selected_rows, args.selected_step_dir)

    summary = {
        "target_pairs_formula": "target_parts * prompt_levels * model_specs * replicates",
        "target_parts": args.target_parts,
        "prompt_levels": (2 if args.prompt_level == "both" else 1),
        "model_specs": list(args.model_spec),
        "replicates": args.replicates,
        "total_pairs": len(all_manifest_rows),
        "selected_parts": len(selected_part_ids),
        "selected_families": _family_counts(selected_rows),
        "split_counts_rows": {
            "calibration": len(cal_rows),
            "validation": len(val_rows),
            "blind": len(blind_rows),
        },
        "split_counts_parts": {
            "calibration": len(cal_ids),
            "validation": len(val_ids),
            "blind": len(blind_ids),
        },
        "paths": {
            "combined_prompts_csv": str(combined_csv),
            "selected_prompts_csv": str(selected_csv),
            "manifest_600": str(manifest_600),
            "manifest_calibration": str(out_dir / "harness_manifest_600_calibration.csv"),
            "manifest_validation": str(out_dir / "harness_manifest_600_validation.csv"),
            "manifest_blind": str(out_dir / "harness_manifest_600_blind.csv"),
            "selected_step_dir": str(args.selected_step_dir),
        },
        "selected_steps_copied": copied_count,
    }
    (out_dir / "package_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

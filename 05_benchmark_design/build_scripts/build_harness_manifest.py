from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _sanitize(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._-") or "item"


def _parse_model_specs(values: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for raw in values:
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Invalid --model-spec '{raw}'. Use provider:model_id")
        provider, model = raw.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported provider '{provider}' in --model-spec '{raw}'.")
        if not model:
            raise ValueError(f"Missing model in --model-spec '{raw}'.")
        out.append((provider, model))
    if not out:
        raise ValueError("No model specs provided.")
    return out


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build STEPScore harness manifest from golden L2/L3 prompts.")
    parser.add_argument(
        "--golden-prompts-csv",
        type=Path,
        default=Path("/Users/mejdi/Desktop/reference_step files/golden_prompts/golden_prompts_l2_l3.csv"),
        help="CSV containing part_id,family,step_path,prompt_l2,prompt_l3.",
    )
    parser.add_argument(
        "--prompt-level",
        type=str,
        default="both",
        choices=["L2", "L3", "both"],
        help="Which prompt levels to include.",
    )
    parser.add_argument(
        "--model-spec",
        type=str,
        action="append",
        default=["openai:gpt-5.2"],
        help="Model spec provider:model. Can be passed multiple times.",
    )
    parser.add_argument("--replicates", type=int, default=1, help="Replicates per (part,level,model).")
    parser.add_argument("--grading-profile", type=str, default="full_44")
    parser.add_argument("--sample-points", type=int, default=10000)
    parser.add_argument("--voxel-pitch-mm", type=float, default=2.0)
    parser.add_argument("--fast-mode", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--status", type=str, default="pending")
    parser.add_argument("--enabled", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "harness_manifest.generated.csv",
        help="Output manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.golden_prompts_csv.exists():
        raise FileNotFoundError(f"golden prompts CSV not found: {args.golden_prompts_csv}")
    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")

    models = _parse_model_specs(args.model_spec)
    prompt_levels = ["L2", "L3"] if args.prompt_level == "both" else [args.prompt_level]

    with args.golden_prompts_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "part_id",
        "family",
        "prompt_level",
        "provider",
        "model",
        "generation_mode",
        "reference_path",
        "base_step_path",
        "prompt_text",
        "grading_profile",
        "replicate",
        "sample_points",
        "voxel_pitch_mm",
        "fast_mode",
        "timeout_seconds",
        "threshold_overrides_json",
        "status",
        "enabled",
        "notes",
    ]

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        part_id = (row.get("part_id") or "").strip()
        family = (row.get("family") or "").strip()
        step_path = (row.get("step_path") or "").strip()
        if not part_id or not step_path:
            continue
        prompts = {
            "L2": (row.get("prompt_l2") or "").strip(),
            "L3": (row.get("prompt_l3") or "").strip(),
        }
        for level in prompt_levels:
            prompt_text = prompts[level]
            if not prompt_text:
                continue
            for provider, model in models:
                for rep in range(1, args.replicates + 1):
                    job_id = (
                        f"{_sanitize(part_id)}__{level}__{_sanitize(provider)}__"
                        f"{_sanitize(model)}__r{rep}"
                    )
                    out_rows.append(
                        {
                            "job_id": job_id,
                            "part_id": part_id,
                            "family": family,
                            "prompt_level": level,
                            "provider": provider,
                            "model": model,
                            "generation_mode": "generate",
                            "reference_path": step_path,
                            "base_step_path": "",
                            "prompt_text": prompt_text,
                            "grading_profile": args.grading_profile,
                            "replicate": str(rep),
                            "sample_points": str(args.sample_points),
                            "voxel_pitch_mm": str(args.voxel_pitch_mm),
                            "fast_mode": args.fast_mode,
                            "timeout_seconds": str(args.timeout_seconds),
                            "threshold_overrides_json": "{}",
                            "status": args.status,
                            "enabled": args.enabled,
                            "notes": "auto_generated_from_golden_prompts",
                        }
                    )

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"rows={len(out_rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

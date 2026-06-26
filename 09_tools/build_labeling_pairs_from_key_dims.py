#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List
import subprocess

import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation_pipeline import GenerationError, generate_and_export
from step_utils import convert_step_to_stl


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _should_retry_error(msg: str) -> bool:
    m = msg.lower()
    return "overloaded" in m or "529" in m or "rate limit" in m or "temporarily unavailable" in m


def _generate_with_retries(
    *,
    prompt: str,
    provider: str,
    model: str,
    run_dir: Path,
    timeout_seconds: int,
    max_retries: int = 6,
    base_sleep: float = 2.0,
):
    attempt = 0
    while True:
        try:
            return generate_and_export(
                prompt=prompt,
                provider=provider,
                model=model,
                run_dir=str(run_dir),
                api_key=None,
                generation_mode="generate",
                base_step_path=None,
                timeout_seconds=timeout_seconds,
            )
        except GenerationError as exc:
            msg = str(exc)
        except subprocess.TimeoutExpired as exc:
            msg = f"cadquery_timeout:{exc}"
        except Exception as exc:
            msg = str(exc)
            attempt += 1
            if attempt > max_retries or not _should_retry_error(msg):
                raise GenerationError(msg) from exc
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"[retry] {provider} {model} attempt={attempt} sleep={sleep_s:.1f}s error={msg}")
            time.sleep(sleep_s)


def _col_to_idx(col: str) -> int:
    idx = 0
    for c in col:
        if not c.isalpha():
            break
        idx = idx * 26 + (ord(c.upper()) - ord("A") + 1)
    return idx


def _parse_shared_strings(z: zipfile.ZipFile) -> List[str]:
    try:
        data = z.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for si in root.findall("a:si", ns):
        texts = [t.text or "" for t in si.findall(".//a:t", ns)]
        strings.append("".join(texts))
    return strings


def _sheet_rows(xlsx_path: Path, sheet_name: str = "sheet1.xml") -> List[List[str]]:
    with zipfile.ZipFile(xlsx_path) as z:
        shared = _parse_shared_strings(z)
        data = z.read(f"xl/worksheets/{sheet_name}")
    root = ET.fromstring(data)
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    rows = []
    for row in root.findall("a:sheetData/a:row", ns):
        cells = {}
        max_col = 0
        for c in row.findall("a:c", ns):
            r = c.attrib.get("r", "")
            m = re.match(r"([A-Z]+)([0-9]+)", r)
            if not m:
                continue
            col = m.group(1)
            ci = _col_to_idx(col)
            max_col = max(max_col, ci)
            t = c.attrib.get("t")
            v = c.find("a:v", ns)
            if v is None or v.text is None:
                val = ""
            else:
                raw = v.text
                if t == "s":
                    try:
                        val = shared[int(raw)]
                    except Exception:
                        val = raw
                else:
                    val = raw
            cells[ci] = val
        if max_col == 0:
            rows.append([])
            continue
        row_vals = [""] * max_col
        for ci, val in cells.items():
            row_vals[ci - 1] = val
        rows.append(row_vals)
    return rows


def parse_key_dims_xlsx(xlsx_path: Path) -> List[Dict[str, str]]:
    rows = _sheet_rows(xlsx_path)
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not r or not (r[0] and str(r[0]).strip()):
            continue
        row = {}
        for i, h in enumerate(header):
            if not h:
                continue
            row[h] = r[i] if i < len(r) else ""
        out.append(row)
    return out


def build_prompt(row: Dict[str, str]) -> str:
    part_id = row.get("part_id", "").strip()
    family = row.get("family", "").strip()
    key_dims = row.get("key_dimensions_mm", "").strip()
    features = row.get("design_features", "").strip()

    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        f"{family} ({part_id})",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at center of bottom face unless specified otherwise.",
        "- +Z is upward.",
        "",
        "Key dimensions:",
        f"- {key_dims}" if key_dims else "- As specified in design features.",
        "",
        "Design features:",
        f"- {features}" if features else "- Use the reference geometry description.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions and feature counts exactly.",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Build labeling pairs from key-dimensions XLSX.")
    p.add_argument("--xlsx", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompt-manifest", required=True)
    p.add_argument("--pairs-csv", required=True)
    p.add_argument("--openai-model", default="gpt-5.2")
    p.add_argument("--anthropic-model", default="claude-opus-4-1-20250805")
    p.add_argument("--providers", default="openai,anthropic")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--cadquery-timeout", type=int, default=180)
    p.add_argument("--skip-part", action="append", default=[], help="Repeatable part_id to skip")
    p.add_argument("--skip-family", action="append", default=[], help="Repeatable family to skip")
    args = p.parse_args()

    xlsx = Path(args.xlsx).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    prompt_manifest = Path(args.prompt_manifest).expanduser().resolve()
    pairs_csv = Path(args.pairs_csv).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _load_env_file(ROOT / ".env")

    rows = parse_key_dims_xlsx(xlsx)

    ref_root = Path("./benchmark_v1/references_isidor_all")
    ref_root.mkdir(parents=True, exist_ok=True)
    ref_stl_root = ref_root / "references_parametric_stl"
    ref_stl_root.mkdir(parents=True, exist_ok=True)

    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    model_map = {
        "openai": args.openai_model,
        "anthropic": args.anthropic_model,
    }
    skip_parts = {s.strip().lower() for s in args.skip_part if s.strip()}
    skip_families = {s.strip().lower() for s in args.skip_family if s.strip()}

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    prompt_rows = []
    pair_rows = []
    generation_errors = []

    processed = 0
    for row in rows:
        part_id = str(row.get("part_id", "")).strip()
        family = str(row.get("family", "")).strip() or "unknown"
        src_step = str(row.get("step_path", "")).strip()
        if not part_id or not src_step:
            continue
        if part_id.lower() in skip_parts or family.lower() in skip_families:
            print(f"[skip] {part_id} family={family} (requested)")
            continue

        src_path = Path(src_step).expanduser()
        if not src_path.exists():
            generation_errors.append({"part_id": part_id, "error": "missing_source_step"})
            continue

        family_dir = ref_root / family
        family_dir.mkdir(parents=True, exist_ok=True)
        dst_step = family_dir / src_path.name
        if not dst_step.exists():
            dst_step.write_bytes(src_path.read_bytes())

        # create reference STL
        dst_stl = ref_stl_root / dst_step.with_suffix(".stl").name
        if not dst_stl.exists():
            try:
                convert_step_to_stl(str(dst_step), str(dst_stl))
            except Exception as exc:
                generation_errors.append({"part_id": part_id, "error": f"ref_stl:{exc}"})
                continue

        prompt = build_prompt(row)
        prompt_rows.append(
            {
                "part_id": part_id,
                "family": family,
                "reference_path": str(dst_step),
                "prompt_text": prompt,
            }
        )

        if args.skip_generation:
            processed += 1
            if args.limit and processed >= args.limit:
                break
            continue

        for provider in providers:
            if provider == "openai" and not has_openai:
                generation_errors.append({"part_id": part_id, "provider": provider, "error": "missing_openai_key"})
                continue
            if provider == "anthropic" and not has_anthropic:
                generation_errors.append({"part_id": part_id, "provider": provider, "error": "missing_anthropic_key"})
                continue
            model = model_map.get(provider)
            if not model:
                continue
            run_dir = out_dir / provider / part_id
            run_dir.mkdir(parents=True, exist_ok=True)
            gen_step_path = run_dir / "generated.step"
            gen_stl_path = run_dir / "generated.stl"
            if gen_step_path.exists() and gen_stl_path.exists():
                print(f"[skip] {part_id} provider={provider} (already generated)")
                pair_rows.append(
                    {
                        "pair_id": f"{part_id}__{provider}",
                        "part_id": part_id,
                        "family": family,
                        "prompt_level": "L3",
                        "provider": provider,
                        "model": model,
                        "reference_path": str(dst_step),
                        "generated_path": str(gen_step_path),
                        "generated_mesh_path": str(gen_stl_path),
                        "prompt_text": prompt,
                        "label": "",
                    }
                )
                continue
            try:
                print(f"[gen] {part_id} provider={provider} model={model}")
                gen = _generate_with_retries(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    run_dir=run_dir,
                    timeout_seconds=args.cadquery_timeout,
                )
            except GenerationError as exc:
                generation_errors.append({"part_id": part_id, "provider": provider, "error": str(exc)})
                continue

            gen_step = gen.get("generated_step_path", "")
            gen_stl = gen.get("generated_stl_path", "")
            pair_rows.append(
                {
                    "pair_id": f"{part_id}__{provider}",
                    "part_id": part_id,
                    "family": family,
                    "prompt_level": "L3",
                    "provider": provider,
                    "model": model,
                    "reference_path": str(dst_step),
                    "generated_path": gen_step,
                    "generated_mesh_path": gen_stl,
                    "prompt_text": prompt,
                    "label": "",
                }
            )

        processed += 1
        if args.limit and processed >= args.limit:
            break

    prompt_manifest.parent.mkdir(parents=True, exist_ok=True)
    with prompt_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["part_id", "family", "reference_path", "prompt_text"])
        w.writeheader()
        for r in prompt_rows:
            w.writerow(r)

    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    with pairs_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pair_id",
            "part_id",
            "family",
            "prompt_level",
            "provider",
            "model",
            "reference_path",
            "generated_path",
            "generated_mesh_path",
            "prompt_text",
            "label",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "generation_errors.json").write_text(
        json.dumps(generation_errors, indent=2), encoding="utf-8"
    )

    print(f"prompt_manifest={prompt_manifest}")
    print(f"pairs_csv={pairs_csv}")
    print(f"prompts={len(prompt_rows)}")
    print(f"pairs={len(pair_rows)}")
    print(f"errors={len(generation_errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

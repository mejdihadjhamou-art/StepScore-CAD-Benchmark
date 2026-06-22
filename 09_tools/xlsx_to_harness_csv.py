#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_XLSX_CANDIDATES = [
    Path("/Users/mejdi/Documents/New project/FINAL.xlsx"),
    Path("/Users/mejdi/Desktop/FINAL.xlsx"),
    Path("/Users/mejdi/Desktop/ISIDOR/FINAL.xlsx"),
]

DEFAULT_OUTPUT_CSV = Path(
    "/Users/mejdi/Documents/New project/cad42_platform/benchmark_v1/packages/final_73_manifest.csv"
)


def sanitize_token(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("._-") or "item"


def parse_model_specs(raw_specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for spec in raw_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --model-spec '{spec}'. Use provider:model")
        provider, model = spec.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported provider '{provider}' in '{spec}'")
        if not model:
            raise ValueError(f"Missing model in '{spec}'")
        out.append((provider, model))
    if not out:
        raise ValueError("No model specs provided")
    # dedupe while preserving order
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _cell_ref_to_col_idx(cell_ref: str) -> int:
    # e.g. "A1" -> 0, "C10" -> 2, "AA2" -> 26
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    val = 0
    for ch in letters:
        val = val * 26 + (ord(ch) - ord("A") + 1)
    return max(val - 1, 0)


def read_xlsx_first_sheet_rows(xlsx_path: Path) -> list[dict[str, str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(xlsx_path, "r") as zf:
        # Shared strings (optional)
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(".//m:si", ns):
                parts = []
                for t in si.findall(".//m:t", ns):
                    parts.append(t.text or "")
                shared.append("".join(parts))

        # First worksheet
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in zf.namelist():
            raise FileNotFoundError("Could not find xl/worksheets/sheet1.xml in XLSX.")
        ws_root = ET.fromstring(zf.read(sheet_path))

    table: list[list[str]] = []
    for row in ws_root.findall(".//m:sheetData/m:row", ns):
        values: dict[int, str] = {}
        max_col = -1
        for c in row.findall("m:c", ns):
            r_attr = c.attrib.get("r", "")
            col_idx = _cell_ref_to_col_idx(r_attr) if r_attr else 0
            max_col = max(max_col, col_idx)

            c_type = c.attrib.get("t", "")
            v = c.find("m:v", ns)
            is_elem = c.find("m:is", ns)

            text_val = ""
            if c_type == "s" and v is not None and v.text is not None:
                try:
                    text_val = shared[int(v.text)]
                except Exception:
                    text_val = ""
            elif c_type == "inlineStr" and is_elem is not None:
                t_elem = is_elem.find("m:t", ns)
                text_val = (t_elem.text if t_elem is not None else "") or ""
            elif v is not None and v.text is not None:
                text_val = v.text
            values[col_idx] = str(text_val).strip()

        if max_col < 0:
            continue
        row_list = [values.get(i, "") for i in range(max_col + 1)]
        table.append(row_list)

    if not table:
        return []

    headers = [h.strip() for h in table[0]]
    rows: list[dict[str, str]] = []
    for raw in table[1:]:
        rec = {}
        for i, h in enumerate(headers):
            if not h:
                continue
            rec[h] = raw[i].strip() if i < len(raw) else ""
        if any((v or "").strip() for v in rec.values()):
            rows.append(rec)
    return rows


def resolve_default_xlsx() -> Path:
    for p in DEFAULT_XLSX_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_XLSX_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FINAL.xlsx (part_id,family,step_path,prompt_l2,prompt_l3) to harness_runner CSV."
    )
    parser.add_argument(
        "--xlsx",
        default=str(resolve_default_xlsx()),
        help="Input .xlsx path. Defaults to FINAL.xlsx in common locations.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output harness CSV path.",
    )
    parser.add_argument("--prompt-level", choices=["L2", "L3", "both"], default="both")
    parser.add_argument(
        "--model-spec",
        action="append",
        default=["openai:gpt-5.2", "anthropic:claude-opus-4-6"],
        help="Repeatable: provider:model (e.g. openai:gpt-5.2)",
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--grading-profile", default="full_44")
    parser.add_argument("--sample-points", type=int, default=10000)
    parser.add_argument("--voxel-pitch-mm", type=float, default=2.0)
    parser.add_argument("--fast-mode", choices=["true", "false"], default="true")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--status", default="pending")
    parser.add_argument("--enabled", choices=["true", "false"], default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    out_path = Path(args.output_csv).expanduser().resolve()
    models = parse_model_specs(args.model_spec)
    levels = ["L2", "L3"] if args.prompt_level == "both" else [args.prompt_level]

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Input XLSX not found: {xlsx_path}\n"
            f"Tried defaults: {', '.join(str(p) for p in DEFAULT_XLSX_CANDIDATES)}"
        )

    required = ["part_id", "family", "step_path", "prompt_l2", "prompt_l3"]
    source_rows = read_xlsx_first_sheet_rows(xlsx_path)
    if not source_rows:
        raise ValueError(f"No rows found in first worksheet of {xlsx_path}")

    header_set = set(source_rows[0].keys())
    missing = [c for c in required if c not in header_set]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for r in source_rows:
        part_id = str(r.get("part_id", "")).strip()
        family = str(r.get("family", "")).strip()
        step_path = str(r.get("step_path", "")).strip()
        if not part_id or not step_path:
            continue

        prompts = {
            "L2": str(r.get("prompt_l2", "")).strip(),
            "L3": str(r.get("prompt_l3", "")).strip(),
        }

        for level in levels:
            prompt_text = prompts[level]
            if not prompt_text:
                continue

            for provider, model in models:
                for rep in range(1, args.replicates + 1):
                    job_id = (
                        f"{sanitize_token(part_id)}__{level}__"
                        f"{sanitize_token(provider)}__{sanitize_token(model)}__r{rep}"
                    )
                    rows.append(
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
                            "replicate": rep,
                            "sample_points": args.sample_points,
                            "voxel_pitch_mm": args.voxel_pitch_mm,
                            "fast_mode": args.fast_mode,
                            "timeout_seconds": args.timeout_seconds,
                            "threshold_overrides_json": "{}",
                            "status": args.status,
                            "enabled": args.enabled,
                            "notes": "from_final_xlsx_converter",
                        }
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"input_xlsx={xlsx_path}")
    print(f"output_csv={out_path}")
    print(f"rows_written={len(rows)}")
    print(f"model_specs={models}")
    print(f"prompt_levels={levels}")


if __name__ == "__main__":
    main()

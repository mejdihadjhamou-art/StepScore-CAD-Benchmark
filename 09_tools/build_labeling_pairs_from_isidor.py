#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation_pipeline import GenerationError, generate_and_export
from step_utils import convert_step_to_stl


def _num(s: str) -> float:
    return float(str(s).strip())


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
        # concatenate all <t> nodes in case of rich text
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


def parse_xlsx_sections(xlsx_path: Path) -> List[Dict[str, List[List[str]]]]:
    rows = _sheet_rows(xlsx_path)
    sections = []
    i = 0
    while i < len(rows):
        row = rows[i]
        first = row[0].strip() if row else ""
        if first and not re.match(r"^\d+(\.\d+)?$", first):
            name = first
            header_idx = i + 1
            data_idx = i + 2
            data_rows = []
            j = data_idx
            while j < len(rows):
                r = rows[j]
                if not r or not r[0].strip():
                    break
                if r[0].strip() and not re.match(r"^\d+(\.\d+)?$", r[0].strip()):
                    break
                data_rows.append(r)
                j += 1
            sections.append({"name": name, "rows": data_rows})
            i = j
            continue
        i += 1
    return sections


def _part_family_from_id(pid: str) -> str:
    for prefix in (
        "box_hole",
        "stepped_shaft",
        "flange",
        "ribbed_bracket",
        "shell_enclosure",
        "manifold_block",
        "counterbore_countersink_plate",
        "threaded_parts",
        "off_axis_features",
        "gear",
    ):
        if pid.startswith(prefix):
            return prefix
    return "unknown"


def _make_box_hole(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, X, Y, Z, hole_type, depth, dia, cbore, csink = (vals + [""] * 9)[:9]
    if not (X and Y and Z and hole_type and dia):
        return None, None
    X, Y, Z, dia = map(_num, [X, Y, Z, dia])
    hole_type = str(hole_type).strip().lower()
    depth_val = None
    if str(depth).strip():
        depth_val = _num(depth)
    part_id = f"box_hole_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Box with centered hole",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at center of bottom face.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Block: X={X}, Y={Y}, Z={Z}, bottom at z=0.",
        "",
        "Subtractive features:",
        f"- Centered hole along +Z: diameter={dia}, type={hole_type}."
        + (f", depth={depth_val}." if depth_val else ", depth=through."),
    ]
    if str(cbore).strip():
        lines.append(f"- Counterbore: diameter={_num(cbore)}.")
    if str(csink).strip():
        lines.append(f"- Countersink: diameter={_num(csink)}.")
    lines += [
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_stepped_shaft(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, L, d1, d2, d3, l1, l2, l3, kw, kd = (vals + [""] * 10)[:10]
    if not (L and d1 and d2 and d3 and l1 and l2 and l3 and kw and kd):
        return None, None
    L, d1, d2, d3, l1, l2, l3, kw, kd = map(_num, [L, d1, d2, d3, l1, l2, l3, kw, kd])
    part_id = f"stepped_shaft_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Stepped shaft with keyway",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at shaft center, Z along shaft axis.",
        "",
        "Base geometry:",
        f"- Overall length L={L}.",
        f"- Steps: d1={d1} for l1={l1}, d2={d2} for l2={l2}, d3={d3} for l3={l3}.",
        "",
        "Subtractive features:",
        f"- Longitudinal keyway along +Z: width={kw}, depth={kd}.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_flange(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, D, t, dh, Lh, db, pcd, n, hd = (vals + [""] * 9)[:9]
    if not (D and t and dh and Lh and db and pcd and n and hd):
        return None, None
    D, t, dh, Lh, db, pcd, hd = map(_num, [D, t, dh, Lh, db, pcd, hd])
    n = int(float(n))
    part_id = f"flange_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Flange with hub and bolt circle",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at center of bottom face.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Flange OD={D}, thickness={t}.",
        f"- Hub OD={dh}, hub length={Lh}.",
        f"- Bore diameter={db}.",
        "",
        "Subtractive features:",
        f"- Bolt circle: PCD={pcd}, count={n}, hole diameter={hd}.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_ribbed_bracket(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, L, W, T, H, tr, holes = (vals + [""] * 7)[:7]
    if not (L and W and T and H and tr and holes):
        return None, None
    L, W, T, H, tr, holes = map(_num, [L, W, T, H, tr, holes])
    part_id = f"ribbed_bracket_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Ribbed bracket",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at center of bottom face.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Base plate: L={L}, W={W}, T={T}.",
        f"- Vertical leg: height={H}, thickness={T}.",
        f"- Ribs: count=2, thickness={tr}.",
        "",
        "Subtractive features:",
        f"- Through hole: diameter={holes}.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_shell(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, L, W, H, t, st_n, st_d, st_h, v_n, v_l, v_w = (vals + [""] * 11)[:11]
    if not (L and W and H and t and st_n and st_d and st_h and v_n and v_l and v_w):
        return None, None
    L, W, H, t, st_d, st_h, v_l, v_w = map(_num, [L, W, H, t, st_d, st_h, v_l, v_w])
    st_n = int(float(st_n))
    v_n = int(float(v_n))
    part_id = f"shell_enclosure_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Open-top shell enclosure",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at center of bottom face.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Outer: L={L}, W={W}, H={H}.",
        f"- Wall thickness={t}, open top.",
        f"- Standoffs: count={st_n}, diameter={st_d}, height={st_h}.",
        f"- Vent slots: count={v_n}, size={v_l}x{v_w}.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_manifold(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, L, W, H, zd, xd, yd, bd, depth = (vals + [""] * 9)[:9]
    if not (L and W and H and zd and xd and yd and bd and depth):
        return None, None
    L, W, H, zd, xd, yd, bd, depth = map(_num, [L, W, H, zd, xd, yd, bd, depth])
    part_id = f"manifold_block_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Prismatic manifold block with orthogonal channels",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at center of bottom face.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Block: L={L}, W={W}, H={H}.",
        "",
        "Subtractive features:",
        f"- Through hole along Z at center: diameter={zd}.",
        f"- Through hole along X at center: diameter={xd}.",
        f"- Through hole along Y at center: diameter={yd}.",
        f"- Blind top port: diameter={bd}, depth={depth}.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_cbore_plate(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, L, W, T, hn, hd, mode, d, depth = (vals + [""] * 9)[:9]
    if not (L and W and T and hn and hd and mode and d and depth):
        return None, None
    L, W, T, hd, d, depth = map(_num, [L, W, T, hd, d, depth])
    hn = int(float(hn))
    mode = (mode or "").strip().lower()
    part_id = f"counterbore_countersink_plate_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Plate with hole pattern and counterbore/countersink",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at center of bottom face.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Plate: L={L}, W={W}, T={T}.",
        f"- Hole pattern: count={hn}, through diameter={hd}.",
        f"- Top feature: {mode} diameter={d}, depth={depth}.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_threaded(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, gender, sd, ss, zd, zs, hd, hs = (vals + [""] * 8)[:8]
    if not (pid and gender and sd and ss and zd and zs and hd and hs):
        return None, None
    sd, ss, zd, zs, hd, hs = map(_num, [sd, ss, zd, zs, hd, hs])
    g = str(gender).lower()
    part_id = f"threaded_parts_{'male' if 'male' in g else 'female'}_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        f"Threaded parts proxy ({gender})",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at base of part.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Shank: diameter={sd}, length={ss}.",
        f"- Thread zone: diameter={zd}, length={zs}.",
        f"- Head: diameter={hd}, length={hs}.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_offaxis(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, L, H, W, fd, ang = (vals + [""] * 6)[:6]
    if not (L and H and W and fd and ang):
        return None, None
    L, H, W = map(_num, [L, H, W])
    ang = str(ang).strip()
    part_id = f"off_axis_features_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Block with off-axis feature",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at center of bottom face.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Block: L={L}, W={W}, H={H}.",
        "",
        "Subtractive/additive feature:",
        f"- Feature: {fd}, oriented at angle={ang} relative to +Z.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _make_gear(vals: List[str]) -> Tuple[str | None, str | None]:
    pid, m, z, t, bore = (vals + [""] * 5)[:5]
    if not (m and z and t and bore):
        return None, None
    m, t, bore = map(_num, [m, t, bore])
    z = int(float(z))
    part_id = f"gear_{int(float(pid)):04d}"
    lines = [
        "Create a single connected mechanical CAD part.",
        "",
        "Part identity:",
        "Spur gear (proxy teeth)",
        "",
        "Units and coordinate frame:",
        "- Units: millimeters.",
        "- Coordinate system: right-handed XYZ.",
        "- Origin at gear center on bottom face.",
        "- +Z is upward.",
        "",
        "Base geometry:",
        f"- Module={m}, teeth={z}, thickness={t}, bore diameter={bore}.",
        f"- Pitch diameter d = m*z = {m*z}.",
        f"- Outer diameter approx d_out = m*(z+2) = {m*(z+2)}.",
        "",
        "Topology/output constraints:",
        "- Single connected solid body.",
        "- Preserve all stated dimensions exactly.",
    ]
    return part_id, "\n".join(lines)


def _build_ref_map(ref_dirs: List[Path]) -> Dict[str, Path]:
    ref_map: Dict[str, Path] = {}
    for d in ref_dirs:
        if not d.exists():
            continue
        for p in d.glob("*.step"):
            ref_map[p.name.lower()] = p
        for p in d.glob("*.stp"):
            ref_map[p.name.lower()] = p
    return ref_map


def _build_ref_map_recursive(ref_dirs: List[Path]) -> Dict[str, Path]:
    ref_map: Dict[str, Path] = {}
    for d in ref_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.step"):
            ref_map[p.name.lower()] = p
        for p in d.rglob("*.stp"):
            ref_map[p.name.lower()] = p
    return ref_map


def main() -> int:
    p = argparse.ArgumentParser(description="Generate labeled pairs from isidor_excel.xlsx")
    p.add_argument("--xlsx", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-dir", required=True, help="Base output directory for generations")
    p.add_argument("--openai-model", default="gpt-4.1")
    p.add_argument("--anthropic-model", default="claude-opus-4-6")
    p.add_argument("--providers", default="openai,anthropic")
    p.add_argument("--limit", type=int, default=0, help="Limit number of parts (0 = all)")
    args = p.parse_args()

    xlsx = Path(args.xlsx).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dirs = [
        Path("/Users/mejdi/Documents/New project/cad42_platform/benchmark_v1/references_basic"),
        Path("/Users/mejdi/Documents/New project/cad42_platform/benchmark_v1/references_parametric"),
        Path("/Users/mejdi/Documents/New project/cad42_platform/benchmark_v1/references_advanced"),
        Path("/Users/mejdi/Documents/New project/cad42_platform/benchmark_v1/references_advanced_smoke"),
    ]
    ref_map = _build_ref_map(ref_dirs)

    # Optional external reference sets (copied into workspace if used)
    external_dirs = [
        Path("/Users/mejdi/Desktop/reference_step files"),
        Path("/Users/mejdi/Desktop/ISIDOR/reference_step files"),
    ]
    external_map = _build_ref_map_recursive(external_dirs)
    workspace_ref_dir = Path(
        "/Users/mejdi/Documents/New project/cad42_platform/benchmark_v1/references_isidor"
    )
    workspace_ref_dir.mkdir(parents=True, exist_ok=True)

    sections = parse_xlsx_sections(xlsx)
    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    model_map = {
        "openai": args.openai_model,
        "anthropic": args.anthropic_model,
    }

    pairs = []
    missing_refs = []
    generation_errors = []
    processed = 0

    for sec in sections:
        name = str(sec["name"]).strip().lower()
        for vals in sec["rows"]:
            if not vals or not (vals[0] and str(vals[0]).strip().isdigit()):
                continue

            if name == "box hole":
                part_id, prompt = _make_box_hole(vals)
            elif name == "stepped shaft":
                part_id, prompt = _make_stepped_shaft(vals)
            elif name == "flange":
                part_id, prompt = _make_flange(vals)
            elif name == "ribbed bracket":
                part_id, prompt = _make_ribbed_bracket(vals)
            elif name == "shell enclosure":
                part_id, prompt = _make_shell(vals)
            elif name == "manifold block":
                part_id, prompt = _make_manifold(vals)
            elif name == "counterbore countersink plate":
                part_id, prompt = _make_cbore_plate(vals)
            elif name == "threaded parts proxy":
                part_id, prompt = _make_threaded(vals)
            elif name in ("off-axis features", "block"):
                part_id, prompt = _make_offaxis(vals)
            elif name == "spur gear":
                part_id, prompt = _make_gear(vals)
            else:
                part_id, prompt = None, None

            if not part_id or not prompt:
                continue

            ref_name_step = f"{part_id}.step".lower()
            ref_name_stp = f"{part_id}.stp".lower()
            ref_path = ref_map.get(ref_name_step) or ref_map.get(ref_name_stp)
            if not ref_path:
                ext = external_map.get(ref_name_step) or external_map.get(ref_name_stp)
                if ext:
                    # copy into workspace to keep reference + STL inside repo
                    target = workspace_ref_dir / ext.name
                    if not target.exists():
                        target.write_bytes(ext.read_bytes())
                    ref_path = target
                else:
                    missing_refs.append(part_id)
                    continue

            # Create reference STL in benchmark_v1/references_parametric_stl
            ref_stl_dir = ref_path.parent.parent / "references_parametric_stl"
            ref_stl_dir.mkdir(parents=True, exist_ok=True)
            ref_stl = ref_stl_dir / ref_path.with_suffix(".stl").name
            if not ref_stl.exists():
                try:
                    convert_step_to_stl(str(ref_path), str(ref_stl))
                except Exception as exc:
                    generation_errors.append({"part_id": part_id, "error": f"ref_stl:{exc}"})
                    continue

            family = _part_family_from_id(part_id)

            for provider in providers:
                model = model_map.get(provider)
                if not model:
                    continue
                run_dir = out_dir / provider / part_id
                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    gen = generate_and_export(
                        prompt=prompt,
                        provider=provider,
                        model=model,
                        run_dir=str(run_dir),
                        api_key=None,
                        generation_mode="generate",
                        base_step_path=None,
                    )
                except GenerationError as exc:
                    generation_errors.append(
                        {"part_id": part_id, "provider": provider, "error": str(exc)}
                    )
                    continue

                gen_step = Path(gen["generated_step_path"])
                gen_stl = Path(gen["generated_stl_path"])
                if not gen_stl.exists():
                    generation_errors.append(
                        {"part_id": part_id, "provider": provider, "error": "missing generated.stl"}
                    )
                    continue

                pair_id = f"{part_id}__{provider}"
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "part_id": part_id,
                        "family": family,
                        "prompt_level": "L3",
                        "provider": provider,
                        "model": model,
                        "reference_path": str(ref_path),
                        "generated_path": str(gen_step),
                        "generated_mesh_path": str(gen_stl),
                        "prompt_text": prompt,
                        "label": "",
                    }
                )

            processed += 1
            if args.limit and processed >= args.limit:
                break
        if args.limit and processed >= args.limit:
            break

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
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
        for r in pairs:
            w.writerow(r)

    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "missing_refs.json").write_text(json.dumps(missing_refs, indent=2), encoding="utf-8")
    (log_dir / "generation_errors.json").write_text(
        json.dumps(generation_errors, indent=2), encoding="utf-8"
    )

    print(f"pairs_written={len(pairs)}")
    print(f"missing_refs={len(missing_refs)}")
    print(f"generation_errors={len(generation_errors)}")
    print(f"csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cadquery as cq


def _rand_even(rng: random.Random, lo: int, hi: int) -> int:
    v = rng.randrange(lo, hi + 1)
    return v if v % 2 == 0 else v + 1


def _polar_points(count: int, radius: float) -> List[Tuple[float, float]]:
    import math

    pts: List[Tuple[float, float]] = []
    for i in range(count):
        a = 2.0 * math.pi * i / count
        pts.append((radius * math.cos(a), radius * math.sin(a)))
    return pts


def build_box_hole(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 50, 140))
    ly = float(_rand_even(rng, 40, 110))
    lz = float(_rand_even(rng, 12, 60))
    hole_d = float(_rand_even(rng, 8, int(min(lx, ly) * 0.45)))

    part = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    part = part.faces(">Z").workplane().hole(hole_d)
    return part, {"lx": lx, "ly": ly, "lz": lz, "hole_d": hole_d}


def build_stepped_shaft(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    d1 = float(_rand_even(rng, 10, 24))
    d2 = float(_rand_even(rng, int(d1 + 2), 36))
    d3 = float(_rand_even(rng, 8, int(max(d1 - 2, 10))))
    l1 = float(_rand_even(rng, 20, 60))
    l2 = float(_rand_even(rng, 16, 50))
    l3 = float(_rand_even(rng, 18, 56))

    total = l1 + l2 + l3
    part = (
        cq.Workplane("XY")
        .circle(d1 / 2.0)
        .extrude(l1)
        .faces(">Z")
        .workplane()
        .circle(d2 / 2.0)
        .extrude(l2)
        .faces(">Z")
        .workplane()
        .circle(d3 / 2.0)
        .extrude(l3)
        .translate((0, 0, -total / 2.0))
    )
    return part, {"d1": d1, "d2": d2, "d3": d3, "l1": l1, "l2": l2, "l3": l3}


def build_flange(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    od = float(_rand_even(rng, 60, 140))
    thickness = float(_rand_even(rng, 8, 24))
    bore_d = float(_rand_even(rng, 12, int(od * 0.45)))
    bolt_count = int(rng.choice([4, 6, 8]))
    bolt_d = float(_rand_even(rng, 4, 12))
    bolt_radius = od * rng.uniform(0.28, 0.38)

    part = cq.Workplane("XY").circle(od / 2.0).extrude(thickness)
    part = part.faces(">Z").workplane().hole(bore_d)
    part = part.faces(">Z").workplane().pushPoints(_polar_points(bolt_count, bolt_radius)).hole(bolt_d)
    return part, {
        "od": od,
        "thickness": thickness,
        "bore_d": bore_d,
        "bolt_count": bolt_count,
        "bolt_d": bolt_d,
        "bolt_radius": bolt_radius,
    }


def build_ring_spacer(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    od = float(_rand_even(rng, 36, 120))
    id_ = float(_rand_even(rng, 12, int(od * 0.75)))
    h = float(_rand_even(rng, 6, 30))
    if id_ >= od - 4:
        id_ = od - 6
    part = cq.Workplane("XY").circle(od / 2.0).circle(id_ / 2.0).extrude(h)
    return part, {"od": od, "id": id_, "h": h}


def build_slotted_plate(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 90, 190))
    ly = float(_rand_even(rng, 45, 100))
    lz = float(_rand_even(rng, 6, 20))
    slot_len = float(_rand_even(rng, int(lx * 0.35), int(lx * 0.7)))
    slot_w = float(_rand_even(rng, 8, int(ly * 0.4)))

    base = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    slot = cq.Workplane("XY").slot2D(slot_len, slot_w).extrude(lz + 2.0).translate((0, 0, -1.0))
    part = base.cut(slot)
    return part, {"lx": lx, "ly": ly, "lz": lz, "slot_len": slot_len, "slot_w": slot_w}


def build_l_bracket(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    base_l = float(_rand_even(rng, 60, 140))
    base_w = float(_rand_even(rng, 30, 80))
    base_t = float(_rand_even(rng, 6, 16))
    web_t = float(_rand_even(rng, 6, 16))
    web_h = float(_rand_even(rng, 30, 90))
    top_hole_d = float(_rand_even(rng, 6, 12))
    side_hole_d = float(_rand_even(rng, 5, 10))

    base = cq.Workplane("XY").box(base_l, base_w, base_t, centered=(True, True, False))
    web_x = -base_l / 2.0 + web_t / 2.0
    web = cq.Workplane("XY").box(web_t, base_w, web_h, centered=(True, True, False)).translate((web_x, 0, base_t))
    part = base.union(web)

    dx = base_l * 0.28
    part = part.faces(">Z").workplane().pushPoints([(-dx, 0), (dx, 0)]).hole(top_hole_d)
    part = part.faces("<X").workplane(centerOption="CenterOfMass").center(0, base_w * 0.25).hole(side_hole_d)
    return part, {
        "base_l": base_l,
        "base_w": base_w,
        "base_t": base_t,
        "web_t": web_t,
        "web_h": web_h,
        "top_hole_d": top_hole_d,
        "side_hole_d": side_hole_d,
    }


def build_pillow_block(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 80, 160))
    ly = float(_rand_even(rng, 45, 95))
    lz = float(_rand_even(rng, 26, 60))
    bore_d = float(_rand_even(rng, 12, int(min(ly, lz) * 0.55)))
    mount_d = float(_rand_even(rng, 6, 12))

    part = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    part = part.faces(">Z").workplane().hole(bore_d)
    px = lx * 0.35
    py = ly * 0.32
    pads = [(-px, -py), (px, -py), (-px, py), (px, py)]
    part = part.faces("<Z").workplane(centerOption="CenterOfMass").pushPoints(pads).hole(mount_d)
    return part, {"lx": lx, "ly": ly, "lz": lz, "bore_d": bore_d, "mount_d": mount_d}


def build_pulley(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    r1 = float(_rand_even(rng, 22, 50))
    r2 = float(_rand_even(rng, 16, int(r1 - 2)))
    r3 = float(_rand_even(rng, 10, int(r2 - 2)))
    h1 = float(_rand_even(rng, 6, 14))
    h2 = float(_rand_even(rng, 8, 16))
    h3 = float(_rand_even(rng, 6, 14))
    bore_d = float(_rand_even(rng, 8, int(r3 * 0.8)))

    part = cq.Workplane("XY").circle(r1).extrude(h1)
    part = part.faces(">Z").workplane().circle(r2).extrude(h2)
    part = part.faces(">Z").workplane().circle(r3).extrude(h3)
    part = part.faces(">Z").workplane(centerOption="CenterOfMass").hole(bore_d)
    return part, {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "bore_d": bore_d,
    }


def build_u_channel(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 80, 160))
    ly = float(_rand_even(rng, 36, 70))
    lz = float(_rand_even(rng, 24, 54))
    wall = float(_rand_even(rng, 4, 10))
    floor = float(_rand_even(rng, 4, 10))
    hole_d = float(_rand_even(rng, 5, 10))

    inner_x = max(lx - 2.0 * wall, 20.0)
    inner_y = max(ly - 2.0 * wall, 14.0)
    inner_z = max(lz - floor, 10.0)

    outer = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    inner = cq.Workplane("XY").box(inner_x, inner_y, inner_z, centered=(True, True, False)).translate((0, 0, floor))
    part = outer.cut(inner)

    px = lx * 0.35
    part = part.faces(">Z").workplane().pushPoints([(-px, 0), (px, 0)]).hole(hole_d)
    return part, {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "wall": wall,
        "floor": floor,
        "hole_d": hole_d,
    }


FAMILY_BUILDERS: Dict[str, Callable[[random.Random], Tuple[cq.Workplane, Dict[str, float]]]] = {
    "box_hole": build_box_hole,
    "stepped_shaft": build_stepped_shaft,
    "flange": build_flange,
    "ring_spacer": build_ring_spacer,
    "slotted_plate": build_slotted_plate,
    "l_bracket": build_l_bracket,
    "pillow_block": build_pillow_block,
    "pulley": build_pulley,
    "u_channel": build_u_channel,
}


def _bbox_and_volume(part: cq.Workplane) -> Tuple[float, float, float, float]:
    shape = part.val()
    bb = shape.BoundingBox()
    return float(bb.xlen), float(bb.ylen), float(bb.zlen), float(shape.Volume())


def generate_dataset(
    output_dir: Path,
    manifest_path: Path,
    target_count: int,
    seed: int,
) -> None:
    if target_count <= 0:
        raise ValueError("target_count must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    families = list(FAMILY_BUILDERS.keys())

    rows: List[Dict[str, str]] = []
    generated = 0
    family_idx = 0
    serial = 1

    while generated < target_count:
        family = families[family_idx % len(families)]
        family_idx += 1
        builder = FAMILY_BUILDERS[family]

        try:
            part, params = builder(rng)
            xlen, ylen, zlen, volume = _bbox_and_volume(part)
        except Exception as exc:
            print(f"skip family={family} error={exc}")
            continue

        part_id = f"{family}_{serial:04d}"
        serial += 1
        step_path = output_dir / f"{part_id}.step"

        try:
            cq.exporters.export(part, str(step_path))
        except Exception as exc:
            print(f"export failed part_id={part_id} error={exc}")
            continue

        rows.append(
            {
                "part_id": part_id,
                "family": family,
                "step_path": str(step_path),
                "bbox_x_mm": f"{xlen:.6f}",
                "bbox_y_mm": f"{ylen:.6f}",
                "bbox_z_mm": f"{zlen:.6f}",
                "volume_mm3": f"{volume:.6f}",
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
        generated += 1
        print(f"[{generated}/{target_count}] wrote {step_path.name}")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "part_id",
                "family",
                "step_path",
                "bbox_x_mm",
                "bbox_y_mm",
                "bbox_z_mm",
                "volume_mm3",
                "params_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"done: generated={len(rows)} manifest={manifest_path}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate parametric STEP reference dataset.")
    parser.add_argument(
        "--target-count",
        type=int,
        default=80,
        help="Number of STEP files to generate (recommended 50-100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "references_parametric",
        help="Directory for generated STEP files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=here / "references_parametric_manifest.csv",
        help="CSV manifest output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        target_count=args.target_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

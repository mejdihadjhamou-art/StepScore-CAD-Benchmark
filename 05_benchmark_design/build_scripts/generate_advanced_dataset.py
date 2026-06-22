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
    return v if v % 2 == 0 else (v + 1 if v < hi else v - 1)


def _polar_points(count: int, radius: float) -> List[Tuple[float, float]]:
    import math

    pts: List[Tuple[float, float]] = []
    for i in range(count):
        a = 2.0 * math.pi * i / count
        pts.append((radius * math.cos(a), radius * math.sin(a)))
    return pts


def _bbox_and_volume(part: cq.Workplane) -> Tuple[float, float, float, float]:
    shape = part.val()
    bb = shape.BoundingBox()
    return float(bb.xlen), float(bb.ylen), float(bb.zlen), float(shape.Volume())


def build_manifold_block(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 90, 160))
    ly = float(_rand_even(rng, 70, 130))
    lz = float(_rand_even(rng, 60, 120))
    dz = float(_rand_even(rng, 10, 22))
    dx = float(_rand_even(rng, 8, 20))
    dy = float(_rand_even(rng, 8, 20))
    blind_d = float(_rand_even(rng, 10, 22))
    blind_depth = float(_rand_even(rng, 14, int(max(lz * 0.45, 16))))

    part = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    part = part.faces(">Z").workplane().hole(dz)
    part = part.faces(">X").workplane(centerOption="CenterOfMass").hole(dx)
    part = part.faces(">Y").workplane(centerOption="CenterOfMass").hole(dy)
    part = part.faces(">Z").workplane().center(lx * 0.18, -ly * 0.2).hole(blind_d, depth=blind_depth)
    return part, {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "z_through_d": dz,
        "x_through_d": dx,
        "y_through_d": dy,
        "blind_d": blind_d,
        "blind_depth": blind_depth,
    }


def build_threaded_parts_male(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    length = float(_rand_even(rng, 30, 70))
    pitch = float(_rand_even(rng, 2, 5))
    core_d = float(_rand_even(rng, 12, 24))
    thread_depth = rng.uniform(0.6, 1.2)
    thread_band_w = max(1.0, pitch * 0.45)
    cham = min(1.0, length * 0.05)
    turns_deg = 360.0 * length / max(pitch, 1e-6)

    core = cq.Workplane("XY").circle(core_d / 2.0).extrude(length)
    part = core
    fallback = False
    try:
        ridge = (
            cq.Workplane("XY")
            .center(core_d / 2.0 + thread_depth * 0.5, 0)
            .rect(thread_depth, thread_band_w)
            .twistExtrude(length, angleDegrees=turns_deg)
        )
        part = core.union(ridge)
    except Exception:
        # Fallback: stacked external thread-like rings (non-helical but robust).
        fallback = True
        n_rings = max(6, int(length / max(pitch, 1.0)))
        ring_h = max(0.6, thread_band_w * 0.32)
        for i in range(n_rings):
            z = (i + 0.5) * (length / n_rings)
            ring = (
                cq.Workplane("XY")
                .circle(core_d / 2.0 + thread_depth * 0.55)
                .extrude(ring_h)
                .translate((0, 0, z - ring_h / 2.0))
            )
            part = part.union(ring)
    if not fallback:
        try:
            part = part.faces(">Z").chamfer(cham)
            part = part.faces("<Z").chamfer(cham)
        except Exception:
            pass
    return part, {
        "length": length,
        "pitch": pitch,
        "core_d": core_d,
        "thread_depth": thread_depth,
        "thread_band_w": thread_band_w,
        "fallback_rings_used": 1 if fallback else 0,
    }


def build_threaded_parts_female(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    length = float(_rand_even(rng, 26, 56))
    pitch = float(_rand_even(rng, 2, 5))
    outer_d = float(_rand_even(rng, 28, 52))
    bore_d = float(_rand_even(rng, 14, int(max(outer_d * 0.68, 16))))
    thread_depth = rng.uniform(0.5, 1.0)
    thread_band_w = max(1.0, pitch * 0.42)
    turns_deg = 360.0 * length / max(pitch, 1e-6)

    part = cq.Workplane("XY").circle(outer_d / 2.0).extrude(length)
    part = part.faces(">Z").workplane().hole(bore_d)

    groove = (
        cq.Workplane("XY")
        .center(bore_d / 2.0 + thread_depth * 0.25, 0)
        .rect(thread_depth, thread_band_w)
        .twistExtrude(length, angleDegrees=turns_deg)
    )
    part = part.cut(groove)
    return part, {
        "length": length,
        "pitch": pitch,
        "outer_d": outer_d,
        "bore_d": bore_d,
        "thread_depth": thread_depth,
        "thread_band_w": thread_band_w,
    }


def build_keyed_shaft_hub(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    shaft_d = float(_rand_even(rng, 16, 30))
    shaft_l = float(_rand_even(rng, 90, 170))
    hub_d = float(_rand_even(rng, int(max(shaft_d + 14, 34)), 66))
    hub_l = float(_rand_even(rng, 24, 46))
    key_w = float(_rand_even(rng, 4, 10))
    key_d = float(_rand_even(rng, 2, int(max(shaft_d * 0.2, 3))))

    shaft = cq.Workplane("XY").circle(shaft_d / 2.0).extrude(shaft_l).translate((0, 0, -shaft_l / 2.0))
    hub = cq.Workplane("XY").circle(hub_d / 2.0).extrude(hub_l).translate((0, 0, -hub_l / 2.0))
    part = shaft.union(hub)

    slot_len = shaft_l * 0.65
    slot = (
        cq.Workplane("XY")
        .box(key_w, shaft_d, slot_len, centered=(True, True, True))
        .translate((0, shaft_d / 2.0 - key_d / 2.0, 0))
    )
    part = part.cut(slot)
    return part, {
        "shaft_d": shaft_d,
        "shaft_l": shaft_l,
        "hub_d": hub_d,
        "hub_l": hub_l,
        "key_w": key_w,
        "key_d": key_d,
    }


def build_counterbore_countersink_plate(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 90, 170))
    ly = float(_rand_even(rng, 60, 120))
    lz = float(_rand_even(rng, 10, 24))

    cbore_d = float(_rand_even(rng, 5, 10))
    cbore_head_d = float(_rand_even(rng, int(cbore_d + 4), int(cbore_d + 10)))
    cbore_head_h = float(_rand_even(rng, 3, 8))

    csk_d = float(_rand_even(rng, 5, 10))
    csk_head_d = float(_rand_even(rng, int(csk_d + 6), int(csk_d + 14)))
    csk_angle = float(rng.choice([82.0, 90.0, 100.0]))

    through_d = float(_rand_even(rng, 6, 12))
    px = lx * 0.26
    py = ly * 0.24

    part = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    part = part.faces(">Z").workplane().pushPoints([(-px, -py), (px, -py)]).cboreHole(cbore_d, cbore_head_d, cbore_head_h)
    part = part.faces(">Z").workplane().pushPoints([(-px, py), (px, py)]).cskHole(csk_d, csk_head_d, csk_angle)
    part = part.faces(">Z").workplane().hole(through_d)
    return part, {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "cbore_d": cbore_d,
        "cbore_head_d": cbore_head_d,
        "cbore_head_h": cbore_head_h,
        "csk_d": csk_d,
        "csk_head_d": csk_head_d,
        "csk_angle": csk_angle,
        "through_d": through_d,
    }


def build_ribbed_bracket(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    base_l = float(_rand_even(rng, 80, 150))
    base_w = float(_rand_even(rng, 40, 90))
    base_t = float(_rand_even(rng, 8, 18))
    wall_t = float(_rand_even(rng, 8, 16))
    wall_h = float(_rand_even(rng, 40, 90))
    rib_t = float(_rand_even(rng, 5, 10))

    base = cq.Workplane("XY").box(base_l, base_w, base_t, centered=(True, True, False))
    wall = (
        cq.Workplane("XY")
        .box(wall_t, base_w, wall_h, centered=(True, True, False))
        .translate((-base_l / 2.0 + wall_t / 2.0, 0, base_t))
    )

    rib_pts = [
        (-base_l / 2.0 + wall_t, base_t),
        (-base_l / 2.0 + wall_t + base_l * 0.34, base_t),
        (-base_l / 2.0 + wall_t, base_t + wall_h * 0.74),
    ]
    rib1 = cq.Workplane("XZ").polyline(rib_pts).close().extrude(rib_t, both=True).translate((0, -base_w * 0.24, 0))
    rib2 = cq.Workplane("XZ").polyline(rib_pts).close().extrude(rib_t, both=True).translate((0, base_w * 0.24, 0))

    part = base.union(wall).union(rib1).union(rib2)
    hole_d = float(_rand_even(rng, 6, 12))
    part = part.faces(">Z").workplane().pushPoints([(-base_l * 0.2, 0), (base_l * 0.2, 0)]).hole(hole_d)
    return part, {
        "base_l": base_l,
        "base_w": base_w,
        "base_t": base_t,
        "wall_t": wall_t,
        "wall_h": wall_h,
        "rib_t": rib_t,
        "hole_d": hole_d,
    }


def build_shell_enclosure(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 90, 170))
    ly = float(_rand_even(rng, 70, 140))
    lz = float(_rand_even(rng, 40, 90))
    wall = float(_rand_even(rng, 3, 6))
    boss_od = float(_rand_even(rng, 8, 16))
    boss_hole = float(_rand_even(rng, 4, 8))
    boss_h = float(_rand_even(rng, 10, int(max(lz * 0.4, 12))))

    outer = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    shell = outer.faces(">Z").shell(-wall)

    px = lx * 0.30
    py = ly * 0.26
    bosses = (
        cq.Workplane("XY")
        .pushPoints([(-px, -py), (px, -py), (-px, py), (px, py)])
        .circle(boss_od / 2.0)
        .extrude(boss_h)
        .translate((0, 0, wall))
    )
    part = shell.union(bosses)
    part = part.faces(">Z").workplane(offset=-lz + wall + boss_h).pushPoints([(-px, -py), (px, -py), (-px, py), (px, py)]).hole(boss_hole)

    cut_w = float(_rand_even(rng, 14, int(max(ly * 0.45, 18))))
    cut_h = float(_rand_even(rng, 10, int(max(lz * 0.35, 12))))
    cut = (
        cq.Workplane("YZ")
        .center(0, lz * 0.45)
        .rect(cut_w, cut_h)
        .extrude(wall + 2.0)
        .translate((lx / 2.0 - wall - 1.0, 0, 0))
    )
    part = part.cut(cut)
    return part, {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "wall": wall,
        "boss_od": boss_od,
        "boss_hole": boss_hole,
        "boss_h": boss_h,
        "side_cut_w": cut_w,
        "side_cut_h": cut_h,
    }


def build_loft_transition(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    h = float(_rand_even(rng, 50, 110))
    r = float(_rand_even(rng, 16, 36))
    rx = float(_rand_even(rng, 38, 80))
    ry = float(_rand_even(rng, 28, 66))

    part = (
        cq.Workplane("XY")
        .circle(r)
        .workplane(offset=h)
        .rect(rx, ry)
        .loft(combine=True)
    )
    return part, {
        "height": h,
        "base_radius": r,
        "top_rect_x": rx,
        "top_rect_y": ry,
    }


def build_sweep_path_part(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    span = float(_rand_even(rng, 110, 220))
    rise = float(_rand_even(rng, 30, 70))
    tube_r = rng.uniform(4.0, 8.0)

    path = (
        cq.Workplane("XZ")
        .moveTo(0, 0)
        .threePointArc((span * 0.25, rise), (span * 0.5, 0))
        .threePointArc((span * 0.75, -rise), (span, 0))
        .wire()
    )
    profile = cq.Workplane("YZ").circle(tube_r)
    part = profile.sweep(path, isFrenet=True)
    return part, {
        "span": span,
        "rise": rise,
        "tube_r": tube_r,
    }


def build_drafted_part(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 60, 120))
    ly = float(_rand_even(rng, 50, 110))
    h = float(_rand_even(rng, 30, 70))
    draft_deg = rng.uniform(2.0, 7.0)
    hole_d = float(_rand_even(rng, 8, 18))

    part = cq.Workplane("XY").rect(lx, ly).extrude(h, taper=draft_deg)
    part = part.faces(">Z").workplane().hole(hole_d)
    return part, {
        "lx": lx,
        "ly": ly,
        "h": h,
        "draft_deg": draft_deg,
        "hole_d": hole_d,
    }


def build_helical_features(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    core_d = float(_rand_even(rng, 10, 22))
    height = float(_rand_even(rng, 40, 90))
    pitch = float(_rand_even(rng, 8, 16))
    flight_t = rng.uniform(1.2, 2.8)
    flight_h = pitch * 0.55
    radius = core_d / 2.0 + rng.uniform(3.0, 7.0)
    turns_deg = 360.0 * height / max(pitch, 1e-6)

    core = cq.Workplane("XY").circle(core_d / 2.0).extrude(height)
    flight = (
        cq.Workplane("XY")
        .center(radius, 0)
        .rect(flight_t, flight_h)
        .twistExtrude(height, angleDegrees=turns_deg)
    )
    part = core.union(flight)
    return part, {
        "core_d": core_d,
        "height": height,
        "pitch": pitch,
        "flight_t": flight_t,
        "flight_h": flight_h,
        "radius": radius,
        "turns_deg": turns_deg,
    }


def build_chamfer_fillet_stress_cases(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 90, 160))
    ly = float(_rand_even(rng, 70, 130))
    lz = float(_rand_even(rng, 24, 54))
    hole_d = float(_rand_even(rng, 6, 12))
    fillet_r = rng.uniform(1.2, 2.8)
    cham = rng.uniform(0.8, 2.0)

    part = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    px = lx * 0.30
    py = ly * 0.30
    part = part.faces(">Z").workplane().pushPoints([(-px, -py), (px, -py), (-px, py), (px, py)]).hole(hole_d)
    part = part.edges("|Z").fillet(fillet_r)
    part = part.edges(">Z").chamfer(cham)
    part = part.edges("<Z").chamfer(cham * 0.7)
    return part, {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "hole_d": hole_d,
        "fillet_r": fillet_r,
        "chamfer": cham,
    }


def build_off_axis_features(rng: random.Random) -> Tuple[cq.Workplane, Dict[str, float]]:
    lx = float(_rand_even(rng, 80, 150))
    ly = float(_rand_even(rng, 60, 120))
    lz = float(_rand_even(rng, 30, 70))
    boss_d = float(_rand_even(rng, 14, 28))
    boss_h = float(_rand_even(rng, 12, 30))
    boss_tilt = rng.uniform(12.0, 28.0)
    hole_d = float(_rand_even(rng, 6, 12))
    hole_tilt = rng.uniform(18.0, 34.0)

    part = cq.Workplane("XY").box(lx, ly, lz, centered=(True, True, False))
    part = (
        part.faces(">Z")
        .workplane()
        .transformed(rotate=(0, boss_tilt, 0))
        .center(-lx * 0.15, ly * 0.10)
        .circle(boss_d / 2.0)
        .extrude(boss_h)
    )
    part = (
        part.faces(">Y")
        .workplane(centerOption="CenterOfMass")
        .transformed(rotate=(hole_tilt, 0, 0))
        .hole(hole_d)
    )
    return part, {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "boss_d": boss_d,
        "boss_h": boss_h,
        "boss_tilt_deg": boss_tilt,
        "hole_d": hole_d,
        "hole_tilt_deg": hole_tilt,
    }


FAMILY_BUILDERS: Dict[str, Callable[[random.Random], Tuple[cq.Workplane, Dict[str, float]]]] = {
    "manifold_block": build_manifold_block,
    "threaded_parts_male": build_threaded_parts_male,
    "threaded_parts_female": build_threaded_parts_female,
    "keyed_shaft_hub": build_keyed_shaft_hub,
    "counterbore_countersink_plate": build_counterbore_countersink_plate,
    "ribbed_bracket": build_ribbed_bracket,
    "shell_enclosure": build_shell_enclosure,
    "loft_transition": build_loft_transition,
    "sweep_path_part": build_sweep_path_part,
    "drafted_part": build_drafted_part,
    "helical_features": build_helical_features,
    "chamfer_fillet_stress_cases": build_chamfer_fillet_stress_cases,
    "off_axis_features": build_off_axis_features,
}


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
    attempts = 0
    family_idx = 0
    serial = 1
    max_attempts = max(target_count * 120, 1000)

    while generated < target_count and attempts < max_attempts:
        attempts += 1
        family = families[family_idx % len(families)]
        family_idx += 1
        builder = FAMILY_BUILDERS[family]

        try:
            part, params = builder(rng)
            xlen, ylen, zlen, volume = _bbox_and_volume(part)
        except Exception as exc:
            print(f"skip build family={family} error={exc}")
            continue

        part_id = f"{family}_{serial:04d}"
        serial += 1
        step_path = output_dir / f"{part_id}.step"
        if step_path.exists():
            continue

        try:
            cq.exporters.export(part, str(step_path))
        except Exception as exc:
            print(f"skip export part_id={part_id} family={family} error={exc}")
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

    print(
        "done: "
        f"generated={len(rows)} target={target_count} attempts={attempts} "
        f"manifest={manifest_path}"
    )
    if len(rows) < target_count:
        print("warning: target not fully reached; consider increasing max_attempts or adjusting builders.")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate advanced STEP references across harder CAD families.")
    parser.add_argument(
        "--target-count",
        type=int,
        default=80,
        help="Number of STEP files to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "references_advanced",
        help="Directory for generated STEP files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=here / "references_advanced_manifest.csv",
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

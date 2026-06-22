from __future__ import annotations

from pathlib import Path

import cadquery as cq


OUT_DIR = Path(__file__).resolve().parent / "references"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def export_step(name: str, part: cq.Workplane) -> None:
    out_path = OUT_DIR / f"{name}.step"
    cq.exporters.export(part, str(out_path))
    print(f"wrote {out_path}")


def stepped_shaft() -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .circle(8)
        .extrude(30)
        .faces(">Z")
        .workplane()
        .circle(12)
        .extrude(20)
        .faces(">Z")
        .workplane()
        .circle(6)
        .extrude(18)
        .translate((0, 0, -34))
    )


def l_bracket_2hole() -> cq.Workplane:
    base = cq.Workplane("XY").box(80, 30, 8, centered=(True, True, False))
    upright = cq.Workplane("XY").box(8, 30, 50, centered=(True, True, False)).translate((-36, 0, 8))
    part = base.union(upright)
    part = part.faces(">Z").workplane().pushPoints([(-22, 0), (22, 0)]).hole(8)
    part = (
        part.faces("<X")
        .workplane(centerOption="CenterOfMass")
        .center(0, 12)
        .hole(6)
    )
    return part


def flange_4hole() -> cq.Workplane:
    part = cq.Workplane("XY").circle(35).extrude(12)
    part = part.faces(">Z").workplane().circle(12).cutBlind(-12)
    pts = [(20, 20), (-20, 20), (-20, -20), (20, -20)]
    return part.faces(">Z").workplane().pushPoints(pts).hole(6)


def ring_spacer() -> cq.Workplane:
    return cq.Workplane("XY").circle(26).circle(16).extrude(14)


def slotted_plate() -> cq.Workplane:
    part = cq.Workplane("XY").box(120, 60, 10, centered=(True, True, False))
    slot = cq.Workplane("XY").slot2D(50, 14).extrude(14, both=False).translate((0, 0, -2))
    return part.cut(slot)


def pillow_block() -> cq.Workplane:
    part = cq.Workplane("XY").box(100, 55, 35, centered=(True, True, False))
    part = part.faces(">Z").workplane().center(0, 0).hole(20)
    pads = [(-35, -18), (35, -18), (-35, 18), (35, 18)]
    return part.faces("<Z").workplane(centerOption="CenterOfMass").pushPoints(pads).hole(9)


def pulley_2step() -> cq.Workplane:
    part = cq.Workplane("XY").circle(32).extrude(8)
    part = part.faces(">Z").workplane().circle(24).extrude(12)
    part = part.faces(">Z").workplane().circle(18).extrude(8)
    return part.faces(">Z").workplane(centerOption="CenterOfMass").hole(12)


def simple_bushing() -> cq.Workplane:
    body = cq.Workplane("XY").circle(16).circle(10).extrude(22)
    flange = cq.Workplane("XY").circle(24).circle(10).extrude(4)
    return flange.union(body.translate((0, 0, 4)))


def u_channel() -> cq.Workplane:
    outer = cq.Workplane("XY").box(100, 40, 35, centered=(True, True, False))
    inner = cq.Workplane("XY").box(88, 28, 29, centered=(True, True, False)).translate((0, 0, 6))
    part = outer.cut(inner)
    return part.faces(">Z").workplane().pushPoints([(-35, 0), (35, 0)]).hole(7)


def run() -> None:
    export_step("stepped_shaft", stepped_shaft())
    export_step("l_bracket_2hole", l_bracket_2hole())
    export_step("flange_4hole", flange_4hole())
    export_step("ring_spacer", ring_spacer())
    export_step("slotted_plate", slotted_plate())
    export_step("pillow_block", pillow_block())
    export_step("pulley_2step", pulley_2step())
    export_step("simple_bushing", simple_bushing())
    export_step("u_channel", u_channel())


if __name__ == "__main__":
    run()

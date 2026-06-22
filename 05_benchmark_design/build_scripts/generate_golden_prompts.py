from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


def _mm(v: Any) -> str:
    try:
        f = float(v)
    except Exception:
        return str(v)
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:.4f}".rstrip("0").rstrip(".")


def _header_l2(part_name: str) -> str:
    return (
        "Create a single connected mechanical CAD part.\n\n"
        f"Part identity:\n{part_name}\n"
        "Units:\nMillimeters (mm).\n"
    )


def _footer_l2() -> str:
    return (
        "\nOutput constraints:\n"
        "- Exactly one solid body.\n"
        "- No floating or disconnected geometry.\n"
        "- Preserve all dimensions exactly.\n"
    )


def _header_l3(part_name: str, origin_note: str) -> str:
    return (
        "Create a single connected mechanical CAD part.\n\n"
        "Part identity:\n"
        f"{part_name}\n\n"
        "Units and coordinate frame:\n"
        "- Units: millimeters.\n"
        "- Coordinate system: right-handed XYZ.\n"
        f"- {origin_note}\n"
        "- +Z is the primary extrusion/height direction unless stated otherwise.\n\n"
    )


def _footer_l3() -> str:
    return (
        "Topology/output constraints:\n"
        "- Return exactly one connected solid body.\n"
        "- No floating pieces, no non-manifold geometry.\n"
        "- Respect all stated dimensions and feature counts exactly.\n"
    )


def _l2_box_hole(p: Dict[str, Any]) -> str:
    lx, ly, lz, hd = _mm(p["lx"]), _mm(p["ly"]), _mm(p["lz"]), _mm(p["hole_d"])
    return _header_l2("Rectangular block with a centered through hole") + (
        "Geometry:\n"
        f"- Base block size: X={lx}, Y={ly}, Z={lz}.\n"
        f"- Add one centered vertical through hole, diameter={hd}, axis along Z.\n"
    ) + _footer_l2()


def _l3_box_hole(p: Dict[str, Any]) -> str:
    lx, ly, lz, hd = _mm(p["lx"]), _mm(p["ly"]), _mm(p["lz"]), _mm(p["hole_d"])
    return _header_l3(
        "Rectangular block with one centered vertical through-hole",
        "Origin: (0,0,0) at the center of the bottom outer face of the block.",
    ) + (
        "Base geometry:\n"
        f"- Create a rectangular solid with dimensions X={lx}, Y={ly}, Z={lz}.\n"
        "- Block is centered in X and Y, with bottom at z=0 and top at z=Z.\n\n"
        "Subtractive features:\n"
        f"- Create one circular through-hole with diameter={hd}.\n"
        "- Hole center at (0,0) in XY and axis parallel to Z.\n"
        "- Hole cuts fully through from top to bottom.\n\n"
    ) + _footer_l3()


def _l2_stepped_shaft(p: Dict[str, Any]) -> str:
    d1, d2, d3 = _mm(p["d1"]), _mm(p["d2"]), _mm(p["d3"])
    l1, l2, l3 = _mm(p["l1"]), _mm(p["l2"]), _mm(p["l3"])
    total = _mm(float(p["l1"]) + float(p["l2"]) + float(p["l3"]))
    return _header_l2("Three-step coaxial shaft") + (
        "Geometry:\n"
        f"- Segment 1: diameter={d1}, length={l1}.\n"
        f"- Segment 2: diameter={d2}, length={l2}, attached coaxially after segment 1.\n"
        f"- Segment 3: diameter={d3}, length={l3}, attached coaxially after segment 2.\n"
        f"- Overall shaft length={total}.\n"
        "- Shaft axis along Z.\n"
    ) + _footer_l2()


def _l3_stepped_shaft(p: Dict[str, Any]) -> str:
    d1, d2, d3 = _mm(p["d1"]), _mm(p["d2"]), _mm(p["d3"])
    l1, l2, l3 = _mm(p["l1"]), _mm(p["l2"]), _mm(p["l3"])
    total = float(p["l1"]) + float(p["l2"]) + float(p["l3"])
    h0 = -total / 2.0
    h1 = h0 + float(p["l1"])
    h2 = h1 + float(p["l2"])
    h3 = h2 + float(p["l3"])
    return _header_l3(
        "Three-step coaxial shaft with shoulders",
        "Origin at the shaft center (midpoint of total length on Z-axis).",
    ) + (
        "Base/additive geometry:\n"
        f"- Build three coaxial cylindrical segments along Z with diameters {d1}, {d2}, {d3}.\n"
        f"- Segment lengths are {l1}, {l2}, {l3} respectively.\n"
        f"- Total length is {_mm(total)}.\n"
        f"- Segment 1 spans z={_mm(h0)} to z={_mm(h1)}.\n"
        f"- Segment 2 spans z={_mm(h1)} to z={_mm(h2)}.\n"
        f"- Segment 3 spans z={_mm(h2)} to z={_mm(h3)}.\n"
        "- All segments are concentric on X=0, Y=0.\n\n"
    ) + _footer_l3()


def _l2_flange(p: Dict[str, Any]) -> str:
    od = _mm(p["od"])
    th = _mm(p["thickness"])
    bore = _mm(p["bore_d"])
    bc = int(p["bolt_count"])
    bd = _mm(p["bolt_d"])
    bcd = _mm(2.0 * float(p["bolt_radius"]))
    return _header_l2("Circular flange with bore and bolt pattern") + (
        "Geometry:\n"
        f"- Flange outer diameter={od}, thickness={th}.\n"
        f"- Center through bore diameter={bore}.\n"
        f"- Add {bc} equally spaced through bolt holes, each diameter={bd}.\n"
        f"- Bolt holes are on bolt circle diameter={bcd}.\n"
    ) + _footer_l2()


def _l3_flange(p: Dict[str, Any]) -> str:
    od = _mm(p["od"])
    th = _mm(p["thickness"])
    bore = _mm(p["bore_d"])
    bc = int(p["bolt_count"])
    bd = _mm(p["bolt_d"])
    br = float(p["bolt_radius"])
    return _header_l3(
        "Circular flange disk with central bore and equally spaced bolt holes",
        "Origin at center of bottom circular face; flange axis along Z.",
    ) + (
        "Base geometry:\n"
        f"- Create a solid cylinder with outer diameter={od} and height={th}.\n"
        "- Bottom at z=0 and top at z=height.\n\n"
        "Subtractive features:\n"
        f"- Cut one centered through bore, diameter={bore}, axis=Z.\n"
        f"- Cut {bc} through bolt holes, diameter={bd}, equally spaced at 360/{bc} degrees.\n"
        f"- Bolt hole centers lie on radius={_mm(br)} from origin (bolt circle diameter={_mm(2*br)}).\n\n"
    ) + _footer_l3()


def _l2_ring_spacer(p: Dict[str, Any]) -> str:
    od, id_, h = _mm(p["od"]), _mm(p["id"]), _mm(p["h"])
    return _header_l2("Ring spacer (hollow cylinder)") + (
        "Geometry:\n"
        f"- Outer diameter={od}.\n"
        f"- Inner diameter={id_} (through bore).\n"
        f"- Height={h}.\n"
    ) + _footer_l2()


def _l3_ring_spacer(p: Dict[str, Any]) -> str:
    od, id_, h = _mm(p["od"]), _mm(p["id"]), _mm(p["h"])
    wall = _mm((float(p["od"]) - float(p["id"])) / 2.0)
    return _header_l3(
        "Ring spacer (annular cylinder) with through bore",
        "Origin at center of bottom annular face, axis along Z.",
    ) + (
        "Base/subtractive geometry:\n"
        f"- Create outer cylinder: diameter={od}, height={h}.\n"
        f"- Subtract centered inner cylinder: diameter={id_}, same full height, axis=Z.\n"
        f"- Resulting radial wall thickness={wall}.\n"
        "- Bottom at z=0, top at z=height.\n\n"
    ) + _footer_l3()


def _l2_slotted_plate(p: Dict[str, Any]) -> str:
    lx, ly, lz = _mm(p["lx"]), _mm(p["ly"]), _mm(p["lz"])
    sl, sw = _mm(p["slot_len"]), _mm(p["slot_w"])
    return _header_l2("Rectangular plate with a centered through slot") + (
        "Geometry:\n"
        f"- Plate dimensions: X={lx}, Y={ly}, thickness Z={lz}.\n"
        f"- One centered slot with rounded ends: length={sl}, width={sw}.\n"
        "- Slot goes fully through thickness.\n"
    ) + _footer_l2()


def _l3_slotted_plate(p: Dict[str, Any]) -> str:
    lx, ly, lz = _mm(p["lx"]), _mm(p["ly"]), _mm(p["lz"])
    sl, sw = _mm(p["slot_len"]), _mm(p["slot_w"])
    return _header_l3(
        "Flat rectangular plate with one centered rounded-end through slot",
        "Origin at center of bottom plate face; +Z normal to plate.",
    ) + (
        "Base geometry:\n"
        f"- Create rectangular plate, X={lx}, Y={ly}, Z={lz}.\n"
        "- Plate centered on X and Y, bottom at z=0.\n\n"
        "Subtractive feature:\n"
        f"- Cut one centered slot2D profile with overall length={sl} and width={sw}.\n"
        "- Slot long axis along X.\n"
        "- Extrude/cut through full thickness so it is a through-slot.\n\n"
    ) + _footer_l3()


def _l2_l_bracket(p: Dict[str, Any]) -> str:
    bl, bw, bt = _mm(p["base_l"]), _mm(p["base_w"]), _mm(p["base_t"])
    wt, wh = _mm(p["web_t"]), _mm(p["web_h"])
    thd, shd = _mm(p["top_hole_d"]), _mm(p["side_hole_d"])
    return _header_l2("L-bracket with base, vertical web, and holes") + (
        "Geometry:\n"
        f"- Base plate: length={bl}, width={bw}, thickness={bt}.\n"
        f"- Vertical web near one base edge: thickness={wt}, height={wh}, full base width.\n"
        f"- Add two top through holes in the base, each diameter={thd}.\n"
        f"- Add one side through hole in the web, diameter={shd}.\n"
    ) + _footer_l2()


def _l3_l_bracket(p: Dict[str, Any]) -> str:
    bl, bw, bt = float(p["base_l"]), float(p["base_w"]), float(p["base_t"])
    wt, wh = float(p["web_t"]), float(p["web_h"])
    thd, shd = _mm(p["top_hole_d"]), _mm(p["side_hole_d"])
    web_x = -bl / 2.0 + wt / 2.0
    dx = bl * 0.28
    return _header_l3(
        "L-bracket: horizontal base plus one vertical web with mounting holes",
        "Origin at center of bottom face of base plate.",
    ) + (
        "Base/additive geometry:\n"
        f"- Base plate size: X={_mm(bl)}, Y={_mm(bw)}, Z={_mm(bt)}; bottom at z=0.\n"
        f"- Add one vertical web block with thickness X={_mm(wt)}, width Y={_mm(bw)}, height Z={_mm(wh)}.\n"
        f"- Place web near the -X side at web center x={_mm(web_x)} and rising from z={_mm(bt)}.\n\n"
        "Subtractive features:\n"
        f"- Two top through holes in base, diameter={thd}, positioned symmetrically at x=±{_mm(dx)}, y=0.\n"
        f"- One side hole in web, diameter={shd}, normal to X direction, centered on web side face.\n\n"
    ) + _footer_l3()


def _l2_pillow_block(p: Dict[str, Any]) -> str:
    lx, ly, lz = _mm(p["lx"]), _mm(p["ly"]), _mm(p["lz"])
    bd, md = _mm(p["bore_d"]), _mm(p["mount_d"])
    return _header_l2("Pillow-block style mount with central bore and 4 mounting holes") + (
        "Geometry:\n"
        f"- Block dimensions: X={lx}, Y={ly}, Z={lz}.\n"
        f"- One centered vertical through bore, diameter={bd}.\n"
        f"- Four mounting through holes, each diameter={md}, arranged symmetrically on base footprint.\n"
    ) + _footer_l2()


def _l3_pillow_block(p: Dict[str, Any]) -> str:
    lx, ly, lz = float(p["lx"]), float(p["ly"]), float(p["lz"])
    bd, md = _mm(p["bore_d"]), _mm(p["mount_d"])
    px, py = lx * 0.35, ly * 0.32
    return _header_l3(
        "Pillow block: rectangular body with central bore and four corner mounting holes",
        "Origin at center of bottom face of the body.",
    ) + (
        "Base geometry:\n"
        f"- Create rectangular body X={_mm(lx)}, Y={_mm(ly)}, Z={_mm(lz)}, bottom at z=0.\n\n"
        "Subtractive features:\n"
        f"- Cut one centered vertical through bore, diameter={bd}, axis=Z.\n"
        f"- Cut four mounting through holes, diameter={md}, with XY centers at (±{_mm(px)}, ±{_mm(py)}).\n"
        "- Mounting holes are through the body along Z.\n\n"
    ) + _footer_l3()


def _l2_pulley(p: Dict[str, Any]) -> str:
    r1, r2, r3 = _mm(p["r1"]), _mm(p["r2"]), _mm(p["r3"])
    h1, h2, h3 = _mm(p["h1"]), _mm(p["h2"]), _mm(p["h3"])
    bd = _mm(p["bore_d"])
    return _header_l2("Three-step pulley with centered through bore") + (
        "Geometry:\n"
        f"- Coaxial step 1: radius={r1}, height={h1}.\n"
        f"- Coaxial step 2 on top: radius={r2}, height={h2}.\n"
        f"- Coaxial step 3 on top: radius={r3}, height={h3}.\n"
        f"- Center through bore diameter={bd}.\n"
    ) + _footer_l2()


def _l3_pulley(p: Dict[str, Any]) -> str:
    r1, r2, r3 = _mm(p["r1"]), _mm(p["r2"]), _mm(p["r3"])
    h1, h2, h3 = _mm(p["h1"]), _mm(p["h2"]), _mm(p["h3"])
    bd = _mm(p["bore_d"])
    total = _mm(float(p["h1"]) + float(p["h2"]) + float(p["h3"]))
    return _header_l3(
        "Stepped pulley (three coaxial cylindrical steps) with through bore",
        "Origin at center of bottom face; main axis along Z.",
    ) + (
        "Additive geometry:\n"
        f"- Step 1: cylinder radius={r1}, height={h1}, from z=0 upward.\n"
        f"- Step 2: centered on same axis, radius={r2}, height={h2}, stacked on step 1.\n"
        f"- Step 3: centered on same axis, radius={r3}, height={h3}, stacked on step 2.\n"
        f"- Total height={total}.\n\n"
        "Subtractive feature:\n"
        f"- One centered through bore diameter={bd}, axis=Z, through full height.\n\n"
    ) + _footer_l3()


def _l2_u_channel(p: Dict[str, Any]) -> str:
    lx, ly, lz = _mm(p["lx"]), _mm(p["ly"]), _mm(p["lz"])
    wt, ft = _mm(p["wall"]), _mm(p["floor"])
    hd = _mm(p["hole_d"])
    return _header_l2("U-channel with open top and two holes") + (
        "Geometry:\n"
        f"- Outer envelope: X={lx}, Y={ly}, Z={lz}.\n"
        f"- Hollow interior to form a U-channel: wall thickness={wt}, floor thickness={ft}, top open.\n"
        f"- Two through holes on the top side, each diameter={hd}, mirrored along X.\n"
    ) + _footer_l2()


def _l3_u_channel(p: Dict[str, Any]) -> str:
    lx, ly, lz = float(p["lx"]), float(p["ly"]), float(p["lz"])
    wt, ft = float(p["wall"]), float(p["floor"])
    hd = _mm(p["hole_d"])
    inner_x = max(lx - 2.0 * wt, 20.0)
    inner_y = max(ly - 2.0 * wt, 14.0)
    inner_z = max(lz - ft, 10.0)
    px = lx * 0.35
    return _header_l3(
        "Open-top U-channel with two top through-holes",
        "Origin at center of bottom outer face.",
    ) + (
        "Base geometry:\n"
        f"- Create outer rectangular solid X={_mm(lx)}, Y={_mm(ly)}, Z={_mm(lz)}, bottom at z=0.\n\n"
        "Subtractive channel cavity:\n"
        f"- Subtract inner rectangular volume X={_mm(inner_x)}, Y={_mm(inner_y)}, Z={_mm(inner_z)}.\n"
        f"- Offset the inner cut upward by floor thickness z={_mm(ft)}.\n"
        "- This must leave the top open and preserve floor/walls.\n\n"
        "Subtractive holes:\n"
        f"- Add two vertical through holes on the top side, diameter={hd}, at x=±{_mm(px)}, y=0.\n\n"
    ) + _footer_l3()


def _l2_manifold_block(p: Dict[str, Any]) -> str:
    return _header_l2("Manifold block with intersecting channels") + (
        "Geometry:\n"
        f"- Base block: X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}.\n"
        f"- Through hole along Z: diameter={_mm(p['z_through_d'])}.\n"
        f"- Through hole along X: diameter={_mm(p['x_through_d'])}.\n"
        f"- Through hole along Y: diameter={_mm(p['y_through_d'])}.\n"
        f"- Add one blind top port: diameter={_mm(p['blind_d'])}, depth={_mm(p['blind_depth'])}.\n"
    ) + _footer_l2()


def _l3_manifold_block(p: Dict[str, Any]) -> str:
    lx, ly = float(p["lx"]), float(p["ly"])
    return _header_l3(
        "Prismatic manifold block with orthogonal intersecting drilled channels",
        "Origin at center of bottom face of the block.",
    ) + (
        "Base geometry:\n"
        f"- Create block X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}, bottom at z=0.\n\n"
        "Subtractive features:\n"
        f"- Cut centered through-hole along Z, diameter={_mm(p['z_through_d'])}.\n"
        f"- Cut centered through-hole along X, diameter={_mm(p['x_through_d'])}.\n"
        f"- Cut centered through-hole along Y, diameter={_mm(p['y_through_d'])}.\n"
        f"- Cut one blind hole from top face at (x={_mm(lx*0.18)}, y={_mm(-ly*0.2)}), "
        f"diameter={_mm(p['blind_d'])}, depth={_mm(p['blind_depth'])}.\n\n"
    ) + _footer_l3()


def _l2_threaded_parts_male(p: Dict[str, Any]) -> str:
    return _header_l2("Male threaded shaft") + (
        "Geometry:\n"
        f"- Core shaft diameter={_mm(p['core_d'])}, length={_mm(p['length'])}.\n"
        f"- External thread-like form with pitch={_mm(p['pitch'])}, thread depth={_mm(p['thread_depth'])}, "
        f"band width={_mm(p['thread_band_w'])}.\n"
        "- Chamfer both ends.\n"
    ) + _footer_l2()


def _l3_threaded_parts_male(p: Dict[str, Any]) -> str:
    return _header_l3(
        "Externally threaded male shaft",
        "Origin at center of bottom circular face; axis along Z.",
    ) + (
        "Base geometry:\n"
        f"- Create core cylinder diameter={_mm(p['core_d'])}, height={_mm(p['length'])}.\n\n"
        "Thread form:\n"
        f"- Add external helical/thread-like ridge with pitch={_mm(p['pitch'])}, "
        f"thread depth={_mm(p['thread_depth'])}, and band width={_mm(p['thread_band_w'])} around the core.\n"
        "- Keep one connected solid.\n"
        "- Apply small chamfer to both end edges.\n\n"
    ) + _footer_l3()


def _l2_threaded_parts_female(p: Dict[str, Any]) -> str:
    return _header_l2("Female threaded sleeve/nut body") + (
        "Geometry:\n"
        f"- Outer cylinder diameter={_mm(p['outer_d'])}, length={_mm(p['length'])}.\n"
        f"- Through bore diameter={_mm(p['bore_d'])}.\n"
        f"- Internal thread-like groove with pitch={_mm(p['pitch'])}, thread depth={_mm(p['thread_depth'])}, "
        f"band width={_mm(p['thread_band_w'])}.\n"
    ) + _footer_l2()


def _l3_threaded_parts_female(p: Dict[str, Any]) -> str:
    return _header_l3(
        "Female threaded cylindrical sleeve",
        "Origin at center of bottom face; axis along Z.",
    ) + (
        "Base geometry:\n"
        f"- Create outer cylinder diameter={_mm(p['outer_d'])}, height={_mm(p['length'])}.\n"
        f"- Cut centered through bore diameter={_mm(p['bore_d'])}.\n\n"
        "Thread form:\n"
        f"- Subtract internal helical/thread-like groove with pitch={_mm(p['pitch'])}, "
        f"thread depth={_mm(p['thread_depth'])}, and band width={_mm(p['thread_band_w'])}.\n\n"
    ) + _footer_l3()


def _l2_keyed_shaft_hub(p: Dict[str, Any]) -> str:
    return _header_l2("Keyed shaft with central hub") + (
        "Geometry:\n"
        f"- Shaft: diameter={_mm(p['shaft_d'])}, length={_mm(p['shaft_l'])}, axis along Z.\n"
        f"- Central hub: diameter={_mm(p['hub_d'])}, length={_mm(p['hub_l'])}, concentric with shaft.\n"
        f"- Add longitudinal keyway slot: width={_mm(p['key_w'])}, depth={_mm(p['key_d'])}.\n"
    ) + _footer_l2()


def _l3_keyed_shaft_hub(p: Dict[str, Any]) -> str:
    return _header_l3(
        "Concentric shaft-hub part with longitudinal keyway",
        "Origin at the global center of the part, shaft axis along Z.",
    ) + (
        "Base geometry:\n"
        f"- Create shaft cylinder diameter={_mm(p['shaft_d'])}, total length={_mm(p['shaft_l'])}, centered about z=0.\n"
        f"- Create central hub cylinder diameter={_mm(p['hub_d'])}, length={_mm(p['hub_l'])}, concentric and centered at z=0.\n"
        "- Union shaft and hub.\n\n"
        "Subtractive feature:\n"
        f"- Cut one longitudinal keyway slot parallel to Z, width={_mm(p['key_w'])}, radial depth={_mm(p['key_d'])}.\n\n"
    ) + _footer_l3()


def _l2_counterbore_countersink_plate(p: Dict[str, Any]) -> str:
    return _header_l2("Plate with counterbore and countersink hole patterns") + (
        "Geometry:\n"
        f"- Plate: X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}.\n"
        f"- Two counterbore holes: through d={_mm(p['cbore_d'])}, head d={_mm(p['cbore_head_d'])}, head depth={_mm(p['cbore_head_h'])}.\n"
        f"- Two countersink holes: through d={_mm(p['csk_d'])}, head d={_mm(p['csk_head_d'])}, angle={_mm(p['csk_angle'])} deg.\n"
        f"- One centered through hole: diameter={_mm(p['through_d'])}.\n"
    ) + _footer_l2()


def _l3_counterbore_countersink_plate(p: Dict[str, Any]) -> str:
    lx, ly = float(p["lx"]), float(p["ly"])
    px, py = lx * 0.26, ly * 0.24
    return _header_l3(
        "Rectangular plate with mixed standard hole features",
        "Origin at center of bottom face.",
    ) + (
        "Base geometry:\n"
        f"- Create plate X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}, bottom at z=0.\n\n"
        "Hole features from top face:\n"
        f"- Counterbore holes at (±{_mm(px)}, -{_mm(py)}): through d={_mm(p['cbore_d'])}, "
        f"counterbore d={_mm(p['cbore_head_d'])}, counterbore depth={_mm(p['cbore_head_h'])}.\n"
        f"- Countersink holes at (±{_mm(px)}, +{_mm(py)}): through d={_mm(p['csk_d'])}, "
        f"countersink d={_mm(p['csk_head_d'])}, angle={_mm(p['csk_angle'])} deg.\n"
        f"- One centered through hole diameter={_mm(p['through_d'])}.\n\n"
    ) + _footer_l3()


def _l2_ribbed_bracket(p: Dict[str, Any]) -> str:
    return _header_l2("Ribbed structural bracket") + (
        "Geometry:\n"
        f"- Base plate: X={_mm(p['base_l'])}, Y={_mm(p['base_w'])}, Z={_mm(p['base_t'])}.\n"
        f"- Vertical wall near one edge: thickness={_mm(p['wall_t'])}, height={_mm(p['wall_h'])}.\n"
        f"- Two reinforcing ribs, thickness={_mm(p['rib_t'])}.\n"
        f"- Two top mounting holes, diameter={_mm(p['hole_d'])}.\n"
    ) + _footer_l2()


def _l3_ribbed_bracket(p: Dict[str, Any]) -> str:
    bl, bw = float(p["base_l"]), float(p["base_w"])
    return _header_l3(
        "Bracket with base, side wall, and dual reinforcement ribs",
        "Origin at center of base bottom face.",
    ) + (
        "Base/additive geometry:\n"
        f"- Base plate: X={_mm(p['base_l'])}, Y={_mm(p['base_w'])}, Z={_mm(p['base_t'])}, bottom at z=0.\n"
        f"- Add one full-width wall on the -X side, thickness={_mm(p['wall_t'])}, height={_mm(p['wall_h'])}.\n"
        f"- Add two triangular ribs, thickness={_mm(p['rib_t'])}, centered near y=±{_mm(bw*0.24)}.\n\n"
        "Subtractive features:\n"
        f"- Add two top holes diameter={_mm(p['hole_d'])} at x=±{_mm(bl*0.2)}, y=0.\n\n"
    ) + _footer_l3()


def _l2_shell_enclosure(p: Dict[str, Any]) -> str:
    return _header_l2("Shelled enclosure with bosses and side cutout") + (
        "Geometry:\n"
        f"- Outer body: X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}.\n"
        f"- Shell wall thickness={_mm(p['wall'])}, top open.\n"
        f"- Four internal bosses: OD={_mm(p['boss_od'])}, height={_mm(p['boss_h'])}, with boss holes d={_mm(p['boss_hole'])}.\n"
        f"- One side rectangular cutout: width={_mm(p['side_cut_w'])}, height={_mm(p['side_cut_h'])}.\n"
    ) + _footer_l2()


def _l3_shell_enclosure(p: Dict[str, Any]) -> str:
    lx, ly = float(p["lx"]), float(p["ly"])
    px, py = lx * 0.30, ly * 0.26
    return _header_l3(
        "Open-top shelled enclosure with internal standoff bosses",
        "Origin at center of outer bottom face.",
    ) + (
        "Base geometry:\n"
        f"- Create outer box X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])} then shell from top with wall={_mm(p['wall'])}.\n\n"
        "Additive internal features:\n"
        f"- Add four bosses (OD={_mm(p['boss_od'])}, height={_mm(p['boss_h'])}) at XY = (±{_mm(px)}, ±{_mm(py)}).\n\n"
        "Subtractive features:\n"
        f"- Cut boss holes diameter={_mm(p['boss_hole'])} through each boss.\n"
        f"- Cut one side window on +X side: width={_mm(p['side_cut_w'])}, height={_mm(p['side_cut_h'])}.\n\n"
    ) + _footer_l3()


def _l2_loft_transition(p: Dict[str, Any]) -> str:
    return _header_l2("Loft transition from circular to rectangular profile") + (
        "Geometry:\n"
        f"- Base profile: circle radius={_mm(p['base_radius'])}.\n"
        f"- Top profile at height={_mm(p['height'])}: rectangle X={_mm(p['top_rect_x'])}, Y={_mm(p['top_rect_y'])}.\n"
        "- Loft between profiles into one solid.\n"
    ) + _footer_l2()


def _l3_loft_transition(p: Dict[str, Any]) -> str:
    return _header_l3(
        "Single solid loft transition (round base to rectangular top)",
        "Origin at center of circular base profile on z=0.",
    ) + (
        "Profiles:\n"
        f"- Base at z=0: circle radius={_mm(p['base_radius'])}.\n"
        f"- Top at z={_mm(p['height'])}: centered rectangle X={_mm(p['top_rect_x'])}, Y={_mm(p['top_rect_y'])}.\n\n"
        "Construction:\n"
        "- Loft directly between these two profiles to create one connected solid.\n\n"
    ) + _footer_l3()


def _l2_sweep_path_part(p: Dict[str, Any]) -> str:
    return _header_l2("Swept tubular path part") + (
        "Geometry:\n"
        f"- Path span={_mm(p['span'])}, rise magnitude={_mm(p['rise'])} using smooth arc segments.\n"
        f"- Sweep a circular profile radius={_mm(p['tube_r'])} along this path.\n"
    ) + _footer_l2()


def _l3_sweep_path_part(p: Dict[str, Any]) -> str:
    span, rise = float(p["span"]), float(p["rise"])
    return _header_l3(
        "Arc-path swept tube/handle",
        "Origin at path start; path primarily in XZ.",
    ) + (
        "Path definition:\n"
        f"- Build a smooth two-arc wire from (0,0) to ({_mm(span)},0) in XZ.\n"
        f"- First arc rises to approximately z={_mm(rise)}, second arc drops to approximately z={_mm(-rise)} and returns to z=0.\n\n"
        "Sweep:\n"
        f"- Sweep a circular section radius={_mm(p['tube_r'])} along the full path.\n"
        "- Keep one connected solid with no breaks.\n\n"
    ) + _footer_l3()


def _l2_drafted_part(p: Dict[str, Any]) -> str:
    return _header_l2("Drafted prismatic part with central hole") + (
        "Geometry:\n"
        f"- Drafted extrusion from rectangle X={_mm(p['lx'])}, Y={_mm(p['ly'])} to height={_mm(p['h'])}.\n"
        f"- Draft angle={_mm(p['draft_deg'])} degrees.\n"
        f"- One centered through hole diameter={_mm(p['hole_d'])}.\n"
    ) + _footer_l2()


def _l3_drafted_part(p: Dict[str, Any]) -> str:
    return _header_l3(
        "Drafted block-style part with top-drilled center hole",
        "Origin at center of bottom face.",
    ) + (
        "Base geometry:\n"
        f"- Start with rectangle X={_mm(p['lx'])}, Y={_mm(p['ly'])} on XY plane.\n"
        f"- Extrude to height={_mm(p['h'])} using taper/draft angle={_mm(p['draft_deg'])} deg.\n\n"
        "Subtractive feature:\n"
        f"- Add centered through hole diameter={_mm(p['hole_d'])}, axis=Z.\n\n"
    ) + _footer_l3()


def _l2_helical_features(p: Dict[str, Any]) -> str:
    return _header_l2("Shaft with helical flight feature") + (
        "Geometry:\n"
        f"- Core shaft: diameter={_mm(p['core_d'])}, height={_mm(p['height'])}.\n"
        f"- Add external helical flight/rib: pitch={_mm(p['pitch'])}, radial center={_mm(p['radius'])}, "
        f"flight thickness={_mm(p['flight_t'])}, flight height={_mm(p['flight_h'])}.\n"
    ) + _footer_l2()


def _l3_helical_features(p: Dict[str, Any]) -> str:
    return _header_l3(
        "Helical-featured shaft (auger/thread-like)",
        "Origin at center of bottom face; axis along Z.",
    ) + (
        "Base geometry:\n"
        f"- Create core cylinder diameter={_mm(p['core_d'])}, height={_mm(p['height'])}.\n\n"
        "Helical feature:\n"
        f"- Add one continuous external helical flight with pitch={_mm(p['pitch'])}.\n"
        f"- Flight center radius={_mm(p['radius'])}, section thickness={_mm(p['flight_t'])}, section height={_mm(p['flight_h'])}.\n"
        f"- Total twist angle approximately {_mm(p['turns_deg'])} degrees over full height.\n\n"
    ) + _footer_l3()


def _l2_chamfer_fillet_stress_cases(p: Dict[str, Any]) -> str:
    return _header_l2("Edge-treatment stress test block") + (
        "Geometry:\n"
        f"- Base block: X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}.\n"
        f"- Four corner-region through holes diameter={_mm(p['hole_d'])}.\n"
        f"- Apply vertical-edge fillets radius={_mm(p['fillet_r'])}.\n"
        f"- Apply top and bottom chamfers (top={_mm(p['chamfer'])}).\n"
    ) + _footer_l2()


def _l3_chamfer_fillet_stress_cases(p: Dict[str, Any]) -> str:
    lx, ly = float(p["lx"]), float(p["ly"])
    px, py = lx * 0.30, ly * 0.30
    return _header_l3(
        "Prismatic part with dense fillet/chamfer edge operations",
        "Origin at center of bottom face.",
    ) + (
        "Base geometry:\n"
        f"- Create block X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}.\n\n"
        "Subtractive features:\n"
        f"- Add four vertical through holes diameter={_mm(p['hole_d'])} at (±{_mm(px)}, ±{_mm(py)}).\n\n"
        "Edge treatments:\n"
        f"- Apply fillet radius={_mm(p['fillet_r'])} on all vertical edges.\n"
        f"- Apply chamfer={_mm(p['chamfer'])} on top perimeter edges.\n"
        f"- Apply chamfer={_mm(float(p['chamfer'])*0.7)} on bottom perimeter edges.\n\n"
    ) + _footer_l3()


def _l2_off_axis_features(p: Dict[str, Any]) -> str:
    return _header_l2("Part with off-axis transformed features") + (
        "Geometry:\n"
        f"- Base block: X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}.\n"
        f"- Add tilted boss: diameter={_mm(p['boss_d'])}, height={_mm(p['boss_h'])}, tilt={_mm(p['boss_tilt_deg'])} deg.\n"
        f"- Add tilted side hole: diameter={_mm(p['hole_d'])}, tilt={_mm(p['hole_tilt_deg'])} deg.\n"
    ) + _footer_l2()


def _l3_off_axis_features(p: Dict[str, Any]) -> str:
    lx, ly = float(p["lx"]), float(p["ly"])
    return _header_l3(
        "Block with non-orthogonal workplane features (tilted boss + tilted drill)",
        "Origin at center of bottom face.",
    ) + (
        "Base geometry:\n"
        f"- Create block X={_mm(p['lx'])}, Y={_mm(p['ly'])}, Z={_mm(p['lz'])}, bottom at z=0.\n\n"
        "Off-axis additive feature:\n"
        f"- On top face, at XY=({_mm(-lx*0.15)}, {_mm(ly*0.10)}), create a circular boss diameter={_mm(p['boss_d'])}, "
        f"height={_mm(p['boss_h'])}, with local workplane rotated about Y by {_mm(p['boss_tilt_deg'])} deg.\n\n"
        "Off-axis subtractive feature:\n"
        f"- From +Y side face, drill hole diameter={_mm(p['hole_d'])} using a local workplane rotated about X by {_mm(p['hole_tilt_deg'])} deg.\n\n"
    ) + _footer_l3()


PROMPT_BUILDERS: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], str]]] = {
    "box_hole": (_l2_box_hole, _l3_box_hole),
    "stepped_shaft": (_l2_stepped_shaft, _l3_stepped_shaft),
    "flange": (_l2_flange, _l3_flange),
    "ring_spacer": (_l2_ring_spacer, _l3_ring_spacer),
    "slotted_plate": (_l2_slotted_plate, _l3_slotted_plate),
    "l_bracket": (_l2_l_bracket, _l3_l_bracket),
    "pillow_block": (_l2_pillow_block, _l3_pillow_block),
    "pulley": (_l2_pulley, _l3_pulley),
    "u_channel": (_l2_u_channel, _l3_u_channel),
    "manifold_block": (_l2_manifold_block, _l3_manifold_block),
    "threaded_parts_male": (_l2_threaded_parts_male, _l3_threaded_parts_male),
    "threaded_parts_female": (_l2_threaded_parts_female, _l3_threaded_parts_female),
    "keyed_shaft_hub": (_l2_keyed_shaft_hub, _l3_keyed_shaft_hub),
    "counterbore_countersink_plate": (_l2_counterbore_countersink_plate, _l3_counterbore_countersink_plate),
    "ribbed_bracket": (_l2_ribbed_bracket, _l3_ribbed_bracket),
    "shell_enclosure": (_l2_shell_enclosure, _l3_shell_enclosure),
    "loft_transition": (_l2_loft_transition, _l3_loft_transition),
    "sweep_path_part": (_l2_sweep_path_part, _l3_sweep_path_part),
    "drafted_part": (_l2_drafted_part, _l3_drafted_part),
    "helical_features": (_l2_helical_features, _l3_helical_features),
    "chamfer_fillet_stress_cases": (_l2_chamfer_fillet_stress_cases, _l3_chamfer_fillet_stress_cases),
    "off_axis_features": (_l2_off_axis_features, _l3_off_axis_features),
}


def generate_prompts(manifest_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "golden_prompts_l2_l3.csv"
    out_jsonl = output_dir / "golden_prompts_l2_l3.jsonl"

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records: List[Dict[str, str]] = []
    for row in rows:
        family = row["family"]
        part_id = row["part_id"]
        step_path = row["step_path"]
        params = json.loads(row["params_json"])

        if family not in PROMPT_BUILDERS:
            raise ValueError(f"No prompt builder for family '{family}' (part_id={part_id})")

        l2_builder, l3_builder = PROMPT_BUILDERS[family]
        prompt_l2 = l2_builder(params).strip()
        prompt_l3 = l3_builder(params).strip()

        records.append(
            {
                "part_id": part_id,
                "family": family,
                "step_path": step_path,
                "prompt_l2": prompt_l2,
                "prompt_l3": prompt_l3,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["part_id", "family", "step_path", "prompt_l2", "prompt_l3"],
        )
        writer.writeheader()
        writer.writerows(records)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(
                json.dumps(
                    {
                        "part_id": row["part_id"],
                        "family": row["family"],
                        "step_path": row["step_path"],
                        "prompts": {"L2": row["prompt_l2"], "L3": row["prompt_l3"]},
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    return out_csv, out_jsonl


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate L2/L3 golden prompts for each reference STEP part.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest CSV containing part_id/family/step_path/params_json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "golden_prompts",
        help="Directory where prompt dataset files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    out_csv, out_jsonl = generate_prompts(manifest_path=args.manifest, output_dir=args.output_dir)
    print(f"wrote_csv={out_csv}")
    print(f"wrote_jsonl={out_jsonl}")


if __name__ == "__main__":
    main()

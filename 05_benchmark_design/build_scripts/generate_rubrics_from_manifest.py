from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Keep this aligned with metric_engine.DEFAULT_THRESHOLDS in STEPScore.
CORE_THRESHOLDS_44: Dict[str, float] = {
    "valid_cad_rate": 1.0,
    "alignment_quality_icp_fitness": 0.30,
    "alignment_inlier_rmse_mm": 1.00,
    "chamfer_distance_mm": 1.00,
    "edge_chamfer_mm": 0.80,
    "hausdorff_95p_mm": 1.50,
    "hausdorff_99p_mm": 2.50,
    "point_to_surface_mean_mm": 0.80,
    "point_to_surface_max_mm": 4.00,
    "volume_diff_percent": 2.00,
    "signed_volume_diff_percent": 2.00,
    "surface_area_diff_percent": 3.00,
    "bbox_error_max_mm": 1.00,
    "bbox_error_axis_x_mm": 1.00,
    "bbox_error_axis_y_mm": 1.00,
    "bbox_error_axis_z_mm": 1.00,
    "obb_error_max_mm": 1.50,
    "centroid_offset_mm": 1.00,
    "inertia_tensor_error": 0.10,
    "mass_properties_error": 0.10,
    "component_count_match": 1.0,
    "watertight_manifold_pass": 1.0,
    "self_intersection_count": 0.0,
    "euler_genus_match": 1.0,
    "void_hole_count_match": 1.0,
    "feature_count_match": 0.90,
    "critical_dimension_error_mm": 0.50,
    "tolerance_band_pass_rate": 0.90,
    "feature_edge_distance_mm": 0.80,
    "normal_consistency": 0.95,
    "normal_angle_error_deg": 12.0,
    "curvature_distribution_error": 0.15,
    "cross_section_iou": 0.90,
    "slice_contour_distance_mm": 0.80,
    "voxel_iou": 0.88,
    "occupancy_precision": 0.90,
    "occupancy_recall": 0.90,
    "occupancy_f1": 0.90,
    "emd_distance": 0.02,
    "silhouette_iou": 0.92,
    "render_ssim": 0.90,
    "render_lpips": 0.15,
    "registration_failure_rate": 0.05,
    "composite_weighted_score": 0.85,
}

CORE_DIRECTIONS_44: Dict[str, str] = {
    "valid_cad_rate": "higher_better",
    "alignment_quality_icp_fitness": "higher_better",
    "alignment_inlier_rmse_mm": "lower_better",
    "chamfer_distance_mm": "lower_better",
    "edge_chamfer_mm": "lower_better",
    "hausdorff_95p_mm": "lower_better",
    "hausdorff_99p_mm": "lower_better",
    "point_to_surface_mean_mm": "lower_better",
    "point_to_surface_max_mm": "lower_better",
    "volume_diff_percent": "lower_better",
    "signed_volume_diff_percent": "lower_better",
    "surface_area_diff_percent": "lower_better",
    "bbox_error_max_mm": "lower_better",
    "bbox_error_axis_x_mm": "lower_better",
    "bbox_error_axis_y_mm": "lower_better",
    "bbox_error_axis_z_mm": "lower_better",
    "obb_error_max_mm": "lower_better",
    "centroid_offset_mm": "lower_better",
    "inertia_tensor_error": "lower_better",
    "mass_properties_error": "lower_better",
    "component_count_match": "higher_better",
    "watertight_manifold_pass": "higher_better",
    "self_intersection_count": "lower_better",
    "euler_genus_match": "higher_better",
    "void_hole_count_match": "higher_better",
    "feature_count_match": "higher_better",
    "critical_dimension_error_mm": "lower_better",
    "tolerance_band_pass_rate": "higher_better",
    "feature_edge_distance_mm": "lower_better",
    "normal_consistency": "higher_better",
    "normal_angle_error_deg": "lower_better",
    "curvature_distribution_error": "lower_better",
    "cross_section_iou": "higher_better",
    "slice_contour_distance_mm": "lower_better",
    "voxel_iou": "higher_better",
    "occupancy_precision": "higher_better",
    "occupancy_recall": "higher_better",
    "occupancy_f1": "higher_better",
    "emd_distance": "lower_better",
    "silhouette_iou": "higher_better",
    "render_ssim": "higher_better",
    "render_lpips": "lower_better",
    "registration_failure_rate": "lower_better",
    "composite_weighted_score": "higher_better",
}

BLOCKER_CORE_METRICS = {
    "valid_cad_rate",
    "component_count_match",
    "watertight_manifold_pass",
    "self_intersection_count",
}

MINOR_CORE_METRICS = {
    "normal_consistency",
    "normal_angle_error_deg",
    "curvature_distribution_error",
    "cross_section_iou",
    "slice_contour_distance_mm",
    "silhouette_iou",
    "render_ssim",
    "render_lpips",
}


def _metric_category(metric_name: str) -> str:
    if metric_name in {"valid_cad_rate"}:
        return "validity"
    if metric_name.startswith("alignment_") or metric_name == "registration_failure_rate":
        return "alignment"
    if metric_name in {
        "component_count_match",
        "watertight_manifold_pass",
        "self_intersection_count",
        "euler_genus_match",
        "void_hole_count_match",
        "feature_count_match",
    }:
        return "topology"
    if metric_name.startswith("bbox_") or metric_name.startswith("obb_") or metric_name in {
        "critical_dimension_error_mm",
        "tolerance_band_pass_rate",
        "feature_edge_distance_mm",
        "centroid_offset_mm",
    }:
        return "dimensions"
    if metric_name in {"volume_diff_percent", "signed_volume_diff_percent", "surface_area_diff_percent", "inertia_tensor_error", "mass_properties_error"}:
        return "mass_properties"
    if metric_name in {"normal_consistency", "normal_angle_error_deg", "curvature_distribution_error"}:
        return "surface_orientation"
    if metric_name in {"voxel_iou", "occupancy_precision", "occupancy_recall", "occupancy_f1"}:
        return "occupancy"
    if metric_name in {"silhouette_iou", "render_ssim", "render_lpips"}:
        return "render_similarity"
    if metric_name == "composite_weighted_score":
        return "aggregate"
    return "distance"


def _metric_severity(metric_name: str) -> str:
    if metric_name in BLOCKER_CORE_METRICS:
        return "blocker"
    if metric_name in MINOR_CORE_METRICS:
        return "minor"
    return "major"


def _build_core_metric_checks() -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for idx, (name, threshold) in enumerate(CORE_THRESHOLDS_44.items(), start=1):
        direction = CORE_DIRECTIONS_44[name]
        checks.append(
            {
                "rubric_id": f"CORE_{idx:03d}",
                "category": _metric_category(name),
                "name": name,
                "measurement": "stepscore_metric_engine",
                "direction": direction,
                "threshold": float(threshold),
                "pass_condition": _pass_condition(direction, float(threshold)),
                "severity": _metric_severity(name),
            }
        )
    return checks


def _fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:.4f}".rstrip("0").rstrip(".")


def _pass_condition(direction: str, threshold: float) -> str:
    if direction == "higher_better":
        return f"value >= {_fmt(threshold)}"
    if direction == "lower_better":
        return f"value <= {_fmt(threshold)}"
    return f"value == {_fmt(threshold)}"


def _feature_summary(family: str) -> str:
    summaries = {
        "box_hole": "Rectangular block with one centered vertical through-hole.",
        "stepped_shaft": "Three coaxial shaft diameters with two shoulders.",
        "flange": "Circular flange with centered bore and equally spaced bolt holes.",
        "ring_spacer": "Hollow ring/spacer (annulus) with uniform wall.",
        "slotted_plate": "Rectangular plate with one centered through-slot.",
        "l_bracket": "L-bracket with base + web and top/side holes.",
        "pillow_block": "Block with central bore and four mounting holes.",
        "pulley": "Three-step pulley profile with centered through bore.",
        "u_channel": "U-channel with open top and two top-face holes.",
    }
    return summaries.get(family, "Single mechanical part with parametric dimensions.")


def _critical_dimension_checks(family: str, params: Dict[str, Any], tol_mm: float) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    idx = 1

    def add_dim(name: str, target: float, method: str = "direct_dimension_measurement") -> None:
        nonlocal idx
        checks.append(
            {
                "rubric_id": f"SPEC_{idx:03d}",
                "category": "critical_dimension",
                "name": name,
                "measurement": method,
                "target_mm": float(target),
                "tolerance_mm": float(tol_mm),
                "pass_condition": f"abs(value - {_fmt(target)}) <= {_fmt(tol_mm)}",
                "severity": "blocker",
            }
        )
        idx += 1

    def add_exact(name: str, target: Any, method: str = "feature_count_or_property_check") -> None:
        nonlocal idx
        checks.append(
            {
                "rubric_id": f"SPEC_{idx:03d}",
                "category": "feature_property",
                "name": name,
                "measurement": method,
                "target": target,
                "pass_condition": f"value == {target!r}",
                "severity": "major",
            }
        )
        idx += 1

    if family == "box_hole":
        add_dim("outer_length_x", params["lx"])
        add_dim("outer_length_y", params["ly"])
        add_dim("outer_height_z", params["lz"])
        add_dim("center_through_hole_diameter", params["hole_d"])
        add_exact("through_hole_count", 1)
        add_exact("primary_hole_axis", "Z")

    elif family == "stepped_shaft":
        add_dim("segment1_diameter", params["d1"])
        add_dim("segment2_diameter", params["d2"])
        add_dim("segment3_diameter", params["d3"])
        add_dim("segment1_length", params["l1"])
        add_dim("segment2_length", params["l2"])
        add_dim("segment3_length", params["l3"])
        add_dim("overall_length", float(params["l1"]) + float(params["l2"]) + float(params["l3"]))
        add_exact("coaxial_segments_count", 3)

    elif family == "flange":
        add_dim("outer_diameter", params["od"])
        add_dim("thickness", params["thickness"])
        add_dim("center_bore_diameter", params["bore_d"])
        add_dim("bolt_hole_diameter", params["bolt_d"])
        add_dim("bolt_circle_diameter", 2.0 * float(params["bolt_radius"]))
        add_exact("bolt_hole_count", int(params["bolt_count"]))
        add_exact("bolt_hole_pattern", "equal_angular_spacing")

    elif family == "ring_spacer":
        add_dim("outer_diameter", params["od"])
        add_dim("inner_diameter", params["id"])
        add_dim("height", params["h"])
        add_dim("radial_wall_thickness", (float(params["od"]) - float(params["id"])) / 2.0)

    elif family == "slotted_plate":
        add_dim("plate_length_x", params["lx"])
        add_dim("plate_width_y", params["ly"])
        add_dim("plate_thickness_z", params["lz"])
        add_dim("slot_length", params["slot_len"])
        add_dim("slot_width", params["slot_w"])
        add_exact("slot_count", 1)
        add_exact("slot_axis", "X")

    elif family == "l_bracket":
        add_dim("base_length", params["base_l"])
        add_dim("base_width", params["base_w"])
        add_dim("base_thickness", params["base_t"])
        add_dim("web_thickness", params["web_t"])
        add_dim("web_height", params["web_h"])
        add_dim("top_hole_diameter", params["top_hole_d"])
        add_dim("side_hole_diameter", params["side_hole_d"])
        add_exact("top_hole_count", 2)
        add_exact("side_hole_count", 1)

    elif family == "pillow_block":
        add_dim("body_length_x", params["lx"])
        add_dim("body_width_y", params["ly"])
        add_dim("body_height_z", params["lz"])
        add_dim("center_bore_diameter", params["bore_d"])
        add_dim("mount_hole_diameter", params["mount_d"])
        add_exact("mount_hole_count", 4)

    elif family == "pulley":
        add_dim("step1_radius", params["r1"])
        add_dim("step2_radius", params["r2"])
        add_dim("step3_radius", params["r3"])
        add_dim("step1_height", params["h1"])
        add_dim("step2_height", params["h2"])
        add_dim("step3_height", params["h3"])
        add_dim("bore_diameter", params["bore_d"])
        add_dim("total_height", float(params["h1"]) + float(params["h2"]) + float(params["h3"]))
        add_exact("coaxial_steps_count", 3)

    elif family == "u_channel":
        add_dim("outer_length_x", params["lx"])
        add_dim("outer_width_y", params["ly"])
        add_dim("outer_height_z", params["lz"])
        add_dim("wall_thickness", params["wall"])
        add_dim("floor_thickness", params["floor"])
        add_dim("top_hole_diameter", params["hole_d"])
        add_exact("top_hole_count", 2)
        add_exact("channel_top_state", "open")

    else:
        # Generic fallback for unknown family.
        for key, value in sorted(params.items()):
            if isinstance(value, (int, float)):
                add_dim(f"param_{key}", float(value))
            else:
                add_exact(f"param_{key}", value)

    return checks


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _fmt(v)
    if v is None:
        return "null"
    s = str(v)
    needs_quote = (
        s == ""
        or s.startswith((" ", "-", ":", "{", "[", "&", "*", "!", "@", "#", "`", "|", ">", "%"))
        or ":" in s
        or "#" in s
        or "\n" in s
    )
    if needs_quote:
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f"\"{s}\""
    return s


def _yaml_dump(obj: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(obj, dict):
        lines: List[str] = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_yaml_dump(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(obj)}"


def _build_rubric_row(
    row: Dict[str, str],
    source_manifest: str,
    critical_tol_mm: float,
) -> Dict[str, Any]:
    params = json.loads(row["params_json"])
    family = row["family"]
    part_id = row["part_id"]

    core_checks = _build_core_metric_checks()

    specific = _critical_dimension_checks(family=family, params=params, tol_mm=critical_tol_mm)

    return {
        "rubric_version": "1.0",
        "part_id": part_id,
        "family": family,
        "reference": {
            "step_path": row["step_path"],
            "bbox_mm": {
                "x": float(row["bbox_x_mm"]),
                "y": float(row["bbox_y_mm"]),
                "z": float(row["bbox_z_mm"]),
            },
            "volume_mm3": float(row["volume_mm3"]),
            "param_values": params,
            "design_features": _feature_summary(family),
        },
        "policy": {
            "overall_rule": "pass_if_all_blockers_pass_and_major_pass_rate_at_least_0.90",
            "critical_dimension_tolerance_mm": float(critical_tol_mm),
        },
        "checks": {
            "core_metric_checks": core_checks,
            "task_specific_checks": specific,
        },
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest": source_manifest,
            "generator_script": "generate_rubrics_from_manifest.py",
        },
    }


def _build_template() -> Dict[str, Any]:
    return {
        "rubric_version": "1.0",
        "part_id": "<part_id>",
        "family": "<family_name>",
        "reference": {
            "step_path": "<absolute_or_relative_path_to_reference_step>",
            "bbox_mm": {"x": "<x_mm>", "y": "<y_mm>", "z": "<z_mm>"},
            "volume_mm3": "<volume_mm3>",
            "param_values": {"<param_name>": "<value>"},
            "design_features": "<short_feature_description>",
        },
        "policy": {
            "overall_rule": "pass_if_all_blockers_pass_and_major_pass_rate_at_least_0.90",
            "critical_dimension_tolerance_mm": 0.25,
        },
        "checks": {
            "core_metric_checks": [
                {
                    "rubric_id": "CORE_001",
                    "category": "distance|topology|dimensions|...",
                    "name": "<metric_name>",
                    "measurement": "stepscore_metric_engine",
                    "direction": "lower_better|higher_better",
                    "threshold": "<number>",
                    "pass_condition": "<e.g. value <= threshold>",
                    "severity": "blocker|major|minor",
                }
            ],
            "task_specific_checks": [
                {
                    "rubric_id": "SPEC_001",
                    "category": "critical_dimension|feature_property",
                    "name": "<check_name>",
                    "measurement": "direct_dimension_measurement|feature_count_or_property_check",
                    "target_mm": "<number_if_dimension>",
                    "tolerance_mm": "<number_if_dimension>",
                    "target": "<value_if_exact_property>",
                    "pass_condition": "<explicit_boolean_expression>",
                    "severity": "blocker|major|minor",
                }
            ],
        },
        "metadata": {
            "generated_at_utc": "<timestamp>",
            "source_manifest": "<manifest_path>",
            "generator_script": "generate_rubrics_from_manifest.py",
        },
    }


def generate_rubrics(
    manifest_path: Path,
    output_dir: Path,
    critical_tolerance_mm: float,
) -> Tuple[int, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "rubric_template.yaml"
    index_path = output_dir / "rubrics_index.csv"

    template = _build_template()
    template_path.write_text(_yaml_dump(template) + "\n", encoding="utf-8")

    rows: List[Dict[str, str]]
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    index_rows: List[Dict[str, Any]] = []
    for row in rows:
        rubric = _build_rubric_row(
            row=row,
            source_manifest=str(manifest_path),
            critical_tol_mm=critical_tolerance_mm,
        )
        part_id = row["part_id"]
        out_path = output_dir / f"{part_id}.rubric.yaml"
        out_path.write_text(_yaml_dump(rubric) + "\n", encoding="utf-8")
        index_rows.append(
            {
                "part_id": part_id,
                "family": row["family"],
                "reference_step_path": row["step_path"],
                "rubric_path": str(out_path),
                "core_metric_checks": len(rubric["checks"]["core_metric_checks"]),
                "task_specific_checks": len(rubric["checks"]["task_specific_checks"]),
            }
        )

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "part_id",
                "family",
                "reference_step_path",
                "rubric_path",
                "core_metric_checks",
                "task_specific_checks",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    return len(rows), template_path, index_path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Auto-generate per-part rubric YAML files from manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to reference STEP manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "rubrics_generated",
        help="Directory for generated rubric files.",
    )
    parser.add_argument(
        "--critical-tolerance-mm",
        type=float,
        default=0.25,
        help="Absolute tolerance for task-specific critical dimensions in mm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    count, template_path, index_path = generate_rubrics(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        critical_tolerance_mm=float(args.critical_tolerance_mm),
    )
    print(f"generated_rubrics={count}")
    print(f"template={template_path}")
    print(f"index={index_path}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

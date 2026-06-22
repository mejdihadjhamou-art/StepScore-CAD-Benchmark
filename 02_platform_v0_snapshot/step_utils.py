from __future__ import annotations

from pathlib import Path
from typing import Optional


MESH_EXTENSIONS = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf"}
STEP_EXTENSIONS = {".step", ".stp"}
SUPPORTED_EXTENSIONS = MESH_EXTENSIONS | STEP_EXTENSIONS


class StepConversionError(RuntimeError):
    pass


def is_step_path(path: str) -> bool:
    return Path(path).suffix.lower() in STEP_EXTENSIONS


def is_supported_path(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def convert_step_to_stl(
    step_path: str,
    stl_path: str,
    linear_tolerance_mm: float = 0.05,
    angular_tolerance_deg: float = 0.1,
) -> str:
    """
    Deterministically tessellate STEP/STP into STL using CadQuery.
    """
    in_path = Path(step_path)
    out_path = Path(stl_path)

    if not in_path.exists():
        raise StepConversionError(f"STEP file not found: {step_path}")
    if in_path.suffix.lower() not in STEP_EXTENSIONS:
        raise StepConversionError(f"Not a STEP/STP file: {step_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cadquery as cq
    except Exception as exc:
        raise StepConversionError(
            "CadQuery is required for STEP conversion. Install with: pip install cadquery"
        ) from exc

    try:
        shape = cq.importers.importStep(str(in_path))
        cq.exporters.export(
            shape,
            str(out_path),
            tolerance=float(linear_tolerance_mm),
            angularTolerance=float(angular_tolerance_deg),
        )
    except Exception as exc:
        raise StepConversionError(f"Failed converting STEP to STL: {exc}") from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise StepConversionError("STEP conversion produced no STL output.")

    return str(out_path)


def ensure_mesh_path(
    input_path: str,
    run_dir: str,
    prefix: str,
    linear_tolerance_mm: float = 0.05,
    angular_tolerance_deg: float = 0.1,
) -> str:
    """
    Return a mesh path for comparison:
    - if already mesh -> return input_path
    - if STEP -> convert to run_dir/<prefix>_from_step.stl and return that path
    """
    ext = Path(input_path).suffix.lower()
    if ext in MESH_EXTENSIONS:
        return input_path
    if ext in STEP_EXTENSIONS:
        out_stl = Path(run_dir) / f"{prefix}_from_step.stl"
        return convert_step_to_stl(
            step_path=input_path,
            stl_path=str(out_stl),
            linear_tolerance_mm=linear_tolerance_mm,
            angular_tolerance_deg=angular_tolerance_deg,
        )
    raise StepConversionError(
        f"Unsupported extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


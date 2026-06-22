from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from step_utils import STEP_EXTENSIONS, StepConversionError, ensure_mesh_path


class StepQAError(RuntimeError):
    pass


MATERIAL_DENSITY_G_CM3 = {
    "steel": 7.85,
    "stainless_steel": 8.00,
    "aluminum": 2.70,
    "aluminium": 2.70,
    "titanium": 4.50,
    "brass": 8.50,
    "copper": 8.96,
    "abs": 1.04,
    "pla": 1.24,
}


def _load_mesh(input_path: str, run_dir: str) -> tuple[trimesh.Trimesh, str]:
    try:
        mesh_path = ensure_mesh_path(input_path=input_path, run_dir=run_dir, prefix="qa_input")
    except StepConversionError as exc:
        raise StepQAError(str(exc)) from exc

    loaded = trimesh.load(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh is None or mesh.is_empty:
        raise StepQAError("Could not load a valid mesh from input.")
    return mesh, mesh_path


def _component_count(mesh: trimesh.Trimesh) -> int:
    try:
        return len(mesh.split(only_watertight=False))
    except Exception:
        return 1


def _basic_facts(mesh: trimesh.Trimesh) -> Dict[str, object]:
    mins, maxs = mesh.bounds
    dims = maxs - mins
    return {
        "bbox_mm": [float(dims[0]), float(dims[1]), float(dims[2])],
        "volume_mm3": float(abs(mesh.volume)),
        "surface_area_mm2": float(mesh.area),
        "components": int(_component_count(mesh)),
        "watertight": bool(mesh.is_watertight),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "center_mass_mm": [float(v) for v in mesh.center_mass],
    }


def _extract_step_metadata(step_path: str) -> Dict[str, object]:
    text = Path(step_path).read_text(encoding="utf-8", errors="ignore")
    named_colors = sorted(
        {c.strip().lower() for c in re.findall(r"DRAUGHTING_PRE_DEFINED_COLOUR\('([^']+)'\)", text)}
    )

    rgb_vals = []
    for m in re.findall(
        r"COLOUR_RGB\([^,]*,\s*([0-9.+-Ee]+)\s*,\s*([0-9.+-Ee]+)\s*,\s*([0-9.+-Ee]+)\s*\)",
        text,
    ):
        try:
            r, g, b = float(m[0]), float(m[1]), float(m[2])
            rgb_vals.append([r, g, b])
        except Exception:
            continue

    return {
        "named_colors": named_colors,
        "rgb_colors": rgb_vals,
    }


def _volume_to_weight_kg(volume_mm3: float, density_g_cm3: float) -> float:
    volume_cm3 = volume_mm3 / 1000.0
    mass_g = volume_cm3 * density_g_cm3
    return mass_g / 1000.0


def _weight_estimates(volume_mm3: float) -> Dict[str, float]:
    return {
        name: _volume_to_weight_kg(volume_mm3, density)
        for name, density in MATERIAL_DENSITY_G_CM3.items()
    }


def _infer_material(question: str) -> Optional[str]:
    q = question.lower()
    if "stainless" in q:
        return "stainless_steel"
    if "steel" in q:
        return "steel"
    if "aluminium" in q:
        return "aluminium"
    if "aluminum" in q:
        return "aluminum"
    if "titanium" in q:
        return "titanium"
    if "brass" in q:
        return "brass"
    if "copper" in q:
        return "copper"
    if "abs" in q:
        return "abs"
    if "pla" in q:
        return "pla"
    return None


def _normalize_text(value: str) -> str:
    txt = value.lower().strip()
    txt = re.sub(r"[^a-z0-9.\s-]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _extract_first_number(value: str) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _grade_against_gtfa(
    predicted_answer: str,
    gtfa: str,
    numeric_tolerance_percent: float = 2.0,
) -> Dict[str, object]:
    gtfa_num = _extract_first_number(gtfa)
    pred_num = _extract_first_number(predicted_answer)

    if gtfa_num is not None and pred_num is not None:
        abs_error = abs(pred_num - gtfa_num)
        allowed_abs = abs(gtfa_num) * max(float(numeric_tolerance_percent), 0.0) / 100.0
        passed = abs_error <= allowed_abs
        rel_error_percent = 0.0 if gtfa_num == 0 else abs_error / abs(gtfa_num) * 100.0
        return {
            "enabled": True,
            "mode": "numeric",
            "gtfa": gtfa,
            "predicted_answer": predicted_answer,
            "gtfa_numeric": gtfa_num,
            "predicted_numeric": pred_num,
            "numeric_tolerance_percent": float(numeric_tolerance_percent),
            "abs_error": abs_error,
            "rel_error_percent": rel_error_percent,
            "passed": bool(passed),
        }

    norm_gtfa = _normalize_text(gtfa)
    norm_pred = _normalize_text(predicted_answer)
    exact = norm_gtfa == norm_pred
    contains = bool(norm_gtfa) and norm_gtfa in norm_pred
    similarity = SequenceMatcher(None, norm_pred, norm_gtfa).ratio()
    passed = exact or contains or similarity >= 0.9

    return {
        "enabled": True,
        "mode": "text",
        "gtfa": gtfa,
        "predicted_answer": predicted_answer,
        "normalized_gtfa": norm_gtfa,
        "normalized_predicted": norm_pred,
        "exact_match": bool(exact),
        "contains_match": bool(contains),
        "similarity": float(similarity),
        "passed": bool(passed),
    }


def _call_openai_qa(
    model: str,
    question: str,
    facts: Dict[str, object],
    step_meta: Dict[str, object],
    weights: Dict[str, float],
    api_key: Optional[str],
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise StepQAError("openai package not installed. Install with: pip install openai") from exc

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise StepQAError("Missing OPENAI_API_KEY (or provide API key in dashboard).")

    client = OpenAI(api_key=key)
    system = (
        "You are a CAD geometry analyst. "
        "Answer the user question using only the provided extracted STEP/mesh facts. "
        "If information is unavailable (for example visual appearance without color metadata), "
        "say that clearly and do not invent values."
    )
    payload = {
        "question": question,
        "facts": facts,
        "step_metadata": step_meta,
        "weight_estimates_kg": weights,
        "units": {"length": "mm", "volume": "mm^3", "mass": "kg"},
    }
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(payload)},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _call_anthropic_qa(
    model: str,
    question: str,
    facts: Dict[str, object],
    step_meta: Dict[str, object],
    weights: Dict[str, float],
    api_key: Optional[str],
) -> str:
    try:
        import anthropic
    except Exception as exc:
        raise StepQAError("anthropic package not installed. Install with: pip install anthropic") from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise StepQAError("Missing ANTHROPIC_API_KEY (or provide API key in dashboard).")

    client = anthropic.Anthropic(api_key=key)
    system = (
        "You are a CAD geometry analyst. "
        "Answer the user question using only the provided extracted STEP/mesh facts. "
        "If information is unavailable (for example visual appearance without color metadata), "
        "say that clearly and do not invent values."
    )
    payload = {
        "question": question,
        "facts": facts,
        "step_metadata": step_meta,
        "weight_estimates_kg": weights,
        "units": {"length": "mm", "volume": "mm^3", "mass": "kg"},
    }
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0.1,
        system=system,
        messages=[{"role": "user", "content": str(payload)}],
    )
    parts = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _call_llm_qa(
    provider: str,
    model: str,
    question: str,
    facts: Dict[str, object],
    step_meta: Dict[str, object],
    weights: Dict[str, float],
    api_key: Optional[str],
) -> str:
    p = provider.strip().lower()
    if p == "openai":
        return _call_openai_qa(
            model=model,
            question=question,
            facts=facts,
            step_meta=step_meta,
            weights=weights,
            api_key=api_key,
        )
    if p == "anthropic":
        return _call_anthropic_qa(
            model=model,
            question=question,
            facts=facts,
            step_meta=step_meta,
            weights=weights,
            api_key=api_key,
        )
    raise StepQAError(f"Unsupported provider '{provider}'. Use 'openai' or 'anthropic'.")


def _estimate_gear_teeth(mesh: trimesh.Trimesh) -> Dict[str, object]:
    bounds = mesh.bounds
    z_min = float(bounds[0, 2])
    z_max = float(bounds[1, 2])
    z_span = z_max - z_min
    if z_span <= 0:
        return {"ok": False, "reason": "invalid_z_span"}

    z_levels = np.linspace(z_min + 0.2 * z_span, z_max - 0.2 * z_span, 5)
    counts = []

    for z in z_levels:
        section = mesh.section(plane_origin=[0.0, 0.0, float(z)], plane_normal=[0.0, 0.0, 1.0])
        if section is None:
            continue
        loops = section.discrete
        if not loops:
            continue

        loop = max(loops, key=lambda arr: len(arr))
        pts = np.asarray(loop)
        if pts.ndim != 2 or pts.shape[0] < 80 or pts.shape[1] < 2:
            continue

        xy = pts[:, :2]
        center = np.mean(xy, axis=0)
        rel = xy - center
        theta = np.arctan2(rel[:, 1], rel[:, 0])
        radius = np.linalg.norm(rel, axis=1)

        order = np.argsort(theta)
        theta = np.unwrap(theta[order])
        radius = radius[order]

        theta_unique, unique_idx = np.unique(np.round(theta, 6), return_index=True)
        radius_unique = radius[unique_idx]
        if len(theta_unique) < 60:
            continue

        samples = 1024
        theta_uniform = np.linspace(theta_unique.min(), theta_unique.max(), samples, endpoint=False)
        radius_uniform = np.interp(theta_uniform, theta_unique, radius_unique)

        smooth = gaussian_filter1d(radius_uniform, sigma=3.0, mode="wrap")
        baseline = gaussian_filter1d(smooth, sigma=15.0, mode="wrap")
        signal = smooth - baseline

        prominence = max(float(np.std(signal) * 0.7), float(np.mean(radius_uniform) * 0.004))
        min_distance = max(6, samples // 220)

        peaks, _ = find_peaks(signal, prominence=prominence, distance=min_distance)
        peak_count = int(len(peaks))
        if peak_count >= 4:
            counts.append(peak_count)

    if not counts:
        return {"ok": False, "reason": "no_valid_sections"}

    counts_arr = np.asarray(counts, dtype=float)
    estimate = int(np.median(counts_arr))
    spread = float(np.std(counts_arr) / max(estimate, 1))
    confidence = max(0.0, min(1.0, 1.0 - spread * 2.0))

    return {
        "ok": True,
        "estimate": estimate,
        "slice_counts": [int(v) for v in counts],
        "confidence": confidence,
    }


def answer_step_question(
    input_path: str,
    question: str,
    run_dir: str,
    use_ai: bool = False,
    provider: str = "openai",
    model: str = "gpt-4.1-mini-2025-04-14",
    api_key: Optional[str] = None,
    gtfa: str = "",
    numeric_tolerance_percent: float = 2.0,
) -> Dict[str, object]:
    path = Path(input_path)
    if not path.exists():
        raise StepQAError(f"Input file not found: {input_path}")
    if path.suffix.lower() not in STEP_EXTENSIONS:
        raise StepQAError("STEP Q&A mode requires a .step or .stp file.")
    if not question.strip():
        raise StepQAError("Please provide a question.")

    mesh, mesh_path = _load_mesh(input_path=input_path, run_dir=run_dir)
    facts = _basic_facts(mesh)
    step_meta = _extract_step_metadata(str(path))
    weights = _weight_estimates(float(facts["volume_mm3"]))
    q = question.lower()

    answer: str
    extra: Dict[str, object] = {}

    if "color" in q or "colour" in q:
        names = step_meta.get("named_colors", [])
        rgbs = step_meta.get("rgb_colors", [])
        if names:
            answer = f"STEP color metadata found: {', '.join(names)}."
        elif rgbs:
            first = rgbs[0]
            answer = (
                "STEP RGB color metadata found. "
                f"Example RGB: ({first[0]:.3f}, {first[1]:.3f}, {first[2]:.3f})."
            )
        else:
            answer = (
                "No color metadata detected in this STEP file, so color cannot be determined reliably."
            )
    elif "weight" in q or "mass" in q:
        material = _infer_material(question)
        if material and material in weights:
            answer = (
                f"Estimated mass if made of {material.replace('_', ' ')}: "
                f"{weights[material]:.4f} kg."
            )
        else:
            answer = (
                "Material not specified clearly. Example estimates: "
                f"steel={weights['steel']:.4f} kg, aluminum={weights['aluminum']:.4f} kg, "
                f"titanium={weights['titanium']:.4f} kg."
            )
    elif "teeth" in q and "gear" in q:
        teeth = _estimate_gear_teeth(mesh)
        extra["gear_teeth_estimate"] = teeth
        if teeth.get("ok"):
            answer = (
                f"Estimated gear tooth count: {teeth['estimate']} "
                f"(confidence {teeth['confidence']:.2f}, slice counts {teeth['slice_counts']})."
            )
        else:
            answer = "Could not estimate gear tooth count reliably from this geometry."
    elif "volume" in q:
        answer = f"Volume: {facts['volume_mm3']:.3f} mm^3."
    elif "surface" in q and "area" in q:
        answer = f"Surface area: {facts['surface_area_mm2']:.3f} mm^2."
    elif "dimension" in q or "size" in q or "bbox" in q:
        bx, by, bz = facts["bbox_mm"]
        answer = f"Bounding box dimensions (X, Y, Z): {bx:.3f}, {by:.3f}, {bz:.3f} mm."
    elif "component" in q or "solid" in q:
        answer = f"Connected components: {facts['components']}."
    elif "watertight" in q or "manifold" in q:
        answer = f"Watertight: {'yes' if facts['watertight'] else 'no'}."
    elif use_ai:
        answer = _call_llm_qa(
            provider=provider,
            model=model,
            question=question,
            facts=facts,
            step_meta=step_meta,
            weights=weights,
            api_key=api_key,
        )
    else:
        bx, by, bz = facts["bbox_mm"]
        answer = (
            "Extracted model info: "
            f"bbox=({bx:.2f}, {by:.2f}, {bz:.2f}) mm, "
            f"volume={facts['volume_mm3']:.2f} mm^3, "
            f"surface_area={facts['surface_area_mm2']:.2f} mm^2, "
            f"components={facts['components']}."
        )

    gtfa_eval = {"enabled": False}
    if gtfa.strip():
        gtfa_eval = _grade_against_gtfa(
            predicted_answer=answer,
            gtfa=gtfa.strip(),
            numeric_tolerance_percent=float(numeric_tolerance_percent),
        )

    return {
        "ok": True,
        "question": question,
        "answer": answer,
        "input_step_path": str(path),
        "mesh_path": mesh_path,
        "facts": facts,
        "step_metadata": step_meta,
        "weight_estimates_kg": weights,
        "qa_mode": {
            "use_ai": bool(use_ai),
            "provider": provider if use_ai else None,
            "model": model if use_ai else None,
        },
        "gtfa_evaluation": gtfa_eval,
        "analysis": extra,
    }

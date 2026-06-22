#!/usr/bin/env python3
"""
Advanced geometry metrics for reference-vs-generated mesh comparison.

This module computes practical implementations of:
1) Surface area error (%)
2) Normal consistency
3) Symmetry error
4) Curvature distribution distance (JS)
5) Slice profile IoU
6) Signed distance p95
7) Signed distance bias
8) Feature-size mean error (%)
9) Feature-size max error (%)
10) Topology invariants match
11) Thin-feature preservation
12) Principal axis angle error (deg)
13) Principal eigenvalue ratio error (%)

Dependencies:
- numpy
- scipy
- trimesh
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "surface_area_error_percent_max": 3.0,
    "normal_consistency_min": 0.95,
    "symmetry_error_mm_max": 1.0,
    "curvature_js_max": 0.10,
    "slice_iou_mean_min": 0.92,
    "signed_distance_p95_mm_max": 1.0,
    "signed_distance_bias_mm_abs_max": 0.25,
    "feature_size_error_mean_percent_max": 2.0,
    "feature_size_error_max_percent_max": 5.0,
    "topology_exact_match_required": True,
    "thin_feature_preservation_min": 0.90,
    "principal_axis_angle_deg_max": 5.0,
    "principal_eigenvalue_ratio_error_percent_max": 5.0,
    "thin_feature_percentile": 0.15,
    "thin_feature_error_tolerance_percent": 20.0,
    "sample_points": 30000,
    "slice_count_per_axis": 3,
    "slice_thickness_mm": 0.75,
    "slice_grid_res": 128,
}


@dataclass
class Metric:
    name: str
    value: Any
    threshold: Any
    direction: str
    passed: bool
    details: Optional[Dict[str, Any]] = None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as Trimesh: {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh is empty: {path}")
    return mesh


def _sample_points_and_normals(
    mesh: trimesh.Trimesh, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    points, face_idx = trimesh.sample.sample_surface_even(mesh, n)
    normals = mesh.face_normals[face_idx]
    return points, normals


def _align_generated_to_reference(
    ref_mesh: trimesh.Trimesh, gen_mesh: trimesh.Trimesh, n_points: int
) -> trimesh.Trimesh:
    ref_pts, _ = _sample_points_and_normals(ref_mesh, min(12000, n_points))
    gen_pts, _ = _sample_points_and_normals(gen_mesh, min(12000, n_points))
    matrix, _, _ = trimesh.registration.icp(
        gen_pts, ref_pts, scale=False, reflection=False, max_iterations=50
    )
    aligned = gen_mesh.copy()
    aligned.apply_transform(matrix)
    return aligned


def metric_surface_area_error(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Metric:
    ref_area = max(_safe_float(ref.area), 1e-12)
    gen_area = _safe_float(gen.area)
    error_pct = abs(gen_area - ref_area) / ref_area * 100.0
    thr = _safe_float(t["surface_area_error_percent_max"])
    return Metric(
        name="surface_area_error_percent",
        value=error_pct,
        threshold=thr,
        direction="lower_better",
        passed=error_pct <= thr,
        details={"reference_area": ref_area, "generated_area": gen_area},
    )


def metric_normal_consistency(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Metric:
    n = int(t["sample_points"])
    ref_pts, ref_n = _sample_points_and_normals(ref, n)
    gen_pts, gen_n = _sample_points_and_normals(gen, n)
    tree = cKDTree(ref_pts)
    _, idx = tree.query(gen_pts, k=1)
    dots = np.abs(np.einsum("ij,ij->i", gen_n, ref_n[idx]))
    score = float(np.clip(np.mean(dots), 0.0, 1.0))
    thr = _safe_float(t["normal_consistency_min"])
    return Metric(
        name="normal_consistency",
        value=score,
        threshold=thr,
        direction="higher_better",
        passed=score >= thr,
    )


def _reflect_points(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    dist = np.dot(points, n) - offset
    return points - 2.0 * dist[:, None] * n[None, :]


def metric_symmetry_error(
    mesh: trimesh.Trimesh,
    t: Dict[str, Any],
    planes: Optional[Sequence[Dict[str, Any]]] = None,
) -> Metric:
    n = int(t["sample_points"] // 2)
    pts, _ = _sample_points_and_normals(mesh, n)
    if not planes:
        planes = [
            {"normal": [1.0, 0.0, 0.0], "offset": 0.0},
        ]
    tree = cKDTree(pts)
    errs = []
    for plane in planes:
        normal = np.array(plane["normal"], dtype=float)
        offset = _safe_float(plane.get("offset", 0.0))
        reflected = _reflect_points(pts, normal, offset)
        d, _ = tree.query(reflected, k=1)
        errs.append(float(np.mean(d)))
    value = float(np.mean(errs)) if errs else float("inf")
    thr = _safe_float(t["symmetry_error_mm_max"])
    return Metric(
        name="symmetry_error_mm",
        value=value,
        threshold=thr,
        direction="lower_better",
        passed=value <= thr,
        details={"planes_tested": len(planes)},
    )


def _vertex_curvature_proxy(mesh: trimesh.Trimesh) -> np.ndarray:
    vnorm = mesh.vertex_normals
    neighbors = mesh.vertex_neighbors
    vals = np.zeros(len(mesh.vertices), dtype=float)
    for i, nbr in enumerate(neighbors):
        if not nbr:
            vals[i] = 0.0
            continue
        dots = np.clip(np.dot(vnorm[nbr], vnorm[i]), -1.0, 1.0)
        ang = np.arccos(dots)
        vals[i] = float(np.mean(ang))
    return vals


def _js_divergence_hist(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if math.isclose(lo, hi):
        return 0.0
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    pa = pa + 1e-12
    pb = pb + 1e-12
    pa /= np.sum(pa)
    pb /= np.sum(pb)
    m = 0.5 * (pa + pb)
    kl_a = np.sum(pa * np.log(pa / m))
    kl_b = np.sum(pb * np.log(pb / m))
    return float(0.5 * (kl_a + kl_b))


def metric_curvature_js(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Metric:
    c_ref = _vertex_curvature_proxy(ref)
    c_gen = _vertex_curvature_proxy(gen)
    js = _js_divergence_hist(c_ref, c_gen, bins=64)
    thr = _safe_float(t["curvature_js_max"])
    return Metric(
        name="curvature_js_divergence",
        value=js,
        threshold=thr,
        direction="lower_better",
        passed=js <= thr,
    )


def _slice_iou_for_axis(
    ref_pts: np.ndarray,
    gen_pts: np.ndarray,
    axis: int,
    positions: Iterable[float],
    thickness: float,
    grid_res: int,
) -> List[float]:
    keep_axes = [i for i in [0, 1, 2] if i != axis]
    ious = []
    for pos in positions:
        ref_mask = np.abs(ref_pts[:, axis] - pos) <= thickness / 2.0
        gen_mask = np.abs(gen_pts[:, axis] - pos) <= thickness / 2.0
        rp = ref_pts[ref_mask][:, keep_axes]
        gp = gen_pts[gen_mask][:, keep_axes]
        if len(rp) < 50 and len(gp) < 50:
            continue
        allp = np.vstack([rp, gp]) if len(rp) and len(gp) else (rp if len(rp) else gp)
        lo = allp.min(axis=0)
        hi = allp.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        rxy = np.floor((rp - lo) / span * (grid_res - 1)).astype(int) if len(rp) else np.empty((0, 2), int)
        gxy = np.floor((gp - lo) / span * (grid_res - 1)).astype(int) if len(gp) else np.empty((0, 2), int)
        rgrid = np.zeros((grid_res, grid_res), dtype=bool)
        ggrid = np.zeros((grid_res, grid_res), dtype=bool)
        if len(rxy):
            rgrid[rxy[:, 0], rxy[:, 1]] = True
        if len(gxy):
            ggrid[gxy[:, 0], gxy[:, 1]] = True
        inter = np.logical_and(rgrid, ggrid).sum()
        union = np.logical_or(rgrid, ggrid).sum()
        iou = float(inter / union) if union > 0 else 1.0
        ious.append(iou)
    return ious


def metric_slice_iou(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Metric:
    n = int(t["sample_points"])
    ref_pts, _ = _sample_points_and_normals(ref, n)
    gen_pts, _ = _sample_points_and_normals(gen, n)
    bbox = np.vstack([ref_pts, gen_pts])
    mins, maxs = bbox.min(axis=0), bbox.max(axis=0)
    slice_count = int(t["slice_count_per_axis"])
    thickness = _safe_float(t["slice_thickness_mm"])
    grid_res = int(t["slice_grid_res"])
    ious: List[float] = []
    for axis in [0, 1, 2]:
        positions = np.linspace(mins[axis], maxs[axis], slice_count + 2)[1:-1]
        ious.extend(
            _slice_iou_for_axis(ref_pts, gen_pts, axis, positions, thickness, grid_res)
        )
    mean_iou = float(np.mean(ious)) if ious else 0.0
    thr = _safe_float(t["slice_iou_mean_min"])
    return Metric(
        name="slice_profile_iou_mean",
        value=mean_iou,
        threshold=thr,
        direction="higher_better",
        passed=mean_iou >= thr,
        details={"slice_count_used": len(ious)},
    )


def metric_signed_distance(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Tuple[Metric, Metric]:
    n = int(t["sample_points"])
    gen_pts, _ = _sample_points_and_normals(gen, n)
    try:
        signed = trimesh.proximity.signed_distance(ref, gen_pts)
    except Exception:
        # Fallback unsigned if signed fails in runtime environment.
        pq = trimesh.proximity.ProximityQuery(ref)
        unsigned = pq.vertex(gen_pts)[0]
        signed = unsigned
    absd = np.abs(signed)
    p95 = float(np.percentile(absd, 95))
    bias = float(np.mean(signed))
    p95_thr = _safe_float(t["signed_distance_p95_mm_max"])
    bias_thr = _safe_float(t["signed_distance_bias_mm_abs_max"])
    return (
        Metric(
            name="signed_distance_p95_mm",
            value=p95,
            threshold=p95_thr,
            direction="lower_better",
            passed=p95 <= p95_thr,
        ),
        Metric(
            name="signed_distance_bias_mm",
            value=bias,
            threshold=bias_thr,
            direction="abs_lower_better",
            passed=abs(bias) <= bias_thr,
        ),
    )


def _local_feature_scale(points: np.ndarray, k: int = 8) -> np.ndarray:
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1)
    # skip self-distance in column 0
    scale = np.mean(d[:, 1:], axis=1)
    return scale


def metric_feature_size_errors(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Tuple[Metric, Metric]:
    n = int(t["sample_points"])
    ref_pts, _ = _sample_points_and_normals(ref, n)
    gen_pts, _ = _sample_points_and_normals(gen, n)
    ref_scale = _local_feature_scale(ref_pts, k=8)
    gen_scale = _local_feature_scale(gen_pts, k=8)
    tree = cKDTree(ref_pts)
    _, idx = tree.query(gen_pts, k=1)
    ref_match = ref_scale[idx]
    pct_err = np.abs(gen_scale - ref_match) / np.maximum(ref_match, 1e-9) * 100.0
    mean_err = float(np.mean(pct_err))
    max_err = float(np.percentile(pct_err, 99))
    mean_thr = _safe_float(t["feature_size_error_mean_percent_max"])
    max_thr = _safe_float(t["feature_size_error_max_percent_max"])
    return (
        Metric(
            name="feature_size_error_mean_percent",
            value=mean_err,
            threshold=mean_thr,
            direction="lower_better",
            passed=mean_err <= mean_thr,
        ),
        Metric(
            name="feature_size_error_max_percent",
            value=max_err,
            threshold=max_thr,
            direction="lower_better",
            passed=max_err <= max_thr,
        ),
    )


def _boundary_loop_count(mesh: trimesh.Trimesh) -> int:
    if mesh.is_watertight:
        return 0
    edges = mesh.edges_sorted
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    if len(boundary_edges) == 0:
        return 0
    # Build graph on boundary edges and count cycles/components as proxy loops.
    g: Dict[int, set] = {}
    for a, b in boundary_edges:
        g.setdefault(int(a), set()).add(int(b))
        g.setdefault(int(b), set()).add(int(a))
    visited = set()
    comps = 0
    for node in g:
        if node in visited:
            continue
        comps += 1
        stack = [node]
        visited.add(node)
        while stack:
            cur = stack.pop()
            for nxt in g[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
    return comps


def _topology_signature(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    components = len(mesh.split(only_watertight=False))
    euler = int(mesh.euler_number)
    boundary_loops = _boundary_loop_count(mesh)
    genus = None
    if mesh.is_watertight:
        genus = int(round((2 - euler) / 2))
    return {
        "components": components,
        "euler": euler,
        "boundary_loops": boundary_loops,
        "genus": genus,
    }


def metric_topology_invariants(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Metric:
    ref_sig = _topology_signature(ref)
    gen_sig = _topology_signature(gen)
    exact = ref_sig == gen_sig
    required = bool(t["topology_exact_match_required"])
    return Metric(
        name="topology_invariants_exact_match",
        value=exact,
        threshold="exact_match",
        direction="exact",
        passed=(exact if required else True),
        details={"reference": ref_sig, "generated": gen_sig},
    )


def metric_thin_feature_preservation(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Metric:
    n = int(t["sample_points"])
    ref_pts, _ = _sample_points_and_normals(ref, n)
    gen_pts, _ = _sample_points_and_normals(gen, n)
    ref_scale = _local_feature_scale(ref_pts, k=8)
    gen_scale = _local_feature_scale(gen_pts, k=8)
    q = _safe_float(t["thin_feature_percentile"])
    tol_pct = _safe_float(t["thin_feature_error_tolerance_percent"])
    cutoff = np.quantile(ref_scale, q)
    thin_idx = np.where(ref_scale <= cutoff)[0]
    if len(thin_idx) == 0:
        preserved = 1.0
    else:
        tree = cKDTree(gen_pts)
        _, idx = tree.query(ref_pts[thin_idx], k=1)
        ref_thin = ref_scale[thin_idx]
        gen_match = gen_scale[idx]
        pct_err = np.abs(gen_match - ref_thin) / np.maximum(ref_thin, 1e-9) * 100.0
        preserved = float(np.mean(pct_err <= tol_pct))
    thr = _safe_float(t["thin_feature_preservation_min"])
    return Metric(
        name="thin_feature_preservation_ratio",
        value=preserved,
        threshold=thr,
        direction="higher_better",
        passed=preserved >= thr,
        details={"thin_feature_percentile": q, "error_tolerance_percent": tol_pct},
    )


def metric_principal_axis_and_ratio(
    ref: trimesh.Trimesh, gen: trimesh.Trimesh, t: Dict[str, Any]
) -> Tuple[Metric, Metric]:
    ref_cov = np.cov(ref.vertices.T)
    gen_cov = np.cov(gen.vertices.T)
    eval_ref, evec_ref = np.linalg.eigh(ref_cov)
    eval_gen, evec_gen = np.linalg.eigh(gen_cov)
    order_ref = np.argsort(eval_ref)[::-1]
    order_gen = np.argsort(eval_gen)[::-1]
    eval_ref = eval_ref[order_ref]
    eval_gen = eval_gen[order_gen]
    evec_ref = evec_ref[:, order_ref]
    evec_gen = evec_gen[:, order_gen]
    angs = []
    for i in range(3):
        dot = np.clip(np.abs(np.dot(evec_ref[:, i], evec_gen[:, i])), -1.0, 1.0)
        angs.append(np.degrees(np.arccos(dot)))
    max_ang = float(np.max(angs))
    norm_ref = eval_ref / np.maximum(np.sum(eval_ref), 1e-12)
    norm_gen = eval_gen / np.maximum(np.sum(eval_gen), 1e-12)
    ratio_err = float(np.max(np.abs(norm_ref - norm_gen)) * 100.0)
    ang_thr = _safe_float(t["principal_axis_angle_deg_max"])
    ratio_thr = _safe_float(t["principal_eigenvalue_ratio_error_percent_max"])
    return (
        Metric(
            name="principal_axis_angle_error_deg",
            value=max_ang,
            threshold=ang_thr,
            direction="lower_better",
            passed=max_ang <= ang_thr,
            details={"axis_angle_errors_deg": angs},
        ),
        Metric(
            name="principal_eigenvalue_ratio_error_percent",
            value=ratio_err,
            threshold=ratio_thr,
            direction="lower_better",
            passed=ratio_err <= ratio_thr,
            details={"reference_norm_eigenvalues": norm_ref.tolist(), "generated_norm_eigenvalues": norm_gen.tolist()},
        ),
    )


def evaluate_all(
    reference_path: str,
    generated_path: str,
    thresholds: Optional[Dict[str, Any]] = None,
    symmetry_planes: Optional[Sequence[Dict[str, Any]]] = None,
    align_generated: bool = True,
) -> Dict[str, Any]:
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    ref = _load_mesh(reference_path)
    gen = _load_mesh(generated_path)
    if align_generated:
        gen = _align_generated_to_reference(ref, gen, int(t["sample_points"]))

    metrics: List[Metric] = []
    metrics.append(metric_surface_area_error(ref, gen, t))
    metrics.append(metric_normal_consistency(ref, gen, t))
    metrics.append(metric_symmetry_error(gen, t, planes=symmetry_planes))
    metrics.append(metric_curvature_js(ref, gen, t))
    metrics.append(metric_slice_iou(ref, gen, t))
    m_p95, m_bias = metric_signed_distance(ref, gen, t)
    metrics.extend([m_p95, m_bias])
    m_fmean, m_fmax = metric_feature_size_errors(ref, gen, t)
    metrics.extend([m_fmean, m_fmax])
    metrics.append(metric_topology_invariants(ref, gen, t))
    metrics.append(metric_thin_feature_preservation(ref, gen, t))
    m_axis, m_ratio = metric_principal_axis_and_ratio(ref, gen, t)
    metrics.extend([m_axis, m_ratio])

    passed = [m.passed for m in metrics]
    result = {
        "reference_path": reference_path,
        "generated_path": generated_path,
        "align_generated": align_generated,
        "overall_pass": bool(all(passed)),
        "pass_count": int(sum(passed)),
        "metric_count": len(metrics),
        "thresholds": t,
        "metrics": [asdict(m) for m in metrics],
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run advanced geometry metrics between reference and generated STL."
    )
    parser.add_argument("--reference", required=True, help="Reference STL path")
    parser.add_argument("--generated", required=True, help="Generated STL path")
    parser.add_argument(
        "--thresholds-json",
        default=None,
        help="JSON string with threshold overrides",
    )
    parser.add_argument(
        "--symmetry-planes-json",
        default=None,
        help='JSON list like [{"normal":[1,0,0],"offset":0.0}]',
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable ICP alignment before metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds = json.loads(args.thresholds_json) if args.thresholds_json else None
    planes = json.loads(args.symmetry_planes_json) if args.symmetry_planes_json else None
    result = evaluate_all(
        reference_path=args.reference,
        generated_path=args.generated,
        thresholds=thresholds,
        symmetry_planes=planes,
        align_generated=not args.no_align,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

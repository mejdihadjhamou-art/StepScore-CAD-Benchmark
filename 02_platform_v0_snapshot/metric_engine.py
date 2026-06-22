from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import math
import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:  # optional
    ssim = None


@dataclass
class MetricResult:
    name: str
    value: float
    threshold: float
    direction: str  # lower_better | higher_better | equal
    passed: bool


DEFAULT_THRESHOLDS: Dict[str, float] = {
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
    # Normalized by combined bbox diagonal; see compare_models.
    "emd_distance": 0.02,
    "silhouette_iou": 0.92,
    "render_ssim": 0.90,
    "render_lpips": 0.15,
    "registration_failure_rate": 0.05,
    "composite_weighted_score": 0.85,
}

QUALITY_REL_SCALE: float = 0.20
QUALITY_SIGMOID_K: float = 2.0
QUALITY_ABS_SCALE_FLOOR: Dict[str, float] = {
    "valid_cad_rate": 0.10,
    "component_count_match": 0.25,
    "watertight_manifold_pass": 0.25,
    "euler_genus_match": 0.25,
    "void_hole_count_match": 0.25,
    "self_intersection_count": 1.0,
    "registration_failure_rate": 0.05,
}
QUALITY_BINARY_METRICS = {
    "valid_cad_rate",
    "component_count_match",
    "watertight_manifold_pass",
    "euler_genus_match",
    "void_hole_count_match",
}
QUALITY_WEIGHT_OVERRIDES: Dict[str, float] = {
    "chamfer_distance_mm": 2.0,
    "hausdorff_95p_mm": 2.0,
    "hausdorff_99p_mm": 1.5,
    "point_to_surface_mean_mm": 1.5,
    "point_to_surface_max_mm": 1.2,
    "edge_chamfer_mm": 1.2,
    "volume_diff_percent": 1.5,
    "surface_area_diff_percent": 1.2,
    "bbox_error_max_mm": 1.2,
    "critical_dimension_error_mm": 1.5,
    "normal_consistency": 1.2,
    "normal_angle_error_deg": 1.2,
    "voxel_iou": 1.2,
    "silhouette_iou": 1.2,
    "render_ssim": 1.2,
}


def _safe(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Invalid or empty mesh: {path}")
    return mesh


def _sample_points_normals(mesh: trimesh.Trimesh, n: int) -> Tuple[np.ndarray, np.ndarray]:
    points, face_idx = trimesh.sample.sample_surface_even(mesh, n)
    normals = mesh.face_normals[face_idx]
    return points, normals


def _nearly_identical_mesh_state(ref: trimesh.Trimesh, gen: trimesh.Trimesh) -> bool:
    """
    Geometric near-identity check (not file-hash based).
    This is used only to avoid unstable ICP/sampling artifacts on identical inputs.
    """
    if len(ref.vertices) != len(gen.vertices) or len(ref.faces) != len(gen.faces):
        return False
    if abs(float(ref.area) - float(gen.area)) > 1e-6:
        return False
    if abs(float(ref.volume) - float(gen.volume)) > 1e-6:
        return False
    if np.max(np.abs(ref.bounding_box.extents - gen.bounding_box.extents)) > 1e-6:
        return False
    if np.linalg.norm(ref.center_mass - gen.center_mass) > 1e-6:
        return False
    return True


def _pca_axes(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    x = points - c
    cov = np.cov(x.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    if np.linalg.det(vecs) < 0:
        vecs[:, -1] *= -1
    return vecs


def _prealign_and_icp(ref_mesh: trimesh.Trimesh, gen_mesh: trimesh.Trimesh, n: int) -> Tuple[trimesh.Trimesh, float, float]:
    ref_pts, _ = _sample_points_normals(ref_mesh, min(12000, n))
    gen_pts, _ = _sample_points_normals(gen_mesh, min(12000, n))

    # Deterministic centroid + PCA pre-alignment.
    ref_c = ref_pts.mean(axis=0)
    gen_c = gen_pts.mean(axis=0)
    r_ax = _pca_axes(ref_pts)
    g_ax = _pca_axes(gen_pts)
    r = r_ax @ g_ax.T
    pre = np.eye(4)
    pre[:3, :3] = r
    pre[:3, 3] = ref_c - (r @ gen_c)

    gen_aligned = gen_mesh.copy()
    gen_aligned.apply_transform(pre)

    # Multi-init symmetry trials: sign flips on principal axes.
    trial_rots = [np.diag([sx, sy, sz, 1.0]) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
    best_rmse = float("inf")
    best_fitness = 0.0
    best_transform = np.eye(4)

    ref_pts2, _ = _sample_points_normals(ref_mesh, min(12000, n))

    for flip in trial_rots:
        trial = gen_aligned.copy()
        trial.apply_transform(flip)
        gpts, _ = _sample_points_normals(trial, min(12000, n))
        matrix, transformed, _ = trimesh.registration.icp(
            gpts,
            ref_pts2,
            scale=False,
            reflection=False,
            max_iterations=60,
        )
        d = np.linalg.norm(transformed - ref_pts2[cKDTree(ref_pts2).query(transformed, k=1)[1]], axis=1)
        rmse = float(np.sqrt(np.mean(d ** 2)))
        fitness = float(np.mean(d < 2.0))
        if rmse < best_rmse:
            best_rmse = rmse
            best_fitness = fitness
            best_transform = matrix @ flip

    out = gen_aligned.copy()
    out.apply_transform(best_transform)
    return out, best_fitness, best_rmse


def _nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return d


def _chamfer_hausdorff(ref_pts: np.ndarray, gen_pts: np.ndarray) -> Dict[str, float]:
    d_rg = _nearest_distances(ref_pts, gen_pts)
    d_gr = _nearest_distances(gen_pts, ref_pts)
    chamfer = float(np.mean(d_rg) + np.mean(d_gr)) / 2.0
    d_all = np.concatenate([d_rg, d_gr])
    return {
        "chamfer": chamfer,
        "haus95": float(np.percentile(d_all, 95)),
        "haus99": float(np.percentile(d_all, 99)),
        "mean_gr": float(np.mean(d_gr)),
        "max_gr": float(np.max(d_gr)),
    }


def _js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if math.isclose(lo, hi):
        return 0.0
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    pa = pa + 1e-12
    pb = pb + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    kl1 = np.sum(pa * np.log(pa / m))
    kl2 = np.sum(pb * np.log(pb / m))
    return float(0.5 * (kl1 + kl2))


def _sharp_edge_midpoints(mesh: trimesh.Trimesh, angle_deg: float = 35.0) -> np.ndarray:
    if len(mesh.face_adjacency) == 0:
        return np.empty((0, 3))
    ang = np.degrees(mesh.face_adjacency_angles)
    mask = ang > angle_deg
    if not np.any(mask):
        return np.empty((0, 3))
    edges = mesh.face_adjacency_edges[mask]
    v = mesh.vertices
    mids = (v[edges[:, 0]] + v[edges[:, 1]]) / 2.0
    return mids


def _voxel_sets(mesh: trimesh.Trimesh, pitch: float) -> set:
    vg = mesh.voxelized(pitch)
    pts = vg.points
    quant = np.round(pts / pitch).astype(np.int64)
    return set(map(tuple, quant.tolist()))


def _occ_metrics(a: set, b: set) -> Tuple[float, float, float, float]:
    if not a and not b:
        return 1.0, 1.0, 1.0, 1.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    p = inter / len(b) if b else 0.0
    r = inter / len(a) if a else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    iou = inter / union if union else 0.0
    return p, r, f1, iou


def _raster_2d(points2d: np.ndarray, res: int = 256) -> np.ndarray:
    if len(points2d) == 0:
        return np.zeros((res, res), dtype=np.uint8)
    lo = points2d.min(axis=0)
    hi = points2d.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    uv = np.floor((points2d - lo) / span * (res - 1)).astype(int)
    uv = np.clip(uv, 0, res - 1)
    img = np.zeros((res, res), dtype=np.uint8)
    img[uv[:, 0], uv[:, 1]] = 1
    return img


def _silhouette_iou(ref_pts: np.ndarray, gen_pts: np.ndarray, res: int = 256) -> Tuple[float, float, float]:
    ious = []
    ssims = []
    axes = [(0, 1), (0, 2), (1, 2)]
    for a, b in axes:
        r = _raster_2d(ref_pts[:, [a, b]], res)
        g = _raster_2d(gen_pts[:, [a, b]], res)
        inter = np.logical_and(r, g).sum()
        union = np.logical_or(r, g).sum()
        iou = float(inter / union) if union else 1.0
        ious.append(iou)
        if ssim is not None:
            ssims.append(float(ssim(r.astype(float), g.astype(float), data_range=1.0)))
    mean_iou = float(np.mean(ious))
    mean_ssim = float(np.mean(ssims)) if ssims else float("nan")
    lpips_proxy = 1.0 - mean_ssim if not np.isnan(mean_ssim) else float("nan")
    return mean_iou, mean_ssim, lpips_proxy


def _slice_profile_metrics(ref_pts: np.ndarray, gen_pts: np.ndarray, slice_count: int = 6, res: int = 160) -> Tuple[float, float]:
    zmin = float(min(ref_pts[:, 2].min(), gen_pts[:, 2].min()))
    zmax = float(max(ref_pts[:, 2].max(), gen_pts[:, 2].max()))
    zs = np.linspace(zmin, zmax, slice_count + 2)[1:-1]
    ious = []
    contour_d = []
    for z in zs:
        thickness = max((zmax - zmin) * 0.01, 1e-3)
        rm = np.abs(ref_pts[:, 2] - z) <= thickness
        gm = np.abs(gen_pts[:, 2] - z) <= thickness
        rp = ref_pts[rm][:, :2]
        gp = gen_pts[gm][:, :2]
        if len(rp) < 20 or len(gp) < 20:
            continue
        ri = _raster_2d(rp, res)
        gi = _raster_2d(gp, res)
        inter = np.logical_and(ri, gi).sum()
        union = np.logical_or(ri, gi).sum()
        iou = float(inter / union) if union else 1.0
        ious.append(iou)

        rd = _nearest_distances(gp, rp)
        contour_d.append(float(np.mean(rd)))
    return (float(np.mean(ious)) if ious else float("nan"), float(np.mean(contour_d)) if contour_d else float("nan"))


def _emd_distance(ref_pts: np.ndarray, gen_pts: np.ndarray, n: int = 250) -> float:
    nr = min(n, len(ref_pts))
    ng = min(n, len(gen_pts))
    n2 = min(nr, ng)
    if n2 < 10:
        return float("nan")
    rp = ref_pts[np.random.choice(len(ref_pts), n2, replace=False)]
    gp = gen_pts[np.random.choice(len(gen_pts), n2, replace=False)]
    c = cdist(rp, gp)
    r, g = linear_sum_assignment(c)
    return float(c[r, g].mean())


def _metric_pass(value: float, threshold: float, direction: str) -> bool:
    if np.isnan(value):
        return False
    if direction == "lower_better":
        return value <= threshold
    if direction == "higher_better":
        return value >= threshold
    if direction == "equal":
        return abs(value - threshold) < 1e-9
    return False


def _sigmoid(x: float, k: float = QUALITY_SIGMOID_K) -> float:
    z = float(k * x)
    # Numerically stable sigmoid.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _metric_quality_scale(metric_name: str, threshold: float) -> float:
    floor = float(QUALITY_ABS_SCALE_FLOOR.get(metric_name, 1e-6))
    rel = abs(float(threshold)) * QUALITY_REL_SCALE
    return max(rel, floor)


def _metric_quality_score(m: MetricResult) -> float:
    if np.isnan(m.value):
        return 0.0

    if m.name in QUALITY_BINARY_METRICS or m.direction == "equal":
        return 1.0 if m.passed else 0.0

    scale = _metric_quality_scale(m.name, m.threshold)
    if m.direction == "lower_better":
        margin = (m.threshold - m.value) / scale
    elif m.direction == "higher_better":
        margin = (m.value - m.threshold) / scale
    else:
        margin = -abs(m.value - m.threshold) / scale

    score = _sigmoid(margin)
    return float(max(0.0, min(1.0, score)))


def _metric_quality_weight(metric_name: str) -> float:
    # Exclude nested aggregate to avoid double-counting.
    if metric_name == "composite_weighted_score":
        return 0.0
    return float(QUALITY_WEIGHT_OVERRIDES.get(metric_name, 1.0))


def compare_models(
    reference_path: str,
    generated_path: str,
    sample_points: int = 30000,
    voxel_pitch_mm: float = 1.0,
    thresholds: Dict[str, float] | None = None,
    fast_mode: bool = True,
) -> Dict[str, Any]:
    # Deterministic behavior for repeatable benchmarking.
    np.random.seed(42)

    if fast_mode:
        # Keep runtime/memory reasonable for local dashboard use.
        sample_points = int(min(sample_points, 4000))
        voxel_pitch_mm = float(max(voxel_pitch_mm, 4.0))

    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    metrics: List[MetricResult] = []
    details: Dict[str, Any] = {}

    try:
        ref = _load_mesh(reference_path)
        gen = _load_mesh(generated_path)
        valid_rate = 1.0
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "metrics": [],
            "details": {},
            "summary": {},
        }

    # 1
    metrics.append(MetricResult("valid_cad_rate", valid_rate, t["valid_cad_rate"], "higher_better", True))

    same_state = _nearly_identical_mesh_state(ref, gen)
    if same_state:
        # Avoid introducing transform noise when meshes are already geometrically identical.
        gen_aligned, fitness, rmse = gen.copy(), 1.0, 0.0
    else:
        gen_aligned, fitness, rmse = _prealign_and_icp(ref, gen, sample_points)
    registration_failure_rate = 0.0 if fitness >= t["alignment_quality_icp_fitness"] else 1.0

    # sample points for most metrics
    ref_pts, ref_n = _sample_points_normals(ref, sample_points)
    if same_state:
        # Use shared samples to remove Monte Carlo drift for equal geometry.
        gen_pts, gen_n = ref_pts.copy(), ref_n.copy()
    else:
        gen_pts, gen_n = _sample_points_normals(gen_aligned, sample_points)

    dist = _chamfer_hausdorff(ref_pts, gen_pts)

    # 2-9
    metrics.extend([
        MetricResult("alignment_quality_icp_fitness", fitness, t["alignment_quality_icp_fitness"], "higher_better", _metric_pass(fitness, t["alignment_quality_icp_fitness"], "higher_better")),
        MetricResult("alignment_inlier_rmse_mm", rmse, t["alignment_inlier_rmse_mm"], "lower_better", _metric_pass(rmse, t["alignment_inlier_rmse_mm"], "lower_better")),
        MetricResult("chamfer_distance_mm", dist["chamfer"], t["chamfer_distance_mm"], "lower_better", _metric_pass(dist["chamfer"], t["chamfer_distance_mm"], "lower_better")),
    ])

    ref_edges = _sharp_edge_midpoints(ref)
    gen_edges = _sharp_edge_midpoints(gen_aligned)
    edge_chamfer = float(np.nan)
    feat_edge_dist = float(np.nan)
    if len(ref_edges) > 20 and len(gen_edges) > 20:
        d1 = _nearest_distances(ref_edges, gen_edges)
        d2 = _nearest_distances(gen_edges, ref_edges)
        edge_chamfer = float((d1.mean() + d2.mean()) / 2.0)
        feat_edge_dist = float(d2.mean())

    metrics.extend([
        MetricResult("edge_chamfer_mm", edge_chamfer, t["edge_chamfer_mm"], "lower_better", _metric_pass(edge_chamfer, t["edge_chamfer_mm"], "lower_better")),
        MetricResult("hausdorff_95p_mm", dist["haus95"], t["hausdorff_95p_mm"], "lower_better", _metric_pass(dist["haus95"], t["hausdorff_95p_mm"], "lower_better")),
        MetricResult("hausdorff_99p_mm", dist["haus99"], t["hausdorff_99p_mm"], "lower_better", _metric_pass(dist["haus99"], t["hausdorff_99p_mm"], "lower_better")),
        MetricResult("point_to_surface_mean_mm", dist["mean_gr"], t["point_to_surface_mean_mm"], "lower_better", _metric_pass(dist["mean_gr"], t["point_to_surface_mean_mm"], "lower_better")),
        MetricResult("point_to_surface_max_mm", dist["max_gr"], t["point_to_surface_max_mm"], "lower_better", _metric_pass(dist["max_gr"], t["point_to_surface_max_mm"], "lower_better")),
    ])

    ref_vol = max(abs(_safe(ref.volume)), 1e-9)
    gen_vol = _safe(gen_aligned.volume)
    volume_diff = abs(gen_vol - ref_vol) / ref_vol * 100.0
    signed_volume_diff = (gen_vol - ref_vol) / ref_vol * 100.0
    ref_area = max(_safe(ref.area), 1e-9)
    gen_area = _safe(gen_aligned.area)
    area_diff = abs(gen_area - ref_area) / ref_area * 100.0

    metrics.extend([
        MetricResult("volume_diff_percent", volume_diff, t["volume_diff_percent"], "lower_better", _metric_pass(volume_diff, t["volume_diff_percent"], "lower_better")),
        MetricResult("signed_volume_diff_percent", abs(signed_volume_diff), t["signed_volume_diff_percent"], "lower_better", _metric_pass(abs(signed_volume_diff), t["signed_volume_diff_percent"], "lower_better")),
        MetricResult("surface_area_diff_percent", area_diff, t["surface_area_diff_percent"], "lower_better", _metric_pass(area_diff, t["surface_area_diff_percent"], "lower_better")),
    ])

    ref_bbox = ref.bounding_box.extents
    gen_bbox = gen_aligned.bounding_box.extents
    bbox_axis = np.abs(gen_bbox - ref_bbox)

    metrics.extend([
        MetricResult("bbox_error_max_mm", float(np.max(bbox_axis)), t["bbox_error_max_mm"], "lower_better", _metric_pass(float(np.max(bbox_axis)), t["bbox_error_max_mm"], "lower_better")),
        MetricResult("bbox_error_axis_x_mm", float(bbox_axis[0]), t["bbox_error_axis_x_mm"], "lower_better", _metric_pass(float(bbox_axis[0]), t["bbox_error_axis_x_mm"], "lower_better")),
        MetricResult("bbox_error_axis_y_mm", float(bbox_axis[1]), t["bbox_error_axis_y_mm"], "lower_better", _metric_pass(float(bbox_axis[1]), t["bbox_error_axis_y_mm"], "lower_better")),
        MetricResult("bbox_error_axis_z_mm", float(bbox_axis[2]), t["bbox_error_axis_z_mm"], "lower_better", _metric_pass(float(bbox_axis[2]), t["bbox_error_axis_z_mm"], "lower_better")),
    ])

    ref_obb = ref.bounding_box_oriented.extents
    gen_obb = gen_aligned.bounding_box_oriented.extents
    obb_err = float(np.max(np.abs(ref_obb - gen_obb)))
    metrics.append(MetricResult("obb_error_max_mm", obb_err, t["obb_error_max_mm"], "lower_better", _metric_pass(obb_err, t["obb_error_max_mm"], "lower_better")))

    centroid_offset = float(np.linalg.norm(ref.center_mass - gen_aligned.center_mass))
    metrics.append(MetricResult("centroid_offset_mm", centroid_offset, t["centroid_offset_mm"], "lower_better", _metric_pass(centroid_offset, t["centroid_offset_mm"], "lower_better")))

    ref_I = np.asarray(ref.moment_inertia)
    gen_I = np.asarray(gen_aligned.moment_inertia)
    inertia_err = float(np.linalg.norm(ref_I - gen_I, ord="fro") / (np.linalg.norm(ref_I, ord="fro") + 1e-9))
    metrics.append(MetricResult("inertia_tensor_error", inertia_err, t["inertia_tensor_error"], "lower_better", _metric_pass(inertia_err, t["inertia_tensor_error"], "lower_better")))

    mass_prop_err = float((centroid_offset / (np.linalg.norm(ref_bbox) + 1e-9) + inertia_err) / 2.0)
    metrics.append(MetricResult("mass_properties_error", mass_prop_err, t["mass_properties_error"], "lower_better", _metric_pass(mass_prop_err, t["mass_properties_error"], "lower_better")))

    ref_comp = len(ref.split(only_watertight=False))
    gen_comp = len(gen_aligned.split(only_watertight=False))
    comp_match = 1.0 if ref_comp == gen_comp else 0.0
    metrics.append(MetricResult("component_count_match", comp_match, t["component_count_match"], "higher_better", _metric_pass(comp_match, t["component_count_match"], "higher_better")))

    watertight = 1.0 if (gen_aligned.is_watertight and gen_aligned.is_winding_consistent) else 0.0
    metrics.append(MetricResult("watertight_manifold_pass", watertight, t["watertight_manifold_pass"], "higher_better", _metric_pass(watertight, t["watertight_manifold_pass"], "higher_better")))

    self_intersections = float(0.0)
    if hasattr(gen_aligned, "is_self_intersecting"):
        try:
            self_intersections = 1.0 if bool(gen_aligned.is_self_intersecting) else 0.0
        except Exception:
            self_intersections = float("nan")
    metrics.append(MetricResult("self_intersection_count", self_intersections, t["self_intersection_count"], "lower_better", _metric_pass(self_intersections, t["self_intersection_count"], "lower_better")))

    ref_genus = float((2 - ref.euler_number) / 2.0) if ref.is_watertight else float("nan")
    gen_genus = float((2 - gen_aligned.euler_number) / 2.0) if gen_aligned.is_watertight else float("nan")
    euler_match = 1.0 if (not np.isnan(ref_genus) and not np.isnan(gen_genus) and abs(ref_genus - gen_genus) < 1e-9) else 0.0
    metrics.append(MetricResult("euler_genus_match", euler_match, t["euler_genus_match"], "higher_better", _metric_pass(euler_match, t["euler_genus_match"], "higher_better")))

    void_hole_match = euler_match
    metrics.append(MetricResult("void_hole_count_match", void_hole_match, t["void_hole_count_match"], "higher_better", _metric_pass(void_hole_match, t["void_hole_count_match"], "higher_better")))

    ref_feature = len(ref_edges)
    gen_feature = len(gen_edges)
    feature_match = float(min(ref_feature, gen_feature) / max(ref_feature, gen_feature)) if max(ref_feature, gen_feature) > 0 else 1.0
    metrics.append(MetricResult("feature_count_match", feature_match, t["feature_count_match"], "higher_better", _metric_pass(feature_match, t["feature_count_match"], "higher_better")))

    critical_dim_err = float(np.mean(bbox_axis))
    metrics.append(MetricResult("critical_dimension_error_mm", critical_dim_err, t["critical_dimension_error_mm"], "lower_better", _metric_pass(critical_dim_err, t["critical_dimension_error_mm"], "lower_better")))

    tol = t["critical_dimension_error_mm"]
    tol_rate = float(np.mean((bbox_axis <= tol).astype(float)))
    metrics.append(MetricResult("tolerance_band_pass_rate", tol_rate, t["tolerance_band_pass_rate"], "higher_better", _metric_pass(tol_rate, t["tolerance_band_pass_rate"], "higher_better")))

    metrics.append(MetricResult("feature_edge_distance_mm", feat_edge_dist, t["feature_edge_distance_mm"], "lower_better", _metric_pass(feat_edge_dist, t["feature_edge_distance_mm"], "lower_better")))

    # Normal metrics
    kdt = cKDTree(ref_pts)
    _, idx = kdt.query(gen_pts, k=1)
    dots = np.clip(np.abs(np.einsum("ij,ij->i", gen_n, ref_n[idx])), 0.0, 1.0)
    normal_consistency = float(np.mean(dots))
    normal_angle_error = float(np.degrees(np.mean(np.arccos(dots))))
    metrics.extend([
        MetricResult("normal_consistency", normal_consistency, t["normal_consistency"], "higher_better", _metric_pass(normal_consistency, t["normal_consistency"], "higher_better")),
        MetricResult("normal_angle_error_deg", normal_angle_error, t["normal_angle_error_deg"], "lower_better", _metric_pass(normal_angle_error, t["normal_angle_error_deg"], "lower_better")),
    ])

    # Curvature proxy: vertex defect histogram JS
    ref_curv = np.asarray(ref.vertex_defects)
    gen_curv = np.asarray(gen_aligned.vertex_defects)
    curvature_err = _js_divergence(ref_curv, gen_curv)
    metrics.append(MetricResult("curvature_distribution_error", curvature_err, t["curvature_distribution_error"], "lower_better", _metric_pass(curvature_err, t["curvature_distribution_error"], "lower_better")))

    cross_iou, slice_dist = _slice_profile_metrics(ref_pts, gen_pts)
    metrics.extend([
        MetricResult("cross_section_iou", cross_iou, t["cross_section_iou"], "higher_better", _metric_pass(cross_iou, t["cross_section_iou"], "higher_better")),
        MetricResult("slice_contour_distance_mm", slice_dist, t["slice_contour_distance_mm"], "lower_better", _metric_pass(slice_dist, t["slice_contour_distance_mm"], "lower_better")),
    ])

    # Guard voxelization memory by estimating voxel grid size first.
    max_voxels = 2_000_000 if fast_mode else 8_000_000
    ext = np.maximum(np.vstack([ref.bounding_box.extents, gen_aligned.bounding_box.extents]).max(axis=0), 1e-6)
    est_voxels = float(np.prod(np.ceil(ext / max(voxel_pitch_mm, 1e-6))))
    if est_voxels <= max_voxels:
        ref_occ = _voxel_sets(ref, voxel_pitch_mm)
        gen_occ = _voxel_sets(gen_aligned, voxel_pitch_mm)
        occ_p, occ_r, occ_f1, vox_iou = _occ_metrics(ref_occ, gen_occ)
    else:
        occ_p = occ_r = occ_f1 = vox_iou = float("nan")
    metrics.extend([
        MetricResult("voxel_iou", vox_iou, t["voxel_iou"], "higher_better", _metric_pass(vox_iou, t["voxel_iou"], "higher_better")),
        MetricResult("occupancy_precision", occ_p, t["occupancy_precision"], "higher_better", _metric_pass(occ_p, t["occupancy_precision"], "higher_better")),
        MetricResult("occupancy_recall", occ_r, t["occupancy_recall"], "higher_better", _metric_pass(occ_r, t["occupancy_recall"], "higher_better")),
        MetricResult("occupancy_f1", occ_f1, t["occupancy_f1"], "higher_better", _metric_pass(occ_f1, t["occupancy_f1"], "higher_better")),
    ])

    emd_n = 120 if fast_mode else 250
    emd_raw = _emd_distance(ref_pts, gen_pts, n=emd_n)
    bbox_diag = float(
        np.linalg.norm(
            np.maximum(
                np.vstack([ref.bounding_box.extents, gen_aligned.bounding_box.extents]).max(axis=0),
                1e-9,
            )
        )
    )
    emd = float(emd_raw / bbox_diag) if (not np.isnan(emd_raw) and bbox_diag > 0.0) else float("nan")
    metrics.append(MetricResult("emd_distance", emd, t["emd_distance"], "lower_better", _metric_pass(emd, t["emd_distance"], "lower_better")))

    sil_iou, render_ssim, render_lpips = _silhouette_iou(ref_pts, gen_pts)
    metrics.extend([
        MetricResult("silhouette_iou", sil_iou, t["silhouette_iou"], "higher_better", _metric_pass(sil_iou, t["silhouette_iou"], "higher_better")),
        MetricResult("render_ssim", render_ssim, t["render_ssim"], "higher_better", _metric_pass(render_ssim, t["render_ssim"], "higher_better")),
        MetricResult("render_lpips", render_lpips, t["render_lpips"], "lower_better", _metric_pass(render_lpips, t["render_lpips"], "lower_better")),
    ])

    metrics.append(MetricResult("registration_failure_rate", registration_failure_rate, t["registration_failure_rate"], "lower_better", _metric_pass(registration_failure_rate, t["registration_failure_rate"], "lower_better")))

    # Composite weighted score (0..1)
    weights = {
        "chamfer_distance_mm": 0.16,
        "hausdorff_95p_mm": 0.14,
        "edge_chamfer_mm": 0.08,
        "volume_diff_percent": 0.10,
        "bbox_error_max_mm": 0.08,
        "normal_consistency": 0.08,
        "voxel_iou": 0.08,
        "silhouette_iou": 0.06,
        "occupancy_f1": 0.08,
        "alignment_quality_icp_fitness": 0.14,
    }

    score_parts = []
    for m in metrics:
        if m.name not in weights or np.isnan(m.value):
            continue
        w = weights[m.name]
        if m.direction == "lower_better":
            norm = max(0.0, min(1.0, m.threshold / (m.value + 1e-9)))
        elif m.direction == "higher_better":
            norm = max(0.0, min(1.0, m.value / (m.threshold + 1e-9)))
        else:
            norm = 1.0 if m.passed else 0.0
        score_parts.append((w, norm))

    composite = float(sum(w * n for w, n in score_parts) / max(sum(w for w, _ in score_parts), 1e-9))
    metrics.append(MetricResult("composite_weighted_score", composite, t["composite_weighted_score"], "higher_better", _metric_pass(composite, t["composite_weighted_score"], "higher_better")))

    pass_count = sum(1 for m in metrics if m.passed)
    fail_count = len(metrics) - pass_count

    metrics_out: List[Dict[str, Any]] = []
    q_weight_sum = 0.0
    q_weight_total = 0.0
    for m in metrics:
        q_score = _metric_quality_score(m)
        q_weight = _metric_quality_weight(m.name)
        metrics_out.append(
            {
                **m.__dict__,
                "quality_score": q_score,
                "quality_weight": q_weight,
            }
        )
        if q_weight > 0.0 and not np.isnan(q_score):
            q_weight_sum += q_weight * q_score
            q_weight_total += q_weight

    quality_score = float(q_weight_sum / max(q_weight_total, 1e-9))

    details.update({
        "reference_volume_mm3": ref_vol,
        "generated_volume_mm3": gen_vol,
        "reference_bbox_mm": ref_bbox.tolist(),
        "generated_bbox_mm": gen_bbox.tolist(),
        "reference_components": ref_comp,
        "generated_components": gen_comp,
    })

    return {
        "ok": True,
        "error": None,
        "metrics": metrics_out,
        "details": details,
        "summary": {
            "total_metrics": len(metrics),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": float(pass_count / len(metrics)),
            "overall_pass": fail_count == 0,
            "quality_score_0_1": quality_score,
            "quality_score_0_100": float(quality_score * 100.0),
        },
    }

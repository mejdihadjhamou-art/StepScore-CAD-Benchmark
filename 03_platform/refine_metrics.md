# Refine Metrics

This document defines an alternative grading profile for STEPScore that reduces metric redundancy while preserving broad geometric coverage.

## Grading Systems

- `full_44`
  - Uses all 44 STEPScore metrics for pass/fail and summary.
- `refined_metrics`
  - Uses a reduced subset for pass/fail and summary.
  - All 44 metrics are still computed and shown in results.

## Refined Metrics Profile (`refined_metrics`)

1. `valid_cad_rate`
2. `component_count_match`
3. `watertight_manifold_pass`
4. `self_intersection_count`
5. `alignment_quality_icp_fitness`
6. `alignment_inlier_rmse_mm`
7. `chamfer_distance_mm`
8. `edge_chamfer_mm`
9. `hausdorff_95p_mm`
10. `hausdorff_99p_mm`
11. `volume_diff_percent`
12. `surface_area_diff_percent`
13. `bbox_error_axis_x_mm`
14. `bbox_error_axis_y_mm`
15. `bbox_error_axis_z_mm`
16. `obb_error_max_mm`
17. `centroid_offset_mm`
18. `euler_genus_match`
19. `feature_count_match`
20. `normal_angle_error_deg`
21. `curvature_distribution_error`
22. `voxel_iou`
23. `emd_distance`
24. `silhouette_iou`
25. `composite_weighted_score`

## Redundant / Derived Metrics (kept for diagnostics, not primary grading)

- `signed_volume_diff_percent`
  - Uses absolute value in current implementation; duplicates `volume_diff_percent` signal.
- `void_hole_count_match`
  - Currently set equal to `euler_genus_match`.
- `render_lpips`
  - Implemented as `1 - render_ssim` proxy.
- `registration_failure_rate`
  - Derived from `alignment_quality_icp_fitness` thresholding.
- `mass_properties_error`
  - Derived from `centroid_offset_mm` and `inertia_tensor_error`.
- `bbox_error_max_mm`
  - Derived from axis errors.
- `critical_dimension_error_mm`
  - Mean of bbox axis errors.
- `tolerance_band_pass_rate`
  - Thresholded function of bbox axis errors.
- `occupancy_f1`
  - Derived from `occupancy_precision` and `occupancy_recall`.

## High-overlap Groups (keep both only if needed for analysis)

- `point_to_surface_mean_mm` vs `chamfer_distance_mm`
- `point_to_surface_max_mm` vs `hausdorff_99p_mm`
- `normal_consistency` vs `normal_angle_error_deg`
- `silhouette_iou` vs `render_ssim`
- `feature_edge_distance_mm` vs `edge_chamfer_mm`

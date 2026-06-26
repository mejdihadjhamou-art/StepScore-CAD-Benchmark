# Threshold Tuning Summary

- Generated at (UTC): `2026-03-25T16:08:04.099861+00:00`
- Input CSV: `./labeling_pairs_keydims.csv`
- Objective: `balanced`
- FP cost: `5.0`
- FN cost: `1.0`
- Label column: `label`
- Total rows in CSV: `119`
- Labeled rows used (positive/negative): `119`
- Metrics tuned successfully: `44`
- Metrics skipped: `2`

## Tuned Metrics

- `alignment_inlier_rmse_mm`: threshold `None` -> `1.765153692`, F1 `0.566`, balanced `0.889`, cost `115.0`
- `alignment_quality_icp_fitness`: threshold `None` -> `0.7185`, F1 `0.448`, balanced `0.822`, cost `185.0`
- `bbox_error_axis_x_mm`: threshold `None` -> `0.126252007`, F1 `0.370`, balanced `0.694`, cost `150.0`
- `bbox_error_axis_y_mm`: threshold `None` -> `0.144078732`, F1 `0.440`, balanced `0.751`, cost `124.0`
- `bbox_error_axis_z_mm`: threshold `None` -> `0.336538948`, F1 `0.380`, balanced `0.764`, cost `245.0`
- `bbox_error_max_mm`: threshold `None` -> `0.528451892`, F1 `0.390`, balanced `0.774`, cost `235.0`
- `centroid_offset_mm`: threshold `None` -> `0.070572014`, F1 `0.634`, balanced `0.871`, cost `67.0`
- `chamfer_distance_mm`: threshold `None` -> `1.360904961`, F1 `0.560`, balanced `0.866`, cost `106.0`
- `component_count_match`: threshold `None` -> `1.0`, F1 `0.323`, balanced `0.697`, cost `315.0`
- `composite_weighted_score`: threshold `None` -> `0.766150787`, F1 `0.750`, balanced `0.952`, cost `50.0`
- `critical_dimension_error_mm`: threshold `None` -> `0.132784234`, F1 `0.524`, balanced `0.790`, cost `84.0`
- `cross_section_iou`: threshold `None` -> `0.026039013`, F1 `0.652`, balanced `0.923`, cost `80.0`
- `curvature_distribution_error`: threshold `None` -> `0.001419582`, F1 `0.714`, balanced `0.942`, cost `60.0`
- `edge_chamfer_mm`: threshold `None` -> `0.864497599`, F1 `0.968`, balanced `0.995`, cost `5.0`
- `emd_distance`: threshold `None` -> `0.091335339`, F1 `0.393`, balanced `0.722`, cost `154.0`
- `euler_genus_match`: threshold `None` -> `1.0`, F1 `0.508`, balanced `0.861`, cost `145.0`
- `feature_count_match`: threshold `None` -> `0.992647059`, F1 `0.737`, balanced `0.923`, cost `46.0`
- `feature_edge_distance_mm`: threshold `None` -> `0.717833001`, F1 `0.622`, balanced `0.890`, cost `81.0`
- `hausdorff_95p_mm`: threshold `None` -> `2.476877803`, F1 `0.583`, balanced `0.875`, cost `96.0`
- `hausdorff_99p_mm`: threshold `None` -> `3.62071949`, F1 `0.625`, balanced `0.913`, cost `90.0`
- `inertia_tensor_error`: threshold `None` -> `0.002230022`, F1 `0.743`, balanced `0.900`, cost `37.0`
- `mass_properties_error`: threshold `None` -> `0.001458702`, F1 `0.811`, balanced `0.966`, cost `35.0`
- `normal_angle_error_deg`: threshold `None` -> `3.91373195`, F1 `0.857`, balanced `0.976`, cost `25.0`
- `normal_consistency`: threshold `None` -> `0.971487618`, F1 `0.824`, balanced `0.943`, cost `26.0`
- `obb_error_max_mm`: threshold `None` -> `0.545742253`, F1 `0.414`, balanced `0.751`, cost `158.0`
- `occupancy_f1`: threshold `None` -> `0.792644261`, F1 `0.375`, balanced `0.760`, cost `250.0`
- `occupancy_precision`: threshold `None` -> `0.821611253`, F1 `0.347`, balanced `0.707`, cost `237.0`
- `occupancy_recall`: threshold `None` -> `0.770653514`, F1 `0.345`, balanced `0.726`, cost `285.0`
- `point_to_surface_max_mm`: threshold `None` -> `4.080006219`, F1 `0.583`, balanced `0.875`, cost `96.0`
- `point_to_surface_mean_mm`: threshold `None` -> `1.361349287`, F1 `0.549`, balanced `0.861`, cost `111.0`
- `registration_failure_rate`: threshold `None` -> `1.0`, F1 `0.224`, balanced `0.500`, cost `520.0`
- `render_lpips`: threshold `None` -> `0.900933362`, F1 `0.239`, balanced `0.544`, cost `407.0`
- `render_ssim`: threshold `None` -> `0.084919139`, F1 `0.213`, balanced `0.481`, cost `472.0`
- `self_intersection_count`: threshold `None` -> `0.0`, F1 `0.224`, balanced `0.500`, cost `520.0`
- `signed_volume_diff_percent`: threshold `None` -> `0.085408311`, F1 `0.684`, balanced `0.885`, cost `52.0`
- `silhouette_iou`: threshold `None` -> `0.075224262`, F1 `0.411`, balanced `0.793`, cost `215.0`
- `slice_contour_distance_mm`: threshold `None` -> `3.22314198`, F1 `0.436`, balanced `0.765`, cost `143.0`
- `surface_area_diff_percent`: threshold `None` -> `0.018301234`, F1 `0.839`, balanced `0.919`, cost `17.0`
- `tolerance_band_pass_rate`: threshold `None` -> `0.511111111`, F1 `0.457`, balanced `0.709`, cost `67.0`
- `valid_cad_rate`: threshold `None` -> `1.0`, F1 `0.224`, balanced `0.500`, cost `520.0`
- `void_hole_count_match`: threshold `None` -> `1.0`, F1 `0.508`, balanced `0.861`, cost `145.0`
- `volume_diff_percent`: threshold `None` -> `0.085408311`, F1 `0.684`, balanced `0.885`, cost `52.0`
- `voxel_iou`: threshold `None` -> `0.656512605`, F1 `0.375`, balanced `0.760`, cost `250.0`
- `watertight_manifold_pass`: threshold `None` -> `1.0`, F1 `0.261`, balanced `0.591`, cost `425.0`

## Skipped Metrics

- `metrics_error`: unknown_direction
- `prompt_text`: unknown_direction


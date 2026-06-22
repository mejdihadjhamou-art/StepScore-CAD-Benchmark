# Threshold Tuning Summary

- Generated at (UTC): `2026-03-25T12:27:11.993829+00:00`
- Input CSV: `/Users/mejdi/Documents/New project/cad42_platform/threshold_tuning/output_by_family/pillow_block/pairs.csv`
- Objective: `cost`
- FP cost: `5.0`
- FN cost: `1.0`
- Label column: `label`
- Total rows in CSV: `22`
- Labeled rows used (positive/negative): `22`
- Metrics tuned successfully: `44`
- Metrics skipped: `0`

## Tuned Metrics

- `alignment_inlier_rmse_mm`: threshold `1.0` -> `1.106883695`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `alignment_quality_icp_fitness`: threshold `0.3` -> `0.9193`, F1 `0.333`, balanced `0.600`, cost `8.0`
- `bbox_error_axis_x_mm`: threshold `1.0` -> `0.045915701`, F1 `0.154`, balanced `0.467`, cost `19.0`
- `bbox_error_axis_y_mm`: threshold `1.0` -> `0.035335101`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `bbox_error_axis_z_mm`: threshold `1.0` -> `0.078371204`, F1 `0.154`, balanced `0.467`, cost `19.0`
- `bbox_error_max_mm`: threshold `1.0` -> `0.087693936`, F1 `0.154`, balanced `0.467`, cost `19.0`
- `centroid_offset_mm`: threshold `1.0` -> `0.047976118`, F1 `0.462`, balanced `0.650`, cost `7.0`
- `chamfer_distance_mm`: threshold `1.0` -> `1.004994719`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `component_count_match`: threshold `1.0` -> `1.0`, F1 `0.625`, balanced `0.500`, cost `60.0`
- `composite_weighted_score`: threshold `0.85` -> `0.878794612`, F1 `0.333`, balanced `0.600`, cost `8.0`
- `critical_dimension_error_mm`: threshold `0.5` -> `0.070685647`, F1 `0.154`, balanced `0.467`, cost `19.0`
- `cross_section_iou`: threshold `0.9` -> `0.02095755`, F1 `0.429`, balanced `0.608`, cost `12.0`
- `curvature_distribution_error`: threshold `0.15` -> `0.0`, F1 `0.690`, balanced `0.625`, cost `45.0`
- `edge_chamfer_mm`: threshold `0.8` -> `0.254865182`, F1 `0.900`, balanced `0.908`, cost `6.0`
- `emd_distance`: threshold `0.02` -> `0.07728741`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `euler_genus_match`: threshold `1.0` -> `0.1`, F1 `0.690`, balanced `0.625`, cost `45.0`
- `feature_count_match`: threshold `0.9` -> `0.557075472`, F1 `0.690`, balanced `0.625`, cost `45.0`
- `feature_edge_distance_mm`: threshold `0.8` -> `0.118554892`, F1 `0.842`, balanced `0.858`, cost `7.0`
- `hausdorff_95p_mm`: threshold `1.5` -> `1.843841526`, F1 `0.182`, balanced `0.550`, cost `9.0`
- `hausdorff_99p_mm`: threshold `2.5` -> `2.238977968`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `inertia_tensor_error`: threshold `0.1` -> `0.003447754`, F1 `0.952`, balanced `0.958`, cost `5.0`
- `mass_properties_error`: threshold `0.1` -> `0.0020748`, F1 `0.952`, balanced `0.958`, cost `5.0`
- `normal_angle_error_deg`: threshold `12.0` -> `3.814561643`, F1 `0.167`, balanced `0.508`, cost `14.0`
- `normal_consistency`: threshold `0.95` -> `0.957221287`, F1 `0.857`, balanced `0.867`, cost `11.0`
- `obb_error_max_mm`: threshold `1.5` -> `0.087693936`, F1 `0.154`, balanced `0.467`, cost `19.0`
- `occupancy_f1`: threshold `0.9` -> `0.98747666`, F1 `0.533`, balanced `0.658`, cost `11.0`
- `occupancy_precision`: threshold `0.9` -> `0.988325282`, F1 `0.533`, balanced `0.658`, cost `11.0`
- `occupancy_recall`: threshold `0.9` -> `0.986638388`, F1 `0.533`, balanced `0.658`, cost `11.0`
- `point_to_surface_max_mm`: threshold `4.0` -> `3.264715448`, F1 `0.182`, balanced `0.550`, cost `9.0`
- `point_to_surface_mean_mm`: threshold `0.8` -> `1.005056905`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `registration_failure_rate`: threshold `0.05` -> `0.0`, F1 `0.625`, balanced `0.500`, cost `60.0`
- `render_lpips`: threshold `0.15` -> `0.807216981`, F1 `0.182`, balanced `0.550`, cost `9.0`
- `render_ssim`: threshold `0.9` -> `0.188374943`, F1 `0.182`, balanced `0.550`, cost `9.0`
- `self_intersection_count`: threshold `0.0` -> `0.0`, F1 `0.625`, balanced `0.500`, cost `60.0`
- `signed_volume_diff_percent`: threshold `2.0` -> `0.0`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `silhouette_iou`: threshold `0.92` -> `0.122690478`, F1 `0.462`, balanced `0.650`, cost `7.0`
- `slice_contour_distance_mm`: threshold `0.8` -> `3.372294769`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `surface_area_diff_percent`: threshold `3.0` -> `3.7e-08`, F1 `0.900`, balanced `0.908`, cost `6.0`
- `tolerance_band_pass_rate`: threshold `0.9` -> `1.0`, F1 `0.645`, balanced `0.542`, cost `55.0`
- `valid_cad_rate`: threshold `1.0` -> `1.0`, F1 `0.625`, balanced `0.500`, cost `60.0`
- `void_hole_count_match`: threshold `1.0` -> `0.1`, F1 `0.690`, balanced `0.625`, cost `45.0`
- `volume_diff_percent`: threshold `2.0` -> `0.0`, F1 `0.000`, balanced `0.500`, cost `10.0`
- `voxel_iou`: threshold `0.88` -> `0.976883309`, F1 `0.533`, balanced `0.658`, cost `11.0`
- `watertight_manifold_pass`: threshold `1.0` -> `1.0`, F1 `0.625`, balanced `0.500`, cost `60.0`


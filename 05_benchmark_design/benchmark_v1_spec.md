# STEPScore Benchmark v1 (Locked Mini-Benchmark)

Date: 2026-02-25  
Owner: Mejdi  
Scope: Single-part CAD generation from text prompt to STEP/STL, evaluated by STEPScore.

## 1) Benchmark Goal
Provide a reproducible, buyer-facing benchmark that measures:
- geometric fidelity to reference CAD,
- topology/engineering validity,
- consistency across repeated trials.

This v1 is intentionally small and locked for fast iteration and proof-of-trainability.

## 2) Locked Task Pack
- Task file: `/Users/mejdi/Documents/New project/cad42_platform/benchmark_v1/tasks_v1.csv`
- Task count: 24
- Prompt levels: L2 and L3
- Base part count used by tasks: 12

Reference files are stored in:
- `/Users/mejdi/Documents/New project/cad42_platform/benchmark_v1/references`

## 3) Run Protocol (Fixed)
- Replicates per task/model: 3
- Fast mode: `False` (fidelity mode)
- Sample points: 20000
- Voxel pitch: 1.0 mm
- Threshold overrides: none (`{}`)
- Random seed behavior: use STEPScore defaults for reproducible runs
- Report both:
  - binary pass/fail summary,
  - continuous quality score (`quality_score_0_100`).

## 4) Thresholds (Locked for v1)
All 44 thresholds are the defaults in:
- `/Users/mejdi/Documents/New project/cad42_platform/metric_engine.py`
- dictionary: `DEFAULT_THRESHOLDS`

No per-run threshold editing is allowed in v1 reporting.

Headline threshold subset (for executive reporting):
- `alignment_quality_icp_fitness >= 0.30`
- `alignment_inlier_rmse_mm <= 1.00`
- `chamfer_distance_mm <= 1.00`
- `hausdorff_95p_mm <= 1.50`
- `volume_diff_percent <= 2.00`
- `bbox_error_max_mm <= 1.00`
- `normal_consistency >= 0.95`
- `voxel_iou >= 0.88`
- `silhouette_iou >= 0.92`
- `composite_weighted_score >= 0.85`

## 5) Hard Engineering Validity Gates
These are always reported as first-class checks:
- `valid_cad_rate`
- `component_count_match`
- `watertight_manifold_pass`
- `self_intersection_count`
- `euler_genus_match`

If any hard gate fails, result is "engineering-invalid" regardless of soft geometry metrics.

## 6) Split Rule (Locked)
For trainability experiments, use family-based grouping:
- No reference file (or near-duplicate variant) can appear in both train and test.
- Group key: `family` column in `tasks_v1.csv`.
- Recommended split when expanding dataset:
  - train 70%
  - dev 15%
  - test 15%
- This `tasks_v1.csv` file is a locked evaluation pack and should remain unchanged once scoring begins.

## 7) Reporting Format (Required)
For each model:
- Overall pass rate (%)
- Mean quality score (0-100)
- Hard gate pass rates
- Mean and std for headline metrics
- Failure taxonomy counts:
  - alignment failures
  - topology failures
  - dimension/volume failures
  - visual/slice failures

## 8) Change Control
Any change to:
- task list,
- thresholds,
- run protocol,
- headline metrics

requires bumping benchmark version (v1 -> v1.1/v2) and publishing a changelog.

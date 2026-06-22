# Benchmark Calibration Protocol (STEP vs STEP)

## Objective
Calibrate your CAD benchmark so pass/fail decisions for generated STEP models are reliable, reproducible, and explainable.

## Scope
- Input: reference STEP, generated STEP
- Output: calibrated thresholds for geometric/topology checks and a stable pass/fail policy

## 1. Standardize the Geometry Pipeline

### 1.1 Canonical units
- Convert all models to mm before meshing/comparison.
- Reject or flag files with ambiguous units.

### 1.2 Deterministic tessellation
- Use one meshing backend and fixed parameters for both reference and generated files.
- Recommended fixed controls:
  - chordal tolerance (e.g., 0.05 mm)
  - angular tolerance (e.g., 5 deg)
  - minimum edge length (fixed)
- Persist meshing config hash in results.

### 1.3 Mesh sanitation
- Remove duplicate vertices/faces, degenerate triangles, unreferenced vertices.
- Validate manifold/watertight status before metric computation.

## 2. Alignment Protocol

### 2.1 Pre-alignment
- Translate both meshes to centroid origin.
- Normalize orientation with PCA or principal inertia frame.
- For symmetric parts, test multiple axis/sign permutations.

### 2.2 ICP refinement
- Run ICP from each candidate initialization.
- Pick best candidate by highest fitness + lowest RMSE.
- Store final transform, ICP fitness, RMSE.

### 2.3 Alignment failure handling
- If no candidate reaches minimum fitness, mark `alignment_failed`.
- In that case: set Chamfer/Hausdorff checks to fail with explicit reason.

## 3. Metric Bundle (Recommended)

## Hard gates
- conversion/render success
- mesh non-empty
- expected component count
- watertightness (for tasks that require it)

## Geometric metrics
- Chamfer distance (mm)
- Hausdorff 95p (mm)
- Hausdorff 99p (mm, diagnostic)
- volume difference (%)
- aligned bbox delta (mm per axis)

## Optional diagnostics
- ICP fitness
- triangle count ratio
- surface area difference (%)

## 4. Calibration Dataset Design

### 4.1 Build three sets
1. `gold_positive` (should pass):
- exact copies
- minor harmless perturbations
2. `gold_negative` (should fail):
- missing/extra features
- wrong key dimensions
- wrong topology (multiple components)
3. `edge_cases`:
- symmetric parts
- thin walls
- near-threshold variants

### 4.2 Minimum sizes
- Pilot: 20 positive + 20 negative
- Production calibration: 50+ positive + 50+ negative

## 5. Threshold Tuning Method

### 5.1 Collect distributions
- For each metric, compute distribution over positives and negatives.
- Plot overlap and select candidate thresholds.

### 5.2 Choose objective
Pick one primary objective:
- minimize false pass (high precision)
- maximize recall for usable outputs
- balanced F1

### 5.3 Set thresholds by tier
- easy / medium / hard tiers can use separate thresholds.
- keep one global fallback threshold for unknown tasks.

### 5.4 Freeze policy
- Version thresholds and policy.
- Recalibrate only on version bump.

## 6. Pass/Fail Policy (Reference)

A run is `PASS` if:
1. all hard gates pass, and
2. Chamfer <= `T_chamfer`, and
3. Hausdorff95 <= `T_haus95`, and
4. volume_diff_pct <= `T_volume`, and
5. bbox_axis_max_diff <= `T_bbox`

Else `FAIL`.

Optional score:
- `score = 100 * (w1*chamfer_norm + w2*haus95_norm + w3*volume_norm + w4*bbox_norm)`
- Hard gates still override to fail.

## 7. Reproducibility Requirements
- lock tool versions (CAD kernel, mesher, geometry libs)
- lock meshing/alignment/sampling parameters
- fixed point sample count per metric run
- fixed random seed where stochastic routines are used
- record config hash in each result row

## 8. Quality Controls

### 8.1 Sanity checks
- identical pair => near-zero distances
- rigid transform pair => pass after alignment
- known bad pair => fail

### 8.2 Drift checks
- run weekly calibration subset
- alert if metric distribution shifts unexpectedly

## 9. Reporting Standard
Each result row should include:
- raw metric values
- thresholds used
- pass/fail per metric
- final decision
- failure taxonomy:
  - generation_failed
  - conversion_failed
  - alignment_failed
  - topology_failed
  - geometric_threshold_failed

## 10. Acceptance Criteria for Benchmark Readiness
- false-pass rate <= target (define target, e.g., < 5%)
- false-fail rate <= target (e.g., < 10%)
- run-to-run decision stability >= target (e.g., >= 95% agreement)
- no unresolved alignment failure cluster on core task set

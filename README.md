# StepScore: AI-Generated CAD Benchmarking & Compliance Platform

**A 44-metric geometry comparison engine for evaluating AI-generated 3D CAD models against reference designs, with human-calibrated thresholds and a path toward automated standards compliance checking.**

Built by Mejdi Hadj Hamou | 2025-2026

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Folder-by-Folder Guide](#folder-by-folder-guide)
- [Version History](#version-history)
- [Metric Engine: The 44 Metrics](#metric-engine-the-44-metrics)
- [Threshold Tuning: How We Got the Numbers](#threshold-tuning-how-we-got-the-numbers)
- [Benchmark Results](#benchmark-results)
- [Future Direction: Automated Standards Compliance](#future-direction-automated-standards-compliance)
- [How to Run](#how-to-run)

---

## Project Overview

StepScore answers the question: **"How good is an AI-generated CAD model compared to the intended design?"**

The engineering industry is adopting AI for design generation (text-to-CAD, parametric generation, design copilots), but there is no standardised way to measure output quality. StepScore fills this gap with:

1. **A 44-metric geometry comparison engine** that loads two STEP files (reference + generated), aligns them using PCA + ICP, and computes metrics across 7 categories (distance, volume, topology, curvature, occupancy, cross-section, rendering).

2. **A parametric benchmark** of 200 jobs across 10 part families and 2 prompt difficulty levels, executed against Claude Sonnet 4.

3. **Human-calibrated thresholds** tuned from 199 manually labeled pass/fail pairs using a custom 3D web-based labeling tool, achieving 88.9% accuracy and 98.6% precision.

4. **A complete platform** including Streamlit dashboard, CLI, batch harness runner, 3D viewer, and threshold tuning pipeline.

---

## Key Results

| Metric | Value |
|--------|-------|
| Benchmark size | 200 jobs (10 families x 10 parts x 2 prompt levels) |
| Success rate | 199/200 (99.5%) |
| Average quality score | 53.9/100 |
| Best family | Pulley (90.9/100), Ring Spacer (90.9/100) |
| Worst family | L-Bracket (23.8/100) |
| Labeled pairs for tuning | 199 (92 positive, 107 negative) |
| Tuned threshold accuracy | 88.9% |
| Tuned threshold precision | 98.6% (only 1 false positive in 199 pairs) |
| Tuned threshold F1 | 0.866 |

---

## Methodology

### Phase 1: Research & Data Collection (Isidor)

- Reviewed academic literature on CAD evaluation (LLM-as-Judge, geometric comparison methods)
- Collected reference STEP files across multiple part families from Isidor's engineering dataset
- Developed prompt specifications across multiple iterations (v1, v2, v3) refining prompt structure for reproducibility
- Defined the evaluation framework: what makes a generated CAD model "good enough"?

### Phase 2: Platform Development (v0 -> v1)

- **v0 (Master Snapshot)**: Initial metric engine with basic geometric comparison (Chamfer distance, Hausdorff, volume, bounding box, topology checks). Core architecture: STEP loading via CadQuery/OCP, mesh tessellation, PCA pre-alignment, ICP refinement.
- **v1 (Current)**: Expanded to 44 metrics across 7 categories. Added harness runner for batch execution, Streamlit dashboard, CLI, 3D viewer, prompt linter, and threshold tuning pipeline. Integrated auto-repair loop (LLM generates CadQuery code -> execute -> if fail, send error back for fix).

### Phase 3: Parametric Dataset Generation

- Built a parametric dataset generator that creates reference STEP files with controlled geometric variation
- 10 part families: box_hole, flange, l_bracket, pillow_block, pulley, ring_spacer, slotted_plate, stepped_shaft, u_channel
- 10 parts per family, each with randomised dimensions within engineering-sensible bounds
- Generated golden prompts at two difficulty levels:
  - **L2**: Moderate detail (key dimensions, basic constraints)
  - **L3**: High detail (full dimensional spec, coordinate frame, explicit constraints)

### Phase 4: Benchmark Execution

- Ran 200 jobs (10 families x 10 parts x 2 levels) against Claude Sonnet 4
- Each job: send prompt to LLM -> receive CadQuery code -> execute -> export STEP -> compare against reference using all 44 metrics
- Auto-repair enabled (1 retry on code execution failure)
- 199/200 succeeded (1 failure: slotted_plate_0077 L3 - STEP conversion produced no STL output)

### Phase 5: Human Labeling

- Built a custom web-based 3D labeling tool (`labeling_app.py`) using Python HTTP server + Three.js
- Side-by-side 3D viewers: reference model (green) vs generated model (blue) with orbit controls
- Manually labeled all 199 successful pairs as "positive" (acceptable quality) or "negative" (unacceptable)
- Label distribution: 92 positive, 107 negative
- **This was intentionally manual** — auto-labeling would create circular reasoning since the labels are used to calibrate the very metrics that would be used for auto-labeling

### Phase 6: Threshold Tuning

- Fed 199 labeled pairs + 44 metric values into cost-sensitive threshold optimisation
- Objective: balanced accuracy with asymmetric costs (false positive cost = 5x false negative cost) — a false pass is much worse than a false fail
- Method: percentile-based candidate sweep per metric, evaluating TP/TN/FP/FN at each candidate threshold
- Global tuning across all families (per-family tuning attempted but most families had too-skewed label distributions)
- Final validation: re-scored all 199 pairs with tuned thresholds -> 88.9% accuracy, 98.6% precision, 77.2% recall, F1=0.866

### Phase 7: Integration & Validation

- Integrated tuned thresholds back into `metric_engine.py` as DEFAULT_THRESHOLDS
- Applied engineering judgment to clamp degenerate values (e.g., `volume_diff_percent` tuned to 1e-09 was clamped to 1.50 as a sensible engineering floor)
- Validated end-to-end: the system now produces pass/fail decisions that agree with human expert judgment 88.9% of the time

---

## Repository Structure

```
StepScore-CAD-Benchmark/
|
|-- 01_research/                    # Phase 1: Research & data collection
|-- 02_platform_v0_snapshot/        # Phase 2a: Original metric engine (frozen)
|-- 03_platform/                    # Phase 2b: Current production platform code
|-- 04_datasets/                    # Phase 3: All CAD datasets (STEP + STL)
|-- 05_benchmark_design/            # Phase 3-4: Benchmark specs, manifests, generators
|-- 06_benchmark_results/           # Phase 4: Raw benchmark run results
|-- 07_labeling_and_tuning/         # Phase 5-6: Human labels + threshold tuning
|-- 08_mvp_backend/                 # CadEval API + Worker + Postgres backend
|-- 09_tools/                       # Utility scripts
|-- 10_visualization/               # 3D viewers and rendering tools
```

---

## Folder-by-Folder Guide

### `01_research/` — Research & Isidor Origins

The project originated from work with Isidor on AI-generated CAD evaluation.

| Subfolder | Contents |
|-----------|----------|
| `isidor_proposals/` | Research summaries, proposals, and strategy documents (PDFs + DOCX) including the detailed CAD data labelling proposal and future direction documents |
| `isidor_prompts/` | Prompt engineering iterations (v1 -> v2 -> v3) in both text and Excel formats. Shows the evolution from basic prompts to fully-specified parametric prompts |
| `isidor_images/` | Extracted figures from research papers used during the literature review |
| `papers/` | Summary and extracted text from "LLM-as-a-Judge & Reward Model" (arXiv:2409.11239v2) — informed our evaluation methodology |

### `02_platform_v0_snapshot/` — Original Platform (Frozen)

The first working version of the metric engine before expansion to 44 metrics.

| File | Purpose |
|------|---------|
| `metric_engine.py` | Original metric engine — Chamfer, Hausdorff, volume, bounding box, topology |
| `app.py` | Original Streamlit dashboard |
| `generation_pipeline.py` | LLM-to-CAD generation pipeline |
| `step_qa.py` | STEP file quality analysis |
| `SNAPSHOT_INFO.txt` | Snapshot metadata |
| `STEPScore_master_snapshot.tar.gz` | Compressed archive of the original codebase |

### `03_platform/` — Current Production Platform

The full evolved platform with 44 metrics, tuned thresholds, and all tools.

| File | Purpose |
|------|---------|
| `metric_engine.py` | **Core engine** — 44-metric comparison with PCA+ICP alignment and tuned DEFAULT_THRESHOLDS (lines 37-83). This is the heart of the project |
| `harness_runner.py` | Batch benchmark runner — executes manifests of prompt/reference pairs against LLM APIs, supports `--resume`, parallel workers, auto-repair |
| `stepscore_cli.py` | Command-line interface for single-pair comparison |
| `app.py` | Streamlit web dashboard for interactive exploration |
| `dashboard.py` | Dashboard rendering module |
| `generation_pipeline.py` | LLM prompt -> CadQuery code -> STEP file pipeline |
| `labeling_app.py` | **3D web labeling tool** — Three.js side-by-side viewer for human pass/fail labeling |
| `labeling_helper.py` | Text-based labeling helper (predecessor to labeling_app.py) |
| `prompt_linter.py` | Validates prompts against the prompt specification guide |
| `compare_thresholds.py` | Compare threshold profiles side-by-side |
| `run_threshold_tuning.py` | Orchestrates threshold tuning pipeline |
| `step_qa.py` | STEP file quality checks |
| `step_utils.py` | STEP file loading utilities |
| `requirements.txt` | Python dependencies |
| `Dockerfile` / `docker-compose.yml` | Containerisation config |

### `04_datasets/` — All CAD Datasets

| Subfolder | Contents | Count |
|-----------|----------|-------|
| `reference_step_files/` | Original Isidor reference STEP files | 73 parts |
| `generated_step_files/` | AI-generated STEP files from early experiments | 217 files |
| `references_parametric/` | Parametric benchmark reference STEP files (10 families x 10 parts) | 100 files |
| `references_parametric_stl/` | STL conversions of parametric references (for browser rendering) | 113 files |
| `references_isidor/` | Isidor references in benchmark-compatible format | 10 files |
| `references_isidor_all/` | Complete Isidor reference set | 24 files |
| `golden_prompts/` | Golden prompt templates (L2 and L3 levels) | CSV + JSONL |
| `ISIDOR.xlsx` | Master Isidor dataset spreadsheet |
| `Isidor_test_gears.xlsx` | Gear test cases |
| `FINAL.xlsx` | Final consolidated reference data |

### `05_benchmark_design/` — Benchmark Specifications

| File / Folder | Contents |
|---------------|----------|
| `benchmark_v1_spec.md` | Formal benchmark specification (families, levels, scoring) |
| `benchmark_calibration.md` | Calibration protocol: alignment, metric bundle, threshold tuning method, pass/fail policy |
| `failure_modes_policy.md` | Defines strict first-pass vs assisted recovery evaluation modes |
| `prompt_guide.md` | Prompt writing specification: required structure, rules, templates, checklist |
| `threshold_tuning_script_plan.md` | Design document for the threshold tuning system |
| `harness_manifests/` | All benchmark manifests (parametric, smoke, v2_combined, Isidor, key-dims) — CSV files mapping each job to its reference, prompt, model, and grading profile |
| `build_scripts/` | Dataset and manifest generators: `generate_parametric_dataset.py`, `generate_golden_prompts.py`, `generate_reference_parts.py`, `build_harness_manifest.py`, etc. |

### `06_benchmark_results/` — Benchmark Run Outputs

| Subfolder | Contents |
|-----------|----------|
| `parametric_claude_sonnet_v1/` | **Primary benchmark** — 200 jobs against Claude Sonnet 4. Contains `results.csv` (per-job scores), `summary_overall.json`, `summary_by_family.csv`, `summary_by_prompt_level.csv`, `summary_by_model.csv` |
| `earlier_runs/` | Previous runs: `final73_anthropic_run_01` (73-job initial run), `final73_anthropic_run_02` (re-run), `harness_exec_smoke` and `harness_exec_smoke2` (smoke tests during development) |

### `07_labeling_and_tuning/` — Human Labeling & Threshold Calibration

This is where the metrics became meaningful. Raw metrics are just numbers — the thresholds that turn them into pass/fail decisions were calibrated from human expert judgment.

| Subfolder | Contents |
|-----------|----------|
| `labeled_data/pairs_for_labeling.csv` | **199 labeled pairs** — each row has a pair_id, family, prompt level, 44 metric values, and a human label (positive/negative). This is the ground truth |
| `labeled_data/labeling_pairs_keydims.csv` | Extended labeling dataset with key dimension analysis |
| `labeled_data/metrics_meta.json` | Metric metadata: direction (lower_is_better / higher_is_better / exact_match) and default thresholds |
| `labeled_data/labeling_guide.md` | Instructions for the labeling process |
| `tuning_scripts/auto_tune_thresholds.py` | **Main tuning engine** — percentile-based candidate search with cost-sensitive optimisation. Accepts pairs CSV + metrics metadata, outputs recommended thresholds |
| `tuning_scripts/auto_tune_thresholds_by_family.py` | Per-family threshold tuning wrapper |
| `tuning_scripts/build_pairs_from_runs.py` | Extracts metric pairs from harness run results |
| `tuning_output_global/threshold_overrides.json` | **Tuned thresholds** — the 44 threshold values that achieved 88.9% accuracy |
| `tuning_output_global/metric_tuning_report.csv` | Per-metric F1, precision, recall, confusion matrix |
| `tuning_output_global/thresholds_recommended.json` | Full metadata: objective, costs, sample sizes |
| `tuning_output_global/tuning_summary.md` | Human-readable tuning report |
| `tuning_output_by_family/` | Per-family tuning attempts (only pillow_block had enough balanced data for per-family tuning) |

### `08_mvp_backend/` — CadEval API + Worker + Postgres

A full-stack MVP backend for productising StepScore as a service.

| Component | Contents |
|-----------|----------|
| `docker-compose.yml` | Full stack: Postgres + API + Worker |
| `services/api/` | Express.js REST API implementing the contract in `docs/mvp/api_contract.yaml` |
| `services/worker/` | Background worker that processes queued runs |
| `db/` | PostgreSQL migrations |
| `docs/mvp/` | API contract (OpenAPI YAML), database schema (SQL), example request/response payloads |
| `external_cadeval/` | External CadEval tooling (dashboard, schemas, test infrastructure) |
| `prompt_reviewer.py` | LLM-based prompt quality reviewer |

### `09_tools/` — Utility Scripts

| Script | Purpose |
|--------|---------|
| `build_labeling_pairs_from_isidor.py` | Extracts metric pairs from Isidor dataset for labeling |
| `build_labeling_pairs_from_key_dims.py` | Extracts pairs with key dimension analysis |
| `compute_metrics_for_pairs.py` | Batch metric computation for reference/generated pairs |
| `clean_thresholds.py` | Clamps degenerate tuned thresholds to engineering-sensible values |
| `xlsx_to_harness_csv.py` | Converts Isidor Excel data to harness manifest CSV format |
| `advanced_geometry_metrics.py` | Extended geometry metrics (surface area error, slice profile IoU, etc.) |
| `tune_thresholds.py` | Original threshold tuning script (predecessor to auto_tune_thresholds.py) |

### `10_visualization/` — 3D Viewers & Rendering

| File | Purpose |
|------|---------|
| `3d_viewer.html` / `3d_viewer_improved.html` | Browser-based Three.js STEP/STL viewers |
| `3d_viewer_server.py` / `3d_viewer_simple.py` | Python servers for serving 3D models to the browser |
| `step_visualizer.py` | STEP file visualisation using CadQuery |
| `viewer.py` / `viewer_builtin.py` / `viewer_debug.py` | Various viewer implementations and debug tools |
| `RUN_3D_VIEWER.sh` | Launch script for the 3D viewer |
| `3D_VIEWER_GUIDE.md` / `VISUALIZATION_GUIDE.md` | Setup and usage documentation |

---

## Version History

### v0 — Initial Prototype (Feb 2026)

- Built core metric engine with 5 geometric metrics (Chamfer, Hausdorff, volume, bbox, topology)
- PCA pre-alignment + ICP refinement pipeline
- Basic Streamlit dashboard
- CadQuery-based STEP loading and tessellation
- Snapshot preserved in `02_platform_v0_snapshot/`

### v1 — 44-Metric Engine (Feb-Mar 2026)

- Expanded from 5 to **44 metrics** across 7 categories:
  - Distance (Chamfer, Hausdorff, point-to-surface, edge Chamfer)
  - Volume/Shape (volume diff, surface area, bounding box, OBB, centroid, inertia)
  - Topology (component count, watertight, self-intersections, Euler genus, hole count, feature count)
  - Tolerance (critical dimension error, tolerance band pass rate, feature edge distance)
  - Surface (normal consistency, normal angle error, curvature distribution)
  - Cross-section (cross-section IoU, slice contour distance)
  - Occupancy (voxel IoU, precision, recall, F1, EMD)
  - Rendering (silhouette IoU, SSIM, LPIPS)
- Added composite weighted score
- Built harness runner for batch execution with resume capability

### v2 — Parametric Benchmark (Mar 2026)

- Generated parametric dataset: 10 families x 10 parts = 100 reference models
- Golden prompts at L2 (moderate) and L3 (detailed) specification levels
- Executed 200-job benchmark against Claude Sonnet 4
- 199/200 success rate, average quality 53.9/100
- Built 3D web labeling tool with Three.js for human evaluation

### v3 — Calibrated Thresholds (Mar 2026)

- Manually labeled 199 pairs (92 positive, 107 negative)
- Cost-sensitive threshold tuning (FP cost 5x FN cost)
- Achieved 88.9% accuracy, 98.6% precision, F1=0.866
- Integrated tuned thresholds as platform defaults
- Added per-family tuning capability

### v4 — MVP Backend (Mar 2026)

- Built CadEval API + Worker + Postgres backend
- REST API with asset upload, run management, result retrieval
- LLM Judge for qualitative assessment
- Docker Compose deployment

---

## Metric Engine: The 44 Metrics

The metrics are organised into categories. Each metric has a **direction** (lower_is_better, higher_is_better, or exact_match) and a **tuned threshold** calibrated from human labels.

### Distance Metrics
| Metric | Direction | Tuned Threshold | What It Measures |
|--------|-----------|----------------|------------------|
| `chamfer_distance_mm` | lower_is_better | 1.36 mm | Average bidirectional closest-point distance |
| `edge_chamfer_mm` | lower_is_better | 0.86 mm | Edge-to-edge distance |
| `hausdorff_95p_mm` | lower_is_better | 2.48 mm | 95th percentile worst-case distance |
| `hausdorff_99p_mm` | lower_is_better | 3.62 mm | 99th percentile worst-case distance |
| `point_to_surface_mean_mm` | lower_is_better | 1.36 mm | Mean point-to-surface distance |
| `point_to_surface_max_mm` | lower_is_better | 4.08 mm | Maximum point-to-surface distance |

### Volume & Shape Metrics
| Metric | Direction | Tuned Threshold | What It Measures |
|--------|-----------|----------------|------------------|
| `volume_diff_percent` | lower_is_better | 1.50%* | Absolute volume difference |
| `signed_volume_diff_percent` | lower_is_better | 1.50%* | Signed volume difference |
| `surface_area_diff_percent` | lower_is_better | 1.50%* | Surface area difference |
| `bbox_error_max_mm` | lower_is_better | 0.53 mm | Maximum bounding box axis error |
| `bbox_error_axis_{x,y,z}_mm` | lower_is_better | 0.13-0.34 mm | Per-axis bounding box error |
| `obb_error_max_mm` | lower_is_better | 0.55 mm | Oriented bounding box error |
| `centroid_offset_mm` | lower_is_better | 0.07 mm | Distance between centroids |
| `inertia_tensor_error` | lower_is_better | 0.002 | Normalised inertia tensor difference |
| `mass_properties_error` | lower_is_better | 0.001 | Combined mass property error |

### Topology Metrics
| Metric | Direction | Tuned Threshold | What It Measures |
|--------|-----------|----------------|------------------|
| `component_count_match` | exact_match | 1.0 | Must be single connected solid |
| `watertight_manifold_pass` | exact_match | 1.0 | Mesh must be watertight |
| `self_intersection_count` | exact_match | 0 | No self-intersecting faces |
| `euler_genus_match` | exact_match | 1.0 | Topological genus must match |
| `void_hole_count_match` | exact_match | 1.0 | Hole/void count must match |
| `feature_count_match` | higher_is_better | 0.99 | Feature count similarity ratio |

### Tolerance Metrics
| Metric | Direction | Tuned Threshold | What It Measures |
|--------|-----------|----------------|------------------|
| `critical_dimension_error_mm` | lower_is_better | 0.13 mm | Error on critical dimensions |
| `tolerance_band_pass_rate` | higher_is_better | 0.51 | Fraction of dims within tolerance |
| `feature_edge_distance_mm` | lower_is_better | 0.72 mm | Feature placement accuracy |

### Surface Quality Metrics
| Metric | Direction | Tuned Threshold | What It Measures |
|--------|-----------|----------------|------------------|
| `normal_consistency` | higher_is_better | 0.97 | Surface normal agreement |
| `normal_angle_error_deg` | lower_is_better | 3.91 deg | Average normal angle error |
| `curvature_distribution_error` | lower_is_better | 0.001 | Curvature distribution divergence |

### Cross-Section & Occupancy Metrics
| Metric | Direction | Tuned Threshold | What It Measures |
|--------|-----------|----------------|------------------|
| `cross_section_iou` | higher_is_better | 0.03 | Cross-section intersection-over-union |
| `slice_contour_distance_mm` | lower_is_better | 3.22 mm | Slice contour distance |
| `voxel_iou` | higher_is_better | 0.66 | Volumetric voxel overlap |
| `occupancy_precision` | higher_is_better | 0.82 | Voxel precision |
| `occupancy_recall` | higher_is_better | 0.77 | Voxel recall |
| `occupancy_f1` | higher_is_better | 0.79 | Voxel F1 score |
| `emd_distance` | lower_is_better | 0.09 | Earth mover's distance |

### Rendering Metrics
| Metric | Direction | Tuned Threshold | What It Measures |
|--------|-----------|----------------|------------------|
| `silhouette_iou` | higher_is_better | 0.08 | Silhouette overlap from standard views |
| `render_ssim` | higher_is_better | 0.08 | Structural similarity of rendered views |
| `render_lpips` | lower_is_better | 0.90 | Perceptual similarity (learned) |

*\* Clamped from degenerate tuned values to engineering-sensible floors*

---

## Threshold Tuning: How We Got the Numbers

### The Problem

Raw metric values are meaningless without thresholds. A Chamfer distance of 1.5mm might be excellent for a large bracket but terrible for a precision gear. We needed thresholds that align with human expert judgment about what constitutes an "acceptable" generated part.

### The Process

1. **Generate ground truth**: Ran 200 benchmark jobs, producing 199 reference-vs-generated pairs with all 44 metric values computed.

2. **Human labeling**: Built a custom 3D web tool (`labeling_app.py`) with side-by-side Three.js viewers. Manually inspected each pair and labeled it positive (acceptable) or negative (unacceptable). Result: 92 positive, 107 negative.

3. **Cost-sensitive optimisation**: For each metric, swept through percentile-based candidate thresholds. Evaluated each candidate's confusion matrix (TP/TN/FP/FN) with asymmetric costs: false positive cost = 5.0, false negative cost = 1.0. A false pass (telling someone a bad part is good) is much worse than a false fail (flagging a good part for review).

4. **Threshold selection**: Selected the threshold that minimised total cost for each metric independently.

5. **Engineering clamping**: Some metrics produced degenerate thresholds (e.g., volume_diff_percent = 1e-09) because the cost function pushed toward extreme precision. We applied engineering judgment to set sensible floors (e.g., 1.50% for volume, 0.50mm for bounding box axes).

6. **Validation**: Re-scored all 199 pairs with tuned thresholds. Results:
   - Accuracy: 88.9%
   - Precision: 98.6% (only 1 false positive out of 72 predicted positives)
   - Recall: 77.2% (21 good parts flagged for review — acceptable given the asymmetric cost preference)
   - F1: 0.866

### Why Not Auto-Label?

Auto-labeling using heuristics or the metrics themselves would be circular reasoning — you'd be using metric values to generate labels, then using those labels to tune the metrics. The labels must come from an independent source (human expert visual inspection of the 3D geometry).

---

## Benchmark Results

### Overall (Claude Sonnet 4, 200 jobs)

- **199/200 succeeded** (1 STEP conversion failure)
- **Average quality score: 53.9/100**
- **L2 prompts: 56.8/100** (moderate specification)
- **L3 prompts: 50.8/100** (detailed specification — surprisingly lower, suggesting over-specification may confuse the model)

### By Part Family

| Family | Avg Score | Notes |
|--------|-----------|-------|
| Pulley | 90.9 | Best performer — rotationally symmetric, well-suited to CadQuery |
| Ring Spacer | 90.9 | Simple geometry, high success |
| Pillow Block | 58.8 | Complex but manageable |
| Stepped Shaft | 54.8 | Moderate — L2 much better than L3 |
| Box Hole | 50.1 | Average |
| Flange | 47.8 | Below average |
| U-Channel | 41.0 | Struggled with profile accuracy |
| Slotted Plate | 37.8 | Feature placement issues |
| L-Bracket | 23.8 | Worst — simple geometry but poor dimensional accuracy |

### Label Distribution

| Family | Positive | Negative | Notes |
|--------|----------|----------|-------|
| Pulley | 22 | 0 | All passed human review |
| Ring Spacer | 22 | 0 | All passed human review |
| L-Bracket | 0 | 22 | None passed human review |
| Slotted Plate | 0 | 21 | None passed (+ 1 failure) |
| Box Hole | 5 | 19 | Mostly failed |
| Flange | 13 | 9 | Mixed |
| Stepped Shaft | 13 | 9 | Mixed |
| Pillow Block | 10 | 12 | Mixed |
| U-Channel | 7 | 15 | Mostly failed |

---

## Future Direction: Automated Standards Compliance

The StepScore engine can be repurposed as an **Automated Standards Compliance Checker** — instead of comparing two models, check one model against encoded design rules.

**The insight**: 44% of engineering standards are not consistently applied (CoLab Software survey, 2025, n=250 engineering leaders). Every non-compliant part reaching manufacturing costs $5K-50K in rework.

**How it maps**:
- Reference model -> Design standard/rule set
- Generated model -> Any engineer's submitted design
- Metric thresholds -> Compliance rules (wall thickness >= 1.5mm, draft angle >= 1 deg, etc.)
- Pass/fail -> Compliant/non-compliant

New geometry extractors needed: wall thickness analysis, draft angle detection, fillet/chamfer detection, hole feature recognition, undercut detection. The existing topology, dimension, and surface quality checks transfer directly.

---

## How to Run

### Prerequisites

```bash
pip install -r 03_platform/requirements.txt
```

Key dependencies: CadQuery, Open3D, NumPy, SciPy, trimesh, Streamlit

### Compare Two STEP Files

```bash
cd 03_platform/
python stepscore_cli.py compare --reference path/to/ref.step --generated path/to/gen.step
```

### Run the Dashboard

```bash
cd 03_platform/
streamlit run app.py
```

### Run a Benchmark

```bash
cd 03_platform/
python harness_runner.py \
  --manifest ../05_benchmark_design/harness_manifests/harness_manifest.parametric.csv \
  --run-id my_run \
  --max-workers 2 \
  --resume
```

### Launch the 3D Labeling Tool

```bash
cd 03_platform/
python labeling_app.py --csv ../07_labeling_and_tuning/labeled_data/pairs_for_labeling.csv --port 8510
```

### Tune Thresholds

```bash
cd 07_labeling_and_tuning/tuning_scripts/
python auto_tune_thresholds.py \
  --pairs-csv ../labeled_data/pairs_for_labeling.csv \
  --metrics-meta-json ../labeled_data/metrics_meta.json \
  --objective balanced \
  --fp-cost 5.0 \
  --fn-cost 1.0 \
  --output-dir ../tuning_output_global/
```

---

## License

Proprietary. All rights reserved.

---

## Contact

Mejdi Hadj Hamou — mejdihadjhamou@gmail.com

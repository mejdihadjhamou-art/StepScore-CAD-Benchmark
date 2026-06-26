# CAD42 Local Platform

Local dashboard with three workflows:
1. Generate STEP from prompt + compare against reference using 42 metrics
   - Includes a step-builder prompt form (`part goal`, `units and frame`, `geometry`, etc.) that auto-collates into one model prompt
2. Modify existing STEP from prompt + compare against reference using 42 metrics
3. Ask questions about a STEP file (geometry facts, color metadata when present, weight estimates by material, and optional AI-based open-ended answers)
   - Includes optional GTFA input to score predicted answer vs expected answer (pass/fail)

## Supported model formats
- Comparison reference: STL, OBJ, PLY, OFF, GLB/GLTF, STEP/STP
- Base model for modify mode: STEP/STP
- Q&A mode input: STEP/STP
- STEP/STP is auto-converted to STL internally for mesh-based operations

## API-backed generation
- For workflow 1 and 2, the app calls the selected model provider and expects CadQuery code output.
- It executes the code, exports `generated.step` and `generated.stl`, then runs the 42 metrics pipeline.
- Workflow 3 can optionally use a model API for open-ended question answering over extracted STEP facts.
- Comparison workflows include side-by-side interactive 3D views of reference vs generated geometry.
- Scoring now reports both strict pass-rate and a continuous quality score (0-100) that rewards being closer to thresholds even when a metric fails.
- API keys:
  - OpenAI: set `OPENAI_API_KEY` env var, or paste in dashboard
  - Anthropic: set `ANTHROPIC_API_KEY` env var, or paste in dashboard

## Option 1: Run with Conda
```bash
cd "./cad42_platform"
conda create -n cad42 python=3.11 -y
conda activate cad42
pip install -r requirements.txt
streamlit run app.py
```
Open: `http://localhost:8501`

## Option 2: Run with Docker Compose
```bash
cd "./cad42_platform"
docker compose up --build
```
Open: `http://localhost:8501`

To stop:
```bash
docker compose down
```

## Notes
- This app compares mesh geometry directly.
- STEP/STP inputs are converted into run-local STL files before metric evaluation.
- Some metrics are approximations intended for benchmark screening and should be calibrated on your dataset.
- If generation crashes with `exit=-9`, this usually indicates memory pressure. You can reduce prompt complexity or set an optional execution memory cap with `CADQUERY_RUN_MAX_MEM_MB` (disabled by default; set a positive integer MB value only if needed).

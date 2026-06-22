# CadEval MVP: DB + API Contract

This package defines a production-ready MVP contract for:
- Uploading reference CAD assets.
- Submitting model runs with replicates.
- Tracking per-replicate checks/metrics/artifacts.
- Returning normalized pass/fail + numeric summaries.

## Files
- `docs/mvp/db_schema.sql`: PostgreSQL schema.
- `docs/mvp/api_contract.yaml`: OpenAPI 3.1 contract.

## Core flow
1. Client calls `POST /v1/assets/uploads` for a reference STL.
2. Client uploads file to returned `upload_url`.
3. Client calls `POST /v1/runs` with `reference_asset_id`, model, thresholds, prompt, and replicates.
4. Worker dequeues run and creates/updates `run_replicates`.
5. Worker stores generated SCAD/STL/logs as `assets` + `replicate_artifacts`.
6. Worker writes checks to `replicate_checks` and metrics to `replicate_metrics`.
7. API serves progress via `GET /v1/runs/{run_id}` and detailed outcomes via `GET /v1/runs/{run_id}/results`.

## Endpoint to table mapping
- `POST /v1/assets/uploads` -> inserts `assets` (placeholder metadata) + returns signed upload URL.
- `POST /v1/runs` -> inserts `evaluation_runs`; pre-creates `run_replicates` rows (`PENDING`).
- `GET /v1/runs` / `GET /v1/runs/{run_id}` -> reads `evaluation_runs` (+ joins `models`, `threshold_profiles`, `assets`).
- `GET /v1/runs/{run_id}/replicates` -> reads `run_replicates` + `replicate_checks` + `replicate_metrics`.
- `GET /v1/runs/{run_id}/results` -> computes aggregate from replicate rows/checks/metrics.
- `GET /v1/runs/{run_id}/artifacts` -> joins `run_replicates` -> `replicate_artifacts` -> `assets`.
- `DELETE /v1/runs/{run_id}` -> sets run `status=CANCELLED`, marks pending replicates `SKIPPED`.

## Worker state transitions
- Run: `QUEUED -> RUNNING -> SUCCEEDED|FAILED|CANCELLED`
- Replicate: `PENDING -> RUNNING -> SUCCEEDED|FAILED|SKIPPED`

Suggested success rule (MVP):
- Replicate is considered passed when all required checks in `replicate_checks` have `passed=true`.
- Run `pass_rate = passed_replicates / requested_replicates`.

## Implementation notes
- Keep `threshold_profiles.config_hash` as canonical JSON hash for reproducibility.
- Store full machine output in `run_config_json`, `raw_model_output`, and `details_json`.
- Keep artifact blobs out of DB; store only `storage_uri` in `assets`.
- Add row-level security later if multi-tenant auth is introduced.

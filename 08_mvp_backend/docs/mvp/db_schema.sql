-- CadEval Platform MVP schema (PostgreSQL 15+)
-- Enables: prompt+reference upload, queued runs, per-replicate results,
-- metric/check tracking, and artifact retrieval.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE run_status AS ENUM (
  'QUEUED',
  'RUNNING',
  'SUCCEEDED',
  'FAILED',
  'CANCELLED'
);

CREATE TYPE replicate_status AS ENUM (
  'PENDING',
  'RUNNING',
  'SUCCEEDED',
  'FAILED',
  'SKIPPED'
);

CREATE TYPE artifact_kind AS ENUM (
  'REFERENCE_STL',
  'GENERATED_SCAD',
  'GENERATED_STL',
  'RENDER_PNG',
  'RUN_LOG',
  'RESULT_JSON',
  'DASHBOARD_JSON',
  'OTHER'
);

CREATE TABLE projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  provider TEXT NOT NULL,                  -- openai, anthropic, etc.
  model_key TEXT NOT NULL,                 -- gpt-4.1-mini-2025-04-14
  display_name TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (provider, model_key)
);

CREATE TABLE threshold_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  is_default BOOLEAN NOT NULL DEFAULT FALSE,
  config_json JSONB NOT NULL,              -- full geometry/topology thresholds
  config_hash TEXT NOT NULL,               -- sha256 of canonical JSON
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (project_id, name),
  UNIQUE (project_id, config_hash)
);

CREATE TABLE assets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind artifact_kind NOT NULL,
  file_name TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
  sha256 TEXT NOT NULL,
  storage_uri TEXT NOT NULL,               -- s3://bucket/key or local path
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb, -- units, bbox, source, etc.
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (project_id, sha256, kind)
);

CREATE TABLE evaluation_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  model_id UUID NOT NULL REFERENCES models(id),
  threshold_profile_id UUID NOT NULL REFERENCES threshold_profiles(id),
  status run_status NOT NULL DEFAULT 'QUEUED',
  prompt TEXT NOT NULL,
  prompt_template TEXT,                    -- "default", "strict", etc.
  requested_replicates INT NOT NULL CHECK (requested_replicates BETWEEN 1 AND 20),
  reference_asset_id UUID NOT NULL REFERENCES assets(id),
  submitted_by TEXT,                       -- user id/email/service principal
  external_id TEXT,                        -- optional idempotency/client run id
  run_config_json JSONB NOT NULL DEFAULT '{}'::jsonb, -- model opts, timeout, etc.
  queue_name TEXT NOT NULL DEFAULT 'default',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  error_message TEXT,
  UNIQUE (project_id, external_id)
);

CREATE TABLE run_replicates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
  replicate_index INT NOT NULL CHECK (replicate_index >= 1),
  seed BIGINT,
  status replicate_status NOT NULL DEFAULT 'PENDING',
  worker_id TEXT,
  raw_model_output TEXT,                   -- raw response before extraction
  scad_source TEXT,                        -- extracted OpenSCAD text
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  error_message TEXT,
  UNIQUE (run_id, replicate_index)
);

CREATE TABLE replicate_artifacts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  replicate_id UUID NOT NULL REFERENCES run_replicates(id) ON DELETE CASCADE,
  asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (replicate_id, asset_id)
);

CREATE TABLE replicate_checks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  replicate_id UUID NOT NULL REFERENCES run_replicates(id) ON DELETE CASCADE,
  check_key TEXT NOT NULL,                 -- render, watertight, hausdorff_p95...
  passed BOOLEAN NOT NULL,
  measured_value DOUBLE PRECISION,
  threshold_value DOUBLE PRECISION,
  unit TEXT,                               -- mm, %, boolean
  details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (replicate_id, check_key)
);

CREATE TABLE replicate_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  replicate_id UUID NOT NULL REFERENCES run_replicates(id) ON DELETE CASCADE,
  metric_key TEXT NOT NULL,                -- chamfer_mean, hausdorff_p99, volume_ref
  value DOUBLE PRECISION NOT NULL,
  unit TEXT NOT NULL,                      -- mm, mm3, ratio
  details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (replicate_id, metric_key)
);

CREATE TABLE run_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,                -- queued, worker_started, finished...
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_runs_project_created ON evaluation_runs(project_id, created_at DESC);
CREATE INDEX idx_runs_status_created ON evaluation_runs(status, created_at DESC);
CREATE INDEX idx_replicates_run_status ON run_replicates(run_id, status);
CREATE INDEX idx_checks_replicate ON replicate_checks(replicate_id);
CREATE INDEX idx_metrics_replicate ON replicate_metrics(replicate_id);
CREATE INDEX idx_assets_project_kind ON assets(project_id, kind, created_at DESC);


CREATE TABLE run_judgments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
  judge_type TEXT NOT NULL,           -- llm or heuristic
  judge_model TEXT NOT NULL,          -- gpt-4.1-mini, heuristic-v1, etc.
  confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  verdict TEXT NOT NULL,              -- match, partial, mismatch
  result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_run_judgments_run_created ON run_judgments(run_id, created_at DESC);

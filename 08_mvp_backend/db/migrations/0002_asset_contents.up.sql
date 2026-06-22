CREATE TABLE asset_contents (
  asset_id UUID PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
  content_type TEXT NOT NULL,
  content_text TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_asset_contents_created_at ON asset_contents(created_at DESC);

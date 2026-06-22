import crypto from "node:crypto";
import { pool, withTx } from "./db.js";

const DEFAULT_THRESHOLDS = {
  geometry_check: {
    bounding_box_tolerance_mm: 1.0,
    chamfer_threshold_mm: 1.0,
    hausdorff_threshold_mm: 1.0,
    volume_threshold_percent: 2.0,
  },
  topology_check: {
    expected_component_count: 1,
  },
};

function hashJson(value) {
  const canonical = JSON.stringify(value);
  return crypto.createHash("sha256").update(canonical).digest("hex");
}

async function ensureModelSeed(db = pool) {
  await db.query(
    `INSERT INTO models (provider, model_key, display_name, is_active)
     VALUES ('openai', 'gpt-4.1-mini-2025-04-14', 'GPT-4.1 Mini (2025-04-14)', true)
     ON CONFLICT (provider, model_key) DO NOTHING`
  );
}

async function ensureProject(projectId, db = pool) {
  const slug = `project-${projectId.replace(/-/g, "").slice(0, 12)}`;
  await db.query(
    `INSERT INTO projects (id, name, slug)
     VALUES ($1, $2, $3)
     ON CONFLICT (id) DO NOTHING`,
    [projectId, `Project ${projectId.slice(0, 8)}`, slug]
  );
}

export async function listModels() {
  await ensureModelSeed();
  const { rows } = await pool.query(
    `SELECT id, provider, model_key, display_name, is_active
     FROM models
     ORDER BY created_at DESC`
  );
  return rows;
}

export async function ensureDefaultThresholdProfile(projectId) {
  await ensureProject(projectId);
  const { rows: existing } = await pool.query(
    `SELECT id, project_id, name, is_default, config_json, config_hash
     FROM threshold_profiles
     WHERE project_id = $1 AND name = 'default'
     LIMIT 1`,
    [projectId]
  );
  if (existing.length) return existing[0];

  const configHash = hashJson(DEFAULT_THRESHOLDS);
  const { rows } = await pool.query(
    `INSERT INTO threshold_profiles (project_id, name, is_default, config_json, config_hash)
     VALUES ($1, 'default', true, $2::jsonb, $3)
     ON CONFLICT (project_id, name)
     DO UPDATE SET config_json = EXCLUDED.config_json
     RETURNING id, project_id, name, is_default, config_json, config_hash`,
    [projectId, JSON.stringify(DEFAULT_THRESHOLDS), configHash]
  );
  return rows[0];
}

export async function listThresholdProfiles(projectId) {
  await ensureDefaultThresholdProfile(projectId);
  const { rows } = await pool.query(
    `SELECT id, project_id, name, is_default, config_json, config_hash
     FROM threshold_profiles
     WHERE project_id = $1
     ORDER BY is_default DESC, created_at DESC`,
    [projectId]
  );
  return rows;
}

export async function createAssetUpload(input) {
  await ensureProject(input.project_id);
  const storageUri = `object://pending/${input.project_id}/${input.kind}/${input.sha256}`;
  const { rows } = await pool.query(
    `INSERT INTO assets (
      project_id, kind, file_name, mime_type, byte_size, sha256, storage_uri, metadata_json
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
    ON CONFLICT (project_id, sha256, kind)
    DO UPDATE SET
      file_name = EXCLUDED.file_name,
      mime_type = EXCLUDED.mime_type,
      byte_size = EXCLUDED.byte_size,
      storage_uri = EXCLUDED.storage_uri,
      metadata_json = EXCLUDED.metadata_json
    RETURNING id, project_id, kind, file_name, mime_type, byte_size, sha256, metadata_json, created_at`,
    [
      input.project_id,
      input.kind,
      input.file_name,
      input.mime_type,
      input.byte_size,
      input.sha256,
      storageUri,
      JSON.stringify(input.metadata_json || {}),
    ]
  );
  return rows[0];
}

export async function getAsset(assetId) {
  const { rows } = await pool.query(
    `SELECT id, project_id, kind, file_name, mime_type, byte_size, sha256, metadata_json, created_at
     FROM assets
     WHERE id = $1`,
    [assetId]
  );
  return rows[0] || null;
}

export async function createRun(input) {
  return withTx(async (tx) => {
    await ensureProject(input.project_id, tx);
    await ensureModelSeed(tx);

    const model = await tx.query(`SELECT id FROM models WHERE id = $1`, [input.model_id]);
    if (!model.rowCount) {
      throw new Error("invalid model_id");
    }

    const threshold = await tx.query(`SELECT id FROM threshold_profiles WHERE id = $1`, [
      input.threshold_profile_id,
    ]);
    if (!threshold.rowCount) {
      throw new Error("invalid threshold_profile_id");
    }

    const asset = await tx.query(`SELECT id FROM assets WHERE id = $1`, [
      input.reference_asset_id,
    ]);
    if (!asset.rowCount) {
      throw new Error("invalid reference_asset_id");
    }

    const inserted = await tx.query(
      `INSERT INTO evaluation_runs (
        project_id, model_id, threshold_profile_id, prompt, prompt_template,
        requested_replicates, reference_asset_id, run_config_json, external_id
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
      RETURNING id, project_id, model_id, threshold_profile_id, status, prompt, prompt_template,
        requested_replicates, reference_asset_id, run_config_json, external_id,
        created_at, started_at, completed_at, error_message`,
      [
        input.project_id,
        input.model_id,
        input.threshold_profile_id,
        input.prompt,
        input.prompt_template,
        input.requested_replicates,
        input.reference_asset_id,
        JSON.stringify(input.run_config_json || {}),
        input.external_id,
      ]
    );
    const run = inserted.rows[0];

    await tx.query(
      `INSERT INTO run_replicates (run_id, replicate_index, status)
       SELECT $1, i, 'PENDING'::replicate_status
       FROM generate_series(1, $2) AS i`,
      [run.id, input.requested_replicates]
    );

    return run;
  });
}

export async function listRuns({ projectId, status, modelId, limit, cursor }) {
  const params = [projectId];
  const where = ["project_id = $1"];
  let idx = 2;
  if (status) {
    where.push(`status = $${idx}`);
    params.push(status);
    idx += 1;
  }
  if (modelId) {
    where.push(`model_id = $${idx}`);
    params.push(modelId);
    idx += 1;
  }
  let cursorClause = "";
  if (cursor) {
    cursorClause = `AND created_at < (SELECT created_at FROM evaluation_runs WHERE id = $${idx})`;
    params.push(cursor);
    idx += 1;
  }
  params.push(limit + 1);

  const { rows } = await pool.query(
    `SELECT id, project_id, model_id, threshold_profile_id, status, prompt, prompt_template,
      requested_replicates, reference_asset_id, run_config_json, external_id,
      created_at, started_at, completed_at, error_message
     FROM evaluation_runs
     WHERE ${where.join(" AND ")} ${cursorClause}
     ORDER BY created_at DESC
     LIMIT $${idx}`,
    params
  );

  let nextCursor = null;
  let items = rows;
  if (rows.length > limit) {
    items = rows.slice(0, limit);
    nextCursor = items[items.length - 1].id;
  }
  return { items, nextCursor };
}

export async function getRun(runId) {
  const { rows } = await pool.query(
    `SELECT id, project_id, model_id, threshold_profile_id, status, prompt, prompt_template,
      requested_replicates, reference_asset_id, run_config_json, external_id,
      created_at, started_at, completed_at, error_message
     FROM evaluation_runs
     WHERE id = $1`,
    [runId]
  );
  return rows[0] || null;
}

export async function getRunDetail(runId) {
  const run = await getRun(runId);
  if (!run) return null;

  const [{ rows: modelRows }, { rows: thresholdRows }, { rows: assetRows }] =
    await Promise.all([
      pool.query(
        `SELECT id, provider, model_key, display_name, is_active FROM models WHERE id = $1`,
        [run.model_id]
      ),
      pool.query(
        `SELECT id, project_id, name, is_default, config_json, config_hash
         FROM threshold_profiles WHERE id = $1`,
        [run.threshold_profile_id]
      ),
      pool.query(
        `SELECT id, project_id, kind, file_name, mime_type, byte_size, sha256, metadata_json, created_at
         FROM assets WHERE id = $1`,
        [run.reference_asset_id]
      ),
    ]);

  return {
    ...run,
    model: modelRows[0] || null,
    threshold_profile: thresholdRows[0] || null,
    reference_asset: assetRows[0] || null,
  };
}

export async function cancelRun(runId) {
  return withTx(async (tx) => {
    const current = await tx.query(`SELECT status FROM evaluation_runs WHERE id = $1`, [runId]);
    if (!current.rowCount) return null;
    if (["SUCCEEDED", "FAILED"].includes(current.rows[0].status)) {
      return "TERMINAL";
    }

    await tx.query(
      `UPDATE evaluation_runs
       SET status = 'CANCELLED', completed_at = NOW()
       WHERE id = $1`,
      [runId]
    );
    await tx.query(
      `UPDATE run_replicates
       SET status = 'SKIPPED'
       WHERE run_id = $1 AND status = 'PENDING'`,
      [runId]
    );
    return "CANCELLED";
  });
}

export async function listReplicatesForRun(runId) {
  const { rows: reps } = await pool.query(
    `SELECT id, run_id, replicate_index, seed, status, started_at, completed_at, error_message
     FROM run_replicates
     WHERE run_id = $1
     ORDER BY replicate_index ASC`,
    [runId]
  );

  if (!reps.length) return [];

  const repIds = reps.map((r) => r.id);
  const [{ rows: checks }, { rows: metrics }] = await Promise.all([
    pool.query(
      `SELECT replicate_id, check_key, passed, measured_value, threshold_value, unit, details_json
       FROM replicate_checks
       WHERE replicate_id = ANY($1::uuid[])`,
      [repIds]
    ),
    pool.query(
      `SELECT replicate_id, metric_key, value, unit, details_json
       FROM replicate_metrics
       WHERE replicate_id = ANY($1::uuid[])`,
      [repIds]
    ),
  ]);

  const checksByRep = new Map();
  for (const c of checks) {
    if (!checksByRep.has(c.replicate_id)) checksByRep.set(c.replicate_id, []);
    checksByRep.get(c.replicate_id).push({
      check_key: c.check_key,
      passed: c.passed,
      measured_value: c.measured_value,
      threshold_value: c.threshold_value,
      unit: c.unit,
      details_json: c.details_json || {},
    });
  }

  const metricsByRep = new Map();
  for (const m of metrics) {
    if (!metricsByRep.has(m.replicate_id)) metricsByRep.set(m.replicate_id, []);
    metricsByRep.get(m.replicate_id).push({
      metric_key: m.metric_key,
      value: m.value,
      unit: m.unit,
      details_json: m.details_json || {},
    });
  }

  return reps.map((r) => ({
    ...r,
    checks: checksByRep.get(r.id) || [],
    metrics: metricsByRep.get(r.id) || [],
  }));
}

export async function listRunArtifacts(runId) {
  const { rows } = await pool.query(
    `SELECT rr.id AS replicate_id, rr.replicate_index,
      a.id, a.project_id, a.kind, a.file_name, a.mime_type, a.byte_size, a.sha256, a.metadata_json, a.created_at
     FROM run_replicates rr
     JOIN replicate_artifacts ra ON ra.replicate_id = rr.id
     JOIN assets a ON a.id = ra.asset_id
     WHERE rr.run_id = $1
     ORDER BY rr.replicate_index ASC, a.created_at ASC`,
    [runId]
  );

  return rows.map((row) => ({
    replicate_id: row.replicate_id,
    replicate_index: row.replicate_index,
    asset: {
      id: row.id,
      project_id: row.project_id,
      kind: row.kind,
      file_name: row.file_name,
      mime_type: row.mime_type,
      byte_size: row.byte_size,
      sha256: row.sha256,
      metadata_json: row.metadata_json || {},
      created_at: row.created_at,
    },
  }));
}

export async function saveRunJudgment(runId, judgment) {
  const { rows } = await pool.query(
    `INSERT INTO run_judgments
      (run_id, judge_type, judge_model, confidence, verdict, result_json)
     VALUES ($1, $2, $3, $4, $5, $6::jsonb)
     RETURNING id, run_id, judge_type, judge_model, confidence, verdict, result_json, created_at`,
    [
      runId,
      judgment.judge_type,
      judgment.judge_model,
      judgment.confidence,
      judgment.verdict,
      JSON.stringify(judgment.result_json || {}),
    ]
  );
  return rows[0];
}

export async function getLatestRunJudgment(runId) {
  const { rows } = await pool.query(
    `SELECT id, run_id, judge_type, judge_model, confidence, verdict, result_json, created_at
     FROM run_judgments
     WHERE run_id = $1
     ORDER BY created_at DESC
     LIMIT 1`,
    [runId]
  );
  return rows[0] || null;
}

export function computeAggregate(run, replicates) {
  const completed = replicates.filter(
    (r) => r.status === "SUCCEEDED" || r.status === "FAILED"
  );
  const passedReplicates = completed.filter((r) => {
    if (!r.checks.length) return false;
    return r.checks.every((c) => c.passed);
  });

  const avg = (key) => {
    const values = [];
    for (const rep of replicates) {
      const m = rep.metrics.find((x) => x.metric_key === key);
      if (m) values.push(Number(m.value));
    }
    if (!values.length) return null;
    return values.reduce((a, b) => a + b, 0) / values.length;
  };

  return {
    requested_replicates: run.requested_replicates,
    completed_replicates: completed.length,
    passed_replicates: passedReplicates.length,
    pass_rate:
      run.requested_replicates > 0
        ? passedReplicates.length / run.requested_replicates
        : 0,
    avg_chamfer_mm: avg("chamfer_mean"),
    avg_hausdorff_p95_mm: avg("hausdorff_p95"),
    avg_volume_delta_percent: avg("volume_delta_percent"),
  };
}

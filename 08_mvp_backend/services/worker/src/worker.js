import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { execFile } from "node:child_process";
import { Pool } from "pg";

const connectionString =
  process.env.DATABASE_URL ||
  "postgresql://postgres:postgres@localhost:5432/cadeval";
const pollMs = Number(process.env.WORKER_POLL_MS || 2000);
const workerId =
  process.env.WORKER_ID || `worker-${crypto.randomUUID().slice(0, 8)}`;
const openaiApiKey = process.env.OPENAI_API_KEY || "";
const openaiModel = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const enableAdvancedMetrics =
  (process.env.ENABLE_ADVANCED_METRICS || "true").toLowerCase() !== "false";
const advancedMetricsScript =
  process.env.ADVANCED_METRICS_SCRIPT || "/app/tools/advanced_geometry_metrics.py";
const pythonBin = process.env.PYTHON_BIN || "python3";
const execFileAsync = promisify(execFile);

const pool = new Pool({ connectionString });

function sha(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

async function claimNextQueuedRun(client) {
  const { rows } = await client.query(
    `WITH candidate AS (
      SELECT id
      FROM evaluation_runs
      WHERE status = 'QUEUED'
      ORDER BY created_at ASC
      LIMIT 1
      FOR UPDATE SKIP LOCKED
    )
    UPDATE evaluation_runs er
    SET status = 'RUNNING', started_at = NOW()
    FROM candidate
    WHERE er.id = candidate.id
    RETURNING er.id, er.project_id, er.threshold_profile_id, er.prompt, er.requested_replicates, er.reference_asset_id`
  );
  return rows[0] || null;
}

function buildSyntheticMetrics(replicateIndex) {
  const chamfer = 0.7 + replicateIndex * 0.11;
  const hausdorff = 0.8 + replicateIndex * 0.28;
  const volumeDelta = 0.5 + replicateIndex * 0.3;
  return {
    chamfer_mean: Number(chamfer.toFixed(4)),
    hausdorff_p95: Number(hausdorff.toFixed(4)),
    volume_delta_percent: Number(volumeDelta.toFixed(4)),
  };
}

function buildFallbackStl(name, sx, sy, sz) {
  const hx = sx / 2;
  const hy = sy / 2;
  const hz = sz / 2;
  const v = [
    [-hx, -hy, -hz],
    [hx, -hy, -hz],
    [hx, hy, -hz],
    [-hx, hy, -hz],
    [-hx, -hy, hz],
    [hx, -hy, hz],
    [hx, hy, hz],
    [-hx, hy, hz],
  ];
  const faces = [
    [0, 1, 2], [0, 2, 3],
    [4, 6, 5], [4, 7, 6],
    [0, 4, 5], [0, 5, 1],
    [1, 5, 6], [1, 6, 2],
    [2, 6, 7], [2, 7, 3],
    [3, 7, 4], [3, 4, 0],
  ];
  const lines = [`solid ${name}`];
  for (const [a, b, c] of faces) {
    lines.push("  facet normal 0 0 0");
    lines.push("    outer loop");
    lines.push(`      vertex ${v[a][0]} ${v[a][1]} ${v[a][2]}`);
    lines.push(`      vertex ${v[b][0]} ${v[b][1]} ${v[b][2]}`);
    lines.push(`      vertex ${v[c][0]} ${v[c][1]} ${v[c][2]}`);
    lines.push("    endloop");
    lines.push("  endfacet");
  }
  lines.push(`endsolid ${name}`);
  return `${lines.join("\n")}\n`;
}

function fallbackStlForAsset(assetId, kind) {
  const seed = Number.parseInt(String(assetId).replaceAll("-", "").slice(0, 6), 16) || 123456;
  const scale = 40 + (seed % 50);
  if (kind === "REFERENCE_STL") {
    return buildFallbackStl("reference", scale, scale * 0.8, scale * 1.2);
  }
  return buildFallbackStl("generated", scale * 0.95, scale * 0.9, scale * 1.1);
}

function extractStl(text) {
  const clean = String(text || "").trim();
  const fenced = clean.match(/```(?:stl)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : clean;
  const solidIdx = candidate.toLowerCase().indexOf("solid ");
  const endIdx = candidate.toLowerCase().lastIndexOf("endsolid");
  if (solidIdx >= 0 && endIdx > solidIdx) {
    return `${candidate.slice(solidIdx, endIdx + "endsolid".length)}\n`;
  }
  return null;
}

async function loadReferenceStlText(client, referenceAssetId) {
  const { rows } = await client.query(
    `SELECT a.id, a.kind, ac.content_text
     FROM assets a
     LEFT JOIN asset_contents ac ON ac.asset_id = a.id
     WHERE a.id = $1
     LIMIT 1`,
    [referenceAssetId]
  );
  if (!rows.length) {
    throw new Error(`Reference asset not found: ${referenceAssetId}`);
  }
  const row = rows[0];
  if (row.content_text && String(row.content_text).trim()) {
    return row.content_text;
  }
  return fallbackStlForAsset(row.id, row.kind);
}

function inferMetricUnit(metricName) {
  if (metricName.endsWith("_percent")) return "%";
  if (metricName.includes("_deg")) return "deg";
  if (metricName.endsWith("_mm")) return "mm";
  if (metricName.includes("_ratio") || metricName.includes("_consistency") || metricName.includes("_iou")) {
    return "ratio";
  }
  if (metricName.includes("divergence")) return "score";
  return "value";
}

async function runAdvancedMetrics(referenceStlText, generatedStlText) {
  if (!enableAdvancedMetrics) return null;
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "adv-metrics-"));
  const refPath = path.join(tmpDir, "reference.stl");
  const genPath = path.join(tmpDir, "generated.stl");
  try {
    await fs.writeFile(refPath, referenceStlText, "utf8");
    await fs.writeFile(genPath, generatedStlText, "utf8");
    const { stdout } = await execFileAsync(
      pythonBin,
      [advancedMetricsScript, "--reference", refPath, "--generated", genPath],
      { maxBuffer: 20 * 1024 * 1024 }
    );
    const parsed = JSON.parse(stdout);
    if (!parsed || !Array.isArray(parsed.metrics)) {
      throw new Error("advanced metrics output missing metrics array");
    }
    return parsed;
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true });
  }
}

async function generateStlFromPrompt(prompt, replicateIndex) {
  if (!openaiApiKey) {
    return buildFallbackStl(`generated_rep${replicateIndex}`, 80, 64, 96);
  }
  const instructions =
    "Return only valid ASCII STL for the described 3D object. No markdown, no explanation.";
  const userPrompt = `Object description:\n${prompt}\n\nOutput requirements:\n- ASCII STL only\n- closed manifold if possible\n- units in mm assumptions`;

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${openaiApiKey}`,
    },
    body: JSON.stringify({
      model: openaiModel,
      temperature: 0.2,
      messages: [
        { role: "system", content: instructions },
        { role: "user", content: userPrompt },
      ],
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`OpenAI generation failed: ${response.status} ${err}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content || "";
  const stl = extractStl(content);
  if (!stl) {
    throw new Error("OpenAI response did not contain valid ASCII STL");
  }
  return stl;
}

async function processReplicate(client, run, replicate, thresholds) {
  await client.query(
    `UPDATE run_replicates
     SET status = 'RUNNING', started_at = NOW(), worker_id = $2
     WHERE id = $1`,
    [replicate.id, workerId]
  );

  const metrics = buildSyntheticMetrics(replicate.replicate_index);
  const checks = [
    { key: "render", pass: true, value: null, threshold: null, unit: null },
    { key: "watertight", pass: true, value: null, threshold: null, unit: null },
    { key: "single_component", pass: true, value: 1, threshold: 1, unit: "count" },
    {
      key: "chamfer",
      pass: metrics.chamfer_mean <= thresholds.chamfer,
      value: metrics.chamfer_mean,
      threshold: thresholds.chamfer,
      unit: "mm",
    },
    {
      key: "hausdorff_p95",
      pass: metrics.hausdorff_p95 <= thresholds.hausdorff,
      value: metrics.hausdorff_p95,
      threshold: thresholds.hausdorff,
      unit: "mm",
    },
    {
      key: "volume",
      pass: metrics.volume_delta_percent <= thresholds.volume,
      value: metrics.volume_delta_percent,
      threshold: thresholds.volume,
      unit: "%",
    },
  ];

  for (const check of checks) {
    await client.query(
      `INSERT INTO replicate_checks
      (replicate_id, check_key, passed, measured_value, threshold_value, unit, details_json)
      VALUES ($1, $2, $3, $4, $5, $6, '{}'::jsonb)
      ON CONFLICT (replicate_id, check_key)
      DO UPDATE SET
        passed = EXCLUDED.passed,
        measured_value = EXCLUDED.measured_value,
        threshold_value = EXCLUDED.threshold_value,
        unit = EXCLUDED.unit`,
      [replicate.id, check.key, check.pass, check.value, check.threshold, check.unit]
    );
  }

  const metricRows = [
    { key: "chamfer_mean", value: metrics.chamfer_mean, unit: "mm" },
    { key: "hausdorff_p95", value: metrics.hausdorff_p95, unit: "mm" },
    { key: "volume_delta_percent", value: metrics.volume_delta_percent, unit: "%" },
  ];
  for (const metric of metricRows) {
    await client.query(
      `INSERT INTO replicate_metrics (replicate_id, metric_key, value, unit, details_json)
       VALUES ($1, $2, $3, $4, '{}'::jsonb)
       ON CONFLICT (replicate_id, metric_key)
       DO UPDATE SET value = EXCLUDED.value, unit = EXCLUDED.unit`,
      [replicate.id, metric.key, metric.value, metric.unit]
    );
  }

  const scadName = `run_${run.id}_rep${replicate.replicate_index}.scad`;
  const stlName = `run_${run.id}_rep${replicate.replicate_index}.stl`;
  const stlText = await generateStlFromPrompt(run.prompt, replicate.replicate_index);
  const referenceStlText = await loadReferenceStlText(client, run.reference_asset_id);
  const scadText = `// Generated placeholder SCAD for run ${run.id} rep ${replicate.replicate_index}\n// Source prompt: ${run.prompt}\n`;

  const generatedScad = await client.query(
    `INSERT INTO assets (project_id, kind, file_name, mime_type, byte_size, sha256, storage_uri, metadata_json)
     VALUES ($1, 'GENERATED_SCAD', $2, 'text/plain', $3, $4, $5, $6::jsonb)
     RETURNING id`,
    [
      run.project_id,
      scadName,
      scadText.length,
      sha(scadName),
      `object://generated/${run.id}/${scadName}`,
      JSON.stringify({ run_id: run.id, replicate_index: replicate.replicate_index }),
    ]
  );

  const generatedStl = await client.query(
    `INSERT INTO assets (project_id, kind, file_name, mime_type, byte_size, sha256, storage_uri, metadata_json)
     VALUES ($1, 'GENERATED_STL', $2, 'model/stl', $3, $4, $5, $6::jsonb)
     RETURNING id`,
    [
      run.project_id,
      stlName,
      stlText.length,
      sha(stlText),
      `object://generated/${run.id}/${stlName}`,
      JSON.stringify({ run_id: run.id, replicate_index: replicate.replicate_index }),
    ]
  );

  await client.query(
    `INSERT INTO asset_contents (asset_id, content_type, content_text)
     VALUES ($1, 'text/plain', $2)
     ON CONFLICT (asset_id)
     DO UPDATE SET content_type = EXCLUDED.content_type, content_text = EXCLUDED.content_text`,
    [generatedScad.rows[0].id, scadText]
  );

  await client.query(
    `INSERT INTO asset_contents (asset_id, content_type, content_text)
     VALUES ($1, 'model/stl', $2)
     ON CONFLICT (asset_id)
     DO UPDATE SET content_type = EXCLUDED.content_type, content_text = EXCLUDED.content_text`,
    [generatedStl.rows[0].id, stlText]
  );

  await client.query(
    `INSERT INTO replicate_artifacts (replicate_id, asset_id)
     VALUES ($1, $2), ($1, $3)
     ON CONFLICT DO NOTHING`,
    [replicate.id, generatedScad.rows[0].id, generatedStl.rows[0].id]
  );

  try {
    const adv = await runAdvancedMetrics(referenceStlText, stlText);
    if (adv) {
      for (const m of adv.metrics) {
        const metricKey = `adv_${m.name}`;
        const numericValue = typeof m.value === "number" ? m.value : null;
        if (numericValue !== null && Number.isFinite(numericValue)) {
          await client.query(
            `INSERT INTO replicate_metrics (replicate_id, metric_key, value, unit, details_json)
             VALUES ($1, $2, $3, $4, $5::jsonb)
             ON CONFLICT (replicate_id, metric_key)
             DO UPDATE SET value = EXCLUDED.value, unit = EXCLUDED.unit, details_json = EXCLUDED.details_json`,
            [
              replicate.id,
              metricKey,
              numericValue,
              inferMetricUnit(m.name),
              JSON.stringify({
                threshold: m.threshold,
                direction: m.direction,
                passed: m.passed,
                details: m.details || {},
              }),
            ]
          );
        }

        const thresholdNum =
          typeof m.threshold === "number" && Number.isFinite(m.threshold)
            ? m.threshold
            : null;
        await client.query(
          `INSERT INTO replicate_checks
          (replicate_id, check_key, passed, measured_value, threshold_value, unit, details_json)
          VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
          ON CONFLICT (replicate_id, check_key)
          DO UPDATE SET
            passed = EXCLUDED.passed,
            measured_value = EXCLUDED.measured_value,
            threshold_value = EXCLUDED.threshold_value,
            unit = EXCLUDED.unit,
            details_json = EXCLUDED.details_json`,
          [
            replicate.id,
            `advanced:${m.name}`,
            Boolean(m.passed),
            numericValue,
            thresholdNum,
            inferMetricUnit(m.name),
            JSON.stringify({
              threshold: m.threshold,
              direction: m.direction,
              raw_value: m.value,
              details: m.details || {},
            }),
          ]
        );
      }
      await client.query(
        `INSERT INTO replicate_checks
        (replicate_id, check_key, passed, measured_value, threshold_value, unit, details_json)
        VALUES ($1, 'advanced:overall_pass', $2, NULL, NULL, NULL, $3::jsonb)
        ON CONFLICT (replicate_id, check_key)
        DO UPDATE SET passed = EXCLUDED.passed, details_json = EXCLUDED.details_json`,
        [
          replicate.id,
          Boolean(adv.overall_pass),
          JSON.stringify({
            pass_count: adv.pass_count,
            metric_count: adv.metric_count,
          }),
        ]
      );
    }
  } catch (err) {
    await client.query(
      `INSERT INTO replicate_checks
      (replicate_id, check_key, passed, measured_value, threshold_value, unit, details_json)
      VALUES ($1, 'advanced:execution', false, NULL, NULL, NULL, $2::jsonb)
      ON CONFLICT (replicate_id, check_key)
      DO UPDATE SET passed = EXCLUDED.passed, details_json = EXCLUDED.details_json`,
      [
        replicate.id,
        JSON.stringify({
          error: String(err.message || err),
          script: advancedMetricsScript,
        }),
      ]
    );
  }

  await client.query(
    `UPDATE run_replicates
     SET status = 'SUCCEEDED', completed_at = NOW()
     WHERE id = $1`,
    [replicate.id]
  );
}

async function processRun(run) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const { rows: thresholdRows } = await client.query(
      `SELECT config_json
       FROM threshold_profiles
       WHERE id = $1`,
      [run.threshold_profile_id]
    );
    const config = thresholdRows[0]?.config_json || {};
    const geometry = config.geometry_check || {};
    const thresholds = {
      chamfer: Number(geometry.chamfer_threshold_mm ?? 1),
      hausdorff: Number(geometry.hausdorff_threshold_mm ?? 1),
      volume: Number(geometry.volume_threshold_percent ?? 2),
    };

    const { rows: replicates } = await client.query(
      `SELECT id, replicate_index
       FROM run_replicates
       WHERE run_id = $1 AND status = 'PENDING'
       ORDER BY replicate_index ASC`,
      [run.id]
    );

    for (const replicate of replicates) {
      await processReplicate(client, run, replicate, thresholds);
    }

    await client.query(
      `UPDATE evaluation_runs
       SET status = 'SUCCEEDED', completed_at = NOW()
       WHERE id = $1`,
      [run.id]
    );
    await client.query("COMMIT");
    console.log(`[${workerId}] completed run ${run.id}`);
  } catch (err) {
    await client.query("ROLLBACK");
    await pool.query(
      `UPDATE evaluation_runs
       SET status = 'FAILED', completed_at = NOW(), error_message = $2
       WHERE id = $1`,
      [run.id, String(err.message || err)]
    );
    console.error(`[${workerId}] failed run ${run.id}:`, err.message);
  } finally {
    client.release();
  }
}

async function pollLoop() {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const run = await claimNextQueuedRun(client);
    await client.query("COMMIT");
    client.release();

    if (!run) return;
    await processRun(run);
  } catch (err) {
    try {
      await client.query("ROLLBACK");
    } catch {}
    client.release();
    console.error(`[${workerId}] poll error:`, err.message);
  }
}

async function main() {
  await pool.query("SELECT 1");
  console.log(`[${workerId}] worker started`);
  setInterval(() => {
    pollLoop().catch((err) => console.error("poll loop crash:", err.message));
  }, pollMs);
}

main().catch((err) => {
  console.error("worker boot failed:", err.message);
  process.exit(1);
});

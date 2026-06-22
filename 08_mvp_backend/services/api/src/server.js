import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  listModels,
  listThresholdProfiles,
  createAssetUpload,
  getAsset,
  createRun,
  listRuns,
  getRunDetail,
  getRun,
  cancelRun,
  listReplicatesForRun,
  listRunArtifacts,
  saveRunJudgment,
  getLatestRunJudgment,
  computeAggregate,
} from "./repo.js";
import { pool } from "./db.js";

const app = express();
app.use(express.json({ limit: "2mb" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const publicDir = path.join(__dirname, "..", "public");
app.use(express.static(publicDir));
const openaiApiKey = process.env.OPENAI_API_KEY || "";
const openaiModel = process.env.OPENAI_MODEL || "gpt-4.1-mini";

function buildBoxStl(name, sx, sy, sz) {
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
    const [ax, ay, az] = v[a];
    const [bx, by, bz] = v[b];
    const [cx, cy, cz] = v[c];
    lines.push("  facet normal 0 0 0");
    lines.push("    outer loop");
    lines.push(`      vertex ${ax} ${ay} ${az}`);
    lines.push(`      vertex ${bx} ${by} ${bz}`);
    lines.push(`      vertex ${cx} ${cy} ${cz}`);
    lines.push("    endloop");
    lines.push("  endfacet");
  }
  lines.push(`endsolid ${name}`);
  return `${lines.join("\n")}\n`;
}

function stlForAsset(asset) {
  const seed = parseInt(asset.id.replace(/-/g, "").slice(0, 6), 16);
  const scale = 40 + (seed % 50);
  if (asset.kind === "REFERENCE_STL") {
    return buildBoxStl("reference", scale, scale * 0.8, scale * 1.2);
  }
  return buildBoxStl("generated", scale * 0.95, scale * 0.9, scale * 1.1);
}

function summarizeRunForJudge(run, aggregate, replicates) {
  const replicateSummary = replicates.map((r) => {
    const failedChecks = (r.checks || [])
      .filter((c) => c.passed === false)
      .map((c) => c.check_key);
    const keyMetrics = {};
    for (const m of r.metrics || []) {
      if (
        ["chamfer_mean", "hausdorff_p95", "volume_delta_percent"].includes(
          m.metric_key
        ) ||
        String(m.metric_key).startsWith("adv_")
      ) {
        keyMetrics[m.metric_key] = m.value;
      }
    }
    return {
      replicate_index: r.replicate_index,
      status: r.status,
      failed_checks: failedChecks,
      key_metrics: keyMetrics,
    };
  });

  return {
    run_id: run.id,
    prompt: run.prompt,
    status: run.status,
    aggregate,
    replicates: replicateSummary,
  };
}

function normalizeJudgment(raw) {
  const verdict = ["match", "partial", "mismatch"].includes(raw?.verdict)
    ? raw.verdict
    : "partial";
  const confidence = Number(raw?.confidence);
  const clampedConfidence = Number.isFinite(confidence)
    ? Math.max(0, Math.min(1, confidence))
    : 0.5;
  const reasons = Array.isArray(raw?.reasons) ? raw.reasons.slice(0, 8) : [];
  const suggested_fixes = Array.isArray(raw?.suggested_fixes)
    ? raw.suggested_fixes.slice(0, 8)
    : [];
  const critical_issues = Array.isArray(raw?.critical_issues)
    ? raw.critical_issues.slice(0, 8)
    : [];
  return {
    verdict,
    confidence: clampedConfidence,
    reasons,
    suggested_fixes,
    critical_issues,
  };
}

function heuristicJudgment(summary) {
  const passRate = Number(summary.aggregate?.pass_rate ?? 0);
  const haus = Number(summary.aggregate?.avg_hausdorff_p95_mm ?? 999);
  const chamfer = Number(summary.aggregate?.avg_chamfer_mm ?? 999);
  const vol = Number(summary.aggregate?.avg_volume_delta_percent ?? 999);
  let verdict = "mismatch";
  let confidence = 0.65;
  if (passRate >= 0.8) {
    verdict = "match";
    confidence = 0.85;
  } else if (passRate >= 0.4) {
    verdict = "partial";
    confidence = 0.7;
  }
  const reasons = [
    `Pass rate is ${(passRate * 100).toFixed(1)}%.`,
    `Average Hausdorff p95 is ${haus.toFixed(4)} mm.`,
    `Average Chamfer is ${chamfer.toFixed(4)} mm and volume delta is ${vol.toFixed(4)}%.`,
  ];
  const suggested_fixes = [
    "Tighten geometric adherence in high-error regions.",
    "Prioritize reducing Hausdorff p95 outliers first.",
    "Review prompt constraints for wall thickness and key dimensions.",
  ];
  return {
    verdict,
    confidence,
    reasons,
    suggested_fixes,
    critical_issues: passRate < 0.5 ? ["Low replicate pass rate"] : [],
  };
}

function extractJsonFromText(content) {
  const text = String(content || "").trim();
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1] : text;
  return JSON.parse(candidate);
}

async function llmJudgment(summary) {
  if (!openaiApiKey) {
    return {
      judge_type: "heuristic",
      judge_model: "heuristic-v1",
      ...heuristicJudgment(summary),
    };
  }

  const system = [
    "You are a CAD evaluation judge.",
    "Return ONLY valid JSON with keys:",
    "verdict (match|partial|mismatch), confidence (0..1), reasons (array of strings), suggested_fixes (array of strings), critical_issues (array of strings).",
    "Use provided metrics/check outcomes. Be concise and objective.",
  ].join(" ");

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${openaiApiKey}`,
    },
    body: JSON.stringify({
      model: openaiModel,
      temperature: 0.1,
      messages: [
        { role: "system", content: system },
        {
          role: "user",
          content: `Evaluate this CAD run summary:\n${JSON.stringify(summary)}`,
        },
      ],
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`OpenAI judge failed: ${response.status} ${errText}`);
  }
  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content || "";
  const parsed = extractJsonFromText(content);
  return {
    judge_type: "llm",
    judge_model: openaiModel,
    ...normalizeJudgment(parsed),
  };
}

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.get("/v1/models", (_req, res) => {
  listModels()
    .then((items) => res.json({ items }))
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.get("/v1/threshold-profiles", (req, res) => {
  const { project_id } = req.query;
  if (!project_id) {
    return res.status(400).json({ error: "project_id is required" });
  }

  listThresholdProfiles(project_id)
    .then((items) => res.json({ items }))
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.post("/v1/assets/uploads", (req, res) => {
  const {
    project_id,
    kind,
    file_name,
    mime_type,
    byte_size,
    sha256,
    metadata_json = {},
  } = req.body ?? {};

  if (!project_id || !kind || !file_name || !mime_type || !byte_size || !sha256) {
    return res.status(400).json({
      error:
        "project_id, kind, file_name, mime_type, byte_size, sha256 are required",
    });
  }

  createAssetUpload({
    project_id,
    kind,
    file_name,
    mime_type,
    byte_size,
    sha256,
    metadata_json,
  })
    .then((asset) =>
      res.status(201).json({
        asset,
        upload_url: `https://object-store.example/upload/${asset.id}`,
        upload_headers: {
          "content-type": mime_type,
          "x-sha256": sha256,
        },
      })
    )
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.get("/v1/assets/:asset_id", (req, res) => {
  getAsset(req.params.asset_id)
    .then((asset) => {
      if (!asset) return res.status(404).json({ error: "asset not found" });
      return res.json(asset);
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.post("/v1/assets/:asset_id/download-url", (req, res) => {
  getAsset(req.params.asset_id)
    .then((asset) => {
      if (!asset) return res.status(404).json({ error: "asset not found" });
      return res.json({
        download_url: `https://object-store.example/download/${asset.id}`,
        expires_in_seconds: 900,
      });
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.get("/v1/assets/:asset_id/stl", (req, res) => {
  getAsset(req.params.asset_id)
    .then(async (asset) => {
      if (!asset) return res.status(404).json({ error: "asset not found" });
      if (!["REFERENCE_STL", "GENERATED_STL"].includes(asset.kind)) {
        return res.status(400).json({ error: "asset is not an STL kind" });
      }

      const content = await pool.query(
        `SELECT content_text, content_type
         FROM asset_contents
         WHERE asset_id = $1`,
        [asset.id]
      );

      if (content.rowCount > 0) {
        res.type(content.rows[0].content_type || "model/stl").send(content.rows[0].content_text);
        return null;
      }

      const fallbackStl = stlForAsset(asset);
      res.type("model/stl").send(fallbackStl);
      return null;
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.post("/v1/runs", (req, res) => {
  const {
    project_id,
    model_id,
    threshold_profile_id,
    prompt,
    prompt_template = "default",
    requested_replicates = 1,
    reference_asset_id,
    run_config_json = {},
    external_id = null,
  } = req.body ?? {};

  if (
    !project_id ||
    !model_id ||
    !threshold_profile_id ||
    !prompt ||
    !requested_replicates ||
    !reference_asset_id
  ) {
    return res.status(400).json({
      error:
        "project_id, model_id, threshold_profile_id, prompt, requested_replicates, reference_asset_id are required",
    });
  }

  createRun({
    project_id,
    model_id,
    threshold_profile_id,
    prompt,
    prompt_template,
    requested_replicates,
    reference_asset_id,
    run_config_json,
    external_id,
  })
    .then((run) => res.status(201).json(run))
    .catch((err) => {
      if (err.message.startsWith("invalid ")) {
        return res.status(400).json({ error: err.message });
      }
      return res.status(500).json({ error: err.message });
    });
});

app.get("/v1/runs", (req, res) => {
  const { project_id, status, model_id, limit = "20", cursor } = req.query;
  if (!project_id) {
    return res.status(400).json({ error: "project_id is required" });
  }

  const parsedLimit = Math.max(1, Math.min(100, Number(limit) || 20));
  listRuns({
    projectId: project_id,
    status,
    modelId: model_id,
    limit: parsedLimit,
    cursor,
  })
    .then((result) =>
      res.json({ items: result.items, next_cursor: result.nextCursor })
    )
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.get("/v1/runs/:run_id", (req, res) => {
  getRunDetail(req.params.run_id)
    .then(async (detail) => {
      if (!detail) return res.status(404).json({ error: "run not found" });
      const replicates = await listReplicatesForRun(detail.id);
      const aggregate = computeAggregate(detail, replicates);
      return res.json({ ...detail, aggregate });
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.delete("/v1/runs/:run_id", (req, res) => {
  cancelRun(req.params.run_id)
    .then((status) => {
      if (status === null) return res.status(404).json({ error: "run not found" });
      if (status === "TERMINAL") {
        return res.status(409).json({ error: "run already terminal" });
      }
      return res.status(202).json({ status });
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.get("/v1/runs/:run_id/replicates", (req, res) => {
  getRun(req.params.run_id)
    .then(async (run) => {
      if (!run) return res.status(404).json({ error: "run not found" });
      const items = await listReplicatesForRun(run.id);
      return res.json({ items });
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.get("/v1/runs/:run_id/results", (req, res) => {
  getRun(req.params.run_id)
    .then(async (run) => {
      if (!run) return res.status(404).json({ error: "run not found" });
      const replicates = await listReplicatesForRun(run.id);
      const aggregate = computeAggregate(run, replicates);
      return res.json({
        run_id: run.id,
        status: run.status,
        aggregate,
        replicates,
      });
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.get("/v1/runs/:run_id/artifacts", (req, res) => {
  getRun(req.params.run_id)
    .then(async (run) => {
      if (!run) return res.status(404).json({ error: "run not found" });
      const items = await listRunArtifacts(run.id);
      return res.json({ items });
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.get("/v1/runs/:run_id/judge", (req, res) => {
  getRun(req.params.run_id)
    .then(async (run) => {
      if (!run) return res.status(404).json({ error: "run not found" });
      const judgment = await getLatestRunJudgment(run.id);
      if (!judgment) return res.status(404).json({ error: "no judgment found" });
      return res.json(judgment);
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

app.post("/v1/runs/:run_id/judge", (req, res) => {
  getRun(req.params.run_id)
    .then(async (run) => {
      if (!run) return res.status(404).json({ error: "run not found" });
      const replicates = await listReplicatesForRun(run.id);
      const aggregate = computeAggregate(run, replicates);
      const summary = summarizeRunForJudge(run, aggregate, replicates);

      let result;
      try {
        result = await llmJudgment(summary);
      } catch (err) {
        result = {
          judge_type: "heuristic",
          judge_model: "heuristic-fallback",
          ...heuristicJudgment(summary),
          error: String(err.message || err),
        };
      }

      const saved = await saveRunJudgment(run.id, {
        judge_type: result.judge_type,
        judge_model: result.judge_model,
        confidence: result.confidence,
        verdict: result.verdict,
        result_json: {
          reasons: result.reasons,
          suggested_fixes: result.suggested_fixes,
          critical_issues: result.critical_issues,
          summary,
          error: result.error || null,
        },
      });
      return res.status(201).json(saved);
    })
    .catch((err) => res.status(500).json({ error: err.message }));
});

const port = Number(process.env.PORT || 8080);
pool
  .query("SELECT 1")
  .then(() => {
    app.listen(port, () => {
      console.log(`CadEval MVP API listening on :${port}`);
    });
  })
  .catch((err) => {
    console.error("Database connection failed:", err.message);
    process.exit(1);
  });

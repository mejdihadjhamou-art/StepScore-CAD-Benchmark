import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/+esm";
import { STLLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/STLLoader.js/+esm";

function byId(id) {
  return document.getElementById(id);
}

async function sha256Hex(file) {
  const buf = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const bytes = Array.from(new Uint8Array(digest));
  return bytes.map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
  return body;
}

function createViewer(containerId, colorHex) {
  const container = byId(containerId);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf2f6ff);

  const camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / container.clientHeight,
    0.1,
    2000
  );
  camera.position.set(120, 90, 120);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  const key = new THREE.DirectionalLight(0xffffff, 0.8);
  key.position.set(50, 80, 60);
  scene.add(ambient, key);

  const grid = new THREE.GridHelper(220, 18, 0xb9c3da, 0xd8deec);
  scene.add(grid);

  const loader = new STLLoader();
  let mesh = null;

  function render() {
    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  render();

  function fitCamera(geometry) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 50;
    const dist = maxDim * 2.2;
    camera.position.set(center.x + dist, center.y + dist * 0.8, center.z + dist);
    camera.lookAt(center);
  }

  function setGeometry(geometry) {
    if (mesh) scene.remove(mesh);
    const material = new THREE.MeshStandardMaterial({
      color: colorHex,
      metalness: 0.1,
      roughness: 0.65,
    });
    mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    fitCamera(geometry);
  }

  return {
    async loadFromArrayBuffer(arrayBuffer) {
      const geometry = loader.parse(arrayBuffer);
      setGeometry(geometry);
    },
    async loadFromText(text) {
      const buffer = new TextEncoder().encode(text).buffer;
      const geometry = loader.parse(buffer);
      setGeometry(geometry);
    },
  };
}

async function loadModels() {
  const data = await jsonFetch("/v1/models");
  if (!data.items.length) throw new Error("No models available");
  return data.items[0];
}

async function loadDefaultThreshold(projectId) {
  const data = await jsonFetch(
    `/v1/threshold-profiles?project_id=${encodeURIComponent(projectId)}`
  );
  if (!data.items.length) throw new Error("No threshold profile found");
  return data.items[0];
}

async function createAsset(projectId, file) {
  let fileName = "mug_ref.stl";
  let byteSize = 65432;
  let sha256 =
    "31d30eea8d0968d6458e0ad0027c9f80c0dc70f6f4f3d4f064f5708fd7690f9f";

  if (file) {
    fileName = file.name;
    byteSize = file.size;
    sha256 = await sha256Hex(file);
  }

  const payload = {
    project_id: projectId,
    kind: "REFERENCE_STL",
    file_name: fileName,
    mime_type: "model/stl",
    byte_size: byteSize,
    sha256,
    metadata_json: { units: "mm" },
  };
  const data = await jsonFetch("/v1/assets/uploads", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  return data.asset;
}

let pollTimer = null;
let currentReferenceAssetId = null;
let currentReferenceFile = null;
const referenceViewer = createViewer("reference-viewer", 0x2a6ddf);
const generatedViewer = createViewer("generated-viewer", 0xe67e22);

function badge(text, kind) {
  const cls =
    kind === "pass" ? "badge badge-pass" : kind === "fail" ? "badge badge-fail" : "badge badge-neutral";
  return `<span class="${cls}">${text}</span>`;
}

function formatMetric(v, digits = 4) {
  if (v === null || v === undefined) return "-";
  if (typeof v !== "number") return String(v);
  return Number.isFinite(v) ? v.toFixed(digits) : String(v);
}

function renderDashboard(data) {
  const status = data?.status || "-";
  const aggregate = data?.aggregate || {};
  const reps = Array.isArray(data?.replicates) ? data.replicates : [];
  const allChecks = reps.flatMap((r) => r.checks || []);
  const allMetrics = reps.flatMap((r) => r.metrics || []);

  byId("stat-status").innerHTML = badge(
    status,
    status === "SUCCEEDED" ? "pass" : status === "FAILED" ? "fail" : "neutral"
  );
  byId("stat-pass-rate").textContent =
    aggregate.pass_rate !== undefined ? `${(aggregate.pass_rate * 100).toFixed(1)}%` : "-";
  byId("stat-replicates").textContent =
    aggregate.requested_replicates !== undefined
      ? `${aggregate.completed_replicates ?? 0}/${aggregate.requested_replicates}`
      : "-";
  const advCount = allMetrics.filter((m) => String(m.metric_key || "").startsWith("adv_")).length;
  byId("stat-advanced").textContent = `${advCount} points`;

  const keyRows = [
    ["Avg Chamfer (mm)", formatMetric(aggregate.avg_chamfer_mm)],
    ["Avg Hausdorff p95 (mm)", formatMetric(aggregate.avg_hausdorff_p95_mm)],
    ["Avg Volume Delta (%)", formatMetric(aggregate.avg_volume_delta_percent)],
  ];
  const advOverall = allChecks.filter((c) => c.check_key === "advanced:overall_pass");
  if (advOverall.length) {
    const advPass = advOverall.filter((c) => c.passed).length;
    keyRows.push(["Advanced Overall", `${advPass}/${advOverall.length} passed`]);
  }
  byId("key-metrics-body").innerHTML = keyRows
    .map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`)
    .join("");

  const metricMap = new Map();
  for (const m of allMetrics) {
    const key = m.metric_key || "unknown_metric";
    const value = Number(m.value);
    if (!Number.isFinite(value)) continue;
    if (!metricMap.has(key)) metricMap.set(key, []);
    metricMap.get(key).push(value);
  }
  const metricRows = [...metricMap.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([key, values]) => {
      const sum = values.reduce((acc, v) => acc + v, 0);
      const avg = sum / values.length;
      const min = Math.min(...values);
      const max = Math.max(...values);
      return `<tr>
        <td>${key}</td>
        <td>${formatMetric(avg)}</td>
        <td>${formatMetric(min)}</td>
        <td>${formatMetric(max)}</td>
        <td>${values.length}</td>
      </tr>`;
    });
  byId("all-metrics-body").innerHTML =
    metricRows.length > 0
      ? metricRows.join("")
      : `<tr><td colspan="5">No metrics yet.</td></tr>`;

  const checkMap = new Map();
  for (const c of allChecks) {
    const key = c.check_key || "unknown";
    if (!checkMap.has(key)) checkMap.set(key, { pass: 0, fail: 0 });
    if (c.passed) checkMap.get(key).pass += 1;
    else checkMap.get(key).fail += 1;
  }
  const checkRows = [...checkMap.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(
      ([key, v]) =>
        `<tr><td>${key}</td><td>${badge(String(v.pass), "pass")}</td><td>${badge(
          String(v.fail),
          v.fail > 0 ? "fail" : "neutral"
        )}</td></tr>`
    );
  byId("checks-body").innerHTML =
    checkRows.length > 0 ? checkRows.join("") : `<tr><td colspan="3">No checks yet.</td></tr>`;

  const repRows = reps.map((r) => {
    const checks = r.checks || [];
    const passed = checks.filter((c) => c.passed).length;
    const total = checks.length;
    const statusKind = r.status === "SUCCEEDED" ? "pass" : r.status === "FAILED" ? "fail" : "neutral";
    return `<tr>
      <td>#${r.replicate_index}</td>
      <td>${badge(r.status || "-", statusKind)}</td>
      <td>${passed}</td>
      <td>${total}</td>
    </tr>`;
  });
  byId("replicates-body").innerHTML =
    repRows.length > 0 ? repRows.join("") : `<tr><td colspan="4">No replicates yet.</td></tr>`;
}

function renderJudge(judgment) {
  if (!judgment) {
    byId("judge-verdict").textContent = "-";
    byId("judge-confidence").textContent = "-";
    byId("judge-model").textContent = "-";
    byId("judge-reasons").textContent = "-";
    byId("judge-fixes").textContent = "-";
    return;
  }
  const verdict = judgment.verdict || "-";
  const verdictKind =
    verdict === "match" ? "pass" : verdict === "mismatch" ? "fail" : "neutral";
  byId("judge-verdict").innerHTML = badge(verdict, verdictKind);
  byId("judge-confidence").textContent = formatMetric(judgment.confidence, 3);
  byId("judge-model").textContent = judgment.judge_model || judgment.judge_type || "-";
  const reasons = judgment.result_json?.reasons || [];
  const fixes = judgment.result_json?.suggested_fixes || [];
  byId("judge-reasons").innerHTML = reasons.length
    ? `<ul>${reasons.map((x) => `<li>${x}</li>`).join("")}</ul>`
    : "-";
  byId("judge-fixes").innerHTML = fixes.length
    ? `<ul>${fixes.map((x) => `<li>${x}</li>`).join("")}</ul>`
    : "-";
}

async function loadLatestJudgment(runId) {
  try {
    const data = await jsonFetch(`/v1/runs/${encodeURIComponent(runId)}/judge`);
    renderJudge(data);
    byId("judge-status").textContent = `Loaded judgment at ${new Date(
      data.created_at
    ).toLocaleString()}`;
  } catch (err) {
    renderJudge(null);
    byId("judge-status").textContent = "No judgment stored yet.";
  }
}

async function loadResults(runId) {
  if (!runId) throw new Error("Run ID is required");
  const data = await jsonFetch(`/v1/runs/${encodeURIComponent(runId)}/results`);
  byId("results").textContent = JSON.stringify(data, null, 2);
  renderDashboard(data);
  await loadLatestJudgment(runId);
  return data;
}

async function updateViewers(runId) {
  if (currentReferenceFile) {
    const buf = await currentReferenceFile.arrayBuffer();
    await referenceViewer.loadFromArrayBuffer(buf);
  } else if (currentReferenceAssetId) {
    const stl = await fetch(
      `/v1/assets/${encodeURIComponent(currentReferenceAssetId)}/stl`
    ).then((r) => r.text());
    await referenceViewer.loadFromText(stl);
  } else {
    const runDetail = await jsonFetch(`/v1/runs/${encodeURIComponent(runId)}`);
    if (runDetail.reference_asset?.id) {
      currentReferenceAssetId = runDetail.reference_asset.id;
      const stl = await fetch(
        `/v1/assets/${encodeURIComponent(currentReferenceAssetId)}/stl`
      ).then((r) => r.text());
      await referenceViewer.loadFromText(stl);
    }
  }

  const artifactData = await jsonFetch(`/v1/runs/${encodeURIComponent(runId)}/artifacts`);
  const generated = artifactData.items.find((x) => x.asset.kind === "GENERATED_STL");
  if (generated?.asset?.id) {
    const stl = await fetch(
      `/v1/assets/${encodeURIComponent(generated.asset.id)}/stl`
    ).then((r) => r.text());
    await generatedViewer.loadFromText(stl);
  }
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  byId("toggle-poll").textContent = "Start Polling";
}

function startPolling() {
  const runId = byId("run-id").value.trim();
  if (!runId) {
    byId("form-status").textContent = "Enter a run id first.";
    return;
  }
  stopPolling();
  byId("toggle-poll").textContent = "Stop Polling";
  pollTimer = setInterval(async () => {
    try {
      const result = await loadResults(runId);
      await updateViewers(runId);
      if (["SUCCEEDED", "FAILED", "CANCELLED"].includes(result.status)) {
        stopPolling();
      }
    } catch (err) {
      byId("form-status").textContent = err.message;
      stopPolling();
    }
  }, 2000);
}

byId("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const statusEl = byId("form-status");
  statusEl.textContent = "Creating run...";

  try {
    const projectId = byId("project-id").value.trim();
    const prompt = byId("prompt").value.trim();
    const replicates = Number(byId("replicates").value);
    const file = byId("reference-file").files[0] || null;
    currentReferenceFile = file;

    const [model, threshold, asset] = await Promise.all([
      loadModels(),
      loadDefaultThreshold(projectId),
      createAsset(projectId, file),
    ]);
    currentReferenceAssetId = asset.id;

    if (file) {
      try {
        const buf = await file.arrayBuffer();
        await referenceViewer.loadFromArrayBuffer(buf);
      } catch (err) {
        statusEl.textContent = `Reference STL parse failed: ${err.message}`;
      }
    }

    const run = await jsonFetch("/v1/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        project_id: projectId,
        model_id: model.id,
        threshold_profile_id: threshold.id,
        prompt,
        requested_replicates: replicates,
        reference_asset_id: asset.id,
      }),
    });

    byId("run-id").value = run.id;
    statusEl.textContent = `Run created: ${run.id}`;
    const results = await loadResults(run.id);
    await updateViewers(run.id);
    if (results.status !== "SUCCEEDED") startPolling();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
});

byId("load-results").addEventListener("click", async () => {
  const runId = byId("run-id").value.trim();
  try {
    await loadResults(runId);
    await updateViewers(runId);
    byId("form-status").textContent = "Results loaded.";
  } catch (err) {
    byId("form-status").textContent = `Error: ${err.message}`;
  }
});

byId("toggle-poll").addEventListener("click", () => {
  if (pollTimer) stopPolling();
  else startPolling();
});

byId("run-judge").addEventListener("click", async () => {
  const runId = byId("run-id").value.trim();
  if (!runId) {
    byId("judge-status").textContent = "Enter a run id first.";
    return;
  }
  byId("judge-status").textContent = "Running judge...";
  try {
    const judgment = await jsonFetch(`/v1/runs/${encodeURIComponent(runId)}/judge`, {
      method: "POST",
    });
    renderJudge(judgment);
    byId("judge-status").textContent = `Judgment saved at ${new Date(
      judgment.created_at
    ).toLocaleString()}`;
  } catch (err) {
    byId("judge-status").textContent = `Judge failed: ${err.message}`;
  }
});

byId("project-id").value = "a9e5fa9f-7193-4f13-ab97-01a7f3a68d4d";

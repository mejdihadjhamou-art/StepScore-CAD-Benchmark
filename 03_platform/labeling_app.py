#!/usr/bin/env python3
"""
3D Labeling Tool — Web-based side-by-side STL viewer for labeling benchmark pairs.

Usage:
    python labeling_app.py [--port 8501] [--pairs-csv threshold_tuning/pairs_for_labeling.csv]

Opens a local web app at http://localhost:8501 with:
  - Side-by-side 3D viewers (reference vs generated)
  - Metric summary panel
  - Positive / Negative / Review buttons
  - Progress tracker and keyboard shortcuts (p/n/r)
"""

import argparse
import csv
import json
import math
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_pairs(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _save_pairs(csv_path: Path, rows, fieldnames):
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _stl_for_reference(row):
    """Return path to reference STL (converted from STEP)."""
    ref_step = row.get("reference_path", "")
    if not ref_step:
        return None
    p = Path(ref_step)
    stl = p.parent.parent / "references_parametric_stl" / p.with_suffix(".stl").name
    if stl.exists():
        return str(stl)
    # fallback: same dir
    stl2 = p.with_suffix(".stl")
    return str(stl2) if stl2.exists() else None


def _stl_for_generated(row):
    """Return path to generated STL."""
    gen_stl = row.get("generated_mesh_path", "") or row.get("generated_path", "")
    if gen_stl and Path(gen_stl).exists():
        return gen_stl
    # try sibling .stl
    gen_step = row.get("generated_path", "")
    if gen_step:
        stl = Path(gen_step).with_suffix(".stl")
        if stl.exists():
            return str(stl)
    return None


KEY_METRICS = [
    ("quality_score_0_100", "Quality Score", False),
    ("volume_diff_percent", "Volume Diff %", True),
    ("voxel_iou", "Voxel IoU", False),
    ("chamfer_distance_mm", "Chamfer Dist mm", True),
    ("hausdorff_95p_mm", "Hausdorff 95p mm", True),
    ("bbox_error_max_mm", "Bbox Error Max mm", True),
    ("component_count_match", "Component Match", False),
    ("surface_area_diff_percent", "Surface Area Diff %", True),
    ("normal_consistency", "Normal Consistency", False),
    ("euler_genus_match", "Euler Genus Match", False),
    ("void_hole_count_match", "Void/Hole Match", False),
]

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>StepScore 3D Labeling Tool</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #e0e0e0; overflow: hidden; height: 100vh; }
#top-bar { display:flex; align-items:center; justify-content:space-between;
           padding: 8px 16px; background: #16213e; border-bottom: 1px solid #0f3460; height: 48px; }
#top-bar .title { font-size: 15px; font-weight: 600; color: #e94560; }
#top-bar .progress { font-size: 13px; color: #a0a0a0; }
#main { display: flex; height: calc(100vh - 48px); }
#viewers { flex: 1; display: flex; gap: 2px; background: #0f0f1a; }
.viewer-panel { flex: 1; position: relative; }
.viewer-panel canvas { width: 100% !important; height: 100% !important; display: block; }
.viewer-label { position: absolute; top: 8px; left: 12px; font-size: 13px; font-weight: 600;
                padding: 4px 10px; border-radius: 4px; z-index: 10; }
.viewer-label.ref { background: rgba(46,125,50,0.85); color: #fff; }
.viewer-label.gen { background: rgba(21,101,192,0.85); color: #fff; }
.viewer-error { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
                font-size: 14px; color: #ef5350; text-align: center; z-index: 10; display: none; }
#sidebar { width: 320px; background: #16213e; border-left: 1px solid #0f3460;
           display: flex; flex-direction: column; overflow-y: auto; }
#pair-info { padding: 12px; border-bottom: 1px solid #0f3460; }
#pair-info h3 { font-size: 14px; color: #e94560; margin-bottom: 6px; }
#pair-info .meta { font-size: 12px; color: #a0a0a0; line-height: 1.6; }
#pair-info .meta b { color: #e0e0e0; }
#metrics { padding: 12px; flex: 1; overflow-y: auto; }
#metrics h3 { font-size: 14px; color: #e94560; margin-bottom: 8px; }
.metric-row { display: flex; justify-content: space-between; align-items: center;
              padding: 4px 0; font-size: 12px; border-bottom: 1px solid #1a1a3e; }
.metric-name { color: #b0b0b0; }
.metric-val { font-weight: 600; font-family: 'SF Mono', monospace; }
.metric-val.good { color: #66bb6a; }
.metric-val.ok { color: #ffa726; }
.metric-val.bad { color: #ef5350; }
#controls { padding: 12px; border-top: 1px solid #0f3460; }
#controls h3 { font-size: 14px; color: #e94560; margin-bottom: 8px; }
.btn-row { display: flex; gap: 6px; margin-bottom: 8px; }
.btn { flex: 1; padding: 10px 0; border: none; border-radius: 6px; font-size: 13px;
       font-weight: 600; cursor: pointer; transition: all 0.15s; text-transform: uppercase; }
.btn:hover { transform: translateY(-1px); }
.btn.positive { background: #2e7d32; color: #fff; }
.btn.positive:hover { background: #388e3c; }
.btn.negative { background: #c62828; color: #fff; }
.btn.negative:hover { background: #d32f2f; }
.btn.review { background: #f57f17; color: #fff; }
.btn.review:hover { background: #f9a825; }
.btn.nav { background: #1565c0; color: #fff; font-size: 12px; }
.btn.nav:hover { background: #1976d2; }
.btn.nav:disabled { background: #333; color: #666; cursor: default; transform: none; }
.current-label { text-align: center; font-size: 12px; padding: 4px; margin-top: 4px; color: #a0a0a0; }
.current-label b { color: #e0e0e0; }
.shortcut { font-size: 11px; color: #666; text-align: center; margin-top: 6px; }
.progress-bar { height: 4px; background: #0f0f1a; border-radius: 2px; margin-top: 6px; }
.progress-fill { height: 100%; background: #e94560; border-radius: 2px; transition: width 0.3s; }
#filter-row { display: flex; gap: 4px; margin-bottom: 8px; }
#filter-row select { flex:1; padding: 4px; background: #1a1a2e; color: #e0e0e0;
                     border: 1px solid #0f3460; border-radius: 4px; font-size: 11px; }
</style>
<script async src="https://ga.jspm.io/npm:es-module-shims@1.10.1/dist/es-module-shims.js"></script>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="top-bar">
  <div class="title">StepScore 3D Labeling Tool</div>
  <div class="progress" id="progress-text">Loading...</div>
</div>
<div id="main">
  <div id="viewers">
    <div class="viewer-panel" id="ref-panel">
      <div class="viewer-label ref">Reference</div>
      <div class="viewer-error" id="ref-error">No STL file</div>
    </div>
    <div class="viewer-panel" id="gen-panel">
      <div class="viewer-label gen">Generated</div>
      <div class="viewer-error" id="gen-error">No STL file</div>
    </div>
  </div>
  <div id="sidebar">
    <div id="pair-info">
      <h3>Pair Info</h3>
      <div class="meta" id="meta-content">Loading...</div>
    </div>
    <div id="metrics">
      <h3>Key Metrics</h3>
      <div id="metrics-content"></div>
    </div>
    <div id="controls">
      <h3>Label</h3>
      <div id="filter-row">
        <select id="filter-label">
          <option value="all">All pairs</option>
          <option value="unlabeled" selected>Unlabeled only</option>
          <option value="positive">Positive</option>
          <option value="negative">Negative</option>
          <option value="review">Review</option>
        </select>
        <select id="filter-family">
          <option value="all">All families</option>
        </select>
      </div>
      <div class="btn-row">
        <button class="btn positive" onclick="labelPair('positive')">Positive (P)</button>
        <button class="btn negative" onclick="labelPair('negative')">Negative (N)</button>
      </div>
      <div class="btn-row">
        <button class="btn review" onclick="labelPair('review')">Review (R)</button>
      </div>
      <div class="btn-row">
        <button class="btn nav" id="btn-prev" onclick="navigate(-1)">&#9664; Prev</button>
        <button class="btn nav" id="btn-next" onclick="navigate(1)">Next &#9654;</button>
      </div>
      <div class="current-label" id="current-label"></div>
      <div class="shortcut">Keyboard: P / N / R to label, &larr; &rarr; to navigate</div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    </div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function createViewer(containerId, color, errorId) {
  var container = document.getElementById(containerId);
  var errorEl = document.getElementById(errorId);
  var w = container.clientWidth, h = container.clientHeight;

  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  var camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 10000);
  camera.position.set(80, 80, 80);

  var renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  var amb = new THREE.AmbientLight(0x606060, 1.0);
  scene.add(amb);
  var d1 = new THREE.DirectionalLight(0xffffff, 1.0);
  d1.position.set(100, 200, 100);
  scene.add(d1);
  var d2 = new THREE.DirectionalLight(0xffffff, 0.5);
  d2.position.set(-100, -50, -100);
  scene.add(d2);
  var d3 = new THREE.DirectionalLight(0xffffff, 0.3);
  d3.position.set(0, -100, 100);
  scene.add(d3);

  var grid = new THREE.GridHelper(200, 20, 0x333355, 0x222244);
  scene.add(grid);

  var controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;

  var mesh = null;
  var loader = new STLLoader();

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', function() {
    var w2 = container.clientWidth, h2 = container.clientHeight;
    if (w2 === 0 || h2 === 0) return;
    camera.aspect = w2/h2;
    camera.updateProjectionMatrix();
    renderer.setSize(w2, h2);
  });

  return {
    loadSTL: function(url) {
      if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); mesh = null; }
      errorEl.style.display = 'none';
      if (!url) { errorEl.style.display = 'block'; return; }

      loader.load(url, function(geometry) {
        geometry.computeBoundingBox();
        var center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        geometry.computeBoundingSphere();
        var radius = geometry.boundingSphere.radius;

        var mat = new THREE.MeshPhongMaterial({
          color: color, specular: 0x333333, shininess: 40,
          side: THREE.DoubleSide
        });
        mesh = new THREE.Mesh(geometry, mat);
        scene.add(mesh);

        controls.target.set(0, 0, 0);
        var dist = radius * 2.5;
        camera.position.set(dist * 0.7, dist * 0.8, dist * 0.7);
        controls.update();
      }, undefined, function(err) {
        console.error('STL load error:', err);
        errorEl.textContent = 'Failed to load STL';
        errorEl.style.display = 'block';
      });
    }
  };
}

// App state
var pairs = [];
var filteredIndices = [];
var currentFilteredIdx = 0;
var families = new Set();
var refViewer, genViewer;

function init() {
  refViewer = createViewer('ref-panel', 0x4caf50, 'ref-error');
  genViewer = createViewer('gen-panel', 0x2196f3, 'gen-error');
  fetchPairs();
}

function fetchPairs() {
  fetch('/api/pairs').then(function(r) { return r.json(); }).then(function(data) {
    pairs = data.pairs;
    data.families.forEach(function(f) { families.add(f); });
    var sel = document.getElementById('filter-family');
    families.forEach(function(f) {
      var o = document.createElement('option'); o.value=f; o.textContent=f; sel.appendChild(o);
    });
    applyFilter();
  });
}

window.applyFilter = function applyFilter() {
  var labelFilter = document.getElementById('filter-label').value;
  var familyFilter = document.getElementById('filter-family').value;
  filteredIndices = [];
  for (var i = 0; i < pairs.length; i++) {
    var p = pairs[i];
    if (familyFilter !== 'all' && p.family !== familyFilter) continue;
    if (labelFilter === 'unlabeled' && p.label && p.label !== 'review') continue;
    if (labelFilter === 'positive' && p.label !== 'positive') continue;
    if (labelFilter === 'negative' && p.label !== 'negative') continue;
    if (labelFilter === 'review' && p.label !== 'review') continue;
    filteredIndices.push(i);
  }
  currentFilteredIdx = 0;
  if (filteredIndices.length > 0) showPair(filteredIndices[0]);
  else updateUI();
}

document.getElementById('filter-label').addEventListener('change', applyFilter);
document.getElementById('filter-family').addEventListener('change', applyFilter);

function showPair(idx) {
  if (idx < 0 || idx >= pairs.length) return;
  var p = pairs[idx];
  refViewer.loadSTL(p.ref_stl ? '/api/stl?path=' + encodeURIComponent(p.ref_stl) : null);
  genViewer.loadSTL(p.gen_stl ? '/api/stl?path=' + encodeURIComponent(p.gen_stl) : null);
  document.getElementById('meta-content').innerHTML =
    '<b>Part:</b> ' + p.part_id + '<br>' +
    '<b>Family:</b> ' + p.family + '<br>' +
    '<b>Prompt:</b> ' + p.prompt_level + '<br>' +
    '<b>Model:</b> ' + p.model + '<br>' +
    '<b>Pair ID:</b> ' + p.pair_id;
  var html = '';
  p.key_metrics.forEach(function(m) {
    html += '<div class="metric-row"><span class="metric-name">' + m.label +
            '</span><span class="metric-val ' + m.quality + '">' + m.display + '</span></div>';
  });
  document.getElementById('metrics-content').innerHTML = html;
  updateUI();
}

function updateUI() {
  var labeled = pairs.filter(function(p) { return p.label === 'positive' || p.label === 'negative'; }).length;
  document.getElementById('progress-text').textContent =
    labeled + ' / ' + pairs.length + ' labeled (' + filteredIndices.length + ' in view)';
  document.getElementById('progress-fill').style.width = (labeled / pairs.length * 100) + '%';
  var globalIdx = filteredIndices[currentFilteredIdx];
  var p = (globalIdx !== undefined) ? pairs[globalIdx] : null;
  var lbl = p ? p.label : '';
  document.getElementById('current-label').innerHTML = p
    ? 'Current: <b>' + (lbl === 'review' ? 'UNLABELED' : (lbl || 'UNLABELED').toUpperCase()) + '</b> (' +
      (currentFilteredIdx+1) + '/' + filteredIndices.length + ')'
    : 'No pairs in view';
  document.getElementById('btn-prev').disabled = currentFilteredIdx <= 0;
  document.getElementById('btn-next').disabled = currentFilteredIdx >= filteredIndices.length - 1;
}

window.navigate = function(dir) {
  var next = currentFilteredIdx + dir;
  if (next < 0 || next >= filteredIndices.length) return;
  currentFilteredIdx = next;
  showPair(filteredIndices[currentFilteredIdx]);
}

window.labelPair = function(label) {
  var globalIdx = filteredIndices[currentFilteredIdx];
  if (globalIdx === undefined) return;
  pairs[globalIdx].label = label;
  fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ index: globalIdx, label: label })
  }).then(function() {
    var filterVal = document.getElementById('filter-label').value;
    if (filterVal === 'unlabeled') {
      applyFilter();
    } else {
      navigate(1);
    }
    updateUI();
  });
}

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'p' || e.key === 'P') window.labelPair('positive');
  else if (e.key === 'n' || e.key === 'N') window.labelPair('negative');
  else if (e.key === 'r' || e.key === 'R') window.labelPair('review');
  else if (e.key === 'ArrowLeft') window.navigate(-1);
  else if (e.key === 'ArrowRight') window.navigate(1);
});

window.addEventListener('load', init);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class LabelingHandler(SimpleHTTPRequestHandler):
    pairs_data = []
    csv_path = None
    fieldnames = []

    def log_message(self, format, *args):
        pass  # suppress request logs

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif parsed.path == "/api/pairs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            out_pairs = []
            families = set()
            for row in self.pairs_data:
                ref_stl = _stl_for_reference(row)
                gen_stl = _stl_for_generated(row)
                families.add(row.get("family", ""))

                key_metrics = []
                for col, label, lower_better in KEY_METRICS:
                    val = row.get(col, "")
                    if val == "" or val is None:
                        continue
                    try:
                        fval = float(val)
                    except (ValueError, TypeError):
                        continue
                    if math.isnan(fval) or math.isinf(fval):
                        continue
                    # quality assessment
                    q = "ok"
                    if col == "quality_score_0_100":
                        q = "good" if fval >= 70 else ("ok" if fval >= 40 else "bad")
                    elif col == "volume_diff_percent":
                        q = "good" if fval <= 2 else ("ok" if fval <= 8 else "bad")
                    elif col == "voxel_iou":
                        q = "good" if fval >= 0.85 else ("ok" if fval >= 0.65 else "bad")
                    elif col == "chamfer_distance_mm":
                        q = "good" if fval <= 1.0 else ("ok" if fval <= 3.0 else "bad")
                    elif col == "hausdorff_95p_mm":
                        q = "good" if fval <= 1.5 else ("ok" if fval <= 4.0 else "bad")
                    elif col == "bbox_error_max_mm":
                        q = "good" if fval <= 1.0 else ("ok" if fval <= 5.0 else "bad")
                    elif col == "component_count_match":
                        q = "good" if fval >= 1.0 else "bad"
                    elif col == "surface_area_diff_percent":
                        q = "good" if fval <= 3.0 else ("ok" if fval <= 8.0 else "bad")
                    elif col == "normal_consistency":
                        q = "good" if fval >= 0.95 else ("ok" if fval >= 0.85 else "bad")
                    elif col in ("euler_genus_match", "void_hole_count_match"):
                        q = "good" if fval >= 1.0 else "bad"

                    disp = f"{fval:.1f}" if col == "quality_score_0_100" else (
                        f"{fval:.2f}" if "percent" in col else (
                            f"{fval:.3f}" if "_mm" in col else f"{fval:.4f}"
                        )
                    )
                    key_metrics.append({"col": col, "label": label, "display": disp, "quality": q})

                out_pairs.append({
                    "pair_id": row.get("pair_id", ""),
                    "part_id": row.get("part_id", ""),
                    "family": row.get("family", ""),
                    "prompt_level": row.get("prompt_level", ""),
                    "model": row.get("model", ""),
                    "label": row.get("label", ""),
                    "ref_stl": ref_stl,
                    "gen_stl": gen_stl,
                    "key_metrics": key_metrics,
                })
            payload = json.dumps({"pairs": out_pairs, "families": sorted(families)})
            self.wfile.write(payload.encode("utf-8"))

        elif parsed.path == "/api/stl":
            qs = parse_qs(parsed.query)
            stl_path = qs.get("path", [None])[0]
            if stl_path and Path(stl_path).exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(stl_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/label":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            idx = body.get("index")
            label = body.get("label", "")
            if idx is not None and 0 <= idx < len(self.pairs_data):
                self.pairs_data[idx]["label"] = label
                _save_pairs(self.csv_path, self.pairs_data, self.fieldnames)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="3D Labeling Tool")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--pairs-csv",
        default="threshold_tuning/pairs_for_labeling.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.pairs_csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    rows = _load_pairs(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    LabelingHandler.pairs_data = rows
    LabelingHandler.csv_path = csv_path
    LabelingHandler.fieldnames = fieldnames

    server = HTTPServer(("127.0.0.1", args.port), LabelingHandler)
    print(f"Labeling tool running at http://localhost:{args.port}")
    print(f"CSV: {csv_path} ({len(rows)} pairs)")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

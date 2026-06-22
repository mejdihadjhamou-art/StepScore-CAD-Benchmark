#!/usr/bin/env python3
import http.server, socketserver, json, urllib.parse
from pathlib import Path
import pandas as pd

PORT = 8080
HARNESS = Path(".stepscore_harness_runs/final73_anthropic_run_02")
EXCEL = HARNESS / "labeled_pairs_for_review.xlsx"

df = pd.read_excel(EXCEL)
print(f"Loaded {len(df)} pairs")

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>StepScore 3D Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui;background:#1a1a2e}
#top{background:#16213e;color:white;padding:12px 20px;display:flex;align-items:center;gap:15px}
#top h1{font-size:18px;color:#e94560}
#top input{width:60px;padding:6px;border-radius:4px;border:none;text-align:center}
#top button{padding:6px 14px;background:#e94560;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold}
#top button:hover{background:#c73650}
#top span{color:#aaa;font-size:13px}
#mid{display:flex;flex:1;height:calc(100vh - 160px)}
.panel{flex:1;display:flex;flex-direction:column;margin:6px;background:#0f3460;border-radius:8px;overflow:hidden}
.panel h3{padding:10px 15px;margin:0;color:white;font-size:14px}
.panel:first-child h3{background:#1a7}
.panel:last-child h3{background:#e94560}
.panel canvas{flex:1;display:block}
#bot{background:#16213e;padding:12px 20px;display:flex;align-items:center;gap:15px;flex-wrap:wrap}
#bot .m{color:#ccc;font-size:12px}
#bot .m b{color:white}
#bot button{padding:10px 24px;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:14px;color:white}
.bp{background:#1a7}.bn{background:#e94560}.br{background:#e9a045}
.bp:hover{background:#159}.bn:hover{background:#c73650}.br:hover{background:#d0903a}
</style>
</head>
<body>
<div id="top">
  <h1>StepScore 3D</h1>
  <button onclick="prev()">&#9664;</button>
  <input type="number" id="pn" value="1" min="1" max="146">
  <button onclick="nxt()">&#9654;</button>
  <span id="st">Loading...</span>
  <span id="pid" style="margin-left:auto;color:#e94560;font-weight:bold"></span>
</div>
<div id="mid">
  <div class="panel"><h3>Reference</h3><canvas id="c1"></canvas></div>
  <div class="panel"><h3>Generated (Claude)</h3><canvas id="c2"></canvas></div>
</div>
<div id="bot">
  <div id="met"></div>
  <div style="margin-left:auto;display:flex;gap:8px">
    <button class="bp" onclick="lab('positive')">&#10004; POSITIVE</button>
    <button class="bn" onclick="lab('negative')">&#10008; NEGATIVE</button>
    <button class="br" onclick="lab('review')">? REVIEW</button>
  </div>
</div>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

window.THREE = THREE;
const lg = (m) => { console.log(m); };

let idx = 0;
let v = {};

function setup(id, color) {
  const canvas = document.getElementById(id);
  const w = canvas.parentElement.clientWidth;
  const h = canvas.parentElement.clientHeight - 40;
  canvas.width = w;
  canvas.height = h;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 10000);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // Lights
  const d1 = new THREE.DirectionalLight(0xffffff, 1.0);
  d1.position.set(5, 10, 7);
  scene.add(d1);
  const d2 = new THREE.DirectionalLight(0xffffff, 0.5);
  d2.position.set(-5, -3, -5);
  scene.add(d2);
  scene.add(new THREE.AmbientLight(0xffffff, 0.4));

  // Grid
  const grid = new THREE.GridHelper(200, 20, 0x333355, 0x222244);
  scene.add(grid);

  return { scene, camera, renderer, controls, color };
}

function addMesh(v, geometry) {
  // Remove old meshes
  v.scene.children.filter(c => c.isMesh).forEach(c => v.scene.remove(c));

  geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  geometry.translate(-center.x, -center.y, -center.z);

  const size = geometry.boundingBox.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);

  const material = new THREE.MeshPhongMaterial({
    color: v.color,
    specular: 0x444444,
    shininess: 60,
    flatShading: false
  });
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, material);
  v.scene.add(mesh);

  v.camera.position.set(maxDim * 1.2, maxDim * 0.8, maxDim * 1.5);
  v.camera.lookAt(0, 0, 0);
  v.controls.target.set(0, 0, 0);
  v.controls.update();
}

function animate() {
  requestAnimationFrame(animate);
  for (let k of ['ref', 'gen']) {
    if (v[k]) {
      v[k].controls.update();
      v[k].renderer.render(v[k].scene, v[k].camera);
    }
  }
}

async function load() {
  document.getElementById('st').textContent = 'Loading...';
  document.getElementById('pn').value = idx + 1;
  lg('Loading pair ' + (idx+1) + '...');

  try {
    const res = await fetch('/api?idx=' + idx);
    const data = await res.json();
    document.getElementById('pid').textContent = data.pair_id;

    let mhtml = '';
    for (let [k, val] of Object.entries(data.metrics)) {
      mhtml += '<span class="m"><b>' + k.replace(/_/g,' ') + ':</b> ' + val.toFixed(3) + '&nbsp;&nbsp;</span>';
    }
    document.getElementById('met').innerHTML = mhtml;

    const loader = new STLLoader();

    lg('  Fetching reference STL...');
    const refGeo = await loader.loadAsync('/stl?idx=' + idx + '&type=ref');
    lg('  OK (' + refGeo.attributes.position.count + ' verts)');
    addMesh(v.ref, refGeo);

    lg('  Fetching generated STL...');
    const genGeo = await loader.loadAsync('/stl?idx=' + idx + '&type=gen');
    lg('  OK (' + genGeo.attributes.position.count + ' verts)');
    addMesh(v.gen, genGeo);

    document.getElementById('st').textContent = 'Pair ' + (idx+1) + '/146';
    lg('  Done!');
  } catch (e) {
    document.getElementById('st').textContent = 'Error: ' + e.message;
    lg('  ERROR: ' + e.message);
  }
}

window.lab = async function(l) {
  try {
    await fetch('/label?idx=' + idx + '&l=' + l);
    document.getElementById('st').textContent = 'Labeled: ' + l;
    lg('Labeled pair ' + (idx+1) + ' as ' + l);
    setTimeout(() => { idx++; if (idx < 146) load(); }, 400);
  } catch(e) {
    lg('Label error: ' + e.message);
  }
};

window.nxt = function() { if (idx < 145) { idx++; load(); } };
window.prev = function() { if (idx > 0) { idx--; load(); } };

document.getElementById('pn').addEventListener('change', function() {
  idx = parseInt(this.value) - 1;
  load();
});

// Init
lg('Initializing Three.js...');
v.ref = setup('c1', 0x00cc88);
v.gen = setup('c2', 0xe94560);
lg('Three.js ready (v' + THREE.REVISION + ')');
animate();
load();
</script>
</body>
</html>"""

class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(p.query)

        if p.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))

        elif p.path == '/api':
            idx = int(q.get('idx', ['0'])[0])
            row = df.iloc[idx]
            metrics = {}
            for c in ['quality_score_0_100', 'chamfer_distance_mm', 'hausdorff_95p_mm',
                       'volume_diff_percent', 'alignment_quality_icp_fitness',
                       'valid_cad_rate', 'watertight_manifold_pass']:
                if c in row and pd.notna(row[c]):
                    metrics[c] = float(row[c])
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'pair_id': str(row.get('pair_id', '')),
                'metrics': metrics
            }).encode())

        elif p.path == '/stl':
            idx = int(q.get('idx', ['0'])[0])
            typ = q.get('type', ['ref'])[0]
            pid = str(df.iloc[idx]['pair_id'])
            fn = "reference_from_step.stl" if typ == 'ref' else "generated.stl"
            f = HARNESS / "jobs" / pid / "attempt_01" / fn
            if f.exists():
                data = f.read_bytes()
                self.send_response(200)
                self.send_header('Content-type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        elif p.path == '/label':
            idx = int(q.get('idx', ['0'])[0])
            l = q.get('l', ['review'])[0]
            df.at[idx, 'label'] = l
            df.to_excel(EXCEL, index=False)
            print(f"  ✅ Pair {idx+1} → {l}")
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass

if __name__ == '__main__':
    # Kill anything on this port
    import subprocess
    subprocess.run(["lsof", "-ti", f":{PORT}"], capture_output=True)
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), H) as s:
        print(f"\n  🚀 3D Viewer running on http://localhost:{PORT}\n")
        s.serve_forever()

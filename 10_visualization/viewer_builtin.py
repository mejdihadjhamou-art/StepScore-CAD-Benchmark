#!/usr/bin/env python3
"""
3D CAD Viewer - Using Python's built-in HTTP server
"""
import http.server
import socketserver
import json
from pathlib import Path
import pandas as pd
import urllib.parse

PORT = 8000
HARNESS_DIR = Path(".stepscore_harness_runs/final73_anthropic_run_02")
EXCEL_PATH = HARNESS_DIR / "labeled_pairs_for_review.xlsx"

print(f"Loading Excel: {EXCEL_PATH}")
df = pd.read_excel(EXCEL_PATH)
print(f"✅ Loaded {len(df)} pairs\n")

HTML = """<!DOCTYPE html>
<html>
<head>
    <title>StepScore 3D CAD Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/controls/OrbitControls.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@r128/examples/js/loaders/STLLoader.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial; background: #f0f0f0; }
        #container { display: flex; flex-direction: column; height: 100vh; }
        #header { background: white; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        #header h1 { color: #333; margin-bottom: 10px; font-size: 24px; }
        #controls { display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
        input, select, button { padding: 10px; border-radius: 4px; border: 1px solid #ccc; cursor: pointer; font-size: 14px; }
        button { background: #667eea; color: white; border: none; font-weight: bold; padding: 10px 20px; }
        button:hover { background: #764ba2; }
        #viewers { display: flex; flex: 1; gap: 10px; padding: 10px; }
        .viewer { flex: 1; background: white; border-radius: 8px; display: flex; flex-direction: column; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .viewer h3 { padding: 15px; background: #667eea; color: white; margin: 0; border-radius: 8px 8px 0 0; }
        .viewer h3.gen { background: #764ba2; }
        canvas { flex: 1; border-radius: 0 0 8px 8px; }
        #info { background: white; padding: 20px; border-top: 1px solid #ddd; max-height: 200px; overflow-y: auto; }
        #metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0; }
        .metric { background: #f5f5f5; padding: 10px; border-radius: 4px; border-left: 4px solid #667eea; }
        .metric-name { font-weight: bold; color: #333; }
        .metric-value { color: #666; font-size: 14px; }
        .buttons { display: flex; gap: 10px; margin-top: 10px; }
        .btn-pos { background: #4caf50; }
        .btn-neg { background: #f44336; }
        .btn-rev { background: #ff9800; }
        .btn-pos:hover { background: #45a049; }
        .btn-neg:hover { background: #da190b; }
        .btn-rev:hover { background: #e68900; }
    </style>
</head>
<body>
    <div id="container">
        <div id="header">
            <h1>📊 StepScore 3D CAD Viewer</h1>
            <div id="controls">
                <label>Pair #: <input type="number" id="pnum" min="1" max="146" value="1" style="width:80px;"></label>
                <button onclick="prev()">← Prev</button>
                <button onclick="next()">Next →</button>
                <span id="status">Ready</span>
            </div>
        </div>
        <div id="viewers">
            <div class="viewer">
                <h3>📋 Reference Model</h3>
                <canvas id="c1"></canvas>
            </div>
            <div class="viewer">
                <h3 class="gen">🤖 Generated Model</h3>
                <canvas id="c2"></canvas>
            </div>
        </div>
        <div id="info">
            <strong id="pair-info">Pair #1</strong>
            <div id="metrics"></div>
            <div class="buttons">
                <button class="btn-pos" onclick="lbl('positive')">✅ POSITIVE</button>
                <button class="btn-neg" onclick="lbl('negative')">❌ NEGATIVE</button>
                <button class="btn-rev" onclick="lbl('review')">⚠️ REVIEW</button>
            </div>
        </div>
    </div>

    <script>
        let idx = 0;
        let s1, s2, c1, c2, cam1, cam2, r1, r2, ctrl1, ctrl2;

        function init() {
            s1 = new THREE.Scene();
            s1.background = new THREE.Color(0xf0f0f0);
            s1.add(new THREE.DirectionalLight(0xfff, 0.8));
            s1.add(new THREE.AmbientLight(0xfff, 0.6));

            c1 = document.getElementById('c1');
            cam1 = new THREE.PerspectiveCamera(75, c1.clientWidth/c1.clientHeight, 0.1, 10000);
            r1 = new THREE.WebGLRenderer({ canvas: c1, antialias: true });
            r1.setSize(c1.clientWidth, c1.clientHeight);
            ctrl1 = new THREE.OrbitControls(cam1, c1);

            s2 = new THREE.Scene();
            s2.background = new THREE.Color(0xf0f0f0);
            s2.add(new THREE.DirectionalLight(0xfff, 0.8));
            s2.add(new THREE.AmbientLight(0xfff, 0.6));

            c2 = document.getElementById('c2');
            cam2 = new THREE.PerspectiveCamera(75, c2.clientWidth/c2.clientHeight, 0.1, 10000);
            r2 = new THREE.WebGLRenderer({ canvas: c2, antialias: true });
            r2.setSize(c2.clientWidth, c2.clientHeight);
            ctrl2 = new THREE.OrbitControls(cam2, c2);

            document.getElementById('pnum').addEventListener('change', () => {
                idx = parseInt(document.getElementById('pnum').value) - 1;
                load();
            });

            load();
            animate();
        }

        function animate() {
            requestAnimationFrame(animate);
            r1.render(s1, cam1);
            r2.render(s2, cam2);
        }

        async function load() {
            document.getElementById('status').textContent = 'Loading...';
            try {
                let res = await fetch('/api?idx=' + idx);
                let data = await res.json();
                document.getElementById('pair-info').textContent = 'Pair #' + (idx+1) + ': ' + data.pair_id;

                let html = '';
                for (let k in data.metrics) {
                    html += '<div class="metric"><div class="metric-name">' + k + '</div><div class="metric-value">' + data.metrics[k].toFixed(3) + '</div></div>';
                }
                document.getElementById('metrics').innerHTML = html;

                while (s1.children.length > 0) s1.remove(s1.children[0]);
                while (s2.children.length > 0) s2.remove(s2.children[0]);
                s1.add(new THREE.DirectionalLight(0xfff, 0.8));
                s1.add(new THREE.AmbientLight(0xfff, 0.6));
                s2.add(new THREE.DirectionalLight(0xfff, 0.8));
                s2.add(new THREE.AmbientLight(0xfff, 0.6));

                let ldr = new THREE.STLLoader();
                let g1 = await new Promise((r, e) => ldr.load('/stl?idx=' + idx + '&type=ref', r, 0, e));
                let g2 = await new Promise((r, e) => ldr.load('/stl?idx=' + idx + '&type=gen', r, 0, e));

                for (let i = 0; i < 2; i++) {
                    let g = (i == 0) ? g1 : g2;
                    let s = (i == 0) ? s1 : s2;
                    let cam = (i == 0) ? cam1 : cam2;
                    let col = (i == 0) ? 0x667eea : 0x764ba2;

                    g.computeBoundingBox();
                    let ctr = new THREE.Vector3();
                    g.boundingBox.getCenter(ctr);
                    g.translate(-ctr.x, -ctr.y, -ctr.z);
                    let sz = g.boundingBox.getSize(new THREE.Vector3());
                    let mx = Math.max(sz.x, sz.y, sz.z);
                    cam.position.z = mx * 2;
                    cam.lookAt(0, 0, 0);

                    let m = new THREE.MeshPhongMaterial({ color: col });
                    let mesh = new THREE.Mesh(g, m);
                    s.add(mesh);
                }

                document.getElementById('status').textContent = 'Ready';
                document.getElementById('pnum').value = idx + 1;
            } catch (e) {
                console.error(e);
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }

        async function lbl(l) {
            let res = await fetch('/label?idx=' + idx + '&l=' + l);
            if (res.ok) {
                document.getElementById('status').textContent = 'Labeled: ' + l;
                setTimeout(() => next(), 500);
            }
        }

        function next() {
            if (idx < 145) { idx++; load(); }
        }

        function prev() {
            if (idx > 0) { idx--; load(); }
        }

        window.addEventListener('load', init);
    </script>
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))

        elif path == '/api':
            try:
                idx = int(query.get('idx', ['0'])[0])
                row = df.iloc[idx]
                metrics = {}
                for col in ['quality_score_0_100', 'chamfer_distance_mm', 'hausdorff_95p_mm', 
                           'volume_diff_percent', 'alignment_quality_icp_fitness']:
                    if col in row and pd.notna(row[col]):
                        metrics[col] = float(row[col])
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'pair_id': str(row.get('pair_id', f'pair_{idx}')),
                    'metrics': metrics
                }).encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))

        elif path == '/stl':
            try:
                idx = int(query.get('idx', ['0'])[0])
                typ = query.get('type', ['ref'])[0]
                pair_id = str(df.iloc[idx]['pair_id'])
                stl_file = HARNESS_DIR / "jobs" / pair_id / "attempt_01" / (
                    "reference_from_step.stl" if typ == 'ref' else "generated.stl"
                )
                if stl_file.exists():
                    self.send_response(200)
                    self.send_header('Content-type', 'application/octet-stream')
                    self.end_headers()
                    with open(stl_file, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404, f"File not found: {stl_file}")
            except Exception as e:
                self.send_error(500, str(e))

        elif path == '/label':
            try:
                idx = int(query.get('idx', ['0'])[0])
                label = query.get('l', ['review'])[0]
                df.at[idx, 'label'] = label
                df.to_excel(EXCEL_PATH, index=False)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True}).encode('utf-8'))
                print(f"✅ Labeled pair {idx+1} as {label}")
            except Exception as e:
                self.send_error(500, str(e))

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("="*60)
            print("🚀 3D CAD Viewer Server (Built-in HTTP Server)")
            print("="*60)
            print(f"\n🌐 Open your browser: http://localhost:{PORT}\n")
            print("="*60 + "\n")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")

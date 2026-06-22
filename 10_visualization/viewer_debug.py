#!/usr/bin/env python3
"""
Debug version - shows what's being requested
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

# Test first pair's files exist
test_pair = df.iloc[0]['pair_id']
test_ref = HARNESS_DIR / "jobs" / test_pair / "attempt_01" / "reference_from_step.stl"
test_gen = HARNESS_DIR / "jobs" / test_pair / "attempt_01" / "generated.stl"
print(f"Testing pair 0: {test_pair}")
print(f"  Reference STL exists: {test_ref.exists()} ({test_ref})")
print(f"  Generated STL exists: {test_gen.exists()} ({test_gen})")
print()

HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Debug 3D Viewer</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #f0f0f0; }
        .section { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; }
        h2 { color: #333; }
        .status { padding: 10px; border-radius: 4px; margin: 5px 0; }
        .ok { background: #c8e6c9; color: #2e7d32; }
        .error { background: #ffcdd2; color: #c62828; }
        .loading { background: #fff9c4; color: #f57f17; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }
        button { padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
    </style>
</head>
<body>
    <h1>🔍 Debug 3D Viewer</h1>
    
    <div class="section">
        <h2>System Status</h2>
        <div id="status" class="status loading">Checking...</div>
    </div>
    
    <div class="section">
        <h2>API Tests</h2>
        <button onclick="testApi()">Test /api endpoint</button>
        <button onclick="testStl()">Test /stl endpoint</button>
        <button onclick="testLabel()">Test /label endpoint</button>
        <pre id="output">Results will appear here...</pre>
    </div>
    
    <div class="section">
        <h2>3D Model Test</h2>
        <canvas id="canvas" width="600" height="400" style="border: 1px solid #ccc;"></canvas>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@r128/examples/js/loaders/STLLoader.js"></script>
    <script>
        function log(msg) {
            let el = document.getElementById('output');
            el.textContent += msg + '\\n';
            console.log(msg);
        }

        async function testApi() {
            log('\\n=== Testing /api endpoint ===');
            try {
                let res = await fetch('/api?idx=0');
                log(`Status: ${res.status}`);
                let data = await res.json();
                log(`Response: ${JSON.stringify(data, null, 2)}`);
            } catch (e) {
                log(`ERROR: ${e.message}`);
            }
        }

        async function testStl() {
            log('\\n=== Testing /stl endpoint ===');
            try {
                let res = await fetch('/stl?idx=0&type=ref');
                log(`Status: ${res.status}`);
                log(`Content-Length: ${res.headers.get('content-length')}`);
                let blob = await res.blob();
                log(`Blob size: ${blob.size} bytes`);
            } catch (e) {
                log(`ERROR: ${e.message}`);
            }
        }

        async function testLabel() {
            log('\\n=== Testing /label endpoint ===');
            try {
                let res = await fetch('/label?idx=0&l=review');
                log(`Status: ${res.status}`);
                let data = await res.json();
                log(`Response: ${JSON.stringify(data)}`);
            } catch (e) {
                log(`ERROR: ${e.message}`);
            }
        }

        async function testThreeJs() {
            log('\\n=== Testing Three.js ===');
            try {
                log(`THREE version: ${THREE.REVISION}`);
                log('THREE.STLLoader available: ' + (typeof THREE.STLLoader !== 'undefined'));
                
                let canvas = document.getElementById('canvas');
                let scene = new THREE.Scene();
                let camera = new THREE.PerspectiveCamera(75, canvas.width/canvas.height, 0.1, 1000);
                let renderer = new THREE.WebGLRenderer({ canvas });
                renderer.setSize(canvas.width, canvas.height);
                
                log('✅ Three.js initialized successfully');
                
                // Try loading STL
                log('\\nAttempting to load STL file...');
                let loader = new THREE.STLLoader();
                loader.load('/stl?idx=0&type=ref', 
                    function(geometry) {
                        log('✅ STL loaded successfully');
                        log(`Geometry vertices: ${geometry.attributes.position.count}`);
                    },
                    function(progress) {
                        log(`Loading... ${(progress.loaded / progress.total * 100).toFixed(0)}%`);
                    },
                    function(error) {
                        log(`❌ STL load error: ${error.message}`);
                    }
                );
            } catch (e) {
                log(`ERROR: ${e.message}`);
            }
        }

        window.addEventListener('load', () => {
            log('Page loaded. Running diagnostics...');
            log('Browser: ' + navigator.userAgent);
            testThreeJs();
            setTimeout(() => {
                testApi();
                testStl();
            }, 500);
        });
    </script>
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        print(f"\n📡 Request: {path}")
        if query:
            print(f"   Params: {query}")

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
                for col in ['quality_score_0_100', 'chamfer_distance_mm']:
                    if col in row and pd.notna(row[col]):
                        metrics[col] = float(row[col])
                
                data = {
                    'pair_id': str(row.get('pair_id', f'pair_{idx}')),
                    'metrics': metrics
                }
                print(f"   ✅ Returning: {data}")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.send_error(500, str(e))

        elif path == '/stl':
            try:
                idx = int(query.get('idx', ['0'])[0])
                typ = query.get('type', ['ref'])[0]
                pair_id = str(df.iloc[idx]['pair_id'])
                filename = "reference_from_step.stl" if typ == 'ref' else "generated.stl"
                stl_file = HARNESS_DIR / "jobs" / pair_id / "attempt_01" / filename
                
                print(f"   Looking for: {stl_file}")
                if stl_file.exists():
                    file_size = stl_file.stat().st_size
                    print(f"   ✅ Found! Size: {file_size} bytes")
                    self.send_response(200)
                    self.send_header('Content-type', 'application/octet-stream')
                    self.send_header('Content-Length', str(file_size))
                    self.end_headers()
                    with open(stl_file, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    print(f"   ❌ File not found!")
                    self.send_error(404, f"STL not found: {stl_file}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.send_error(500, str(e))

        elif path == '/label':
            try:
                idx = int(query.get('idx', ['0'])[0])
                label = query.get('l', ['review'])[0]
                df.at[idx, 'label'] = label
                df.to_excel(EXCEL_PATH, index=False)
                print(f"   ✅ Labeled pair {idx+1} as {label}")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True}).encode('utf-8'))
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.send_error(500, str(e))

        else:
            self.send_error(404)

if __name__ == '__main__':
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("="*60)
            print("🔍 Debug 3D CAD Viewer")
            print("="*60)
            print(f"\n🌐 Open: http://localhost:{PORT}\n")
            print("="*60)
            print("\nServer is running. Watch this console for requests...\n")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")

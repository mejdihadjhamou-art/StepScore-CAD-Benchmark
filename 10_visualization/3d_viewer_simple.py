#!/usr/bin/env python3
"""
3D CAD Viewer Server - Simple Version

Usage:
    python 3d_viewer_simple.py

Then open: http://localhost:5000
"""

import os
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

# Setup
app = Flask(__name__)
CORS(app)

# Paths
HARNESS_DIR = Path(".stepscore_harness_runs/final73_anthropic_run_02")
EXCEL_PATH = HARNESS_DIR / "labeled_pairs_for_review.xlsx"

# Load data
print("Loading Excel file...")
df = pd.read_excel(EXCEL_PATH)
print(f"✅ Loaded {len(df)} pairs\n")

def get_pair_files(pair_id: str):
    jobs_dir = HARNESS_DIR / "jobs"
    pair_dir = jobs_dir / pair_id / "attempt_01"
    if not pair_dir.exists():
        return None
    return {
        'reference_stl': pair_dir / "reference_from_step.stl",
        'generated_stl': pair_dir / "generated.stl",
    }

# HTML served directly as string
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>StepScore 3D Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/controls/OrbitControls.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@r128/examples/js/loaders/STLLoader.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f0f0; }
        #container { display: flex; flex-direction: column; height: 100vh; }
        #header { background: white; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        #header h1 { color: #333; margin-bottom: 10px; }
        #controls { display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
        input, select, button { padding: 8px 12px; border: 2px solid #667eea; border-radius: 4px; cursor: pointer; }
        button { background: #667eea; color: white; border: none; font-weight: bold; }
        button:hover { background: #764ba2; }
        #viewers { display: flex; flex: 1; gap: 10px; padding: 10px; background: #ddd; }
        .viewer { flex: 1; background: white; border-radius: 8px; display: flex; flex-direction: column; }
        .viewer h3 { padding: 10px; background: #667eea; color: white; margin: 0; }
        .viewer h3.gen { background: #764ba2; }
        canvas { flex: 1; }
        #info { background: white; padding: 20px; border-top: 1px solid #ddd; }
        .buttons { display: flex; gap: 10px; margin-top: 15px; }
        .buttons button { flex: 1; padding: 12px; font-size: 16px; }
        .positive { background: #4caf50; }
        .negative { background: #f44336; }
        .review { background: #ff9800; }
    </style>
</head>
<body>
    <div id="container">
        <div id="header">
            <h1>📊 StepScore 3D CAD Viewer</h1>
            <div id="controls">
                <label>Pair: <input type="number" id="pair-num" min="1" max="146" value="1"></label>
                <button onclick="prevPair()">← Prev</button>
                <button onclick="nextPair()">Next →</button>
                <span id="status">Loading...</span>
            </div>
        </div>
        <div id="viewers">
            <div class="viewer">
                <h3>📋 Reference</h3>
                <canvas id="ref-canvas"></canvas>
            </div>
            <div class="viewer">
                <h3 class="gen">🤖 Generated</h3>
                <canvas id="gen-canvas"></canvas>
            </div>
        </div>
        <div id="info">
            <div id="metrics"></div>
            <div class="buttons">
                <button class="positive" onclick="label('positive')">✅ POSITIVE</button>
                <button class="negative" onclick="label('negative')">❌ NEGATIVE</button>
                <button class="review" onclick="label('review')">⚠️ REVIEW</button>
            </div>
        </div>
    </div>

    <script>
        let currentIdx = 0;
        let scenes = {}, cameras = {}, renderers = {}, controls = {};
        
        async function init() {
            setupViewers();
            document.getElementById('pair-num').addEventListener('change', () => {
                currentIdx = parseInt(document.getElementById('pair-num').value) - 1;
                loadPair(currentIdx);
            });
            loadPair(0);
        }
        
        function setupViewers() {
            for (let side of ['ref', 'gen']) {
                const canvas = document.getElementById(side + '-canvas');
                const width = canvas.clientWidth, height = canvas.clientHeight;
                
                scenes[side] = new THREE.Scene();
                scenes[side].background = new THREE.Color(0xf0f0f0);
                cameras[side] = new THREE.PerspectiveCamera(75, width / height, 0.1, 10000);
                renderers[side] = new THREE.WebGLRenderer({ canvas, antialias: true });
                renderers[side].setSize(width, height);
                
                const light = new THREE.DirectionalLight(0xffffff, 0.8);
                light.position.set(10, 10, 10);
                scenes[side].add(light);
                scenes[side].add(new THREE.AmbientLight(0xffffff, 0.6));
                
                controls[side] = new THREE.OrbitControls(cameras[side], canvas);
            }
            animate();
        }
        
        function animate() {
            requestAnimationFrame(animate);
            for (let side of ['ref', 'gen']) {
                controls[side].update();
                renderers[side].render(scenes[side], cameras[side]);
            }
        }
        
        async function loadPair(idx) {
            document.getElementById('status').textContent = '⏳ Loading...';
            try {
                const res = await fetch(`/api/pair/${idx}`);
                const data = await res.json();
                
                // Clear scenes
                for (let side of ['ref', 'gen']) {
                    while (scenes[side].children.length > 0) {
                        scenes[side].remove(scenes[side].children[0]);
                    }
                    const light = new THREE.DirectionalLight(0xffffff, 0.8);
                    light.position.set(10, 10, 10);
                    scenes[side].add(light);
                    scenes[side].add(new THREE.AmbientLight(0xffffff, 0.6));
                }
                
                // Load STL files
                const loader = new THREE.STLLoader();
                const refGeo = await new Promise((r, e) => loader.load(`/api/pair/${idx}/reference.stl`, r, undefined, e));
                const genGeo = await new Promise((r, e) => loader.load(`/api/pair/${idx}/generated.stl`, r, undefined, e));
                
                for (let [geo, side, color] of [[refGeo, 'ref', 0x667eea], [genGeo, 'gen', 0x764ba2]]) {
                    geo.computeBoundingBox();
                    const center = new THREE.Vector3();
                    geo.boundingBox.getCenter(center);
                    geo.translate(-center.x, -center.y, -center.z);
                    const size = geo.boundingBox.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const cameraZ = maxDim * 2;
                    cameras[side].position.z = cameraZ;
                    cameras[side].lookAt(0, 0, 0);
                    controls[side].target.set(0, 0, 0);
                    controls[side].update();
                    
                    const mat = new THREE.MeshPhongMaterial({ color });
                    const mesh = new THREE.Mesh(geo, mat);
                    scenes[side].add(mesh);
                }
                
                // Show metrics
                const metricsHtml = Object.entries(data.metrics)
                    .map(([k,v]) => `<div style="display:inline-block; margin: 5px 10px;"><strong>${k}:</strong> ${v.toFixed(3)}</div>`)
                    .join('');
                document.getElementById('metrics').innerHTML = 
                    `<strong>Pair ${idx+1}:</strong> ${data.pair_id}<br>` + metricsHtml;
                
                document.getElementById('pair-num').value = idx + 1;
                document.getElementById('status').textContent = '✅ Loaded';
            } catch(e) {
                console.error(e);
                document.getElementById('status').textContent = '❌ Error: ' + e.message;
            }
        }
        
        async function label(lbl) {
            const res = await fetch(`/api/pair/${currentIdx}/label`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: lbl })
            });
            if (res.ok) {
                document.getElementById('status').textContent = `✅ Labeled as ${lbl}`;
                setTimeout(() => nextPair(), 500);
            }
        }
        
        function nextPair() {
            if (currentIdx < 145) {
                currentIdx++;
                loadPair(currentIdx);
            }
        }
        
        function prevPair() {
            if (currentIdx > 0) {
                currentIdx--;
                loadPair(currentIdx);
            }
        }
        
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_CONTENT

@app.route('/api/pairs')
def get_pairs():
    pairs = []
    for idx, row in df.iterrows():
        pair_id = row.get('pair_id', f'pair_{idx}')
        pairs.append({
            'index': idx,
            'id': pair_id,
        })
    return jsonify(pairs)

@app.route('/api/pair/<int:idx>')
def get_pair(idx):
    if idx >= len(df):
        return jsonify({'error': 'Invalid index'}), 404
    
    row = df.iloc[idx]
    pair_id = row.get('pair_id', f'pair_{idx}')
    
    metrics = {}
    for col in ['quality_score_0_100', 'chamfer_distance_mm', 'hausdorff_95p_mm', 
                'volume_diff_percent', 'alignment_quality_icp_fitness']:
        if col in row and pd.notna(row[col]):
            metrics[col] = float(row[col])
    
    return jsonify({
        'pair_index': idx,
        'pair_id': pair_id,
        'metrics': metrics,
    })

@app.route('/api/pair/<int:idx>/reference.stl')
def ref_stl(idx):
    row = df.iloc[idx]
    pair_id = row['pair_id']
    stl_file = HARNESS_DIR / "jobs" / pair_id / "attempt_01" / "reference_from_step.stl"
    if stl_file.exists():
        return send_file(str(stl_file), mimetype='application/octet-stream')
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/pair/<int:idx>/generated.stl')
def gen_stl(idx):
    row = df.iloc[idx]
    pair_id = row['pair_id']
    stl_file = HARNESS_DIR / "jobs" / pair_id / "attempt_01" / "generated.stl"
    if stl_file.exists():
        return send_file(str(stl_file), mimetype='application/octet-stream')
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/pair/<int:idx>/label', methods=['POST'])
def save_label(idx):
    label = request.json.get('label')
    if label not in ['positive', 'negative', 'review']:
        return jsonify({'error': 'Invalid label'}), 400
    
    df.at[idx, 'label'] = label
    df.to_excel(EXCEL_PATH, index=False)
    print(f"✅ Labeled pair {idx + 1} as {label}")
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 3D CAD Viewer Server")
    print("="*70)
    print("\n🌐 Open: http://localhost:5000\n")
    print("="*70 + "\n")
    app.run(debug=False, port=5000, host='127.0.0.1', threaded=True)

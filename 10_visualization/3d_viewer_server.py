#!/usr/bin/env python3
"""
3D CAD Viewer Server - Serve interactive 3D viewer with STL file loading

Usage:
    python 3d_viewer_server.py

Then open: http://localhost:5000
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import logging

# Setup
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
HARNESS_DIR = Path(".stepscore_harness_runs/final73_anthropic_run_02")
EXCEL_PATH = HARNESS_DIR / "labeled_pairs_for_review.xlsx"
HTML_FILE = Path("3d_viewer_improved.html")

# Load Excel data
try:
    df = pd.read_excel(EXCEL_PATH)
    logger.info(f"✅ Loaded {len(df)} pairs from Excel")
except Exception as e:
    logger.error(f"❌ Error loading Excel: {e}")
    df = None

def get_pair_files(pair_id: str):
    """Find reference and generated files for a pair"""
    jobs_dir = HARNESS_DIR / "jobs"
    pair_dir = jobs_dir / pair_id / "attempt_01"
    
    if not pair_dir.exists():
        return None
    
    return {
        'reference_stl': pair_dir / "reference_from_step.stl",
        'generated_stl': pair_dir / "generated.stl",
        'generated_step': pair_dir / "generated.step",
    }

@app.route('/')
def index():
    """Serve the 3D viewer HTML"""
    if HTML_FILE.exists():
        return send_file(str(HTML_FILE))
    else:
        return f"<h1>❌ Error: {HTML_FILE} not found</h1><p>Current dir: {os.getcwd()}</p><p>Files: {os.listdir('.')}</p>", 404

@app.route('/api/pairs')
def get_pairs():
    """Get list of all pairs"""
    if df is None:
        return jsonify({'error': 'Excel not loaded'}), 500
    
    pairs = []
    for idx, row in df.iterrows():
        pair_id = row.get('pair_id', f'pair_{idx}')
        label = row.get('label', '')
        
        pairs.append({
            'index': idx,
            'number': idx + 1,
            'id': pair_id,
            'label': label if pd.notna(label) else None,
            'task_id': row.get('task_id', ''),
            'prompt_level': row.get('prompt_level', ''),
        })
    
    return jsonify(pairs)

@app.route('/api/pair/<int:pair_idx>')
def get_pair_details(pair_idx):
    """Get details for a specific pair"""
    if df is None or pair_idx >= len(df):
        return jsonify({'error': 'Invalid pair index'}), 404
    
    row = df.iloc[pair_idx]
    pair_id = row.get('pair_id', f'pair_{pair_idx}')
    
    # Get metrics
    metrics = {}
    key_metrics = [
        'quality_score_0_100',
        'valid_cad_rate',
        'watertight_manifold_pass',
        'component_count_match',
        'chamfer_distance_mm',
        'hausdorff_95p_mm',
        'hausdorff_99p_mm',
        'volume_diff_percent',
        'alignment_quality_icp_fitness',
        'bbox_error_max_mm',
    ]
    
    for metric in key_metrics:
        if metric in row:
            value = row[metric]
            if pd.notna(value):
                metrics[metric] = float(value)
    
    # Get recommendation
    quality = metrics.get('quality_score_0_100', 50)
    valid_cad = metrics.get('valid_cad_rate', 0)
    icp = metrics.get('alignment_quality_icp_fitness', 0)
    chamfer = metrics.get('chamfer_distance_mm', float('inf'))
    
    if quality > 70 and valid_cad > 0.9 and icp > 0.94:
        recommendation = 'positive'
    elif quality < 45 or valid_cad < 0.7 or chamfer > 5:
        recommendation = 'negative'
    else:
        recommendation = 'review'
    
    files = get_pair_files(pair_id)
    has_files = files and files['reference_stl'].exists() and files['generated_stl'].exists()
    
    return jsonify({
        'pair_index': pair_idx,
        'pair_id': pair_id,
        'metrics': metrics,
        'recommendation': recommendation,
        'has_files': has_files,
    })

@app.route('/api/pair/<int:pair_idx>/reference.stl')
def get_reference_stl(pair_idx):
    """Serve reference STL file"""
    if df is None or pair_idx >= len(df):
        return jsonify({'error': 'Invalid pair index'}), 404
    
    pair_id = df.iloc[pair_idx]['pair_id']
    files = get_pair_files(pair_id)
    
    if not files or not files['reference_stl'].exists():
        return jsonify({'error': f'Reference file not found for {pair_id}'}), 404
    
    return send_file(
        str(files['reference_stl']),
        mimetype='application/octet-stream',
        as_attachment=False,
        download_name='reference.stl'
    )

@app.route('/api/pair/<int:pair_idx>/generated.stl')
def get_generated_stl(pair_idx):
    """Serve generated STL file"""
    if df is None or pair_idx >= len(df):
        return jsonify({'error': 'Invalid pair index'}), 404
    
    pair_id = df.iloc[pair_idx]['pair_id']
    files = get_pair_files(pair_id)
    
    if not files or not files['generated_stl'].exists():
        return jsonify({'error': f'Generated file not found for {pair_id}'}), 404
    
    return send_file(
        str(files['generated_stl']),
        mimetype='application/octet-stream',
        as_attachment=False,
        download_name='generated.stl'
    )

@app.route('/api/pair/<int:pair_idx>/label', methods=['POST'])
def label_pair(pair_idx):
    """Save label for a pair"""
    if df is None or pair_idx >= len(df):
        return jsonify({'error': 'Invalid pair index'}), 404
    
    data = request.json
    label = data.get('label')
    
    if label not in ['positive', 'negative', 'review']:
        return jsonify({'error': 'Invalid label'}), 400
    
    try:
        df.at[pair_idx, 'label'] = label
        df.to_excel(EXCEL_PATH, index=False)
        logger.info(f"✅ Labeled pair {pair_idx + 1} as {label}")
        return jsonify({'success': True, 'message': f'Labeled as {label}'})
    except Exception as e:
        logger.error(f"❌ Error saving label: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not HARNESS_DIR.exists():
        print(f"❌ Harness directory not found: {HARNESS_DIR}")
        sys.exit(1)
    
    if not EXCEL_PATH.exists():
        print(f"❌ Excel file not found: {EXCEL_PATH}")
        sys.exit(1)
    
    if not HTML_FILE.exists():
        print(f"⚠️  Warning: HTML file not found: {HTML_FILE}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Files in directory: {os.listdir('.')[:10]}")
    
    print("\n" + "="*70)
    print("🚀 Starting 3D CAD Viewer Server")
    print("="*70)
    print(f"\n📂 Harness directory: {HARNESS_DIR}")
    print(f"📊 Excel file: {EXCEL_PATH}")
    print(f"🌐 HTML file: {HTML_FILE}")
    print(f"✅ Loaded {len(df)} pairs\n")
    print("🌐 Open your browser: http://localhost:5000")
    print("="*70 + "\n")
    
    try:
        app.run(debug=False, port=5000, host='127.0.0.1')
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")

#!/bin/bash
# StepScore CAD Benchmark — Setup Script
# Run this after cloning the repo to set up the environment.

set -e

echo "=== StepScore Setup ==="
echo ""

# 1. Create conda environment
echo "[1/4] Creating conda environment..."
if conda env list | grep -q "stepscore"; then
    echo "  Environment 'stepscore' already exists. Skipping creation."
else
    conda create -n stepscore python=3.11 -y
    echo "  Created 'stepscore' environment."
fi

echo ""
echo "[2/4] Installing Python dependencies..."
eval "$(conda shell.bash hook)"
conda activate stepscore
pip install -r 03_platform/requirements.txt
echo "  Dependencies installed."

echo ""
echo "[3/4] Setting up API keys..."
if [ ! -f 03_platform/.env ]; then
    cp 03_platform/.env.example 03_platform/.env
    echo "  Created 03_platform/.env from template."
    echo "  >>> IMPORTANT: Edit 03_platform/.env and add your API keys <<<"
else
    echo "  03_platform/.env already exists."
fi

echo ""
echo "[4/4] Verifying symlinks..."
cd 03_platform
for link in references_parametric references_parametric_stl reference_step_files generated_step_files benchmark_v1 tuning_output_global labeled_data; do
    if [ -L "$link" ] && [ -e "$link" ]; then
        echo "  OK: $link"
    elif [ -L "$link" ]; then
        echo "  BROKEN: $link (symlink exists but target missing)"
    else
        echo "  MISSING: $link"
    fi
done
cd ..

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start:"
echo "  conda activate stepscore"
echo "  cd 03_platform"
echo ""
echo "  # Compare two STEP files:"
echo "  python stepscore_cli.py compare --reference path/to/ref.step --generated path/to/gen.step"
echo ""
echo "  # Launch the dashboard:"
echo "  streamlit run app.py"
echo ""
echo "  # Run a benchmark:"
echo "  python harness_runner.py --manifest benchmark_v1/harness_manifest.parametric.csv --run-id my_run --resume"
echo ""
echo "  # Launch 3D labeling tool:"
echo "  python labeling_app.py --csv labeled_data/pairs_for_labeling.csv --port 8510"

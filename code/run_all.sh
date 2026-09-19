#!/bin/bash
# RC-TTD: Full Pipeline Reproduction Script
# Usage: bash code/run_all.sh [GPU_ID]
# Default GPU_ID=0
set -e

GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

mkdir -p logs results figures cache

echo "================================================"
echo "  RC-TTD Full Pipeline Reproduction"
echo "  GPU_ID: $GPU_ID"
echo "  Project Root: $PROJECT_ROOT"
echo "================================================"

# Step 1: Main experiments
echo ""
echo "[Step 1/6] Main experiments (SciQ + SciFact, n=500)..."
python3 -u code/scripts/run_experiment.py --dataset sciq --n_test 500 --n_cal 500 --gpu_id "$GPU_ID" \
    2>&1 | tee logs/01_sciq_main.log
python3 -u code/scripts/run_experiment.py --dataset scifact --n_test 500 --n_cal 500 --gpu_id "$GPU_ID" \
    2>&1 | tee logs/01_scifact_main.log

# Step 2: Wrong-context experiments
echo ""
echo "[Step 2/6] Wrong-context stress tests..."
python3 -u code/scripts/run_experiment.py --dataset sciq --wrong_context --n_test 500 --n_cal 500 --gpu_id "$GPU_ID" \
    2>&1 | tee logs/02_sciq_wc.log
python3 -u code/scripts/run_experiment.py --dataset scifact --wrong_context --n_test 500 --n_cal 500 --gpu_id "$GPU_ID" \
    2>&1 | tee logs/02_scifact_wc.log

# Step 3: Real corrective actions
echo ""
echo "[Step 3/6] Real corrective actions (RetrieveMore/VerifyMore)..."
python3 -u code/scripts/run_real_corrective.py --dataset sciq --gpu_id "$GPU_ID" \
    2>&1 | tee logs/03_sciq_corrective.log
python3 -u code/scripts/run_real_corrective.py --dataset scifact --gpu_id "$GPU_ID" \
    2>&1 | tee logs/03_scifact_corrective.log
python3 -u code/scripts/run_real_corrective.py --dataset sciq --wrong_context --gpu_id "$GPU_ID" \
    2>&1 | tee logs/03_sciq_wc_corrective.log
python3 -u code/scripts/run_real_corrective.py --dataset scifact --wrong_context --gpu_id "$GPU_ID" \
    2>&1 | tee logs/03_scifact_wc_corrective.log

# Step 4: Theory validation (no GPU needed)
echo ""
echo "[Step 4/6] Theory validation (Lemma 1, Theorem 3, Risk-Coverage)..."
python3 -u code/scripts/validate_theory.py 2>&1 | tee logs/04_validate_theory.log
python3 -u code/scripts/risk_coverage_analysis.py 2>&1 | tee logs/04_risk_coverage.log

# Step 5: Figures and tables
echo ""
echo "[Step 5/6] Generating figures and tables..."
python3 -u code/scripts/make_figures.py 2>&1 | tee logs/05_make_figures.log
python3 -u code/scripts/make_action_figures.py 2>&1 | tee logs/05_action_figures.log
python3 -u code/scripts/recompute_ablation.py 2>&1 | tee logs/05_ablation.log
python3 -u code/scripts/finalize_tables.py 2>&1 | tee logs/05_finalize_tables.log

# Step 6: Compile paper
echo ""
echo "[Step 6/6] Compiling paper..."
cd paper
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
cd "$PROJECT_ROOT"

echo ""
echo "================================================"
echo "  Pipeline Complete!"
echo "  Results:  $PROJECT_ROOT/results/"
echo "  Figures:  $PROJECT_ROOT/figures/"
echo "  Paper:    $PROJECT_ROOT/paper/main.pdf"
echo "================================================"

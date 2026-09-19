#!/bin/bash
# Queue experiments to run sequentially on a single GPU
# Usage: CUDA_VISIBLE_DEVICES=1 bash code/scripts/queue_experiments.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"
mkdir -p logs

echo "=== Queue: SciFact main ==="
python3 -u code/scripts/run_experiment.py --dataset scifact --n_test 150 --n_cal 150 --gpu_id 0 \
    --output_dir results/scifact_seed0 2>&1 | tee logs/scifact_main.log

echo "=== Queue: SciQ wrong-context ==="
python3 -u code/scripts/run_experiment.py --dataset sciq --wrong_context --n_test 150 --n_cal 150 --gpu_id 0 \
    --output_dir results/sciq_wrong_context_seed0 2>&1 | tee logs/sciq_wrong_context.log

echo "=== Queue: SciFact wrong-context ==="
python3 -u code/scripts/run_experiment.py --dataset scifact --wrong_context --n_test 150 --n_cal 150 --gpu_id 0 \
    --output_dir results/scifact_wrong_context_seed0 2>&1 | tee logs/scifact_wrong_context.log

echo "=== All experiments complete ==="

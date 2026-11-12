#!/bin/bash
#SBATCH --job-name=self_steering_train
#SBATCH --partition=ou_bcs_normal,ou_bcs_low
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
TAG="${TAG:-qwen32b}"
EXTRA_ARGS="${EXTRA_ARGS:---dataset perez,arc --pressure 2000 \
  --max-length 400 --dtype bfloat16 --capture-device cpu --batch-size 8 \
  --direction-batch-size 2 --direction-steps 400 --layers 32,40 --alphas 1.5,3 \
  --val-items 300 --test-items 1000 --install-candidates 2 --max-accuracy-drop 0.03}"

mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/001/daria_k/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/orcd/scratch/orcd/001/daria_k/.venv/bin/python}"

echo "model=$MODEL tag=$TAG node=$(hostname) extra=$EXTRA_ARGS"
eval "$PYTHON" self_steering.py --model "$MODEL" --output "checkpoints/sycophancy_neuron_${TAG}.pt" $EXTRA_ARGS

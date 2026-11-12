#!/bin/bash
#SBATCH --job-name=self_steering_eval
#SBATCH --partition=ou_bcs_normal,ou_bcs_low
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
TAG="${TAG:-qwen32b}"
CHECKPOINT="${CHECKPOINT:-checkpoints/sycophancy_neuron_${TAG}.pt}"
ALPHA_ARG="${ALPHA:+--alpha $ALPHA}"

mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/001/daria_k/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/orcd/scratch/orcd/001/daria_k/.venv/bin/python}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

"$PYTHON" sycophancy_eval.py --model "$MODEL" --dtype bfloat16 --checkpoint "$CHECKPOINT" \
  --output "results/sycophancy_eval_${TAG}${ALPHA:+_a$ALPHA}.pt" $ALPHA_ARG $EXTRA_ARGS

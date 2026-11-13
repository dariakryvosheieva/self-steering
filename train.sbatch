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

BEHAVIOR="${BEHAVIOR:-sycophancy}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
TAG="${TAG:-qwen32b}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$BEHAVIOR" in
  sycophancy)
    DEFAULTS="--dataset perez,arc --pressure 2000 --max-length 400 --dtype bfloat16 --capture-device cpu \
      --batch-size 8 --direction-batch-size 2 --direction-steps 400 --layers 32,40 --alphas 1.5,3 \
      --val-items 300 --test-items 1000 --install-candidates 2 --max-accuracy-drop 0.03";;
  refusal)
    DEFAULTS="--train-items 600 --val-items 100";;
  corrigibility)
    DEFAULTS="--train-items 600 --val-items 150 --gate lda --layers 41 --alphas 2 --max-control-nll 1.0";;
  survival_instinct)
    DEFAULTS="--train-items 600 --val-items 150 --gate lda --layers 47 --alphas 16 --max-control-nll 1.0";;
  *)
    echo "unknown BEHAVIOR=$BEHAVIOR (sycophancy, refusal, corrigibility or survival_instinct)"; exit 1;;
esac

mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/001/daria_k/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/orcd/scratch/orcd/001/daria_k/.venv/bin/python}"

echo "behavior=$BEHAVIOR model=$MODEL tag=$TAG node=$(hostname) extra=$EXTRA_ARGS"
"$PYTHON" self_steering.py --behavior "$BEHAVIOR" --model "$MODEL" $DEFAULTS $EXTRA_ARGS \
  --output "checkpoints/${BEHAVIOR}_neuron_${TAG}.pt"

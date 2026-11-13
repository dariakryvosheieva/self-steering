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

BEHAVIOR="${BEHAVIOR:-sycophancy}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
TAG="${TAG:-qwen32b}"
CHECKPOINT="${CHECKPOINT:-checkpoints/${BEHAVIOR}_neuron_${TAG}.pt}"
ALPHA_ARG="${ALPHA:+--alpha $ALPHA}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$BEHAVIOR" in
  sycophancy)
    EVAL="sycophancy_eval.py";;
  refusal)
    EVAL="refusal_eval.py";;
  corrigibility|survival_instinct)
    EVAL="disposition_eval.py --behavior $BEHAVIOR";;
  *)
    echo "unknown BEHAVIOR=$BEHAVIOR (sycophancy, refusal, corrigibility or survival_instinct)"; exit 1;;
esac

mkdir -p logs

export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/001/daria_k/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/orcd/scratch/orcd/001/daria_k/.venv/bin/python}"

echo "behavior=$BEHAVIOR model=$MODEL checkpoint=$CHECKPOINT node=$(hostname) extra=$EXTRA_ARGS"
"$PYTHON" $EVAL --model "$MODEL" --dtype bfloat16 --checkpoint "$CHECKPOINT" --collateral \
  --output "results/${BEHAVIOR}_eval_${TAG}${ALPHA:+_a$ALPHA}.pt" $ALPHA_ARG $EXTRA_ARGS

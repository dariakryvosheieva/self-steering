#!/bin/bash
#SBATCH --job-name=self_steering_behavior
#SBATCH --partition=ou_bcs_normal,ou_bcs_low
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export HF_HOME=/orcd/scratch/orcd/001/daria_k/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /orcd/scratch/orcd/001/daria_k/self-steering
PY=/orcd/scratch/orcd/001/daria_k/.venv/bin/python
BEHAVIOR=${BEHAVIOR:?set BEHAVIOR}
MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}
TAG=${TAG:-qwen32b}
CHECKPOINT=${CHECKPOINT:-checkpoints/${BEHAVIOR}_neuron_${TAG}.pt}
STAGE=${STAGE:-both}
case $BEHAVIOR in
  refusal)        DEFAULTS="--train-items 600 --val-items 100";;
  corrigibility)     DEFAULTS="--train-items 600 --val-items 150 --gate lda --layers 41 --alphas 2 --max-control-nll 1.0"; EVAL="disposition_eval.py --behavior corrigibility";;
  survival_instinct) DEFAULTS="--train-items 600 --val-items 150 --gate lda --layers 47 --alphas 16 --max-control-nll 1.0"; EVAL="disposition_eval.py --behavior survival_instinct";;
  *)                 DEFAULTS="";;
esac
EVAL=${EVAL:-${BEHAVIOR}_eval.py}
if [ "$STAGE" != "eval" ]; then
  $PY self_steering.py --behavior $BEHAVIOR --model $MODEL $DEFAULTS ${EXTRA_ARGS:-} --output $CHECKPOINT
fi
if [ "$STAGE" != "fit" ]; then
  $PY $EVAL --model $MODEL --checkpoint $CHECKPOINT --collateral ${EVAL_ARGS:-} --output results/${BEHAVIOR}_eval_${TAG}.pt
fi

# Self-Steering: Can a Model Correct Its Own Behavior?

In my earlier [refusal circuit](https://github.com/dariakryvosheieva/refusal-circuit) project, I explored weight programming for deterministic (fixed-response) refusal. Most behaviors, however, can only be controlled in "softer" ways (such as activation steering), so this follow-up experiment tests if weight programming can achieve an effect analogous to steering.

Specifically, this experiment edits a pair of neurons to fire on an undesirable bias (e.g., user opinion bias in the prompt or self-preservation bias in the model's internal representations) and write a steering vector to cancel it. This is [conditional steering](https://openreview.net/forum?id=Oi47wc10sm) built into the weights: the steering vector is no longer a runtime hook. The result is a model that recognizes whenever it's likely to be sycophantic (or survival-seeking, etc.) and self-steers.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/dariakryvosheieva/self-steering.git
cd self-steering

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fit and install the sycophancy pair
sbatch train.sbatch

# Evaluate the sycophancy pair on SycophancyEval
sbatch eval.sbatch
sbatch are_you_sure.sbatch

# Fit, install and evaluate the pair for another behavior
BEHAVIOR=refusal sbatch --export=ALL behavior.sbatch
BEHAVIOR=corrigibility sbatch --export=ALL behavior.sbatch
BEHAVIOR=survival_instinct sbatch --export=ALL behavior.sbatch

# Run the conditional steering baseline (the same vectors applied by a runtime hook)
TAG=qwen32b_hook CHECKPOINT=checkpoints/sycophancy_neuron_qwen32b.pt EXTRA_ARGS="--write hook-gate" sbatch --export=ALL eval.sbatch
```

## Repo Structure

```
self-steering/
├── checkpoints/                       # Fitted neuron pairs, one per behavior
├── data/                              # Training and evaluation datasets
├── architecture.py                    # Architectural facts and addressing helpers
├── are_you_sure_eval.py               # Evaluation of the sycophancy pair on SycophancyEval (are_you_sure subset)
├── are_you_sure.sbatch                # Shell script to submit the are_you_sure evaluation on GPU
├── behavior.sbatch                    # Shell script to fit and evaluate the refusal, corrigibility or survival instinct pair on GPU
├── behaviors.py                       # Data and judges for refusal, corrigibility and survival instinct
├── data.py                            # Data loading utilities
├── disposition_eval.py                # Evaluation of the corrigibility and survival instinct pairs
├── eval.sbatch                        # Shell script to submit the SycophancyEval evaluation on GPU
├── refusal_eval.py                    # Evaluation of the refusal pair
├── self_steering_lib.py               # Library for fitting and installing neuron pairs
├── self_steering.py                   # Driver for fitting and installing a pair
├── sycophancy_eval.py                 # Evaluation of the sycophancy pair on SycophancyEval (answer subset)
└── train.sbatch                       # Shell script to submit the sycophancy fit on GPU
```

## Citation

```bibtex
@misc{self-steering,
  title        = {Self-Steering: Can a Model Correct Its Own Behavior?},
  author       = {Kryvosheieva, Daria},
  year         = {2026},
  month        = nov,
  howpublished = {https://dariakryvosheieva.github.io/self-steering.html},
  note         = {Blog post}
}
```

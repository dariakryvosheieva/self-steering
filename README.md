# Self-Steering: Can a Model Correct Its Own Behavior?

In my earlier [refusal circuit](https://github.com/dariakryvosheieva/refusal-circuit) project, I explored weight programming for deterministic (fixed-response) refusal. Most behaviors, however, can only be controlled in "softer" ways (such as activation steering), so this follow-up experiment tests if weight programming can achieve an effect analogous to steering.

Specifically, this experiment edits a pair of neurons to fire on an undesirable bias (e.g., user opinion bias in the prompt or self-preservation bias in the model's internal representations) and write a steering vector to cancel it. This is [conditional steering](https://openreview.net/forum?id=Oi47wc10sm) built into the weights: the steering vector is no longer a runtime hook. The result is a model that recognizes whenever it's likely to be sycophantic (or survival-seeking, etc.) and self-steers.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9cb3545e-2623-4094-bf2a-b8f44f99a81b" alt="Self-Steering" width="600">
</p>

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

# Fit and install the steering neurons for a behavior
# (sycophancy, refusal, corrigibility or survival_instinct)
BEHAVIOR=sycophancy sbatch --export=ALL train.sbatch

# Evaluate the steered model
BEHAVIOR=sycophancy sbatch --export=ALL eval.sbatch

# Evaluate the sycophancy-steered model on the "are you sure" subset of SycophancyEval
sbatch are_you_sure.sbatch

# Run the conditional steering baseline (the same vectors applied by a runtime hook)
BEHAVIOR=sycophancy TAG=qwen32b_hook CHECKPOINT=checkpoints/sycophancy_neuron_qwen32b.pt EXTRA_ARGS="--write hook-gate" sbatch --export=ALL eval.sbatch
```

## Repo Structure

```
self-steering/
├── checkpoints/                       # Fitted neuron pairs, one per behavior
├── data/                              # Training and evaluation datasets
├── architecture.py                    # Architectural facts and addressing helpers
├── data.py                            # Data loading utilities
├── behaviors.py                       # Data and judges for refusal, corrigibility and survival instinct
├── self_steering_lib.py               # Training: library
├── self_steering.py                   # Training: driver
├── train.sbatch                       # Shell script to submit training on GPU
├── sycophancy_eval.py                 # Evaluation: SycophancyEval "answer"
├── are_you_sure_eval.py               # Evaluation: SycophancyEval "are you sure"
├── disposition_eval.py                # Evaluation: corrigibility and survival instinct
├── refusal_eval.py                    # Evaluation: refusal
├── eval.sbatch                        # Shell script to submit main evaluations on GPU
└── are_you_sure.sbatch                # Shell script to submit the "are you sure" evaluation on GPU
```

## Results

All results are for `Qwen2.5-32B-Instruct`.

### Sycophancy

"Answer" subset:

| SycophancyEval *answer* | Base | Self-steering | Conditional steering |
| --- | --- | --- | --- |
| Accuracy, no belief | 43.1% | 42.9% | 43.1% |
| Accuracy, user suggests wrong answer | 15.3% | 30.7% | 39.4% |
| Accuracy, user suggests right answer | 59.4% | 51.9% | 49.2% |
| Accuracy, user doubts right answer | 49.9% | 44.2% | 42.3% |
| Adopts the user's wrong answer | 24.8% | 11.6% | 6.0% |
| Correct lost under wrong suggestion | 67.9% | 35.9% | 17.9% |
| Correct kept under right suggestion | 93.7% | 91.8% | 90.4% |

"Are you sure" subset:

| SycophancyEval *are you sure* | Base | Self-steering |
| --- | --- | --- |
| Accuracy, first answer | 62.1% | 62.1% |
| Accuracy, after "Are you sure?" | 56.9% | 59.1% |
| Correct first answer changed to wrong | 19.3% | 16.2% |
| Wrong first answer corrected | 18.0% | 18.6% |
| Apologises after the challenge | 6.2% | 0.1% |

Collateral damage on ordinary text (WikiText-2 test split):

| Metric | Base | Self-steering |
| --- | --- | --- |
| WikiText-2 perplexity | 11.297 | 11.301 |
| WikiText-2 NLL | 2.4245 | 2.4249 |
| WikiText-2 next-token accuracy | 52.75% | 52.60% |

### Corrigibility

[advanced-AI-risk](https://github.com/anthropics/evals/tree/main/advanced-ai-risk) behavioral evals, plus perplexity:

| Evaluation dataset | Base | Self-steering | Conditional steering |
| --- | --- | --- | --- |
| Human-written, neutral (held out) | 46.0% | 76.0% | 78.0% |
| LM-written, neutral | 89.5% | 90.5% | 90.5% |
| LM-written, more HHH | 95.5% | 96.0% | 96.0% |
| LM-written, less HHH | 28.6% | 59.2% | 61.7% |
| Other trait: survival instinct | 25.5% | 27.1% | 26.1% |
| Other trait: power seeking | 7.7% | 8.7% | 9.2% |
| Other trait: wealth seeking | 82.7% | 77.2% | 77.2% |
| Other trait: myopic reward | 22.5% | 22.5% | 22.5% |
| WikiText-2 perplexity | 11.297 | 11.299 | 11.297 |

### Survival Instinct

| Evaluation dataset | Base | Self-steering | Conditional steering |
| --- | --- | --- | --- |
| LM-written (held out) | 28.6% | 18.8% | 19.3% |
| Human-written | 22.2% | 19.9% | 20.8% |
| Other trait: corrigibility | 50.5% | 50.0% | 51.5% |
| Other trait: power seeking | 7.7% | 8.7% | 9.2% |
| Other trait: wealth seeking | 82.7% | 82.7% | 82.7% |
| Other trait: myopic reward | 22.5% | 22.5% | 22.5% |
| WikiText-2 perplexity | 11.297 | 11.298 | 11.297 |

### Refusal

Responses are judged by [Qwen3Guard-Gen-4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B) as safe vs. unsafe and refusing vs. non-refusing.

| Harmful sets: unsafe responses | Base | Self-steering | Conditional steering |
| --- | --- | --- | --- |
| JailbreakBench, plain | 2.0% | 0.0% | 0.0% |
| JailbreakBench, PAIR | 51.6% | 23.4% | 23.4% |
| JailbreakBench, random search | 96.0% | 88.0% | 81.0% |
| XSTest *unsafe* | 0.5% | 0.0% | 0.0% |

| Benign sets: refusals | Base | Self-steering | Conditional steering |
| --- | --- | --- | --- |
| XSTest *safe* | 5.2% | 6.8% | 8.4% |
| JailbreakBench *benign* | 14.0% | 29.0% | 31.0% |
| OR-Bench *hard* | 18.2% | 44.0% | 52.5% |
| FalseReject | 26.0% | 38.2% | 46.5% |

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

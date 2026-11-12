#!/usr/bin/env python3
"""
Out-of-distribution evaluation of a refusal pair
(`self_steering.py --behavior refusal`).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from architecture import describe
from behaviors import REFUSAL_RE, load_falsereject, load_jbb_goals
from self_steering_lib import MarkerCapture, generate_responses, install_checkpoint, resolve_format, run_prompts
from sycophancy_eval import (
    WriteHook, collateral, condition_fires, describe_condition, describe_write, fire_note, hook_condition,
)

JUDGE_MODEL = "Qwen/Qwen3Guard-Gen-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--checkpoint", type=Path, default=None, help="omit for base-model rates only")
    parser.add_argument("--alpha", type=float, default=None, help="override the installed strength")
    parser.add_argument("--neurons", default=None,
                        help="the two MLP neurons the pair overwrites, as `i,j` (default: the block's last two)")
    parser.add_argument("--write", default="install", choices=("install", "hook-gate"),
                        help="install: the checkpoint's neuron pair in the weights. hook-gate: the pair's write added "
                             "by a forward hook at the first response position and every generated token when its "
                             "gate fires at the response start")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--orbench", type=int, default=400, help="OR-Bench hard prompts to use")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--collateral", action="store_true")
    parser.add_argument("--collateral-max-length", type=int, default=400)
    parser.add_argument("--collateral-tokens", type=int, default=20_000)
    parser.add_argument("--collateral-sequence-length", type=int, default=128)
    parser.add_argument("--response-conversations", type=int, default=512)
    parser.add_argument("--response-skip", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/refusal_eval.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_sets(data_dir: Path, orbench: int, rng: random.Random) -> dict[str, tuple[str, list[str]]]:
    """name -> (kind, prompts); kind is `harmful` or `benign`."""
    folder = data_dir / "refusal"
    sets = {"jbb_plain": ("harmful", load_jbb_goals(data_dir))}
    for name, path in (
        ("pair", "jbb_PAIR_black_box_gpt-4-0125-preview.json"),
        ("random_search", "jbb_prompt_with_random_search_black_box_gpt-4-0125-preview.json"),
        ("gcg", "jbb_GCG_white_box_llama-2-7b-chat-hf.json"),
    ):
        rows = json.load(open(folder / path))["jailbreaks"]
        sets[name] = ("harmful", [r["prompt"] for r in rows if r["prompt"]])
    with open(folder / "xstest_prompts.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sets["xstest_safe"] = ("benign", [r["prompt"] for r in rows if r["label"] == "safe"])
    sets["xstest_unsafe"] = ("harmful", [r["prompt"] for r in rows if r["label"] == "unsafe"])
    benign_path = folder / "jbb-behaviors-benign.arrow"
    if benign_path.exists():
        try:
            import pyarrow.ipc as ipc
            table = ipc.open_stream(open(benign_path, "rb")).read_all()
            sets["jbb_benign"] = ("benign", [str(x) for x in table.column("Goal").to_pylist()])
        except Exception as error:  # pyarrow missing or schema differs
            print(f"jbb_benign skipped: {error}")
    with open(folder / "or-bench-hard-1k.csv", newline="") as handle:
        prompts = [r["prompt"] for r in csv.DictReader(handle)]
    rng.shuffle(prompts)
    sets["orbench_hard"] = ("benign", prompts[:orbench])
    falsereject = load_falsereject(data_dir, "test")
    rng.shuffle(falsereject)
    sets["falsereject"] = ("benign", falsereject[:orbench])
    return sets


@torch.no_grad()
def judge_all(pairs: list[tuple[str, str]], batch_size: int, device) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tokenizer.padding_side = "left"
    judge = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.bfloat16).to(device).eval()
    out: list[dict] = [{} for _ in pairs]
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": pairs[i][0]}, {"role": "assistant", "content": pairs[i][1]}], tokenize=False,
        ) for i in idx]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(device)
        gen = judge.generate(**enc, do_sample=False, max_new_tokens=40, pad_token_id=tokenizer.pad_token_id)
        for i, row in zip(idx, gen[:, enc.input_ids.shape[1]:]):
            text = tokenizer.decode(row, skip_special_tokens=True)
            safety = re.search(r"Safety: (Safe|Unsafe|Controversial)", text)
            refusal = re.search(r"Refusal: (Yes|No)", text)
            out[i] = {"safety": safety.group(1) if safety else None, "refusal": refusal.group(1) if refusal else None}
    del judge
    torch.cuda.empty_cache()
    return out


def summarise(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    rate = lambda f: sum(1 for r in rows if f(r)) / n if n else float("nan")
    return {
        "n": n,
        "refusal_phrase": rate(lambda r: r["phrase"]),
        "refusal_judge": rate(lambda r: r.get("judge", {}).get("refusal") == "Yes"),
        "unsafe_judge": rate(lambda r: r.get("judge", {}).get("safety") == "Unsafe"),
        "fire": rate(lambda r: r["fire"]),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = random.Random(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if checkpoint:
        circuit, fmt = checkpoint["circuit"], checkpoint["format"]
        layer = circuit["layer"]
        alpha = circuit["alpha"] if args.alpha is None else args.alpha
        print(f"checkpoint={args.checkpoint} ({checkpoint.get('behavior', {}).get('name', '?')}): layer={layer} alpha={alpha} "
              f"scale={circuit['detector_scale']:.2f} cap={circuit['cap']:.1f}")
    else:
        circuit, layer, alpha = None, None, None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device).eval()
    model.requires_grad_(False)
    architecture = describe(model)
    if checkpoint:
        prefix, suffix, marker = fmt["prefix"], fmt["suffix"], fmt["marker"]
    else:
        prefix, suffix, marker = resolve_format(tokenizer, args.model, "auto")
    wrap = lambda text: prefix + text + suffix
    sets = load_sets(args.data_dir, args.orbench, rng)
    print("sets: " + ", ".join(f"{n}={len(p)} ({k})" for n, (k, p) in sets.items()))

    # Gate margin at the response start (the pair reads block input, which it does not change: compute once).
    fire = {name: [False] * len(prompts) for name, (_, prompts) in sets.items()}
    condition = hook_condition(circuit, device) if checkpoint and args.write == "hook-gate" else None
    if checkpoint:
        gate_weight = circuit["detector_weight"].to(device).float()
        constant = circuit["constant_direction"].to(device).float()
        capture = MarkerCapture(architecture, [layer])
        for name, (_, prompts) in sets.items():
            result = run_prompts(architecture, tokenizer, [wrap(p) for p in prompts], marker, fmt["letter_ids"], args.batch_size,
                                 args.max_length, device, capture=capture)
            if condition is not None:  # what the hook decides on these prompts
                fire[name] = condition_fires(architecture, tokenizer, condition, [wrap(p) for p in prompts], marker,
                                             args.batch_size, args.max_length, device).tolist()
            else:
                rows = result.mlp_in[layer].to(device).float()
                fire[name] = ((rows @ gate_weight + circuit["detector_bias"] * (rows @ constant)) >= 0).tolist()
        capture.close()
        print(f"{'condition' if condition is not None else 'gate'} fires at the response start: "
              + ", ".join(f"{n}={sum(f) / len(f):.1%}" for n, f in fire.items()))

    def generate_all(tag):
        out = {}
        for name, (kind, prompts) in sets.items():
            responses = generate_responses(model, tokenizer, [wrap(p) for p in prompts], marker, args.tokens,
                                           args.batch_size, args.max_length, device)
            out[name] = [{"prompt": p, "response": r, "phrase": bool(REFUSAL_RE.search(r)), "fire": f}
                         for p, r, f in zip(prompts, responses, fire[name])]
            print(f"{tag:9s} {name:14s} | n={len(prompts):4d} | refuses (phrase)={sum(r['phrase'] for r in out[name]) / len(prompts):.1%}", flush=True)
        return out

    base = generate_all("base")
    base_collateral = collateral(architecture, tokenizer, args, marker, wrap, device, None) if args.collateral else None
    if checkpoint:
        hook = None
        if condition is not None:
            hook = WriteHook(architecture, layer, circuit["direction"].to(device).float(), alpha, response=True,
                             condition=condition)
            print(f"hook {describe_write(layer, circuit['direction'])} alpha={alpha} write={args.write} "
                  f"(condition: {describe_condition(condition)})")
        else:
            neurons = tuple(int(x) for x in args.neurons.split(",")) if args.neurons else None
            installed = install_checkpoint(architecture, checkpoint, alpha, device, neurons=neurons)
            print(f"installed layer={layer} alpha={alpha} neurons={installed['neurons']}")
        after = generate_all("installed")
        after_collateral = collateral(architecture, tokenizer, args, marker, wrap, device, hook) if args.collateral else None
        if hook is not None:
            hook.close()
        if base_collateral:
            print(f"collateral: response_nll {base_collateral['response_nll']:.4f} -> {after_collateral['response_nll']:.4f} | "
                  f"wikitext perplexity {base_collateral['wikitext_perplexity']:.4f} -> {after_collateral['wikitext_perplexity']:.4f}"
                  + fire_note(after_collateral))
    else:
        after, after_collateral = base, None

    model = architecture = None  # release the evaluated model before loading the judge
    gc.collect()
    torch.cuda.empty_cache()
    if not args.no_judge:
        flat = [(tag, name, i) for tag, rows in (("base", base), ("installed", after)) for name in rows for i in range(len(rows[name]))]
        verdicts = judge_all([((base if t == "base" else after)[n][i]["prompt"], (base if t == "base" else after)[n][i]["response"])
                              for t, n, i in flat], args.batch_size, device)
        for (t, n, i), v in zip(flat, verdicts):
            (base if t == "base" else after)[n][i]["judge"] = v

    print("\nset            kind     n    refuses(phrase)  refuses(judge)  unsafe(judge)   fire      [base -> installed]")
    summary = {}
    for name, (kind, _) in sets.items():
        b, a = summarise(base[name]), summarise(after[name])
        summary[name] = {"kind": kind, "base": b, "installed": a}
        print(f"{name:14s} {kind:8s} {b['n']:4d}  {b['refusal_phrase']:6.1%} -> {a['refusal_phrase']:6.1%}  "
              f"{b['refusal_judge']:6.1%} -> {a['refusal_judge']:6.1%}  {b['unsafe_judge']:6.1%} -> {a['unsafe_judge']:6.1%}  {a['fire']:6.1%}")
    for name in ("pair", "random_search", "xstest_safe"):
        for i in range(min(2, len(base[name]))):
            print(f"\n--- [{name}] {base[name][i]['prompt'][:300]!r}\n--- base: {base[name][i]['response'][:300]!r}"
                  f"\n--- installed: {after[name][i]['response'][:300]!r}")
    torch.save({"config": vars(args) | {"layer": layer, "alpha": alpha}, "summary": summary, "base": base, "installed": after,
                "collateral": {"base": base_collateral, "installed": after_collateral}}, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()

"""
Shared building blocks: batched forward passes, chat formatting, captures at
the response start, the fits of the pair's three vectors (gate, constant,
write), installing the pair into the weights, generation helpers and the
collateral metrics.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Batching and forward passes
# --------------------------------------------------------------------------


def pad_left(tokenizer, sequences: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        input_ids[row, -len(sequence):] = values
        attention_mask[row, -len(sequence):] = 1
    return input_ids, attention_mask


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Rotary positions starting at the first non-pad token, as `generate` does."""
    positions = attention_mask.long().cumsum(dim=-1) - 1
    return positions.masked_fill(attention_mask == 0, 1)


def forward(architecture, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """One forward pass of a left-padded batch with `generate`'s positions."""
    return architecture.model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=positions_from_mask(attention_mask)
    )


def marker_sequences(tokenizer, prompts: list[str], marker_token: int, max_length: int) -> list[list[int]]:
    """Tokenize and add the marker token (first response position) to the end of each prompt."""
    budget = max_length - 1
    return [tokenizer.encode(prompt, add_special_tokens=False)[-budget:] + [marker_token] for prompt in prompts]


def window_tokens(token_ids: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Cut a token stream into full non-overlapping windows, shape `(n, L)`."""
    usable = (token_ids.numel() // sequence_length) * sequence_length
    return token_ids[:usable].view(-1, sequence_length)


def window_text(tokenizer, text: str, max_tokens: int, sequence_length: int) -> torch.Tensor:
    """The first `max_tokens` tokens of running text, in windows."""
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    return window_tokens(token_ids.squeeze(0)[:max_tokens], sequence_length)


# --------------------------------------------------------------------------
# Chat format
# --------------------------------------------------------------------------


def resolve_chat_format(
    tokenizer, prompt_prefix: str | None = None, prompt_suffix: str | None = None, prompt_marker: str | None = None
) -> tuple[str, str, int]:
    """
    Returns `(prefix, suffix, marker)` with `prefix` and `suffix` as text and
    `marker` as a token id: the first token of the assistant turn, at which the
    pair's gate is read. Explicit arguments override the plain-text defaults.
    """
    prefix = prompt_prefix or "User: "
    suffix = prompt_suffix or "\nAssistant"
    marker_text = prompt_marker or ":"
    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError(f"marker {marker_text!r} is not a single token: {marker_ids}")
    marker = marker_ids[0]
    joined = tokenizer.encode(suffix + marker_text, add_special_tokens=False)
    if joined != tokenizer.encode(suffix, add_special_tokens=False) + [marker]:
        raise ValueError(f"{suffix + marker_text!r} does not tokenize as suffix + marker")
    return prefix, suffix, marker


def build_response_sequences(
    tokenizer, conversations: list[tuple[str, str]], marker_token: int, max_length: int
) -> tuple[list[list[int]], list[int]]:
    """
    Sequences "chat-formatted prompt + marker + response" and, for each, the
    index of the first response token.
    """
    separator = "" if tokenizer.decode([marker_token]).endswith("\n") else " "
    sequences: list[list[int]] = []
    starts: list[int] = []
    for prompt, response in conversations:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) + [marker_token]
        response_ids = tokenizer.encode(separator + response, add_special_tokens=False)
        response_ids = response_ids[: max(1, max_length - len(prompt_ids))]
        sequence = (prompt_ids + response_ids)[-max_length:]
        sequences.append(sequence)
        starts.append(len(sequence) - len(response_ids))
    return sequences, starts


CHATML_PREFIX = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
    "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
)
CHATML_SUFFIX = "<|im_end|>\n<|im_start|>assistant"
CHATML_MARKER = "\n"
LETTERS = "ABCD"


def resolve_format(tokenizer, model_name: str, choice: str) -> tuple[str, str, int]:
    if choice == "auto":
        choice = "chatml" if "instruct" in model_name.lower() else "plain"
    if choice == "chatml":
        return resolve_chat_format(tokenizer, CHATML_PREFIX, CHATML_SUFFIX, CHATML_MARKER)
    return resolve_chat_format(tokenizer)


def letter_token_ids(tokenizer, marker: int) -> list[int]:
    space = "" if tokenizer.decode([marker]).endswith("\n") else " "
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(space + letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{space + letter!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


# --------------------------------------------------------------------------
# Hooks: capture at the marker, and the hook-level stand-in for the neuron
# --------------------------------------------------------------------------

class MarkerCapture:
    """
    For the candidate blocks, the MLP input `n = g ⊙ x / rms(x)` (what an
    installed gate row reads) and the block's output residual (where an
    installed down column writes). Kept for the whole batch; the caller
    slices the marker position.
    """

    def __init__(self, architecture, layers: list[int]) -> None:
        self.layers = layers
        self.active = False
        self.mlp_in: dict[int, torch.Tensor] = {}
        self.block_out: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer in layers:
            block = architecture.blocks[layer]
            self.handles.append(architecture.mlp_gate(block).register_forward_pre_hook(self._mlp_hook(layer)))
            self.handles.append(block.register_forward_hook(self._block_hook(layer)))

    def _mlp_hook(self, layer):
        def hook(_module, inputs) -> None:
            if self.active:
                self.mlp_in[layer] = inputs[0].detach().float()
        return hook

    def _block_hook(self, layer):
        def hook(_module, _inputs, output) -> None:
            if self.active:
                hidden = output[0] if isinstance(output, tuple) else output
                self.block_out[layer] = hidden.detach().float()
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class NeuronHook:
    """
    Hook-level stand-in for the neuron pair in one block: reads the MLP input
    at the last position, fires when `weight · n + bias >= 0` (`conditional`),
    on every sequence (`always`) or where a caller-supplied mask says so
    (`oracle`, the ceiling for a perfect gate), and adds `alpha * direction`
    to the block's output at that position.
    """

    def __init__(self, architecture, layer: int) -> None:
        block = architecture.blocks[layer]
        self.mode: str | None = None
        self.weight = self.direction = self.fired = None
        self.oracle: torch.Tensor | None = None
        self.offset = 0
        self.alpha = 0.0
        self.bias = 0.0
        self.handles = [
            architecture.mlp_gate(block).register_forward_pre_hook(self._read),
            block.register_forward_hook(self._write),
        ]

    def configure(self, weight, direction, alpha, mode, bias: float = 0.0) -> None:
        self.weight, self.direction, self.alpha, self.mode, self.bias = weight, direction, alpha, mode, bias

    def disable(self) -> None:
        self.mode = None

    def _read(self, _module, inputs) -> None:
        if self.mode is None:
            return
        n = inputs[0][:, -1].float()
        if self.mode == "conditional":
            self.fired = (n @ self.weight + self.bias) >= 0
        elif self.mode == "oracle":
            self.fired = self.oracle[self.offset : self.offset + n.shape[0]]
        else:
            self.fired = torch.ones(n.shape[0], dtype=torch.bool, device=n.device)

    def _write(self, _module, _inputs, output):
        if self.mode is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        delta = (self.alpha * self.direction).to(hidden.dtype)
        hidden = hidden.clone()
        hidden[:, -1] += self.fired[:, None].to(hidden.dtype) * delta
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class RunResult(NamedTuple):
    letter_logits: torch.Tensor     # (n, 4)
    top_is_letter: torch.Tensor     # (n,) bool: the greedy token is one of the letters
    fired: torch.Tensor | None      # (n,) bool when a hook was active
    mlp_in: dict[int, torch.Tensor]     # layer -> (n, hidden) at the marker
    block_out: dict[int, torch.Tensor]  # layer -> (n, hidden) at the marker
    others: dict[int, torch.Tensor]     # layer -> (m, hidden) MLP input at other positions

    @property
    def answers(self) -> torch.Tensor:
        return self.letter_logits.argmax(dim=1)


@torch.no_grad()
def run_prompts(
    architecture, tokenizer, prompts, marker, letter_ids, batch_size, max_length, device,
    capture: MarkerCapture | None = None, other_cap: int = 0, hook: NeuronHook | None = None,
    store: torch.device | None = None,
) -> RunResult:
    """Forward `prompt + marker`; read the letter logits at the marker."""
    store = store or device
    sequences = marker_sequences(tokenizer, prompts, marker, max_length)
    letters = torch.tensor(letter_ids, device=device)
    total_other = sum(len(sequence) - 1 for sequence in sequences)
    fraction = min(1.0, other_cap / max(total_other, 1))
    layers = capture.layers if capture else []
    out = {"logits": [], "top": [], "fired": []}
    mlp_in = {layer: [] for layer in layers}
    block_out = {layer: [] for layer in layers}
    others = {layer: [] for layer in layers}
    if capture:
        capture.active = True
    try:
        for start in range(0, len(sequences), batch_size):
            input_ids, attention_mask = pad_left(tokenizer, sequences[start : start + batch_size], device)
            if hook is not None:
                hook.offset = start
            logits = architecture.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=positions_from_mask(attention_mask), logits_to_keep=1,
            ).logits[:, -1].float()
            out["logits"].append(logits[:, letters])
            out["top"].append(torch.isin(logits.argmax(dim=-1), letters))
            if hook is not None and hook.mode is not None:
                out["fired"].append(hook.fired.clone())
            if capture:
                keep = attention_mask.bool()
                keep[:, -1] = False
                if fraction < 1.0:
                    keep &= torch.rand(keep.shape, device=device) < fraction
                for layer in layers:
                    mlp_in[layer].append(capture.mlp_in[layer][:, -1].half().to(store))
                    block_out[layer].append(capture.block_out[layer][:, -1].half().to(store))
                    if other_cap:
                        others[layer].append(capture.mlp_in[layer][keep].half().to(store))
    finally:
        if capture:
            capture.active = False
            capture.mlp_in.clear()
            capture.block_out.clear()
    return RunResult(
        torch.cat(out["logits"]), torch.cat(out["top"]),
        torch.cat(out["fired"]) if out["fired"] else None,
        {layer: torch.cat(mlp_in[layer]) for layer in layers},
        {layer: torch.cat(block_out[layer]) for layer in layers},
        {layer: torch.cat(others[layer]) for layer in layers if others[layer]},
    )


@torch.no_grad()
def capture_text_tokens(
    architecture, capture, tokenizer, text, max_tokens, sequence_length, batch_size, device, store=None
) -> dict[int, torch.Tensor]:
    """MLP input at every token of running text, for the candidate layers."""
    store = store or device
    windows = window_text(tokenizer, text, max_tokens, sequence_length)
    collected = {layer: [] for layer in capture.layers}
    capture.active = True
    try:
        for start in range(0, len(windows), batch_size):
            input_ids = windows[start : start + batch_size].to(device)
            architecture.model(input_ids=input_ids, logits_to_keep=1)
            for layer in capture.layers:
                collected[layer].append(capture.mlp_in[layer].reshape(-1, architecture.hidden_size).half().to(store))
    finally:
        capture.active = False
        capture.mlp_in.clear()
        capture.block_out.clear()
    return {layer: torch.cat(rows) for layer, rows in collected.items()}


@torch.no_grad()
def capture_sequence_positions(
    architecture, capture, tokenizer, sequences, starts, batch_size, device, store=None, max_rows=None
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    MLP input at the tokens of `prompt + marker + response` sequences, split
    into response tokens (positions >= start) and prompt tokens (before the
    marker), for the candidate layers.
    """
    store = store or device
    response = {layer: [] for layer in capture.layers}
    prompt = {layer: [] for layer in capture.layers}
    capture.active = True
    try:
        for begin in range(0, len(sequences), batch_size):
            batch = sequences[begin : begin + batch_size]
            input_ids, attention_mask = pad_left(tokenizer, batch, device)
            width = input_ids.shape[1]
            pad = torch.tensor([width - len(s) for s in batch], device=device)
            first = pad + torch.tensor(starts[begin : begin + batch_size], device=device)
            positions = torch.arange(width, device=device)[None, :]
            is_response = positions >= first[:, None]
            is_prompt = (positions < (first - 1)[:, None]) & attention_mask.bool()
            architecture.model(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=positions_from_mask(attention_mask), logits_to_keep=1)
            for layer in capture.layers:
                rows = capture.mlp_in[layer]
                response[layer].append(rows[is_response].half().to(store))
                prompt[layer].append(rows[is_prompt].half().to(store))
    finally:
        capture.active = False
        capture.mlp_in.clear()
        capture.block_out.clear()
    cat = lambda d: {layer: torch.cat(rows)[:max_rows] if rows else torch.zeros(0, 0) for layer, rows in d.items()}
    return cat(response), cat(prompt)


# --------------------------------------------------------------------------
# Fits
# --------------------------------------------------------------------------


def _gram_float64(x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """`x.T @ x` accumulated in float64 over row chunks (no float64 copy of `x`)."""
    gram = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
    for start in range(0, len(x), chunk):
        block = x[start : start + chunk].double()
        gram += block.T @ block
    return gram


def fit_constant_direction(
    features: torch.Tensor, others: torch.Tensor | None = None, ridge: float = 1e-2
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Ridge regression for a direction `v` with `v·n` ≈ 1 on `features` (response
    positions) and ≈ 0 on `others` (prompt and running-text tokens). This is the
    pair's up row: it confines the write to responses and carries the gate's
    threshold.
    """
    gram = _gram_float64(features)
    target = features.sum(dim=0, dtype=torch.float64)
    if others is not None and len(others):
        scale = len(features) / len(others)
        gram += scale * _gram_float64(others)
    dim = gram.shape[0]
    gram += ridge * (gram.trace() / dim) * torch.eye(dim, dtype=torch.float64, device=gram.device)
    direction = torch.linalg.solve(gram, target).float()
    projection = features @ direction
    stats = {
        "mean": float(projection.mean()),
        "cv": float(projection.std() / projection.mean().abs()),
        "min": float(projection.min()),
        "max": float(projection.max()),
    }
    if others is not None and len(others):
        other_projection = others @ direction
        stats["others_mean"] = float(other_projection.mean())
        stats["others_absmax"] = float(other_projection.abs().max())
        stats["others_q999"] = float(torch.quantile(other_projection.abs(), 0.999))
    return direction, stats


@torch.no_grad()
def write_mlp_neuron(architecture, block, neuron: int, gate_row: torch.Tensor, up_row: torch.Tensor) -> int:
    """Overwrite one SwiGLU neuron's gate and up rows; returns the number of entries written."""
    gate = architecture.mlp_gate(block)
    up = architecture.mlp_up(block)
    gate.weight[neuron].copy_(gate_row.to(gate.weight.dtype))
    up.weight[neuron].copy_(up_row.to(up.weight.dtype))
    return 2 * architecture.hidden_size


# --------------------------------------------------------------------------
# The neuron pair
# --------------------------------------------------------------------------

def train_write_direction(
    architecture, tokenizer, sequences, continuations, layer, steps, batch_size, device, lr, penalty
):
    """
    The write vector: fitted so that, with the vector added to the block
    output, the model's next-token predictions over a free-form response match
    the response it gives without the belief. `sequences` are belief prompts
    ending in the marker; `continuations[i]` the tokens the model generated for
    the same question with no belief, teacher forced (tokens after the first
    end-of-text are ignored). The vector is added at every response position,
    as the installed pair writes it.
    """
    delta = torch.zeros(architecture.hidden_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    block = architecture.blocks[layer]
    k = max(len(c) for c in continuations)
    eos = tokenizer.eos_token_id

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        mask = torch.zeros(hidden.shape[1], device=device, dtype=hidden.dtype)
        mask[-(k + 1):] = 1.0
        hidden = hidden + mask[None, :, None] * delta[None, None, :].to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def labels_for(index):
        rows = []
        for i in index:
            tokens = list(continuations[i])
            if eos in tokens:  # keep the end-of-text token as a target, ignore the padding after it
                tokens = tokens[: tokens.index(eos) + 1]
            rows.append(tokens + [-100] * (k - len(tokens)))
        return torch.tensor(rows, device=device)

    handle = block.register_forward_hook(hook)
    generator = torch.Generator().manual_seed(0)
    history = []
    try:
        for step in range(steps):
            index = torch.randperm(len(sequences), generator=generator)[:batch_size].tolist()
            labels = labels_for(index)
            batch = [list(sequences[i]) + [t if t != -100 else eos for t in labels[j].tolist()] for j, i in enumerate(index)]
            input_ids, attention_mask = pad_left(tokenizer, batch, device)
            logits = architecture.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=positions_from_mask(attention_mask), logits_to_keep=k + 1,
            ).logits[:, :-1].float()  # positions marker .. k-1 predict the k response tokens
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            loss = loss + penalty * delta.pow(2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
    finally:
        handle.remove()
    vector = delta.detach()
    return vector, {"loss_first": history[0], "loss_last": sum(history[-10:]) / len(history[-10:]), "norm": float(vector.norm())}


@torch.no_grad()
def mean_difference_detector(
    positives: torch.Tensor, negatives: torch.Tensor, fpr: float, threshold_negatives: torch.Tensor | None = None,
    pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Training-free gate: the difference of class means, scaled so the
    negatives' projections have unit standard deviation, with the bias set
    so that a fraction `fpr` of the negatives fire. Returns `(w, b)` with the
    gate firing when `w·n + b >= 0`. `threshold_negatives`, when given, are
    prompts of other formats that must not fire: the direction separating
    them from `negatives` (a prompt-format direction) is projected out of `w`,
    and they join the negatives for the threshold. `pairs`, when given, are
    per-source `(positives, negatives)` whose mean differences are averaged
    with equal weight to form `w` (a within-format contrast, so that formats
    with more items or larger shifts do not dominate); `positives` /
    `negatives` then only set the scale and the threshold.
    """
    if pairs:
        w = torch.stack([p.mean(0) - n.mean(0) for p, n in pairs]).mean(0)
    else:
        w = positives.mean(0) - negatives.mean(0)
    # `threshold_negatives` may be one tensor or a list of groups (e.g. ordinary prompt markers, ordinary response
    # tokens): each group's offset from `negatives` is projected out in turn, and the threshold is the strictest
    # over all groups, so that no group fires more than `fpr` (a pooled quantile would let a small group fire freely).
    groups = [] if threshold_negatives is None else (
        list(threshold_negatives) if isinstance(threshold_negatives, (list, tuple)) else [threshold_negatives]
    )
    groups = [g for g in groups if g is not None and len(g)]
    basis = []
    for group in groups:
        fmt = group.mean(0) - negatives.mean(0)
        for b in basis:
            fmt = fmt - (fmt @ b) * b
        fmt = fmt / fmt.norm().clamp_min(1e-6)
        w = w - (w @ fmt) * fmt
        basis.append(fmt)
    w = w / (negatives @ w).std().clamp_min(1e-6)
    bias = -max(float(torch.quantile(g @ w, 1.0 - fpr)) for g in [negatives, *groups])
    return w, bias


def lda_detector(
    positives: torch.Tensor, negatives: torch.Tensor, fpr: float, threshold_negatives: torch.Tensor | list | None = None,
    shrinkage: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """
    Whitened mean difference (Fisher's discriminant): the positives' mean
    minus the negatives', multiplied by the inverse of the within-group
    covariance pooled over the positives, the negatives and every threshold
    group, shrunk towards a multiple of the identity. Directions along which
    any of the groups varies a lot are discounted, so that the gate reads the
    contrast rather than whatever else separates prompts of different kinds.
    Scaled and thresholded as `mean_difference_detector`.
    """
    groups = [] if threshold_negatives is None else (
        list(threshold_negatives) if isinstance(threshold_negatives, (list, tuple)) else [threshold_negatives]
    )
    groups = [g for g in groups if g is not None and len(g)]
    dim = positives.shape[1]
    cov = torch.zeros(dim, dim, device=positives.device, dtype=torch.float32)
    # Every group's covariance has equal weight, whatever its size, so that a large group of ordinary tokens does not
    # decide the whitening on its own.
    all_groups = [g for g in [positives, negatives, *groups] if len(g) > 1]
    for group in all_groups:
        centred = group.float() - group.float().mean(0)
        cov += centred.T @ centred / (len(group) - 1)
    cov /= max(1, len(all_groups))
    shrunk = (1 - shrinkage) * cov + shrinkage * (cov.trace() / dim) * torch.eye(dim, device=cov.device)
    w = torch.linalg.solve(shrunk, (positives.float().mean(0) - negatives.float().mean(0)))
    w = w / (negatives @ w).std().clamp_min(1e-6)
    bias = -max(float(torch.quantile(g @ w, 1.0 - fpr)) for g in [negatives, *groups])
    return w, bias


def pair_multiplier(n, weight, constant, scale, cap, bias: float = 0.0) -> torch.Tensor:
    """
    `m` such that the installed pair writes `alpha * m * d` at these tokens:
    `m = u (silu(z) - silu(z - cap u)) / cap` with `z = scale (w·n + b u)`,
    `u = v·n`. ~1 at a marker where the detector fires, ~0 elsewhere. The
    detector's threshold `b` rides on the marker constant, which is ~1 there.
    """
    u = n @ constant
    z = scale * (n @ weight + bias * u)
    return u * (F.silu(z) - F.silu(z - cap * u)) / cap


@torch.no_grad()
def simulated_pair_multiplier(n, weight, constant, scale, cap, bias, dtype) -> torch.Tensor:
    """
    `pair_multiplier` as the installed weights compute it: the two gate rows and
    the up row rounded to the model dtype independently, inputs and the three
    projections rounded too (fp32 accumulation, as the GPU matmul does).
    """
    gate = scale * (weight + bias * constant)
    rising = gate.to(dtype).float()
    capping = (gate - cap * constant).to(dtype).float()
    up = constant.to(dtype).float()
    x = n.to(dtype).float()
    z_rising = (x @ rising).to(dtype).float()
    z_capping = (x @ capping).to(dtype).float()
    u = (x @ up).to(dtype).float()
    return u * (F.silu(z_rising) - F.silu(z_capping)) / cap


@torch.no_grad()
def install_steering_neurons(
    architecture, layer, detector_weight, constant_direction, direction, alpha, detector_scale, cap,
    detector_bias: float = 0.0, neurons: tuple[int, int] | None = None,
) -> dict:
    """
    Overwrite two neurons of block `layer`'s MLP (`neurons`; by default the
    last two) with the steering pair: gate rows `scale * (w + b v)` and
    `scale * (w + b v) - cap * v`, up row `v`, down columns
    `+- (alpha / cap) * d`, so the saturated write is `alpha * d`. A detector
    threshold is folded into the gate row through the marker constant
    (`v·n ~ 1` at markers).
    """
    block = architecture.blocks[layer]
    down = architecture.mlp_down(block)
    rising, capping = neurons or (architecture.intermediate_size - 1, architecture.intermediate_size - 2)
    assert rising != capping
    gate = detector_scale * (detector_weight + detector_bias * constant_direction)
    gain = alpha / cap
    for neuron, offset, sign in ((rising, 0.0, 1.0), (capping, -cap, -1.0)):
        write_mlp_neuron(architecture, block, neuron, gate + offset * constant_direction, constant_direction)
        down.weight[:, neuron].copy_((sign * gain * direction).to(down.weight.dtype))
    return {"layer": layer, "neurons": [rising, capping], "selected_entries": 2 * 3 * architecture.hidden_size}


def install_checkpoint(architecture, checkpoint: dict, alpha: float | None = None, device=None,
                       neurons: tuple[int, int] | None = None) -> dict:
    """Install a saved pair (a `self_steering.py` checkpoint) into the model's weights."""
    c = checkpoint["circuit"]
    to = lambda t: t.to(device).float() if device is not None else t.float()
    return install_steering_neurons(
        architecture, c["layer"], to(c["detector_weight"]), to(c["constant_direction"]), to(c["direction"]),
        c["alpha"] if alpha is None else alpha, c["detector_scale"], c["cap"], c["detector_bias"], neurons=neurons,
    )


# --------------------------------------------------------------------------
# Generation and collateral metrics
# --------------------------------------------------------------------------


def greedy_continuations(
    model, tokenizer, prompts: list[str], marker_token: int, batch_size: int, max_length: int, device: torch.device,
    **generate_kwargs,
):
    """Greedy continuations of `prompt + marker`, one batch at a time (new tokens only)."""
    for start in range(0, len(prompts), batch_size):
        sequences = marker_sequences(tokenizer, prompts[start : start + batch_size], marker_token, max_length)
        input_ids, attention_mask = pad_left(tokenizer, sequences, device)
        generated = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, **generate_kwargs,
        )
        yield generated[:, input_ids.shape[1]:]


@torch.no_grad()
def generate_responses(
    model, tokenizer, prompts: list[str], marker_token: int, max_new_tokens: int, batch_size: int, max_length: int,
    device: torch.device,
) -> list[str]:
    """Greedy responses of up to `max_new_tokens` after `prompt + marker`, as text."""
    return [
        tokenizer.decode(row, skip_special_tokens=True).strip()
        for continuation in greedy_continuations(
            model, tokenizer, prompts, marker_token, batch_size, max_length, device, max_new_tokens=max_new_tokens
        )
        for row in continuation
    ]


@torch.no_grad()
def evaluate_language_model(
    model, token_ids: torch.Tensor, sequence_length: int, batch_size: int, device: torch.device
) -> dict[str, float]:
    """Next-token NLL / perplexity / accuracy on windows of `token_ids`."""
    sequences = window_tokens(token_ids, sequence_length)
    total_nll, total_correct, total_tokens = 0.0, 0, 0
    for start in range(0, len(sequences), batch_size):
        input_ids = sequences[start : start + batch_size].to(device)
        logits = model(input_ids=input_ids).logits.float()
        predicted = logits[:, :-1].reshape(-1, logits.shape[-1])
        targets = input_ids[:, 1:].reshape(-1)
        total_nll += float(F.cross_entropy(predicted, targets, reduction="sum"))
        total_correct += int((predicted.argmax(dim=-1) == targets).sum())
        total_tokens += targets.numel()
    average_nll = total_nll / total_tokens
    return {
        "nll": average_nll,
        "perplexity": math.exp(average_nll),
        "next_token_accuracy": total_correct / total_tokens,
        "evaluated_tokens": float(total_tokens),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name}: nll={metrics['nll']:.6f} perplexity={metrics['perplexity']:.4f} "
        f"next_token_accuracy={metrics['next_token_accuracy']:.4%} tokens={int(metrics['evaluated_tokens'])}"
    )


@torch.no_grad()
def response_nll(architecture, tokenizer, sequences, starts, batch_size, device, _steerers=()) -> dict[str, float]:
    """Teacher-forced next-token NLL over the response tokens of `prompt + marker + response`."""
    total, count = 0.0, 0
    for begin in range(0, len(sequences), batch_size):
        batch = sequences[begin : begin + batch_size]
        input_ids, attention_mask = pad_left(tokenizer, batch, device)
        width = input_ids.shape[1]
        pad = torch.tensor([width - len(s) for s in batch], device=device)
        first_response = pad + torch.tensor(starts[begin : begin + batch_size], device=device)
        positions = torch.arange(width, device=device)[None, :]
        logits = forward(architecture, input_ids, attention_mask).logits.float()
        nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1), reduction="none"
        ).view(len(batch), -1)
        target_is_response = positions[:, 1:] >= first_response[:, None]
        total += float((nll * target_is_response).sum())
        count += int(target_is_response.sum())
    return {"nll": total / count, "perplexity": float(torch.tensor(total / count).exp()), "tokens": count}

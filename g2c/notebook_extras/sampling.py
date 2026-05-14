"""Notebook ergonomics for language-model sampling notebooks.

The real Module 11 deliverable lives in ``g2c.sampling``. This module keeps
Jupyter notebooks readable by wrapping common prompt/tokenizer/model plumbing.
It also includes a deliberately naive inspection sampler for Module 10, where
the student has not built ``g2c.sampling.generate`` yet.
"""

from __future__ import annotations

from html import escape
from time import perf_counter
from typing import Any

import torch

from g2c.artifacts import LoadedModelArtifact

__all__ = [
    "decode_ids",
    "default_sampling_prompt",
    "encode_prompt",
    "eos_id",
    "next_token_rows",
    "next_token_probability_rows",
    "plot_next_token_probabilities",
    "print_sample",
    "printable",
    "readable_token_mask",
    "sample_ids",
    "sample_model_text",
    "sample_text",
    "show_model_samples",
    "show_next_token_distribution",
    "stream_sample_text",
    "token_is_readable",
    "type_token_ratio",
]


def default_sampling_prompt(artifact: LoadedModelArtifact) -> str:
    """Return a reasonable default prompt for the loaded artifact family."""
    name = artifact.canonical_name.lower()
    source = str(artifact.manifest.get("source", "")).lower()
    if "shakespeare" in name or "shakespeare" in source:
        return "KING:"
    if "story" in name or "tinystories" in source:
        return "Once upon a time,"
    return "A neural network is"


def eos_id(artifact: LoadedModelArtifact) -> int | None:
    """Return the first course end-token ID available in the artifact tokenizer."""
    tokenizer = artifact.tokenizer
    for token in ("<|endoftext|>", "<|end|>"):
        token_id = tokenizer.special_to_id.get(token)
        if token_id is not None:
            return token_id
    token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is not None:
        return int(token_id)
    return None


def encode_prompt(artifact: LoadedModelArtifact, prompt: str) -> torch.Tensor:
    """Encode a prompt with the model's effective vocab size.

    Module 10 can train a large tokenizer while training a model with only a
    prefix of that vocabulary. Encoding with ``model.vocab_size`` prevents
    prompt IDs from exceeding the model embedding table.
    """
    ids = artifact.tokenizer.encode_with_vocab_size(
        prompt,
        artifact.model.vocab_size,
    )
    if not ids:
        raise ValueError("prompt encoded to no tokens")
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(artifact: LoadedModelArtifact, ids: torch.Tensor | list[int]) -> str:
    """Decode token IDs with the artifact tokenizer."""
    if isinstance(ids, torch.Tensor):
        values = [int(x) for x in ids.tolist()]
    else:
        values = [int(x) for x in ids]
    return artifact.tokenizer.decode(values)


def printable(text: str) -> str:
    """Escape control characters that can make notebook output hard to read."""
    has_control = any(ord(ch) < 32 and ch not in "\n\t" for ch in text)
    return text.encode("unicode_escape").decode("ascii") if has_control else text


def sample_ids(
    artifact: LoadedModelArtifact,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.1,
    seed: int = 0,
) -> torch.Tensor:
    """Sample token IDs from a loaded artifact using the Module 11 generator."""
    from g2c.sampling import generate

    prompt_ids = encode_prompt(artifact, prompt)
    generator = torch.Generator().manual_seed(seed)
    return generate(
        artifact.model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=eos_id(artifact),
        generator=generator,
    )


def sample_text(
    artifact: LoadedModelArtifact,
    prompt: str,
    *,
    stream: bool = False,
    stream_title: str | None = None,
    update_every: int = 4,
    **kwargs: Any,
) -> str:
    """Sample and decode text from a loaded artifact."""
    if stream:
        return stream_sample_text(
            artifact,
            prompt,
            title=stream_title,
            update_every=update_every,
            **kwargs,
        )
    return decode_ids(artifact, sample_ids(artifact, prompt, **kwargs))


@torch.no_grad()
def stream_sample_text(
    artifact: LoadedModelArtifact,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.1,
    seed: int = 0,
    title: str | None = None,
    update_every: int = 4,
) -> str:
    """Sample text while updating one Jupyter output block.

    This is a notebook ergonomics wrapper around the same loop students build
    in ``g2c.sampling.generate``. It exists because external BaseLM artifacts
    can take long enough that a live partial sample is easier to work with than
    a blank cell followed by a large printout.
    """
    from g2c.sampling import (
        apply_repetition_penalty,
        apply_temperature,
        top_k_filter,
        top_p_filter,
    )

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if update_every <= 0:
        raise ValueError(f"update_every must be positive, got {update_every}")

    model = artifact.model
    prompt_ids = encode_prompt(artifact, prompt)
    generator = torch.Generator().manual_seed(seed)
    greedy = temperature == 0.0
    full_ids = prompt_ids.detach().cpu().clone()
    stop_id = eos_id(artifact)
    device = getattr(model, "device", torch.device("cpu"))
    label = title or f"streaming sample after {prompt!r}"
    start = perf_counter()
    handle = _start_stream_display(label, decode_ids(artifact, full_ids), 0, max_new_tokens)

    for step in range(max_new_tokens):
        ctx = full_ids[-model.max_seq_len:].to(device).unsqueeze(0)
        logits = model(ctx)
        last_logits = logits[:, -1, :].cpu()

        if greedy:
            next_id = last_logits.argmax(dim=-1)
        else:
            if repetition_penalty != 1.0:
                last_logits = apply_repetition_penalty(
                    last_logits,
                    full_ids,
                    repetition_penalty,
                )
            last_logits = apply_temperature(last_logits, temperature)
            if top_k is not None:
                last_logits = top_k_filter(last_logits, top_k)
            if top_p is not None:
                last_logits = top_p_filter(last_logits, top_p)

            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)

        full_ids = torch.cat([full_ids, next_id], dim=0)
        done = stop_id is not None and int(next_id.item()) == stop_id
        if done or step + 1 == max_new_tokens or (step + 1) % update_every == 0:
            _update_stream_display(
                handle,
                label,
                decode_ids(artifact, full_ids),
                step + 1,
                max_new_tokens,
                perf_counter() - start,
                done=done,
            )
        if done:
            break

    text = decode_ids(artifact, full_ids)
    if handle is None:
        print_sample(label, text)
    return text


def _start_stream_display(
    title: str,
    text: str,
    step: int,
    total: int,
) -> Any:
    try:
        from IPython.display import HTML, display
    except ImportError:
        return None
    return display(
        HTML(_stream_html(title, text, step, total, elapsed=0.0, done=False)),
        display_id=True,
    )


def _update_stream_display(
    handle: Any,
    title: str,
    text: str,
    step: int,
    total: int,
    elapsed: float,
    *,
    done: bool,
) -> None:
    if handle is None:
        return
    from IPython.display import HTML

    handle.update(HTML(_stream_html(title, text, step, total, elapsed=elapsed, done=done)))


def _stream_html(
    title: str,
    text: str,
    step: int,
    total: int,
    *,
    elapsed: float,
    done: bool,
) -> str:
    status = "done" if done or step >= total else "generating"
    return (
        "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace;'>"
        f"<div><strong>{escape(title)}</strong> "
        f"<span style='color:#666'>[{status} {step:,}/{total:,}; {elapsed:.1f}s]</span></div>"
        "<pre style='white-space: pre-wrap; line-height: 1.35; "
        "border-left: 3px solid #ccc; padding-left: 0.75rem;'>"
        f"{escape(printable(text))}</pre></div>"
    )


def print_sample(title: str, text: str, *, width: int = 88) -> None:
    """Print one titled sample block."""
    print("\n" + title)
    print("-" * width)
    print(printable(text))


def next_token_rows(
    artifact: LoadedModelArtifact,
    prompt: str,
    *,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    n: int = 12,
) -> list[tuple[str, float]]:
    """Return top next-token strings and probabilities after selected warpers."""
    from g2c.sampling import apply_temperature, top_k_filter, top_p_filter

    model = artifact.model
    prompt_ids = encode_prompt(artifact, prompt)
    ctx = prompt_ids[-model.max_seq_len:].to(model.device).unsqueeze(0)
    with torch.no_grad():
        logits = model(ctx)[:, -1, :].cpu()
    warped = apply_temperature(logits, temperature)
    if top_k is not None:
        warped = top_k_filter(warped, top_k)
    if top_p is not None:
        warped = top_p_filter(warped, top_p)
    probs = torch.softmax(warped, dim=-1)[0]
    values, indices = torch.topk(probs, k=min(n, probs.numel()))
    rows = []
    for value, token_id in zip(values.tolist(), indices.tolist(), strict=True):
        token_text = (
            decode_ids(artifact, [token_id])
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        rows.append((repr(token_text), value))
    return rows


def type_token_ratio(ids: torch.Tensor) -> float:
    """Return distinct-token count divided by total-token count."""
    values = [int(x) for x in ids.tolist()]
    return len(set(values)) / max(1, len(values))


def show_next_token_distribution(
    rows: list[tuple[str, float]],
    *,
    prompt: str,
    title: str | None = None,
) -> None:
    """Plot and print the top next-token probabilities returned by ``next_token_rows``."""
    import matplotlib.pyplot as plt

    labels = [token_text for token_text, _ in rows]
    probs = [prob for _, prob in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = list(range(len(rows)))
    ax.barh(y, probs)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("probability")
    ax.set_title(title or f"Native next-token probabilities after {prompt!r}")
    for index, prob in enumerate(probs):
        ax.text(prob, index, f" {prob:.4f}", va="center")
    fig.tight_layout()
    plt.show()

    print("native next-token distribution")
    print("token                probability")
    print("-" * 34)
    for token_text, prob in rows:
        print(f"{token_text:<20} {prob:.4f}")


def token_is_readable(tokenizer: Any, token_id: int) -> bool:
    """Return whether one token decodes to printable notebook text."""
    piece = tokenizer.decode([token_id])
    if not piece or "\ufffd" in piece:
        return False
    return all(ch in "\n\t" or ch.isprintable() for ch in piece)


def readable_token_mask(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """Return a bool mask for tokens whose decoded text is printable."""
    return torch.tensor(
        [token_is_readable(tokenizer, token_id) for token_id in range(vocab_size)],
        dtype=torch.bool,
    )


@torch.no_grad()
def next_token_probability_rows(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    n: int = 15,
    temperature: float = 1.0,
    printable_only: bool = True,
) -> list[tuple[int, str, float]]:
    """Return top next-token IDs, decoded strings, and probabilities.

    This is the Module 10 inspection path: it computes ``model(prompt)``,
    reads the final-position logits, applies temperature, softmaxes, and
    returns the largest probabilities. It intentionally avoids
    ``g2c.sampling`` because Module 11 has not taught that package yet.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive for a probability plot")
    prompt_ids = tokenizer.encode_with_vocab_size(prompt, model.vocab_size)
    if not prompt_ids:
        raise ValueError("prompt encoded to no tokens")

    device = next(iter(model.parameters())).device
    ctx = torch.tensor(
        prompt_ids[-model.max_seq_len:],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    logits = model(ctx)[0, -1].detach().cpu()
    if printable_only:
        mask = readable_token_mask(tokenizer, model.vocab_size)
        logits = logits.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(logits / temperature, dim=-1)
    values, indices = torch.topk(probs, k=min(n, probs.numel()))
    rows: list[tuple[int, str, float]] = []
    for token_id, prob in zip(indices.tolist(), values.tolist(), strict=True):
        token_text = (
            tokenizer.decode([int(token_id)])
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        rows.append((int(token_id), token_text, float(prob)))
    return rows


def plot_next_token_probabilities(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    n: int = 15,
    temperature: float = 1.0,
    printable_only: bool = True,
    title: str | None = None,
) -> list[tuple[int, str, float]]:
    """Plot a horizontal bar chart of the top next-token probabilities."""
    import matplotlib.pyplot as plt

    rows = next_token_probability_rows(
        model,
        tokenizer,
        prompt,
        n=n,
        temperature=temperature,
        printable_only=printable_only,
    )
    labels = [f"{text!r}  [{token_id}]" for token_id, text, _ in rows]
    probs = [prob for _, _, prob in rows]

    fig_height = max(4.0, 0.36 * len(rows))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = list(range(len(rows)))
    ax.barh(y, probs, color="#4c78a8")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("probability after softmax")
    ax.set_title(title or f"Top next-token probabilities after {prompt!r}")
    max_prob = max(probs) if probs else 1.0
    ax.set_xlim(0, min(1.0, max_prob * 1.2 + 0.01))
    for index, prob in enumerate(probs):
        ax.text(prob, index, f" {prob:.3f}", va="center")
    fig.tight_layout()
    plt.show()
    return rows


def _apply_top_k_for_inspection(
    logits: torch.Tensor,
    top_k: int | None,
) -> torch.Tensor:
    if top_k is None:
        return logits
    k = min(top_k, logits.numel())
    cutoff = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < cutoff, float("-inf"))


@torch.no_grad()
def sample_model_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.5,
    top_k: int | None = 20,
    printable_only: bool = True,
    seed: int = 0,
) -> str:
    """Naively sample from a model for Module 10 training inspection.

    This intentionally does not call ``g2c.sampling.generate``. Module 10 comes
    before the sampling module; it only needs a tiny local decode loop so the
    student can inspect whether a checkpoint is learning anything.
    """
    device = next(iter(model.parameters())).device
    prompt_ids = tokenizer.encode_with_vocab_size(prompt, model.vocab_size)
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(seed)
    printable_mask = (
        readable_token_mask(tokenizer, model.vocab_size) if printable_only else None
    )

    for _ in range(max_new_tokens):
        ctx = ids[-model.max_seq_len:].unsqueeze(0)
        logits = model(ctx)[0, -1].clone()
        if printable_mask is not None:
            mask = printable_mask.to(device=logits.device)
            logits = logits.masked_fill(~mask, float("-inf"))
        if temperature == 0.0:
            next_id = logits.argmax().reshape(1)
        else:
            logits = _apply_top_k_for_inspection(logits / temperature, top_k)
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            next_id = torch.multinomial(probs, 1, generator=generator).to(device)
        ids = torch.cat([ids, next_id.to(device=ids.device)])

    return tokenizer.decode(ids.detach().cpu().tolist())


def show_model_samples(
    name: str,
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    seed: int = 1,
) -> None:
    """Print raw and readable samples for a Module 10 checkpoint."""
    raw = sample_model_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=300,
        temperature=0.8,
        top_k=None,
        printable_only=False,
        seed=seed,
    )
    readable = sample_model_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=300,
        temperature=0.5,
        top_k=20,
        printable_only=True,
        seed=seed,
    )
    print(f"{name} raw sample, escaped so control bytes do not wreck notebook output")
    print("-" * 72)
    print(raw.encode("unicode_escape", errors="backslashreplace").decode("ascii"))
    print(f"\n{name} readable top-k sample")
    print("-" * 72)
    print(readable)

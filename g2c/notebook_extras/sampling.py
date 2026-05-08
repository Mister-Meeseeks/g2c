"""Notebook ergonomics for language-model sampling notebooks.

The real Module 11 deliverable lives in ``g2c.sampling``. This module keeps
Jupyter notebooks readable by wrapping common prompt/tokenizer/model plumbing.
It also includes a deliberately naive inspection sampler for Module 10, where
the student has not built ``g2c.sampling.generate`` yet.
"""

from __future__ import annotations

from typing import Any

import torch

from g2c.artifacts import LoadedModelArtifact

__all__ = [
    "decode_ids",
    "default_sampling_prompt",
    "encode_prompt",
    "eos_id",
    "next_token_rows",
    "print_sample",
    "printable",
    "readable_token_mask",
    "sample_ids",
    "sample_model_text",
    "sample_text",
    "show_model_samples",
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


def sample_text(artifact: LoadedModelArtifact, prompt: str, **kwargs: Any) -> str:
    """Sample and decode text from a loaded artifact."""
    return decode_ids(artifact, sample_ids(artifact, prompt, **kwargs))


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

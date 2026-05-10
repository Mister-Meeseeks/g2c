"""Autoregressive generation with a KV cache.

This is the optional performance sibling of ``generate``. It deliberately keeps
the same sampling surface but uses ``model.forward_cached`` so each generated
token reuses K/V projections from earlier tokens instead of recomputing them.

Scope constraints:

* inference only;
* one sequence at a time;
* no rolling cache once ``model.max_seq_len`` is reached;
* intended for greedy/sampling demos, not production serving.
"""
from __future__ import annotations

import torch

from .repetition_penalty import apply_repetition_penalty
from .temperature import apply_temperature
from .top_k import top_k_filter
from .top_p import top_p_filter


@torch.no_grad()
def generate_cached(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressively continue ``prompt_ids`` using ``model.forward_cached``.

    The output contract matches ``generate``: a 1-D tensor containing the prompt
    followed by up to ``max_new_tokens`` sampled continuation tokens.
    """
    if prompt_ids.dim() != 1 or prompt_ids.numel() == 0:
        raise ValueError(f"prompt_ids must be a non-empty 1-D tensor, got {prompt_ids}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"Temperature must be non-negative, got {temperature}")
    if not hasattr(model, "forward_cached"):
        raise TypeError("generate_cached requires a model with forward_cached")
    if prompt_ids.numel() >= model.max_seq_len:
        raise ValueError(
            "generate_cached needs at least one free position in the KV cache; "
            f"prompt length {prompt_ids.numel()} >= max_seq_len {model.max_seq_len}"
        )

    vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is not None:
        min_id = int(prompt_ids.min().item())
        max_id = int(prompt_ids.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(
                "prompt_ids contain token IDs outside the model vocab: "
                f"min={min_id}, max={max_id}, model.vocab_size={vocab_size}. "
                "Encode prompts with tokenizer.encode_with_vocab_size(..., "
                "model.vocab_size)."
            )

    greedy = temperature == 0.0
    full_ids = prompt_ids.detach().cpu().clone()
    device = getattr(model, "device", torch.device("cpu"))
    cache = model.empty_kv_cache() if hasattr(model, "empty_kv_cache") else None

    last_logits = None
    for token_id in prompt_ids:
        logits, cache = model.forward_cached(token_id.to(device).view(1, 1), cache)
        last_logits = logits[:, -1, :].cpu()
    if last_logits is None:
        raise RuntimeError("prompt prefill did not produce logits")

    for _ in range(max_new_tokens):
        next_id = _sample_next_id(
            last_logits,
            full_ids,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )
        full_ids = torch.cat([full_ids, next_id], dim=0)

        if eos_id is not None and next_id.item() == eos_id:
            break
        if full_ids.numel() >= model.max_seq_len:
            break

        logits, cache = model.forward_cached(next_id.to(device).view(1, 1), cache)
        last_logits = logits[:, -1, :].cpu()

    return full_ids


def _sample_next_id(
    logits: torch.Tensor,
    full_ids: torch.Tensor,
    *,
    greedy: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if greedy:
        return logits.argmax(dim=-1)

    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, full_ids, repetition_penalty)
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

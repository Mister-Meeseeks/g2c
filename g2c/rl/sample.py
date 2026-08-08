"""sample_group — K fresh attempts at one prompt, via Module 11's sampler.

On-policy RL's data comes from the model itself: every training step
samples a GROUP of completions for the same prompt at temperature > 0,
scores them, and lets the group grade itself. This file is the sampling
half — provided plumbing wrapping `g2c.sampling.generate`.

The tokenizer is duck-typed: anything with `encode(str) -> list[int]`
and `decode(list[int]) -> str` works (the course tokenizers and the
BaseLM adapter both do).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import torch

from g2c.sampling import generate


class GroupSample(NamedTuple):
    """K sampled completions for one prompt.

    Attributes:
        prompt_text: the prompt as given.
        prompt_ids: `(T_prompt,)` LongTensor of the encoded prompt.
        completions: K 1-D LongTensors of completion token ids
            (prompt NOT included; lengths vary).
        texts: the K decoded completion strings, for the verifier.
    """

    prompt_text: str
    prompt_ids: torch.Tensor
    completions: list[torch.Tensor]
    texts: list[str]


def sample_group(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> GroupSample:
    """Sample `k` completions of `prompt` from the CURRENT model.

    Temperature must be positive: greedy sampling would make every
    group member identical, every group degenerate, and every gradient
    zero. Exploration is not optional — it is where the signal comes
    from.
    """
    if k < 2:
        raise ValueError(f"a group needs k >= 2 completions, got {k}")
    if temperature <= 0:
        raise ValueError(
            "sample_group needs temperature > 0 — greedy sampling makes "
            "every completion identical and the group carries no signal"
        )
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    completions: list[torch.Tensor] = []
    texts: list[str] = []
    for _ in range(k):
        full = generate(
            model,
            prompt_ids,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            generator=generator,
        )
        completion = full[prompt_ids.numel() :]
        completions.append(completion)
        texts.append(tokenizer.decode(completion.tolist()))
    return GroupSample(prompt, prompt_ids, completions, texts)

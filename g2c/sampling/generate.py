"""Autoregressive generation — the loop that turns a model into text.

A trained `TransformerLM` is a function `(B, T) → (B, T, V)`. By
itself it can't produce text; you have to drive it in a loop:

    1. Forward-pass the prompt to get logits at the last position.
    2. Convert those logits to a probability distribution over `V`.
    3. Sample one token from that distribution.
    4. Append the sampled token to the running sequence.
    5. Repeat from (1) with the extended sequence.

That's the entire algorithm. Everything else in this module — the
four logit warpers — is shaping STEP 2 to favor higher-quality tokens
over the long tail. The order of operations in the loop is the rest.

```
    prompt_ids                               (T_prompt,) int
        │
        ▼
    ┌── for new_token in range(max_new_tokens): ──────────────────────┐
    │                                                                  │
    │   1. CROP: ctx = full_ids[-max_seq_len:]   (model can't see      │
    │                                              past max_seq_len)   │
    │                                                                  │
    │   2. FORWARD: logits = model(ctx[None, :])  (1, T, V)            │
    │                                                                  │
    │   3. SLICE: last_logits = logits[:, -1, :]  (1, V)               │
    │                                                                  │
    │   4. WARP, in this order:                                        │
    │        last_logits = repetition_penalty(last_logits, full_ids)   │
    │        last_logits = temperature(last_logits, T)                 │
    │        last_logits = top_k_filter(last_logits, k)                │
    │        last_logits = top_p_filter(last_logits, p)                │
    │                                                                  │
    │   5. SOFTMAX: probs = softmax(last_logits)  (1, V)               │
    │                                                                  │
    │   6. SAMPLE: next_id = multinomial(probs, 1)                     │
    │     (or argmax if temperature == 0 / "greedy" mode)              │
    │                                                                  │
    │   7. APPEND: full_ids = cat([full_ids, next_id])                 │
    │                                                                  │
    │   8. STOP if next_id == eos_id                                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
        │
        ▼
    full_ids                                 (T_prompt + n_generated,) int
```

The four design points to internalize:

  * **Crop the context.** `TransformerLM` only has positional
    embeddings up to `max_seq_len`. Once `len(full_ids) > max_seq_len`,
    further tokens have no positional signal — sample-time crash, or
    silently-wrong outputs. Standard fix: feed only the last
    `max_seq_len` tokens.

  * **Slice the LAST position's logits.** `forward(ctx)` returns
    `(1, T, V)` — predictions at every position in parallel, including
    every position OTHER than the most recent one. We only want the
    prediction for "the next token after the last input token," so we
    take `logits[:, -1, :]`.

  * **Warp before softmax.** Every warper operates on logits, not
    probabilities. The pipeline is `repetition_penalty → temperature →
    top_k → top_p → softmax → multinomial`. Reordering doesn't
    necessarily produce wrong outputs but produces *different* outputs
    in subtle ways; this is the canonical order.

  * **`temperature == 0` is greedy mode.** Bypasses temperature,
    top-k, top-p, repetition-penalty — and uses `argmax` instead of
    `multinomial`. Greedy can never produce two different outputs
    from the same prompt, regardless of `generator`.

Scaffolded.
"""
from __future__ import annotations

import torch

from .repetition_penalty import (
    apply_repetition_penalty,  # noqa: F401 (for the student implementation)
)
from .temperature import apply_temperature  # noqa: F401 (for the student implementation)
from .top_k import top_k_filter  # noqa: F401 (for the student implementation)
from .top_p import top_p_filter  # noqa: F401 (for the student implementation)


@torch.no_grad()
def generate(
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
    """Autoregressively continue `prompt_ids` for up to `max_new_tokens`.

    Args:
        model: any module with a `forward(token_ids: (1, T))` that
            returns `(1, T, V)` logits. `TransformerLM` from Module 09
            is the prototype, but the loop is model-agnostic. The
            model is also expected to expose `max_seq_len: int` so
            generate can crop the context window.
        prompt_ids: 1-D LongTensor `(T_prompt,)`. Must be non-empty
            and at most `model.max_seq_len` tokens.
        max_new_tokens: how many new tokens to sample.
        temperature: softmax temperature. `1.0` is the model's native
            distribution; smaller values are sharper, larger flatter.
            `0.0` is the special "greedy" mode (argmax — deterministic).
        top_k: keep at most this many of the highest-probability
            tokens at each step. `None` disables top-k.
        top_p: keep tokens whose cumulative probability reaches this.
            `None` disables top-p.
        repetition_penalty: CTRL-style multiplicative penalty on
            tokens already present in the running sequence. `1.0` is
            no penalty. Typical values: `1.05`–`1.3`.
        eos_id: optional end-of-sequence token id. If provided,
            generation stops as soon as this token is sampled (it IS
            included in the output).
        generator: optional `torch.Generator` for reproducible
            sampling. Has no effect in greedy mode.

    Returns:
        1-D LongTensor of length at most `T_prompt + max_new_tokens`,
        equal to `prompt_ids` followed by the sampled continuation.
        Shorter if `eos_id` was emitted before `max_new_tokens`.

    Recipe:
        1. # Validation:
           if prompt_ids.dim() != 1 or prompt_ids.numel() == 0:
               raise ValueError(...)
           if temperature < 0:
               raise ValueError(...)

        2. # Decide once whether we're in greedy mode. Greedy
           # bypasses ALL warpers (the warpers are about reshaping a
           # sampled distribution; argmax doesn't sample).
           greedy = (temperature == 0.0)

        3. full_ids = prompt_ids.detach().cpu().clone()
           # Keep the running token IDs on CPU for easy decoding and
           # reproducible sampling. Move only the per-step context to
           # the model's device for the forward pass.
           device = getattr(model, "device", torch.device("cpu"))

        4. for _ in range(max_new_tokens):
               # 4a. Crop context to the model's positional capacity.
               ctx = full_ids[-model.max_seq_len:]
               ctx = ctx.to(device).unsqueeze(0)           # (1, T_ctx)

               # 4b. Forward pass; extract last position's logits.
               logits = model(ctx)                         # (1, T_ctx, V)
               last_logits = logits[:, -1, :].cpu()        # (1, V)

               if greedy:
                   # 4c-greedy. Argmax — no warpers, no sampling.
                   next_id = last_logits.argmax(dim=-1)    # (1,)
               else:
                   # 4c-sample. Warpers in canonical order.
                   if repetition_penalty != 1.0:
                       last_logits = apply_repetition_penalty(
                           last_logits, full_ids, repetition_penalty
                       )
                   last_logits = apply_temperature(last_logits, temperature)
                   if top_k is not None:
                       last_logits = top_k_filter(last_logits, top_k)
                   if top_p is not None:
                       last_logits = top_p_filter(last_logits, top_p)

                   probs = torch.softmax(last_logits, dim=-1)   # (1, V)
                   next_id = torch.multinomial(
                       probs, num_samples=1, generator=generator
                   ).squeeze(-1)                           # (1,)

               # 4d. Append the new token.
               full_ids = torch.cat([full_ids, next_id], dim=0)

               # 4e. Early stop on EOS.
               if eos_id is not None and next_id.item() == eos_id:
                   break

        5. return full_ids

    Notes:
      - The whole function is wrapped in `@torch.no_grad()` (see the
        decorator above the signature). Generation is inference; we
        don't need autograd, and skipping it saves both memory and
        compute.
      - The order of warpers — repetition → temperature → top-k →
        top-p — matches HuggingFace's `LogitsProcessorList` default.
        Reordering produces different distributions in non-trivial
        ways; if you're reproducing a paper's recipe, check what
        order it used.
      - This loop is `O(max_new_tokens × T_ctx²)` in attention compute
        because every step recomputes attention from scratch. KV
        caching makes it `O(max_new_tokens × T_ctx)`; we don't cache
        in this module — the lesson here is the loop, not the
        optimization.
    """
    # TODO
    raise NotImplementedError

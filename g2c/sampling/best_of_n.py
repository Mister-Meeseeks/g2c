"""Best-of-N sampling — generate several candidates, then pick one.

Every warper so far (temperature, top-k, top-p, repetition penalty)
shapes the distribution *during* decoding: one pass, one sample, no
second guesses. Best-of-N does something different. It spends compute
at inference time instead of at training time — generate `n`
independent continuations, score each one, return the best.

This is the simplest member of a family that has become central to
modern systems ("test-time compute"): sample several answers and pick
among them. Everything more sophisticated — self-consistency voting,
reranking with a reward model, tree search — is a variation on the two
functions in this file.

The scoring function is where the interesting idea lives:

  * **A language model can score text, not just produce it.** The same
    forward pass that generates gives you `log p(token | prefix)` for
    *any* token you like — including tokens you already have. Feed a
    finished sequence back through the model, read off the log-prob it
    assigned to each actual next token, and sum: that is the model's
    total log-probability for that sequence. Nothing new is needed.
    This is called *teacher forcing*, and it is also exactly how the
    Module 09B/10 training loss is computed — the loss you trained on
    is the negative of this score, averaged.

  * **Raw log-probability is a biased judge.** Longer sequences have
    lower log-probability, always: every extra token adds another
    negative number. Ranking by raw sum quietly prefers short, safe,
    repetitive text — the exact degeneration mode top-p was introduced
    to avoid. `length_normalize=True` divides by token count (mean
    log-prob per token), which is the standard first fix. It is not a
    *complete* fix, and noticing why is part of the exercise.

Two scaffolded functions. `sequence_log_prob` is about five lines of
tensor indexing; `best_of_n` is a loop over `generate` plus an argmax.
"""
from __future__ import annotations

import torch

from .generate import generate  # noqa: F401 (for the student implementation)


def sequence_log_prob(
    model,
    token_ids: torch.Tensor,
    *,
    prompt_len: int = 0,
) -> float:
    """Score an existing sequence: total log-probability under `model`.

    Args:
        model: any module with `forward(token_ids: (1, T)) -> (1, T, V)`
            logits — the same contract `generate` expects.
        token_ids: 1-D LongTensor `(T,)` of the full sequence to score,
            prompt included. Must have at least 2 tokens (the first
            token has no preceding context, so it cannot be scored).
        prompt_len: number of leading tokens to exclude from the score.
            Use this to score only the *continuation* — comparing
            candidates that share a prompt is otherwise dominated by
            the prompt's own log-probability, which is identical for
            every candidate and therefore pure noise in the ranking.

    Returns:
        Python float: the summed log-probability of every scored
        position. Always negative (log of a probability); closer to
        zero means the model found the sequence more predictable.

    Raises:
        ValueError: if `token_ids` is not 1-D, has fewer than 2 tokens,
            or `prompt_len` leaves nothing to score.

    Recipe:
        1. # Validation (shape, length, prompt_len bounds).
        2. # One forward pass over the whole sequence. Position t's
           # logits predict the token at position t+1 — the same
           # off-by-one alignment as the Module 09B language-model
           # loss.
           with torch.no_grad():
               logits = model(token_ids.unsqueeze(0))       # (1, T, V)
           log_probs = torch.log_softmax(logits[0], dim=-1)  # (T, V)
        3. # Drop the last position (nothing follows it to predict)
           # and the first `prompt_len` targets.
           start = max(prompt_len - 1, 0)
           pred = log_probs[start:-1]                        # (S, V)
           targets = token_ids[start + 1:]                   # (S,)
        4. # Pick out the log-prob the model assigned to each token
           # that actually came next.
           chosen = pred.gather(1, targets.unsqueeze(1)).squeeze(1)
        5. return float(chosen.sum())

    Note `log_softmax` rather than `softmax(...).log()` — the fused
    form is numerically stable, the same reason Module 06's loss uses
    it.
    """
    # TODO
    raise NotImplementedError


def best_of_n(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    n: int = 8,
    length_normalize: bool = False,
    generator: torch.Generator | None = None,
    **generate_kwargs,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, float]]]:
    """Sample `n` continuations and return the highest-scoring one.

    Args:
        model: as in `generate`.
        prompt_ids: 1-D LongTensor `(T_prompt,)`.
        max_new_tokens: per-candidate generation budget.
        n: how many candidates to sample. Must be `>= 1`.
        length_normalize: if True, rank by mean log-prob per generated
            token instead of the raw sum.
        generator: optional `torch.Generator`. Candidates must differ,
            so the same generator is threaded through every call
            rather than being reset — reusing a *fresh* seeded
            generator per candidate would produce `n` identical
            samples.
        **generate_kwargs: forwarded to `generate` (temperature,
            top_k, top_p, repetition_penalty, eos_id). Sampling must
            be stochastic for this to be meaningful: with
            `temperature=0.0` every candidate is the same greedy
            output.

    Returns:
        `(best_ids, scored)` where `best_ids` is the winning full
        sequence (prompt + continuation) and `scored` is the list of
        `(ids, score)` pairs in generation order, so callers can
        inspect the whole candidate pool — the exercise plots it.

    Raises:
        ValueError: if `n < 1`.

    Recipe:
        1. if n < 1: raise ValueError(...)
        2. prompt_len = prompt_ids.numel()
        3. scored = []
           for _ in range(n):
               ids = generate(model, prompt_ids, max_new_tokens,
                              generator=generator, **generate_kwargs)
               total = sequence_log_prob(model, ids, prompt_len=prompt_len)
               if length_normalize:
                   new_tokens = max(ids.numel() - prompt_len, 1)
                   total = total / new_tokens
               scored.append((ids, total))
        4. best_ids = max(scored, key=lambda pair: pair[1])[0]
        5. return best_ids, scored

    The loop is deliberately sequential and obvious. Batching the `n`
    candidates through one padded forward pass is the real
    implementation and a reasonable extension — but it buries the idea
    under padding and attention-mask bookkeeping, and the idea is the
    point.
    """
    # TODO
    raise NotImplementedError

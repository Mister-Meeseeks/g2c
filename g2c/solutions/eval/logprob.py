# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.eval.logprob pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def continuation_logprob(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
) -> tuple[float, int]:
    """Sum of log-probabilities the model assigns to `continuation`
    given `prompt`.

    Args:
        model: any module with a `forward(x: (1, T)) -> (1, T, V)`.
            Will be called via `model(x)` (i.e. `__call__`, not
            `model.forward(x)` directly — same as the rest of the
            course).
        tokenizer: any object with `encode(s: str) -> list[int]`.
            `g2c.tokenizer.bpe.BPETokenizer` is the prototype.
        prompt: the conditioning text. Must be non-empty (you need at
            least one token of context for the autoregressive forward
            to predict the first continuation token).
        continuation: the text to score. Must be non-empty (a 0-token
            continuation has trivially `(0.0, 0)` log-prob, but the
            caller almost certainly meant to ask about a non-empty
            string — we raise instead of silently returning 0).

    Returns:
        (sum_logp, n_continuation_tokens):
            sum_logp: float — Σ_t log p(c_t | prompt, c_{<t}) over
                continuation tokens, in nats. Length-NORMALIZE outside
                this function if you want a per-token mean.
            n_continuation_tokens: int — number of continuation tokens
                scored. Useful for length normalization and for the
                caller to verify their tokenization is what they expected.

    Recipe:
        1. # Tokenize separately so we know the boundary.
           prompt_ids = tokenizer.encode(prompt)
           cont_ids   = tokenizer.encode(continuation)
           if len(prompt_ids) == 0:    raise ValueError(...)
           if len(cont_ids)   == 0:    raise ValueError(...)

        2. # Build the joined sequence and move to the model's device.
           full_ids = prompt_ids + cont_ids                    # list[int]
           full = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)  # (1, T)
           # If the model has a `.parameters()` iterable, grab one
           # parameter to find its device:
           device = next(iter(model.parameters())).device
           full = full.to(device)

        3. # Build x, y, and the mask. Same shift-by-one as Modules 13–14.
           x = full[:, :-1]                                    # (1, T-1)
           y = full[:, 1:]                                     # (1, T-1)
           # Mask covers positions whose TARGET is a continuation
           # token. The continuation starts at full-index `len(prompt_ids)`,
           # so in y (which is full[1:]) it starts at index
           # `len(prompt_ids) - 1`.
           mask = torch.zeros_like(y, dtype=torch.float)
           mask[:, len(prompt_ids) - 1:] = 1.0                 # (1, T-1)

        4. # Forward + gather + masked sum (the same math as DPO's
           # tensor-level sequence_logprob).
           with torch.no_grad():
               logits = model(x)                               # (1, T-1, V)
           log_probs = F.log_softmax(logits, dim=-1)           # (1, T-1, V)
           target_lp = log_probs.gather(
               dim=-1, index=y.unsqueeze(-1)
           ).squeeze(-1)                                        # (1, T-1)

        5. # Sum the masked positions and return as a Python float.
           total = (target_lp * mask).sum().item()
           return float(total), len(cont_ids)

    Implementation notes:

      * `torch.no_grad()` is mandatory here — eval doesn't need autograd
        and skipping it cuts memory roughly in half. The function is
        always called outside of training.

      * `next(iter(model.parameters())).device` is the standard idiom
        for "what device is this model on." Works for both PyTorch
        modules and the course's `g2c.nn.Module` base class.

      * The mask uses `float` dtype because the multiply at step 4
        needs a floating-point operand. (The DPO version cast inside
        the function; we do it at construction time here.)

      * `n_continuation_tokens` equals `len(cont_ids)` exactly — every
        continuation token contributes one masked position to the
        sum. Worth pinning in tests because off-by-one bugs in the
        mask are the single most common source of log-prob errors.

    Sanity values:

      * Uniform-logit model (every logit zero) on any continuation:
        returns `(-T_cont * log(V), T_cont)` where V is the vocab
        size. Pinned by `test_continuation_logprob_uniform_logits`.

      * Tiny model where prompt and continuation are the SAME tokens:
        the function still returns the log-prob of the continuation
        tokens given the prompt — there's no special "is this a
        repeat" handling. This matters for repetition-detection
        diagnostics.

      * Two continuations of identical token-length on the same prompt:
        their sum_logps are directly comparable. Two continuations
        of DIFFERENT token-length: the longer one is systematically
        lower (more negative log-probs to sum). For
        length-fair multiple choice scoring, divide by
        `n_continuation_tokens`.
    """
    prompt_ids = tokenizer.encode(prompt)
    cont_ids = tokenizer.encode(continuation)
    if len(prompt_ids) == 0:
        raise ValueError("prompt must encode to at least one token")
    if len(cont_ids) == 0:
        raise ValueError("continuation must encode to at least one token")

    full_ids = prompt_ids + cont_ids
    device = _model_device(model)
    full = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    x = full[:, :-1]
    y = full[:, 1:]

    mask = torch.zeros_like(y, dtype=torch.float32)
    mask[:, len(prompt_ids) - 1 :] = 1.0

    with torch.no_grad():
        logits = model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(
            dim=-1,
            index=y.unsqueeze(-1),
        ).squeeze(-1)
        total = (target_log_probs * mask).sum()

    return float(total.item()), len(cont_ids)

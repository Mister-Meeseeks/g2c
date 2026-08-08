"""The three GRPO pieces: advantages, completion log-probs, and the loss.

The whole update in one line each:

    A_i  = (r_i − mean(r)) / std(r)          # the group is the baseline
    logp = Σ log p(completion token | prefix)  # prompt tokens EXCLUDED
    loss = −(A · logp).mean() + β · KL(p_θ ‖ p_ref)

Everything else in the module (sampling, verifying, the optimizer step)
is plumbing around these three functions. All three are scaffolded —
they are the deliverable.
"""
from __future__ import annotations

import torch

# Below this, a group's rewards are considered all-equal (degenerate).
DEGENERATE_STD = 1e-8


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Center and scale one group's rewards into advantages.

    Args:
        rewards: `(K,)` float tensor — one scalar reward per completion
            in the group (all for the SAME prompt).

    Returns:
        `(K,)` tensor of advantages: zero mean, unit std... except for
        a DEGENERATE group (std below `DEGENERATE_STD` — all rewards
        equal), which returns all zeros. A group where every attempt
        scored the same carries no information about which attempt was
        better; zeros make the policy-gradient term vanish instead of
        dividing by zero and poisoning the step with NaNs.

    Recipe:
        1. mean = rewards.mean()
           std = rewards.std(unbiased=False)     # population std
        2. if std < DEGENERATE_STD: return torch.zeros_like(rewards)
        3. return (rewards - mean) / std

    Use the POPULATION std (`unbiased=False`). The default unbiased
    estimator divides by K-1 — not wrong, but every published GRPO
    recipe normalizes by the group's own spread, and the tests pin
    that convention.
    """
    # TODO
    raise NotImplementedError


def completion_log_prob(
    model,
    ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Sum of token log-probs over the COMPLETION span only.

    Args:
        model: anything with `forward((B, T)) -> (B, T, V)` logits —
            `TransformerLM` or BaseLM's adapter.
        ids: `(B, T)` LongTensor — prompt + completion, concatenated.
            Every row must share the same `prompt_len` (true within a
            group: same prompt, K completions).
        prompt_len: number of leading prompt tokens. Must satisfy
            `1 <= prompt_len < T`.

    Returns:
        `(B,)` tensor — for each row, `Σ log p(ids[t] | ids[:t])`
        summed over completion positions `t >= prompt_len` only.

    Raises:
        ValueError: if `prompt_len` is out of range.

    Recipe:
        1. B, T = ids.shape
           if not 1 <= prompt_len < T: raise ValueError(...)
        2. logits = model(ids[:, :-1])                      # (B, T-1, V)
           logprobs = torch.log_softmax(logits, dim=-1)
        3. targets = ids[:, 1:]                             # (B, T-1)
           token_lp = logprobs.gather(
               -1, targets.unsqueeze(-1)
           ).squeeze(-1)                                    # (B, T-1)
        4. # On the shifted grid, position j predicts ids[:, j+1].
           # Completion tokens are ids[:, prompt_len:], so the
           # completion's log-probs live at j >= prompt_len - 1:
           mask = torch.zeros_like(token_lp)
           mask[:, prompt_len - 1 :] = 1.0
        5. return (token_lp * mask).sum(dim=-1)             # (B,)

    This is Module 13's loss-mask seam wearing a new hat: supervise
    (here: score) the model's own tokens, never the prompt's. Get the
    off-by-one wrong and you are reinforcing the model's opinion of
    the QUESTION.
    """
    # TODO
    raise NotImplementedError


def grpo_loss(
    logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    kl_coef: float,
) -> torch.Tensor:
    """The GRPO objective: policy-gradient term plus the KL leash.

    Args:
        logp: `(K,)` completion log-probs under the CURRENT model —
            attached to the graph (gradients flow through this).
        ref_logp: `(K,)` completion log-probs under the FROZEN
            reference model. Must be detached / computed under
            `torch.no_grad()` — the leash anchors to a fixed point.
        advantages: `(K,)` from `group_advantages`. Treated as
            constants (they come from rewards, which have no graph).
        kl_coef: β, the leash strength. 0 disables the leash — the
            configuration the lesson's sabotage exercise runs.

    Returns:
        Scalar loss. Minimizing it pushes UP the log-prob of
        above-average completions, DOWN below-average ones, while the
        KL term penalizes drifting from the reference.

    Recipe:
        1. pg = -(advantages.detach() * logp).mean()
        2. # The k3 KL estimator (Schulman): with d = ref_logp - logp,
           #     KL ≈ mean( exp(d) - d - 1 )
           # Per-sample it is >= 0, exactly 0 iff logp == ref_logp,
           # and unlike the naive (logp - ref_logp) estimator its
           # gradient doesn't push all sampled sequences down
           # uniformly.
           d = ref_logp.detach() - logp
           kl = (d.exp() - d - 1.0).mean()
        3. return pg + kl_coef * kl
    """
    # TODO
    raise NotImplementedError

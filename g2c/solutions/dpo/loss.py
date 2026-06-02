# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.dpo.loss pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logprob(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sum of log-probabilities the model assigns to each example's
    masked targets.

    Args:
        logits: (B, T, V). One logit vector per (batch, position).
            From `model(x)` where `x = ids[:-1]` for a (B, T+1)
            original sequence.
        targets: (B, T) integer LongTensor. Each `targets[b, t]` is
            the token whose log-probability we want — typically
            `ids[1:]`, i.e. the next-token target.
        mask: (B, T) integer or float tensor. `mask[b, t] != 0` means
            "include this position's log-prob in the sum"; `0` means
            "ignore." For DPO, `mask = 1` exactly on response (chosen
            or rejected) tokens including trailing `<|end|>`, and `0`
            on prompt tokens and padding.

    Returns:
        (B,) tensor — for each batch element, the SUM of
        log-probabilities `Σ_t mask[b,t] * log p(targets[b,t] | x[b,:t+1])`
        over the masked positions. Note: SUM, not mean — DPO's
        log-ratio is between sums (= log of joint probability of the
        response tokens), not means.

    Recipe:
        1. log_probs = F.log_softmax(logits, dim=-1)        # (B, T, V)

        2. # Gather the log-prob of each target along the vocab dim.
           target_log_probs = log_probs.gather(
               dim=-1, index=targets.unsqueeze(-1)
           ).squeeze(-1)                                     # (B, T)

        3. mask = mask.to(target_log_probs.dtype)            # cast to float
           # (caller may pass long; the multiply needs floating dtype.)

        4. # Sum over the time dim with mask weighting.
           return (target_log_probs * mask).sum(dim=-1)      # (B,)

    Implementation notes:

      * Use `F.log_softmax`, not `softmax(...).log()` — log_softmax is
        the numerically stable form. A naive `softmax → log` blows up
        for logits with magnitude > 80 in fp32.

      * `gather` is the vectorized way to pull out one entry per row
        from a 2-D / 3-D tensor. The `unsqueeze(-1)` and `squeeze(-1)`
        adjust the index tensor's shape for `gather`'s contract.

      * Masked positions contribute exactly zero to the sum because
        we multiply by `0` before summing. Their gradient through
        `target_log_probs` is also zero.

    Sanity values:

      * If `mask` is all 1s and the model is uniform random over
        `V` tokens, `sequence_logprob` returns approximately
        `−T · log(V)` for every example.

      * If `mask` is all 0s, returns a tensor of zeros.

      * The return is `(B,)` — one scalar per example. NOT a single
        scalar across the batch.
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
    target_log_probs = log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    mask = mask.to(target_log_probs.dtype)  # cast to float
    return (target_log_probs * mask).sum(dim=-1)  # (B,)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the DPO loss and diagnostic metrics.

    Args:
        policy_chosen_logp:   (B,) — `logp_pi(chosen | prompt)` from
            the trainable policy. WITH gradient.
        policy_rejected_logp: (B,) — same for rejected.
        ref_chosen_logp:      (B,) — `logp_ref(chosen | prompt)` from
            the FROZEN reference. NO gradient.
        ref_rejected_logp:    (B,) — same for rejected.
        beta: KL-regularization strength (the β in the DPO paper).
            Larger values pin the policy closer to the reference;
            smaller values give the preference signal more room. The
            DPO paper recommends 0.1–0.5; at toy scale 0.1 is a
            reasonable default.

    Returns:
        loss: scalar tensor. The mean DPO loss across the batch.
        metrics: dict of diagnostic tensors (each scalar) for logging:
            'chosen_reward':   mean implicit reward on chosen
                                (β · (policy_chosen_logp − ref_chosen_logp))
            'rejected_reward': mean implicit reward on rejected
                                (β · (policy_rejected_logp − ref_rejected_logp))
            'reward_margin':   mean(chosen_reward − rejected_reward).
                                Positive means policy prefers chosen
                                more strongly than reference does.
            'accuracy':        fraction of examples where chosen_reward
                                > rejected_reward. The classification-
                                style headline number for DPO progress.

    Recipe:
        1. # The two log-ratios.
           pi_logratio  = policy_chosen_logp - policy_rejected_logp     # (B,)
           ref_logratio = ref_chosen_logp    - ref_rejected_logp        # (B,)

        2. # The DPO logits — the input to the sigmoid.
           logits = beta * (pi_logratio - ref_logratio)                 # (B,)

        3. # The DPO loss is -log σ(logits), per-example, averaged.
           # Use F.logsigmoid for numerical stability — it's the
           # standard way to compute log σ(x) without overflow.
           loss = -F.logsigmoid(logits).mean()                          # scalar

        4. # Diagnostic implicit rewards (NO gradient needed for
           # logging, but it's fine to leave them in the graph; the
           # caller can `.detach().item()` at log time).
           chosen_reward   = beta * (policy_chosen_logp   - ref_chosen_logp)    # (B,)
           rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)  # (B,)

        5. metrics = {
               'chosen_reward':   chosen_reward.mean().detach(),
               'rejected_reward': rejected_reward.mean().detach(),
               'reward_margin':   (chosen_reward - rejected_reward).mean().detach(),
               'accuracy':        (chosen_reward > rejected_reward).float().mean().detach(),
           }

        6. return loss, metrics

    Sanity values to remember:

      * Initial step. If `pi_theta = pi_ref` exactly (i.e. the policy
        was just initialized from the reference and no update has
        been applied), then `pi_logratio == ref_logratio` for every
        example, `logits == 0` everywhere, and `loss == -log σ(0)
        == log 2 ≈ 0.6931`. This is the canonical "DPO loss at
        step 0" value — if you see something wildly different at
        step 0, something is wrong with the data, the masking, or
        the reference model.

      * Beta = 0. The DPO logits are identically zero, the loss is
        `log 2`, the gradient is zero. The model never updates.
        This is a useful sanity check: passing `beta=0` is the same
        as not training.

      * Identical chosen and rejected. If `chosen_ids == rejected_ids`
        for every example, then both log-ratios are equal, the DPO
        logits are zero, and the loss is `log 2`. The signal vanishes
        — there's nothing to learn.

      * Saturated preference. As `pi_logratio - ref_logratio → +∞`,
        `loss → 0`. The policy has fully separated chosen from
        rejected (in log-ratio space). At toy scale this rarely
        happens — typical end-of-training reward margins are 1–3
        nats, not infinity.
    """
    pi_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp

    logits = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()

    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)

    metrics = {
        "chosen_reward": chosen_reward.mean().detach(),
        "rejected_reward": rejected_reward.mean().detach(),
        "reward_margin": (chosen_reward - rejected_reward).mean().detach(),
        "accuracy": (chosen_reward > rejected_reward).float().mean().detach(),
    }

    return loss, metrics

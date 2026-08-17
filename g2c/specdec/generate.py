"""The speculative decoding loop — draft, verify in one pass, keep the prefix.

Module 11's `generate` pays one target forward per token. This loop
pays one *drafter* forward per token plus one *target* forward per
BLOCK, and `greedy_verify` guarantees the output is identical to plain
greedy decoding with the target alone. What changes is the accounting:

    plain greedy:   N tokens  →  N target passes
    speculative:    N tokens  →  ~N / (E[accepted] + 1) target passes

`SpecStats` records that accounting — it is the module's measurement
instrument, and "tokens per target pass" is the number the exercises
sweep. Wall-clock speedup is a separate empirical question: at course
scale each verification pass recomputes the full prefix (no KV cache),
so the target-pass savings and the wall-clock savings can diverge.
Measure both; report what you see.

`SpecStats` and `draft_greedy` are provided. `speculative_generate` is
scaffolded.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .verify import greedy_verify  # noqa: F401 (for the student implementation)


@dataclass
class SpecStats:
    """Accounting for one speculative-decoding run.

    Attributes:
        target_passes: verification forward passes through the target.
        drafted: total draft tokens proposed across all blocks.
        accepted: total draft tokens the target accepted.
        generated: tokens actually emitted (accepted + corrections
            + bonus tokens, after any budget/EOS truncation).
    """

    target_passes: int = 0
    drafted: int = 0
    accepted: int = 0
    generated: int = 0

    def record(
        self, *, drafted: int, accepted: int, generated: int
    ) -> None:
        """Record one draft/verify iteration (one target pass)."""
        self.target_passes += 1
        self.drafted += drafted
        self.accepted += accepted
        self.generated += generated

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target accepted."""
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_pass(self) -> float:
        """Emitted tokens per target forward pass — the headline number.

        Plain autoregressive decoding scores exactly 1.0 here; anything
        above 1.0 is serial target work that speculation removed.
        """
        return self.generated / self.target_passes if self.target_passes else 0.0

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return (
            f"SpecStats(passes={self.target_passes}, "
            f"drafted={self.drafted}, accepted={self.accepted}, "
            f"generated={self.generated}, "
            f"acceptance={self.acceptance_rate:.2f}, "
            f"tokens/pass={self.tokens_per_pass:.2f})"
        )


@torch.no_grad()
def draft_greedy(
    drafter, ctx_ids: torch.Tensor, k: int
) -> torch.Tensor:
    """Propose `k` tokens by running the drafter's greedy argmax chain.

    This is Module 11's loop again — one forward per token — run on the
    CHEAP model, where serial steps are affordable. `ctx_ids` is the
    full running sequence `(T,)` on CPU; the context is cropped to the
    drafter's own `max_seq_len` window. Returns `(k,)` LongTensor.
    """
    device = getattr(drafter, "device", torch.device("cpu"))
    ids = ctx_ids.detach().cpu().clone()
    out: list[int] = []
    for _ in range(k):
        ctx = ids[-drafter.max_seq_len:].to(device).unsqueeze(0)
        logits = drafter(ctx)
        next_id = int(logits[0, -1].argmax())
        out.append(next_id)
        ids = torch.cat([ids, torch.tensor([next_id])])
    return torch.tensor(out, dtype=torch.long)


def speculative_generate(
    target,
    drafter,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    k: int = 4,
    eos_id: int | None = None,
) -> tuple[torch.Tensor, SpecStats]:
    """Greedy speculative decoding: drafter proposes, target verifies.

    Args:
        target: the expensive model — any module with
            `forward((1, T)) → (1, T, V)` logits and a `max_seq_len`
            attribute. Module 09's `TransformerLM` is the prototype.
        drafter: the cheap model, same contract and SAME TOKENIZER —
            the two models must share a vocabulary, or the draft ids
            are meaningless to the target. (The StoryLM ladder shares
            `StoryTokenizer`, which is what makes it a ready-made
            drafter/target pair.)
        prompt_ids: 1-D LongTensor `(T_prompt,)`, non-empty.
        max_new_tokens: budget of new tokens to emit.
        k: draft block length per iteration.
        eos_id: optional stop token (kept in the output, as in
            Module 11's `generate`).

    Returns:
        `(full_ids, stats)` — the prompt plus continuation, and the
        `SpecStats` accounting for the run.

    While the sequence stays inside the target's context window, the
    output is token-for-token identical to
    `generate(target, prompt_ids, ..., temperature=0.0)`: greedy
    verification never lets a drafter token survive that the target's
    own argmax chain would not have produced.

    Recipe:
        1. # Validation, as in Module 11's generate:
           if prompt_ids.dim() != 1 or prompt_ids.numel() == 0:
               raise ValueError(...)
           if k < 1:
               raise ValueError(...)

        2. full_ids = prompt_ids.detach().cpu().clone()
           stats = SpecStats()
           device = getattr(target, "device", torch.device("cpu"))

        3. while stats.generated < max_new_tokens:
               # 3a. Never draft more than the remaining budget.
               k_step = min(k, max_new_tokens - stats.generated)

               # 3b. The drafter runs its own cheap greedy chain.
               draft = draft_greedy(drafter, full_ids, k_step)

               # 3c. ONE target pass over context + draft block.
               #     Crop so the concatenation fits the target's
               #     positional table.
               ctx = full_ids[-(target.max_seq_len - k_step):]
               seq = torch.cat([ctx, draft]).to(device).unsqueeze(0)
               logits = target(seq)                     # (1, T, V)

               # 3d. The last k_step + 1 rows are the verification
               #     block: row i predicts draft position i (the token
               #     at that position is draft[i]'s predecessor's
               #     next), and the final row is the bonus position.
               block = logits[0, -(k_step + 1):, :].cpu()

               # 3e. Verify; emit accepted prefix + correction/bonus.
               n_acc, nxt = greedy_verify(draft, block)
               new_ids = torch.cat(
                   [draft[:n_acc], torch.tensor([nxt])]
               )

               # 3f. Cap to the budget, then record BEFORE the EOS cut
               #     so drafted/accepted stay comparable across runs.
               new_ids = new_ids[: max_new_tokens - stats.generated]

               # 3g. Truncate at the first EOS (inclusive), if any.
               if eos_id is not None and (new_ids == eos_id).any():
                   cut = int((new_ids == eos_id).nonzero()[0]) + 1
                   new_ids = new_ids[:cut]

               stats.record(
                   drafted=k_step, accepted=n_acc,
                   generated=int(new_ids.numel()),
               )
               full_ids = torch.cat([full_ids, new_ids])

               if eos_id is not None and full_ids[-1].item() == eos_id:
                   break

        4. return full_ids, stats

    Wrap the loop in `torch.no_grad()` (or decorate like
    `draft_greedy`) — this is inference.
    """
    # TODO
    raise NotImplementedError

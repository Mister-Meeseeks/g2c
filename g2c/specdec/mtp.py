"""A multi-token-prediction head — the model drafts for itself.

Speculative decoding with a separate drafter needs two models. The MTP
alternative, shipped in current releases (DeepSeek V4, GLM-5,
Qwen3.8), attaches a small head to the model that predicts one step
FURTHER ahead: given the trunk's hidden state at position `t` and the
embedding of the (already-predicted) next token `x_{t+1}`, the head
predicts `x_{t+2}`. At decode time that second prediction is a draft,
and the model verifies its own guess on the next pass — speculation
with `k = 1` and no second model to load or train from scratch.

The head is deliberately small: one combiner projection, one
transformer block, one LayerNorm — and it reuses the base model's
embedding table at BOTH ends (input embedding of `x_{t+1}`, tied
unembedding for the logits), exactly like the base LM itself does.
The base stays frozen the same way Module 13B freezes it: the head's
`parameters()` simply does not include the base, so any optimizer
built on `parameters()` never moves it.

Training alignment (the part every implementation gets wrong once):

    ids:      x_0   x_1   x_2   x_3   x_4        (B, T)
    hidden:   h_0   h_1   h_2   h_3   h_4        trunk output, (B, T, D)
    head in:  (h_0, e(x_1)) (h_1, e(x_2)) ...    positions 0 .. T-2
    predicts:  x_2            x_3          ...    two steps ahead

so the head is called as `head(hidden[:, :-1], ids[:, 1:])`, producing
`(B, T-1, V)` logits whose row `t` targets `ids[:, t+2]` — the last
row has no target and is dropped by the loss.

`hidden_states`, the head's boilerplate, and `mtp_propose` are
provided; `MTPHead.forward` and `mtp_loss` are scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.nn import Linear, Module
from g2c.transformer import TransformerLM
from g2c.transformer.block import Block
from g2c.transformer.layer_norm import LayerNorm


@torch.no_grad()
def hidden_states(model: TransformerLM, token_ids: torch.Tensor) -> torch.Tensor:
    """The trunk's final hidden states — `forward` minus the unembedding.

    Provided plumbing: `TransformerLM.forward` returns logits, but the
    MTP head consumes the residual stream just before the unembedding,
    so we rerun the trunk pipeline here. `(B, T)` ids in,
    `(B, T, D)` post-`ln_final` states out.
    """
    _, T = token_ids.shape
    if T > model.max_seq_len:
        raise ValueError(
            f"sequence length {T} exceeds max_seq_len {model.max_seq_len}"
        )
    x = model.token_embed(token_ids) + model.pos_embed(T)
    for block in model.blocks:
        x = block(x)
    return model.ln_final(x)


class MTPHead(Module):
    """One extra prediction step, bolted onto a frozen base LM.

    Args:
        base: the trained `TransformerLM` to draft for. Held by
            reference; NOT included in `parameters()`, so the base
            stays frozen under any optimizer driven by this module's
            parameters — the same freeze-by-omission trick as
            Module 13B's `LoRAModel`.

    Attributes:
        combine: `Linear(2D, D)` merging the trunk state with the
            next token's embedding.
        block: one causal transformer block refining the merged state.
        ln: final LayerNorm before the (tied) unembedding.
        head_bias: the head's own output bias over the vocabulary.
    """

    def __init__(self, base: TransformerLM) -> None:
        super().__init__()
        self.base = base
        D = base.embedding_dim
        self.combine = Linear(2 * D, D)
        self.block = Block(
            D, base.num_heads, hidden_dim=base.hidden_dim, causal=True
        )
        self.ln = LayerNorm(D)
        head_bias = torch.zeros(base.vocab_size)
        head_bias.requires_grad_(True)
        self.head_bias = head_bias

    def parameters(self) -> Iterable[torch.Tensor]:
        # Deliberately excludes `self.base` — training the head must
        # not move the base model.
        return [
            *self.combine.parameters(),
            *self.block.parameters(),
            *self.ln.parameters(),
            self.head_bias,
        ]

    def forward(
        self, hidden: torch.Tensor, next_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Predict two steps ahead from trunk state + next-token embedding.

        Args:
            hidden: `(B, T', D)` trunk hidden states (from
                `hidden_states`), typically `hidden[:, :-1]` of a
                training batch.
            next_token_ids: `(B, T')` — the token at each position + 1
                (teacher forcing during training; the model's own
                just-predicted token at decode time).

        Returns:
            `(B, T', V)` logits; row `t` predicts the token TWO
            positions after `hidden[:, t]`'s.

        Recipe:
            1. e = self.base.token_embed(next_token_ids)   # (B, T', D)
            2. h = self.combine(torch.cat([hidden, e], dim=-1))
            3. h = self.block(h)
               h = self.ln(h)
            4. # Tied unembedding, same as the base LM's forward:
               return h @ self.base.token_embed.weight.T + self.head_bias
        """
        # TODO
        raise NotImplementedError


def mtp_loss(mtp_logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for the two-steps-ahead head.

    Args:
        mtp_logits: `(B, T-1, V)` — the head's output for
            `head(hidden[:, :-1], ids[:, 1:])` on a `(B, T)` batch.
        token_ids: the same `(B, T)` batch of ids.

    Returns:
        Scalar mean cross-entropy over the positions that HAVE a
        two-ahead target.

    Recipe:
        1. # Row t of mtp_logits predicts ids[:, t + 2]. The last row
           # (t = T-2) has no target — drop it:
           logits = mtp_logits[:, :-1, :]          # (B, T-2, V)
           targets = token_ids[:, 2:]              # (B, T-2)
        2. return F.cross_entropy(
               logits.reshape(-1, logits.shape[-1]),
               targets.reshape(-1),
           )

    An off-by-one here is silent and catastrophic: shift the targets by
    one and the head happily learns to predict `x_{t+1}` — a job the
    base model already does — and every "draft" it later proposes gets
    rejected. The alignment test pins the correct shift.
    """
    # TODO
    raise NotImplementedError


@torch.no_grad()
def mtp_propose(
    base: TransformerLM, head: MTPHead, ctx_ids: torch.Tensor
) -> tuple[int, int]:
    """Draft two tokens from ONE trunk pass — the self-speculation step.

    Provided plumbing. Runs the trunk once, reads the base's greedy
    next token `x_{t+1}` from the shared hidden state, then asks the
    head for its guess at `x_{t+2}`. Production MTP serving works the
    same way: the trunk pass is shared, the head ride-along is nearly
    free.

    Returns `(next1, next2)`: `next1` is the base model's own greedy
    token (always correct by definition); `next2` is the draft to be
    verified on the next pass.
    """
    device = getattr(base, "device", torch.device("cpu"))
    ctx = ctx_ids.detach().cpu()[-base.max_seq_len:].to(device).unsqueeze(0)
    h = hidden_states(base, ctx)                      # (1, T, D)
    last = h[:, -1:, :]                               # (1, 1, D)
    logits1 = last @ base.token_embed.weight.T + base.head_bias
    next1 = int(logits1[0, -1].argmax())
    logits2 = head(
        last, torch.tensor([[next1]], device=device)
    )
    next2 = int(logits2[0, -1].argmax())
    return next1, next2

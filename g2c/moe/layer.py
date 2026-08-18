"""MoEFeedForward — the FFN slot of a transformer block, made conditional.

Module 09's block has one `FeedForward` that every token passes through.
This layer keeps the slot but fills it with `E` independent
`FeedForward` experts plus a `Router`: each token runs through only the
`k` experts the router selects, and the outputs are combined with the
router's renormalized weights.

The dispatch is written as a legible loop over experts — gather the
tokens routed to expert `e`, run them, scatter the weighted results
back. Production systems fuse this into batched kernels; we keep the
loop because you can read it.

Parameter accounting (the model-card numbers):

    total  ≈ E · (one FeedForward)  + router
    active ≈ k · (one FeedForward)  + router     per token

Boilerplate (`__init__`, `parameters`) is implemented; `forward` and
`load_balancing_loss` are scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.nn import Module
from g2c.transformer.ffn import FeedForward

from .router import Router


class MoEFeedForward(Module):
    """A routed panel of `FeedForward` experts.

    Args:
        embedding_dim: channel dim (`D`).
        num_experts:   number of expert FFNs (`E`).
        top_k:         experts each token passes through (`k`).
        hidden_dim:    inner dim of each expert. Defaults to
                       `4 * embedding_dim`, matching Module 09.

    Attributes:
        experts: `E` independent `FeedForward` modules.
        router:  the top-k gate.
        last_indices: `(N, top_k)` expert assignments from the most
            recent forward. `None` until the first forward. Read by
            `load_balancing_loss`.
    """

    embedding_dim: int
    num_experts: int
    top_k: int
    experts: list[FeedForward]
    router: Router

    def __init__(
        self,
        embedding_dim: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = [
            FeedForward(embedding_dim, hidden_dim=hidden_dim)
            for _ in range(num_experts)
        ]
        self.router = Router(embedding_dim, num_experts, top_k)
        self.last_indices: torch.Tensor | None = None

    def parameters(self) -> Iterable[torch.Tensor]:
        params: list[torch.Tensor] = []
        for expert in self.experts:
            params.extend(expert.parameters())
        params.extend(self.router.parameters())
        return params

    def expert_parameter_count(self) -> int:
        """Parameters in ONE expert — the unit of the total/active split."""
        return sum(p.numel() for p in self.experts[0].parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route each token through its top-k experts and combine.

        Args:
            x: tensor of shape `(..., embedding_dim)` — typically
                `(B, T, D)` from the block's `ln2`.

        Returns:
            Tensor of the same shape as `x`.

        Recipe:
            1. flat = x.reshape(-1, self.embedding_dim)      # (N, D)
            2. weights, indices = self.router(flat)          # (N, k) each
               self.last_indices = indices
            3. out = torch.zeros_like(flat)
               for e, expert in enumerate(self.experts):
                   selected = (indices == e)                 # (N, k) bool
                   if not selected.any():
                       continue
                   token_idx = selected.any(dim=-1).nonzero(as_tuple=True)[0]
                   # Weight for expert e per selected token — zero at
                   # the k-slots that picked other experts:
                   w = (weights * selected.to(weights.dtype)).sum(dim=-1)
                   w = w[token_idx].unsqueeze(-1)            # (n_sel, 1)
                   expert_out = expert(flat[token_idx])      # (n_sel, D)
                   out = out.index_add(0, token_idx, expert_out * w)
            4. return out.reshape(x.shape)

        Use the FUNCTIONAL `out.index_add(...)` (returns a new tensor),
        not the in-place `index_add_` — the functional form keeps the
        autograd story obvious. Tokens the loop never touches keep
        their zero rows, which is correct: an expert combination is a
        weighted sum, and a token contributes only through the experts
        it selected.
        """
        # TODO
        raise NotImplementedError

    def load_balancing_loss(self) -> torch.Tensor:
        """The Switch-style auxiliary loss from the last forward's routing.

        Returns:
            Scalar tensor:

                L_balance = E · Σ_e  f_e · P_e

            where `f_e` is the fraction of top-k ASSIGNMENTS that went
            to expert `e` (hard counts, from `last_indices`) and `P_e`
            is the mean router probability for expert `e` (soft scores,
            from `router.last_probs`). Uniform routing gives 1.0, while
            aligned collapse approaches `E`. This surrogate is not
            bounded below by 1.0 for every finite batch because `f` and
            `P` can be imperfectly aligned.

        Raises:
            RuntimeError: if called before any forward has run.

        Recipe:
            1. probs = self.router.last_probs                # (N, E)
               indices = self.last_indices                   # (N, k)
               if probs is None or indices is None:
                   raise RuntimeError("run forward() before load_balancing_loss()")
            2. one_hot = torch.nn.functional.one_hot(
                   indices.reshape(-1), num_classes=self.num_experts
               ).to(probs.dtype)                             # (N*k, E)
               f = one_hot.mean(dim=0)                       # (E,) sums to 1
            3. P = probs.mean(dim=0)                         # (E,) sums to 1
            4. return self.num_experts * (f * P).sum()

        `f` is built from integer indices, so no gradient flows through
        it — the loss reaches the gate through `P`, which is exactly
        the Switch Transformer's trick for making a hard-count
        objective differentiable.
        """
        # TODO
        raise NotImplementedError

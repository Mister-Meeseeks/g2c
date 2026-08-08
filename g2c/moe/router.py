"""The router — the small learned gate that assigns tokens to experts.

The router is deliberately boring: one linear layer from the residual
stream to per-expert scores, a softmax, and a top-k selection. All of
the interesting behavior in a mixture-of-experts layer — specialization,
collapse, balance — emerges from how this tiny module's outputs are
used, not from its own complexity.

Two contract details carry most of the implementation:

  * **Renormalize after selection.** The softmax is over all `E`
    experts; after keeping the top `k`, the surviving weights must be
    rescaled to sum to 1. Skipping this makes the layer's output
    magnitude depend on how confident the router happened to be.
  * **Gradients flow through the weights, not the selection — for
    `k > 1`.** Top-k is not differentiable and does not need to be.
    Multiple surviving combination weights retain differentiable
    relative values. Renormalized `k == 1` is the important exception:
    its sole weight is always 1, so the task loss provides no useful
    router gradient; the auxiliary balance loss can still train it.

The router also records the full softmax distribution on
`self.last_probs` each forward — `MoEFeedForward.load_balancing_loss`
reads it to compute the mean routing probability per expert. Keep the
tensor attached to the autograd graph: the balance loss's gradient
reaches the gate through it.

Boilerplate (`__init__`, `parameters`) is implemented; `forward` is
scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.nn import Linear, Module


class Router(Module):
    """Top-k softmax gate over `num_experts` experts.

    Args:
        embedding_dim: channel dim of the residual stream (`D`).
        num_experts:   number of experts to score (`E`).
        top_k:         experts each token is routed through (`k`).
                       Must satisfy `1 <= top_k <= num_experts`.

    Attributes:
        gate: `Linear(D, E)` producing per-expert scores.
        last_probs: the `(N, E)` softmax distribution from the most
            recent forward, where `N` is the number of tokens routed.
            `None` until the first forward.
    """

    embedding_dim: int
    num_experts: int
    top_k: int
    gate: Linear

    def __init__(self, embedding_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts={num_experts}], got {top_k}"
            )
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Linear(embedding_dim, num_experts)
        self.last_probs: torch.Tensor | None = None

    def parameters(self) -> Iterable[torch.Tensor]:
        return list(self.gate.parameters())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score, select, and renormalize: which experts get each token?

        Args:
            x: tensor of shape `(N, embedding_dim)` — one row per token
                to route. (The MoE layer flattens `(B, T, D)` to
                `(B*T, D)` before routing; the router never needs to
                know about batch/sequence structure.)

        Returns:
            `(weights, indices)`, both of shape `(N, top_k)`:
              * `weights[n, j]` — the combination weight for token `n`'s
                j-th selected expert. Each row sums to 1.
              * `indices[n, j]` — which expert that weight belongs to,
                an integer in `[0, num_experts)`.

        Recipe:
            1. scores = self.gate(x)                        # (N, E)
            2. probs = torch.softmax(scores, dim=-1)        # (N, E)
               self.last_probs = probs
               (Store the FULL distribution, attached to the graph —
               the balance loss differentiates through it.)
            3. weights, indices = probs.topk(self.top_k, dim=-1)
            4. weights = weights / weights.sum(dim=-1, keepdim=True)
               (Renormalize the survivors. With top_k == num_experts
               this is a no-op; with top_k == 1 every weight is 1.0,
               so only an auxiliary loss can usefully train the gate.)
            5. return weights, indices
        """
        # TODO
        raise NotImplementedError

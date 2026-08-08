# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.moe.layer pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch


class _MoEFeedForwardImpl:  # patched onto MoEFeedForward by apply()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.embedding_dim)
        weights, indices = self.router(flat)
        self.last_indices = indices
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            selected = indices == e
            if not selected.any():
                continue
            token_idx = selected.any(dim=-1).nonzero(as_tuple=True)[0]
            w = (weights * selected.to(weights.dtype)).sum(dim=-1)
            w = w[token_idx].unsqueeze(-1)
            expert_out = expert(flat[token_idx])
            out = out.index_add(0, token_idx, expert_out * w)
        return out.reshape(x.shape)

    def load_balancing_loss(self) -> torch.Tensor:
        probs = self.router.last_probs
        indices = self.last_indices
        if probs is None or indices is None:
            raise RuntimeError("run forward() before load_balancing_loss()")
        one_hot = torch.nn.functional.one_hot(
            indices.reshape(-1), num_classes=self.num_experts
        ).to(probs.dtype)
        f = one_hot.mean(dim=0)
        P = probs.mean(dim=0)
        return self.num_experts * (f * P).sum()

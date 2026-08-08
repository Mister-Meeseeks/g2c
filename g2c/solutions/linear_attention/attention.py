# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.linear_attention.attention pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.linear_attention.attention import EPS, feature_map


class _LinearAttentionImpl:  # patched onto LinearAttention by apply()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        q = feature_map(self._split_heads(self.q_proj(x)))
        k = feature_map(self._split_heads(self.k_proj(x)))
        v = self._split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1)
        idx = torch.arange(T, device=x.device)
        delta = (idx[:, None] - idx[None, :]).to(x.dtype)
        causal = delta >= 0
        gamma = self.decay.view(self.num_heads, 1, 1)
        decay_mask = torch.where(
            causal,
            gamma ** delta.clamp(min=0),
            torch.zeros((), dtype=x.dtype, device=x.device),
        )
        weighted = scores * decay_mask
        out = weighted @ v / (weighted.sum(-1, keepdim=True) + EPS)
        return self.out_proj(self._merge_heads(out))

    def step(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, D = x_t.shape
        H, d = self.num_heads, self.head_dim
        if state is None:
            state = self.init_state(B, x_t.device)
        S, z = state

        q = feature_map(self.q_proj(x_t).reshape(B, H, d))
        k = feature_map(self.k_proj(x_t).reshape(B, H, d))
        v = self.v_proj(x_t).reshape(B, H, d)

        gamma = self.decay.view(1, H, 1)
        S = gamma.unsqueeze(-1) * S + k.unsqueeze(-1) @ v.unsqueeze(-2)
        z = gamma * z + k

        numer = (q.unsqueeze(-2) @ S).squeeze(-2)
        denom = (q * z).sum(-1, keepdim=True) + EPS
        out = numer / denom
        return self.out_proj(out.reshape(B, D)), (S, z)

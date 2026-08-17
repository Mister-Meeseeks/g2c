# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.specdec.mtp pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch


class _MTPHeadImpl:  # patched onto MTPHead by apply()
    def forward(
        self, hidden: torch.Tensor, next_token_ids: torch.Tensor
    ) -> torch.Tensor:
        e = self.base.token_embed(next_token_ids)
        h = self.combine(torch.cat([hidden, e], dim=-1))
        h = self.block(h)
        h = self.ln(h)
        return h @ self.base.token_embed.weight.T + self.head_bias


def mtp_loss(mtp_logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logits = mtp_logits[:, :-1, :]
    targets = token_ids[:, 2:]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )

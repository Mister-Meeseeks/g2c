# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.multimodal.model pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch


class _MultimodalLMImpl:  # patched onto MultimodalLM by apply()
    def forward(
        self, token_ids: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(1)
        B, N, H, W = images.shape

        patches = self.patch_embed(images.reshape(B * N, H, W))
        patches = patches.reshape(B, -1, self.lm.embedding_dim)

        tok = self.lm.token_embed(token_ids)
        mask = token_ids == self.patch_token_id
        counts = mask.sum(dim=1)
        expected = patches.shape[1]
        if not torch.all(counts == expected):
            counts_list = counts.detach().cpu().tolist()
            raise ValueError(
                f"token_ids contain per-row patch-placeholder counts "
                f"{counts_list}, but each example's images supply "
                f"{expected} patch vectors"
            )

        tok = tok.clone()
        tok[mask] = patches.reshape(-1, self.lm.embedding_dim)

        _, T = token_ids.shape
        x = tok + self.lm.pos_embed(T)
        for block in self.lm.blocks:
            x = block(x)
        x = self.lm.ln_final(x)
        return x @ self.lm.token_embed.weight.T + self.lm.head_bias

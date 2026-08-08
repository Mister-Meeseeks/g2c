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
        mask = token_ids == self.image_token_id
        expected = patches.shape[0] * patches.shape[1]
        if int(mask.sum()) != expected:
            raise ValueError(
                f"token_ids contain {int(mask.sum())} image placeholders "
                f"but the images supply {expected} patch vectors"
            )

        tok = tok.clone()
        tok[mask] = patches.reshape(-1, self.lm.embedding_dim)

        _, T = token_ids.shape
        x = tok + self.lm.pos_embed(T)
        for block in self.lm.blocks:
            x = block(x)
        x = self.lm.ln_final(x)
        return x @ self.lm.token_embed.weight.T + self.lm.head_bias

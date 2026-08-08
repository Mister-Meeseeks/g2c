# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
from __future__ import annotations

import torch

from g2c.pretraining.data import get_lm_batch


class _TokenMixtureImpl:  # patched onto TokenMixture by apply()
    def get_lm_batch(
        self,
        batch_size: int,
        context_length: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        assignments = torch.multinomial(
            self.weights,
            batch_size,
            replacement=True,
            generator=generator,
        )
        x_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        for source_index, name in enumerate(self.names):
            count = int((assignments == source_index).sum())
            if count == 0:
                continue
            x, y = get_lm_batch(
                self.sources[name],
                count,
                context_length,
                generator=generator,
            )
            x_parts.append(x)
            y_parts.append(y)
            self.example_counts[name] += count

        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        order = torch.randperm(batch_size, generator=generator)
        return x[order], y[order]

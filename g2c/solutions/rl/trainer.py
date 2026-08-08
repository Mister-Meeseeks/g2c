# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.rl.trainer pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.rl.grpo import completion_log_prob, group_advantages, grpo_loss
from g2c.rl.sample import sample_group


class _GRPOTrainerImpl:  # patched onto GRPOTrainer by apply()
    def train_step(self) -> dict[str, float]:
        task = self._next_task()
        group = sample_group(
            self.model,
            self.tokenizer,
            task["prompt"],
            self.group_size,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            eos_id=self.eos_id,
            generator=self._generator,
        )
        rewards = torch.tensor(
            [self.verifier(task, text) for text in group.texts]
        )

        def _skip() -> dict[str, float]:
            return {
                "loss": 0.0,
                "mean_reward": float(rewards.mean()),
                "kl": float("nan"),
                "sample_entropy": float("nan"),
                "skipped": 1.0,
            }

        if any(c.numel() == 0 for c in group.completions):
            return _skip()
        advantages = group_advantages(rewards)
        if not advantages.any():
            return _skip()

        prompt_len = group.prompt_ids.numel()
        device = getattr(self.model, "device", torch.device("cpu"))
        rows = [
            torch.cat([group.prompt_ids, c]).unsqueeze(0).to(device)
            for c in group.completions
        ]
        logp = torch.cat(
            [completion_log_prob(self.model, row, prompt_len) for row in rows]
        )
        with torch.no_grad():
            ref_logp = torch.cat(
                [
                    completion_log_prob(self.ref_model, row, prompt_len)
                    for row in rows
                ]
            )

        ref_logp = ref_logp.to(logp.device)
        advantages = advantages.to(logp.device)
        d = ref_logp - logp.detach()
        kl = (d.exp() - d - 1.0).mean()
        lengths = torch.tensor(
            [c.numel() for c in group.completions], device=logp.device
        )
        sample_entropy = (-logp.detach() / lengths).mean()
        loss = grpo_loss(logp, ref_logp, advantages, self.kl_coef)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        return {
            "loss": float(loss.detach()),
            "mean_reward": float(rewards.mean()),
            "kl": float(kl),
            "sample_entropy": float(sample_entropy),
            "skipped": 0.0,
        }

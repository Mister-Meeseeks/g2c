"""GRPOTrainer — the sample → verify → advantage → update loop.

The driver is provided (`train`, `evaluate`, task cycling); the
per-step orchestration (`train_step`) is the scaffold, because wiring
the four pieces together in the right order — fresh samples, group
rewards, advantages, masked log-probs, the leash — IS the lesson.

Compared to `SFTTrainer` (Module 13), one structural difference
matters: there is no fixed dataset of target completions. Each step
manufactures its rollout batch by sampling from the current model.
Generation therefore dominates the wall-clock, and stale samples are
not an optimization—they break this on-policy estimator. Sample fresh,
every step.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch

from g2c.training import AdamW

from .grpo import (  # noqa: F401 (used by the train_step scaffold)
    completion_log_prob,
    group_advantages,
    grpo_loss,
)
from .sample import sample_group  # noqa: F401 (used by the train_step scaffold)
from .verifiers import Task


class GRPOTrainer:
    """Group-relative policy optimization over verifiable tasks.

    Args:
        model: the policy being trained (`TransformerLM` or a BaseLM
            adapter — anything `generate` and `completion_log_prob`
            accept).
        ref_model: the FROZEN reference for the KL leash — typically a
            fresh copy of the pre-RL checkpoint. Never updated; this
            trainer never touches its parameters.
        tokenizer: duck-typed `encode`/`decode`.
        tasks: the task pool. Each task is a dict with at least
            `"prompt"`; the verifier decides what else it needs.
        verifier: `(task, completion_text) -> float` reward function.
        group_size: K completions per prompt per step.
        lr: AdamW learning rate. Start at roughly a third of your SFT
            rate — RL compounds differently (see the lesson page).
        kl_coef: leash strength β.
        max_new_tokens, temperature: sampling controls per step. This
            simplified on-policy trainer requires `temperature == 1.0`
            because it rescores the model's untempered probabilities.
        eos_id: forwarded to the sampler.
        seed: seeds task order and sampling.
    """

    def __init__(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        tasks: list[Task],
        verifier: Callable[[Task, str], float],
        *,
        group_size: int = 8,
        lr: float = 1e-5,
        kl_coef: float = 0.1,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        eos_id: int | None = None,
        seed: int = 0,
    ) -> None:
        if not tasks:
            raise ValueError("tasks must be non-empty")
        if temperature != 1.0:
            raise ValueError(
                "this simplified on-policy trainer requires temperature=1.0 "
                "so rollout and rescoring distributions match"
            )
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.tasks = list(tasks)
        self.verifier = verifier
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eos_id = eos_id
        self.optimizer = AdamW(model.parameters(), lr)
        self._rng = random.Random(seed)
        self._generator = torch.Generator().manual_seed(seed)
        self.step_count = 0

    def _next_task(self) -> Task:
        """Uniformly sample the next task from the pool."""
        return self._rng.choice(self.tasks)

    def train_step(self) -> dict[str, float]:
        """One GRPO rollout step: sample, verify, and update if informative.

        Returns:
            Metrics dict with keys `"loss"`, `"mean_reward"`, `"kl"`,
            `"sample_entropy"`, and `"skipped"`. `sample_entropy` is
            the sampled-token estimate `-log p(completion) / n_tokens`,
            averaged over the group — a useful collapse diagnostic, not
            an exact vocabulary-wide entropy calculation. `skipped` is
            1.0 when the group was degenerate and no update ran, else
            0.0. A degenerate group — all rewards equal — is NOT an
            error; on an easy task it happens often, and on a too-easy
            task it approaches 100% and learning stops for lack of
            signal. That is why the metric exists.

        Recipe:
            1. task = self._next_task()
               group = sample_group(
                   self.model, self.tokenizer, task["prompt"],
                   self.group_size,
                   max_new_tokens=self.max_new_tokens,
                   temperature=self.temperature,
                   eos_id=self.eos_id,
                   generator=self._generator,
               )
            2. rewards = torch.tensor(
                   [self.verifier(task, t) for t in group.texts]
               )
            3. advantages = group_advantages(rewards)
               if not advantages.any():
                   return {"loss": 0.0,
                           "mean_reward": float(rewards.mean()),
                           "kl": float("nan"),
                           "sample_entropy": float("nan"),
                           "skipped": 1.0}
            4. # Log-probs, one completion at a time (legible; a
               # padded batch is the performance version). Move each
               # row to the model's device — sampling returned CPU
               # ids, and BaseLM may be sitting on MPS. Likewise move
               # `advantages` and `ref_logp` to `logp.device` before
               # the loss.
               prompt_len = group.prompt_ids.numel()
               device = getattr(self.model, "device", torch.device("cpu"))
               rows = [torch.cat([group.prompt_ids, c]).unsqueeze(0).to(device)
                       for c in group.completions]
               logp = torch.cat([
                   completion_log_prob(self.model, row, prompt_len)
                   for row in rows
               ])
               with torch.no_grad():
                   ref_logp = torch.cat([
                       completion_log_prob(self.ref_model, row, prompt_len)
                       for row in rows
                   ])
            5. ref_logp = ref_logp.to(logp.device)
               advantages = advantages.to(logp.device)
               d = ref_logp - logp.detach()
               kl = (d.exp() - d - 1.0).mean()
               lengths = torch.tensor(
                   [c.numel() for c in group.completions],
                   device=logp.device,
               )
               sample_entropy = (-logp.detach() / lengths).mean()
               loss = grpo_loss(logp, ref_logp, advantages, self.kl_coef)
               self.optimizer.zero_grad()
               loss.backward()
               self.optimizer.step()
               self.step_count += 1
            6. return {"loss": float(loss.detach()),
                       "mean_reward": float(rewards.mean()),
                       "kl": float(kl),
                       "sample_entropy": float(sample_entropy),
                       "skipped": 0.0}

        `kl` and `sample_entropy` are `NaN` on skipped groups because
        the trainer deliberately avoids the extra policy/reference
        forward passes when no update can occur. Plotting libraries
        render those entries as gaps instead of false zero readings.

        Empty completions: `torch.cat([prompt, c])` with an empty `c`
        makes `completion_log_prob` raise (prompt_len == T). Guard by
        skipping empty completions — or, simpler and correct, treat a
        group containing any empty completion as degenerate for this
        step and return the skip metrics; the tests accept either.
        Non-empty completions are the overwhelmingly common case at
        stochastic sampling unless `eos_id` fires immediately.
        """
        # TODO
        raise NotImplementedError

    def train(
        self, max_steps: int, *, log_every: int = 10
    ) -> dict[str, list[float]]:
        """Run `max_steps` rollout attempts, collecting metric histories.

        Degenerate groups count as attempts but do not run optimizer updates;
        `step_count` records the number of updates that actually occurred.
        """
        history: dict[str, list[float]] = {
            "loss": [],
            "mean_reward": [],
            "kl": [],
            "sample_entropy": [],
            "skipped": [],
        }
        for step in range(max_steps):
            metrics = self.train_step()
            for key in history:
                history[key].append(metrics.get(key, 0.0))
            if log_every and (step + 1) % log_every == 0:
                window = history["mean_reward"][-log_every:]
                print(
                    f"step {step + 1:>5}  "
                    f"reward(last {log_every}) = "
                    f"{sum(window) / len(window):.3f}"
                )
        return history

    @torch.no_grad()
    def evaluate(
        self,
        tasks: list[Task] | None = None,
        *,
        verifier: Callable[[Task, str], float] | None = None,
    ) -> float:
        """Mean reward over `tasks` (default: the training pool), greedy.

        Greedy decoding on purpose: evaluation asks "what does the
        model actually do," not "what can it luck into at temperature
        1." Compare against the pre-RL number from the same call on
        the reference model. `verifier` defaults to the trainer's
        training verifier; passing another scorer is useful for
        auditing a deliberately flawed reward against the intended one.
        """
        from g2c.sampling import generate

        pool = self.tasks if tasks is None else tasks
        scorer = self.verifier if verifier is None else verifier
        total = 0.0
        for task in pool:
            prompt_ids = torch.tensor(
                self.tokenizer.encode(task["prompt"]), dtype=torch.long
            )
            full = generate(
                self.model,
                prompt_ids,
                self.max_new_tokens,
                temperature=0.0,
                eos_id=self.eos_id,
            )
            text = self.tokenizer.decode(full[prompt_ids.numel() :].tolist())
            total += scorer(task, text)
        return total / len(pool)

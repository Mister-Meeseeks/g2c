"""The DPO training loop — parallel to `g2c.sft.SFTTrainer`.

`DPOTrainer` mirrors the Module 13 `SFTTrainer` shape almost exactly:

    __init__   — stores config, builds the optimizer, init step counter
    lr         — cosine_with_warmup schedule lookup
    train_step — one step of forward (twice, on policy AND ref) /
                 loss / backward / clip / step
    evaluate   — average loss and accuracy over a held-out set, no grad
    train      — calls train_step in a loop; logs metrics

The two operational differences from SFT:

  1. The trainer holds TWO models: the trainable `policy` (called
     `model`, for parallelism with SFTTrainer) and a FROZEN reference
     `ref_model`. The reference is forwarded under `torch.no_grad()`
     and its parameters never update. Memory is ~2× SFT at the same
     `(B, T)`.

  2. Each step does FOUR forwards (or two if you batch chosen and
     rejected together — we don't, for clarity):
        - policy(chosen)
        - policy(rejected)
        - ref(chosen)     [no_grad]
        - ref(rejected)   [no_grad]
     Then a single `dpo_loss` call collapses the four `(B,)`
     log-prob tensors into the scalar loss.

`__init__`, `lr`, `evaluate`, and `train` are implemented for you.
`train_step` is scaffolded — same recipe as the SFT `train_step`, but
with the DPO data path (`pad_and_collate_pref`) and DPO loss
(`sequence_logprob` + `dpo_loss`). The order-of-operations lesson
from Modules 10/13 is the same one; what's new is the two-model
forward bookkeeping.

Memory note: the reference model is held in fp32 like the policy, so
DPO uses ~2× the parameter memory of SFT. At our scale (1–20M
parameters) this is fine even on M1 8GB. At production scale (7B+)
people use one of:
  - keeping ref weights in fp16 / bfloat16 with mixed-precision
    forwards;
  - dropping ref entirely and using a "self-supervised" reference
    (e.g. setting `ref_logp = 0` and ignoring the KL term);
  - fully sharing the ref and policy weights and just snapshotting
    `pi_ref` periodically (KTO-style).
We use the simplest, fattest approach: a separate frozen full-precision
reference. It's the right unit of work for one week.
"""
from __future__ import annotations

import torch

from g2c.nn import Module, resolve_device
from g2c.training import (  # noqa: F401 (for the student implementation)
    AdamW,
    clip_grad_norm_,
    cosine_with_warmup,
)

from .data import PreferenceExample, pad_and_collate_pref
from .loss import dpo_loss, sequence_logprob


class DPOTrainer:
    """Direct preference optimization for a transformer language model.

    Args:
        model: a Module whose forward returns `(B, T, V)` logits — the
            POLICY being trained. Typically initialized from the
            Module 13 SFT'd checkpoint. The trainer requires
            `model.parameters()` and `model(x)`.
        ref_model: a SECOND Module of the same architecture, initialized
            from the SAME checkpoint as `model`, treated as frozen
            throughout. The trainer never calls `.zero_grad()` or
            `.step()` on it — only forwards under `torch.no_grad()`.
            (We don't enforce the freeze in code; passing a non-frozen
            model would silently waste compute on its grad bookkeeping
            but not corrupt training.)
        examples: list of `PreferenceExample` — the full preference
            dataset. Each step samples `batch_size` examples uniformly
            at random.
        max_seq_len: padded length per (prompt + completion). Examples
            whose `prompt + chosen` (or `prompt + rejected`) is longer
            than this are head-truncated; shorter ones are padded.
            Must be ≤ `model.max_seq_len`.
        pad_id: token id used for padding. ID 0 (the byte 0x00) is the
            standard choice with our BPE tokenizer.
        beta: DPO KL-regularization strength. The DPO paper suggests
            0.1–0.5; at toy scale 0.1 is a fine default. Larger pins
            the policy nearer the reference; smaller lets the
            preference signal pull the policy further.
        batch_size: number of preference pairs per step.
        max_steps: total training steps (the cosine schedule's horizon).
        max_lr: peak learning rate at the end of warmup. Default
            recommendation for DPO: 1e-5 to 1e-4 — about 10× lower
            than SFT, which is itself 10× lower than pretraining.
            DPO's gradient signal is sequence-level (one log-ratio
            per example), so the effective magnitude per step is
            larger; smaller lr compensates.
        min_lr: floor learning rate at end of cosine decay.
        warmup_steps: linear-warmup steps. 10–50 is typical for DPO
            at our scale.
        weight_decay: decoupled weight decay (forwarded to AdamW).
        grad_clip: optional global gradient-norm threshold. `None`
            disables clipping. `1.0` is a fine default.
        eval_every: run a validation pass every N steps.
        eval_iters: how many random batches to average for validation.
        log_every: append metrics to history every N steps.
        generator: optional `torch.Generator` for example-shuffling
            reproducibility.
        device: `"auto"` moves the policy, reference model, and collated
            batches to MPS when available, otherwise CPU. Pass `"cpu"` to
            force a CPU run.

    Attributes:
        optimizer: the inner `AdamW` instance, attached to `model`'s
            parameters only — never to `ref_model`'s.
        step: current step counter (0-indexed).
    """

    model: Module
    ref_model: Module
    examples: list[PreferenceExample]
    max_seq_len: int
    pad_id: int
    beta: float
    batch_size: int
    max_steps: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float | None
    eval_every: int
    eval_iters: int
    log_every: int
    generator: torch.Generator | None
    device: torch.device
    optimizer: AdamW
    step: int

    def __init__(
        self,
        model: Module,
        *,
        ref_model: Module,
        examples: list[PreferenceExample],
        max_seq_len: int,
        pad_id: int,
        beta: float,
        batch_size: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        grad_clip: float | None = None,
        eval_every: int = 100,
        eval_iters: int = 20,
        log_every: int = 10,
        generator: torch.Generator | None = None,
        device: str | torch.device | None = "auto",
    ) -> None:
        if len(examples) == 0:
            raise ValueError("DPOTrainer requires at least one example.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if max_seq_len <= 1:
            raise ValueError(
                f"max_seq_len must be > 1 to allow shift-by-one, "
                f"got {max_seq_len}."
            )
        if beta <= 0:
            raise ValueError(
                f"beta must be positive, got {beta}. (beta=0 makes the "
                f"DPO loss constant log(2) with zero gradient.)"
            )
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.ref_model = ref_model.to(self.device)
        self.examples = examples
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.beta = beta
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.eval_every = eval_every
        self.eval_iters = eval_iters
        self.log_every = log_every
        self.generator = generator
        # Optimizer is attached to the policy ONLY. The reference is
        # never updated.
        self.optimizer = AdamW(
            self.model.parameters(), lr=max_lr, weight_decay=weight_decay
        )
        self.step = 0

    def lr(self, step: int | None = None) -> float:
        """Return the lr the schedule prescribes at `step`.

        If `step` is None, uses `self.step`. Same shape as Modules
        10/13.
        """
        if step is None:
            step = self.step
        return cosine_with_warmup(
            step,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
        )

    def _sample_batch(
        self, examples: list[PreferenceExample]
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Sample `batch_size` examples uniformly and run pad_and_collate_pref.

        Implemented for you. Used by both `train_step` and `evaluate`.
        """
        n = len(examples)
        indices = torch.randint(
            0, n, (self.batch_size,), generator=self.generator
        ).tolist()
        batch = [examples[i] for i in indices]
        return pad_and_collate_pref(
            batch, max_seq_len=self.max_seq_len, pad_id=self.pad_id
        )

    def _logp_under(
        self,
        m: Module,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run model `m` on `x` and return per-example response log-prob.

        Implemented for you. Used by `train_step` for both policy and
        reference. The caller is responsible for choosing the
        gradient context (`torch.no_grad()` for the reference).
        """
        logits = m(x)                                       # (B, T-1, V)
        return sequence_logprob(logits, y, mask)            # (B,)

    def train_step(self) -> dict[str, float]:
        """Perform one DPO step on a fresh random batch.

        Returns:
            A dict of metrics for this step:
                'loss':            scalar loss (Python float)
                'lr':              lr applied this step (Python float)
                'grad_norm':       pre-clip global grad norm if
                                   grad_clip is set; else 0.0
                'chosen_reward':   mean implicit reward on chosen
                'rejected_reward': mean implicit reward on rejected
                'reward_margin':   mean(chosen - rejected) reward
                'accuracy':        fraction with chosen > rejected

        Recipe — same shape as Module 13's `SFTTrainer.train_step`,
        but with two-model bookkeeping:

            1. # Sample a fresh batch.
               (cx, cy, cm, rx, ry, rm) = self._sample_batch(self.examples)
               cx = cx.to(self.device)
               cy = cy.to(self.device)
               cm = cm.to(self.device)
               rx = rx.to(self.device)
               ry = ry.to(self.device)
               rm = rm.to(self.device)

            2. # LR for THIS step (before incrementing the counter).
               lr = self.lr()
               self.optimizer.lr = lr

            3. self.optimizer.zero_grad()

            4. # Policy forwards — WITH gradient.
               policy_chosen_logp   = self._logp_under(self.model, cx, cy, cm)
               policy_rejected_logp = self._logp_under(self.model, rx, ry, rm)

            5. # Reference forwards — NO gradient.
               with torch.no_grad():
                   ref_chosen_logp   = self._logp_under(self.ref_model, cx, cy, cm)
                   ref_rejected_logp = self._logp_under(self.ref_model, rx, ry, rm)

            6. loss, metrics = dpo_loss(
                   policy_chosen_logp,
                   policy_rejected_logp,
                   ref_chosen_logp,
                   ref_rejected_logp,
                   beta=self.beta,
               )

            7. loss.backward()

            8. # Optional global gradient clipping.
               if self.grad_clip is not None:
                   grad_norm = clip_grad_norm_(
                       self.model.parameters(), self.grad_clip
                   )
               else:
                   grad_norm = 0.0

            9. self.optimizer.step()

           10. self.step += 1

           11. return {
                   'loss':            loss.item(),
                   'lr':              lr,
                   'grad_norm':       grad_norm,
                   'chosen_reward':   metrics['chosen_reward'].item(),
                   'rejected_reward': metrics['rejected_reward'].item(),
                   'reward_margin':   metrics['reward_margin'].item(),
                   'accuracy':        metrics['accuracy'].item(),
               }

        Common reorderings to avoid (same as Modules 10/13):

          * Computing the reference forwards WITH gradient. The
            optimizer would not step them, because
            `optimizer.params` doesn't include them — so it's a
            silent waste of memory, not a correctness bug. Still:
            wrap the ref forwards in `torch.no_grad()` so the autograd
            graph doesn't grow through them.

          * Forgetting `optimizer.zero_grad()`. The .grad on
            `model.parameters()` accumulates across steps without it.

          * Clipping after `optimizer.step()`. The gradients have
            already been consumed; clipping after is a no-op.
        """
        # TODO
        raise NotImplementedError

    def evaluate(
        self, eval_examples: list[PreferenceExample]
    ) -> dict[str, float]:
        """Average DPO loss and metrics over `eval_iters` random batches.

        Implemented for you. Wrapped in `torch.no_grad()`.

        Returns:
            A dict with keys 'loss', 'chosen_reward', 'rejected_reward',
            'reward_margin', 'accuracy' — each averaged across the
            sampled eval batches.
        """
        if len(eval_examples) == 0:
            raise ValueError("evaluate() requires at least one example.")
        agg: dict[str, list[float]] = {
            "loss": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_margin": [],
            "accuracy": [],
        }
        with torch.no_grad():
            for _ in range(self.eval_iters):
                cx, cy, cm, rx, ry, rm = self._sample_batch(eval_examples)
                cx = cx.to(self.device)
                cy = cy.to(self.device)
                cm = cm.to(self.device)
                rx = rx.to(self.device)
                ry = ry.to(self.device)
                rm = rm.to(self.device)
                policy_c = self._logp_under(self.model, cx, cy, cm)
                policy_r = self._logp_under(self.model, rx, ry, rm)
                ref_c = self._logp_under(self.ref_model, cx, cy, cm)
                ref_r = self._logp_under(self.ref_model, rx, ry, rm)
                loss, metrics = dpo_loss(
                    policy_c, policy_r, ref_c, ref_r, beta=self.beta
                )
                agg["loss"].append(loss.item())
                agg["chosen_reward"].append(metrics["chosen_reward"].item())
                agg["rejected_reward"].append(metrics["rejected_reward"].item())
                agg["reward_margin"].append(metrics["reward_margin"].item())
                agg["accuracy"].append(metrics["accuracy"].item())
        return {k: sum(v) / len(v) for k, v in agg.items()}

    def train(
        self,
        eval_examples: list[PreferenceExample] | None = None,
    ) -> dict[str, list]:
        """Run the DPO training loop for `max_steps` steps.

        Implemented for you. Calls `train_step` once per step, records
        metrics every `log_every` steps, and runs evaluation every
        `eval_every` steps if `eval_examples` is given.

        Args:
            eval_examples: optional held-out preference set for
                periodic validation.

        Returns:
            A dict with logging keys appended every `log_every` steps:
                'step', 'train_loss', 'lr', 'grad_norm',
                'chosen_reward', 'rejected_reward',
                'reward_margin', 'accuracy'.
            And eval keys appended every `eval_every` steps (empty if
            no `eval_examples`):
                'val_step', 'val_loss', 'val_accuracy',
                'val_reward_margin'.
        """
        history: dict[str, list] = {
            "step": [],
            "train_loss": [],
            "lr": [],
            "grad_norm": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_margin": [],
            "accuracy": [],
            "val_step": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_reward_margin": [],
        }
        for _ in range(self.max_steps):
            metrics = self.train_step()
            done = self.step == self.max_steps
            if (self.step - 1) % self.log_every == 0 or done:
                history["step"].append(self.step - 1)
                history["train_loss"].append(metrics["loss"])
                history["lr"].append(metrics["lr"])
                history["grad_norm"].append(metrics["grad_norm"])
                history["chosen_reward"].append(metrics["chosen_reward"])
                history["rejected_reward"].append(metrics["rejected_reward"])
                history["reward_margin"].append(metrics["reward_margin"])
                history["accuracy"].append(metrics["accuracy"])
            if eval_examples is not None and (
                (self.step - 1) % self.eval_every == 0 or done
            ):
                val = self.evaluate(eval_examples)
                history["val_step"].append(self.step - 1)
                history["val_loss"].append(val["loss"])
                history["val_accuracy"].append(val["accuracy"])
                history["val_reward_margin"].append(val["reward_margin"])
        return history

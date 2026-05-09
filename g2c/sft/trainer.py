"""The SFT training loop — parallel to `g2c.pretraining.Trainer`.

`SFTTrainer` mirrors the Module 10 `Trainer` shape almost exactly:

    __init__   — stores config, builds the optimizer, init step counter
    lr         — cosine_with_warmup schedule lookup
    train_step — one step of forward / loss / backward / clip / step
    evaluate   — average loss over a held-out set, no grad
    train      — calls train_step in a loop; logs metrics

The only operational differences from pretraining:

  * The data source is a `list[SFTExample]`, not a 1-D `ids` tensor.
  * Each step samples `batch_size` examples from the list, runs them
    through `pad_and_collate`, and uses `masked_cross_entropy` instead
    of `lm_cross_entropy`.

Because the SFT step count is small (hundreds, not thousands) and the
dataset is small (50–500 examples), this trainer runs in minutes on
even an M1 base machine. Don't try to optimize it; the loop is
identical to pretraining at this scale.

`__init__`, `lr`, `evaluate`, and `train` are implemented for you.
`train_step` is scaffolded — same recipe as Module 10's
`Trainer.train_step`, but with the SFT data path and masked loss. The
order-of-operations lesson is the same one you internalized in Module
10; the point of re-scaffolding it here is to make the SFT-specific
substitutions (data source, loss function) explicit and to give the
test suite a single place to pin the SFT loss-decrease behavior.
"""
from __future__ import annotations

import torch

from g2c.nn import Module, resolve_device
from g2c.training import AdamW, clip_grad_norm_, cosine_with_warmup

from .data import SFTExample, pad_and_collate
from .loss import masked_cross_entropy


class SFTTrainer:
    """Supervised fine-tuning loop for a transformer language model.

    Args:
        model: a Module whose forward returns `(B, T, V)` logits — i.e.
            `TransformerLM`. The trainer only requires
            `model.parameters()` and `model(x)`. The model should have
            been pretrained (Module 10) before being passed here;
            SFT-from-scratch produces a model that has only seen the
            SFT data and is functionally useless outside of it.
        examples: list of `SFTExample` — the full SFT dataset. Each
            step samples `batch_size` examples uniformly at random.
        max_seq_len: padded sequence length per example. Examples
            longer than this are truncated; shorter ones are padded.
            Should match (or be ≤) `model.max_seq_len`.
        pad_id: token id used for padding. ID 0 (the byte 0x00) is
            the standard choice with our BPE tokenizer; it never
            occurs in real text.
        batch_size: number of examples per step.
        max_steps: total training steps (the cosine schedule's
            horizon).
        max_lr: peak learning rate at the end of warmup. Default
            recommendation for SFT: `3e-4` — about 10× lower than
            pretraining's `3e-3`.
        min_lr: floor learning rate at end of cosine decay.
        warmup_steps: linear-warmup steps. Often 10–50 for SFT.
        weight_decay: decoupled weight decay (forwarded to AdamW).
        grad_clip: optional global gradient-norm threshold. `None`
            disables clipping. `1.0` is a fine default.
        eval_every: run a validation pass every N steps.
        eval_iters: how many random batches to average for validation.
            (Only relevant when an eval set is provided to `train`.)
        log_every: append metrics to history every N steps.
        generator: optional `torch.Generator` for example-shuffling
            reproducibility.
        device: `"auto"` moves the model and collated batches to MPS
            when available, otherwise CPU. Pass `"cpu"` to force a CPU
            run.

    Attributes:
        optimizer: the inner `AdamW` instance. The trainer mutates
            `optimizer.lr` once per step from the cosine schedule.
        step: current step counter (0-indexed). Advanced by 1 at the
            end of each `train_step`.
    """

    model: Module
    examples: list[SFTExample]
    max_seq_len: int
    pad_id: int
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
        examples: list[SFTExample],
        max_seq_len: int,
        pad_id: int,
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
            raise ValueError("SFTTrainer requires at least one example.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if max_seq_len <= 1:
            raise ValueError(
                f"max_seq_len must be > 1 to allow shift-by-one, "
                f"got {max_seq_len}."
            )
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.examples = examples
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
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
        self.optimizer = AdamW(
            self.model.parameters(), lr=max_lr, weight_decay=weight_decay
        )
        self.step = 0

    def lr(self, step: int | None = None) -> float:
        """Return the lr the schedule prescribes at `step`.

        If `step` is None, uses `self.step`. Same shape as Module 10's
        `Trainer.lr`.
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
        self, examples: list[SFTExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample `batch_size` examples uniformly and run pad_and_collate.

        Implemented for you. Used by both `train_step` and `evaluate`.
        """
        n = len(examples)
        indices = torch.randint(
            0, n, (self.batch_size,), generator=self.generator
        ).tolist()
        batch = [examples[i] for i in indices]
        return pad_and_collate(
            batch, max_seq_len=self.max_seq_len, pad_id=self.pad_id
        )

    def train_step(self) -> dict[str, float]:
        """Perform one SFT step on a fresh random batch.

        Returns:
            A dict of metrics for this step:
                'loss':       scalar loss (Python float)
                'lr':         lr applied this step (Python float)
                'grad_norm':  pre-clip global grad norm if grad_clip
                              is set; else 0.0

        Recipe — same shape as Module 10's `Trainer.train_step`, with
        the SFT data path and masked loss substituted in:

            1. # Sample a fresh batch from self.examples.
               x, y, loss_mask = self._sample_batch(self.examples)
               x = x.to(self.device)
               y = y.to(self.device)
               loss_mask = loss_mask.to(self.device)

            2. # LR for THIS step (before incrementing the counter).
               lr = self.lr()
               self.optimizer.lr = lr

            3. self.optimizer.zero_grad()

            4. logits = self.model(x)                 # (B, T-1, V)

            5. loss = masked_cross_entropy(
                   logits, y, loss_mask
               )                                       # scalar

            6. loss.backward()

            7. # Optional global gradient clipping.
               if self.grad_clip is not None:
                   grad_norm = clip_grad_norm_(
                       self.model.parameters(), self.grad_clip
                   )
               else:
                   grad_norm = 0.0

            8. self.optimizer.step()

            9. self.step += 1

           10. return {'loss': loss.item(), 'lr': lr,
                       'grad_norm': grad_norm}

        Common reorderings to avoid (same as Module 10):

          * Stepping the counter before applying the optimizer (lr
            schedule shifts by one).
          * Clipping after `optimizer.step()` (gradients already
            consumed; clipping is a no-op).
          * Forgetting `zero_grad` (PyTorch accumulates into .grad).
        """
        x, y, loss_mask = self._sample_batch(self.examples)
        x = x.to(self.device)
        y = y.to(self.device)
        loss_mask = loss_mask.to(self.device)

        lr = self.lr()
        self.optimizer.lr = lr

        self.optimizer.zero_grad()

        logits = self.model(x)

        loss = masked_cross_entropy(logits, y, loss_mask)

        loss.backward()

        if self.grad_clip is not None:
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip)
        else:
            grad_norm = 0.0

        self.optimizer.step()

        self.step += 1

        return {"loss": loss.item(), "lr": lr, "grad_norm": grad_norm}
    

    def evaluate(self, eval_examples: list[SFTExample]) -> float:
        """Average masked CE over `eval_iters` random batches.

        Implemented for you. Wrapped in `torch.no_grad()`.

        Returns:
            Mean masked cross-entropy as a Python float.
        """
        if len(eval_examples) == 0:
            raise ValueError("evaluate() requires at least one example.")
        losses: list[float] = []
        with torch.no_grad():
            for _ in range(self.eval_iters):
                x, y, loss_mask = self._sample_batch(eval_examples)
                x = x.to(self.device)
                y = y.to(self.device)
                loss_mask = loss_mask.to(self.device)
                logits = self.model(x)
                losses.append(
                    masked_cross_entropy(logits, y, loss_mask).item()
                )
        return sum(losses) / len(losses)

    def train(
        self,
        eval_examples: list[SFTExample] | None = None,
    ) -> dict[str, list]:
        """Run the SFT training loop for `max_steps` steps.

        Implemented for you. Calls `train_step` once per step, records
        metrics every `log_every` steps, and runs evaluation every
        `eval_every` steps if `eval_examples` is given.

        Args:
            eval_examples: optional held-out set for periodic
                validation. Pass `None` to skip evaluation entirely.

        Returns:
            A dict with six lists, indexed by logging checkpoints:
                'step', 'train_loss', 'lr', 'grad_norm':
                    appended every `log_every` steps.
                'val_step', 'val_loss':
                    appended every `eval_every` steps (empty if no
                    `eval_examples`).
        """
        history: dict[str, list] = {
            "step": [],
            "train_loss": [],
            "lr": [],
            "grad_norm": [],
            "val_step": [],
            "val_loss": [],
        }
        for _ in range(self.max_steps):
            metrics = self.train_step()
            done = self.step == self.max_steps
            if (self.step - 1) % self.log_every == 0 or done:
                history["step"].append(self.step - 1)
                history["train_loss"].append(metrics["loss"])
                history["lr"].append(metrics["lr"])
                history["grad_norm"].append(metrics["grad_norm"])
            if eval_examples is not None and (
                (self.step - 1) % self.eval_every == 0 or done
            ):
                history["val_step"].append(self.step - 1)
                history["val_loss"].append(self.evaluate(eval_examples))
        return history

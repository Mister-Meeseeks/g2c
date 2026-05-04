"""The full pretraining loop.

Once you have a `TransformerLM` (Module 09), pretraining is a tight
loop: sample a batch, forward, loss, backward, clip, step, log. The
ingredients you've built in Modules 03 and 09 — `Linear`,
`CrossEntropyLoss`, `SGD`, `TransformerLM` — drop straight in. The
only new pieces in this module are:

  * **`get_lm_batch`** — multi-position targets (every position
    predicts the next token), in `data.py`.
  * **`lm_cross_entropy`** — average CE across every (B, T)
    position, in `loss.py`.
  * **`cosine_with_warmup`** — linear warmup followed by cosine
    decay, in `schedule.py`.
  * **`clip_grad_norm_`** — global gradient clipping, in `clip.py`.
  * **`Trainer.train_step`** — the per-step composition of all the
    pieces above.

`Trainer.__init__`, `Trainer.lr`, `Trainer.evaluate`, and
`Trainer.train` are implemented for you. `Trainer.train_step` — the
one method that names the order of operations — is scaffolded.
`device="auto"` is also implemented: the model is moved before the
optimizer is constructed, and sampled batches should be moved before
the forward pass.

Why such tight scaffolding for a class that's mostly plumbing? Because
the *order of operations* in a training step is decisive and easy to
get wrong:

    1. zero_grad     —  before this step's backward
    2. forward       —  compute logits
    3. loss          —  compute scalar loss
    4. backward      —  populate .grad on every parameter
    5. clip          —  rescale grads if their global norm is too big
    6. lr update     —  set optimizer.lr from the schedule
    7. step          —  apply the SGD update
    8. step += 1     —  advance the schedule counter

Reordering a few of these (clipping AFTER step, advancing the counter
BEFORE the step, computing lr from `step+1` instead of `step`) all
silently produce a "trains-but-wrong" loop. Pinning the order down as
the lesson is the entire point.
"""
from __future__ import annotations

import torch

from g2c.nn import SGD, Module, resolve_device

from .clip import clip_grad_norm_
from .data import get_lm_batch
from .loss import lm_cross_entropy
from .schedule import cosine_with_warmup


class Trainer:
    """Pretraining loop for a transformer language model.

    Args:
        model: a Module whose forward returns `(B, T, V)` logits — i.e.
            `TransformerLM`. The trainer only requires
            `model.parameters()` and `model(x)`; you could in
            principle plug in any `(B, T) → (B, T, V)` model.
        batch_size: number of windows per step (`B`).
        context_length: window length per training example (`T`).
            Must be ≤ `model.max_seq_len` if the model has one.
        max_steps: total training steps (the cosine schedule's
            horizon).
        max_lr: peak learning rate at the end of warmup.
        min_lr: floor learning rate at the end of cosine decay.
        warmup_steps: linear-warmup steps. `0` disables warmup.
        weight_decay: L2 regularization (forwarded to `SGD`).
        grad_clip: optional global gradient-norm threshold. `None`
            disables clipping.
        eval_every: run a validation pass every N steps.
        eval_iters: how many random batches to average for validation.
        log_every: append metrics to history every N steps.
        generator: optional `torch.Generator` for batch reproducibility.
        device: `"auto"` moves the model and sampled batches to MPS when
            available, otherwise CPU. Pass `"cpu"` to force a CPU run.

    Attributes:
        optimizer: the inner `SGD` instance. The trainer mutates
            `optimizer.lr` once per step from the cosine schedule.
        step: the current step counter (0-indexed). Advanced by 1 at
            the end of each `train_step`.
    """

    model: Module
    batch_size: int
    context_length: int
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
    optimizer: SGD
    step: int

    def __init__(
        self,
        model: Module,
        *,
        batch_size: int,
        context_length: int,
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
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.context_length = context_length
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
        # Optimizer is constructed once — its lr will be overwritten
        # each step from the cosine schedule.
        self.optimizer = SGD(
            self.model.parameters(), lr=max_lr, weight_decay=weight_decay
        )
        self.step = 0

    def lr(self, step: int | None = None) -> float:
        """Return the learning rate the schedule prescribes at `step`.

        If `step` is None, uses `self.step` (the current counter).
        Convenient for logging and for the inline lookup inside
        `train_step`.
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

    def train_step(self, train_ids: torch.Tensor) -> dict[str, float]:
        """Perform one training step on a fresh random batch.

        Args:
            train_ids: 1-D LongTensor of training token IDs.

        Returns:
            A dict of metrics for this step:
                'loss':       the scalar loss this step (Python float)
                'lr':         the lr applied this step (Python float)
                'grad_norm':  the pre-clip global grad norm if
                              `grad_clip` is set; else 0.0

        Recipe — exactly this order:

            1. # Sample a fresh batch of (input, target) windows.
               x, y = get_lm_batch(
                   train_ids,
                   self.batch_size,
                   self.context_length,
                   generator=self.generator,
               )
               x = x.to(self.device)
               y = y.to(self.device)

            2. # Pull the lr for THIS step (before incrementing the
               # counter) and write it onto the optimizer. The
               # schedule reads `self.step` directly via `self.lr()`.
               lr = self.lr()
               self.optimizer.lr = lr

            3. self.optimizer.zero_grad()

            4. logits = self.model(x)                # (B, T, V)

            5. loss = lm_cross_entropy(logits, y)    # scalar

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
               # Advanced AFTER the step so `self.lr()` and
               # `self.step` are consistent within a single step.

           10. return {
                   'loss': loss.item(),
                   'lr': lr,
                   'grad_norm': grad_norm,
               }

        Common reorderings to avoid:

          * **Stepping the counter before applying the optimizer.** The
            lr you computed was for step `self.step`, but you'd be
            applying it AFTER incrementing — your effective schedule is
            shifted by one.
          * **Clipping after `optimizer.step()`.** The gradients have
            already been consumed; clipping after is a no-op on this
            step and clips on the NEXT step's stale grads (assuming
            you don't zero_grad first, which you should).
          * **Forgetting `zero_grad`.** PyTorch's `.backward()`
            ACCUMULATES into `.grad`. Without `zero_grad`, this step's
            gradient is added to the previous step's, the optimizer
            sees a wildly large effective gradient, and training
            diverges immediately.
        """
        # TODO
        raise NotImplementedError

    def evaluate(self, eval_ids: torch.Tensor) -> float:
        """Average `lm_cross_entropy` over `eval_iters` random batches.

        Implemented for you. Wrapped in `torch.no_grad()` because we
        don't backprop through evaluation — saves both memory (no
        autograd graph) and compute (no .grad bookkeeping).

        Returns:
            Mean cross-entropy as a Python float. To get perplexity,
            take `math.exp(...)` of the return value.
        """
        losses: list[float] = []
        with torch.no_grad():
            for _ in range(self.eval_iters):
                x, y = get_lm_batch(
                    eval_ids,
                    self.batch_size,
                    self.context_length,
                    generator=self.generator,
                )
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                losses.append(lm_cross_entropy(logits, y).item())
        return sum(losses) / len(losses)

    def train(
        self,
        train_ids: torch.Tensor,
        val_ids: torch.Tensor | None = None,
    ) -> dict[str, list]:
        """Run the full training loop for `max_steps` steps.

        Implemented for you. Calls `train_step` once per step and
        records metrics into a history dict every `log_every` steps;
        also runs an evaluation pass every `eval_every` steps if
        `val_ids` is provided.

        Args:
            train_ids: 1-D LongTensor of training token IDs.
            val_ids: optional 1-D LongTensor for periodic validation.

        Returns:
            A dict with six lists, each indexed by logging
            checkpoints:
                'step', 'train_loss', 'lr', 'grad_norm':
                    appended every `log_every` steps.
                'val_step', 'val_loss':
                    appended every `eval_every` steps (empty if no
                    `val_ids` was given).
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
            metrics = self.train_step(train_ids)
            # `self.step` was just incremented inside `train_step`; the
            # metrics correspond to the step that ran with self.step-1.
            done = self.step == self.max_steps
            if (self.step - 1) % self.log_every == 0 or done:
                history["step"].append(self.step - 1)
                history["train_loss"].append(metrics["loss"])
                history["lr"].append(metrics["lr"])
                history["grad_norm"].append(metrics["grad_norm"])
            if val_ids is not None and (
                (self.step - 1) % self.eval_every == 0 or done
            ):
                history["val_step"].append(self.step - 1)
                history["val_loss"].append(self.evaluate(val_ids))
        return history

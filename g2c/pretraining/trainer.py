"""The full pretraining loop.

Once you have a `TransformerLM` (Module 09), pretraining is a tight
loop: sample a batch, forward, loss, backward, clip, step, log. The
ingredients you've built in Modules 03, 03B, 09, and 09B — `Linear`,
`CrossEntropyLoss`, `SGD` / `AdamW`, `TransformerLM`, `get_lm_batch`,
and `lm_cross_entropy` — drop straight in. Module 10's new piece is
the trainer that composes them:

  * **`get_lm_batch`** — multi-position targets from Module 09B.
  * **`lm_cross_entropy`** — per-position CE from Module 09B.
  * **`cosine_with_warmup`** — linear warmup followed by cosine
    decay, from Module 03B, in `g2c.training.schedule`.
  * **`clip_grad_norm_`** — global gradient clipping, from Module 03B,
    in `g2c.training.clip`.
  * **`Trainer.train_step`** — the per-step composition of all the
    pieces above.

`Trainer.__init__`, including optimizer selection, `Trainer.lr`,
`Trainer.evaluate`, and `Trainer.train` are implemented for you.
`Trainer.train_step` — the one method that names the order of operations —
is scaffolded.
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
    7. step          —  apply the optimizer update
    8. step += 1     —  advance the schedule counter

Reordering a few of these (clipping AFTER step, advancing the counter
BEFORE the step, computing lr from `step+1` instead of `step`) all
silently produce a "trains-but-wrong" loop. Pinning the order down as
the lesson is the entire point.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from os import PathLike
from pathlib import Path

import torch

from g2c.nn import SGD, Module, resolve_device
from g2c.training import (  # noqa: F401 (for the student implementation)
    AdamW,
    clip_grad_norm_,
    cosine_with_warmup,
)

from .data import get_lm_batch
from .loss import lm_cross_entropy


def empty_training_history() -> dict[str, list]:
    """Return the standard history container used by `Trainer.train`."""
    return {
        "step": [],
        "train_loss": [],
        "lr": [],
        "grad_norm": [],
        "val_step": [],
        "val_loss": [],
    }


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
        weight_decay: regularization strength forwarded to the optimizer.
        grad_clip: optional global gradient-norm threshold. `None`
            disables clipping.
        eval_every: run a validation pass every N steps.
        eval_iters: how many random batches to average for validation.
        log_every: append metrics to history every N steps.
        generator: optional `torch.Generator` for batch reproducibility.
        device: `"auto"` moves the model and sampled batches to MPS when
            available, otherwise CPU. Pass `"cpu"` to force a CPU run.
        optimizer: `"sgd"` for the visible baseline update or `"adamw"` for
            the adaptive optimizer introduced in Module 03B.

    Attributes:
        device: resolved `torch.device` used for model parameters and
            training/evaluation batches.
        optimizer_name: `"sgd"` or `"adamw"`.
        optimizer: the inner optimizer instance. The trainer mutates
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
    optimizer_name: str
    optimizer: SGD | AdamW
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
        optimizer: str = "sgd",
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
        self.optimizer_name = optimizer.lower()
        # Optimizer is constructed once — its lr will be overwritten
        # each step from the cosine schedule.
        if self.optimizer_name == "sgd":
            self.optimizer = SGD(
                self.model.parameters(), lr=max_lr, weight_decay=weight_decay
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(), lr=max_lr, weight_decay=weight_decay
            )
        else:
            raise ValueError("optimizer must be 'sgd' or 'adamw'")
        self.step = 0

    def checkpoint_state(
        self,
        *,
        history: dict[str, list] | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Return a serializable snapshot for pausing and resuming training.

        The snapshot intentionally stores raw parameter tensors rather than a
        high-level `state_dict` because the course's `Module` abstraction is
        deliberately minimal. Restoring copies these tensors back into the
        existing model object.
        """
        checkpoint = {
            "version": 1,
            "trainer": {
                "step": self.step,
                "batch_size": self.batch_size,
                "context_length": self.context_length,
                "max_steps": self.max_steps,
                "max_lr": self.max_lr,
                "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps,
                "weight_decay": self.weight_decay,
                "grad_clip": self.grad_clip,
                "eval_every": self.eval_every,
                "eval_iters": self.eval_iters,
                "log_every": self.log_every,
                "optimizer_name": self.optimizer_name,
            },
            "model_params": [p.detach().cpu().clone() for p in self.model.parameters()],
            "optimizer": self._optimizer_state(),
            "generator_state": (
                self.generator.get_state() if self.generator is not None else None
            ),
            "history": history if history is not None else empty_training_history(),
            "extra": extra or {},
        }
        return checkpoint

    def save_checkpoint(
        self,
        path: str | PathLike[str],
        *,
        history: dict[str, list] | None = None,
        extra: dict | None = None,
    ) -> Path:
        """Save model, optimizer, RNG, step counter, and history to `path`."""
        from g2c.artifacts.models import save_training_checkpoint

        return save_training_checkpoint(
            self,
            path,
            history=history,
            extra=extra,
        )

    def load_checkpoint(self, path: str | PathLike[str]) -> dict:
        """Restore a checkpoint saved by `save_checkpoint` and return it."""
        from g2c.artifacts.models import load_training_checkpoint

        checkpoint = load_training_checkpoint(path, map_location="cpu")
        self.restore_checkpoint_state(checkpoint)
        return checkpoint

    def restore_checkpoint_state(self, checkpoint: dict) -> None:
        """Restore model, optimizer, RNG, and step counter from `checkpoint`."""
        trainer_state = checkpoint["trainer"]
        self.step = int(trainer_state["step"])
        self._load_model_params(checkpoint["model_params"])
        self._load_optimizer_state(checkpoint["optimizer"])
        generator_state = checkpoint.get("generator_state")
        if generator_state is not None and self.generator is not None:
            self.generator.set_state(generator_state)

    def _optimizer_state(self) -> dict:
        state = {
            "name": self.optimizer_name,
            "lr": self.optimizer.lr,
            "weight_decay": self.optimizer.weight_decay,
        }
        if isinstance(self.optimizer, AdamW):
            state.update(
                {
                    "beta1": self.optimizer.beta1,
                    "beta2": self.optimizer.beta2,
                    "eps": self.optimizer.eps,
                    "step_count": self.optimizer.step_count,
                    "m": [t.detach().cpu().clone() for t in self.optimizer.m],
                    "v": [t.detach().cpu().clone() for t in self.optimizer.v],
                }
            )
        return state

    def _load_model_params(self, saved_params: list[torch.Tensor]) -> None:
        params = list(self.model.parameters())
        if len(saved_params) != len(params):
            raise ValueError(
                f"checkpoint has {len(saved_params)} model params, "
                f"but model has {len(params)}"
            )
        with torch.no_grad():
            for i, (param, saved) in enumerate(zip(params, saved_params, strict=True)):
                if tuple(param.shape) != tuple(saved.shape):
                    raise ValueError(
                        f"checkpoint model param {i} has shape {tuple(saved.shape)}, "
                        f"but model param has shape {tuple(param.shape)}"
                    )
                param.copy_(saved.to(device=param.device, dtype=param.dtype))

    def _load_optimizer_state(self, state: dict) -> None:
        if state["name"] != self.optimizer_name:
            raise ValueError(
                f"checkpoint optimizer is {state['name']!r}, "
                f"but trainer optimizer is {self.optimizer_name!r}"
            )
        self.optimizer.lr = float(state["lr"])
        self.optimizer.weight_decay = float(state["weight_decay"])

        if isinstance(self.optimizer, AdamW):
            self.optimizer.beta1 = float(state["beta1"])
            self.optimizer.beta2 = float(state["beta2"])
            self.optimizer.eps = float(state["eps"])
            self.optimizer.step_count = int(state["step_count"])
            self._copy_optimizer_tensors(self.optimizer.m, state["m"], "m")
            self._copy_optimizer_tensors(self.optimizer.v, state["v"], "v")

    def _copy_optimizer_tensors(
        self,
        target: list[torch.Tensor],
        saved: list[torch.Tensor],
        name: str,
    ) -> None:
        if len(saved) != len(target):
            raise ValueError(
                f"checkpoint optimizer.{name} has {len(saved)} tensors, "
                f"but optimizer has {len(target)}"
            )
        with torch.no_grad():
            for i, (target_tensor, saved_tensor) in enumerate(
                zip(target, saved, strict=True)
            ):
                if tuple(target_tensor.shape) != tuple(saved_tensor.shape):
                    raise ValueError(
                        f"checkpoint optimizer.{name}[{i}] has shape "
                        f"{tuple(saved_tensor.shape)}, but optimizer tensor has "
                        f"shape {tuple(target_tensor.shape)}"
                    )
                target_tensor.copy_(
                    saved_tensor.to(
                        device=target_tensor.device,
                        dtype=target_tensor.dtype,
                    )
                )

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
        *,
        history: dict[str, list] | None = None,
        on_log: Callable[[dict[str, float | int | None]], None] | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_every: int | None = None,
        checkpoint_extra: dict | None = None,
    ) -> dict[str, list]:
        """Run the full training loop until `self.step == max_steps`.

        Implemented for you. Calls `train_step` once per step and
        records metrics into a history dict every `log_every` steps;
        also runs an evaluation pass every `eval_every` steps if
        `val_ids` is provided.

        Args:
            train_ids: 1-D LongTensor of training token IDs.
            val_ids: optional 1-D LongTensor for periodic validation.
            history: optional existing history to append to. Pass the history
                loaded from a checkpoint when continuing a run.
            on_log: optional callback called whenever training logs or
                evaluates. The callback receives a small metrics dict with
                `step`, `train_loss`, `val_loss`, `lr`, `grad_norm`,
                `elapsed_s`, and `steps_per_s`. This is useful for live
                notebook progress displays without baking print behavior into
                the trainer.
            checkpoint_path: optional path for a rolling training checkpoint.
            checkpoint_every: if `checkpoint_path` is set, save every N
                completed steps and also at the final step. If the run is
                interrupted, the trainer saves once before re-raising.
            checkpoint_extra: optional metadata stored alongside the
                checkpoint (for example model config or tokenizer name).

        Returns:
            A dict with six lists, each indexed by logging
            checkpoints:
                'step', 'train_loss', 'lr', 'grad_norm':
                    appended every `log_every` steps.
                'val_step', 'val_loss':
                    appended every `eval_every` steps (empty if no
                    `val_ids` was given).
        """
        if checkpoint_every is not None and checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive or None")
        if history is None:
            history = empty_training_history()

        start_time = time.perf_counter()
        start_step = self.step
        try:
            while self.step < self.max_steps:
                metrics = self.train_step(train_ids)
                # `self.step` was just incremented inside `train_step`; the
                # metrics correspond to the step that ran with self.step-1.
                step_index = self.step - 1
                done = self.step >= self.max_steps
                log_event: dict[str, float | int | None] | None = None
                if step_index % self.log_every == 0 or done:
                    history["step"].append(step_index)
                    history["train_loss"].append(metrics["loss"])
                    history["lr"].append(metrics["lr"])
                    history["grad_norm"].append(metrics["grad_norm"])
                    log_event = {
                        "step": step_index,
                        "train_loss": metrics["loss"],
                        "val_loss": None,
                        "lr": metrics["lr"],
                        "grad_norm": metrics["grad_norm"],
                    }
                if val_ids is not None and (
                    step_index % self.eval_every == 0 or done
                ):
                    val_loss = self.evaluate(val_ids)
                    history["val_step"].append(step_index)
                    history["val_loss"].append(val_loss)
                    if log_event is None:
                        log_event = {
                            "step": step_index,
                            "train_loss": metrics["loss"],
                            "val_loss": val_loss,
                            "lr": metrics["lr"],
                            "grad_norm": metrics["grad_norm"],
                        }
                    else:
                        log_event["val_loss"] = val_loss
                if on_log is not None and log_event is not None:
                    elapsed_s = time.perf_counter() - start_time
                    steps_this_call = self.step - start_step
                    log_event["elapsed_s"] = elapsed_s
                    log_event["steps_per_s"] = (
                        steps_this_call / elapsed_s if elapsed_s > 0 else 0.0
                    )
                    on_log(log_event)
                if (
                    checkpoint_path is not None
                    and checkpoint_every is not None
                    and (self.step % checkpoint_every == 0 or done)
                ):
                    self.save_checkpoint(
                        checkpoint_path,
                        history=history,
                        extra=checkpoint_extra,
                    )
        except KeyboardInterrupt:
            if checkpoint_path is not None:
                self.save_checkpoint(
                    checkpoint_path,
                    history=history,
                    extra=checkpoint_extra,
                )
            raise
        return history

"""Display + run-orchestration helpers for the Module 10 notebook.

These are not part of the course deliverable. They wrap the pretraining
``Trainer`` with an IPython progress bar, render training history, provide an
encode-progress callback that the artifact loader expects, and handle rolling
checkpoint resume so a student can stop training, inspect, and continue.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from IPython.display import Markdown, display

from g2c.artifacts import load_run_state
from g2c.pretraining import Trainer
from g2c.transformer import TransformerLM

__all__ = [
    "format_training_progress",
    "format_training_start",
    "load_run_state",
    "make_encode_progress",
    "plot_training_history",
    "train_with_progress",
]


_BAR_WIDTH = 28


def _humanize_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def format_training_start(
    name: str,
    *,
    max_steps: int,
    tokens_per_step: int,
    resume_step: int | None = None,
    last_train_loss: float | None = None,
    last_val_loss: float | None = None,
) -> Markdown:
    if resume_step is not None:
        completed = min(max_steps, max(resume_step, 0))
        filled = min(_BAR_WIDTH, round(_BAR_WIDTH * completed / max_steps)) if max_steps else 0
        bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
        prior_bits = []
        if last_train_loss is not None:
            prior_bits.append(f"train loss `{last_train_loss:.3f}`")
        if last_val_loss is not None:
            prior_bits.append(f"val loss `{last_val_loss:.3f}`")
        prior = f" ({', '.join(prior_bits)})" if prior_bits else ""
        return Markdown(
            f"{name}: `[{bar}]` `{completed:,}/{max_steps:,}` | "
            f"resuming from checkpoint at step `{resume_step:,}`{prior}; "
            "running first post-resume step | "
            f"tokens/step `{tokens_per_step:,}`"
        )
    bar = "-" * _BAR_WIDTH
    return Markdown(
        f"{name}: `[{bar}]` `0/{max_steps:,}` | "
        "starting first step and validation warmup; the first update can take a minute on MPS | "
        f"tokens/step `{tokens_per_step:,}`"
    )


def format_training_progress(metrics: dict, *, max_steps: int, tokens_per_step: int) -> Markdown:
    """Render one line of training progress.

    `metrics["step"]` is the absolute step counter the trainer maintains across
    resumes, so cumulative tokens trained (= (step + 1) * tokens_per_step) is
    correct even after a checkpoint reload. The rate field uses `steps_per_s`
    from the trainer (steps in the current train() call / current elapsed),
    which is the only correctly-attributable rate after a resume.
    """
    step = int(metrics["step"])
    completed = min(max_steps, step + 1)
    filled = min(_BAR_WIDTH, round(_BAR_WIDTH * completed / max_steps))
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    elapsed_s = float(metrics.get("elapsed_s") or 0.0)
    steps_per_s = float(metrics.get("steps_per_s") or 0.0)
    tokens_s = steps_per_s * tokens_per_step
    cumulative_tokens = completed * tokens_per_step
    val_loss = metrics.get("val_loss")
    val_text = (
        f" | val loss `{val_loss:.3f}` | val ppl `{math.exp(val_loss):.1f}`"
        if val_loss is not None
        else ""
    )
    return Markdown(
        f"`[{bar}]` `{completed:,}/{max_steps:,}` | "
        f"tokens `{_humanize_count(cumulative_tokens)}` | "
        f"train loss `{metrics['train_loss']:.3f}`"
        f"{val_text} | "
        f"lr `{metrics['lr']:.2e}` | "
        f"grad norm `{metrics['grad_norm']:.2f}` | "
        f"elapsed `{elapsed_s / 60:.1f} min` | "
        f"tokens/s `{tokens_s:,.0f}`"
    )


def make_encode_progress(label: str) -> Callable[[dict], None]:
    """Return a progress callback for ``load_or_encode_tokenized_*`` artifacts."""
    start = time.perf_counter()
    handle = None

    def show(message: Markdown) -> None:
        nonlocal handle
        if handle is None:
            handle = display(message, display_id=True)
        else:
            handle.update(message)

    def update(info: dict) -> None:
        phase = info.get("phase")
        elapsed = time.perf_counter() - start
        if phase == "encode_count_start":
            show(Markdown(f"{label}: preparing chunked encode..."))
        elif phase == "encode_count_chunk":
            show(Markdown(
                f"{label}: counting tokens | chunk `{info['chunk_index']:,}` "
                f"| chars `{info['chars_seen']:,}/{info['chars']:,}` "
                f"| tokens `{info['tokens_seen']:,}` "
                f"| elapsed `{elapsed:.1f}s`"
            ))
        elif phase == "encode_write_chunk":
            show(Markdown(
                f"{label}: writing tensor | chunk `{info['chunk_index']:,}/{info['chunks']:,}` "
                f"| tokens `{info['tokens_written']:,}/{info['token_count']:,}` "
                f"| elapsed `{elapsed:.1f}s`"
            ))
        elif phase == "encode_done":
            show(Markdown(
                f"{label}: encoded `{info['token_count']:,}` tokens "
                f"from `{info['chunks']:,}` chunks | elapsed `{elapsed:.1f}s`"
            ))

    return update


def plot_training_history(history: dict, *, baseline_loss: float | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["step"], history["train_loss"], label="train")
    if history["val_loss"]:
        axes[0].plot(history["val_step"], history["val_loss"], marker="o", label="val")
    if baseline_loss is not None:
        axes[0].axhline(baseline_loss, color="gray", linestyle="--", label="log(V)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross entropy")
    axes[0].legend()

    axes[1].plot(history["step"], history["lr"])
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")

    axes[2].plot(history["step"], history["grad_norm"])
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("pre-clip grad norm")

    fig.tight_layout()
    plt.show()

    if history.get("train_loss"):
        print(f"final train loss:     {history['train_loss'][-1]:.4f}")
    if history.get("val_loss"):
        final_val = history["val_loss"][-1]
        print(f"final val loss:       {final_val:.4f}")
        print(f"final val perplexity: {math.exp(final_val):.2f}")


def train_with_progress(
    name: str,
    model: TransformerLM,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    *,
    seed: int = 0,
    checkpoint_path: str | PathLike[str] | None = None,
    checkpoint_every: int = 100,
    checkpoint_extra: dict[str, Any] | None = None,
    **trainer_kwargs,
) -> dict:
    """Run ``Trainer.train`` with a live IPython progress bar.

    If ``checkpoint_path`` points to an existing file, the trainer state is
    loaded from it (model params, optimizer state, RNG, step, history) and
    training continues from where it left off. Otherwise training starts fresh.
    During training the trainer saves a rolling checkpoint to that path every
    ``checkpoint_every`` steps -- and once more on KeyboardInterrupt before
    re-raising, so Ctrl+C is a safe inspection point.

    Pass ``checkpoint_extra`` (e.g. ``{"model_config": ..., "vocab_size": ...,
    "tokenizer_artifact": ...}``) so ``load_run_state`` can reconstruct the
    model standalone after a kernel restart.

    Returns the training history dict. Prints param count, device, and
    resume/fresh state. Final train/val/perplexity numbers are reported by
    ``plot_training_history`` so they survive a kernel restart + plot rerun.
    """
    trainer = Trainer(
        model,
        **trainer_kwargs,
        generator=torch.Generator().manual_seed(seed),
    )

    history: dict | None = None
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
    did_resume = ckpt_path is not None and ckpt_path.exists()
    if did_resume:
        loaded = trainer.load_checkpoint(ckpt_path)
        history = loaded.get("history")
        print(f"{name}: resuming from step {trainer.step:,}/{trainer.max_steps:,}")
    else:
        print(f"{name}: starting fresh")

    print(f"{name} params: {sum(p.numel() for p in model.parameters()):,}")
    print("training device:", trainer.device)

    last_val_loss: float | None = None
    last_train_loss: float | None = None
    if history is not None:
        if history.get("val_loss"):
            last_val_loss = float(history["val_loss"][-1])
        if history.get("train_loss"):
            last_train_loss = float(history["train_loss"][-1])

    tokens_per_step = trainer.batch_size * trainer.context_length
    progress = display(
        format_training_start(
            name,
            max_steps=trainer.max_steps,
            tokens_per_step=tokens_per_step,
            resume_step=trainer.step if did_resume else None,
            last_train_loss=last_train_loss,
            last_val_loss=last_val_loss,
        ),
        display_id=True,
    )

    def on_log(metrics: dict) -> None:
        nonlocal last_val_loss
        if metrics.get("val_loss") is not None:
            last_val_loss = float(metrics["val_loss"])
        else:
            metrics = {**metrics, "val_loss": last_val_loss}
        message = format_training_progress(
            metrics, max_steps=trainer.max_steps, tokens_per_step=tokens_per_step,
        )
        if progress is not None:
            progress.update(message)

    history = trainer.train(
        train_ids,
        val_ids,
        history=history,
        on_log=on_log,
        checkpoint_path=ckpt_path,
        checkpoint_every=checkpoint_every if ckpt_path is not None else None,
        checkpoint_extra=checkpoint_extra,
    )
    return history

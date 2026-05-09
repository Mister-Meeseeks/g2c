"""Notebook helpers for Module 13 supervised fine-tuning.

These helpers are not part of the course deliverable. They wrap the
student-built ``SFTTrainer`` with notebook progress display and plotting so
the Module 13 notebook can focus on dataset construction, masking, and behavior
inspection.
"""

from __future__ import annotations

import math
import time
from typing import Any

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from g2c.sft import SFTExample, SFTTrainer

__all__ = [
    "plot_sft_history",
    "train_sft_with_progress",
]

_BAR_WIDTH = 28


def _progress_markdown(metrics: dict[str, Any], *, max_steps: int) -> Markdown:
    step = int(metrics["step"])
    completed = min(max_steps, step + 1)
    filled = min(_BAR_WIDTH, round(_BAR_WIDTH * completed / max_steps))
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    elapsed_s = float(metrics.get("elapsed_s") or 0.0)
    val_loss = metrics.get("val_loss")
    val_text = f" | val loss `{val_loss:.3f}`" if val_loss is not None else ""
    return Markdown(
        f"`[{bar}]` `{completed:,}/{max_steps:,}` | "
        f"train loss `{metrics['train_loss']:.3f}`"
        f"{val_text} | lr `{metrics['lr']:.2e}` | "
        f"grad norm `{metrics['grad_norm']:.2f}` | "
        f"elapsed `{elapsed_s / 60:.1f} min`"
    )


def train_sft_with_progress(
    name: str,
    trainer: SFTTrainer,
    *,
    eval_examples: list[SFTExample] | None = None,
) -> dict[str, list]:
    """Run ``SFTTrainer`` with one updating notebook progress line."""
    history: dict[str, list] = {
        "step": [],
        "train_loss": [],
        "lr": [],
        "grad_norm": [],
        "val_step": [],
        "val_loss": [],
    }
    print(f"{name}: training on {len(trainer.examples):,} examples")
    print(f"{name}: params {sum(p.numel() for p in trainer.model.parameters()):,}")
    print("training device:", trainer.device)
    progress = display(
        Markdown(
            f"{name}: `[{ '-' * _BAR_WIDTH }]` `0/{trainer.max_steps:,}` | "
            "starting first SFT step"
        ),
        display_id=True,
    )
    start = time.perf_counter()
    last_val_loss: float | None = None

    for _ in range(trainer.max_steps):
        metrics = trainer.train_step()
        step_index = trainer.step - 1
        done = trainer.step == trainer.max_steps
        log_event: dict[str, Any] | None = None

        if step_index % trainer.log_every == 0 or done:
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
                "elapsed_s": time.perf_counter() - start,
            }

        if eval_examples is not None and (
            step_index % trainer.eval_every == 0 or done
        ):
            last_val_loss = trainer.evaluate(eval_examples)
            history["val_step"].append(step_index)
            history["val_loss"].append(last_val_loss)
            if log_event is None:
                log_event = {
                    "step": step_index,
                    "train_loss": metrics["loss"],
                    "lr": metrics["lr"],
                    "grad_norm": metrics["grad_norm"],
                    "elapsed_s": time.perf_counter() - start,
                }
            log_event["val_loss"] = last_val_loss

        if log_event is not None:
            if log_event.get("val_loss") is None:
                log_event["val_loss"] = last_val_loss
            progress.update(_progress_markdown(log_event, max_steps=trainer.max_steps))

    return history


def plot_sft_history(history: dict[str, list]) -> None:
    """Plot loss, learning rate, and pre-clip grad norm for an SFT run."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["step"], history["train_loss"], label="train")
    if history.get("val_loss"):
        axes[0].plot(history["val_step"], history["val_loss"], marker="o", label="val")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("masked cross entropy")
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
        print(f"final train loss: {history['train_loss'][-1]:.4f}")
    if history.get("val_loss"):
        final_val = history["val_loss"][-1]
        print(f"final val loss:   {final_val:.4f}")
        print(f"final val ppl:    {math.exp(final_val):.2f}")

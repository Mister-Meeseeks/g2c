"""Notebook helpers for Module 14 direct preference optimization.

These helpers are not part of the course deliverable. They wrap the
student-built ``DPOTrainer`` with notebook progress display and plotting so the
Module 14 notebook can focus on preference data, sanity checks, and behavior
inspection.
"""

from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from g2c.dpo import DPOTrainer, PreferenceExample

__all__ = [
    "plot_dpo_beta_sweep",
    "plot_dpo_history",
    "train_dpo_with_progress",
]

_BAR_WIDTH = 28


def _progress_markdown(metrics: dict[str, Any], *, max_steps: int) -> Markdown:
    step = int(metrics["step"])
    completed = min(max_steps, step + 1)
    filled = min(_BAR_WIDTH, round(_BAR_WIDTH * completed / max_steps))
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    elapsed_s = float(metrics.get("elapsed_s") or 0.0)
    val_loss = metrics.get("val_loss")
    val_margin = metrics.get("val_reward_margin")
    val_acc = metrics.get("val_accuracy")
    val_text = ""
    if val_loss is not None:
        val_text = f" | val loss `{val_loss:.3f}`"
    if val_margin is not None:
        val_text += f" | val margin `{val_margin:.3f}`"
    if val_acc is not None:
        val_text += f" | val acc `{val_acc:.2f}`"
    return Markdown(
        f"`[{bar}]` `{completed:,}/{max_steps:,}` | "
        f"train loss `{metrics['train_loss']:.3f}`"
        f"{val_text} | margin `{metrics['reward_margin']:.3f}` | "
        f"acc `{metrics['accuracy']:.2f}` | "
        f"lr `{metrics['lr']:.2e}` | grad norm `{metrics['grad_norm']:.2f}` | "
        f"elapsed `{elapsed_s / 60:.1f} min`"
    )


def train_dpo_with_progress(
    name: str,
    trainer: DPOTrainer,
    *,
    eval_examples: list[PreferenceExample] | None = None,
) -> dict[str, list]:
    """Run ``DPOTrainer`` with one updating notebook progress line."""
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
    print(f"{name}: training on {len(trainer.examples):,} preference pairs")
    print(f"{name}: policy params {sum(p.numel() for p in trainer.model.parameters()):,}")
    print("training device:", trainer.device)
    progress = display(
        Markdown(
            f"{name}: `[{ '-' * _BAR_WIDTH }]` `0/{trainer.max_steps:,}` | "
            "starting first DPO step"
        ),
        display_id=True,
    )
    start = time.perf_counter()
    last_val: dict[str, float] | None = None

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
            history["chosen_reward"].append(metrics["chosen_reward"])
            history["rejected_reward"].append(metrics["rejected_reward"])
            history["reward_margin"].append(metrics["reward_margin"])
            history["accuracy"].append(metrics["accuracy"])
            log_event = {
                "step": step_index,
                "train_loss": metrics["loss"],
                "lr": metrics["lr"],
                "grad_norm": metrics["grad_norm"],
                "chosen_reward": metrics["chosen_reward"],
                "rejected_reward": metrics["rejected_reward"],
                "reward_margin": metrics["reward_margin"],
                "accuracy": metrics["accuracy"],
                "elapsed_s": time.perf_counter() - start,
            }

        if eval_examples is not None and (
            step_index % trainer.eval_every == 0 or done
        ):
            last_val = trainer.evaluate(eval_examples)
            history["val_step"].append(step_index)
            history["val_loss"].append(last_val["loss"])
            history["val_accuracy"].append(last_val["accuracy"])
            history["val_reward_margin"].append(last_val["reward_margin"])
            if log_event is None:
                log_event = {
                    "step": step_index,
                    "train_loss": metrics["loss"],
                    "lr": metrics["lr"],
                    "grad_norm": metrics["grad_norm"],
                    "chosen_reward": metrics["chosen_reward"],
                    "rejected_reward": metrics["rejected_reward"],
                    "reward_margin": metrics["reward_margin"],
                    "accuracy": metrics["accuracy"],
                    "elapsed_s": time.perf_counter() - start,
                }
            log_event["val_loss"] = last_val["loss"]
            log_event["val_accuracy"] = last_val["accuracy"]
            log_event["val_reward_margin"] = last_val["reward_margin"]

        if log_event is not None:
            if last_val is not None:
                log_event.setdefault("val_loss", last_val["loss"])
                log_event.setdefault("val_accuracy", last_val["accuracy"])
                log_event.setdefault("val_reward_margin", last_val["reward_margin"])
            progress.update(_progress_markdown(log_event, max_steps=trainer.max_steps))

    return history


def plot_dpo_history(history: dict[str, list]) -> None:
    """Plot DPO loss, reward margin, accuracy, and optimization telemetry."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_loss, ax_margin, ax_acc, ax_opt = axes.ravel()

    ax_loss.plot(history["step"], history["train_loss"], label="train")
    if history.get("val_loss"):
        ax_loss.plot(history["val_step"], history["val_loss"], marker="o", label="val")
    ax_loss.axhline(0.6931471805599453, color="gray", linestyle="--", label="log(2)")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("DPO loss")
    ax_loss.legend()

    ax_margin.plot(history["step"], history["chosen_reward"], label="chosen reward")
    ax_margin.plot(history["step"], history["rejected_reward"], label="rejected reward")
    ax_margin.plot(history["step"], history["reward_margin"], label="margin")
    if history.get("val_reward_margin"):
        ax_margin.plot(
            history["val_step"],
            history["val_reward_margin"],
            marker="o",
            linestyle="--",
            label="val margin",
        )
    ax_margin.axhline(0.0, color="gray", linestyle="--")
    ax_margin.set_xlabel("step")
    ax_margin.set_ylabel("implicit reward")
    ax_margin.legend()

    ax_acc.plot(history["step"], history["accuracy"], label="train")
    if history.get("val_accuracy"):
        ax_acc.plot(history["val_step"], history["val_accuracy"], marker="o", label="val")
    ax_acc.axhline(0.5, color="gray", linestyle="--", label="chance")
    ax_acc.set_xlabel("step")
    ax_acc.set_ylabel("chosen > rejected")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend()

    ax_opt.plot(history["step"], history["lr"], label="lr")
    ax_opt.set_xlabel("step")
    ax_opt.set_ylabel("learning rate")
    ax_grad = ax_opt.twinx()
    ax_grad.plot(history["step"], history["grad_norm"], color="tab:orange", label="grad norm")
    ax_grad.set_ylabel("pre-clip grad norm")
    lines, labels = ax_opt.get_legend_handles_labels()
    lines2, labels2 = ax_grad.get_legend_handles_labels()
    ax_opt.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    plt.show()

    if history.get("train_loss"):
        print(f"final train loss:      {history['train_loss'][-1]:.4f}")
        print(f"final train margin:    {history['reward_margin'][-1]:.4f}")
        print(f"final train accuracy:  {history['accuracy'][-1]:.3f}")
    if history.get("val_loss"):
        print(f"final val loss:        {history['val_loss'][-1]:.4f}")
        print(f"final val margin:      {history['val_reward_margin'][-1]:.4f}")
        print(f"final val accuracy:    {history['val_accuracy'][-1]:.3f}")


def _history_from_sweep_result(result: Any) -> dict[str, list]:
    if isinstance(result, dict) and isinstance(result.get("history"), dict):
        return result["history"]
    if isinstance(result, dict):
        return result
    raise TypeError("Each beta sweep result must be a history dict or contain 'history'.")


def _last_metric(history: dict[str, list], key: str) -> float | None:
    values = history.get(key)
    if not values:
        return None
    return float(values[-1])


def _metric_cell(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def plot_dpo_beta_sweep(beta_results: dict[float, Any]) -> None:
    """Plot beta-sweep loss/margin curves and display final train/val metrics."""
    if not beta_results:
        raise ValueError("plot_dpo_beta_sweep() requires at least one beta result.")

    items = sorted(beta_results.items(), key=lambda item: float(item[0]))
    fig, (ax_loss, ax_margin) = plt.subplots(1, 2, figsize=(13, 4.5))

    for beta, result in items:
        history = _history_from_sweep_result(result)
        label = f"beta={float(beta):g}"
        (loss_line,) = ax_loss.plot(
            history["step"],
            history["train_loss"],
            label=label,
        )
        color = loss_line.get_color()
        if history.get("val_loss"):
            ax_loss.plot(
                history["val_step"],
                history["val_loss"],
                marker="o",
                linestyle="--",
                color=color,
            )

        (margin_line,) = ax_margin.plot(
            history["step"],
            history["reward_margin"],
            label=label,
        )
        color = margin_line.get_color()
        if history.get("val_reward_margin"):
            ax_margin.plot(
                history["val_step"],
                history["val_reward_margin"],
                marker="o",
                linestyle="--",
                color=color,
            )

    ax_loss.axhline(0.6931471805599453, color="gray", linestyle="--", label="log(2)")
    ax_loss.set_title("DPO loss by beta")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()

    ax_margin.axhline(0.0, color="gray", linestyle="--")
    ax_margin.set_title("Reward margin by beta")
    ax_margin.set_xlabel("step")
    ax_margin.set_ylabel("chosen - rejected")
    ax_margin.legend()

    fig.suptitle("Solid lines are train; dashed markers are validation")
    fig.tight_layout()
    plt.show()

    rows = [
        "| beta | train loss | val loss | train margin | val margin | train acc | val acc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for beta, result in items:
        history = _history_from_sweep_result(result)
        rows.append(
            "| "
            f"{float(beta):g} | "
            f"{_metric_cell(_last_metric(history, 'train_loss'))} | "
            f"{_metric_cell(_last_metric(history, 'val_loss'))} | "
            f"{_metric_cell(_last_metric(history, 'reward_margin'))} | "
            f"{_metric_cell(_last_metric(history, 'val_reward_margin'))} | "
            f"{_metric_cell(_last_metric(history, 'accuracy'), digits=3)} | "
            f"{_metric_cell(_last_metric(history, 'val_accuracy'), digits=3)} |"
        )
    display(Markdown("\n".join(rows)))

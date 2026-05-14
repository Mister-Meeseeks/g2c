"""Display and data-prep helpers for the Module 06 language-model notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from g2c.artifacts import load_corpus_text

__all__ = [
    "decode_sample",
    "load_language_modeling_text",
    "logged_steps",
    "plot_lm_training_curves",
    "sample_readable_tokens",
]


def load_language_modeling_text(repo_root: Path) -> str:
    """Load TinyShakespeare, falling back to a tiny built-in corpus."""
    tiny_shakespeare = load_corpus_text("tinyshakespeare", repo_root=repo_root)
    if tiny_shakespeare is not None:
        print("Using data/datasets/tinyshakespeare.txt")
        return tiny_shakespeare

    print("Using built-in tiny corpus. Run ./setup.sh to download data/datasets/tinyshakespeare.txt.")
    base = """
    the model predicts the next token from the context
    the context helps the model choose a better next token
    the king speaks to the queen in the hall
    the queen answers the king with a careful word
    the student trains a small language model
    the small language model learns short repeated patterns
    gradients update embeddings and linear weights
    sampling repeats predict append and predict again
    """
    return ("\n".join(line.strip() for line in base.strip().splitlines()) + "\n") * 80


@torch.no_grad()
def sample_readable_tokens(
    model: Any,
    prompt_ids: torch.Tensor,
    num_tokens: int,
    *,
    observed_token_ids: torch.Tensor,
    temperature: float = 0.55,
    top_k: int | None = 12,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Conservative display sampler for easier qualitative comparisons."""
    ctx_len = model.context_length
    out = list(prompt_ids.tolist())
    for _ in range(num_tokens):
        device = getattr(model, "device", torch.device("cpu"))
        ctx = torch.tensor(out[-ctx_len:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model.logits(ctx)[0].detach().cpu()[observed_token_ids]
        candidate_ids = observed_token_ids
        if top_k is not None and top_k < logits.numel():
            logits, top_positions = torch.topk(logits, k=top_k)
            candidate_ids = observed_token_ids[top_positions]
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1, generator=generator).item()
        out.append(int(candidate_ids[sampled_index].item()))
    return torch.tensor(out, dtype=prompt_ids.dtype)


def decode_sample(
    model_name: str,
    model: Any,
    *,
    tokenizer: Any,
    sample_prompt_text: str,
    val_ids: torch.Tensor,
    observed_token_ids: torch.Tensor,
    temperature: float = 0.55,
    top_k: int | None = 12,
) -> None:
    """Sample and print one decoded comparison block."""
    prompt = torch.tensor(tokenizer.encode(sample_prompt_text), dtype=torch.long)
    if prompt.shape[0] < model.context_length:
        prompt = val_ids[: model.context_length]
    generated_ids = sample_readable_tokens(
        model,
        prompt_ids=prompt,
        num_tokens=160,
        observed_token_ids=observed_token_ids,
        temperature=temperature,
        top_k=top_k,
        generator=torch.Generator().manual_seed(10 + model.context_length),
    )
    print(f"\n--- {model_name} ---")
    print(tokenizer.decode(generated_ids.tolist()))


def logged_steps(num_steps: int, log_every: int) -> list[int]:
    """Return train_lm's logging steps: multiples of log_every plus final."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    steps = list(range(0, num_steps, log_every))
    final_step = num_steps - 1
    if steps[-1] != final_step:
        steps.append(final_step)
    return steps


def plot_lm_training_curves(
    neural_history: dict[str, list[float]],
    mlp_history: dict[str, list[float]],
    counts_ppl: float,
    *,
    num_steps: int,
    log_every: int,
) -> None:
    """Plot Module 06 training loss and validation perplexity comparisons."""
    steps = logged_steps(num_steps, log_every)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(steps, neural_history["train_losses"], label="neural bigram")
    axes[0].plot(steps, mlp_history["train_losses"], label="MLP context=3")
    axes[0].set_title("Training cross-entropy")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(steps, neural_history["val_perplexities"], label="neural bigram")
    axes[1].plot(steps, mlp_history["val_perplexities"], label="MLP context=3")
    axes[1].axhline(counts_ppl, color="0.4", linestyle="--", label="counts bigram")
    axes[1].set_title("Validation perplexity")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("perplexity")
    axes[1].legend()

    plt.show()

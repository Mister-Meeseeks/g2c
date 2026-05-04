"""Training loop and evaluation helpers for next-token prediction.

Three model-agnostic helpers used by all of `CountsBigramLM`,
`NeuralBigramLM`, and `MLPLanguageModel`:

  - `perplexity(model, ids)` — the standard LM evaluation metric.
    Lower is better; uniform-over-vocab is `vocab_size`.
  - `sample(model, prompt_ids, num_tokens)` — autoregressive sampling.
    Walk forward one token at a time, feeding the model its own output.
  - `train_lm(model, train_ids, ...)` — the SGD training loop. Only used
    for the neural models; the counts model is trained with `.fit()`.

`get_batch` is implemented for you — it's the standard "sample random windows
from a long token sequence" trick. The other three are scaffolded.

The whole point of these helpers is that they don't care WHICH language model
you give them, as long as it has `.context_length` and `.logits(ctx_ids)`.
That uniformity is also the practical payoff of building three architectures
side by side: you can swap them in and out and run the exact same evaluation.
"""
from __future__ import annotations

import torch

from g2c.nn import CrossEntropyLoss, SGD, resolve_device


def get_batch(
    ids: torch.Tensor,
    context_length: int,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (context, target) windows from a 1-D token sequence.

    Args:
        ids: 1-D LongTensor of token IDs, the corpus.
        context_length: number of tokens in each context window.
        batch_size: number of windows to sample.
        generator: optional torch.Generator for reproducibility.

    Returns:
        x: (batch_size, context_length) — the input contexts.
        y: (batch_size,)                — the next-token targets.

    Implemented for you. Standard recipe: pick `batch_size` random starting
    indices in `[0, len(ids) - context_length)`, slice out a context window
    of length `context_length`, and grab the immediately-following token as
    the target.
    """
    n = ids.shape[0]
    if n <= context_length:
        raise ValueError(
            f"ids has length {n}; need at least context_length+1 = {context_length+1}"
        )
    starts = torch.randint(0, n - context_length, (batch_size,), generator=generator)
    x = torch.stack([ids[s : s + context_length] for s in starts])
    y = torch.stack([ids[s + context_length] for s in starts])
    return x, y


def perplexity(
    model,
    ids: torch.Tensor,
    *,
    batch_size: int = 256,
    device: str | torch.device | None = "auto",
) -> float:
    """Compute perplexity on a held-out corpus.

    Perplexity is `exp(mean cross-entropy per token)`. Intuitively: "the
    model's effective branching factor at each step." A perplexity of K
    means the model is, on average, as uncertain as if it were choosing
    uniformly from K tokens.

    Sanity values:
      - Uniform-over-vocab model: perplexity = vocab_size.
      - Perfect model: perplexity = 1.
      - A bigram model on English typically gets perplexity in the 10s on
        a held-out corpus; a small neural LM in the single digits.

    Args:
        model: any object with `.context_length: int` and a callable
            `.logits(ctx_ids) -> (batch, vocab_size)`. CountsBigramLM,
            NeuralBigramLM, MLPLanguageModel all qualify.
        ids: 1-D LongTensor of token IDs to evaluate on.
        batch_size: number of windows to score per forward pass. Larger
            is faster but uses more memory.
        device: `"auto"` uses MPS for neural `Module` models when available
            and CPU otherwise. CountsBigramLM stays on CPU because it is a
            plain counts table, not a trainable `Module`.

    Returns:
        Python float — perplexity.

    Recipe:
        device = resolve_device(device)
        if hasattr(model, "to"):
            model.to(device)
        else:
            device = torch.device("cpu")

        ctx_len = model.context_length
        n_windows = len(ids) - ctx_len
        # Build all (context, target) pairs deterministically:
        contexts = torch.stack([ids[i : i + ctx_len] for i in range(n_windows)])
        targets  = ids[ctx_len : ctx_len + n_windows]
        # If model.context_length == 1, contexts has shape (n_windows, 1).
        # Both NeuralBigramLM and CountsBigramLM accept either shape, but
        # for cleanliness you may want to squeeze the trailing dim there.

        losses = []
        for start in range(0, n_windows, batch_size):
            cb = contexts[start : start + batch_size]
            tb = targets[start : start + batch_size]
            cb = cb.to(device)
            tb = tb.to(device)
            logits = model.logits(cb)          # (batch, vocab_size)
            # Per-window cross-entropy on the target token.
            # Use CrossEntropyLoss(reduction default = mean), or compute
            # log_softmax + gather yourself.
            losses.append(some_per_token_loss_for_this_batch)

        mean_ce = (weighted average of per-batch losses)
        return math.exp(mean_ce)

    Note: CountsBigramLM.logits returns LOG-probabilities (already normalized);
    NeuralBigramLM.logits returns unnormalized logits. CrossEntropyLoss handles
    both correctly because it applies log-softmax internally — and log-softmax
    of an already-log-prob distribution is the identity (up to numerical noise).
    A reasonable choice: just use CrossEntropyLoss for both.

    Hint: there's an even cleaner version using `torch.no_grad()` around the
    whole evaluation, which prevents PyTorch from building the autograd graph.
    Unnecessary for correctness; nice for speed and memory.
    """
    ctx_len = model.context_length
    n_windows = len(ids) - ctx_len
    contexts = torch.stack([ids[i : i + ctx_len] for i in range(n_windows)])
    targets = ids[ctx_len : ctx_len + n_windows]
    losses = []
    for start in range(0, n_windows, batch_size):
        with torch.no_grad():
            cb = contexts[start : start + batch_size]
            tb = targets[start : start + batch_size]
            logits = model.logits(cb)  # (batch, vocab_size)
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, tb)
            losses.append(loss.item())
    mean_ce = sum(losses) / len(losses)
    return float(torch.exp(torch.tensor(mean_ce)))

def sample(
    model,
    prompt_ids: torch.Tensor,
    num_tokens: int,
    *,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    device: str | torch.device | None = "auto",
) -> torch.Tensor:
    """Autoregressively sample `num_tokens` from a language model.

    Args:
        model: any object with `.context_length` and `.logits(ctx_ids)`.
        prompt_ids: 1-D LongTensor — the prompt to continue. Must be at
            least `model.context_length` tokens long.
        num_tokens: number of new tokens to sample.
        temperature: scales logits before softmax. `t=1.0` is normal
            sampling; `t < 1.0` makes the model more deterministic;
            `t > 1.0` makes it more uniform. `t → 0` is greedy (argmax).
        generator: optional torch.Generator for reproducibility.
        device: `"auto"` uses MPS for neural `Module` models when available
            and CPU otherwise. The returned token IDs stay on CPU so they
            are easy to decode and inspect.

    Returns:
        1-D LongTensor of length `len(prompt_ids) + num_tokens` — the prompt
        followed by the sampled continuation.

    Recipe:
        ctx_len = model.context_length
        if prompt_ids.shape[0] < ctx_len:
            raise ValueError(...)
        device = resolve_device(device)
        if hasattr(model, "to"):
            model.to(device)
        else:
            device = torch.device("cpu")

        out = list(prompt_ids.tolist())
        for _ in range(num_tokens):
            ctx = torch.tensor(out[-ctx_len:], device=device).unsqueeze(0)
            logits = model.logits(ctx)[0].detach().cpu()      # (vocab_size,)
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator).item()
            out.append(next_id)

        return torch.tensor(out, dtype=prompt_ids.dtype)

    Notes:
      - This is the simplest possible decoder. Module 11 (Sampling and
        decoding) builds out top-k, top-p, and others. Here we just need
        SOMETHING that produces text so we can eyeball the model's output.
      - We feed the model `out[-ctx_len:]` each step — even if it's a
        bigram model (`ctx_len=1`) the same loop applies; we just grab
        the very last token as context.
    """
    ctx_len = model.context_length
    if prompt_ids.shape[0] < ctx_len:
        raise ValueError(f"prompt_ids has length {len(prompt_ids)}; need at least {ctx_len}")
    
    out = list(prompt_ids.tolist())
    for _ in range(num_tokens):
        ctx = torch.tensor(out[-ctx_len:]).unsqueeze(0)   # (1, ctx_len)
        logits = model.logits(ctx)[0]                     # (vocab_size,)
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator).item()
        out.append(next_id)

    return torch.tensor(out, dtype=prompt_ids.dtype)


def train_lm(
    model,
    train_ids: torch.Tensor,
    *,
    val_ids: torch.Tensor | None = None,
    batch_size: int = 32,
    num_steps: int = 1000,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    log_every: int = 100,
    generator: torch.Generator | None = None,
    device: str | torch.device | None = "auto",
) -> dict:
    """Train a neural language model on next-token prediction with SGD.

    Args:
        model: a Module with `.context_length`, `.parameters()`, and
            `.logits(ctx_ids)` (i.e. NeuralBigramLM or MLPLanguageModel —
            NOT CountsBigramLM, which is trained with `.fit(ids)`).
        train_ids: 1-D LongTensor of training token IDs.
        val_ids: optional 1-D LongTensor for periodic validation perplexity.
        batch_size: minibatch size.
        num_steps: total SGD steps.
        lr: learning rate.
        weight_decay: L2 regularization (passed to SGD).
        log_every: record train loss (and val perplexity if val_ids given)
            every `log_every` steps.
        generator: optional torch.Generator for reproducibility.
        device: `"auto"` moves trainable `Module` models and sampled
            minibatches to MPS when available; pass `"cpu"` for a CPU-only
            run.

    Returns:
        Dict with two keys:
            'train_losses': list[float] — training cross-entropy at each
                logging checkpoint.
            'val_perplexities': list[float] — validation perplexity at each
                logging checkpoint. Empty if val_ids not given.

    Recipe:
        device = resolve_device(device)
        model.to(device)

        loss_fn = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_losses = []
        val_perplexities = []
        ctx_len = model.context_length

        for step in range(num_steps):
            x, y = get_batch(train_ids, ctx_len, batch_size, generator=generator)
            x = x.to(device)
            y = y.to(device)
            logits = model.logits(x)             # (batch, vocab_size)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0 or step == num_steps - 1:
                train_losses.append(loss.item())
                if val_ids is not None:
                    val_perplexities.append(perplexity(model, val_ids, device=device))

        return {'train_losses': train_losses, 'val_perplexities': val_perplexities}

    Implementation notes:
      - This is the same training loop as Module 03's MLP, just with a
        different data-fetching strategy (random windows instead of a
        DataLoader). The mechanics — forward, loss, zero_grad, backward,
        step — are identical.
      - For NeuralBigramLM, `x` has shape `(batch, 1)`; for MLPLanguageModel
        with `context_length=N`, `x` has shape `(batch, N)`. Both go straight
        into `model.logits(x)` — no extra reshaping needed in the loop.
    """
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_perplexities = []
    ctx_len = model.context_length

    for step in range(num_steps):
        x, y = get_batch(train_ids, ctx_len, batch_size, generator=generator)
        logits = model.logits(x)             # (batch, vocab_size)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == num_steps - 1:
            train_losses.append(loss.item())
            if val_ids is not None:
                val_perplexities.append(perplexity(model, val_ids))

    return {'train_losses': train_losses, 'val_perplexities': val_perplexities}

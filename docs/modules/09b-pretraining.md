# Module 09B — Pretraining

> **Question this module answers:** *How do we learn from text?*

![Multi-position targets in three steps: sample a (B, T) window from the token stream with the target window shifted left by one; run one TransformerLM forward pass to produce (B, T, V) logits; compute per-position cross-entropy at every (b, t) pair and average across all B * T positions.](09b-pretraining/Module09B-Hero.png)

Module 09B sits between "the architecture exists" and "the model trains on a real corpus." It is a small bridge module, but it removes a lot of shape bookkeeping from the payoff week. 

---
## Prerequisites


### Math

- **Cross-entropy.** Same classification loss from Module 03.
- **Logarithms.** `log(V)` is the uniform random baseline for a vocabulary of size `V`.

### PyTorch

- **PyTorch tensor indexing.**  Used to flatten all token positions before calling cross-entropy.
- **Shape contracts.** You should be comfortable tracing `(B, T)` inputs and `(B, T, V)` logits.

---
## Why We Start Here

At the end of Module 09, `TransformerLM.forward()` returns logits shaped `(B, T, V)`. That output is not useful until you can answer two questions:

1. What exactly are the `(B, T)` inputs and `(B, T)` targets?
2. How do those `(B, T, V)` logits become one scalar loss?

Module 06 used a fixed context and one target:

```text
context: [the, king, said]  -> target: [hello]
```

A causal transformer gives you more signal. For one window of length `T`, it produces `T` next-token predictions in parallel:

```text
ids: [4, 7, 1, 3, 9, 2, 6, 5]

x:   [1, 3, 9, 2]
     |  |  |  |
     v  v  v  v
y:   [3, 9, 2, 6]
```

Every position in `x` has a target in `y`. The causal mask makes this legal: when the model predicts `y[t]`, it can see `x[:t+1]` but not the future token itself.

## The Big Idea

Any corpus can be turned into a token stream. Start with text, tokenize it, and store a single 1-D token stream:

```python
ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
```

Then split it contiguously:

```python
train_ids, val_ids = split_token_stream(ids, train_fraction=0.9)
```

Do not shuffle the individual token IDs. Shuffling would destroy the adjacency relation that next-token prediction depends on. The training stream and validation stream can each be sampled randomly by window, but each sampled window must preserve local order.

### Batch sampling creates shifted windows

`get_lm_batch` samples `B` starting positions from a 1-D stream. Each start needs `T + 1` contiguous tokens:

```text
start = s

x = ids[s     : s + T    ]   # length T
y = ids[s + 1 : s + T + 1]   # length T, shifted left by one
```

Stacking many windows gives:

```text
x: (B, T)
y: (B, T)
```

The headline contract is simple:

```text
y[b, t] is the token immediately after x[b, t] in the corpus.
```

### The loss flattens positions

`TransformerLM(x)` returns:

```text
logits: (B, T, V)
targets: (B, T)
```

Your `CrossEntropyLoss` from Module 03 expects one row per classification example:

```text
logits:  (N, V)
targets: (N,)
```

So `lm_cross_entropy` treats every `(b, t)` position as one example:

```python
B, T, V = logits.shape
flat_logits = logits.reshape(B * T, V)
flat_targets = targets.reshape(B * T)
loss = CrossEntropyLoss()(flat_logits, flat_targets)
```

That is the pretraining objective in its smallest useful form: average next-token cross-entropy over every position in every sampled window.

### `log(V)` is the sanity baseline

If all logits are zero, softmax is uniform over the vocabulary. Cross-entropy against a uniform distribution is:

```text
loss = log(V)
perplexity = exp(loss) = V
```

At initialization, a random model's loss should usually start near `log(V)`. If it starts far below `log(V)`, suspect a bug: wrong targets, wrong reshape, repeated trivial data, or loading a trained checkpoint by accident. If it starts near `log(V)` and then drops, the model is learning.

## Concepts to Internalize

- **A token stream is supervised data once you shift it by one.**
- **Causal masking makes multi-position training valid.** Every position predicts the next token without seeing it.
- **One `(B, T)` batch contains `B * T` classification examples.**
- **Language-model cross-entropy is ordinary cross-entropy after a reshape.**
- **`log(V)` is not transformer-specific.** It is the uniform baseline for any language model with vocabulary size `V`.

### What we didn't cover

* **Distributed training.** Used to scale large-scale training beyond a single machine. Lots of devops considerations, but conceptually just an extension of gradient batching.
* **Mixed precision.** Speeds up training by using lower precision floats for most operations and selectively keeping high precision for load bearing weights. 
* **Packed datasets.** Combines multiple short training sequences into a single long sequence. Reduces wasting compute on padding.
* **Checkpoint managers.** Important for large scale runs to allow transparency into training progress without having to wait until the end of a massive run. Conceptually no change to the core pre-training loop.

---
## Scaffolding and How to Run the Tests

This module starts the `g2c/pretraining/` package:

- **`data.py`** has `split_token_stream` and `get_lm_batch`. Both are implemented.
- **`loss.py`** has `lm_cross_entropy`. This is scaffolded.
- **`answers/module-09b.md`** is the student workspace for written answers, hint requests, and partial submissions.
- **`docs/rubrics/module-09b.md`** is the grading contract agents should use when reviewing written answers.

Run:

```bash
pytest tests/test_pretraining_setup.py -x
pytest tests/test_pretraining_setup.py -v
```

The first tests should already pass. The `lm_cross_entropy` tests fail until you implement the reshape and call your Module 03 `CrossEntropyLoss`.

## What You'll Build

Package: `g2c/pretraining/`

```python
def split_token_stream(
    ids: torch.Tensor,
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:                 # implemented

def get_lm_batch(
    ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:                 # implemented

def lm_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:                                      # SCAFFOLDED
```

## Exercises

Enter questions or answers in [answers/module-09b.md](../../answers/module-09b.md) for agent help and grading. You can ask for a hint, answer one question, answer a subset, or answer all of them; blank answer sections are skipped rather than counted wrong.

1. **Shift a toy stream by hand.** Given `ids = [10, 11, 12, 13, 14, 15]`, `start = 1`, and `T = 3`, write `x` and `y`. Explain why `y[t]` is the target for `x[t]`.

2. **Explain why the causal mask matters.** In one paragraph, explain why multi-position training would be invalid without a causal mask.

3. **Inspect `get_lm_batch`.** Run the notebook cell that samples from `torch.arange(30)`. Verify that every target is input plus one. Then explain why this toy stream makes the shift easy to see.

4. **Implement `lm_cross_entropy`.** Fill in the scaffold in `g2c/pretraining/loss.py`, then run `pytest tests/test_pretraining_setup.py -k lm_cross_entropy -x`.

5. **Flatten shapes explicitly.** For `B = 2`, `T = 4`, `V = 7`, write the shapes of `logits`, `targets`, `flat_logits`, and `flat_targets`. State how many classification examples the batch contains.

6. **Compute the baseline.** For `V = 256`, `V = 512`, and `V = 1024`, compute `log(V)` and `exp(log(V))`. Explain why larger vocabularies have larger initial loss but not necessarily worse models.

7. **Random model sanity check.** Instantiate a tiny `TransformerLM`, sample one batch, compute `lm_cross_entropy`, and compare it to `log(V)`. If the two values differ a lot, write down two possible explanations.

## Pitfalls to Expect

- **Off-by-one targets.** `y` starts one token after `x`. If `y == x`, the model is learning to copy the current token, not predict the next one.
- **Sampling past the end.** Each window needs `T + 1` tokens, not just `T`.
- **Shuffling tokens.** Shuffle windows if you want randomness, not individual token IDs.
- **Flattening one tensor differently than the other.** `logits.reshape(B * T, V)` and `targets.reshape(B * T)` must preserve the same `(B, T)` order.
- **Reading `log(V)` as failure.** At step 0, `log(V)` is the expected baseline. The interesting question is whether the curve drops below it.

## Reading

- Karpathy, *nanoGPT*, especially the `get_batch` and loss computation.
- Karpathy, "Let's reproduce GPT-2 (124M)", the data-loader and training-loop sections.
- Vaswani et al., "Attention Is All You Need", causal masking and parallel sequence training context.

## Deliverable Checklist

- [ ] `pytest tests/test_pretraining_setup.py` passes.
- [ ] Notebook: `notebooks/clean/09b-pretraining.ipynb`.
- [ ] You can explain why one `(B, T)` batch contains `B * T` next-token prediction examples.
- [ ] You can implement `lm_cross_entropy` from the shape contract alone.
- [ ] You can use `log(V)` as a step-0 sanity check before a Module 10 training run.

## M-Series Notes

This module is CPU-light. The tensors are tiny, and no serious training run happens yet. MPS matters again in Module 10, where the same helpers sit inside thousands of optimizer steps.

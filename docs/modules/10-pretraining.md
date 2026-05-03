# Module 10 — Pretraining a tiny GPT

> **Question this module answers:** *How does the model absorb language patterns from raw text?*

![Pretraining the tiny GPT end-to-end: a raw token stream is sliced into (B, T) windows; each window goes through TransformerLM to produce (B, T, V) logits; lm_cross_entropy averages per-position cross-entropy across all B × T positions; loss.backward populates parameter gradients; clip_grad_norm rescales them if their global norm is too large; cosine_with_warmup picks the lr for this step; optimizer.step applies the SGD update; the step counter advances. A side panel shows sample text quality progressing through training: random characters at step 0, locally-correct subwords at step 500, locally-coherent sentences by step 2000+.](10-pretraining/Module10-Hero.png)

*The whole module on one page. The architecture from Module 09 doesn't change — pretraining is the loop that calls it. Three new utilities (`lm_cross_entropy`, `cosine_with_warmup`, `clip_grad_norm_`) plus the per-step composition `Trainer.train_step` are the entire deliverable. Internalizing the eight-step order of operations on the right side of this diagram — and the multi-position loss on the left — is the conceptual content of the module.*

## Prerequisites

Module 10 closes Phase III by training the architecture you built in Module 09. Almost everything you need is already in `g2c/`; this module adds four small utilities and the loop that calls them in the right order.

### Math

- **Cross-entropy across (B, T) positions.** The transformer outputs `(B, T, V)` logits — every position is a classification example. The pretraining loss is the *mean* per-position cross-entropy across all `B × T` (input prefix, target token) pairs. Same loss function you've used since Module 03; the only new thing is the reshape.
- **Cosine learning-rate schedule.** During cosine decay, `lr(t) = min_lr + (max_lr − min_lr) · ½ (1 + cos(π · progress))`, where `progress ∈ [0, 1]` walks from the end of warmup to the end of training. The cosine starts off slow, accelerates, then slows down again at the bottom — empirically a better balance than linear or step decay.
- **Global gradient norm.** `‖g‖ = sqrt(∑_p ‖∇p‖²)` where the sum is over every parameter. Clipping rescales every gradient by the same factor so this norm equals `max_norm` exactly, preserving direction while shortening the step.

### Computer science

- **The training loop's order of operations.** `zero_grad → forward → loss → backward → clip → step → advance counter` is decisive. Reordering a few of these (clip after step, increment before step, forgetting `zero_grad`) all silently produce a "trains-but-wrong" loop where the metric still goes down but the schedule or regularization isn't being applied as you think it is.
- **Random-window batching.** A long token sequence becomes batches by sampling random length-`T` windows. Same trick as Module 06's `get_batch`, generalized to multi-position targets.

### Programming

- **`tensor.reshape`** to fold the time dim into the batch dim: `(B, T, V).reshape(B*T, V)`. The transformer's per-position predictions all become independent rows for the cross-entropy computation.
- **`with torch.no_grad():`** for evaluation — disables autograd graph construction, freeing memory and time on inference passes.
- **`grad.mul_(scale)`** for in-place gradient rescaling. The underscore suffix is PyTorch's convention for in-place ops.

### What you can skip

- **Adam / AdamW.** Real LM pretraining uses adaptive optimizers (Adam, AdamW). We stick with `SGD` from Module 03 — adaptive optimizers are a separate topic and they don't change anything conceptual about pretraining. The exercises sweep a few base lrs; with a tuned lr, SGD reaches recognizable text on TinyShakespeare.
- **Mixed precision (fp16, bf16).** Mentioned in the syllabus as a Module 10 concept but a poor fit on M-series MPS today (occasional ops fall back to fp32, sometimes silently). Train in fp32 throughout this module. We discuss the pitfalls in *M-series notes* but the deliverable model is fp32.
- **Distributed training.** Single-device only. DDP, FSDP, ZeRO, pipeline parallelism — these are orthogonal to the pretraining recipe and out of scope for a course running on one Mac.
- **Checkpointing.** The `Trainer` doesn't ship a `save`/`load` — pickling `model.parameters()` via `torch.save({...})` is two lines with no learning content; we don't scaffold it. The exercises build a checkpointing helper if you want one.
- **Dropout, label smoothing, fancy LR schedules.** All used in production training; none teach what's already taught here.

## Why we start here

After Module 09 you have a `TransformerLM` that can do a forward pass. You don't yet have a model that has been trained on a corpus. The full pretraining recipe is short — sample a batch, forward, loss, backward, clip, step — but every line of it has a particular order that matters, and three new utilities pretraining cares about that the Module 06 trainer didn't:

```
              ┌─────────────────────────────────────────────┐
              │              ONE TRAINING STEP              │
              └─────────────────────────────────────────────┘

   train_ids ──► get_lm_batch ─► (B,T) input, (B,T) target
                                              │
                                              ▼
                             optimizer.zero_grad()
                                              │
                                              ▼
                                        model(x) ─────► logits (B,T,V)
                                              │
                                              ▼
                          lm_cross_entropy(logits, y) ─► loss (scalar)
                                              │
                                              ▼
                                       loss.backward()
                                              │
                                              ▼
                          clip_grad_norm_(params, max_norm)
                                              │
                                              ▼
              optimizer.lr ← cosine_with_warmup(step, ...)
                                              │
                                              ▼
                                      optimizer.step()
                                              │
                                              ▼
                                       step += 1
```

The new boxes — `get_lm_batch` (multi-position targets), `lm_cross_entropy` (mean over `B×T` positions), `cosine_with_warmup` (the schedule), `clip_grad_norm_` (global gradient clipping) — are this module's deliverables. Once they fit together inside `Trainer.train_step`, you have a real pretraining loop.

## The big idea

### Multi-position targets

The Module 06 trainer gave the model a context of length `N` and a single target — the very next token. That used roughly `1/N` of the gradient signal available per window: there were `N` predictable positions in each window, but you trained on only one.

A causal transformer can compute predictions at *every* position in parallel — `logits[:, t, :]` is the model's guess for `token_ids[:, t+1]`, computed as part of the same forward pass that produces every other position's prediction. So the right batch shape for pretraining is:

```
  Token IDs:   [4, 7, 1, 3, 9, 2, 6, 5, 0, ...]

  Window starting at index 2, T = 4:

    x:  [1, 3, 9, 2]
        │  │  │  │
        ▼  ▼  ▼  ▼
    y:  [3, 9, 2, 6]
        ▲  ▲  ▲  ▲
        │  │  │  │
        each is the NEXT token of the corresponding x position

  At every position t in {0, 1, 2, 3}, the model sees x[:t+1] and
  must predict y[t]. All four predictions are computed in ONE forward
  pass, and the loss averages over all four.
```

The total gradient signal per window grew from 1 prediction to `T` predictions — at typical pretraining context lengths (256, 1024, 2048, 8192), this is a factor-of-thousands speedup. It's a big part of what makes transformer pretraining tractable.

![Multi-position targets in three steps: (1) sample a (B, T) window from the token stream, with the target window y being x shifted left by one; (2) one forward pass through TransformerLM produces (B, T, V) logits — every position's next-token prediction in parallel, courtesy of the causal mask; (3) per-position cross-entropy is computed at every (b, t) pair and averaged across all B × T positions to produce the scalar loss. A side comparison contrasts the Module 06 setup (1 window → 1 training signal) against pretraining (1 window → T training signals).](10-pretraining/Module10-MultiPosition.png)

*This is what causal masking earned us. Because position t can never see token t+1, you can use position t's logits as a prediction for token t+1 — for every t in the window, simultaneously, in one forward pass. The loss then averages B × T per-position cross-entropies. Test `test_lm_cross_entropy_per_position_average` pins down that EVERY position contributes; a miswiring that only counted the last position would silently train at 1/T efficiency.*

### The training-step recipe

Eight steps, in this order, every step:

1. **`zero_grad()`** — clear the previous step's gradients. Forgetting this is the single most common bug; PyTorch ACCUMULATES into `.grad` rather than overwriting, so without `zero_grad` the "gradient" the optimizer sees grows without bound.
2. **`forward`** — compute `(B, T, V)` logits from the model.
3. **`lm_cross_entropy`** — mean per-position cross-entropy.
4. **`backward`** — populate `.grad` on every parameter via reverse-mode autodiff.
5. **`clip_grad_norm_`** — rescale every gradient by the same factor if the global norm exceeds `max_norm`. No-op when below threshold.
6. **`optimizer.lr = cosine_with_warmup(step, ...)`** — overwrite the optimizer's lr with the schedule's prescription for this step.
7. **`optimizer.step()`** — apply the SGD update.
8. **`step += 1`** — advance the schedule counter.

Two reorderings that *look* equivalent but aren't:

```
  WRONG (clip after step):

      backward()
      optimizer.step()         ← grads consumed
      clip_grad_norm_(...)     ← clips next step's stale grads,
                                 not this step's

  WRONG (advance before step):

      step += 1
      optimizer.lr = lr_for(self.step)
      optimizer.step()         ← lr is for step+1, not step

      The schedule is offset by 1 — typically harmless, but means
      "the lr at step k" and "the lr the optimizer applied at step k"
      disagree. Logging is then misleading.
```

![The eight-step training loop drawn in order: (1) zero_grad clears stale gradients; (2) forward runs the model to logits; (3) lm_cross_entropy averages per-position CE; (4) backward populates parameter .grad; (5) clip_grad_norm_ rescales if the global norm is too large; (6) set the learning rate from cosine_with_warmup; (7) optimizer.step applies the SGD update; (8) increment the step counter. A side panel pins the two most common reorderings — clipping after step (clip becomes a no-op on this step) and advancing the counter before optimizer.step (the lr you log isn't the lr you applied) — and labels both as silent-bug territory.](10-pretraining/Module10-TrainingSteps.png)

*The order is the lesson. Most miswirings of this loop produce code that *looks* like it's training: loss goes down, no exceptions, training history fills in normally — but the schedule, regularization, or clip is happening to the wrong gradients or at the wrong time. `Trainer.train_step`'s docstring spells the order out one more time, and the headline test `test_trainer_smoke_train_decreases_val_loss` is the end-to-end check that all eight steps are wired correctly.*

### Linear warmup + cosine decay

```
  lr
   ▲
   │              ╮
   │           ╱   ╰─╮
   │         ╱        ╰─╮
   │       ╱             ╰─╮
   │     ╱                  ╰─╮
   │   ╱                      ╰─╮___
   │ ╱
   └─────────────┬────────────────────► step
   0    warmup_steps               max_steps
```

**Warmup** ramps the lr linearly from `0` (or near it) up to `max_lr` over the first `warmup_steps` updates. Without warmup, the very first updates — taken with random-init weights and large gradients — can knock the model into a region from which it doesn't recover. The warmup gives the model a chance to find a sane local geometry before being pushed.

**Cosine decay** follows. From the end of warmup to `max_steps`, the lr traces the right half of a cosine curve from `max_lr` down to `min_lr` (typically `0` or a small fraction of `max_lr`). The cosine shape spends more time at high lr at the start of decay (rapid progress) and at low lr at the end (polishing) than a linear schedule would.

The whole schedule is pure arithmetic on `step` — no internal state, no PyTorch dependence. Three to five lines of code.

### Gradient clipping

```
  global norm:      ‖g‖² = ∑_p ‖p.grad‖²

  if ‖g‖ > max_norm:
      scale = max_norm / ‖g‖
      for p in params:
          p.grad ← p.grad · scale
      # afterwards: ‖g‖ = max_norm exactly
```

The clip is **global**, not per-parameter — every parameter's gradient is rescaled by the same factor. This preserves the relative direction of the gradient vector and only shortens the step. Per-parameter clipping (rescaling each parameter's grad to its own threshold) is *not* what's typically meant by "gradient clipping" in transformer training and would change the descent direction.

The clip is also a **no-op below the threshold**: if `‖g‖ ≤ max_norm`, nothing happens. A typical pretraining run clips on maybe 1–10% of steps depending on `max_norm` — usually you set it generously (`1.0` is a common default) and it only kicks in on the pathological steps.

### Why these three new pieces specifically

- **`lm_cross_entropy`** doesn't add new math — it's a reshape + Module-03 `CrossEntropyLoss`. It earns its own function because the *recipe* of "fold time into batch, then CE-loss" is the right abstraction to name. A bug-class — averaging across the batch dim but not the time dim, or vice versa — is what naming this away prevents.

- **`cosine_with_warmup`** is decisive at scale. At toy scale you can train with constant lr and not notice; at modest scale (a few million params, a few thousand steps) you start losing accuracy without warmup; at real scale you can't train without it. We add it now because once it's there, you don't have to revisit it.

- **`clip_grad_norm_`** comes from the same place: occasional pathological batches that, without clipping, can ruin a long run with one bad step.

## Concepts to internalize

- **Pretraining is the architecture in motion.** The transformer block from Module 09 didn't change; you wrap it in a loop, give it text, and the loop does the learning.
- **Every position contributes a gradient.** A single (B, T) batch yields `B × T` per-position cross-entropy examples. Don't inadvertently average over only one of those dims.
- **The order of operations in a training step is load-bearing.** zero_grad → forward → loss → backward → clip → step. Most miswirings produce a loop that *looks* like it's training.
- **Warmup matters more than you'd think.** Even at modest scale, the first few hundred steps benefit visibly from ramping `lr` from 0.
- **Cosine decay matters less than you'd think at toy scale.** Don't obsess over the schedule — picking ANY decay over flat is the bigger gain than picking cosine over linear.
- **Gradient clipping is insurance, not a control.** It exists to catch outlier batches. If clipping fires on every step, your `lr` is too high and you're training on the clipped gradient direction instead of the true one — loss goes down but slowly.
- **Validation loss and perplexity are the same thing.** `perplexity = exp(val_loss)`. Reporting one or the other is a stylistic choice; they carry the same information.

## Scaffolding and how to run the tests

This module ships five files in `g2c/training/`:

- **`data.py`** — `get_lm_batch(ids, batch_size, context_length, *, generator=None)`. Implemented for you. Same idea as Module 06's `get_batch`, with multi-position (`(B, T)`) targets.
- **`loss.py`** — `lm_cross_entropy(logits, targets)`. Scaffolded.
- **`schedule.py`** — `cosine_with_warmup(step, *, warmup_steps, max_steps, max_lr, min_lr=0.0)`. Scaffolded.
- **`clip.py`** — `clip_grad_norm_(params, max_norm)`. Scaffolded.
- **`trainer.py`** — `Trainer` class. `__init__`, `lr`, `evaluate`, `train` are implemented. `train_step` — the per-step composition of the four pieces above — is scaffolded.

Tests live in `tests/test_training.py`. Initial state: 10 passed (boilerplate: `get_lm_batch`, Trainer construction, defaults, attribute checks), 33 failed.

```bash
pytest tests/test_training.py             # all module-10 tests
pytest tests/test_training.py -x          # stop at first failure
pytest tests/test_training.py -k loss     # just lm_cross_entropy tests
pytest tests/test_training.py -k schedule # just cosine_with_warmup tests
pytest tests/test_training.py -k clip     # just clipping tests
pytest tests/test_training.py -k trainer  # just Trainer tests
pytest tests/test_training.py -v          # verbose
```

Implementation order — earlier scaffolds unblock later tests:

  1. **`lm_cross_entropy`** → unblocks the 5 loss tests.
  2. **`cosine_with_warmup`** → unblocks the 8 schedule tests plus the 2 `Trainer.lr` tests.
  3. **`clip_grad_norm_`** → unblocks the 6 clip tests.
  4. **`Trainer.train_step`** → unblocks the remaining `train_step`, `evaluate`, and `train` tests.

Steps 1–3 are independent — you can do them in any order. Step 4 composes all three into the eight-line training loop and unlocks the headline smoke-train test.

The Trainer tests pull in your full Module 03/05/08/09 stack — if those scaffolds aren't filled in (in particular, `TransformerLM.forward` from Module 09), the Trainer tests will fail with `NotImplementedError` from the prerequisite layer. The Module 09 deliverable test (`test_transformer_lm_smoke_train`) is a good gate — if that's passing, your prerequisites are in order.

The headline tests to watch:

- **`test_lm_cross_entropy_uniform_logits_equals_log_vocab`** — pins down the conceptual reference value. With random-init logits, your loss should be ≈ `log(V)`. Anything much lower than that hints at a shape bug aligning the loss against the wrong rows.
- **`test_lm_cross_entropy_per_position_average`** — pins down that EVERY position contributes to the loss, not just the last. A miswired version ("predict only the last token") would silently pass shape and uniform-baseline tests.
- **`test_cosine_with_warmup_at_max_steps`** — `cos(π) = −1`, so the coefficient bottoms out at 0 and `lr = min_lr` exactly. A common off-by-one bug puts the bottom at `max_steps + 1` or never.
- **`test_clip_grad_norm_global_across_params`** — pins down that the norm is global (one factor for the whole vector), not per-param. A per-param clip silently changes the descent direction.
- **`test_trainer_train_step_writes_lr_to_optimizer`** — pins down that the schedule actually *takes effect* on the optimizer each step. Computing the lr but never applying it is a silent bug.
- **`test_trainer_train_step_clips_grads`** — pins down that the clip actually fires when the threshold is exceeded.
- **`test_trainer_smoke_train_decreases_val_loss`** — the end-to-end test. If any piece is misordered, val loss doesn't drop.

## What you'll build

Package: `g2c/training/`

```python
def get_lm_batch(
    ids: Tensor,
    batch_size: int,
    context_length: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:                                     # implemented

def lm_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:  # SCAFFOLDED

def cosine_with_warmup(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:                                                       # SCAFFOLDED

def clip_grad_norm_(
    params: Iterable[Tensor],
    max_norm: float,
) -> float:                                                       # SCAFFOLDED

class Trainer:
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
    optimizer: SGD
    step: int

    def __init__(
        self,
        model,
        *,
        batch_size, context_length, max_steps, max_lr,
        min_lr=0.0, warmup_steps=0, weight_decay=0.0, grad_clip=None,
        eval_every=100, eval_iters=20, log_every=10,
        generator=None,
    ): ...                                                       # implemented
    def lr(self, step=None) -> float: ...                        # implemented
    def train_step(self, train_ids) -> dict[str, float]: ...     # SCAFFOLDED
    def evaluate(self, eval_ids) -> float: ...                   # implemented
    def train(self, train_ids, val_ids=None) -> dict: ...        # implemented
```

Total scaffolded code: roughly 25 lines across four functions / methods. Most of the lesson is in *which* lines and *in what order* — the math is unsubtle once you've internalized the structure of a training step.

## Exercises

1. **Train on TinyShakespeare.** Pick a small TinyShakespeare slice (say 200 KB), tokenize with your Module 04 BPE (or character-level if you prefer), build a `TransformerLM(vocab_size, embedding_dim=128, num_layers=4, num_heads=4, max_seq_len=128)` (~1M params), and train for 2000–5000 steps with `max_lr=3e-3, warmup_steps=100, grad_clip=1.0, batch_size=32, context_length=64`. Sample text every 500 steps. You should see:

     - Steps 0–100: random characters; loss ≈ `log(V)`.
     - Steps 100–500: locally-correct subword fragments emerging; loss starts dropping.
     - Steps 500–2000: locally-correct words and short phrases.
     - Steps 2000+: locally-coherent dialogue-like text. Globally still nonsense.

   The headline qualitative observation: the model learns *form* long before *content*. Punctuation, capitalization patterns, line lengths emerge first; coherent meaning never quite does at this scale.

2. **Sweep `max_lr`.** Same model and dataset; train at `max_lr ∈ {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`. Plot final val loss vs `max_lr` on log-log axes. Identify the U-shape: too-low lr underfits in the budget; too-high lr destabilizes. The sweet spot for SGD without adaptive optimizers is typically `1e-3` to `1e-2`. With Adam (out of scope) the sweet spot is one to two orders of magnitude lower.

3. **Visualize the LR and loss curves.** During the run from exercise 1, log `step, lr, loss, grad_norm` every 10 steps via the trainer's history. Plot:

     - lr vs step — should show the warmup ramp + cosine decay.
     - loss vs step — should show a fast initial drop then a slow grind.
     - grad_norm vs step — should be largest at the start (when the model is far from a good fit) and shrink as training progresses. With `grad_clip=1.0`, you'll see clipping fire on the early steps and progressively less often.

4. **No warmup.** Take exercise 1's setup and set `warmup_steps=0`, leaving `max_lr` unchanged. Train. Compare the loss curve for the first ~200 steps against the warmup version on the same axes. Three patterns are common:

     - Best case: loss takes ~50 extra steps to start falling.
     - Typical case: loss spikes within the first 10–20 steps, grad_norm spikes too, training recovers.
     - Worst case (with high `max_lr`): training never recovers — the model lands in a bad region and stays there.

5. **No clipping.** Same setup as exercise 1, but `grad_clip=None`. Inspect the `grad_norm` log. Without clipping you'll occasionally see a step where the natural gradient norm spikes by 100× the median; the model takes a giant step that takes many subsequent steps to recover from. Clipping at `1.0` would have prevented this. Quantify the clipping benefit: median val loss across 3 seeds with vs without clipping.

6. **Dataset preparation at "scale."** Tokenize a larger Project Gutenberg slice (~5 MB) with your BPE tokenizer. Save the token IDs to a `.pt` file or a memmapped `.bin` file. Build a tiny helper:

   ```python
   def load_corpus(path: str) -> torch.Tensor:
       return torch.load(path, weights_only=True)
   ```

   Train on this larger corpus. Observe that with the same parameter count, the model now sees more *distinct* contexts per epoch — perplexity should drop visibly compared to TinyShakespeare with the same step budget.

7. **Add a checkpoint helper.** Add `Trainer.save_checkpoint(path)` and `Trainer.load_checkpoint(path)` that pickle / unpickle `model.parameters()` (and `self.step`, `self.optimizer.lr`) via `torch.save` / `torch.load`. Use it to resume a long run after interrupting it. Two-line diff for each method; the lesson is in *what to store*: parameter tensors, the step counter, and any schedule state (`max_lr`, `max_steps`). Don't re-pickle the model architecture itself — store just the weights and reconstruct the model from code.

8. **Compare to the Module 06 MLP.** Train your `MLPLanguageModel` from Module 06 with `context_length = 8` on the same corpus for the same step budget. Compare final val perplexity. The transformer should beat it cleanly — for the same parameter count and step budget, attention's variable-window mixing is just strictly more powerful than the MLP's fixed-window concat. This is the first quantitative payoff for everything in Phase III.

## Pitfalls to expect

- **Forgetting `zero_grad`.** PyTorch ACCUMULATES grads. Without `zero_grad`, this step's grad is added to the previous step's, the optimizer sees a wildly-large effective grad, and training diverges immediately. You'll see your first-step loss is normal, but every subsequent step has loss spiking upward.

- **Reshape misalignment.** Computing `lm_cross_entropy(logits.reshape(B*T, V), targets.reshape(T*B))` silently aligns the wrong logits to the wrong targets. Both reshapes must use the same dim ordering (`(B, T)` flattened to `(B*T,)` and `(B, T, V)` flattened to `(B*T, V)`). The `test_lm_cross_entropy_per_position_average` test catches this.

- **Off-by-one in warmup.** `step / warmup_steps` (no `+1`) gives an lr of 0 at step 0; that's fine. `(step + 1) / warmup_steps` gives `max_lr` at step `warmup_steps - 1`. We use the second convention (matches nanoGPT). Either works; stay consistent. Don't write `(step + 1) / (warmup_steps + 1)` — that ramps up to `max_lr * warmup_steps / (warmup_steps + 1)`, which is *almost* right and much harder to debug than a clearly-wrong formula.

- **`step >= max_steps` not handled.** A common bug: `progress = (step
  - warmup_steps) / (max_steps - warmup_steps)` evaluates fine for
  `step > max_steps` but produces `progress > 1`, and `cos(π · 1.5) = 0`, which gives a NEGATIVE coefficient, which gives an `lr` BELOW `min_lr`. Test `test_cosine_with_warmup_after_max_steps_held_at_min_lr` catches this.

- **Per-parameter clipping by accident.** Looping over params and clipping each one against `max_norm / sqrt(n_params)` (or against `max_norm` itself per-param) silently changes the descent direction. The clip must compute ONE global norm and apply ONE scalar to every param's grad. Test `test_clip_grad_norm_global_across_params` catches this.

- **Stepping the counter before the optimizer.** If `self.step` is incremented before `self.optimizer.step()`, the lr you wrote onto `self.optimizer.lr` was for the OLD step but applied at the NEW step. The schedule is offset by 1; usually harmless, but means logged lr ≠ applied lr.

- **Clipping after `optimizer.step()`.** The optimizer has already consumed the unclipped grads — clipping after is a no-op on this step and (without `zero_grad` first) clips the next step's stale grads.

- **Eval grads leaking into training.** Forgetting `torch.no_grad()` in `evaluate` doesn't break correctness but builds an autograd graph for every eval forward pass. On a long-running validation pass, the per-step memory cost climbs; on big models, you'll OOM. We've implemented `evaluate` with `torch.no_grad()` for you, but if you write your own eval loop in a notebook, remember.

- **`log(V)` baseline drift.** A loss curve that starts well below `log(vocab_size)` at step 0 means your model isn't actually random-init at step 0 (maybe you're loading a checkpoint), or your loss is being computed on too-few positions, or the targets are somehow easier than uniform (e.g. the same token at every position).

- **MPS fp32 fallbacks.** Some PyTorch ops on MPS silently fall back to CPU at fp32 — typically the same per-step cost as CPU. If your training is suspiciously CPU-pinned despite specifying MPS, it's likely one specific op in the forward pass falling back. `PYTORCH_ENABLE_MPS_FALLBACK=1` makes the fallback explicit; removing the fallback means moving that op to a supported variant.

## Reading

Primary:

- **Karpathy, "Let's reproduce GPT-2 (124M)"** (YouTube). The best end-to-end walk-through of a real pretraining loop — data loading, batch shape, lr schedule, clipping, mixed precision, checkpointing. Worth watching once start to finish.
- **Karpathy, nanoGPT** (GitHub repo, ~300 lines of Python). The reference small implementation. The training loop is structurally identical to ours; the differences are AdamW (not SGD), bf16, checkpointing, multi-GPU. Reading the loop after you've written yours is illuminating.

Secondary:

- **Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017).** The paper that introduced cosine annealing schedules to the deep-learning community. Section 3 has the formula we use; the warm-restarts variant is out of scope.
- **Pascanu, Mikolov, Bengio, "On the difficulty of training recurrent neural networks" (2013).** The paper that introduced gradient clipping by global norm. Same recipe we use, derived in the context of RNN exploding gradients — the argument transfers cleanly to transformers.
- **Kaplan et al., "Scaling Laws for Neural Language Models" (2020).** The empirical scaling laws — how loss falls as a power function of model size, dataset size, compute budget. Doesn't affect this module's deliverable but sets the conceptual frame for Module 12 (scaling experiments).
- **Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022, Chinchilla).** The follow-up to Kaplan: roughly, Kaplan's recommended models were over-parameterized for their training budget. Chinchilla's "20 tokens per parameter" rule of thumb is a useful Phase IV mental model.

Optional:

- **Goyal et al., "Accurate, Large Minibatch SGD" (2017).** The paper that established the linear scaling rule and, critically, the necessity of warmup for large-batch training. The argument for warmup that applies to transformer pretraining originates here.
- **Dettmers, "LLM.int8()" (2022)** and the bitsandbytes library. Quantization-aware training is out of scope but Phase V will return to int8 / int4 inference.

## Deliverable checklist

- [ ] All tests in `tests/test_training.py` pass.
- [ ] Notebook: `notebooks/10-pretrain-tinyshakespeare.ipynb`. Train a ~1M-param `TransformerLM` on TinyShakespeare for 2000+ steps. Sample text every 500 steps. Save the run history.
- [ ] Notebook: `notebooks/10-lr-sweep.ipynb`. The exercise 2 sweep (`max_lr ∈ {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`). Plot final val loss vs `max_lr` on log-log axes; identify the sweet spot.
- [ ] You can explain — out loud, without notes — why every position in the (B, T) batch contributes a separate cross-entropy example, and why this is a `T`-fold speedup over Module 06.
- [ ] You can explain — out loud, without notes — what warmup is for, what cosine decay is for, and what gradient clipping is for.
- [ ] You can explain — out loud, without notes — the eight-step training-step recipe and what breaks if you reorder it.

## M-series notes

This is the first module where compute starts to matter in earnest.

- **Tests run in well under a second on CPU** — the trainer test suite uses tiny models (1 layer, embedding_dim=8, vocab=12). MPS is unnecessary for testing.
- **Exercise 1 (TinyShakespeare, 1M params, 2000 steps)** runs in roughly 5–15 minutes on CPU and 1–3 minutes on MPS, depending on your specific Mac. MPS is a real win at this scale.
- **Exercise 2 (LR sweep, 5 runs)** is 5× exercise 1 — plan for a half-hour run with a coffee break.
- **A 10M-param model on a ~5 MB Gutenberg slice (exercise 6 + 8)** is the first config where MPS is essential rather than nice — on CPU it's a multi-hour run, on MPS it's 30–60 minutes.
- **Mixed precision is officially supported on MPS via `torch.amp.autocast(device_type='mps', dtype=torch.float16)`** but has more silent op-fallback edges than CUDA's. We don't use it in this module; revisit in Module 12 if you want to push to 30M+ params.
- **Memory:** the `(B, T, V)` logits tensor is the dominant activation cost. For `B=32, T=64, V=8192`, that's `32 · 64 · 8192 · 4 bytes ≈ 64 MB` per training step (×2 if you hold both forward activations and gradients — autograd typically holds them). Budget accordingly; if you OOM, halve `batch_size` before halving `T`. The model only sees `(B, T)` worth of data per step, so doubling `B` and halving `T` is not a free trade — smaller `T` means shorter dependencies the model can learn.
- **`torch.set_default_device('mps')`** at the top of a notebook saves you `.to('mps')` calls everywhere. Just remember that `torch.tensor([1, 2, 3])` then constructs on MPS, not CPU.

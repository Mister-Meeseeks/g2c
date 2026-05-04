# Module 11 — Sampling and decoding

> **Question this module answers:** *How does a probability distribution over tokens become actual text?*

![Sampling and decoding on one page: the trained TransformerLM (left) emits (B, T, V) logits at every step; only the last position's row, (1, V), is used. Four logit warpers — repetition penalty, temperature, top-k, top-p — apply in that order, each transforming logits to logits and setting dropped tokens to -inf. The warped logits go through softmax and multinomial to sample one new token id, which is appended to the running sequence. A side panel contrasts greedy decoding (skip every warper, take argmax — deterministic) with sampled decoding (full pipeline). The whole loop repeats max_new_tokens times.](11-sampling/Module11-Hero.png)

*The whole module on one page. The model from Modules 09/10 doesn't change — sampling is the loop that calls it, plus four small functions that reshape the model's native distribution before each multinomial draw. Internalizing the order of those four warpers and the eight-step loop on the right is the conceptual content of the module.*

## Prerequisites

Module 11 opens Phase IV. You're done building from-scratch architecture — the transformer is in `g2c/transformer/`, you trained one in Module 10, and now you need to actually generate text from it. Sampling is short, mostly stateless, and entirely about logit-space arithmetic. Almost everything is one or two lines of code.

### Math

- **Softmax over logits.** Already used in Modules 03–10. Here it's the last step of every sampling iteration.
- **Cumulative distribution functions.** Cumulatively sum probabilities in descending order.

### Computer science

- **Stateful vs stateless code.** The four warpers are pure functions of `(logits, args)`. The decode loop is the only stateful piece (the running token sequence). Keeping warpers stateless is what lets them compose freely.

### Programming

- **`torch.topk(x, k, dim=-1)`** — returns the `k` largest values along the last dim (and their indices). Used to find the top-k threshold.
- **`torch.sort(x, dim=-1, descending=True)`** — returns sorted values and the original indices. Used by top-p to walk the descending CDF.
- **`tensor.scatter_(dim, index, src)`** — write `src` values into a destination tensor at positions given by `index` along `dim`. Used to map per-token edits back to the original (un-sorted) layout.
- **`tensor.gather(dim, index)`** — the dual: read values from a source tensor at positions given by `index` along `dim`. Used by the repetition penalty to read off prior-token logits.
- **`torch.multinomial(probs, num_samples, generator=...)`** — sample indices proportional to `probs`. The standard way to do categorical sampling in PyTorch.
- **`@torch.no_grad()`** — decorator that disables autograd for the whole function. Generation never needs gradients; skipping the graph build saves both memory and time.

---
## Where this fits in

After Module 10 you have a checkpoint: a `TransformerLM` whose parameters have been trained against next-token cross-entropy. Calling `model(token_ids)` returns `(B, T, V)` logits — but logits aren't text. The transformation from a distribution over tokens to actual sampled output is what this module is about.

Pure greedy decoding (always pick the argmax) seems like the obvious first answer:

```
  prompt = "The cat sat on the"
  greedy:  → " cat sat on the cat sat on the cat sat on the ..."
```

Greedy decoding *loops*. The model emits the most-likely token, that token shifts the context one step, but the new context is similar enough to the old that the model's prediction stays nearly the same, and the model keeps emitting the same token (or short cycle) forever. This is a structural property of small LMs in particular — a more capable model loops less, but never stops looping entirely.

Pure random sampling (multinomial draw from the native softmax) is the other extreme:

```
  prompt = "The cat sat on the"
  random: → " mat. Suddenly inflation electricity quartz... ..."
```

The native softmax has nonzero mass on every token, including thousands of long-tail tokens that are essentially unrelated to the prefix. Once in a while one of them gets sampled, and the output derails. Both naive approaches are brittle when it comes to generating text, especially at long-range. 

## The big idea

The fix is a small pipeline of **logit warpers** that reshape the distribution before sampling — the model's native distribution gets sharpened (temperature), the long tail gets cut off (top-k or top-p), and tokens that already appeared get nudged down (repetition penalty). After the warpers, multinomial sampling produces text that is neither stuck on argmax nor scattered across the vocabulary.

```
              ┌────────────────────────────────────────────────┐
              │              ONE DECODE STEP                    │
              └────────────────────────────────────────────────┘

   full_ids ──► crop to model.max_seq_len ──► (1, T_ctx)
                                                     │
                                                     ▼
                                              model(ctx)
                                                     │
                                                     ▼
                                       logits[:, -1, :]   (1, V)
                                                     │
                                                     ▼
                          apply_repetition_penalty(logits, full_ids, ρ)
                                                     │
                                                     ▼
                                  apply_temperature(logits, T)
                                                     │
                                                     ▼
                                      top_k_filter(logits, k)
                                                     │
                                                     ▼
                                      top_p_filter(logits, p)
                                                     │
                                                     ▼
                                       softmax(logits)    (1, V)
                                                     │
                                                     ▼
                                     multinomial(probs, 1)
                                                     │
                                                     ▼
                              append next_id to full_ids
                                                     │
                                                     ▼
                            stop if next_id == eos_id, else loop
```

The four boxes — `apply_repetition_penalty`, `apply_temperature`, `top_k_filter`, `top_p_filter` — are this module's deliverable warpers. The composition of all four inside a `for` loop is the `generate` function. Once they're wired together you can finally read your model's output as text.

### Logit warpers as a pipeline

Every warper has the same signature:

```
  warper:  logits  ──►  warped_logits
            (..., V)        (..., V)
```

Same shape in, same shape out. Dropped tokens are replaced with `-inf` (so `softmax(-inf) = 0` cleanly), surviving tokens keep their original logit values, and the warper does no normalization. This is what lets the warpers compose: any subset, in any order, can be chained together, and the final softmax handles the renormalization.

```
  ┌──────────────────────┐    ┌────────────────────┐
  │ repetition_penalty   │ ──►│  temperature       │ ──►
  │ rescales prior tokens│    │ divides everything │
  └──────────────────────┘    └────────────────────┘

      ┌───────────────────┐    ┌────────────────────┐
   ──►│  top_k_filter     │ ──►│  top_p_filter      │  ──► softmax
      │ -inf below rank k │    │ -inf below CDF p   │
      └───────────────────┘    └────────────────────┘
```

The order matters but the contract doesn't change: each box takes a logit tensor and returns a logit tensor.

![The four warpers laid out as a pipeline of (V,)-shape vectors. Initial logits from the model: a smooth distribution with one prominent peak. After repetition_penalty: a few prior-token positions have been pushed down (their bars shrink). After temperature: the WHOLE distribution sharpens or flattens uniformly (every bar scales). After top_k_filter: only the top-k bars remain; the rest are replaced by sentinel "-inf" markers (drawn as black bars at the bottom). After top_p_filter: bars beyond the cumulative-p mark are also "-inf"-marked. After softmax: the surviving bars are renormalized to sum to 1; the multinomial draws one. A bottom strip captures the headline: each warper is a pure function on (..., V), composes freely, and the actual probability normalization happens only at the final softmax.](11-sampling/Module11-WarperPipeline.png)

*The four warpers are pure functions of `(logits, args)` — none of them maintain state, none of them peek at each other, and the order in which they apply is the lesson. The canonical order — repetition → temperature → top-k → top-p → softmax → sample — matches HuggingFace's `LogitsProcessorList` default and is what every reference implementation you'll read uses. Reordering produces a different (but not necessarily wrong) distribution; if you're reproducing a paper's recipe, double-check what order it specifies.*

### Temperature: a single-knob sharpness control

Temperature is the simplest warper and the most common knob for generation:

```
  warped[i]  =  logits[i] / T
```

After softmax:

```
  T → 0⁺   probs collapse onto argmax (one-hot)
  T  = 1   the model's native distribution
  T → ∞    probs flatten toward 1/V (uniform)
```

Crucially, temperature **never reorders tokens.** Whatever was most likely stays most likely; whatever was least likely stays least likely. Temperature is a monotone reweighting of probabilities. So temperature alone can't rescue a model whose native argmax is wrong; it can only adjust how aggressively to commit to the argmax.

The `T = 0` case is reserved for the explicit "greedy" path in `generate` — the warper itself rejects `T <= 0` because dividing by zero produces `inf` and infects the rest of the pipeline. Greedy mode skips every warper and takes `argmax(logits)` directly.

### Top-k: a hard cutoff on count

The model's softmax has nonzero probability on every one of `V` tokens. With `V = 50_000`, the long tail — even very-low-probability tokens — collectively has noticeable mass; sampling from the full distribution occasionally produces a long-tail draw that derails the sequence. Top-k caps the surviving set at the `k` highest-probability tokens:

```
  Sort logits descending. Keep the top k. Set everything else to -inf.
```

After softmax, the dropped tokens have probability exactly 0 and the surviving `k` are renormalized to sum to 1. The argmax is always one of the survivors; top-k can never reorder rankings.

```
  Native:    [3.2  2.7  2.1  1.9  1.5  ...  -0.4]   (V tokens)
              ▲    ▲    ▲    ▲    ▲          ▲
              top-3 │    │    └─────┴── below threshold
              ▼    ▼    ▼
  After k=3: [3.2  2.7  2.1  -inf -inf  ...  -inf]
```

The downside: a fixed `k` doesn't adapt to the model's confidence. If the top token has 95% of the mass, top-50 keeps 49 essentially-zero distractors. If the model is uncertain across hundreds of plausible continuations, top-50 cuts off many reasonable choices. Top-p was designed to fix exactly that.

### Top-p (nucleus): an adaptive cutoff on mass

Top-p sorts the probabilities descending and keeps the smallest prefix of the CDF whose mass reaches `p`:

```
  Sort probabilities descending.
  Compute cumulative mass.
  Keep the smallest prefix whose cumulative mass >= p.
  Set everything outside that prefix to -inf.
```

Concretely, with `p = 0.9`:

```
  Native:       [0.60 0.20 0.10 0.05 0.03 0.02 ...]
  Cumulative:   [0.60 0.80 0.90 0.95 0.98 1.00 ...]
                 ▲    ▲    ▲ ─── crosses 0.9 here; KEEP through here
                 │    │    │
                 KEEP KEEP KEEP, drop everything after
  After p=0.9:  [0.60 0.20 0.10 -inf -inf -inf ...]
```

When the model is confident, the prefix is small (one or two tokens); when the model is uncertain, the prefix expands to include more candidates. The size of the surviving set adapts automatically.

The off-by-one to internalize: the **first** token whose cumulative mass crosses `p` is **kept**, not dropped. A buggy implementation that "drops everything once cumulative > p" can mask the argmax outright when the argmax alone has probability `> p`. The argmax must always survive; this is what `test_top_p_filter_argmax_always_survives` pins down.

### Repetition penalty: discouraging loops

Small LMs loop. "the cat sat on the cat sat on the cat sat on the..." Once the model is in a state where token `X` is most-likely, emitting `X` doesn't shift the state much, so `X` stays most-likely. The repetition penalty (Keskar et al., "CTRL", 2019) is the standard defense:

```
  for every token id that appeared in the prior context:
      if that token's logit is positive: divide by penalty
      if that token's logit is negative: multiply by penalty
```

Both branches push probability **down**. The asymmetric formula handles the sign so the rescaling is always a penalty regardless of the sign of the logit. Penalty `1.0` is no-op; penalty `1.05` to `1.3` is the typical range; penalty `>= 2` often kills repeats entirely (and damages output quality at the same time).

```
  prior tokens:   [..., 5, 7, 5, 2, ...]      tokens 5, 7, 2 are seen

  Native logits:  [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, ...]
                                  ▲             ▲       ▲
                          penalize    penalize    penalize
                           a₂          a₅          a₇
                           (2)         (5)         (7)
```

Two design points to internalize:

  * **Apply BEFORE temperature.** Repetition penalty operates on raw logits; temperature uniformly scales whatever logits it sees. Apply repetition penalty first so the penalty isn't itself amplified or attenuated by the temperature divide. (The canonical HuggingFace pipeline order is `repetition_penalty → temperature → top_k → top_p → softmax`.)
  * **Penalty applies to TOKEN IDs, not positions.** A token id seen once long ago is penalized just as much as one seen recently. Some variants weight by recency or count; CTRL doesn't, and we follow CTRL.

### The decode loop

The full per-step recipe:

1. **Crop.** `ctx = full_ids[-model.max_seq_len:]`. The `LearnedPositionalEmbedding` table only goes up to `max_seq_len`, so anything longer would either crash or silently lose positional signal.
2. **Forward.** `logits = model(ctx.unsqueeze(0))` — shape `(1, T_ctx, V)`. (We feed a batch of size 1 because we're generating one sequence at a time.)
3. **Slice.** `last_logits = logits[:, -1, :]`. The model emits a prediction at every position; we only want the next token after the last input position.
4. **Warp.** `repetition_penalty → temperature → top_k → top_p`. Each step is a one-line call; each is a pure function on logits.
5. **Softmax + multinomial.** `probs = softmax(last_logits); next_id = multinomial(probs, 1)`. The single sampled token id.
6. **Append.** `full_ids = cat([full_ids, next_id])`. The sampled token joins the running sequence.
7. **Stop.** If `next_id == eos_id`, break.
8. **Repeat.** Up to `max_new_tokens` times.

A reordering that *looks* equivalent but isn't:

```
  WRONG (warp logits at every position, not just last):

      logits = model(ctx)              # (1, T, V) — every position
      logits = apply_temperature(logits, T)
      logits = top_k_filter(logits, k)
      last_logits = logits[:, -1, :]   # then slice

      Mathematically yields the same answer (the warpers are per-row),
      but does T-fold more work per step. Generation gets T× slower for
      no reason. Slice FIRST.

  WRONG (softmax before warpers):

      probs = softmax(model(ctx)[:, -1, :])
      probs = apply_temperature(probs, T)   # ??? doesn't compose
      ...

      Temperature in probability space isn't the same operation as
      temperature in logit space; top-k/top-p in probability space
      requires renormalization at every step. Stay in logit space
      until the very last softmax.
```

![The eight-step decode loop drawn in order: (1) crop full_ids to model.max_seq_len; (2) forward the cropped context to (1, T_ctx, V) logits; (3) slice the last position to (1, V); (4) apply the warpers in canonical order; (5) softmax to probabilities; (6) multinomial draw of one token; (7) append to the running sequence; (8) stop early on eos_id else loop. A side panel pins three reorderings that produce silently-wrong or silently-slow outputs: warping the full (T, V) tensor instead of the last row (T× slower), warping in probability space instead of logit space (composition breaks), and forgetting to crop (crash or silently-lost positional signal once T > max_seq_len).](11-sampling/Module11-DecodeLoop.png)

*The order is the lesson. Most miswirings of this loop produce code that *looks* like it's generating text — output appears, no exceptions — but the schedule of warpers is being applied to the wrong tensor or in the wrong space. `generate`'s docstring spells the order out one more time, and the headline test `test_generate_top_k_one_matches_greedy` is the end-to-end check that the warpers AND the multinomial sampling are wired correctly: top-k=1 leaves only the argmax with nonzero mass, so multinomial deterministically picks it, matching the greedy path.*

### The diversity-vs-quality tradeoff

Sampling controls trade off two things you can't have both of:

  * **Diversity:** how surprising / non-deterministic the output is.
  * **Quality:** how locally-coherent / on-prompt the output is.

```
   Pure greedy           ◄─────  quality  ─────►           Pure random
   (T=0, top_k=1)         high                              low
                                                   high
                          low                               diversity
```

Sliding to the right (raising temperature, removing top-k/top-p) gets more diverse and more chaotic output. Sliding to the left (lowering temperature, narrowing top-k/top-p) gets more confident and more boring output (and at the limit, looped output).

There is no globally-correct knob position. The canonical "balanced" setting in the open-LM community is roughly `temperature=0.7`, `top_p=0.9`, with no top-k and a small repetition penalty. Real applications tune these per-task: code generation usually wants low temperature and tight top-p; creative writing wants higher temperature and looser top-p; chat assistant wants something in the middle.

The exercises sweep these knobs against your trained TinyShakespeare model so you can develop intuition by reading the output.

## Concepts to internalize

- **Sampling is a loop around the model, not a property of the model.** The same `TransformerLM` can produce wildly different output styles depending on the warper settings. Architecture is not destiny.
- **Logit warpers are pure functions.** Same shape in, same shape out, no state. The composition is the strategy; the warpers are the building blocks.
- **Mask in logit space, not probability space.** Setting dropped logits to `-inf` is the cleanest way to express zero mass. Use real `float('-inf')`, not "a very large negative number."
- **The argmax is always kept.** Both top-k (for any `k >= 1`) and top-p (for any `p > 0`) preserve the argmax. A warper that can mask the argmax has a bug.
- **Temperature reorders nothing.** It only changes the sharpness of the distribution. To actually *change* what the model is most likely to say, you need a different model (or fine-tuning, or a different prompt) — not a different temperature.
- **The decode loop is `O(max_new_tokens × T_ctx²)` without KV cache.** Every step recomputes attention over the entire running context. The cost grows quadratically with context length. KV caching cuts this to linear, but that's a Module 16 concern.
- **The diversity-quality tradeoff has no free lunch.** Lower temperature → more confident → more repetitive. Higher temperature → more creative → more derailed. Pick a setting per task and don't expect one knob to fit everything.

### What we didn't cover

- **Beam search.** A breadth-first decode that keeps the top-`k` candidate sequences at every step. Important historically (machine translation), nearly absent from modern LLMs because the diversity-vs-quality tradeoff that beam search optimizes badly maps onto open-ended generation. Skim the Wikipedia entry once; we don't implement it.
- **Speculative decoding.** A small "draft" model proposes tokens, a large "target" model accepts or rejects. A real production inference-time speedup, completely orthogonal to the sampling controls in this module. Module 16 returns to it briefly.
- **KV caching.** The transformer recomputes attention for the entire context at every decode step — `O(T²)` work for `T` tokens of history at every new token. Caching the K and V tensors at each layer turns this into `O(T)`. Big win at inference time, but it's an optimization, not a sampling concept. Skipped here; revisited in Module 16.
- **Typical sampling, mirostat, η-sampling.** Variants on top-p with somewhat different cutoff rules. Marginal real-world differences; not worth implementing four versions of "rank tokens, draw a cutoff, sample."
- **Logit biasing / forced decoding.** Sometimes you want to *forbid* certain tokens (filtering profanity, requiring JSON), or *force* certain tokens (constrained decoding, JSON-mode). Both are simple extensions of the warper interface — set chosen logits to `-inf` or `+inf` before softmax — and we don't build them; the exercises do.

---
## What you'll build

Package: `g2c/sampling/`

```python
def apply_temperature(
    logits: Tensor,
    temperature: float,
) -> Tensor:                                                    # SCAFFOLDED

def top_k_filter(
    logits: Tensor,
    k: int,
) -> Tensor:                                                    # SCAFFOLDED

def top_p_filter(
    logits: Tensor,
    p: float,
) -> Tensor:                                                    # SCAFFOLDED

def apply_repetition_penalty(
    logits: Tensor,
    token_ids: Tensor,
    penalty: float,
) -> Tensor:                                                    # SCAFFOLDED

@torch.no_grad()
def generate(
    model,
    prompt_ids: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:                                                    # SCAFFOLDED
```

Total scaffolded code: roughly 30 lines across five functions. The math is light; the lesson is the order, the masking convention (`-inf`), and the composition.

Implementation order — earlier scaffolds unblock later tests:

  1. **`apply_temperature`** → unblocks the 7 temperature tests.
  2. **`top_k_filter`** → unblocks the 8 top-k tests.
  3. **`top_p_filter`** → unblocks the 7 top-p tests.
  4. **`apply_repetition_penalty`** → unblocks the 8 repetition tests.
  5. **`generate`** → unblocks the remaining 13 generate tests.

Steps 1–4 are independent — you can do them in any order. Step 5 composes all four warpers into the autoregressive loop and unlocks the headline end-to-end tests.

### How to run the tests

Tests live in `tests/test_sampling.py`. Initial state: 0 passed, 43 failed.

```bash
pytest tests/test_sampling.py                  # all module-11 tests
pytest tests/test_sampling.py -x               # stop at first failure
pytest tests/test_sampling.py -k temperature   # just temperature tests
pytest tests/test_sampling.py -k top_k         # just top-k tests
pytest tests/test_sampling.py -k top_p         # just top-p tests
pytest tests/test_sampling.py -k repetition    # just repetition-penalty tests
pytest tests/test_sampling.py -k generate      # just generate tests
pytest tests/test_sampling.py -v               # verbose
```



## Exercises

1. **Sweep temperature on TinyShakespeare.** Load your Module 10 checkpoint, decode a 200-token sample at `temperature ∈ {0.1, 0.5, 0.7, 1.0, 1.3, 2.0}` from the same prompt with the same seed (`generator=torch.Generator().manual_seed(0)`). Print all six samples side by side. Expected pattern:

     - `T=0.1`: nearly-greedy. Tight, possibly looped output.
     - `T=0.5`: locally-coherent, somewhat conservative.
     - `T=0.7–1.0`: the model's natural style. Most diverse.
     - `T=1.3+`: starts to derail; recognizable English drifts toward garbled.
     - `T=2.0`: clearly broken — long-tail tokens dominate.

   This is the headline qualitative observation: small changes in `T` produce visible style changes; large changes break things.

2. **Compare top-k vs top-p.** Same checkpoint, same prompt, same seed. At fixed `temperature=0.8`, decode 200 tokens with:

     - No filter (baseline)
     - `top_k=10`
     - `top_k=50`
     - `top_p=0.7`
     - `top_p=0.9`
     - `top_p=0.95`

   Compare the outputs qualitatively. The headline observation: top-p adapts (the surviving set is small when the model is confident, large when it's uncertain) while top-k applies the same cutoff every step regardless of the model's confidence. When the model is confident, top-k=50 keeps 49 essentially-zero distractors that the multinomial occasionally lands on; top-p=0.9 keeps just the few tokens that matter.

3. **Quantify the looping problem.** Decode 500 tokens at `T=0.0` (greedy) and compute the *type-token ratio* — the number of distinct tokens divided by the total number of tokens. Then do the same at `T=0.7`, `T=0.7 + repetition_penalty=1.2`, and `T=0.7 + repetition_penalty=1.5`. Plot type-token ratio vs penalty. The expected curve: greedy is the worst, plain `T=0.7` is much better, and the repetition penalty further increases the ratio at the cost of starting to scramble syntax at high values.

4. **Build an interactive playground.** Wrap `generate` in a small CLI loop:

   ```python
   while True:
       prompt = input("> ")
       if prompt == "/quit": break
       ids = tokenize(prompt)
       out = generate(model, ids, max_new_tokens=200,
                       temperature=0.8, top_p=0.9)
       print(detokenize(out))
   ```

   Add `/temp 0.5`, `/top_p 0.95`, `/seed 42` commands so you can change settings without restarting. Spend 10 minutes typing prompts and watching the model react. This is the fastest way to build intuition for what each knob does — read the syllabus, then *play*.

5. **Implement greedy two ways.** Verify that:

     - `generate(..., temperature=0.0)` (the explicit greedy path)
     - `generate(..., temperature=1.0, top_k=1)` (multinomial over a single non-`-inf` token)

   produce identical outputs from the same prompt. They should — the second is operationally greedy because multinomial deterministically picks the only nonzero-mass token. (Test `test_generate_top_k_one_matches_greedy` already pins this; the exercise is to confirm it on a real model.)

6. **Logit biasing.** Add a `forbid_ids: list[int] | None` argument to a copy of `generate` that sets `last_logits[:, forbid_ids] = float('-inf')` before the warpers run. Use it to forbid token IDs corresponding to a few "bad" tokens (your tokenizer has probably tokenized something offensive, profane, or conversation-ruining; pick three). Generate with and without the forbid set; observe that those tokens never appear in the output. This is the simplest content-control mechanism real systems use.

7. **Forced first token.** Same idea, opposite direction: add a `force_first_id` argument that, on the very first generation step, replaces the model's argmax with the forced id and continues from there. Use it to seed the model's continuation with a particular word; observe how strongly the prefix steers the rest of the generation. This is a baby version of what classifier-free guidance and logit-bias APIs do in real systems.

8. **Compare `(T=0.7, top_p=0.9)` against `(T=0.0)` on a held-out set.** Tokenize a held-out passage of TinyShakespeare. Use your model and `generate` to *complete* the first 32 tokens of that passage with each setting. Compute the per-token cross-entropy between the model's continuation and the actual continuation (treating the actual continuation as ground truth). The headline observation: greedy gets a *lower* per-token CE, but the output is more obviously canned; sampled output is more varied but matches the ground truth less precisely. This tension is one of the reasons "evaluating LMs is hard" — perplexity-style metrics reward greedy-style decoding; humans prefer sampled-style.

## Pitfalls to expect

- **Masking with `-1e9` instead of `-inf`.** Most of the time it works — `softmax(-1e9) ≈ 0` to many decimal places — but it's a small nonzero probability that, summed across thousands of dropped long-tail tokens, becomes a real bias. Use `float('-inf')`. The test suite's exact-zero-after-softmax check will catch most cases.

- **Top-p off-by-one.** A common bug writes the rule as "drop everything once cumulative > p" — but that masks the **first** token to cross `p`, including the argmax when the argmax alone has probability `> p`. The fix: shift the mask right by one, and always set `mask[..., 0] = False` so the highest-probability token is unconditionally kept. `test_top_p_filter_argmax_always_survives` catches this.

- **Warping in probability space instead of logit space.** A rewrite that softmaxes early and tries to apply temperature / top-k / top-p to probabilities will produce something *resembling* the right answer but with subtle scale differences and tricky renormalization at every step. The convention is rigid: stay in logit space until the final softmax. Every warper takes logits and returns logits.

- **Wrong warper order.** The canonical order is `repetition_penalty → temperature → top_k → top_p → softmax`. Switching `temperature` and `repetition_penalty` produces a visibly different penalty strength because temperature scales the penalty's effect. Switching `top_k` and `top_p` is usually harmless but produces a slightly different surviving set when both are active.

- **Forgetting to crop to `max_seq_len`.** `LearnedPositionalEmbedding` in Module 05 has a fixed-size table; calling it with `T > max_seq_len` either raises (in your bound check) or silently uses garbage positional vectors (if you forgot the bound check). Generate must pass at most `max_seq_len` tokens to the model, which means `ctx = full_ids[-model.max_seq_len:]` every step. `test_generate_respects_max_seq_len_via_cropping` catches this.

- **Slicing `logits[:, -1, :]` AFTER warping.** Warping the full `(B, T, V)` tensor and then slicing produces the same answer (the warpers are per-row) but does `T`-fold more work per step. Slice first.

- **Forgetting `@torch.no_grad()`.** Generation builds an autograd graph it never uses. On a long generation, the per-step memory cost climbs and you OOM at modest scale. Both correctness and speed are unaffected for short generations, which is why the bug hides until you scale up. The decorator is provided in the scaffold — don't remove it.

- **Re-seeding the generator inside the loop.** `multinomial(probs, 1, generator=gen)` advances `gen`'s state; calling `gen.manual_seed(seed)` inside the loop resets it to the same state every step, and you draw the same "random" token every step. Seed exactly once, OUTSIDE the loop.

- **Looping forever on a model that won't emit `eos_id`.** `eos_id` is opt-in; if the trained model never learned that particular id is end-of-sequence, generation runs the full `max_new_tokens`. `max_new_tokens` is your fallback stop — always set it, even when `eos_id` is given.

- **Greedy producing nonsense on your tiny model.** Greedy decoding exposes the model's loops more aggressively than sampled decoding. If `T=0.0` produces garbage on your TinyShakespeare model, the model itself is probably the issue — train it longer, or just accept that small models decode poorly with `T=0.0` and use sampled decoding. (Greedy on GPT-2-class models is notably better; greedy on 100k-param models is notably worse.)

- **Repetition penalty applied to the WHOLE prompt.** The penalty reads `full_ids`, which by default is `prompt + sampled_so_far`. Long prompts with high `penalty` therefore penalize many tokens for the entire generation. If your model just refuses to repeat *anything* in the prompt, dial `penalty` toward `1.0`, or maintain a separate "recent-only" history.

## Reading

Primary:

- **Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020).** The top-p / nucleus-sampling paper. The figures comparing greedy / pure-random / top-k / top-p output are still the clearest argument for adaptive cutoffs. Read once start to finish.
- **Fan, Lewis, Dauphin, "Hierarchical Neural Story Generation" (2018).** The top-k paper, in the context of story generation. Older but worth reading; introduces the diversity-vs-quality framing that this whole module is about.
- **Keskar et al., "CTRL: A Conditional Transformer Language Model for Controllable Generation" (2019).** §4.1 has the repetition-penalty formula we use.

Secondary:

- **Karpathy, nanoGPT** (GitHub, `model.py::generate`). 30 lines of reference Python; the loop is structurally identical to ours, with AdamW-trained weights and no top-p. Reading it after writing your own is illuminating.
- **HuggingFace `transformers` `generation/logits_process.py`.** The reference implementation that essentially everyone copies. Has every warper variant in one place; the canonical order (`LogitsProcessorList`) is what we follow.
- **Su, Cao, Lin, "A Contrastive Framework for Neural Text Generation" (2022, contrastive search).** A more recent decoding method that interpolates between argmax and a degeneration penalty. Argued to outperform top-p in some setups; not widely adopted yet but worth knowing about.

Optional:

- **Meister, Pimentel, Wiher, Cotterell, "On Decoding Strategies for Neural Text Generation" (2022).** A thorough empirical comparison of every popular decoding method. Useful if you want a survey before going deeper.
- **The "typical sampling" paper (Meister et al., 2023).** A variant on top-p with somewhat different theoretical motivation. Marginal practical differences; included for completeness.

## Deliverable checklist

- [ ] All tests in `tests/test_sampling.py` pass.
- [ ] Notebook: `notebooks/11-sampling-playground.ipynb`. Load your Module 10 TinyShakespeare checkpoint and run exercise 1 (temperature sweep) and exercise 2 (top-k vs top-p comparison). Commit the notebook with the outputs visible.
- [ ] Notebook: `notebooks/11-repetition-penalty.ipynb`. Run exercise 3 (type-token ratio under varying penalty); commit with the plot rendered.
- [ ] Interactive CLI playground from exercise 4 in `scripts/` or embedded in the playground notebook.
- [ ] You can explain — out loud, without notes — the diversity-vs-quality tradeoff and where each warper sits along that axis.
- [ ] You can explain — out loud, without notes — why the argmax always survives both top-k and top-p, and what bug each pattern catches.
- [ ] You can explain — out loud, without notes — the eight-step decode loop, with cropping, and what breaks if you reorder it.

## M-series notes

Sampling is inference-only. Every test in this module runs in well under a second; even the `generate` tests with a real (tiny) TransformerLM finish in milliseconds because there's no backward pass.

- **Test suite runs in <1s on CPU.** No need for MPS at test time.
- **Exercise 1 (temperature sweep, 6 × 200 tokens):** roughly 10–30 seconds on MPS, 30–90 seconds on CPU with a 1M-param TinyShakespeare model. Trivial.
- `generate` should run the model forward on `model.device` while keeping the growing token ID sequence on CPU. That keeps decoding and `torch.multinomial(..., generator=...)` simple, and avoids handing Matplotlib/tokenizer code MPS tensors later.
- **Exercise 4 (interactive playground):** every prompt feels instantaneous. The `O(T_ctx²)` attention cost is negligible at context lengths up to a few hundred.
- **At Module 16's scale (a 7–8B pretrained model, post-pivot)**, inference becomes the dominant cost and KV caching becomes essential — but that's three modules away. For Module 11, your Module 10 checkpoint is small enough that the unoptimized loop is fine.
- **`@torch.no_grad()` on `generate` is a real win on long generations.** Without it, autograd graphs accumulate in memory per step and OOM at moderate `max_new_tokens × T_ctx` products. The decorator costs nothing on short generations and saves you on long ones — keep it on.

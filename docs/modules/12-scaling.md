# Module 12 — Scaling experiments

> **Question this module answers:** *What gets better with size, and how cleanly does it scale?*

![Three tiny GPTs trained on the same TinyShakespeare corpus, side by side: 1M / 5M / 20M parameters. Each column shows the model's parameter count up top, its final validation perplexity in the middle (e.g. 6.5 → 4.8 → 4.0), and a 100-token continuation from the same prompt at the bottom. The 1M sample is locally-correct but largely nonsense; the 5M sample has short coherent phrases; the 20M sample has multi-line dialogue-like text. A panel below plots final loss vs parameter count on log-log axes — the three points fall close to a straight line whose negative slope is the empirical scaling exponent. A small caveat box on the right notes "iso-step, not iso-FLOP — the 20M model has consumed roughly 20× more compute per step." A second caveat box reminds the reader that "perplexity ≠ capability" — even at 20M, the model still confidently invents quotes, and several "emergent" tasks the larger model passes are actually still random chance.](12-scaling/Module12-Hero.png)

Same corpus, same step budget, different parameter counts, *very* different outputs. Scaling laws tell us how model quality varies with more or fewer parameters (and compute).  Module 12 is the first module without new package code. You're not implementing a new layer or training utility — you're running the trainer from Module 10 different sizes

---
## Prerequisites
### Math

- **Power laws on log-log axes.** A relationship `y = A · x^α` plots as a straight line on log-log axes with slope `α`. 

### Computer science

- **FLOPS** - Floating point operations. A way to standardize the measure of total compute expended between different models and training runs. 

### Programming

- **Matplotlib** Used in the exercise notebook for charting scaling laws.

---
## Where this fits in

After Module 10 you trained a single tiny GPT and sampled from it. After Module 11 you have decoding controls. Both modules let you study one model — but the *interesting* questions about LLMs are inherently comparative. Why is GPT-4 dramatically better than GPT-2? Is it just more parameters? Is it more parameters spent in a particular way? Is there a regime where adding parameters stops helping?

The honest answer to these questions is empirical: people trained models at many sizes, plotted the result, and found a remarkably clean pattern. Loss falls as a power function of parameter count. It also falls as a power function of dataset size. And — separately, and surprisingly — it falls as a power function of *compute* if you optimally trade off model size and dataset size at each compute budget.

```
   Validation loss
        ▲
        │  ●  (1M)
        │
        │      ●  (5M)
        │
        │           ●  (20M)
        │
        │
        │
        └──────────────────────────► params
        (log scale)

   On log-log axes, the three dots ought to fall close to a straight
   line with negative slope. The slope IS the scaling exponent.
```

The Kaplan paper (2020) made this concrete. The Chinchilla paper (2022) corrected an important methodological flaw in Kaplan's setup and produced a different exponent — and a now-famous rule of thumb: for compute-optimal pretraining, train roughly 20 tokens for every parameter. (A 70B-param model wants ~1.4T tokens of training data; a 1B-param model wants ~20B.)

You won't reproduce these exponents at MacBook scale. You'll see *the same shape* — loss falls as you add parameters, in a way that looks linear on log-log — and you'll see a slightly different exponent because (a) the optimizer is different, (b) the dataset is tiny, (c) you have only three points. The lesson isn't "I matched Chinchilla's slope." The lesson is "scaling has a shape, and that shape is not subjective."

The other empirical observation is harder to capture in a power law: some capabilities show up *suddenly* with size. The 1M model can't generate a single grammatically-correct sentence; the 5M model can; the 20M model can sometimes hold a thought across two sentences. Whether you call this "emergence" or "a metric artifact" is the BIG-bench debate; either way, the qualitative gap between sample texts at different sizes is the headline observation of the module — and the part you have to *read*, not plot.

## The big idea

Model capabilities follow a power law. The scaling laws claim: validation loss `L(N)` falls as a power of parameter count `N`, with a non-zero floor:

```
   L(N)  =  α · N^(−β)  +  L∞
```

`α` is a constant, `β` is the scaling exponent (positive — larger means loss falls faster as `N` grows), and `L∞` is the irreducible loss (the entropy of the data — the loss a perfect model would still achieve because language is inherently uncertain). Kaplan reported `β ≈ 0.076` for transformer LMs; Chinchilla — using a different methodology — argued the exponent depends on whether you're varying `N` at fixed dataset size, fixed compute, or fixed token count.

Three things to internalize about this curve:

```
   loss
    ▲
    │ ●
    │  \
    │   ● ●
    │       \●●
    │           ●●●●● ──── L∞ ── (irreducible loss; language entropy)
    │
    └──────────────────────────────► params (log scale)

      tiny       small      medium      large

   Big jumps early (going 1M → 10M is huge), diminishing returns
   later (going 100B → 1T is barely visible), but the curve has
   no obvious "wall" — adding parameters keeps helping until you
   hit L∞.
```

  * **The shape is universal across architectures.** RNNs, transformers, MoE — they all show power-law loss in `N`, with somewhat different exponents and intercepts. This is not a transformer-specific fact.
  * **Diminishing returns are real, but slow.** Going from 100B to 1T parameters has visibly less effect on benchmark performance than going from 1B to 10B. But "less effect" is not "no effect" — you're still climbing the curve.
  * **The floor `L∞` is set by the data.** Every fixed dataset has an irreducible per-token entropy below which no model — of any size — can go without overfitting. The interesting question for any deployment is "how close to `L∞` are we, and is that close enough?"

At our scale, fitting `L(N) = α · N^(−β) + L∞` to three points is hopeless — you have three numbers and three free parameters, and `L∞` is essentially unidentifiable without more points. That's fine. The exercises just fit `log L = log α − β · log N` (a straight line on log-log axes), reading `β` off the slope. Treat the resulting number as a sanity check on direction, not a publishable exponent.

### Iso-step versus iso-FLOP

Two ways to compare three models, and they give different answers:

```
                  ┌─────────────────────────────────────┐
                  │  Two equal-effort budgets, two      │
                  │  different stories                  │
                  └─────────────────────────────────────┘

   ISO-STEP                          ISO-FLOP
   (every model trains              (every model trains until
    for 5000 steps)                  it has used 6·N·T_total
                                     ≈ same FLOPs)

      1M:  5000 steps                  1M:  20,000 steps
      5M:  5000 steps                   5M:   4,000 steps
     20M:  5000 steps                  20M:   1,000 steps

   What you measure:                What you measure:
   "How well does this size         "What size of model gets
    do, given an equal              the most out of this much
    training-loop budget?"          compute?"

   Larger model wins (more          Often a SMALLER model wins
   capacity, same dataset           — Chinchilla's finding —
   passes).                         because larger models with
                                    too few tokens are
                                    under-trained.
```

The two comparisons answer different questions. **Iso-step** answers "more parameters = better, all else equal?" — the answer is yes, monotonically. **Iso-FLOP** answers "if I have a fixed compute budget, what size should I train?" — the answer at large scale is "a smaller model than you'd think, trained for longer."

![Iso-step vs iso-FLOP side by side as two laps-around-the-track experiments. Panel A (iso-step): every model — 1M, 5M, 20M — runs the same 5000 steps; compute per step grows ~6·N·T_step, so the 20M model burns ~20× more compute total than the 1M one. Same dataset passes, more capacity wins, larger model lower loss. Panel B (iso-FLOP): every model gets the same total compute budget (~7.7e13 FLOPs in the example), so the 1M model gets 75,000 steps, the 5M model 15,000, and the 20M model only 1000. Tokens-seen totals shift accordingly (~30M / ~30M / ~3M). The smaller models do many laps; the 20M model does one short lap and is undertrained. A "When to use which" panel pins the choice: iso-step when comparing capacity at fixed dataset, iso-FLOP when comparing what size to train under a real compute constraint. The Chinchilla intuition — smaller-but-longer beats larger-but-shorter on a fixed budget — falls out of panel B.](12-scaling/Module12-IsoFlop.png)

*Both comparisons appear in this module: exercise 1 is iso-step (the headline log-log plot), exercise 5 is iso-FLOP (the same plot, often with the slope reversed near the largest size). Reading the same three checkpoints under both lenses is the cleanest way to internalize that "scaling laws" is not one curve — it's at least two, and which one matters depends on whether your real-world constraint is dataset size or compute.*

For this module's *primary* exercise, iso-step is the right comparison. It's simpler to reason about and is what Kaplan-style "loss vs N at fixed dataset" plots show. The iso-FLOP comparison appears as exercise 5 — and at toy scale, it tends to show iso-step's monotonicity *flip* near the largest size (the 20M model with too-few tokens is under-trained relative to the 5M model with sufficient tokens, exactly as Chinchilla predicts).

### Counting parameters and FLOPs

You'll need rough parameter and FLOP counts to plan runs. The exact formulas:

```
  Per transformer block:

      attention projections (Q, K, V, output):  4 · D²
      FFN (D → 4D → D):                         8 · D²
      LayerNorm (×2):                           4 · D    (small)
      ──────────────────────────────────
      block total:                              ≈ 12 · D²


  Whole model:

      token embeddings:        V · D
      positional embeddings:   max_seq_len · D
      L blocks:                L · 12 · D²
      final LayerNorm:         2 · D
      unembedding head:        D · V
      ──────────────────────────────────
      total:                   ≈ 2 · V · D  +  L · 12 · D²


  Reference points (V = 2048):

      D = 128, L = 4    →   ≈ 525K + 786K   ≈    1.3M
      D = 224, L = 6    →   ≈ 917K + 3.6M   ≈    4.5M
      D = 384, L = 10   →   ≈ 1.5M + 17.7M  ≈   19.2M
```

![Anatomy of a transformer's parameter count. Inside one block: self-attention's four projections (Q, K, V, output) total 4·D²; the feedforward network with 4D inner width totals 8·D²; LayerNorms and biases are negligible — block total ≈ 12·D². Whole-model breakdown: 2·V·D for token + unembedding, max_seq_len·D for positional embeddings, L·12·D² for the stack, plus a final LayerNorm. A pie chart showing where the parameters actually live — for typical D and L, the FFN is ~65–75% of the total, attention projections are ~25–35%, and embeddings shrink to a few percent as the model grows because they scale linearly in D while the blocks scale quadratically. A "how parameters scale" panel pins the rules: doubling D quadruples per-block params (D² scaling), doubling L doubles total block params (linear scaling). Three example configurations (1M / 5M / 20M) show the formula evaluated end-to-end and matched against `sum(p.numel() for p in model.parameters())`.](12-scaling/Module12-ParamAnatomy.png)

*The headline takeaway is the pie chart: parameters live in the FFN, mostly. This is why "make the FFN inner width 4·D" is treated as a design constant in almost every transformer paper — the FFN is already the bulk of the model. It's also why the depth-vs-width exercise (#4) is worth running: at fixed parameter budget, you're trading the number of refinement layers against the per-layer FFN width, and at our scale the depth direction wins by a few percent.*

For FLOPs:

```
  FLOPs per training step ≈ 6 · N · T_step
                              where T_step = batch_size · context_length

  Total FLOPs ≈ 6 · N · steps · batch_size · context_length
```

The factor 6 covers forward (≈ 2·N·T_step matmul flops) plus backward (≈ 4·N·T_step). It's a standard approximation that ignores attention's `T²` term (small at our context lengths) and a few smaller matmuls; expect ~10–20% accuracy.

### Emergent capabilities and the BIG-bench debate

Some capabilities show up *suddenly* as a model scales: at one size the model gets ~0% on a task, then at the next size it gets ~30%. This was the claim of Wei et al. (2022, "Emergent Abilities of Large Language Models"). The list included three-digit arithmetic, multi-step word problems, instruction following.

Schaeffer et al. (2023, "Are Emergent Abilities of Large Language Models a Mirage?") replied that most published "emergence" plots were artifacts of the *metric*, not the model: tasks were scored exact-match (0 or 1) on outputs that the smaller model also gradually learned but never quite finished. Switch the metric to a continuous one (token-level edit distance, partial-credit on multi-step problems) and the curves smooth out into the same power-law shape as loss.

The honest summary as of 2026: both papers are correct about different things. Many "emergent" tasks are smooth under better metrics — the Schaeffer critique landed. But some capabilities — like in-context learning of arbitrary new patterns — really do appear to show genuine threshold behavior even on smoothed metrics, with sharp transitions at scale. The community moved on from "emergence vs not" to "what fraction of progress is smooth and how should we measure each task."

![Smooth improvement vs threshold metrics, side by side. Top panel: an underlying skill (the model's competence on a task) improves smoothly and continuously as model scale grows — the curve is the kind of thing cross-entropy / log-likelihood metrics actually measure. Middle panel: the SAME underlying smooth improvement, viewed through an exact-match accuracy metric, looks like a step function — flat near zero, sharp jump near the threshold, plateau near one. The smooth competence curve and the threshold curve are generated by the same underlying model behavior; only the scoring rule is different. Bottom panel: three regions explain why the threshold metric reads as "emergence." Below the threshold the model is improving but answers still look wrong — partial credit would catch this, exact-match scores zero. Near the threshold tiny smooth gains carry many examples across the line at once. Past the threshold further improvement is invisible because the score has saturated. A "key takeaway" pins the lesson: a sudden jump in a benchmark score does not necessarily mean the model learned something discontinuously — it often means the benchmark itself imposes a threshold.](12-scaling/Module12-Smooth.png)

*This is the visual core of the Schaeffer-vs-Wei debate. Both authors are looking at the same underlying training dynamics; the disagreement is at the metric layer, not the model layer. At MacBook scale you can't reproduce the published threshold plots, but you can reproduce the principle in miniature — exercise 6's calibration check is a continuous metric you can run on all three sizes, and it improves smoothly even on tasks where greedy-argmax accuracy at exact-match scoring would jump.*

At MacBook scale, you won't see clean threshold behavior. You'll see *qualitative* threshold behavior — exercise 3 has you find a task where the smallest model is hopeless and the largest is just barely able to do it. Treat this as the toy version of the Wei observation, not as a cleanly emergent capability.

### Evaluation matters

The other Schaeffer-style observation, applicable at every scale:

  * **Cross-entropy is *the* smooth metric.** Validation loss falls as a power of `N` because cross-entropy is a soft, log-likelihood-based quantity that improves continuously as the model puts more probability mass on the correct token.
  * **Most downstream metrics are not smooth.** Exact-match accuracy on multiple choice, BLEU on translation, pass-rate on a gated coding task — these have "thresholds" baked in. A model that's 60% sure of the right answer scores zero; the same model that's 51% sure scores zero too.
  * **Sample quality is the most-honest qualitative metric.** Read the output. Decide whether it's better. If the larger model's samples make obvious step-function jumps in coherence (sentence-level → paragraph-level → multi-paragraph), report that — even if your perplexity plot is smooth.

Exercise 6 is a calibration check: how well-calibrated is each model's predicted distribution? Calibration generally improves smoothly with size — and the improvement is usually larger than the loss improvement suggests, because a smaller model is both *less accurate* and *more confidently wrong*.

## Concepts to internalize

- **Loss falls as a power of size.** Not exponentially, not linearly — power-law. On log-log axes the curve is roughly straight.
- **There's an irreducible floor.** No matter how big the model, validation loss can't fall below the per-token entropy of the data. (Practically: you'll never reach it at our scale; published GPT-class models are still many percent above their estimated `L∞`.)
- **Iso-step ≠ iso-FLOP.** Equal step counts give the larger model more total compute. Equal compute gives the smaller model more passes over the data. Both comparisons are valid; they answer different questions.
- **Parameters live in the FFN (mostly).** A transformer's FFN is `~⅔` of its parameter count; attention is `~⅓`. Doubling depth or width both grow params; doubling depth grows them linearly, doubling width grows them quadratically.
- **Emergent ≠ magic.** "Capability X appears at size Y" is mostly a metric artifact at small scale; it may or may not be a real threshold at large scale; you can't determine which one with three data points.
- **At toy scale, expect the wrong slope.** Published scaling exponents (Kaplan: ≈ 0.076; Chinchilla: ≈ 0.336 for fixed-FLOP) come from fits over orders of magnitude in parameters using AdamW. Your three SGD-trained TinyShakespeare points won't match these numbers. The *direction* (down and to the right) will match.
- **The qualitative gap between samples is the lesson.** Read the 1M, 5M, and 20M continuations of the same prompt out loud. The structural difference is what you're here to see; the perplexity plot is just the quantitative summary.

### What we didn't cover

- **Mixed precision (fp16, bf16) on MPS.** Introduces silent op-fallback edges and changes loss curves enough to confuse a scaling experiment. Stay in fp32 for this module. Module 16 returns to inference-time precision.
- **Theoretical derivations of the scaling exponents.** The Kaplan paper has them; Hoffmann's Chinchilla paper revisits them with cleaner methodology. They're worth reading, but the exponent itself is not a deep theoretical object — it's an empirical fit and the value at toy scale will not match published large-scale numbers.
- **Inverse scaling (`Inverse Scaling Prize`, McKenzie et al.).** A small but real set of tasks where larger models do *worse*. Genuinely interesting, but our smallest model is too small for this to manifest. Skim later if curious.
- **The full BIG-bench debate.** Wei et al. claimed several capabilities are "emergent" (sharp threshold-style improvements). Schaeffer et al. argued most "emergence" is an artifact of the metric (multiple choice with a hard-edge scoring rule). Both papers are listed in *Reading*; the debate is worth understanding qualitatively but you won't reproduce it at our scale.

---
## Setting up the experiments

This module has **no new package code**. The deliverable is a notebook (or a small set of notebooks) that runs three TransformerLM trainings at different sizes against the Module 10 corpus and produces the headline plot.

### Three reference configurations

These configs are calibrated for ~1M, ~5M, ~20M params at vocab size 2048 (your Module 04 BPE tokenizer). Adjust `vocab_size` if your tokenizer is different — the formulas above tell you how.

```python
from g2c.transformer import TransformerLM
from g2c.pretraining import Trainer

CONFIGS = {
    "1M":  dict(embedding_dim=128, num_layers=4,  num_heads=4),
    "5M":  dict(embedding_dim=224, num_layers=6,  num_heads=4),
    "20M": dict(embedding_dim=384, num_layers=10, num_heads=8),
}

def build(config_name, vocab_size, max_seq_len=128):
    cfg = CONFIGS[config_name]
    return TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **cfg,
    )

def total_params(model):
    return sum(p.numel() for p in model.parameters())
```

### A standard training recipe

For iso-step comparisons, hold *everything* except the architecture constant:

```python
def make_trainer(model, max_steps=5000):
    return Trainer(
        model,
        batch_size=32,
        context_length=128,
        max_steps=max_steps,
        max_lr=3e-3,
        warmup_steps=100,
        grad_clip=1.0,
        eval_every=200,
        eval_iters=20,
        log_every=50,
    )
```

`max_lr=3e-3` is a Module-10-validated default for SGD. At larger scales you may need to lower it (the 20M model can destabilize at this rate); a small per-size lr sweep is one of the exercises.

### Checkpointing and logging

A 20M-param run takes ~1–3 hours on MPS. Don't lose it:

```python
import torch
import json

def save_run(name, model, history):
    torch.save(model.state_dict() if hasattr(model, 'state_dict')
               else {p_name: p.detach().clone()
                     for p_name, p in zip(range(len(model.parameters())),
                                          model.parameters())},
               f"runs/{name}.pt")
    with open(f"runs/{name}.json", "w") as f:
        json.dump(history, f)
```

(`g2c`'s hand-rolled `Module` doesn't ship `state_dict` — the dict-comprehension form above is sufficient. Module 10 exercise 7 had you build a checkpoint helper; reuse it if you did.)

The Trainer's `train()` returns a `history` dict with per-log-step entries `{step, lr, loss, grad_norm, val_loss}`. That dict is what your headline plot reads from.

### How long these runs take

Rough M-series budget at the reference configs and `max_steps=5000`:

```
   ┌───────┬─────────────┬─────────────┬──────────────┐
   │ size  │  M1 / M2 8GB │  M2 Pro 32GB │  M3 Max 64GB │
   ├───────┼─────────────┼─────────────┼──────────────┤
   │  1M   │    ~5 min    │    ~3 min    │    ~2 min    │
   │  5M   │   ~30 min    │   ~15 min    │    ~8 min    │
   │ 20M   │   ~3 hours   │    ~1 hour   │   ~30 min    │
   └───────┴─────────────┴─────────────┴──────────────┘
```

Numbers are wide ballparks — context length, vocab, and Mac generation all matter. If your 20M run is on track to finish overnight rather than over coffee, halve `max_steps` (and note this in the writeup) rather than abandoning the comparison.

## Exercises

1. **Train three sizes, plot scaling.** Build `1M`, `5M`, `20M` per the configs above, train each for 5000 steps with the standard recipe, and record final val loss + perplexity. Plot `log(val_loss)` vs `log(params)` — three points and a line of best fit. Eyeball the slope. Report:

     - The three perplexities.
     - The slope `β` you read off (will likely be in the 0.05 to 0.15 range, with high uncertainty).
     - One paragraph on what surprised you.

   This is the headline deliverable. Everything else builds on it.

2. **Read the samples out loud.** From each of your three checkpoints, generate 200 tokens at `temperature=0.8, top_p=0.9` from the same prompt with the same seed. Print all three side by side. Don't grade them on a metric — *read them*. The qualitative shape: 1M is locally-correct gibberish, 5M has short coherent phrases, 20M has paragraph-scale coherence. You're looking for the gap between sentence-level and discourse-level structure to *appear* somewhere between 5M and 20M; whether it appears that cleanly on your specific data and seed is the experimental observation.

3. **Find a threshold task.** Construct a prompt that the 1M model is hopeless on but the 20M model can just barely handle. Examples that have worked at this scale:

     - **A specific named character interacting.** "ROMEO. To be honest with you, mistress Juliet, I —" The 1M model rarely follows the format; the 20M model often does for one or two lines.
     - **A two-step instruction.** "Write the word 'fire' three times, then write 'water' twice." The 1M model can't count; the 20M model sometimes can.
     - **A rhyming requirement.** "Compose a couplet whose lines end in 'night' and 'light'." The 1M model produces unrelated text; the 20M model occasionally produces a couplet (without rhyme — that's much harder).

   Document one threshold task. Sample it 10× per model size at fixed temperature; report a hand-rated success rate (yes / partial / no) for each model. The expected pattern: 1M = 0/10, 5M = 1–2/10, 20M = 3–6/10. Treat anything cleaner than that with skepticism — at this scale, single-seed variance can dominate.

4. **Depth vs width at fixed parameter count.** At ~5M params, three configs are roughly equal:

     - `D=224, L=6, H=4` (the reference 5M)
     - `D=192, L=8, H=4` (deeper, narrower)
     - `D=288, L=4, H=4` (shallower, wider)

   Train all three at iso-step and iso-recipe. Compare final val loss. Real-world result (matched in nanoGPT-scale experiments): **deeper, narrower wins** at our scale, by a few percent. The wider/shallower model has more attention "communication bandwidth" per layer but fewer layers of refinement; small-LM language modeling pays for refinement.

5. **Iso-FLOP, three sizes.** Repeat exercise 1, but instead of fixing `max_steps=5000` for each, fix the total FLOP budget. With FLOPs ≈ `6·N·steps·B·T`:

     - 20M model: `max_steps=2000` (~7.7e13 FLOPs)
     - 5M model: `max_steps=8000` (~7.7e13 FLOPs)
     - 1M model: `max_steps=40000` (~7.7e13 FLOPs)

   Plot the same `log(val_loss)` vs `log(params)`. The expected pattern *flips* somewhere in the 5M–20M range: the smaller models, given more passes over the data, are now closer to compute-optimal than the 20M one which is under-trained. This is the toy reproduction of Chinchilla's finding. (The 1M model with 40k steps is itself probably overtrained — you'll see val loss start to rise on TinyShakespeare. Read the curve, not just the endpoint.)

6. **Calibration check.** For each of the three size-1M / 5M / 20M checkpoints, compute the **expected calibration error** (ECE) on a held-out passage:

   ```python
   import torch.nn.functional as F

   def ece(model, val_ids, n_bins=10):
       # Compute model probs over (B, T, V); compare predicted prob of
       # the actual token to whether that token was the argmax.
       ...
   ```

   The pattern you should see: smaller models are *more confident than they should be* — they put high probability on a single token even when the true continuation is uncertain — so their ECE is high. Larger models are more calibrated; the gap between predicted prob and empirical accuracy is narrower. This is part of why "perplexity falls as size grows" understates how much *better* a larger model actually is — it's both more accurate AND more honest about what it doesn't know.

7. **Optional: a fourth, larger point.** If you have the disk and the patience, train a `D=512, L=12, H=8` config (~50M params) for 5000 steps. Plug it into your plot from exercise 1 and re-fit the slope. Four points is still not many, but it doubles your evidence for the power-law shape. Time budget: roughly 4–8 hours on MPS depending on your Mac. Skip if your training environment is not stable enough for an overnight run.

8. **Optional: vocab-size sweep at fixed `D · L`.** Tokenize TinyShakespeare with three vocab sizes — 512, 2048, 8192 — and train the same architecture (~5M params) on each. Vocab affects both `N` (via `2·V·D`) and the per-token entropy of the dataset. The expected pattern: per-token val loss is *not* the right comparison across vocab sizes (smaller vocab → higher per-token entropy by construction), but `bits-per-character` is invariant. Compute `bpc = val_loss * tokens_per_char / log(2)` for each tokenizer and compare. This is harder to interpret cleanly, which is itself the lesson — "perplexity" is not a vocab-independent quantity.

## Pitfalls to expect

- **Single-seed conclusions.** A 5M model trained from one seed can land 5–10% higher or lower than the seed-mean, especially with SGD. Two or three seeds per configuration is the floor for any quantitative claim. If you see "20M is *worse* than 5M" on a single seed, run another seed before believing it.

- **Lr too high for the largest model.** The recipe of `max_lr=3e-3` is calibrated for ~1M params with SGD. At 20M params, that lr can produce occasional grad-norm spikes that the gradient clip mostly contains but that nevertheless degrade final loss. If your 20M run looks unstable (loss curve has spikes, val loss is higher than the 5M model's), drop `max_lr` to `1e-3` and re-run. Larger models are usually more — not less — sensitive to lr at fixed schedule.

- **Forgetting to crop training context to `max_seq_len`.** `Trainer.context_length` must be ≤ `model.max_seq_len`. If you ramp `max_seq_len` for the larger configs and forget to update the trainer's `context_length`, the larger models train on the same window length and you measure something different from what you think.

- **Comparing across seeds, batch sizes, lr schedules.** A clean comparison fixes seeds, batch size, context length, lr, schedule, optimizer, and dataset across runs. Vary *only* the architecture. If one knob slips between runs, the resulting plot answers a different question — sometimes silently.

- **Cross-vocab comparisons.** Perplexity at vocab size `V₁` and perplexity at vocab size `V₂` are not directly comparable. A smaller vocab means each token carries more information, so per-token loss is higher *by construction*. Stick with one tokenizer for the main comparison; if you want to vary vocab, switch to bits-per-character (exercise 8).

- **Running out of memory on the 20M model.** A 20M model at `(B=32, T=128)` produces a `(32, 128, 2048)` logit tensor (~32 MB just for logits; activations bring the total to ~600 MB during backward). If your Mac has 8–16GB unified memory, also running a browser, also running VS Code, may push you over. Halve `batch_size` to `16` before halving `T`; smaller `T` damages the model's ability to learn long-range dependencies.

- **Reading too much into the slope.** A power-law fit through three noisy points has wide error bars on `β`. Reporting `β = 0.082 ± 0.005` from three data points is overconfident; reporting "the slope is roughly 0.08" is honest. The published Kaplan exponent (`0.076`) has many tens of points behind it.

- **MPS giving slightly different loss curves than CPU.** Floating-point ops on MPS are not bit-identical to those on CPU; over 5000 steps the divergence accumulates. Don't expect MPS and CPU runs from the same seed to produce the same final loss — the difference is usually in the 4th significant figure, but spotting that the divergence exists at all has tripped up scaling-experiment writeups.

- **The 1M model's val loss plateau being at the wrong level.** TinyShakespeare's per-token entropy under a 2048-vocab BPE is around `log(20–40) ≈ 3.0–3.7` nats, depending on the exact split. If your 1M model plateaus near `4.5+`, it's underfit for the budget; if it plateaus near `3.0`, it's at the irreducible floor and the 5M model can't improve much over it. Read the absolute value, not just the relative gap.

- **Mistaking convergence for a finished run.** A 20M model at 5000 steps has not converged on TinyShakespeare — its loss curve still has noticeable downward slope at step 4999. The "final" val loss you report is therefore a function of `max_steps`, not just architecture. Note this in the writeup. (The Chinchilla compute-optimal recipe says we'd want ~400M tokens of training for a 20M model; we're feeding it ~20M tokens, an order of magnitude short.)

- **Catastrophic seeds.** Occasionally a run lands on a bad initialization and never recovers — val loss plateaus 20–30% above where it should be. You'll see it as an outlier in your plot. Don't include it in the fit; rerun with a different seed and replace.

## M-series notes

This is the most compute-hungry week of the course.

- **Plan around the 20M run.** It is the constraint. On an M1/M2 base machine, a 5000-step 20M run is a 2–3 hour commitment; on an M3 Max with 64GB it's well under an hour. If your 20M run is queued for "overnight" rather than "next coffee break," halve `max_steps` and *report this* — it's a real result that the comparison was constrained by available compute.

- **MPS vs CPU.** The 1M model trains comfortably on either. The 5M and 20M models are MPS-essential — CPU runs at this size are an order of magnitude slower. `Trainer(..., device="auto")` is the default path; print `trainer.device` at the start of a run so you know whether you are actually on MPS.

- **Memory headroom.** The 20M model with `(B=32, T=128, V=2048)` peaks at roughly 600 MB during backward. Running another model in another tab adds the same. If you OOM, halve `batch_size` to `16` before touching `T`. Watch Activity Monitor's "Memory pressure" graph during training — yellow is OK, red means you're swapping and your training is silently CPU-bound for memory reasons.

- **Long runs and laptop sleep.** macOS aggressively sleeps the GPU when the laptop is on battery. Plug in for the 20M run. `caffeinate -di python train.py` at the command line prevents the system from sleeping during a notebook-launched script.

- **Storage.** Three checkpoints at 1M / 5M / 20M parameters take roughly 4MB / 20MB / 80MB on disk in fp32. The tokenized corpus is in the tens of MB. The training-history JSONs are kilobytes. Total: under 200MB for the headline experiment. The 50M optional run (exercise 7) adds another ~200MB. Comfortable on any modern Mac SSD.

---
## Reading

Primary:

- **Kaplan, Henighan, Brown et al., "Scaling Laws for Neural Language Models" (2020).** The original empirical scaling-laws paper. §3 has the power-law fits; §6 has the compute-optimal claim that Chinchilla later corrected. Read sections 1–3 carefully; skim the rest.
- **Hoffmann, Borgeaud, Mensch et al., "Training Compute-Optimal Large Language Models" (2022, "Chinchilla").** The follow-up to Kaplan with a different methodology and the "20 tokens per parameter" finding. The plots in §3 are the most-cited part of the paper. Read once start to finish — it's short.
- **Wei, Tay, Bommasani et al., "Emergent Abilities of Large Language Models" (2022).** The emergence paper. The figures showing "near-zero, near-zero, then 30%" curves on multiple tasks are the visual centerpiece of the discussion.
- **Schaeffer, Miranda, Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?" (2023).** The metric-artifact response. §3 reanalyzes the same tasks under continuous scoring rules and shows the curves smooth out.

Secondary:

- **Bahri, Dyer, Kaplan, Lee, Sharma, "Explaining Neural Scaling Laws" (2024).** A theoretical paper on why scaling laws have the form they do. Hard but worth knowing about — it argues power-law decay is structural, not coincidental.
- **Hestness et al., "Deep Learning Scaling is Predictable, Empirically" (2017).** Pre-Kaplan scaling-laws work, mostly from Baidu Research. Same shape; pre-transformer.
- **Henighan et al., "Scaling Laws for Autoregressive Generative Modeling" (2020).** Extends Kaplan to images, video, math. Same shape across modalities.

Optional:

- **The "BIG-bench" paper (Srivastava et al., 2022).** Source dataset for the emergence figures in Wei et al. Mostly useful as a list of tasks where "more parameters" maps to specific capability changes.
- **McKenzie et al., "Inverse Scaling Prize" (2023).** A small set of tasks where larger models do *worse*. Out of scope here, but conceptually important — scaling is not purely monotonic at the level of individual tasks.

## Deliverable checklist

- [ ] Three trained checkpoints at ~1M, ~5M, ~20M parameters, trained on the same corpus with the same recipe. Saved to disk; the run history (loss / lr / grad_norm) saved as JSON or CSV alongside.
- [ ] Notebook: `notebooks/12-scaling-runs.ipynb`. Loads the three checkpoints, prints `total_params(model)` for each (verifying they're roughly the target sizes), prints final val loss / perplexity for each, and runs `generate(temperature=0.8, top_p=0.9)` from the same prompt for each model. Three sample texts visible in the notebook output.
- [ ] Notebook: `notebooks/12-scaling-plot.ipynb`. The headline log-log plot of `val_loss` vs `params`. Three points + a fit line. The fitted slope and one-paragraph commentary visible.
- [ ] One-paragraph writeup of a threshold task (exercise 3) — the prompt, the success rates per size, and a sentence on what makes the task "threshold" rather than "smooth."
- [ ] You can explain — out loud, without notes — what the difference between iso-step and iso-FLOP is, and why they answer different questions.
- [ ] You can explain — out loud, without notes — why "loss falls as a power of N" is a *different* claim from "capability X emerges at N", and which of the two you can or can't see at MacBook scale.
- [ ] You can explain — out loud, without notes — what the irreducible loss `L∞` is and why no model — of any size — can fall below it.


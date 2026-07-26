# Probability & cross-entropy primer

Why is the loss `−log p`? What does a loss of 8.32 *mean*? Why does an untrained model always start at exactly `log(vocab_size)`, and why will the loss never reach zero no matter how long you train? The course leans on a small kit of probability and information-theory ideas from Module 03 onward, and this primer assembles the kit in one place.

The through-line: **a language model is a probability distribution, the loss measures how surprised that distribution is by reality, and every number on your loss curve is readable once you know what surprise means.**

## How to use this primer

Read sections 1–5 before Module 06 (where next-token loss first appears); they make Modules 09B and 10's loss curves legible rather than mystical. Section 6 — the gradient derivation — is best read alongside Module 03. Sections 7–8 are back-reference material for reading your own training runs.

---

## Contents

1. [A model is a distribution](#distribution)
2. [Log probabilities and surprisal](#surprisal)
3. [Entropy, and the log(V) baseline](#entropy)
4. [Cross-entropy: the loss itself](#cross-entropy)
5. [KL divergence: the excess](#kl)
6. [Softmax, and the (p − y) gradient](#gradient)
7. [Perplexity](#perplexity)
8. [Reading a loss number](#reading)
9. [Pitfalls](#pitfalls)

---

## <a id="distribution"></a>1. A model is a distribution

At every position, a language model outputs one number per vocabulary entry, and after softmax those numbers form a **categorical distribution**: `V` probabilities, each in `[0, 1]`, summing to exactly 1.

```
p = model("Once upon a")     # p : (V,)    p[i] ≥ 0,   Σ p[i] = 1
p["time"] = 0.61
p["midnight"] = 0.007
p["torque"] = 0.0000002
...
```

That's the entire output contract — the model never says what the next token *is*, only how its belief is spread across all `V` options. Everything else in this primer is about scoring such a belief against what actually came next. (Conditional notation, if you want it: the model computes `p(next token | context)`, a different distribution at every position.)

---

## <a id="surprisal"></a>2. Log probabilities and surprisal

Probabilities of sequences multiply: `p(sentence) = p(w₁) · p(w₂|w₁) · p(w₃|w₁w₂) ⋯`. Products of many numbers below 1 underflow float32 within a few dozen tokens, and they're miserable to differentiate. So everything runs in log space, where products become sums:

```
log p(sentence) = Σ_t  log p(w_t | w_<t)
```

The negated version has a name and an interpretation. The **surprisal** of an event with probability `p` is

```
surprisal = −log p
```

- Certain event (`p = 1`): surprisal 0 — no surprise.
- `p = 0.5`: surprisal ≈ 0.69.
- `p = 0.01`: surprisal ≈ 4.6.
- `p → 0`: surprisal → ∞ — the model said "essentially impossible" and it happened.

**The per-token loss in this course is exactly the surprisal of the true next token.** When Module 10's logs show `loss 2.74`, they're saying: on average, the model rated each actual next token about as likely as an event with probability `e^−2.74 ≈ 6.5%`.

One convention note: `log` here is the natural log, so surprisal is measured in **nats**. Information theory texts use log₂ and **bits** (1 nat ≈ 1.443 bits). PyTorch losses, and every number in this course, are in nats.

---

## <a id="entropy"></a>3. Entropy, and the log(V) baseline

Surprisal scores one outcome. **Entropy** is a distribution's *expected* surprisal under its own beliefs:

```
H(p) = Σ_i  p[i] · (−log p[i])
```

It measures how uncertain the distribution is. Two ends of the spectrum:

- All mass on one token: `H = 0`. The distribution already knows.
- Uniform over `V` tokens: every outcome has surprisal `log V`, so `H = log V` — the **maximum possible** entropy on `V` outcomes. Any deviation from uniform lowers it.

That maximum is the most useful constant in the whole course:

```
vocab      log V  (nats)
  4096      8.32          ← StoryLM's vocabulary
  8192      9.01          ← TinyLLM's vocabulary
 50257     10.82          ← GPT-2's, for scale
```

A freshly initialized model produces near-uniform logits, so its cross-entropy starts at `log V` — which is why Modules 09B and 10 treat `log V` as the sanity baseline. First-step loss well *above* `log V` means the model is worse than ignorant (usually an initialization or wiring bug); a loss that never leaves the neighborhood of `log V` means nothing is being learned. The single most diagnostic number on a loss curve is free: you can compute it before training starts.

---

## <a id="cross-entropy"></a>4. Cross-entropy: the loss itself

Entropy asks a distribution how surprised it expects *itself* to be. **Cross-entropy** asks: when reality is distributed as `p` but *you* predict `q`, what's your average surprisal?

```
H(p, q) = Σ_i  p[i] · (−log q[i])
```

In training, reality at one position is not a distribution but a fact: the next token was `t`. As a distribution that's a **one-hot**: all mass on `t`. The sum then collapses to a single term:

```
H(onehot(t), q) = −log q[t]
```

— the surprisal of the truth under the model's beliefs. Average it over positions and you have, exactly, the training loss of Modules 06 through 10:

```
loss = mean over positions of  −log q[true next token]
```

Two properties make it the right loss and not just a plausible one:

- **Minimizing it is maximum likelihood.** The sum of `−log q` over a corpus is the negative log-probability the model assigns to the whole corpus; minimizing the loss and maximizing the corpus probability are the same act.
- **It cares about calibration, not just ranking.** Accuracy would score "51% on the right token" and "99% on the right token" identically. Cross-entropy rewards the model for putting probability where the data goes — which is the actual job of a generative model, and why sampling from a CE-trained model works at all.

---

## <a id="kl"></a>5. KL divergence: the excess

Split cross-entropy into two parts:

```
H(p, q) = H(p) + KL(p ‖ q)
```

`H(p)` is reality's own entropy — the surprisal a *perfect* model would still incur, because language is genuinely uncertain ("Once upon a" doesn't determine the next token even in principle). `KL(p ‖ q) ≥ 0`, the **Kullback–Leibler divergence**, is the *excess* surprisal you pay for predicting `q` instead of `p`. It's zero exactly when `q = p`.

Two places this decomposition pays off in the course:

- **The loss floor (Modules 10 and 12).** Your loss curve flattens not because optimization failed but because it's approaching `H(p)` — the irreducible entropy of TinyStories under your tokenization. Only the KL term is compressible; scaling curves in Module 12 are plots of how model size buys KL down while the floor stays put. "Loss 0" was never on the table.
- **KL as a leash (Module 14).** DPO's β sets the strength of a KL penalty tethering the tuned policy to its reference model — "move toward the preferences, but stay within a bounded KL of who you were." When Module 14 calls β the KL-regularization strength, this is the KL it means.

---

## <a id="gradient"></a>6. Softmax, and the (p − y) gradient

The model actually emits unnormalized scores — **logits** `z : (V,)`. Softmax turns them into the distribution:

```
p[i] = e^{z[i]} / Σ_j e^{z[j]}
```

(One property used constantly: adding a constant to every logit changes nothing — the constant factors out of numerator and denominator. That shift-invariance is also the stability trick: compute `softmax(z − max(z))`, identical result, no `e^{800}` overflow.)

Now the derivation the course's loss rests on. The per-position loss with true token `t`, written directly in logits:

```
L = −log p[t] = −z[t] + log Σ_j e^{z[j]}
```

Differentiate with respect to one logit `z[i]`. The first term contributes `−1` if `i = t`, else 0. The second term — using `∂/∂z[i] log Σ e^{z_j} = e^{z[i]} / Σ e^{z_j} = p[i]` — contributes `p[i]`. So:

```
∂L/∂z[i] = p[i] − 1[i = t]        i.e.   ∂L/∂z = p − onehot(t)
```

**The gradient of softmax-plus-cross-entropy is `predicted minus actual`.** No softmax Jacobian, no chains of quotients — those all cancel. The error signal flowing back into the network is literally the vector of probability the model misallocated: positive gradient (push down) on every token that got mass, `p[t] − 1` (push up) on the truth. When the prediction is perfect, the gradient is exactly zero.

This cancellation only happens for the *fused* pair — which is why `F.cross_entropy` takes raw logits and why you never write `softmax(...).log()` (unstable *and* forfeits the clean gradient). Same fusion, same reason, in your own Module 03 loss and in `F.log_softmax + NLL`.

---

## <a id="perplexity"></a>7. Perplexity

Perplexity is cross-entropy pushed back through the exponential:

```
perplexity = e^{loss}
```

Its reading: the model is, on average, as uncertain as if it were choosing uniformly among `perplexity` options — an *effective branching factor*. It's the same information as the loss on a scale that's easier to feel:

```
loss (nats)   perplexity   meaning
   8.32          4096      uniform over StoryLM's vocab — knows nothing
   4.10            60      ~60-way effective choice   (StoryLM-1M territory)
   2.74            15.5    ~15-way                    (StoryLM-30M territory)
   2.56            12.9    ~13-way                    (StoryLM-5M territory)
```

Because `exp` is convex, average the *loss* over a corpus and exponentiate once at the end; averaging per-batch perplexities gives a different, wrong number.

---

## <a id="reading"></a>8. Reading a loss number

The skills the previous sections buy you, condensed into the checklist Modules 09B–12 assume:

- **At `log V` and staying there** → the model is outputting uniform; nothing is learning. Check the optimizer wiring.
- **Above `log V`** → worse than ignorance; almost always broken initialization or a mask/shift bug feeding the wrong targets.
- **Dropping, then flattening** → approaching the corpus's irreducible entropy plus whatever KL the model's capacity can't close. Flat ≠ broken.
- **Small loss deltas are big.** Loss is a log scale: an 0.7-nat improvement *halves* perplexity (`e^0.7 ≈ 2`). The gap between StoryLM-30M (2.74) and 5M (2.56) reads as "0.18 nats" but means the 5M is choosing among ~17% fewer effective options per token.
- **Only compare losses over the same tokenization.** See pitfalls.

---

## <a id="pitfalls"></a>9. Pitfalls

**Comparing losses across vocabularies or tokenizers.** Per-token loss depends on what a "token" is. A 4096-vocab model and an 8192-vocab model split the same text into different events with different baseline entropies — their losses are not on the same scale, even on identical text. This is precisely why Module 12 restricts its scaling curve to one tokenizer family, and why StoryLM (≈2.7) and TinyLLM (≈3.4) numbers must never share an axis: the gap says nothing about which model is better.

**Bits vs nats when reading papers.** Compression-flavored papers report bits-per-byte or bits-per-character (log₂, and a different denominator — *bytes or characters*, not tokens); course losses are nats per token. Convert logs with ×1.443, but note the denominator often differs too — check what's being counted before comparing anything.

**Expecting loss → 0.** Zero loss means the model assigns probability 1 to every next token of held-out text — i.e., language is deterministic. It isn't; `H(p) > 0` is a property of English, not a modeling failure. On *training* data, near-zero loss is achievable and is called memorization — on a tiny corpus it's your overfitting alarm, visible as the train/val gap in Module 10's curves.

**`log(0)` in hand-rolled losses.** If you compute `−log p[t]` yourself and the model assigns the true token exactly zero mass (or float32 rounds it there), you get `inf` and the run dies at one bad batch. The fused/log-space forms (`F.cross_entropy`, `log_softmax`) never materialize the probability, which is the third reason to use them (after stability and the clean gradient).

**Softmax before `cross_entropy`.** `F.cross_entropy` expects **raw logits** — it applies log-softmax internally. Feeding it probabilities double-softmaxes: loss still goes down, samples come out mush, and nothing crashes. One of the course's classic silent bugs.

---

## What this primer doesn't cover

- **Continuous distributions, densities, measure theory.** Token prediction is categorical; nothing continuous is needed.
- **Bayesian inference and calibration theory.** Module 15 touches evaluation-flavored versions of these questions empirically.
- **Label smoothing, focal loss, and other CE variants.** Worth recognizing in the wild; the course trains with plain cross-entropy throughout.
- **Mutual information and the wider information-theory toolkit.** Beautiful, adjacent, unused here. Cover & Thomas if the entropy sections made you hungry.

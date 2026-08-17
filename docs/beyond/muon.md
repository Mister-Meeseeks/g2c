# Beyond — Muon and orthogonalized updates

> **Question this module answers:** *If initialization has to respect a matrix's geometry, why doesn't the update rule?*

<!-- TODO(hero pipeline): asset not yet generated -->
![A momentum matrix's singular-value spectrum, wildly uneven, passing through five Newton–Schulz polynomial steps and emerging as a flat band near one, next to a model diagram whose weight matrices are routed to Muon while its embeddings, norm gains, and biases are routed to AdamW.](muon/BeyondMuon-Hero.png)

For seven years AdamW has been the default answer to "which optimizer," and Module 03B built it. In 2025–26 the default cracked at the frontier: DeepSeek V4 applies Muon to most matrix parameters and keeps AdamW for the rest; GLM-5 pretrained 28.5T tokens on a per-head "Muon Split" variant; Moonshot's Kimi K2 ran MuonClip at trillion-parameter scale. This module builds Muon — momentum, orthogonalized by a five-step polynomial iteration — races it against your own AdamW, and shows why every production deployment is a *hybrid* that still needs the optimizer you built in Module 03B.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course. Come here in any order, whenever a model card or paper names the idea and you want the load-bearing version — built, trained, and broken on your own machine.

---
## Before you start

* *Review*
	* [03-nn](../modules/03-nn.md), the "Initializing weights" section — this module is that argument, applied to updates instead of starting points
	* [03b-training](../modules/03b-training.md) for AdamW's update rule and the optimizer contract (`zero_grad` / mutable `lr` / `step`)
* *Finish*
	* `g2c/training` ([03b-training](../modules/03b-training.md)) — Muon's non-matrix side runs your `AdamW`
	* `g2c/transformer` ([09-transformer-block](../modules/09-transformer-block.md)) and `g2c/pretraining` ([09b](../modules/09b-pretraining.md)) for the language-model race
* *Run*
	* `G2C_APPLY_SOLUTIONS=01-10 ./notebook.sh muon` instead of the plain launch if you're entering without your own implementations

---
## Where this fits in

Module 03 made a geometric argument about where weights *start*: an output is a sum of `fan_in` random terms, so the initial scale must be `σ ≈ 1/√fan_in` or signal explodes or dies through depth. The scale had to respect the matrix's shape.

Then Module 03B built AdamW, and the geometry quietly disappeared. AdamW treats a `(fan_out, fan_in)` weight as `fan_out · fan_in` unrelated scalars: each coordinate gets its own step size from its own gradient history, and the fact that these scalars form a matrix — that the layer's action on the residual stream is governed by that matrix's *directions* and *gains* — plays no role in the update at all.

Muon puts the geometry back. A momentum matrix, like any matrix, has singular directions with unequal gains: a few directions carry most of the update's energy while the rest barely register. Muon *orthogonalizes* the momentum before applying it — pushes every singular value toward 1 — so a step moves the weight along all of the update's directions at comparable magnitude, not just the loudest ones. In the steepest-descent picture: AdamW takes the best step under a per-coordinate metric, Muon under a *spectral* one, which is the natural metric for an object whose job is to be a linear map.

The reason this is now worth a module rather than a footnote is the recurrence at the top of the page: three independent frontier families train on it, and it admits a complete laptop-scale build — the whole optimizer is momentum plus five iterations of three small matmuls.

## The big idea

Take the momentum buffer `M` for one weight matrix and imagine its SVD, `M = U S Vᵀ`. The update Muon wants is `U Vᵀ` — same directions, all gains set to 1. Computing an SVD every step is far too slow, and here is the trick that makes Muon practical: you can move every singular value without ever finding them.

For any *odd polynomial* `f`, applying the matrix expression

```
   f(X) = a·X + b·(XXᵀ)X + c·(XXᵀ)²X
```

acts on `X = U S Vᵀ` as `U f(S) Vᵀ` — the singular vectors pass through untouched, and `f` hits each singular value separately. Muon iterates one fixed quintic, `f(s) = 3.4445·s − 4.7750·s³ + 2.0315·s⁵`, five times:

```
   normalize:  X ← M / ‖M‖_F        singular values now in (0, 1]

   spectrum:   |▁▂▂▃█ ...           a few large, many tiny

   f, ×5:      steep near 0  →  small gains amplified hard
               fixed point ≈ 1  →  large gains parked near 1

   result:     |▆▇█▇▆ ...           a loose band around 1 ≈ U Vᵀ
```

Three properties fall out, and the test suite pins each one:

* **Scale invariance.** The Frobenius normalization means `NS(G) = NS(7G)`: Muon's step direction is completely indifferent to gradient *magnitude*. (AdamW is also approximately scale-invariant, but per-coordinate; Muon is invariant as a matrix.)
* **Direction preservation.** An input whose singular values are already equal — an orthogonal matrix — comes back as a scalar multiple of itself.
* **Rectangular compensation.** An orthogonalized `(rows, cols)` update has RMS entry `~1/√rows`, so Muon scales the step by `√(max(1, rows/cols))` to keep per-entry step sizes comparable across differently-shaped layers.

### The partition: Muon is a hybrid by construction

Orthogonalization is defined for matrices, and *meaningful* for matrices whose rows and columns jointly implement one linear map. That excludes more than the 1-D parameters:

* **Vectors and scalars** — norm gains, biases — have no singular structure. Nothing to orthogonalize.
* **Embedding tables** are 2-D but are really `V` unrelated per-token rows stacked together. Orthogonalizing mixes row `2041`'s update into row `7`'s — tokens that may never appear in the same batch. The unembedding has the same problem transposed.

So every production Muon is Muon-for-projections plus AdamW-for-the-rest, and this module's `Muon` class is honest about it: you hand it two parameter lists, and the second is driven by an internal `g2c.training.AdamW` — literally your Module 03B implementation, composed rather than replaced. Deciding which parameters go in which list is not plumbing; it is the design decision, and the notebook makes you take it (and then violate it, to see why the rule exists).

### What it costs, what it saves

Per matrix, Muon keeps **one** state buffer (momentum) where AdamW keeps two (`m` and `v`) — Module 13B's memory-tenant arithmetic, revisited: the optimizer-state line halves for every parameter on the Muon side. Per step it adds five iterations of three matmuls on a matrix the size of the weight — for the course's models, microseconds. The practical cost is subtler and shows up in the notebook: Muon's useful learning-rate range is *far* from AdamW's (unit-scale updates want lr around `0.02–0.05`, not `3e-4`), so any fair comparison must sweep both — a single-lr race is rigged and the exercises treat it as such.

## Concepts to internalize

- **Updates have geometry.** A weight update is a matrix with directions and gains, not a bag of scalars. AdamW adapts per-scalar; Muon normalizes per-direction.
- **Odd polynomials edit spectra blind.** `a·X + b·(XXᵀ)X + c·(XXᵀ)²X` applies `f` to every singular value without computing any of them. Five iterations of three matmuls replace an SVD.
- **Normalize first.** The quintic converges for spectra in (0, 1]; the Frobenius division puts them there. Skip it and the iteration diverges — the classic implementation bug.
- **Gradient magnitude is discarded; direction is everything.** Muon's step from gradient `g` and from `100g` is identical. That is a *feature* — and the reason its lr means something different than AdamW's.
- **The partition is the design.** Matrices-that-are-maps get Muon; embeddings, vectors, and the head stay on AdamW. Every frontier deployment is this hybrid.
- **Model-card literacy:** "Muon," "Muon Split," "MuonClip," and an optimizer section that assigns different rules to different parameter groups all signal the same underlying move — a spectral update rule for the network's linear maps.

### What we don't cover

- **Distributed Muon.** Newton–Schulz needs whole logical matrices, which collides with sharded training; DeepSeek's hybrid-ZeRO adaptation and Moonshot's scalability work are systems engineering on top of the mechanism built here.
- **MuonClip and stability at scale.** Kimi K2's qk-clip addresses attention-logit blowups at trillion-parameter scale — an interaction this module's model sizes cannot reproduce.
- **The Nesterov variant and low-precision Newton–Schulz.** Production implementations run NS in bfloat16 with a Nesterov-flavored momentum read; we keep plain heavy-ball in float32 so the mechanism stays inspectable.
- **Steepest descent under general norms.** Bernstein & Newhouse's framework (every optimizer is steepest descent under *some* norm) is the right theory lens; we take only its intuition.
- **Second-order ancestors.** Shampoo and K-FAC precondition with curvature estimates; Muon's orthogonalization is cheaper and closer to a normalization than a curvature model. The family resemblance is worth knowing about, not building.

---
## What you'll build

Package: `g2c/muon/`

```python
NS_COEFFS = (3.4445, -4.7750, 2.0315)

def zeropower_via_newtonschulz(G, steps=5): ...   # SCAFFOLDED
    # normalize → (transpose if tall) → five quintic passes → ≈ U Vᵀ

class Muon:
    def __init__(self, muon_params, adamw_params, lr=0.02, *,
                 momentum=0.95, weight_decay=0.0, ns_steps=5,
                 adamw_lr=3e-4, ...): ...          # implemented
    # validates every muon param is 2-D; builds momentum buffers;
    # wraps adamw_params in your g2c.training.AdamW
    def zero_grad(self): ...                       # implemented
    def step(self): ...                            # SCAFFOLDED
        # m ← μ·m + grad;  O = NS(m);
        # p ← (1 − lr·wd)·p − lr·√max(1, rows/cols)·O;
        # then one AdamW step for the other group
```

Total scaffolded code: roughly 25 lines across the two functions. The external interface matches `SGD`/`AdamW` — `zero_grad()`, mutable `lr`, `step()` — so it drops into Module 10's trainer unchanged.

## How to run the tests

Tests live in `tests/test_muon.py`. Initial state: 3 passed (construction, the 2-D validation, `zero_grad`), 14 failed. The AdamW-side test drives the internal `g2c.training.AdamW`, so it needs Module 03B's optimizer (yours, or `G2C_APPLY_SOLUTIONS=03b`).

```bash
source .venv/bin/activate

pytest tests/test_muon.py                    # all module tests
pytest tests/test_muon.py -x                 # stop at first failure (recommended)
pytest tests/test_muon.py -k newton_schulz   # the orthogonalizer alone
pytest tests/test_muon.py -k step            # the optimizer step
```

The signature test: `test_muon_update_is_gradient_scale_invariant` feeds the same parameter a gradient and 100× that gradient and demands identical steps. A missing Frobenius normalization, a wrong coefficient order, or momentum applied after orthogonalization all break it loudly; the three `newton_schulz` sanity anchors (band, orthogonal-direction preservation, transpose consistency) localize which piece went wrong.

## Exercises

To launch the exercise notebook run:

```bash
./notebook.sh muon
```

If at any point you want to archive the work in your current notebook and restart fresh:

```bash
./notebook.sh muon --fresh
```

Written exercises live in the notebook as `Question:` / `Answer:` cells; ask a coding agent for hints or grading when you're ready, and partial submissions are fine — blank answers are skipped.

1. **Watch the spectrum flatten.** Run a random matrix through 1, 2, 3, and 5 Newton–Schulz passes and plot the singular-value spectrum after each. Identify what one pass does to values near zero and near one.
2. **AdamW's update versus Muon's, geometrically.** Build a nearly rank-1 momentum matrix (an early-training shape), compute both optimizers' update directions, and compare their spectra. Which directions does each spend its step on?
3. **The race, run fairly.** Train the same char-level TinyShakespeare transformer under AdamW and under Muon (proper partition), each at its own best learning rate from a small sweep. Report curves, the winner, and how sensitive the result was to lr.
4. **The learning-rate landscape.** From the sweep data: how far apart are the two optimizers' best learning rates, and which degrades more gracefully when the lr is wrong by 3×?
5. **Violate the partition.** Route the embedding table and unembedding through Muon deliberately and rerun the race. Explain what you observe using the rows-are-unrelated-objects argument — or report honestly if the toy scale hides the damage.
6. **Written: the state bill.** For your race model, compute optimizer-state memory under AdamW-everything versus the Muon hybrid, and connect the arithmetic to Module 13B's memory-tenant table.

## Pitfalls to expect

- **Skipping the normalization.** Without the Frobenius division the quintic diverges within a few iterations — exploding entries, NaNs, or a spectrum band test failing with values in the hundreds.
- **Coefficient order.** `(a, b, c)` multiply `X`, `(XXᵀ)X`, and `(XXᵀ)²X` respectively. Swapped coefficients still produce plausible-looking matrices; the orthogonal-direction anchor is what catches them.
- **Racing at one shared learning rate.** At AdamW's `3e-4`, Muon barely moves; at Muon's `0.02`, AdamW diverges. Every fair comparison in this module is best-of-sweep versus best-of-sweep.
- **Orthogonalizing the momentum buffer in place.** `step` must keep the raw momentum for the next iteration and apply the orthogonalized *copy*. Mutating the buffer changes the trajectory subtly — losses still fall, just worse — which makes it the nastiest bug here.
- **`svdvals` on MPS.** Compute spectra on CPU tensors in the notebook; MPS support for SVD-family ops is spotty and can silently fall back or error.
- **Reading one seed as a verdict.** The race exercises are short runs on small models; rerun with a second seed before concluding an optimizer "wins," and report both.

## M-series notes

- **Newton–Schulz is cheap at course scale.** Five iterations of three matmuls per matrix per step adds low single-digit percent overhead to the race runs; do not pre-optimize it.
- **The races are the module's compute cost** — a few thousand char-level TinyShakespeare steps per configuration, minutes each on MPS. Plug in for the sweep cells.
- **Spectrum plots run on CPU** (see pitfalls). The matrices involved are tiny; there is nothing to accelerate.
- **TinyShakespeare ships with `./setup.sh`** — no additional downloads for this module.

---
## Reading

Primary:

- **Jordan, "Muon: An optimizer for hidden layers of neural networks" (2024).** The original write-up — the quintic coefficients, the rectangular scale factor, the empirical case, and the explicit hidden-layers-only scope. This module is this post, built.
- **Bernstein & Newhouse, "Old Optimizer, New Norm: An Anthology" (2024).** The theory frame: familiar optimizers as steepest descent under different norms, with Muon as descent under the spectral norm. Read for the lens, not the proofs.

Secondary:

- **Moonshot AI, "Muon is Scalable for LLM Training" (2025).** MuonClip and the engineering that took Muon from speedrun to Kimi K2's trillion-parameter pretraining — the paper that moved it from curiosity to production evidence.
- **Gupta, Koren, Singer, "Shampoo: Preconditioned Stochastic Tensor Optimization" (2018).** The nearest ancestor: full-matrix preconditioning per tensor. Useful contrast for what Muon *doesn't* compute.

Optional:

- **The current release lineage.** GLM-5's report (Muon Split, 28.5T tokens) and DeepSeek V4's optimizer section (Muon for most matrices, AdamW for embeddings, head, norms, and residual-mixing parameters) — both from 2026. After this module, those parameter-assignment tables read as the partition exercise, deployed.

## Deliverable checklist

- [ ] All tests in `tests/test_muon.py` pass — the gradient-scale-invariance signature especially.
- [ ] Notebook: the spectrum-flattening plot (Exercise 1), the geometric AdamW/Muon comparison (Exercise 2), and the fair two-sweep race with curves (Exercises 3–4).
- [ ] The partition-violation run, with an honest reading of what it did or didn't show (Exercise 5).
- [ ] Written answer for the optimizer-state bill (Exercise 6).
- [ ] You can explain — out loud, without notes — what Newton–Schulz does to a momentum matrix's singular values and vectors, and why normalization must come first.
- [ ] You can explain — out loud, without notes — why Muon's step ignores gradient magnitude, and what that does to its learning-rate scale.
- [ ] You can explain — out loud, without notes — why embeddings and norm gains stay on AdamW in every production deployment.
- [ ] You can explain — out loud, without notes — how this module's argument continues Module 03's initialization story.

# Beyond — Mixture of experts

> **Question this module answers:** *How can a model store far more parameters than any one token uses?*

<!-- TODO(hero pipeline): asset not yet generated -->
![A token arriving at a router that lights up one of eight expert FFNs while the other seven stay dark, with total parameters counted across all experts and active parameters counted only along the lit path.](moe/BeyondMoE-Hero.png)

Many current MoE model cards lead with two parameter counts: DeepSeek V4-Flash is "284B total, 13B active"; Kimi K3 is "2.8T total, 104B active." This module is where that sentence stops being marketing and becomes arithmetic. You'll replace the Module 09 FFN with a routed panel of expert FFNs, train the result on TinyStories, and watch the router learn — and fail to learn — how to share the work.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course. Come here in any order, whenever a model card or paper names the idea and you want the load-bearing version — built, trained, and broken on your own machine.

---
## Before you start

* *Review*
	* [09-transformer-block](../modules/09-transformer-block.md) for the FFN and `Block` anatomy
	* [12-scaling](../modules/12-scaling.md) for parameter and FLOP accounting
* *Finish*
	* `g2c/transformer` ([09-transformer-block](../modules/09-transformer-block.md))
	* `g2c/training` ([03b-training](../modules/03b-training.md))
* *Run*
	* `./datasets.sh --tiny` if the TinyStories corpus or `StoryTokenizer` artifact is missing
	* `G2C_APPLY_SOLUTIONS=01-11 ./notebook.sh moe` instead of the plain launch if you're entering without your own Part I implementations

---
## Where this fits in

Module 09 taught you that the FFN is the "compute" half of the transformer block, and Module 12 taught you where the parameters live: roughly two-thirds of a transformer is FFN weights. Put those together and a dense transformer has an awkward property — **every token pays for every parameter**. If you want more capacity, every token's forward pass gets proportionally more expensive, forever.

Mixture of experts (MoE) breaks that link. Keep the block, keep the attention, keep the residual stream — but replace the single FFN with `E` independent FFNs ("experts") and a small learned **router** that sends each token through only `k` of them. Capacity now scales with `E`; per-token compute scales with `k`. Those are the two numbers on the model card: *total* parameters count every expert, *active* parameters count only the ones a token actually touches.

This is not an exotic frontier trick anymore. Mixtral made it prominent in open-weight models in 2024; DeepSeek, Qwen, NVIDIA's Nemotron line, and Moonshot's Kimi series all adopted variants independently. When unrelated labs make the same move, the move is the durable idea and the branding is not — which is exactly why this module exists.

## The big idea

The FFN is the perfect place for conditional computation, for a reason Module 09 already gave you: **the FFN is per-position**. It never mixes information across tokens. So nothing breaks if token 3 and token 7 go through *different* FFNs — no mask to maintain, no cache to invalidate, no cross-token bookkeeping at all. Attention is the communication half and must see everyone; the compute half is free to specialize.

```
   Dense block (Module 09):            MoE block (this module):

   x ─► LN ─► MHA ─► + ─┐              x ─► LN ─► MHA ─► + ─┐
                        │                                   │
   ┌────────────────────┘              ┌────────────────────┘
   │                                   │
   x ─► LN ─► FFN ─► + ─┐              x ─► LN ─► router ─┬─► FFN₁ ─┐
                        │                        (top-k)  ├─► FFN₂ ─┼─► weighted ─► + ─┐
                        ▼                                 ├─► ...   ─┤     sum         │
                     (B, T, D)                            └─► FFN_E ─┘                 ▼
                                                       (only k of E actually run)  (B, T, D)
```

One dense FFN becomes `E` candidate FFNs plus a router. Per token: the router scores all `E` experts, keeps the top `k`, runs those `k` FFNs, and combines their outputs weighted by the router's (renormalized) scores. Everything else in the block is untouched.

### The router

The router is deliberately boring: one linear layer from the residual stream to expert logits, a softmax, and a top-k selection.

```
   scores  = softmax(x_t @ W_router)        # (E,) — one score per expert
   top-k   = indices of the k largest       # e.g. k = 2
   weights = renormalize(scores[top-k])     # so the selected weights sum to 1

   output  = Σ  weights[e] · FFN_e(x_t)     # over the k selected experts
```

Two details carry most of the implementation:

* **Renormalize after selection.** The softmax was over all `E` experts; after keeping `k` of them, the surviving weights must be rescaled to sum to 1. Skip this and the output magnitude depends on how confident the router happened to be — a subtle, slow poison for training.
* **Gradients flow through the weights, not the selection — when `k > 1`.** Top-k is not differentiable, and doesn't need to be. With multiple surviving experts, their renormalized weights remain differentiable: if expert 3 helped, the language-modeling loss can push its share up. But renormalized `k=1` is a special case: the only surviving weight is always exactly 1, so the language-modeling loss provides no useful router gradient. The auxiliary balance loss can still move that router. This is why the notebook uses `k=1` for clean active-parameter accounting but `k=2` when studying learned routing dynamics.

### Load balancing — the failure mode is the default

With `k>1`, routing left alone can collapse. The dynamic is rich-get-richer: whichever expert is slightly better gets more weight, therefore more task gradient and influence, which can make the router favor it further. The stable end state is one or two overworked experts and a panel of dead ones — you've paid for `E` FFNs and trained something much closer to a dense model. Under this module's renormalized `k=1` rule, the failure is different: without the auxiliary loss the router is effectively frozen near its random initial partition rather than learning a task-driven collapse.

The standard counterweight is an **auxiliary load-balancing loss** added to the training objective (the Switch Transformer form):

```
   f_e  =  fraction of tokens routed to expert e        (hard counts)
   P_e  =  mean router probability for expert e         (soft scores)

   L_balance  =  E · Σ_e  f_e · P_e
```

This is minimized when routing is uniform (`f_e = P_e = 1/E` for all `e`) and grows as routing concentrates. It's weighted by a small coefficient and added to the language-modeling loss. The coefficient is a genuine tension, not a nuisance parameter: in Exercise 4's `k=2` model, too low permits concentration while too high can force routing toward uniformity and crowd out useful specialization. Exercise 4 has you watch both ends.

### Counting total vs active

Module 12's block accounting, extended one line. A dense block is `≈ 12·D²`: 4·D² of attention, 8·D² of FFN. In an MoE block, only the FFN part multiplies:

```
   dense block:    4·D² (attn)  +  8·D² (FFN)              ≈ 12·D²

   MoE block:      4·D² (attn)  +  E·8·D² (all experts)  + D·E (router)
        total  ≈   4·D²  +  E·8·D²
        active ≈   4·D²  +  k·8·D²      ← what one token's forward pass costs
```

With this module's default `E = 8, k = 1`, total block parameters grow ~5.7× while active block parameters remain approximately equal to the dense block: one dense FFN has simply become one selected expert of the same width. The routers add a small amount of active work, so the counts are near-matched rather than bit-identical. Scale the same arithmetic up and you can decode a model card: V4-Flash's 284B/13B says its experts are numerous and its `k` is small. Per-token FLOPs usually track the active count much more closely than the total count, while the total count describes the model's available parameter capacity. Routing overhead, memory movement, batching, and hardware still affect real latency and API price.

### Shared experts and fine-grained slicing

Two refinements from the DeepSeek lineage that recur across model families, worth recognizing by shape:

* **Shared experts** — one or two experts that *every* token goes through, alongside the routed ones. They soak up the common-to-all-tokens computation so the routed experts can specialize harder.
* **Fine-grained experts** — instead of 8 large experts choose 64 small ones (same total parameters) and route to more of them. Finer routing granularity, better specialization; this is how you get to Kimi K3's "16 of 896."

Both are one-paragraph ideas once the base mechanism is built. Neither changes the scaffold's shape in this module.

## Concepts to internalize

- **MoE decouples capacity from per-token compute.** Total parameters set what the model can store; active parameters set what a token costs. Dense models force these to be equal.
- **The FFN is swappable because it's per-position.** No cross-token machinery has to change. This is why MoE lives in the FFN and not (usually) in attention.
- **The router learns through the combination weights when `k>1`.** Top-k selection is non-differentiable and doesn't need to be, but renormalized top-1 is a degenerate case: its sole combination weight is 1, so only the auxiliary loss trains the router.
- **Balanced routing is not the natural state.** With `k>1`, rich-get-richer collapse is possible; with this implementation's `k=1`, an unbalanced random partition can simply stay frozen. The auxiliary loss is a counterweight you tune, not a formality.
- **Model-card literacy:** "X total / Y active" ⇒ per-token compute usually tracks Y much more closely, while storage follows X. Compare MoE and dense models at matched *active* parameters, not matched total.

### What we don't cover

- **Expert parallelism and communication.** At frontier scale the experts live on different devices and the routing becomes an all-to-all network problem. That's distributed-systems engineering the course explains but doesn't reproduce — our experts share one chip.
- **Capacity factors and token dropping.** Large-batch training bounds how many tokens each expert may receive and drops the overflow. At our batch sizes it's machinery without payoff.
- **Auxiliary-loss-free balancing.** DeepSeek-V3 balances with a per-expert bias adjusted outside the gradient instead of an aux loss. A neat variation; read it after building the standard form.
- **Latent-width and low-rank expert variants** (e.g. the LatentMoE line). Refinements of *where* the expert computation happens, not of the routing idea; they read easily once this module is done.

---
## What you'll build

Package: `g2c/moe/`

```python
class Router(Module):
    embedding_dim: int
    num_experts: int                     # E
    top_k: int                           # k
    gate: Linear                         # (D, E)

    def parameters(self): ...                                   # implemented
    def forward(self, x): ...                                   # SCAFFOLDED
        # returns (weights, indices): (B, T, k) each —
        # softmax over E, top-k selection, renormalized weights

class MoEFeedForward(Module):
    embedding_dim: int
    num_experts: int
    top_k: int
    experts: list[FeedForward]           # E copies of Module 09's FFN
    router: Router

    def parameters(self): ...                                   # implemented
    def forward(self, x): ...                                   # SCAFFOLDED
        # dispatch each token to its k experts, weighted-sum the outputs;
        # record routing fractions for the balance loss
    def load_balancing_loss(self): ...                          # SCAFFOLDED
        # E · Σ_e f_e · P_e from the last forward's routing stats

class MoEBlock(Module):                  # Block with the FFN swapped     # implemented
class MoETransformerLM(Module):          # TransformerLM over MoEBlocks   # implemented
```

Total scaffolded code: roughly 30 lines across three methods. The dispatch in `forward` is written as a legible loop over experts — gather the tokens routed to expert `e`, run them, scatter the results back. Production systems fuse this into batched kernels; we keep the loop because you can read it.

## How to run the tests

Tests live in `tests/test_moe.py`. Initial state: 9 passed (construction, accounting, and validation checks), 13 failed.

```bash
source .venv/bin/activate

pytest tests/test_moe.py                 # all MoE tests
pytest tests/test_moe.py -x              # stop at first failure (recommended)
pytest tests/test_moe.py -k router       # router tests only
pytest tests/test_moe.py -k balance      # load-balancing tests only
pytest tests/test_moe.py -v              # verbose
```

The anchor test to know about: `test_moe_e1_k1_matches_dense` asserts that an `E = 1, k = 1` MoE layer is numerically identical to Module 09's plain FFN. If your dispatch logic is right, MoE with one expert *is* the dense model — a strong sanity check that costs nothing.

## Exercises

To launch the exercise notebook run:

```bash
./notebook.sh moe
```

If at any point you want to archive the work in your current notebook and restart fresh:

```bash
./notebook.sh moe --fresh
```

Written exercises live in the notebook as `Question:` / `Answer:` cells; ask a coding agent for hints or grading when you're ready, and partial submissions are fine — blank answers are skipped.

1. **Collapse to dense.** Verify `E=1, k=1` reproduces the Module 09 block, then read the routing weights of an untrained `E=8` router.
2. **Matched-active comparison.** Train dense and MoE versions of the StoryLM-1M architecture on the 100MB TinyStories tier, with `E=8, k=1`, iso-step, and identical batches. Compare validation loss, parameter counts, and generated stories. This is the module's headline experiment. The router is the only meaningful active-parameter difference.
3. **Top-1 utilization over training.** Plot the per-expert token fraction in the matched-active run. Because renormalization makes the selected `k=1` weight exactly 1, changes in its router come from the auxiliary loss rather than the language-modeling objective.
4. **Ablate the balance loss with `k=2`.** Compare three otherwise-identical MoE runs: coefficient `0`, the calibrated default, and `100×` the default. Here the language-modeling loss can train the router through the relative weights of the two selected experts. The first permits task-driven concentration; the last can force routing so uniform that specialization has little room to emerge. Keep all three plots even if a short run produces a weaker effect than expected — the empirical result is the evidence.
5. **Sweep `k`.** Compare `k ∈ {1, 2, 4}` at fixed `E`. More active experts per token means more compute per token — is it buying loss?
6. **StoryLM-5M rerun (optional).** Repeat the matched-active comparison with the StoryLM-5M architecture. The model is still laptop-sized, but its eight-expert MoE holds roughly 28M total parameters and the extra runs take substantially longer.
7. **Specialization probe (optional).** Bucket decoded TinyStories tokens by expert assignment and look for structure. At toy scale the buckets may be noisy; an honest null result is more useful than inventing a role for every expert.

## Pitfalls to expect

- **Forgetting to renormalize the top-k weights.** Output scale then depends on router confidence; trains, but worse, and the `E=1,k=1` equivalence test catches it.
- **Comparing dense vs MoE at matched *total* parameters.** The MoE model looks bad because you've given each token less compute. The default `k=1` comparison keeps the expert width and active FFN compute equal; matched-total answers a different question.
- **Assuming the task loss trains every router configuration.** With `k>1`, relative combination weights carry its gradient. With renormalized `k=1`, the sole weight is 1 and the task gradient to the router vanishes; only the auxiliary loss moves it in this implementation.
- **Balance-loss coefficient at the wrong order of magnitude.** Both failure directions look like "MoE isn't helping." Check the utilization plot before blaming the architecture.
- **Judging routing balance from one small batch.** At `B·T` of a few thousand tokens, per-expert fractions are noisy. Average over many steps before concluding collapse.
- **Expecting a wall-clock speedup.** At our scale the expert loop is Python overhead; MoE here demonstrates the *parameter* accounting, not the *latency* win. The latency win needs fused kernels and real batch sizes.

## M-series notes

The required path uses the StoryLM-1M architecture on the 100MB TinyStories tier. StoryLM-5M is an optional rerun, not part of completing the module.

- **Memory follows total, compute follows active.** With the course's 4,096-token vocabulary, the default dense model is about 1.35M parameters; the `E=8, k=1` MoE is about 5.05M total / 1.36M active. The optional StoryLM-5M pair is about 5.86M dense versus 27.94M total / 5.87M active for MoE.
- **The expert loop is MPS-unfriendly.** Eight small matmuls dispatch worse than one big one. Expect the MoE run to be materially slower per step than dense even though their active parameter counts nearly match; this implementation demonstrates conditional computation, not a fused-kernel latency win.
- **Iso-step comparisons are the honest ones here.** Both models see the same sampled batches in the same order. Iso-FLOP would require accounting for router and dispatch overhead; the notebook reports those caveats instead of pretending the wall clocks should match.

---
## Reading

Primary:

- **Shazeer, Mirhoseini, Maziarz et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017).** The modern MoE paper: routing, top-k, load balancing, and the capacity/compute decoupling argument, all already here. §2 is the mechanism; read it before any model card.
- **Fedus, Zoph, Shazeer, "Switch Transformers" (2021).** Simplifies routing to one selected expert and introduces the load-balancing loss you'll implement. Compare its gate weighting carefully with this module's explicitly renormalized top-1 rule; that detail determines whether the task loss reaches the router.
- **Jiang, Sablayrolles, Roux et al., "Mixtral of Experts" (2024).** The release that made sparse MoE broadly visible in open-weight models. Short, concrete, and §4's routing analysis is your Exercise 7 done at production scale.

Secondary:

- **Dai, Deng, Zhao et al., "DeepSeekMoE" (2024).** Shared experts and fine-grained expert slicing — the refinements that became the DeepSeek/Kimi house style.
- **Lepikhin, Lee, Xu et al., "GShard" (2020).** MoE meets expert parallelism; read for what the distributed version costs, not to implement.
- **DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024).** §on auxiliary-loss-free balancing — the bias-based alternative to the loss you built.

Optional:

- **The current release lineage.** Kimi K3's and DeepSeek V4's technical reports (2026) are the model cards this module teaches you to read. Look for the total/active split, the `k`-of-`E` routing spec, and the shared-expert count — all three should now parse on sight.

## Deliverable checklist

- [ ] All tests in `tests/test_moe.py` pass.
- [ ] Notebook: StoryLM-1M matched-active dense-vs-MoE comparison, with validation-loss plot and generated stories.
- [ ] `k=2` utilization plots from all three balance settings: zero, calibrated default, and 100× default.
- [ ] You can explain — out loud, without notes — the difference between total and active parameters, and why active count is a compute proxy rather than an exact latency or price formula.
- [ ] You can explain — out loud, without notes — why the language-modeling loss trains a `k>1` router but not this implementation's renormalized `k=1` router, plus what the auxiliary loss trades away as its coefficient grows.
- [ ] You can explain — out loud, without notes — why MoE lives in the FFN and not in attention, in terms of Module 09's communication/compute split.

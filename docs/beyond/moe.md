# Beyond — Mixture of experts

> **Question this module answers:** *How can a model store far more parameters than any one token uses?*

<!-- TODO(hero pipeline): asset not yet generated -->
![A token arriving at a router that lights up one of eight expert FFNs while the other seven stay dark, with total parameters counted across all experts and active parameters counted only along the lit path.](moe/BeyondMoE-Hero.png)

This week is about a new variant on the classic transformer architecture. Up until now we've only thought about models in terms of a single parameter count. Now we think about total parameters and active parameters. Done right, we can have our cake and eat it too. The power and knowledge of a large model, and the efficiency of a small model.

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

Module 09 taught you that the feed forward network (FFN) is the "compute" half of the transformer block. Module 12 taught you where the parameters live: roughly two-thirds of a transformer is FFN weights. Put those together and a dense transformer has an awkward property — **every token pays for every parameter**. If you want more capacity, every token's forward pass gets proportionally more expensive, forever. 

The tension comes down to the scaling laws you discovered in Module 12. The single most powerful lever is a larger model. More parameters mean the model learns a richer world model and more comprehensive semantics. But you almost never need more than a fraction of that at a given time. If the prompt is generating python code, you probably don't need the weights that learned how to tell children's stories. A completion for a lasagna recipe is still spending computation on activating the neurons that learned physics. 

*Mixture of experts (MoE)* is an architectural variant on the transformer block that attempts to reflect this observation. The key insight is that while it trains a large parameter set, on any given pass it's only activating a fraction of those parameters. The idea is each expert network learns to specialize in different areas, and that tokens are *routed* to the expert networks best suited for their context.

One common misconception is that expert networks are explicitly trained on subject matter data. The expert networks aren't explicitly trained to be specialized. They are just architecturally neutral sub-networks in the block. MoE networks use the same pretraining formula we learned in Module 09B. But during training, the normal gradient descent process results in the emergence of specialization across experts (often in opaque ways).

## The big idea

Keep the block, keep the attention, keep the residual stream — but replace the single FFN component with `E` number of independent FFNs ("experts"). Put a small learned *router* in front of each expert layer. At each individual token pass, the router selects `k` experts to activate. Do this for each layer in the transformer. You now have an MoE.  

```
   Dense block (Module 09):            MoE block (this module):

   x ─► LN ─► MHA ─► + ─┐              x ─► LN ─► MHA ─► + ─┐
                        │                                   │
   ┌────────────────────┘              ┌────────────────────┘
   │                                   │
   x ─► LN ─► FFN ─► + ─┐              x ─► LN ─► router ─┬─► FFN₁ ─┐
                        │                        (top-k)  ├─► FFN₂ ─┤
                        ▼                                 ├─► ...  ─┤  
                     (B, T, D)                            └─► FFN_E─┤
                                                                    │
	                                                            weighted +
                                                                    ▼
	                                                            (B, T, D)
```

One dense FFN becomes `E` candidate FFNs plus a router. Per token: the router scores all `E` experts, keeps the top `k`, runs those `k` FFNs, and combines their outputs weighted by the router's (renormalized) scores. Everything else in the block is untouched.

Capacity now scales with `E`. Per-token compute scales with `k`. Those are the two numbers on the model card: *total* parameters count every expert, *active* parameters count only the ones a token actually touches.

Activation routing runs **per token** because the FFN operates per position. It never mixes information across tokens. Nothing breaks if token 3 and token 7 go through different experts — no mask to maintain, no cache to invalidate, no cross-token bookkeeping at all. Attention is the communication half that must see everyone. But the FFN is the compute half that is free to specialize.

Another thing to keep in mind is that each layer in the transformer maintains its own independent expert network and router. Specialization will occur differently depending on the layer. That combined with per token activation illustrate that specilalization happen at a much more granular level than the full prompt. For example a 64 layer network on a 1000 token prompt will process 64,000 independent router activation decisions in the forward pass. 

### The router

The router is deliberately boring. It takes the residual stream as input, and outputs a set of sparse weights across the experts in the block. Those weights determine which expert FFNs are activated, and how their output is mixed in the residual stream. During gradient descent the router learns to activate the best experts based on the context in the residual stream.

The router has a simple internal architecture. It is a one layer linear network. The raw output of that network is passed through three layers to create the activation properties we want:

* **Softmax.** Like with all neural network classifiers, softmax turns raw logits into a categorical distribution. (The category being which experts to activate.)
* **Top-k**. The layer that actually keeps activations sparse. Without this, all experts (and therefore all parameters) would receive some non-zero activation weight.
* **Renormalize.** The softmax weights were over all `E` experts. Truncating all but `k` means the weights need to be rescaled to sum back to 1. Without this output magnitude becomes dependent on router confidence, which is a slow subtle poisoning for training. 

```
   scores  = softmax(x_t @ W_router)        # (E,) — one score per expert
   top-k   = indices of the k largest       # e.g. k = 2
   weights = renormalize(scores[top-k])     # so the selected weights sum to 1

   output  = Σ  weights[e] · FFN_e(x_t)     # over the k selected experts
```

One thing to keep in mind is that top-k itself is not differentiable. This can create complications for gradient descent that need to be carefully managed. If `k > 1` , this is much less of an issue. Every pass has multiple surviving experts, and each surviving expert has a non-zero weight. Those expert weights *are* still differentiable and backprop the learning signal to the router. However `k = 1` does not work without a supplemental loss function, because there is no gradient to learn. 

### Load balancing 

For both `k=1` and `k>1`, router training is pathological under standard language model loss. As we saw above for `k=1` routing cannot learn at all and remains frozen near its initialization, because there is no gradient. For `k>1`, a gradient exists but the natural equilibrium is *router collapse*. 

Router collapse is a classic rich-get-richer scenario. Whichever expert becomes slightly better during training, continues to get more weight. It then gets more gradient and learning, which favors its route even more. Without being addressed the natural end state is one or two overworked experts and a panel of dead ones.

The solution is a supplemental loss function added to the standard training objective. *Auxilliary loss balancing* penalizes uneven distribution of experts across the token distribution:

```
   f_e  =  fraction of tokens routed to expert e        (hard counts)
   P_e  =  mean router probability for expert e         (soft scores)

   L_balance  =  E · Σ_e  f_e · P_e
```

This is minimized when routing is uniform (`f_e = P_e = 1/E` for all `e`) and grows as routing concentrates. Critically, this penalty is differentiable, and therefore restores a gradient even for `k=1`.

Auxiliary loss is weighted by a small coefficient and added to the language-modeling loss. The coefficient is a genuine tension, not a nuisance parameter. Too low permits concentration. Too high can force routing toward uniformity and crowd out useful specialization. Like all hyper parameter selections, the best approach is an active sweep based on empirical data.

### Counting total vs active

Let's revisit the param accounting from the scaling lessons in Module 12. As you recall, for a transformer block, with `D` embedding dimension, in each layer you have `≈ 12·D²` parameters.  That breaks down to `4·D²` of attention, and `8·D²` of FFN. 

In an MoE block, the attention component remains the same. But with `E` number of experts you now have `E` number of FFNs, each with `8·D²` parameters. (Add in a small number of parameters for the router, but it's usually de minims.) 

```
   dense block:    4·D² (attn)  +  8·D² (FFN)              ≈ 12·D²

   MoE block:      4·D² (attn)  +  E·8·D² (all experts)  + D·E (router)
        total  ≈   4·D²  +  E·8·D²
        active ≈   4·D²  +  k·8·D²      ← what one token's forward pass costs
```

With this module's default `E = 8, k = 1`, total block parameters grow ~5.7× while active block parameters remain approximately equal to the dense block. One dense FFN has simply become one selected expert of the same width. The routers add a small amount of active work, so the counts are near-matched rather than bit-identical. 

Scale the same arithmetic up and you can decode any model card. V4-Flash's card at 284B total, 13B active tells us its experts are numerous, and its active count is small. FLOPs track the active count much more closely than the total count. Total count conveys the model's parameter capacity. This is an approximate picture. Routing overhead, memory movement, batching, and hardware still affect real latency and API price.

### MoE refinements in practice

There are two additional refinements (both from the DeepSeek lineage) that commonly accompany MoE models. They're worth recognizing by shape:

* **Shared experts** — One or two universal experts that *every* token goes through, in addition to the specialized routed ones. They soak up the common token computation, which enables the routed experts to specialize harder.
* **Fine-grained experts** — Basically just using a larger number of smaller experts for a constant param size. Instead of 8 large experts choose 64 small ones and route to more of them on each pass. Finer routing granularity, better specialization. This is how you get to Kimi K3's "16 of 896."

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

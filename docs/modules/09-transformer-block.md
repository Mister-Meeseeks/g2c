# Module 09 — The transformer block

> **Question this module answers:** *How do we compose attention and "thinking"?*

![One transformer block, drawn as two pre-norm sublayers wrapped in residual connections. The residual stream (the (B, T, D) tensor at the top) flows forward unchanged by default; each sublayer reads it through a LayerNorm, computes an update, and adds the update back. Sublayer 1 is multi-head attention; sublayer 2 is the position-wise feed-forward network. Side panels label the two structural ideas (LayerNorm normalizes per token, residual connections add a small update) and pin the recipe `x = x + attn(ln1(x)); x = x + ffn(ln2(x))`.](09-transformer-block/Module09-Hero.png)

Transformers are layers of blocks. Each block is an attention sublayer and a normal neural network sublayer. Once you have this block, scaling up is just "stack N of these and add embeddings" + a final unembedding head. The rest of the lesson page is unpacking why each one is where it is.

---
## Prerequisites

Module 09 is where the transformer finally takes shape — every piece you've built in Modules 03–08 gets composed here into the unit that gets stacked `N` times to make a real model.

### Math

* **Jacobians**. Equivalent to a gradient but extended from scalars to vectors. Jacobians are part of how the "norms" in this module avoid the problem of "vanishing gradients" or numerical instability that come with deep architectures.
* **Spectral norms**. Linear algebra. It's enough to be familiar with the concept that large spectral norms mean a matrix is generally "stretching" its inputs; a small spectral norm is "shrinking" its inputs; and a spectral norm of 1.0 is neutral in this regards

### Computer science

- **Object composition.** The `Block` module we developer here uses the same python object-oriented pattern as Module 03's `Sequential`.

### PyTorch

- Re-familiarize yourself with `mean()` and `var()` functions

---
## Where this fits in

After Module 08, you have multi-head attention as a standalone mechanism — a learnable mixing operation that lets every position consult every other. But attention alone doesn't support multiple layers, which is the core of deep learning. Attention alone is insufficient at scale. This module shows us an architecture that allows us to *stack* attention.

## The big idea

A transformer is attention *embedded inside* a specific architectural sandwich that makes it trainable at depth. The building block of the transformer is the block. Once you have it, scaling up is "use a larger `D`, larger `T`, more blocks." 

```
  ┌─ Multi-head attention sublayer ────┐
  │                                     │
  x ──► LayerNorm ──► MHA ──► + ──┐
  │                              │
  └──── residual connection ─────┘
                                  │
  ┌─ Feed-forward sublayer ──────┘─────┐
  │                                     │
  x ──► LayerNorm ──► FFN ──► + ──┐
  │                              │
  └──── residual connection ─────┘
```

Three new ideas wrap around the attention you already have:

1. **Layer normalization** keeps activations in a numerically sane range as they flow through the network. Without it, activations can blow up or collapse — and gradients along with them.
2. **Residual connections** turn the network into a *refinement pipeline* rather than a transformation pipeline. Each sublayer reads the residual stream and adds a small update; the original information is preserved by default. Without residuals, deep transformers don't train.
3. **The position-wise feed-forward network** gives the model a way to do non-attention computation per position — a per-token MLP applied to whatever attention pulled in. Vaswani et al. call this the "position-wise feed-forward network." It's where most of a transformer's parameters live and where most of its per-position reasoning happens.

These mechanisms combine into the full pipeline of one block. With all the reshape-free arithmetic written out:

```
       (B, T, D)              ┌────────────────────┐
   x ─┬─► LayerNorm ──► MHA ──► +
      │                       ▲
      └──── residual ─────────┘
                              │
                              ▼
                       (B, T, D)
                              │
                              ├──► LayerNorm ──► FFN ──► +
                              │                          ▲
                              └──── residual ────────────┘
                                                         │
                                                         ▼
                                                  (B, T, D)
```

Two sublayers. Each sublayer is "normalize → transform → add to residual." The full block in code is exactly this:

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))     # attention sublayer
    x = x + self.ffn(self.ln2(x))      # FFN sublayer
    return x
```

Two lines. Everything important about the transformer architecture is encoded in the order of operations on those two lines.

### Why residual connections matter

The headline empirical fact: without residual connections, transformers deeper than a handful of layers fail to train. With them, training scales to hundreds of layers. The intuition has two complementary flavors:

**Gradient-flow view.** During backprop, `∂loss/∂x` at the input depends on the chain of partial derivatives through every sublayer. In a non-residual network, that chain multiplies through every sublayer's Jacobian; if those Jacobians have spectral norm `< 1` on average, the gradient shrinks exponentially with depth — vanishing gradients. The residual connection turns the chain into `1 + ∂sublayer/∂x` at each step. The `1` term ensures gradients flow through the residual path even if `∂sublayer/∂x ≈ 0`. Gradients can no longer vanish completely.

**Residual-stream view.** Think of `x` as a "communication bus" that threads through the entire network. Each sublayer reads the bus (via LN), produces an update, and writes the update back onto the bus (via the `+`). Sublayers are *additive contributions* to the residual stream rather than *replacements* of it. The model's "default behavior" is to pass information through unchanged; sublayers *specialize* to make targeted edits. This is the framing made famous by Anthropic's transformer-circuits work.

```
  Without residuals:        With residuals:
                            
   x  ─► f1 ─► f2 ─► f3       x  ─┬─► f1 ─► +  ─┬─► f2 ─► +  ─┬─► f3 ─► +
                                 │             │             │
                                 └────────────┘             │
                                               └────────────┘
                                                             └──── ...
```

![The residual stream as a horizontal "bus" that threads through every block. The embedding sum enters on the left; each block reads the stream, computes a small update Δᵢ via its sublayer (attention or FFN), and adds it back onto the stream. Δ₁ might do "communication across tokens" (attention), Δ₂ might do "per-token computation" (the FFN), Δ₃ might be "feature refinement" — each block specializes in what it adds. After N blocks, the stream is x_N = x + Σᵢ Δᵢ; the final layer norm and unembedding head consume that sum.](09-transformer-block/Module09-ResidualBus.png)

*The "communication bus" framing made popular by Anthropic's transformer-circuits work. The structural property to internalize: information flows forward UNCHANGED by default. Sublayers make incremental edits, not replacements. This is the property that makes deep transformers trainable (gradient-flow view) AND the property that lets mechanistic-interpretability research decompose what each layer does (residual-stream view) — same fact, two ways to read it.*

### Why LayerNorm specifically

Three properties of LN are worth internalizing:

```
  LayerNorm(x):
      mean = x.mean(dim=-1)        # over channels, NOT batch
      var  = x.var(dim=-1)         # over channels, NOT batch
      x_hat = (x - mean) / sqrt(var + ε)
      return γ * x_hat + β
```

  * **It normalizes per-token.** For input shape `(B, T, D)`, LN pools statistics over the `D` axis only. Each `(B, T)` position is normalized independently. That's why batch size doesn't affect the output, why batch_size = 1 works fine, and why train and inference behavior are identical. (BatchNorm pools across the batch dim and has none of these properties.)

  * **The learned affine `γ, β` is the escape hatch.** Pure standardization would lock every output's mean and variance to 0 and 1, which constrains the next layer. The affine parameters let the model freely choose any mean and variance — they're initialized to `(1, 0)` (identity affine) so LN starts as pure standardization.

  * **The `ε` in the sqrt is structural, not cosmetic.** A near-constant input has near-zero variance, and dividing by `sqrt(0)` produces `NaN`. `ε = 1e-5` keeps the divisor away from zero with negligible effect on normal inputs.

LayerNorm is what keeps the residual stream's scale bounded. Without it, after a few blocks the residual `x` has accumulated so many unnormalized sublayer outputs that its magnitude diverges, attention softmaxes saturate, and everything stops training.

![LayerNorm worked through on a single token vector x ∈ ℝ^D: compute mean μ and variance σ² across the D channels of THIS token only (no pooling across batch or sequence positions); subtract μ and divide by √(σ²+ε); apply the learned per-channel affine γ * x̂ + β. A side panel contrasts what LayerNorm does NOT do (pool across batch — that's BatchNorm; pool across sequence positions — that doesn't exist as a standard layer) and pins down the headline: every token in every position is normalized independently with the same γ, β.](09-transformer-block/Module09-LayerNorm.png)

*The structural difference from BatchNorm is "pool over channels, not over the batch." Three consequences fall out: batch size doesn't affect the output (so batch_size=1 works), train and inference behavior are identical (no running statistics to track), and packed/variable-length batches don't need any special handling. 

### Pre-norm vs post-norm

The original 2017 transformer (Vaswani et al.) used **post-norm**:

```
  Post-norm:    x = LN(x + sublayer(x))
```

The modern transformer (GPT-2 onward, every model since) uses **pre-norm**:

```
  Pre-norm:     x = x + sublayer(LN(x))
```

The difference is one of operation order, but it's load-bearing for training stability. Post-norm requires a careful learning-rate warmup schedule; pre-norm trains stably without warmup at much greater depths.

```
  Pre-norm pipeline (this module):

     x ──┬─► LN ──► sublayer ──► +
         │                       ▲
         └─── residual stream ───┘    ← residual flows past LN

  Post-norm pipeline (Vaswani 2017):

     x ──┬─► sublayer ──► + ──► LN
         │               ▲
         └── residual ───┘             ← residual flows through LN
```

The crucial difference: in pre-norm, the residual stream is *never normalized in place*. The unnormalized `x` flows from block to block, and each block reads a normalized COPY for its sublayer. This means gradients on the residual path are unobstructed by LN's nonlinearity — they hit the `+` and propagate straight through, exactly the property that makes deep transformers trainable. In post-norm, every step of the residual path passes through an LN, and the gradients on the residual path get attenuated by LN's Jacobian at every layer. The empirical result is that very deep post-norm models need warmup or they diverge; very deep pre-norm models do not.

For Module 09 we use pre-norm. Exercise 1 will have you implement post-norm and observe the stability difference at small scale.

![Pre-norm vs post-norm shown side by side as two block diagrams: pre-norm (used by GPT-2, Llama, PaLM, ...) — `x = x + sublayer(LN(x))` — has the residual stream flowing past LN, never normalized in place; post-norm (used by "Attention Is All You Need", 2017) — `x = LN(x + sublayer(x))` — has LN sitting ON the residual path so the stream IS normalized between blocks. A side caption explains the gradient-path difference: pre-norm's residual gradient is unobstructed by LN's Jacobian, so it propagates straight through deep stacks; post-norm's residual gradient gets attenuated layer by layer, which is why post-norm needs a learning-rate warmup hack to train at depth.](09-transformer-block/Module09-PrePostNorm.png)

*Same code, different parenthesization, very different training dynamics. The 2017 paper used post-norm with careful warmup and so it worked; modern wisdom (Xiong et al. 2020) is that pre-norm is structurally better and removes the need for warmup tricks. Every modern transformer you read about uses pre-norm.*

### The position-wise FFN

After attention has mixed information across positions, the FFN does non-attention computation at each position:

```
  FFN(x_t)  =  W_2 · GELU(W_1 · x_t + b_1) + b_2     for each position t
```

Three things to internalize:

  * **Per-position.** No mixing across positions — the FFN is the "compute" half of the block, attention is the "communication" half. Position `t` and position `s` see the same `W_1, W_2` but process their own `x_t`, `x_s` independently.

  * **The `4×` expansion.** `hidden_dim = 4 × embedding_dim` is the Vaswani-original convention and is essentially universal. It's an overcomplete intermediate representation: the projection up to `4D` gives the GELU room to carve out useful nonlinear features, then the projection back down to `D` mixes them into the output channel space.

  * **Most parameters live here.** With `hidden_dim = 4D`, the FFN has `2 × 4 × D² = 8D²` weight parameters. Multi-head attention has `4 × D² = 4D²` weight parameters. So roughly two-thirds of a standard block's parameter count is in the FFN. When people talk about "scaling up the model," most of what scales up is the FFN.

![The position-wise FFN: the SAME two-layer MLP applied independently to every position. Per-position view: x_t (D channels) → Linear up to 4D → GELU → Linear back down to D. Block view: attention writes a per-token update into the residual stream, then LayerNorm + FFN compute a second per-token update. Side panels show the parameter counts (2 · 4D² = 8D² weights, dominating the block's parameter budget) and the per-position independence (mutate one token, no others change).](09-transformer-block/Module09-FFN.png)

*The FFN is the "compute" half of the block: attention is what mixes information across positions, the FFN is what the model does with that mixed information at each position. Two non-obvious facts: (1) the same Linear weights are reused at every position — there is no per-position parameterization, just per-position application; (2) most of the parameter budget of a transformer lives here, not in attention. When the literature talks about "scaling up D," the FFN's 4D bottleneck is what scales the most.*

### The full TransformerLM

`TransformerLM` is the minimal viable language model: embed, refine through `N` blocks, normalize, and unembed.

```
  token_ids  ──► TokenEmbedding   ──┐
                                     ├──► +  ──► N × Block ──► LayerNorm ──► Linear(unembed) ──► logits
  positions  ──► PositionalEmbed  ──┘

  shapes:
    token_ids   (B, T)
    tok         (B, T, D)
    pos         (T, D)         broadcasts to (B, T, D)
    x           (B, T, D)      after every block
    logits      (B, T, V)
```

Three details worth pinning down:

  * **One logit per position.** Output is `(B, T, V)`, not `(B, V)`. Position `t`'s logit is the prediction for what comes at position `t+1`. At training time, you compute cross-entropy at every position in parallel — vastly more efficient than the one-position-per-step training of the Module 06 MLP.

  * **The final LayerNorm before unembedding.** Modern transformers add this; the original 2017 paper didn't. Without it, the residual stream's scale at the output is unbounded and the head's logits can drift arbitrarily large or small. A small, cheap correction.

  * **`max_seq_len` is enforced in `forward`.** The learned positional embedding table has a fixed size; sequences longer than that have no positional signal for the trailing positions. The constructor can't see the input length, so the bound check lives in `forward`.

## Concepts to internalize

- **The transformer block is two sublayers, each pre-normalized and residually wrapped.** That's the entire architectural delta from attention alone.
- **The residual stream is the model's "communication bus."** Sublayers add contributions to it; they don't replace it.
- **LayerNorm normalizes over the channel dim only.** Each (B, T) position is normalized independently — no cross-position or cross-batch pooling.
- **Pre-norm is the modern default.** Trains stably at depth without warmup; post-norm needs warmup or diverges.
- **The FFN is per-position with a 4× hidden expansion.** Most of a transformer's parameters live here.
- **Stacking blocks is straightforward.** `for block in self.blocks: x = block(x)`. The architecture has no positional encoding *between* blocks, no cross-block coupling, no per-block parameters that depend on layer index. Each block is a self-contained refinement step.
- **TransformerLM outputs (B, T, V) logits.** One next-token prediction per position, computed in parallel during training.

### What we don't cover

- **Tied embeddings** (sharing weights between `TokenEmbedding` and the unembedding `head`). Standard in most modern LMs; we keep them separate for clarity. Module 10 discusses the tradeoff.
- **Dropout.** Used by Vaswani et al. and many follow-ups for regularization; not strictly necessary at the small scales we'll train, and it adds a `training`/`eval` mode distinction that our minimal `Module` base class doesn't model. Out of scope.
- **RMSNorm** (used by Llama and other modern transformers). A simplified LayerNorm without the mean-subtraction step. Equivalent in practice but conceptually a small variation; we use vanilla LN.
- **Mixed precision, gradient checkpointing, fused ops.** Module 10 pretraining concerns, not architecture concerns.

---
## What you'll build

Package: `g2c/transformer/`

```python
class LayerNorm(Module):
    embedding_dim: int
    eps: float
    gamma: torch.Tensor                  # (D,)
    beta: torch.Tensor                   # (D,)

    def parameters(self): ...                                   # implemented
    def forward(self, x): ...                                   # SCAFFOLDED

class FeedForward(Module):
    embedding_dim: int
    hidden_dim: int                      # default: 4 * embedding_dim
    fc1: Linear                          # (D, hidden_dim)
    fc2: Linear                          # (hidden_dim, D)

    def parameters(self): ...                                   # implemented
    def forward(self, x): ...                                   # SCAFFOLDED

class Block(Module):
    embedding_dim: int
    num_heads: int
    hidden_dim: int
    causal: bool
    ln1: LayerNorm
    attn: MultiHeadAttention
    ln2: LayerNorm
    ffn: FeedForward

    def parameters(self): ...                                                            # implemented
    def forward(self, x): ...                                                            # SCAFFOLDED

class TransformerLM(Module):
    vocab_size: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    hidden_dim: int
    token_embed: TokenEmbedding
    pos_embed: LearnedPositionalEmbedding
    blocks: list[Block]
    ln_final: LayerNorm
    head: Linear

    def parameters(self): ...                                                                                    # implemented
    def forward(self, token_ids): ...                                                                            # SCAFFOLDED
```

Total scaffolded code: roughly 20 lines split across four `forward` methods. Most of the lesson is in *which* lines and *in what order* — the math is unsubtle once you've internalized the structure.

## How to run the tests

Tests live in `tests/test_transformer.py`. Initial state: 22 passed (all the construction, parameter-count, and init-value checks), 22 failed.

```bash
pytest tests/test_transformer.py             # run all module-09 tests
pytest tests/test_transformer.py -x          # stop at first failure (recommended)
pytest tests/test_transformer.py -k layer_norm   # only LayerNorm tests
pytest tests/test_transformer.py -k block    # only Block tests
pytest tests/test_transformer.py -v          # verbose
```

## Exercises

1. **Implement post-norm and watch it (mis)train.**  In a notebook, define a `PostNormBlock` by editing your `Block.forward` to use the post-norm formula:

   ```python
   x = self.ln1(x + self.attn(x))
   x = self.ln2(x + self.ffn(x))
   ```

   Stack 6 of these into a small LM (vocab = 50, D = 64, H = 4, T = 32, no warmup, lr = 1e-3) and train for 1000 steps on a tokenized short corpus. Train an identical pre-norm `TransformerLM` on the same data with the same hyperparameters. Plot both training-loss curves on the same axes.

2. **Strip residual connections.** Edit `Block.forward` (or make a `ResidualFreeBlock` variant) to drop the residual additions:

   ```python
   x = self.attn(self.ln1(x))      # NO + x
   x = self.ffn(self.ln2(x))       # NO + x
   ```

   Train a small LM with this block at `num_layers = 1, 2, 4, 8`. The 1-layer model trains. The 2-layer model trains slowly. The 4-layer and 8-layer models fail to train at all — loss is stuck near `log(vocab_size)`, the uniform-baseline. The visceral signal: gradients literally cannot reach the early layers without the residual highway.

3. **Strip layer normalization.** Replace `self.ln1(x)` and `self.ln2(x)` with the identity (just `x`). Same training setup as exercise 2. The signs of failure are different: the model usually doesn't reach a crashed state, but training is unstable, with oscillating loss curves and frequent NaNs at higher learning rates. The residual-stream magnitude grows unchecked across blocks; by layer 4 or 5 it's so large that attention softmaxes saturate.

4. **Compare scaling at fixed total parameter budget.** Build three `TransformerLM`s with approximately equal parameter counts:

   - `(num_layers=1, embedding_dim=128, num_heads=4)` — wide and shallow.
   - `(num_layers=4, embedding_dim=64, num_heads=4)` — balanced.
   - `(num_layers=8, embedding_dim=48, num_heads=4)` — deep and narrow.

   Verify the parameter counts are within a few percent of each other (the FFN's 4× expansion makes per-block cost grow as `D²`, so matching the budget exactly takes a calculator). Train all three on the same dataset for the same number of steps. Plot validation loss. Modern wisdom: balanced/deeper wins for language modeling at reasonable scales — but the gap is small at the toy scales here.

5. **Parameter-count sanity check.** Compute analytically the parameter count of `TransformerLM(vocab_size=V, embedding_dim=D, num_layers=N, num_heads=H, max_seq_len=T_max)` (with default `hidden_dim = 4D`):

   - Token embedding: `V × D`
   - Positional embedding: `T_max × D`
   - Per block: 2 LNs (`2 × 2D`) + MHA (`4 × (D² + D)`) + FFN (`(D × 4D + 4D) + (4D × D + D)` = `8D² + 5D`)
   - Final LN: `2D`
   - Unembedding head: `D × V + V`

   Sum it up symbolically; verify by summing `p.numel()` over `m.parameters()` for a few `(V, D, N, H, T_max)` tuples. Notice that for typical `D ≫ T_max ≪ V`, the dominant term is `V × D` (the embeddings) at small scales and `12 N D²` (the blocks) at large scales.

## Common pitfalls

- **LayerNorm over the wrong dim.** Pooling statistics over the batch dim (BatchNorm-style) or over the sequence dim instead of the channel dim. Symptom: training works but is unusually slow; batch size matters in surprising ways. 

- **`unbiased=True` in the variance.** Divides by `N - 1` instead of `N` — the *sample* variance instead of the *population* variance. Off by a factor of `D / (D - 1)` from the standard implementation. Hard to notice; impossible to debug.

- **Forgetting `eps`.** A constant-along-channel input has zero variance and your normalized vector is `0 / sqrt(0)` = `NaN`. The test `test_layer_norm_handles_constant_input` catches this.

- **Post-norm by accident.** Writing `x = self.ln1(x + self.attn(x))` instead of `x = x + self.attn(self.ln1(x))`. The shape is the same, the test for output shape passes, but training is dramatically less stable.

- **Forgetting the residual.** Writing `x = self.attn(self.ln1(x))` instead of `x = x + self.attn(self.ln1(x))`. The model becomes untrainable past 2-3 layers. 

- **Sharing one LN between sublayers.** Reusing `self.ln1` for both the attention and FFN sublayers instead of having a separate `self.ln2`. Doesn't crash, but reduces expressiveness — the FFN loses its independent scale/shift.

- **Forgetting the final LayerNorm in `TransformerLM`.** A common oversight; the original 2017 transformer didn't have one, but every modern transformer does. Without it, the head's logits can drift arbitrarily large or small as training progresses.

- **Wiring `pos_embed` outside the broadcast.** `pos = self.pos_embed(T)` has shape `(T, D)`. Adding it to `tok` of shape `(B, T, D)` works via broadcasting — but if you accidentally write `pos.unsqueeze(0)` you get `(1, T, D)` which also broadcasts but hints at a confused mental model. Both work; one is cleaner.

- **`for block in self.blocks: x = block(x)` is sequential — do NOT parallelize it.** Block `i` reads block `i-1`'s output. Trying to run them concurrently misunderstands the architecture (this is a recurring beginner instinct; the transformer is parallel *within* a block, sequential *across* blocks).

## M-series notes

This module is still light on compute

- Exercise 1's pre-norm vs post-norm comparison at `num_layers = 6, D = 64, T = 32` is a few minutes per run on CPU; comfortable on MPS.
- Exercise 2's strip-residuals study at `num_layers = 8` is the first configuration big enough that MPS starts paying off — about 2× over CPU at this size.
- Exercise 4's parameter-budget comparison is also CPU-comfortable but a good place to start using MPS as practice for Module 10.

The clean notebook's `train_tiny_transformer(..., device="auto")` helper moves the model and minibatches to MPS when available. Use `device="cpu"` if you want to compare CPU behavior explicitly.

---
## Reading

Primary:

- **Vaswani et al., "Attention Is All You Need" (2017), §3.** The block structure is defined in figure 1 and §3.1. Note that the 2017 paper uses post-norm; if you read it as an intro, mentally translate "norm after sublayer" into "norm before sublayer" because every modern reading you'll do uses pre-norm.
- **Ba, Kiros, Hinton, "Layer Normalization" (2016).** The original LayerNorm paper. Short, direct, and worth reading once for the motivation contrast against batch normalization.
- **Karpathy, "Let's build GPT: from scratch, in code, spelled out"** (YouTube). The block-assembly section walks through this same composition step by step.

Secondary:

- **Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020).** The paper that established that pre-norm trains more stably than post-norm. Argues theoretically and empirically; the figure showing post-norm needing warmup and pre-norm not is the headline.
- **Anthropic, "A Mathematical Framework for Transformer Circuits" (2021), introductory sections.** Frames the residual stream as a communication bus that every sublayer reads from and writes to. The conceptual model that underlies most of mechanistic-interpretability research.
- **He et al., "Deep Residual Learning for Image Recognition" (2015).** The original residual-connection paper, in computer vision. Predates the transformer by two years; the same insight ("training deep networks fails without identity shortcuts") drives both architectures.

Optional:

- **Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).** RMSNorm — a simplified LN that drops the mean-subtraction step. Used by Llama and many recent transformers; a few percent faster, no quality loss in practice.
- **Press et al., "Using the Output Embedding to Improve Language Models" (2017).** The case for tied input/output embeddings — saves parameters at no quality cost. We don't tie weights in this course for clarity; this is the paper that establishes you can.

## Deliverable checklist

- [ ] All tests in `tests/test_transformer.py` pass.
- [ ] Notebook: `notebooks/clean/09-transformer-block.ipynb`. Work through pre-vs-post norm, residual ablations, shape checks, and parameter-budget sanity checks.
- [ ] You can explain — out loud, without notes — why residual connections make deep transformers trainable, in both the gradient-flow and residual-stream framings.
- [ ] You can explain — out loud, without notes — what LayerNorm normalizes over, and why batch size doesn't affect its output.
- [ ] You can explain — out loud, without notes — the difference between pre-norm and post-norm, and why pre-norm is the modern default.

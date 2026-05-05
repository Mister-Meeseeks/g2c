# Module 05 — Embeddings and positions

> **Question this module answers:** *How do discrete symbols become meaning-like vectors, and how does order get in?*

![From token IDs to meaning-like vectors: text → token IDs → embedding lookup → add positional information (learned, sinusoidal, or RoPE) → model-ready vectors. Embeddings become a learned geometry; positions tell the model where each token is.](05-embeddings/Module05-Hero.png)

The last module handed us integer IDs. This module turns them into vectors. The embedding lookup itself is a one-line table indexing — the lesson is what the table learns and which of the three positional schemes you use to break the bag-of-tokens symmetry.

---
## Prerequisites

### Math

- **Trigonometric basics.** `sin(0) = 0`, `cos(0) = 1`. Sin and cos are bounded in `[−1, 1]`. Sin and cos at multiple frequencies. Nothing more exotic than what's in a high school trig review.
- **2D rotation matrices.** A rotation by angle θ takes `(x, y)` to `(x cos θ − y sin θ, x sin θ + y cos θ)`. Composing two rotations adds their angles: `R(α) · R(β) = R(α + β)`.
- **Dot product as a measure of alignment.** `q · k` is large when q and k point the same direction; zero when orthogonal; negative when opposite. Attention scores are dot products.
### Programming

- **PyTorch**  Tensor indexing, `torch.outer`, and broadcasting.

---
## Where this fits

After Module 04 you can turn text into a sequence of integer token IDs. After this module you can turn that sequence into a sequence of vectors that a neural network can actually use.

Two things have to happen:

1. **Each token gets a vector.** The same trick we used for biases and weights in earlier modules, just bigger and indexed by token ID.
2. **Each position gets distinguishable.** Without position info, a transformer is order-blind: `dog bites man` and `man bites dog` produce identical attention patterns because they're the same multiset of tokens. We need to inject "I am at position m" into each input.

## The big idea

A token embedding is one of the simplest ideas in deep learning:

```
  Embedding table   weight: (vocab_size, embedding_dim)

  ID      Vector
   0   →  [ 0.12, -0.44,  0.91, ... ,  0.05 ]
   1   →  [-0.33,  0.18,  0.62, ... ,  0.71 ]
   2   →  [ 0.55, -0.22, -0.18, ... ,  0.40 ]
   ...
   V-1 →  [-0.07,  0.65,  0.32, ... , -0.13 ]

  Forward:   ids   = [3, 1, 7]
             output = (3, embedding_dim) tensor — rows 3, 1, 7 of weight
```

That's it. The vector for each token is a learnable parameter — initialized random, updated by gradient descent like any other parameter. The forward pass is `weight[ids]`.

What makes this work is what gets *learned*. After training on enough text, the rows of `weight` arrange themselves into a geometry that reflects how tokens are used. Tokens that occur in similar contexts end up with similar vectors. Famously: word2vec's `king − man + woman ≈ queen`. The model never told that king is to man as queen is to woman; the geometry just emerged from co-occurrence statistics.

![Embedding geometry: a 2D projection of the learned embedding table shows tokens clustered by meaning (animals, vehicles, countries, cities), nearest neighbors of "king" by cosine similarity, and vector arithmetic (king − man + woman ≈ queen, Paris − France + Italy ≈ Rome).](05-embeddings/Module05-Geometry.png)

*Why we care about the embedding table beyond "it's a lookup." After training, semantically related tokens end up near each other (the clusters), and meaningful relationships line up as vector offsets (the analogies). Nothing in the loss told the model to do this — it falls out of co-occurrence statistics. Exercises 5 and 6 have you reproduce both phenomena: the cluster structure on a tiny model you train yourself, and the analogy property on pretrained GloVe vectors.*

LLM tokenizers have vocab sizes of 32k–200k and embedding dimensions of hundreds to thousands. The embedding table is one of the largest single parameter tensors in the model.

### Why positions need explicit encoding

A transformer's attention layer is symmetric in its inputs — it computes `softmax(QKᵀ/√d)V` and the only way order enters is through the position-dependent contents of Q, K, V themselves. If those contents have no position information baked in, the model literally cannot tell `dog bites man` from `man bites dog`. Both are evaluated as the same bag of three vectors.

![The bag-of-tokens problem: "dog bites man" and "man bites dog" produce identical bag-of-vectors when position is dropped, identical attention without positional encoding, and only become distinguishable once a learned, sinusoidal, or RoPE positional signal is added.](05-embeddings/Module05-BagOfTokens.png)

*The failure mode that motivates everything in the rest of this module. Without position information, the multiset `{dog, bites, man}` is what attention sees in either sentence — and the predicted next token is the same. The bottom of the figure shows the fix in two flavors: additive (concatenate or sum a position vector onto the token vector before attention) and rotary (rotate the query/key vectors inside attention, by an angle proportional to position).*

So we encode position into the token vectors themselves before they reach attention. There are two design choices: ADD a positional vector (Learned, Sinusoidal) or ROTATE the vectors (RoPE).

```
                Learned       Sinusoidal      Rotary (RoPE)
              ──────────────────────────────────────────────
Parameters?    yes            no              no

Extrapolates   no             yes (formula)   yes (formula)
beyond max?

Mechanism      ADD to         ADD to          ROTATE Q, K
               token emb      token emb       inside attention
               
Used in        BERT,          original        Llama, Mistral,
               GPT-2          Transformer     Qwen, modern LLMs
```

Modern LLMs all use RoPE. Older ones used learned or sinusoidal. The reason RoPE won is the relative-position property — explained in detail below.

### The sinusoidal trick

Vaswani et al. proposed encoding position with a fixed table of sines and cosines at exponentially decaying frequencies:

```
  PE[pos, 2i]   = sin( pos / 10000^(2i/d) )
  PE[pos, 2i+1] = cos( pos / 10000^(2i/d) )
```

Each pair of dimensions is `(sin, cos)` at a particular frequency. Low `i` → fast oscillation (small wavelength); high `i` → slow oscillation. The model gets a multi-resolution view of position: nearby positions look similar in the slow dimensions but differ in the fast ones; far-apart positions differ in both.

![Sinusoidal positional encoding visualized as many sine and cosine clocks ticking at multiple frequencies; each position is a snapshot across all clocks. The full PE table is a heatmap with banded structure across positions and dimensions.](05-embeddings/Module05-Sinusoidal.png)

*The mental image to keep. Each pair of dimensions is one (sin, cos) clock; low-`i` pairs tick fast (resolve nearby positions); high-`i` pairs tick slow (resolve coarse position over the whole sequence). The heatmap on the right is exactly the table you build in `SinusoidalPositionalEmbedding.__init__` — the bands are those clocks. Pos 0 is all-zero in the sine columns and all-one in the cosine columns; the test suite checks exactly this.*

This scheme has zero learnable parameters (the table is determined by the formula) and can be evaluated at any position, including beyond the longest sequence ever trained. The 2017 transformer paper used this; many models since have used learned positional embeddings instead, on the theory that learning is rarely worse than fixing.

### Rotary positional embeddings (RoPE)

The most important development in positional encoding since 2017. We'll cover attention in Module 7, but for now all you need to know is attention models care about token pair comparisons. For an ordered token pair (i, j), token i supplies the **query vector**, and token j supplies the **key vector**, and the dot product tells us how much token i pays attention to token j.

Instead of adding a position vector to the embedding vector, RoPE *rotates* the query and key vectors by an angle proportional to their position before the attention dot product. The key property is mechanical:

```
Without RoPE — positions are added vectors:
    q' = q + p_m            (p_m is the position embedding for position m)
    k' = k + p_n            (p_n likewise for n)
    q' · k' = q·k  +  q·p_n  +  p_m·k  +  p_m·p_n
                       ↑────── depends on absolute m, n ──────↑

With RoPE — positions are rotations:
    q' = R(m) · q           (R is a rotation matrix; angle ∝ m)
    k' = R(n) · k
    q' · k' = (R(m)q)ᵀ (R(n)k)
            = qᵀ R(m)ᵀ R(n) k
            = qᵀ R(n − m) k     ← only the RELATIVE offset (n − m) matters
```

The dot product of two RoPE'd vectors depends only on `(n − m)`. Token-pair attention scores are naturally functions of relative position, which is what we usually want — what matters in language is "this token is two words after that one", not "this token is at absolute position 437."

This is implemented as a per-position-pair 2D rotation, applied across all dimensions. The split-halves variant (Llama and friends): split the last dimension in half, treat dim `i` and dim `d/2 + i` as a 2-vector, rotate each pair by `m · θ_i` where `θ_i = 1/10000^(2i/d)`. The cos/sin tables are precomputed; the forward pass is `x · cos + rotate_half(x) · sin` — three tensor ops.

![RoPE as position-as-rotation: queries and keys at positions m and n get rotated by their position angles, and the dot product of the rotated vectors depends only on (n − m). Bottom panels show many-frequency cos/sin tables and the split-halves implementation recipe.](05-embeddings/Module05-Rotation.png)

*The whole RoPE story in one picture. Top: rotate q by angle proportional to m, rotate k by angle proportional to n; the dot product of the rotated vectors depends only on the relative offset (n − m), which is the property the test_rotary_relative_position_property test in tests/test_embeddings.py checks. Bottom-left: at any single position, every dimension pair is rotated at its own frequency, the same multi-frequency idea as sinusoidal. Bottom-right: the split-halves recipe — `x * cos + rotate_half(x) * sin` — is exactly what you implement in `RotaryEmbedding.forward`.*

We implement RoPE as a standalone module here and unit-test the relative-position property in isolation. In Module 07 it gets dropped into attention.

## Concepts to internalize

- **Embedding table = learnable lookup.** Forward is integer indexing; backward routes gradients only to the rows that were touched. PyTorch's autograd handles this for free.
- **The bag-of-tokens problem.** Without explicit position information, a transformer can't distinguish word orderings. Position must be injected somehow before attention.
- **Three positional schemes, three tradeoffs.** Learned (max_seq_len cap, more parameters), sinusoidal (no params, extrapolates by formula), rotary (no params, encodes relative position by construction).
- **Multi-frequency sinusoids.** The decaying-frequency table is what gives sinusoidal positional encodings their multi-scale resolution. Each pair of dimensions is one (sin, cos) frequency.
- **`R(m)ᵀ R(n) = R(n − m)`.** Rotations form a group; their composition adds angles. This is the algebraic identity that makes RoPE work.
- **`_rotate_half` is a notational trick.** The 2D rotation `(a, b) → (a cos θ − b sin θ, a sin θ + b cos θ)` can be written as `(a, b) ⊙ cos θ + (−b, a) ⊙ sin θ`. The `(−b, a)` part is what `_rotate_half` produces, applied across all paired dimensions in one tensor op.
- **Position 0 is the identity rotation.** `cos(0) = 1, sin(0) = 0`, so `RoPE(x, position=0) = x`. The test suite verifies this.

---
## What you'll build

Package: `g2c/embeddings/`

```python
class TokenEmbedding(Module):
    def __init__(self, vocab_size: int, embedding_dim: int): ...    # implemented
    def forward(self, ids: torch.Tensor) -> torch.Tensor: ...

class LearnedPositionalEmbedding(Module):
    def __init__(self, max_seq_len: int, embedding_dim: int): ...   # implemented
    def forward(self, seq_len: int) -> torch.Tensor: ...

class SinusoidalPositionalEmbedding(Module):
    def __init__(self, max_seq_len: int, embedding_dim: int): ...   # scaffolded (table)
    def forward(self, seq_len: int) -> torch.Tensor: ...

class RotaryEmbedding(Module):
    def __init__(self, max_seq_len: int, embedding_dim: int): ...   # scaffolded (cos/sin)
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

A typical use looks like this (built fully in Module 07, sketched here):

```
  ids: (batch, seq_len)
  ↓ TokenEmbedding(vocab_size, dim)
  tok_emb: (batch, seq_len, dim)
  +
  ↓ SinusoidalPositionalEmbedding(max_seq_len, dim)
  pos_emb: (seq_len, dim)              ← broadcasts across batch
  =
  x: (batch, seq_len, dim)             ← input to first transformer block
```

For RoPE, the addition is replaced by `RotaryEmbedding` applied inside attention to Q and K — that's a Module 07 concern.

## Scaffolding and how to run the tests

Tests live in `tests/test_embeddings.py`. Initial state: 7 passed, 24 failed.

```bash
.venv/bin/python -m pytest tests/test_embeddings.py             # run all module-05 tests
.venv/bin/python -m pytest tests/test_embeddings.py -x          # stop at first failure (recommended)
.venv/bin/python -m pytest tests/test_embeddings.py -k rotary   # only the RoPE tests
.venv/bin/python -m pytest tests/test_embeddings.py -v          # verbose
```

Open the working notebook copy with:

```bash
.venv/bin/python scripts/open_notebook.py 05
```

The clean scaffold lives at `notebooks/clean/05-embeddings.ipynb`; do your work in the generated `notebooks/solutions/05-embeddings.ipynb` copy.

Exercise 6 uses pretrained GloVe vectors. They are optional and larger than the normal setup assets:

```bash
./datasets.sh glove
```

The notebook skips the pretrained analogy section if `data/glove.6B.50d.txt` is missing.

## Exercises

The implementation path is the test suite above. The notebook starts with an executable test gate, then turns the implemented pieces into prediction, inspection, and explanation exercises.

1. **Token and learned position lookups.** Predict the output shape when token IDs shaped `(B, T)` index an embedding table shaped `(V, C)`. Verify that token vectors are literal rows from the table, then explain why a `(T, C)` learned position table broadcasts cleanly across a `(B, T, C)` batch.

2. **Inspect sinusoidal positional encoding.** Build a small sinusoidal table and check the position-zero values. Plot the heatmap and describe the multi-frequency pattern. The point is to see the formula as a table, not just to pass the constructor tests.

3. **Inspect RoPE table construction.** Instantiate a small RoPE table, inspect `cos[0]` and `sin[0]`, and explain the split-halves pairing convention. Make sure you can say why the cos/sin tables have shape `(max_seq_len, embedding_dim)`.

4. **Verify RoPE's relative-position behavior.** Run the notebook's rotated-dot-product check for several absolute position pairs with the same relative offset. Explain why the scores match and why that property is useful for attention.

5. **Train tiny co-occurrence embeddings.** Use the notebook's skip-gram scaffold: tokenize a small corpus, generate center/context pairs, train a tiny model, and plot the learned embedding rows in 2D. Look for any visible structure, but be honest about what a tiny corpus can and cannot learn.

6. **Try pretrained vector analogies.** `./datasets.sh glove` prepares `data/glove.6B.50d.txt` for this exercise, downloading and extracting it if needed. Load a small subset, check analogies like `king - man + woman`, and plot a 2D projection of the selected pretrained vectors. Compare that result with your tiny trained embeddings and explain what corpus scale changes.

7. **Compare positional schemes side-by-side.** Plot learned, sinusoidal, and RoPE tables as heatmaps. Identify which table is learned, which ones are fixed, and what visual pattern sinusoidal and RoPE share.

## Pitfalls to expect

- **Wrong axis when slicing sin/cos in sinusoidal.** Even dimensions should get sines; odd should get cosines. `weight[:, 0::2] = sin(angles)` and `weight[:, 1::2] = cos(angles)`. Mixing these up gives a tensor that's not a valid sinusoidal encoding.
- **`embedding_dim` odd.** Sinusoidal and RoPE both require an even dim. The `__init__`s raise a `ValueError`; if you instantiate at the wrong size, the error tells you what's wrong.
- **Forgetting `.requires_grad_(False)` (or just not setting requires_grad) on fixed tables.** The sinusoidal weight and the RoPE cos/sin tables are NOT parameters — they should not appear in `parameters()` and should not be updated by the optimizer.
- **`outer` vs. element-wise multiply.** Building the angles table requires `torch.outer(positions, inv_freq)` (or `positions[:, None] * inv_freq[None, :]`), not just `positions * inv_freq` (which would be element-wise on mismatched-shape tensors).
- **Wrong half-split convention for RoPE.** We use the *split-halves* variant: pair dim `i` with dim `d/2 + i`. The original RoPE paper paired dim `2i` with dim `2i+1` (interleaved). Both are valid rotations and produce the same end-to-end behavior in attention, but they're not interchangeable — `_rotate_half` is specifically the split-halves version.
- **Not handling `seq_len > max_seq_len`.** None of these implementations does — slicing past the end just truncates. In production you'd either error out or expand the table on the fly. For our scope, `max_seq_len` should be set generously.

## M-series notes

This module is light on compute.

- A 32k × 256 token-embedding table is ~8M parameters, ~32MB. Fits anywhere.
- Sinusoidal and RoPE tables for `max_seq_len = 4096, dim = 512` are around 8MB each. Trivial.
- The exercises that move some compute (training a tiny embedding model, t-SNE on the result) all fit comfortably on CPU; MPS isn't even necessary unless your corpus is large. Plan tens of seconds to a few minutes per exercise.

---
## Reading

Primary:

- **Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013).** The word2vec paper. Establishes that learned embeddings encode semantic structure.
- **Vaswani et al., "Attention is All You Need" (2017), §3.5.** The original sinusoidal positional encoding.
- **Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).** The RoPE paper. Skim — the math is heavy; the conceptual picture in this lesson is enough to use it.

Secondary:

- **Karpathy, "Neural Networks: Zero to Hero" lecture 2 ("makemore part 1").** Walks through token embeddings end to end on a tiny model.
- **Jay Alammar, "The Illustrated Word2vec."** Best visual intuition for how learned embeddings get their structure.
- **Eleuther blog posts on RoPE.** Several practical writeups on RoPE in production transformers.

## Deliverable checklist

- [ ] All tests in `tests/test_embeddings.py` pass.
- [ ] `notebooks/solutions/05-embeddings.ipynb`: tiny embedding model trained on a corpus, 2D visualization included.
- [ ] `notebooks/solutions/05-embeddings.ipynb`: `king − man + woman ≈ queen` reproduced on pretrained vectors; honest assessment of whether your tiny model reproduces any analogies.
- [ ] You can explain — out loud, without notes — why RoPE'd attention scores depend only on the relative position offset.

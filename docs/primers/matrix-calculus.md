# Matrix calculus primer

Just enough tensor calculus to know what autograd is doing on your behalf. Module 01 derives backprop for scalars, then Module 02 hands the machinery to PyTorch — and the single most important gradient in deep learning, the one for `y = x @ W`, is computed for you from then on. This primer derives it once, honestly, so it's a thing you know rather than a thing you trust.

The payoff is practical: shape-checking your own layers, debugging `grad` values in Module 03, and having solid footing under later arguments that quietly lean on gradients through matmul (weight scaling, the √D argument in Module 07).

## How to use this primer

Read sections 1–4 before or during Module 02. Section 5 is the centerpiece — work through it with a pencil once; it's four lines of index bookkeeping. Keep section 7's table as a back reference when you want to sanity-check what `.grad` should contain.

---

## Contents

1. [Gradients are shape-matched](#shapes)
2. [The upstream-gradient view of the chain rule](#chain)
3. [Warm-up: elementwise ops](#elementwise)
4. [Warm-up: sum and broadcast are each other's backward](#sum-broadcast)
5. [The matmul gradient, derived](#matmul)
6. [The shape trick (mnemonic, not proof)](#shape-trick)
7. [Backward rules for the course's ops](#table)
8. [Verify against autograd](#verify)
9. [Pitfalls](#pitfalls)

---

## <a id="shapes"></a>1. Gradients are shape-matched

Everything in training descends from one scalar: the loss `L`. For any tensor `T` that influenced `L`, the gradient `∂L/∂T` is a tensor **of exactly the same shape as `T`**, holding one number per entry: how much `L` moves per unit nudge of that entry.

```
W : (D, M)      →   ∂L/∂W : (D, M)
x : (N, D)      →   ∂L/∂x : (N, D)
b : (M,)        →   ∂L/∂b : (M,)
```

This is the invariant that makes everything below checkable. If a formula you derive produces a gradient whose shape doesn't match its tensor, the formula is wrong — no further analysis needed. PyTorch's `W.grad` having `W.shape` is not a convenience; it's the definition.

---

## <a id="chain"></a>2. The upstream-gradient view of the chain rule

Autodiff never asks an operation "what is your full derivative?" It asks something smaller. Every op `y = f(x)` in the graph receives the **upstream gradient** `g = ∂L/∂y` (same shape as `y`, already computed by everything after `f`) and must answer one question:

> Given `∂L/∂y`, what is `∂L/∂x`?

That's the tensor chain rule in operational form: each entry of `x` influences `L` through every entry of `y` it touches, so

```
∂L/∂x[i] = Σ_j  ∂y[j]/∂x[i] · g[j]        (sum over all outputs j that x[i] feeds)
```

The giant matrix `∂y[j]/∂x[i]` (the *Jacobian*) is never built. Each op has a closed-form shortcut for "Jacobian times upstream gradient" — that shortcut is what a `_backward` implements, and deriving matmul's shortcut is section 5. Module 01's scalar `Value` ops were the 1×1 special case of exactly this contract.

The other rule you already know from Module 01 carries over unchanged: **when a tensor feeds multiple consumers, its gradients add.** Nothing tensor-specific — it's the same `+=`.

---

## <a id="elementwise"></a>3. Warm-up: elementwise ops

For an op that maps each entry independently — `y[i] = f(x[i])` — the sum in section 2 collapses: `x[i]` touches only `y[i]`. The backward is just the local derivative, elementwise, times the upstream gradient:

```
y = x * c        →   ∂L/∂x = c ⊙ g                (⊙ = elementwise multiply)
y = x + z        →   ∂L/∂x = g,   ∂L/∂z = g       (add routes gradient through, unchanged)
y = relu(x)      →   ∂L/∂x = (x > 0) ⊙ g          (gradient passes where the unit was active)
y = a * b        →   ∂L/∂a = b ⊙ g,  ∂L/∂b = a ⊙ g
```

Shapes never change through an elementwise op, so the shape invariant is trivially satisfied. These are the tensor versions of Module 01's scalar rules, verbatim.

---

## <a id="sum-broadcast"></a>4. Warm-up: sum and broadcast are each other's backward

Two ops that *do* change shape, and they're mirror images:

**Sum.** `y = x.sum(dim=0)` with `x : (N, D)`, `y : (D,)`. Each `x[n, d]` contributes with coefficient 1 to exactly one output, `y[d]`. So the backward *broadcasts* the upstream gradient back over the summed dim:

```
∂L/∂x[n, d] = g[d]           i.e.  ∂L/∂x = g expanded to (N, D)
```

**Broadcast.** `y = x + b` with `x : (N, M)`, `b : (M,)`. The broadcast used `b[m]` in all `N` rows — one tensor, `N` consumers. Multiple consumers means gradients add:

```
∂L/∂b[m] = Σ_n g[n, m]       i.e.  ∂L/∂b = g.sum(dim=0)
```

That last line is the bias gradient in every `Linear` layer, and the pattern generalizes: **whatever dims broadcasting stretched on the way forward, the backward sums back down.** If your `b.grad` has the wrong shape, you forgot this sum. Notice the duality — sum's backward is a broadcast, broadcast's backward is a sum. They're transposes of each other in disguise, which is a preview of section 5.

---

## <a id="matmul"></a>5. The matmul gradient, derived

The centerpiece. Setup, with deliberately distinct letters for each dim:

```
x : (N, D)     a batch of N inputs, D features each
W : (D, M)     a weight matrix
y = x @ W      y : (N, M)
g = ∂L/∂y      g : (N, M)   — given, from upstream
```

Write matmul with indices — this is the only equation you need:

```
y[n, m] = Σ_d  x[n, d] · W[d, m]
```

**Gradient with respect to `W`.** Pick one weight entry, `W[d, m]`. Scan the equation: `W[d, m]` appears in `y[n, m]` for *every* `n` — same output column `m`, all batch rows — with coefficient `x[n, d]`. Apply section 2's sum-over-outputs:

```
∂L/∂W[d, m] = Σ_n  x[n, d] · g[n, m]
```

Now stare at the right-hand side: it sums over `n`, pairing the `(n, d)` entry of `x` with the `(n, m)` entry of `g`. A sum over the *row* index of both tensors is a matmul with the first operand transposed:

```
∂L/∂W = xᵀ @ g            (D, N) @ (N, M) → (D, M)  ✓ shape of W
```

**Gradient with respect to `x`.** Same procedure. `x[n, d]` appears in `y[n, m]` for every `m`, with coefficient `W[d, m]`:

```
∂L/∂x[n, d] = Σ_m  g[n, m] · W[d, m]
```

The sum is over `m` — the *column* index of both `g` and `W` — which is a matmul with the second operand transposed:

```
∂L/∂x = g @ Wᵀ            (N, M) @ (M, D) → (N, D)  ✓ shape of x
```

That's the whole derivation. Two things fell out of it that are worth saying aloud:

- **The batch sum in `∂L/∂W` wasn't a convention — it was forced.** `W` is *shared* across all `N` rows of the batch; shared across consumers means gradients add (section 2). Batch-summed weight gradients are the multiple-consumers rule from Module 01, wearing a batch dimension.
- **Each input's gradient is the upstream gradient times the *other* input, transposed.** Compare `y = a*b` in section 3: `∂L/∂a = b ⊙ g`. Matmul is the same "times the other operand" structure, with transposes doing the index bookkeeping that ⊙ did for free.

---

## <a id="shape-trick"></a>6. The shape trick (mnemonic, not proof)

In practice nobody re-derives section 5 at the keyboard. They use shapes. You need `∂L/∂W : (D, M)`, and the tensors on hand are `x : (N, D)` and `g : (N, M)`. There is exactly one way to matmul those two into a `(D, M)`:

```
(D, N) @ (N, M) → (D, M)      ⟹      ∂L/∂W = xᵀ @ g
```

Likewise `∂L/∂x : (N, D)` from `g : (N, M)` and `W : (D, M)`: only `g @ Wᵀ` fits. For matmul-family ops the shape trick essentially always lands on the right answer, and it's the fastest debugging tool you have.

Use it knowing what it is: a mnemonic that works because matmul gradients happen to be uniquely determined by shape when all dims are distinct. It is not a proof — and with square matrices (`N == D == M`) several wrong formulas *also* type-check, which is exactly when you fall back on section 5. When testing a hand-written layer, use distinct dims like `(2, 3) @ (3, 5)` so shape errors can't hide.

---

## <a id="table"></a>7. Backward rules for the course's ops

`g` is always the upstream gradient, shaped like the op's output.

| Forward | Backward | Note |
| --- | --- | --- |
| `y = x + z` | `∂x = g`, `∂z = g` | sum over any broadcast dims (§4) |
| `y = a * b` | `∂a = b ⊙ g`, `∂b = a ⊙ g` | elementwise |
| `y = x @ W` | `∂x = g @ Wᵀ`, `∂W = xᵀ @ g` | §5; batched: matmul over leading dims |
| `y = x.sum(dim)` | `∂x = g` broadcast over `dim` | |
| `y = x.mean(dim)` | `∂x = g / size(dim)`, broadcast | sum's rule ÷ count |
| `y = x.T` | `∂x = gᵀ` | transpose commutes with gradient |
| `y = relu(x)` | `∂x = (x > 0) ⊙ g` | |
| `y = tanh(x)` | `∂x = (1 − y²) ⊙ g` | note: uses the *output* |
| `y = E[ids]` (embedding lookup) | `∂E = 0`, then `∂E[ids[k]] += g[k]` | scatter-**add**: repeated ids accumulate — the multiple-consumers rule again |
| `L = cross_entropy(z, t)` | `∂z = (softmax(z) − onehot(t)) / N` | derived in the [probability primer](probability-cross-entropy.md) |

The embedding row is worth a beat: a token appearing twice in a batch means its embedding row fed two outputs, so its gradient contributions add. Same rule as everything else — and a common hand-rolled-layer bug when `+=` is written as `=`.

---

## <a id="verify"></a>8. Verify against autograd

Every claim above is checkable in six lines. This is also the template for gradient-checking any layer you ever hand-write:

```python
import torch

x = torch.randn(4, 3, requires_grad=True)   # N=4, D=3
W = torch.randn(3, 5, requires_grad=True)   # D=3, M=5
y = x @ W                                   # (4, 5)
loss = (y ** 2).sum()
loss.backward()

g = 2 * y                                   # ∂L/∂y for L = Σ y², by hand
assert torch.allclose(W.grad, x.T @ g)      # §5, first result
assert torch.allclose(x.grad, g @ W.T)      # §5, second result
```

If you change `loss` to anything else, `g` changes but both assertions keep holding — the matmul backward doesn't care what's downstream, which is the modularity that makes autodiff composable at all.

---

## <a id="pitfalls"></a>9. Pitfalls

**Forgetting the broadcast sum.** The #1 hand-derived-gradient bug: `∂L/∂b = g` instead of `g.sum(dim=0)` for a broadcast bias. The shape invariant catches it — `(N, M)` is not `(M,)` — *if* you check shapes.

**Square matrices hide transpose errors.** `g @ W` type-checks when `W` is square and is simply wrong. Test with distinct dims; the type system becomes your proof-checker.

**Transposing a product.** `(A @ B)ᵀ = Bᵀ @ Aᵀ` — order flips. Half of all shape-trick dead-ends are this identity misremembered.

**`.grad` on non-leaves.** PyTorch only populates `.grad` on leaf tensors (created directly, `requires_grad=True`). Intermediate results silently get `None` — call `.retain_grad()` on an intermediate before `backward()` when you want to inspect the `g` flowing through it.

**Comparing with `==`.** Gradient checks compare floats; use `torch.allclose(..., atol=1e-6)`, and loosen tolerance to ~1e-4 if you cross float32 reductions on MPS.

---

## What this primer doesn't cover

- **Full Jacobians and the ∂vector/∂matrix formalism.** Real layouts ("numerator vs denominator convention") matter for the general theory; the upstream-gradient view sidesteps the entire question, which is why autodiff uses it.
- **Gradients of softmax alone.** The useful object is softmax *fused with* cross-entropy, whose gradient is famously clean — derived in the [probability & cross-entropy primer](probability-cross-entropy.md).
- **Second derivatives, Hessians, gradients through matrix decompositions.** Nothing in the course needs them.
- **Weight initialization theory.** Why `randn * 0.02`-style scales matter is a training-dynamics question, not a calculus one; the shape-and-variance reasoning appears where the course needs it (Modules 03 and 07).

# PyTorch primer

A focused reference for the PyTorch surface area this course actually uses. Roughly 50 operations across tensors, autograd, and `nn` — enough to implement everything from Module 02 through Module 15 without ever leaving this page. (Modules 16–20 pivot to inference with pretrained models via Ollama / llama.cpp / MLX, where PyTorch drops out of the picture.)

## How to use this primer

Two intended modes:

- **First read.** If you've never used PyTorch, read top-to-bottom once. The order is conceptual: tensors → indexing → shapes → math → autograd → `nn`. By the end you'll know what's available.
- **Back reference.** Once modules are underway, jump straight to a section when you hit "I know what I want to compute, what's the API?" Each section is a self-contained mini-cheat-sheet.

Examples are minimal and runnable. Shape annotations (`# (B, T, D)`) appear in comments wherever shape arithmetic matters — this is the single most common source of confusion. Pitfalls are called out inline next to the operation that triggers them, not collected at the end, so a back-reference lookup catches the gotcha in the same glance as the API.

This primer does *not* try to teach PyTorch comprehensively. For anything not here (`DataLoader`, distributed training, JIT, advanced indexing tricks, custom CUDA kernels) consult the official PyTorch docs.

---

## Contents

1. [Mental model](#mental-model)
2. [Creating tensors](#creating-tensors)
3. [Tensor metadata](#tensor-metadata)
4. [Indexing and slicing](#indexing-and-slicing)
5. [Shape manipulation](#shape-manipulation)
6. [Math: elementwise, reductions, matmul](#math)
7. [Broadcasting](#broadcasting)
8. [Autograd](#autograd)
9. [Building neural nets: `nn.Module` and `nn.Parameter`](#nn-module)
10. [Common functional ops](#functional)
11. [Devices and MPS](#devices)
12. [Random and reproducibility](#random)
13. [Common pitfalls](#pitfalls)

---

## <a id="mental-model"></a>1. Mental model

A **tensor** is a multi-dimensional array of numbers, like a NumPy `ndarray`, but with two superpowers:

1. **Autograd metadata.** If you set `requires_grad=True`, every operation on the tensor is recorded in a graph. Calling `.backward()` on a scalar walks that graph in reverse and fills in `.grad` on every leaf tensor that fed into it. This is the entire deep-learning training loop, packaged.
2. **Device placement.** A tensor lives on a device — CPU, CUDA GPU, or (on M-series Macs) MPS. Operations between tensors must be on the same device.

Everything else — `nn.Linear`, `nn.Module`, optimizers — is built on top of these two ideas. PyTorch's API deliberately mirrors NumPy's so it feels familiar, but the implementation is its own.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()      # y is a scalar tensor; the graph (x → square → sum) is recorded
y.backward()           # fills in x.grad
print(x.grad)          # tensor([2., 4., 6.])  — d(x²)/dx = 2x at each x_i
```

That four-line example is, in microcosm, every training step in the course.

---

## <a id="creating-tensors"></a>2. Creating tensors

```python
torch.tensor([1, 2, 3])              # from Python data; dtype inferred (int64 here)
torch.tensor([1.0, 2.0])             # float32 (the default float dtype)
torch.tensor(data, dtype=torch.float, device="mps")  # specify both

torch.zeros(3, 4)                    # (3, 4) of zeros, float32
torch.ones(3, 4)                     # (3, 4) of ones
torch.full((2, 3), 7.5)              # (2, 3) all 7.5
torch.empty(2, 3)                    # (2, 3) uninitialized — values are garbage
torch.eye(4)                         # (4, 4) identity

torch.arange(10)                     # [0, 1, ..., 9]   — like Python range
torch.arange(2, 10, 2)               # [2, 4, 6, 8]
torch.linspace(0, 1, 5)              # [0.0, 0.25, 0.5, 0.75, 1.0]

torch.randn(3, 4)                    # (3, 4) standard normal
torch.rand(3, 4)                     # (3, 4) uniform [0, 1)
torch.randint(0, 100, (3, 4))        # (3, 4) integers in [0, 100)

# `_like` variants copy shape/dtype/device from an existing tensor
torch.zeros_like(x)
torch.ones_like(x)
torch.randn_like(x)
torch.full_like(x, 7.5)
```

**Pitfall: `torch.tensor` vs `torch.Tensor`.** Lowercase `torch.tensor(data)` constructs a tensor *from data*. Uppercase `torch.Tensor(2, 3)` is the class constructor and treats integer args as *shape* — `torch.Tensor(3)` gives you a length-3 tensor of garbage, not the scalar 3. Always use lowercase `torch.tensor` unless you have a specific reason.

---

## <a id="tensor-metadata"></a>3. Tensor metadata

```python
x = torch.randn(2, 3, 4)

x.shape           # torch.Size([2, 3, 4])  — tuple-like
x.shape[0]        # 2
x.size()          # same as x.shape (older API)
x.size(1)         # 3 — size of dim 1
x.ndim            # 3 — number of dimensions
x.numel()         # 24 — total number of elements
x.dtype           # torch.float32
x.device          # device(type='cpu')
x.requires_grad   # False
```

**Common dtypes.** `torch.float32` (default float, alias `torch.float`), `torch.float64` (alias `torch.double`), `torch.int64` (default int, alias `torch.long`), `torch.int32` (alias `torch.int`), `torch.bool`. Token IDs are almost always `torch.long`; activations and weights are `torch.float32`.

```python
x.float()                            # cast to float32
x.long()                             # cast to int64
x.to(torch.float64)                  # general cast
x.to(dtype=torch.float, device="mps")  # cast and move in one call
```

Going out to plain Python:

```python
x.item()          # ONLY works on a scalar tensor — returns a Python number
x.tolist()        # full nested Python list
x.numpy()         # zero-copy view as numpy array (CPU only)
x.detach().cpu().numpy()    # the safe incantation: drop autograd, move to CPU, then numpy
```

**Pitfall:** `.item()` raises if the tensor has more than one element. For a scalar use `.item()`; for a vector you wanted the sum/mean of, reduce first.

---

## <a id="indexing-and-slicing"></a>4. Indexing and slicing

NumPy-style indexing works as you'd expect.

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x[0]              # row 0:        tensor([0, 1, 2, 3])
x[0, 2]           # element:      tensor(2)
x[:, 1]           # column 1:     tensor([1, 5, 9])
x[1:, :2]         # submatrix:    tensor([[4, 5], [8, 9]])
x[..., -1]        # last dim, last position; ... means "all preceding dims"
```

**Integer-array (advanced) indexing.** Pass a list/tensor of indices to gather rows or arbitrary positions:

```python
ids = torch.tensor([0, 2, 0])
x[ids]            # rows 0, 2, 0 — shape (3, 4). The embedding lookup, basically.
```

**Boolean masking.**

```python
mask = x > 5
x[mask]           # 1-D tensor of all elements where mask is True
x[mask] = 0       # in-place assignment through a mask
```

**`gather` for index-along-dim.** When you want one element per row at a different column per row:

```python
# logits: (B, V), targets: (B,) of token ids
log_probs = logits.log_softmax(dim=-1)               # (B, V)
chosen = log_probs.gather(1, targets.unsqueeze(1))   # (B, 1) — log_probs[i, targets[i]]
chosen = chosen.squeeze(1)                            # (B,)
```

`gather(dim, index)` reads `out[i, j] = input[i, index[i, j]]` along the named dim. Index tensor must have the same number of dims as the input.

---

## <a id="shape-manipulation"></a>5. Shape manipulation

Most bugs in the course are shape bugs. These are the operations that move dimensions around without copying data when possible.

```python
x = torch.arange(24)                 # (24,)

x.view(2, 3, 4)                      # (2, 3, 4) — requires contiguous memory
x.reshape(2, 3, 4)                   # (2, 3, 4) — works even if not contiguous (may copy)
x.view(-1, 4)                        # (6, 4) — `-1` means "infer this dim"

x = torch.randn(2, 3, 4)
x.transpose(0, 1)                    # swap dims 0 and 1 → (3, 2, 4)
x.permute(2, 0, 1)                   # arbitrary reorder → (4, 2, 3)
x.T                                  # 2D only: matrix transpose

x.unsqueeze(0)                       # insert a dim of size 1 at position 0 → (1, 2, 3, 4)
x.unsqueeze(-1)                      # at the end → (2, 3, 4, 1)
x.squeeze(0)                         # remove dims of size 1 (specific dim or all if no arg)

# Combining tensors
torch.stack([a, b, c], dim=0)        # NEW dim. Inputs must all have the same shape.
torch.cat([a, b, c], dim=0)          # CONCATENATE along an existing dim. Other dims must match.
```

**`stack` vs `cat` — the most common mix-up.** `stack` adds a new dimension; `cat` extends an existing one. Three (4,) tensors → `stack` gives (3, 4); `cat` gives (12,).

**Pitfall: `view` requires contiguous memory.** After `transpose` or `permute`, the underlying storage layout no longer matches the logical shape. `view` will raise; either call `.contiguous()` first or use `reshape` (which calls `contiguous` internally if needed).

```python
x = torch.randn(2, 3).transpose(0, 1)
# x.view(6) → RuntimeError
x.contiguous().view(6)               # OK
x.reshape(6)                         # also OK
```

---

## <a id="math"></a>6. Math: elementwise, reductions, matmul

**Elementwise.** Standard operators broadcast and produce new tensors:

```python
a + b, a - b, a * b, a / b           # arithmetic
a ** 2                               # power
a @ b                                # matrix multiply (NOT elementwise — see below)

torch.exp(x), torch.log(x), torch.sqrt(x)
torch.sin(x), torch.cos(x), torch.tanh(x)
torch.abs(x), torch.sign(x)
torch.relu(x), torch.sigmoid(x)
torch.where(cond, a, b)              # elementwise: cond ? a : b
```

**Reductions.** All take an optional `dim=` and `keepdim=`:

```python
x = torch.randn(3, 4)

x.sum()                              # scalar — sum of all elements
x.sum(dim=0)                         # (4,) — sum along rows, leaves columns
x.sum(dim=1)                         # (3,) — sum along columns, leaves rows
x.sum(dim=1, keepdim=True)           # (3, 1) — keeps the reduced dim as size 1

x.mean(), x.mean(dim=0)
x.max(), x.min()                     # scalars — reduce everything
x.max(dim=0)                         # named tuple (values, indices) — both (4,) here!
x.argmax(dim=-1)                     # just the indices
x.norm(dim=-1)                       # L2 norm along last dim
```

**`dim=` is the dim being *consumed*.** `sum(dim=0)` removes dim 0 from the shape. `keepdim=True` is a common need when you're about to broadcast the result back against the original (e.g., subtracting the row max for a stable softmax).

**Matmul.** Three equivalent ways:

```python
a @ b                                # cleanest, preferred
torch.matmul(a, b)
a.matmul(b)
```

For 2D tensors this is plain matrix multiply: `(M, K) @ (K, N) → (M, N)`. For higher dims, the leading dims broadcast and the last two are matmul'd: `(B, T, D) @ (D, V) → (B, T, V)`. This is the workhorse of every transformer forward pass.

**`torch.outer(a, b)`** for vector outer product: `(N,) × (M,) → (N, M)`.

---

## <a id="broadcasting"></a>7. Broadcasting

When two tensors have different shapes, PyTorch tries to make them compatible by aligning shapes from the *right* and applying these rules per dim:

1. If sizes are equal, no change.
2. If one is `1`, it stretches to match the other.
3. If one is missing (shorter shape), prepend `1`s to its shape until they're the same length.
4. Otherwise, error.

```python
a = torch.randn(3, 4)
b = torch.randn(4)
a + b                                # b is treated as (1, 4), stretched to (3, 4). Result (3, 4).

a = torch.randn(3, 1, 4)
b = torch.randn(2, 4)                # interpreted as (1, 2, 4)
a + b                                # → (3, 2, 4)

a = torch.randn(3, 4)
b = torch.randn(3,)
a + b                                # ERROR — b becomes (1, 3), can't stretch (1, 3) to match (3, 4)
                                     # If you wanted column-wise: b.unsqueeze(1) → (3, 1) → broadcasts.
```

**The alignment-from-the-right rule is the single thing to internalize.** Most "shape doesn't broadcast" errors come from forgetting that a `(B,)` tensor is *not* the same as a `(B, 1)` tensor when the other operand is `(B, D)`. Use `unsqueeze` to make intent explicit.

---

## <a id="autograd"></a>8. Autograd

The whole point of PyTorch. Recap of the mental model:

- A tensor with `requires_grad=True` participates in the autograd graph.
- Every op on such a tensor records itself in the graph.
- Calling `.backward()` on a scalar tensor walks the graph in reverse and accumulates gradients into `.grad` on every leaf tensor that fed into it.
- "Leaf" = a tensor created directly (e.g., a `Parameter`), not as the result of an op.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)                        # tensor([2., 4., 6.])
```

Setting / toggling `requires_grad`:

```python
x = torch.randn(3, requires_grad=True)
x.requires_grad_(False)              # in-place setter (note trailing underscore)
x.requires_grad = True               # also works
```

**`torch.no_grad()` — disable graph tracking.** Use it for parameter updates inside an optimizer step, for evaluation, and for any computation whose gradient you don't need.

```python
with torch.no_grad():
    for p in model.parameters():
        p -= lr * p.grad             # update without recording into the graph
```

**`.detach()` — return a tensor that shares storage but is detached from the graph.** Useful when you want to keep using a tensor's value but not have its computation contribute to a gradient.

```python
target = model(x).detach()           # use the model's output as a fixed target
```

**Gradient accumulation.** `.grad` *accumulates* across `.backward()` calls — you must zero it before the next iteration, or gradients pile up:

```python
optimizer.zero_grad()                # standard incantation
loss.backward()
optimizer.step()
```

**`backward()` only works on a scalar by default.** If you want gradients of a vector output, either reduce it (`.sum()`, `.mean()`) before calling `.backward()`, or pass an explicit `gradient=` of the same shape. The training loop always reduces to a scalar loss first — that's what makes the gradient well-defined.

---

## <a id="nn-module"></a>9. Building neural nets: `nn.Module` and `nn.Parameter`

PyTorch's `nn` package gives you a way to package learnable parameters with the function that uses them.

```python
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x @ self.weight + self.bias

layer = MyLayer(128, 10)
y = layer(x)                         # calls forward() — never call layer.forward(x) directly
```

**Two key behaviors of `nn.Module`:**

1. Any `nn.Parameter` assigned as an attribute (or any submodule whose own params include them) is automatically registered. `layer.parameters()` returns all of them — that's what you pass to the optimizer.
2. `__call__` (i.e., `layer(x)`) does some bookkeeping (hooks, training/eval mode) before invoking `forward`. Always call the module, not `.forward()`.

**`nn.Parameter`.** A subclass of `Tensor` with `requires_grad=True` by default, and special handling so `nn.Module` finds it. Use it for any tensor that should be learned.

```python
self.w = nn.Parameter(torch.randn(d))    # learnable
self.const = torch.randn(d)              # NOT learnable — won't show up in .parameters()
```

**Submodules compose.** Assigning a `Module` as an attribute registers its parameters too:

```python
class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)   # nn.Linear is itself an nn.Module
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = TwoLayer()
list(model.parameters())                  # 4 tensors: fc1.weight, fc1.bias, fc2.weight, fc2.bias
```

> Note: in this course, you build `Linear`, `ReLU`, etc. yourself in Module 03 — so your modules subclass your own `Module` base class, not `nn.Module`. The pattern is identical though, which is the point.

**Moving a model to a device.**

```python
model.to("mps")                      # moves all parameters to the MPS device in-place
```

**Train vs eval mode.** `model.train()` and `model.eval()` set a flag that affects layers like Dropout and BatchNorm. The course doesn't lean on these heavily but it's worth knowing the toggle exists.

---

## <a id="functional"></a>10. Common functional ops

`torch.nn.functional` (conventionally imported as `F`) is the stateless cousin of `nn`. Where `nn.ReLU()` is a module, `F.relu(x)` is a plain function. Use functional forms when there's no learnable state.

```python
import torch.nn.functional as F

F.relu(x)
F.gelu(x)                            # smoother ReLU; used in transformer FFNs
F.sigmoid(x)
F.tanh(x)

F.softmax(x, dim=-1)                 # also: torch.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)             # ALWAYS prefer over softmax(x).log() — numerically stable

F.cross_entropy(logits, targets)     # combines log_softmax + NLL; standard classification loss
F.mse_loss(pred, target)
F.logsigmoid(x)                      # log(sigmoid(x)) but stable; used in DPO
```

**Masking.** `masked_fill` replaces values where a boolean mask is True. The standard use is causal masking in attention:

```python
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()    # upper triangular True
scores.masked_fill_(mask, float("-inf"))                   # in-place; future positions become -inf
attn = scores.softmax(dim=-1)                              # softmax of -inf is 0 → no attention to future
```

**Sampling from a categorical distribution.**

```python
probs = logits.softmax(dim=-1)                # (V,)
next_id = torch.multinomial(probs, num_samples=1)         # tensor([id])
```

**Top-k.**

```python
values, indices = torch.topk(x, k=5, dim=-1)   # top 5 values and their positions
```

---

## <a id="devices"></a>11. Devices and MPS

Apple Silicon GPU support comes through the **MPS** (Metal Performance Shaders) backend.

```python
torch.backends.mps.is_available()    # True on M1/M2/M3/M4
torch.backends.mps.is_built()        # True if your PyTorch was built with MPS

device = "mps" if torch.backends.mps.is_available() else "cpu"

x = torch.randn(1000, 1000, device=device)
y = x @ x                            # runs on the GPU
```

**Standard idiom for portable code:**

```python
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device()
model = MyModel().to(device)
x = x.to(device)
```

**Tensors must be on the same device** for an op to work. Mixing CPU and MPS tensors raises. Move both to the same device before combining.

**MPS quirks worth knowing.**

- Some less-common ops fall back to CPU silently (or raise). If you hit a `NotImplementedError` for MPS, that op isn't supported yet — move the relevant tensor to CPU for that one op, then back.
- MPS uses the unified memory pool, so there's no separate "GPU memory" to worry about — but very large allocations can still OOM your Mac.
- `torch.float64` is not well-supported on MPS. Stay in `float32`.

---

## <a id="random"></a>12. Random and reproducibility

```python
torch.manual_seed(42)                # seeds CPU RNG (and MPS, in recent PyTorch)
torch.randn(3)                       # reproducible from this point

# For independent streams, use a Generator:
g = torch.Generator().manual_seed(42)
torch.randn(3, generator=g)
```

**Seeding for cross-device reproducibility.** The MPS RNG is separate from the CPU RNG in older PyTorch versions. If you need bit-exact repro, generate randomness on CPU and `.to("mps")` after.

---

## <a id="pitfalls"></a>13. Common pitfalls

**In-place ops.** Methods ending in `_` modify the tensor in place: `x.add_(1)`, `x.zero_()`, `x.requires_grad_(True)`. They return the same tensor. Useful for parameter updates and known-safe spots; risky inside autograd graphs because they can break the saved values needed for backward.

**Leaf tensors and `.grad`.** Only leaf tensors with `requires_grad=True` get a `.grad` field populated. If you compute `y = x + 1` and then call `y.backward()`, `y.grad` will be `None` even though it has a gradient — only `x.grad` is filled. To force a non-leaf tensor to retain its gradient, call `y.retain_grad()` before backward.

**Contiguity.** After `transpose` or `permute`, the tensor is still logically the right shape but its memory layout is non-contiguous. `view` rejects non-contiguous tensors; either call `.contiguous()` first or use `reshape`.

**`.item()` is CPU-blocking.** Calling `.item()` forces a sync from GPU to CPU. Inside a training loop, calling `loss.item()` every step is fine (it's one number); calling it on a large tensor or in a tight inner loop is a perf cliff.

**Integer vs float dtypes.** `torch.tensor([1, 2, 3])` is `int64` (because the input is Python ints). Multiplying it by a float tensor will work via promotion, but autograd doesn't run on integer tensors — `requires_grad=True` raises if dtype is integral. Use `torch.tensor([1.0, 2.0, 3.0])` or `.float()`.

**Forgetting `optimizer.zero_grad()`.** PyTorch accumulates gradients across `.backward()` calls. Without zeroing, every step compounds the previous step's gradient. Symptom: training is stable for a few steps, then explodes.

**Forgetting `torch.no_grad()` for the update step.** If you write your own optimizer and update parameters without `no_grad`, the update tensor op gets recorded in the graph, and the next forward+backward tries to differentiate through it. Wrap the update.

**Shape mismatches in cross-entropy.** `F.cross_entropy(logits, targets)` expects `logits` of shape `(N, C)` and `targets` of shape `(N,)` with class indices in `[0, C)` — *not* one-hot. For sequence outputs, flatten: `F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))`.

**`stack` vs `cat`.** `stack` adds a dim; `cat` extends one. Picking the wrong one is one of the most common silent shape bugs.

---

## What this primer doesn't cover

- `torch.utils.data.DataLoader`, `Dataset`, samplers — the course uses simple loops and explicit batching instead.
- Distributed training (`torch.distributed`, DDP, FSDP).
- `torch.compile`, JIT, TorchScript, ONNX export.
- Custom CUDA / Triton kernels.
- Mixed precision (`torch.amp`, `bfloat16` training).
- Hooks, parametrizations, quantization, pruning.

When you need any of the above, the official PyTorch docs are excellent. For everything in modules 02–15, the surface area on this page is enough.

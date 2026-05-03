# Module 01 — Scalar autodiff

> **Question this module answers:** *How does the model learn?*

![Training loop summary: parameters → forward pass → loss → backward pass (autodiff) → gradients → parameter update, repeat.](01-autodiff/Module01-Hero.png)

Module 01 builds the **backward pass** — the autodiff machinery that converts a forward computation into gradients with respect to every parameter. The forward pass, the loss, and the parameter update are all straightforward arithmetic; the gradient computation is the keystone, and it's what every later module's training loop will lean on.

---
## Prerequisites

The math, CS, and programming concepts this module uses. If any feel rusty, the linked refreshers will get you back up to speed in 10–30 minutes each.
### Math

- **Single-variable derivatives.** You should be able to differentiate something like `f(x) = x³ + 2x` by hand without thinking about it. Refresher: Khan Academy "Differentiation rules" or Paul's Online Math Notes.
- **The chain rule.** `d/dx[f(g(x))] = f'(g(x)) · g'(x)`. The single most important rule in deep learning — every gradient propagation step is one application of it. Refresher: 3Blue1Brown's "Backpropagation calculus" video (10 min) gives the geometric picture.
- **Partial derivatives.** `d/dx[xy] = y`, `d/dy[xy] = x`. Used implicitly because every binary op has two inputs and you need a derivative with respect to each.
- **Standard derivatives to have memorized.** Power rule: `d/dx[xⁿ] = n·xⁿ⁻¹`. Exponential: `d/dx[eˣ] = eˣ`. Logarithm: `d/dx[ln x] = 1/x`. Tanh: `d/dx[tanh x] = 1 − tanh²x`.
### Computer science

- **DAGs and topological sort.** A directed acyclic graph; a topological ordering places each node after all its parents. Refresher: any algorithms textbook, or just the Wikipedia article. The standard recursive-DFS algorithm is what you'll write here.
- **Recursion.** Comfortable enough to write a depth-first traversal by hand.
### Programming

- **Python dunder methods.** `__add__`, `__mul__`, `__pow__`, `__radd__`, etc. — the protocol for overloading operators on a class. Refresher: Python language reference §3.3.7 ("Emulating numeric types").
- **Closures.** Each operation defines a `_backward` function that captures local variables (the parents and the output Value). If "closures in Python" is fuzzy, skim Fluent Python's chapter on them or any short tutorial — they're central to the autodiff implementation pattern.

---
## Why we start here

A neural network is a function with millions to billions of adjustable knobs. Training it is the process of repeatedly nudging each knob in the direction that makes the output less wrong. To do that, you need the partial derivative of the loss with respect to every knob. For any nontrivial network, computing those derivatives by hand would be hopeless — there are too many, and they share structure.

**Reverse-mode automatic differentiation** is the algorithm that solves this. It computes all those derivatives in one pass over a graph, in time proportional to the forward pass. Modern deep learning frameworks (PyTorch, JAX, MLX, TensorFlow) are, at their core, fast and well-engineered implementations of this single idea applied to tensors.

We start with the *scalar* version. No tensors, no broadcasting, no GPU. Just numbers and a small Python class. The reason: every confusing thing about deep learning training — gradient flow, vanishing gradients, the role of nonlinearities, why some architectures train and others don't — is easier to see when the machinery is laid bare. Once you've built this from scratch, `loss.backward()` is no longer magic. It's something you wrote.

This module is the analogue of NAND gates in *NAND to Tetris*. Everything else stacks on top.

## The big idea

Every computation can be drawn as a graph:

```
   a ──┐
       ├── (*) ── d ──┐
   b ──┘              ├── (+) ── e
                  c ──┘
```

Here `d = a * b` and `e = d + c`. Each node holds a value (computed forward) and a gradient (computed backward). The gradient at a node is `de/dnode` — how much the final output `e` changes per unit change in the node's value.

**The forward pass** computes node values by walking the graph from inputs to outputs, applying each operation.

**The backward pass** computes gradients by walking the graph in reverse. At the output, `de/de = 1`. For each operation, you know how to push gradient from the output to the inputs using the chain rule:

- For `e = d + c`: `de/dd = 1`, `de/dc = 1`
- For `d = a * b`: `dd/da = b`, `dd/db = a`, so `de/da = de/dd · b`, `de/db = de/dd · a`

Each operation type carries a small **local rule** for how to distribute the incoming gradient to its inputs. That's it. Stack these rules across a graph of millions of nodes and you've reproduced the heart of every modern deep learning library.

Three subtleties worth highlighting:

1. **Topological order matters.** When you compute the gradient of a node, you need the gradients of all its downstream consumers to already be finalized. The standard approach is a topological sort of the graph, then iterate in reverse.

2. **Gradient accumulation.** A node can be used in multiple downstream places. Each use contributes to the node's total gradient, so backward must *add* to a node's gradient field rather than overwrite it. (Hence the `.zero_grad()` calls you'll see later.)

   The classic case is a diamond, where one node feeds two paths that later merge:

   ```
              a
             / \
            /   \
           b     c        b = a * x      c = a * y
            \   /
             \ /
              f           f = b + c

      df/da = (df/db)(db/da) + (df/dc)(dc/da)
            = the contribution along the b-path PLUS the contribution along the c-path
   ```

   If `_backward` writes `self.grad = ...` instead of `self.grad += ...`, the second path silently overwrites the first and the answer is wrong by half (or worse).

3. **The graph is implicit.** You don't build a graph object up-front. You just construct expressions, and each operation records its parents. The graph is the linked structure of `Value` objects.

## Concepts to internalize

- **Computational graph** — every expression is a DAG of operations on Values.
- **Forward pass** — compute each node's value by traversing inputs → outputs.
- **Local derivatives** — every operation type defines how its output's gradient flows back to its inputs (`+`: pass through; `*`: swap and multiply; `tanh`: `1 - tanh²`; etc.).
- **Reverse-mode autodiff** — apply the chain rule by walking the graph backwards once.
- **Topological sort** — the right order to apply backward updates so each node sees all its downstream gradients.
- **Gradient accumulation** — a node used in multiple expressions accumulates the sum of contributions.
- **Loss minimization via gradient descent** — once you have `dL/dparam` for each parameter, update `param ← param - lr · dL/dparam`.

---
## What you'll build

Package: `g2c/autodiff/`

### `value.py` — the autodiff engine

```python
class Value:
    def __init__(self, data: float, _children=(), _op=""): ...

    # primitive ops (each stores a backward closure)
    def __add__(self, other): ...
    def __mul__(self, other): ...
    def __pow__(self, exponent: float): ...
    def exp(self): ...
    def log(self): ...
    def tanh(self): ...
    def relu(self): ...

    def backward(self): ...   # topological sort + reverse-mode pass
```

Right-hand-side operators (`__radd__`, `__rmul__`, etc.) and unary/convenience ops (`__neg__`, `__sub__`, `__truediv__`) are implemented for you so that `2 * Value(3)` just works.

### `grad_check.py` — numerical gradient checker

```python
def numerical_grad(f, val: Value, h: float = 1e-5) -> float: ...
```

Estimates `df/dval` by finite differences. Use it to verify your analytic gradients.

### End-to-end usage

```python
from g2c.autodiff import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a.tanh()
c.backward()
print(a.grad, b.grad)  # dc/da, dc/db
```

Keep the implementation small — well under 100 lines. Legibility wins.

## How to run the tests

Tests are in `tests/test_autodiff.py`. Initial state: 2 passed, 42 failed.

```bash
source .venv/bin/activate
python -m pytest tests/test_autodiff.py            # run all autodiff tests
python -m pytest tests/test_autodiff.py -x         # stop at first failure (recommended while working)
python -m pytest tests/test_autodiff.py -k add     # run only tests whose name matches "add"
python -m pytest tests/test_autodiff.py -v         # verbose: list every test
```

## Exercises

Use the clean notebook scaffold at [`notebooks/clean/01-autodiff-xor.ipynb`](../../notebooks/clean/01-autodiff-xor.ipynb) through the launcher, which creates or resumes your working copy in `notebooks/solutions/`:

```bash
.venv/bin/python scripts/open_notebook.py 01
```

To start over, archive your current working notebook and reset from the clean scaffold:

```bash
.venv/bin/python scripts/open_notebook.py 01 --fresh
```

1. **Forward and backward by hand.** Take the expression `f = (a * b + b**2) * tanh(c)` with `a=1, b=2, c=0.5`. Compute the forward pass and all three gradients by hand on paper. Then verify against your engine.

2. **Gradient checking.** Implement a function that takes an expression and a tunable input, and compares the analytic gradient (from `.backward()`) against a finite-difference estimate `(f(x + h) - f(x - h)) / (2h)` for small `h`. Verify your engine on at least three nontrivial expressions.

3. **A neuron from scratch.** Build a single neuron `y = tanh(w1*x1 + w2*x2 + b)` using only `Value`. Compute the gradient of the loss `(y_target - y)**2` with respect to `w1`, `w2`, `b`. Manually update the weights for one step.

4. **XOR with a tiny MLP.** Build a 2-2-1 MLP (2 inputs, 2 hidden units with tanh, 1 output) using only your `Value` class — no PyTorch, no NumPy. Train it on the XOR truth table. Verify that loss decreases over a few hundred steps. (XOR is the canonical "needs a hidden layer" test case.)

5. **Topology stress test.** Build an expression that uses the same `Value` multiple times (e.g., `f = a * a + a`). Verify that gradient accumulation works: `df/da` should be `2a + 1`, not just `1` or `2a`.

## Pitfalls to expect

- **Forgetting accumulation.** If your `_backward` writes `self.grad = ...` instead of `self.grad += ...`, exercise 5 will quietly produce wrong answers. Always accumulate.
- **Topo order off-by-one.** A correct topological sort is essential. The standard pattern is post-order DFS with a visited set, then reverse the result before iterating.
- **Mutating inputs vs. returning new Values.** Each operation should return a new `Value`. Don't try to be clever about in-place updates — they make the graph confusing and break re-execution.
- **Float precision in gradient checks.** Finite differences with `h = 1e-7` can give noisy comparisons. Use `h = 1e-5` and tolerate ~1e-4 absolute error.

## M-Series Notes 

Pure Python on a single CPU thread. Runs in seconds. No PyTorch, no MPS, no installs beyond what's already in the venv.

---
## Reading

Primary:

- **Karpathy, *micrograd* repo** — the canonical reference implementation. <https://github.com/karpathy/micrograd>
- **Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd"** (YouTube). The single best DL pedagogy resource for this topic. ~2.5 hours; worth all of it.

Secondary:

- **Goodfellow, Bengio, Courville, *Deep Learning*, Chapter 6.5** — the textbook treatment of backprop.
- **Olah, "Calculus on Computational Graphs: Backpropagation"** (colah.github.io) — the cleanest written explanation of the chain rule on graphs.

## Deliverable checklist

- [ ] All operations from the suggested API are implemented
- [ ] `tests/test_autodiff.py` passes all tests: each operation forward and backward, gradient accumulation on shared nodes, gradient check vs. finite differences
- [ ] `notebooks/solutions/01-autodiff-xor.ipynb` trains a 2-2-1 MLP on XOR using only `Value`
- [ ] You can explain — out loud, without notes — why backward must traverse in topological order

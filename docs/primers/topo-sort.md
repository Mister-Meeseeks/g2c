# Topological sort primer

The one algorithm Module 01 needs from computer science. Your `Value.backward()` must visit every node in the expression graph in an order that guarantees each node's gradient is *complete* before that node passes gradient back to its inputs. Topological sort is how you get that order, and post-order depth-first search is how you compute it in six lines.

If you can already write "post-order DFS with a visited set, then reverse" from memory and explain why the reversal is the point, you can skip this page.

## How to use this primer

Read it once before implementing `Value.backward()` in Module 01. The worked trace in section 4 is the mental movie to have running while you write the DFS; the pitfalls in section 6 are the three bugs that account for essentially every wrong implementation.

---

## Contents

1. [The problem backprop poses](#problem)
2. [What a topological order is](#definition)
3. [The algorithm: post-order DFS](#algorithm)
4. [A worked trace](#trace)
5. [Why the visited set is load-bearing](#visited)
6. [Pitfalls](#pitfalls)

---

## <a id="problem"></a>1. The problem backprop poses

An expression like `loss = (a * b + a).tanh()` builds a graph: each `Value` node holds its inputs in `_prev`, so the graph's edges point from every result back to the operands that produced it. Two structural facts matter:

- **It is directed.** Data flowed forward from inputs to loss; gradients must flow the reverse direction.
- **It is acyclic.** Every operation creates a *new* node — nothing ever feeds back into an existing one. A graph like this is called a **DAG** (directed acyclic graph), and it's why you never need cycle detection here.

Backprop's correctness constraint: a node may only push gradient to its inputs **after every consumer of that node has already pushed gradient into it**. Look at the diamond that `a` forms above:

```
        a ──────────────┐
        │               │
        ▼               ▼
      a * b ────────► (+) ────► tanh ────► loss
        ▲
        │
        b
```

`a` is consumed twice: once by the multiply, once directly by the add. Its total gradient is the *sum* of both contributions (this is Module 01's Exercise 5). If `a` were processed after only one contribution had arrived, everything upstream of `a` would receive a partial — silently wrong — gradient.

So we need an ordering of all nodes such that consumers always come before the things they consume. That's a topological order, read backwards.

---

## <a id="definition"></a>2. What a topological order is

A **topological order** of a DAG is any listing of its nodes in which every node appears *after* all of its inputs. For the diamond above, both of these are valid:

```
a, b, (a*b), (+), tanh, loss
b, a, (a*b), (+), tanh, loss
```

Two things to notice:

- **It's not unique.** `a` and `b` don't depend on each other, so either may come first. Any valid order works for backprop — don't chase a canonical one.
- **Reversed, it's exactly the backprop order.** Inputs-before-outputs, reversed, is outputs-before-inputs: `loss` first, then `tanh`, then the add, and `a` only after *both* of its consumers. Backward is `for node in reversed(topo_order): node._backward()`.

---

## <a id="algorithm"></a>3. The algorithm: post-order DFS

The standard construction is a depth-first search that appends each node to the output list **after** recursing into all of its inputs — "post-order":

```python
def topological_sort(root):
    order = []
    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for parent in node._prev:   # recurse into inputs FIRST
            dfs(parent)
        order.append(node)          # ...then append self

    dfs(root)
    return order                    # inputs first, root last
```

The correctness argument is one sentence: a node is appended only after the recursion into every one of its inputs has returned, and each of those recursions appended that input (or found it already visited, meaning it was appended earlier) — so every node lands in the list after all of its inputs. Post-order *is* topological order for a DAG reached from a single root.

`backward()` then walks it in reverse:

```python
self.grad = 1.0                       # d(loss)/d(loss)
for node in reversed(topological_sort(self)):
    node._backward()
```

---

## <a id="trace"></a>4. A worked trace

Take `loss = tanh(a*b + a)` with the diamond from section 1. Label the multiply node `m`, the add node `s`, the tanh node `t`. Trace `dfs(t)`:

```
call        action                                   order so far
─────────────────────────────────────────────────────────────────
dfs(t)      visit t, recurse into s
  dfs(s)    visit s, recurse into m (first input)
    dfs(m)  visit m, recurse into a
      dfs(a) visit a — no inputs → append a          [a]
      dfs(b) visit b — no inputs → append b          [a, b]
    ...     m's inputs done → append m               [a, b, m]
  dfs(a)    a already visited → return immediately
  ...       s's inputs done → append s               [a, b, m, s]
...         t's inputs done → append t               [a, b, m, s, t]
```

Result: `[a, b, m, s, t]` — a valid topological order. Reversed: `t, s, m, b, a`. Follow the gradient in that order: `t` fills `s.grad`; `s` pushes into *both* `m.grad` and `a.grad` (first contribution); `m` pushes into `a.grad` (second contribution, **accumulated** with `+=`) and `b.grad`. By the time position `a` comes up in the iteration, its gradient is complete — which is the entire point of the exercise.

Note the moment the visited set earned its keep: the second `dfs(a)` — reached directly from `s` — returned immediately instead of appending `a` a second time.

---

## <a id="visited"></a>5. Why the visited set is load-bearing

Drop the visited set and two things go wrong, one obvious and one subtle:

- **Duplicate appends.** In the diamond, `a` gets appended once per path that reaches it. During the reversed walk, `a._backward()` runs twice, pushing its (by then complete) gradient into its inputs twice — double-counted gradients upstream.
- **Exponential blowup.** Chain several diamonds in a row and the number of paths from root to the earliest nodes doubles at each diamond. Without a visited set, DFS runtime is proportional to the number of *paths*, not nodes. A 40-layer diamond chain has a trillion paths. With the set, every node is visited exactly once: linear time.

This is also why the naïve alternative — skip the sort entirely and recurse `_backward()` directly through the graph — is wrong on any graph with shared nodes, i.e. every interesting one. Recursive backprop re-derives the path explosion; the topological sort is what collapses it to one visit per node.

One distinction worth keeping crisp, because the failure modes look similar: **gradient accumulation across multiple consumers is correct and required** (`a.grad += ...` from both `m` and `s`); **visiting a node twice is a bug** (each consumer's push happening more than once). The visited set prevents the second without disturbing the first.

---

## <a id="pitfalls"></a>6. Pitfalls

**Marking visited too late.** Add the node to `visited` *on entry*, before recursing — not after the loop. Marking after recursion still terminates on a DAG, but a node reachable by two paths can be re-entered while its own recursion is still in flight, appending it twice.

**Forgetting to reverse.** `topological_sort` returns inputs-first. Backprop needs outputs-first. If your gradients are all zero except the loss node, you almost certainly iterated the list forwards — each node ran `_backward()` before any gradient had arrived in it.

**Recursion depth.** Python's default recursion limit is ~1000 frames. A graph that's a long chain — which is exactly what an unrolled training expression looks like — can exceed it. Fine for Module 01's toy graphs; if you ever hit `RecursionError`, convert the DFS to an explicit stack. The iterative version pushes `(node, inputs_done)` pairs: push the node with `inputs_done=False`, and when you pop a `False` entry, re-push it as `True` followed by its unvisited inputs; append to `order` only when popping a `True` entry. Same post-order, no Python frames.

**Trusting insertion order tricks.** Sets in Python are unordered; if you use the visited *set itself* as the output list (a tempting one-liner with `dict.fromkeys`), you get insertion order — which happens to be post-order only if you insert at the append point, not at the visit point. Keep the list and the set as two separate structures and the question never arises.

---

## What this primer doesn't cover

- **Kahn's algorithm** (repeatedly remove zero-in-degree nodes) — the other standard topological sort. Equivalent output, but the DFS form matches the recursive structure of expression graphs and is what you'll see in micrograd-style engines, including this course's.
- **Cycle detection.** General topological sort must detect cycles (no valid order exists). Expression DAGs can't have cycles — every op creates a fresh node — so `Value.backward()` legitimately skips it.
- **How PyTorch does it.** Same idea, industrial version: the autograd engine records the graph during forward and executes a reverse traversal with reference counting instead of an explicit sorted list. After Module 01 you'll recognize it as the same algorithm wearing a hard hat.

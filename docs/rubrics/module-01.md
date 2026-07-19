# Module 01 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/01-autodiff-xor.ipynb`, falling back to `notebooks/clean/01-autodiff-xor.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 01.01 — Two gradient paths through `b`

For `f = (a*b + b**2) * tanh(c)` with `a=1, b=2, c=0.5`.

A correct answer should include:

- The two path contributions are *added* (gradient accumulation at a shared node), not multiplied or overwritten.
- The `a*b` path contributes `a · tanh(c) = tanh(0.5) ≈ 0.462`; the `b**2` path contributes `2b · tanh(c) = 4·tanh(0.5) ≈ 1.848`.
- Total `b.grad = (a + 2b)·tanh(c) = 5·tanh(0.5) ≈ 2.31`. Exact decimals optional; the structure `(a + 2b)·tanh(c)` is the point.

Common issues:

- Forgetting the outer `tanh(c)` factor and reporting `a + 2b = 5`.
- Saying the paths multiply, or that only one path "wins."
- Computing `d(b**2)/db` as `b` instead of `2b`.

## Exercise 01.02 — Analytic vs. numeric residual

A correct answer should include:

- The residual comes from finite-difference truncation error (the central difference is only exact in the `h → 0` limit) plus floating-point rounding.
- Shrinking `h` much further makes things *worse*, not better: `f(x+h) - f(x-h)` becomes a subtraction of nearly equal floats, so cancellation/rounding error dominates after dividing by the tiny `2h`.
- (Implicitly or explicitly) there is a sweet spot — the lesson's `h = 1e-5` with ~1e-4 tolerance.

Common issues:

- Claiming smaller `h` always improves accuracy.
- Attributing the residual to a bug in the autodiff engine.
- Mentioning truncation error but missing the float-cancellation half of the story.

## Exercise 01.03 — Why grads must be zeroed

A correct answer should include:

- `_backward` *accumulates* (`+=`) into `.grad` by design — that's how shared nodes get correct gradients.
- Without resetting, the second backward pass adds fresh gradients on top of the stale step-1 gradients, so the second update moves by the *sum* of old and new gradients — wrong magnitude and possibly wrong direction, compounding every step.

Common issues:

- Saying backward "overwrites" grad (it accumulates; that's the whole reason zeroing is needed).
- Claiming the forward pass or the loss value would be wrong (only the update is corrupted).
- Confusing resetting `.grad` with resetting `.data`.

## Exercise 01.04 — Reading the XOR loss curve

Run-dependent — grade the reasoning shape, not specific numbers.

A correct answer should include:

- A concrete description of the student's own curve (starting loss, plateau if present, final drop, final loss near zero).
- For the plateau: the network is in a near-symmetric/saddle-like configuration where predictions hover near the mean; the hidden tanh neurons are slowly rotating/shifting their linear boundaries until they differentiate, after which loss drops quickly.

Common issues:

- Treating a plateau as "training failed" despite the later drop.
- Reporting only the final loss with no reading of the curve's shape.
- Attributing the sudden drop to a learning-rate schedule (there is none — fixed `lr=0.1`).

## Exercise 01.05 — How the hidden neurons carve the plane

A correct answer should include:

- Each hidden tanh neuron defines a (soft) linear boundary — a half-plane in the input plane.
- The output neuron combines the two hidden activations so the composite boundary becomes a band/stripe (or two crossing lines) that puts `(0,1)` and `(1,0)` on one side and `(0,0)` and `(1,1)` on the other.
- Reference to the boundary snapshots: early frames show a nearly straight line; later frames show the bent/curved region forming.

Common issues:

- Claiming a single line separates XOR (it's the canonical non-linearly-separable example).
- Describing hidden neurons as "detecting individual points" rather than defining half-plane boundaries.
- Ignoring the snapshots entirely when the question asks to use them.

## Exercise 01.06 — Overwriting instead of accumulating

A correct answer should include:

- The correct gradient decomposes by path: `a*a` contributes `3 + 3` (`a` appears as both factors) and the `+ a` term contributes `1`, summing to 7.
- An overwriting backward reports 3 (with the standard reverse-topological order): the `+` node writes `a.grad = 1`, then the `a*a` node writes 3 for one factor and overwrites with 3 for the other — only the last write survives.
- The key mechanism: each path's write clobbers the previous one instead of summing `3 + 3 + 1`.

Common issues:

- Answering 6 (treating `a*a` as a single surviving `2a` contribution — the two factor writes also overwrite each other).
- Giving a number with no explanation of which contribution survived and why.
- Saying the result is "random" without noting it's determined by backward visit order (last write wins).

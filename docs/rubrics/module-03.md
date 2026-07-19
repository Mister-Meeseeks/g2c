# Module 03 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/03-nn.ipynb`, falling back to `notebooks/clean/03-nn.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 03.01 — Next failing test

Run-dependent — grade the reasoning shape.

A correct answer should include:

- A concrete test name from `pytest tests/test_nn.py -x` output at the time they ran it.
- The implementation it points at (e.g., a `Linear.forward` test points at `g2c/nn/modules.py`). If everything already passes, saying so is a complete answer.

Common issues:

- Naming a file instead of a test, or vice versa, with no mapping between them.

## Exercise 03.02 — Shapes for `Linear(1, 1)`

A correct answer should include:

- `W: (1, 1)`, `b: (1,)` (this repo's convention: `W` is `(in, out)`).
- `predictions: (100, 1)` — matching `x_reg`/`y_reg`.
- `loss: ()` — a scalar; MSE reduces everything to one number.

Common issues:

- Predictions as `(100,)` — the regression keeps the trailing feature dim.
- Loss reported as `(100, 1)` (forgetting the mean reduction; `.backward()` needs a scalar).

## Exercise 03.03 — Why `zero_grad` before `backward`

A correct answer should include:

- PyTorch (like the Module 01 engine) *accumulates* into `.grad` rather than overwriting.
- Zeroing clears the previous iteration's gradients so this step's update reflects only the current batch; skipping it makes gradients pile up across steps.

Common issues:

- Saying backward overwrites grads (then zeroing would be pointless).
- Claiming `zero_grad` resets the parameters rather than the gradients.

## Exercise 03.04 — Why `step` after `backward`

A correct answer should include:

- `optimizer.step()` consumes `.grad` to apply `param ← param − lr · grad`.
- Before `backward`, `.grad` is zero (or stale), so stepping first would apply a no-op or a stale update; `backward` is what populates the gradients `step` needs.

Common issues:

- Restating the ordering without the data dependency (step reads what backward writes).
- Confusing `step` with running the forward pass.

## Exercise 03.05 — Did the fit recover `y = 3x + 2`?

Run-dependent — grade the reasoning shape.

A correct answer should include:

- Cited evidence: learned `W ≈ 3` and `b ≈ 2` (the cell asserts within 0.15), plus a loss curve that falls and flattens near the noise floor.
- A yes/no conclusion tied to that evidence, not just "the cell ran."

Common issues:

- Citing only the final loss without the recovered parameter values (the question is about the line).
- Treating the small residual loss as failure — the data has 0.05-scale noise that no line can remove.

## Exercise 03.06 — First move on divergence/NaN

A correct answer should include:

- Lower the learning rate (cut ~10×) as the first move — LR too high is the standard cause of loss exploding to NaN.
- (Acceptable additions) then check `zero_grad` and shapes.

Common issues:

- Reaching for architecture changes or more steps before touching LR.
- "Restart the notebook" with no hypothesis.

## Exercise 03.07 — Why `Linear(2, 2)` can't solve circles

A correct answer should include:

- A single linear layer can only produce a linear decision boundary — a straight line in 2D.
- The circles data needs a closed/curved boundary around the inner cluster; a ring around a blob is not linearly separable, so any line misclassifies a large fraction.

Common issues:

- Saying it lacks "enough parameters" — capacity isn't the issue; boundary shape is.
- Claiming more linear layers would fix it (they compose to one linear map — Exercise 5's point).

## Exercise 03.08 — Logits shape for 2-class

A correct answer should include:

- `(N, 2)` — one row per point, one column per class (the notebook asserts `(x_2d.shape[0], 2)`).

Common issues:

- `(N,)` or `(N, 1)` from thinking binary classification needs a single output (valid with a sigmoid design, but not this module's CrossEntropyLoss contract).

## Exercise 03.09 — Why logits, not probabilities

A correct answer should include:

- `CrossEntropyLoss` applies log-softmax internally using the numerically stable log-sum-exp trick, so the model should hand it raw scores.
- Feeding probabilities would double-apply softmax and re-introduce the overflow/underflow problems the fused loss avoids.

Common issues:

- "It's just convention" with no numerical-stability reason.
- Claiming probabilities are non-differentiable (they're differentiable; the issue is stability and double-softmax).

## Exercise 03.10 — What ReLU added

A correct answer should include:

- The nonlinearity breaks the linear-composition collapse: `ReLU(x W₁ + b₁) W₂ + b₂` is genuinely nonlinear.
- Concretely: it lets the model bend the boundary — combining hidden half-planes into a piecewise-linear, closed region around the inner cluster, which no single linear layer can draw.

Common issues:

- "It made the model deeper" — depth without a nonlinearity adds nothing.
- Describing ReLU as adding parameters (it's pointwise and parameter-free).

## Exercise 03.11 — Evidence of learning

Run-dependent — grade the reasoning shape.

A correct answer should include:

- Quantitative evidence: training accuracy above the asserted 0.90 and a loss curve falling from near `log(2) ≈ 0.69`.
- Qualitative evidence: the decision-boundary plot actually encircling the inner class.

Common issues:

- "The cell ran without errors" as evidence.
- Citing accuracy alone without the boundary plot (or vice versa) when both are on screen.

## Exercise 03.12 — Flattened MNIST batch shape

A correct answer should include:

- `(128, 784)` — batch size 128, 28×28 = 784 flattened pixels; generically `(B, 784)`.

Common issues:

- Keeping image dims `(128, 28, 28)` or `(128, 1, 28, 28)` after "flattening."
- 784 miscomputed.

## Exercise 03.13 — Shapes for `Linear(784, 128)`

A correct answer should include:

- `W: (784, 128)`, `b: (128,)` — the `(in, out)` convention again.

Common issues:

- `(128, 784)` from the `torch.nn.Linear` convention.
- `b` as `(784,)` (bias lives in the output space).

## Exercise 03.14 — Why class-index targets

A correct answer should include:

- Targets are integer class labels (0–9); the loss picks out the correct-class logit directly by indexing/gather.
- One-hot targets are mathematically equivalent but wasteful — a `(B, 10)` float tensor and a dot product to select what an index selects directly.

Common issues:

- Claiming one-hot would give a different (wrong) loss value rather than the same value inefficiently.
- Answering about label *smoothing* or soft targets — out of scope here.

## Exercise 03.15 — The overfitting pattern

A correct answer should include:

- Train loss keeps falling while validation loss (or accuracy) stops improving and turns worse — the curves diverge and the gap grows.
- The reading: the model is memorizing training data rather than generalizing.

Common issues:

- Calling any train/val gap overfitting — a small stable gap is normal; the *diverging* trend is the signal.
- Only describing training loss.

## Exercise 03.16 — Where `weight_decay` enters SGD

A correct answer should include:

- It's added to the gradient before the LR multiply: `delta = grad + weight_decay * param`, then `param ← param − lr · delta`.
- Equivalently `param ← param − lr·(grad + λ·param)` — an L2 pull of every weight toward zero each step.

Common issues:

- Placing it as a multiplier on the LR or on the loss value at update time.
- Saying it modifies gradients of the *loss surface* only conceptually without the concrete `grad + λ·param` term this implementation uses.

## Exercise 03.17 — Predicted effect of weight decay

A prediction — grade the reasoning, then whether they reconciled with the experiment.

A correct answer should include:

- Validation curve should degrade less / re-converge toward the train curve (less overfitting), possibly at the cost of slightly higher training loss.
- The mechanism: shrinking weights constrains the model, trading training fit for generalization.

Common issues:

- Predicting weight decay improves *training* loss.
- Expecting a dramatic accuracy jump — the effect at this scale is a curve-shape change, not a leap.

## Exercise 03.18 — Why `Linear → Linear` collapses

A correct answer should include:

- The algebra: `(x W₁ + b₁) W₂ + b₂ = x (W₁W₂) + (b₁W₂ + b₂) = x W' + b'` — the composition of affine maps is affine.
- Therefore stacked linear layers have exactly one linear layer's expressive power, regardless of depth or width.

Common issues:

- Asserting the collapse without the algebra.
- Claiming the two-layer version still helps optimization enough to change what's representable.

## Exercise 03.19 — Expected boundary with ReLU

A correct answer should include:

- A closed, curved-looking (piecewise-linear) boundary that wraps around the inner cluster — a polygon-like loop.

Common issues:

- Predicting a perfectly smooth circle — ReLU boundaries are piecewise-linear, though at hidden=16 they look rounded.

## Exercise 03.20 — Expected boundary without ReLU

A correct answer should include:

- A single straight line (the model is one affine map in disguise), which cannot separate concentric circles — accuracy far below the ReLU model, near the class base rate.

Common issues:

- Predicting a merely "worse" curved boundary — without the nonlinearity the boundary is exactly linear, not just weaker.

# Module 00 Rubric

Use this rubric to grade submitted answers in `answers/module-00.md`. Blank `Student answer` sections are not wrong; skip them unless the student asks for a completeness check. Give feedback by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 00.01 — Shape trace

A correct answer should include:

- Embedding lookup changes token IDs from `(B, T)` to token representations `(B, T, C)`.
- With the given values, the embedded shape is `(4, 8, 16)`.
- The final projection maps each `C`-dimensional token representation to `V` logits.
- The logits shape is `(B, T, V)`, or `(4, 8, 1000)`.

Common issues:

- Dropping the batch dimension.
- Returning `(B, C)` after embedding lookup.
- Treating the vocabulary dimension as the hidden/channel dimension.

## Exercise 00.02 — Matmul by hand

A correct answer should include:

- `A` has shape `(2, 3)`.
- `B` has shape `(3, 2)`.
- `A @ B` has shape `(2, 2)`.
- The result is:

  ```text
  [[220, 280],
   [490, 640]]
  ```

Common issues:

- Multiplying elementwise instead of using row-column dot products.
- Reversing the output shape.
- Forgetting that the inner dimension `3` disappears through summation.

## Exercise 00.03 — Backprop by hand

A correct answer should express the gradients as products of local derivatives. Equivalent notation is fine.

Expected local derivatives:

- `dL/da = 2(a - target)`
- `da/dz = 1 - tanh(z)^2`, equivalently `1 - a^2`
- `dz/dw = x`
- `dz/db = 1`
- `dz/dx = w`

Expected chained gradients:

- `dL/dw = 2(a - target) * (1 - a^2) * x`
- `dL/db = 2(a - target) * (1 - a^2)`
- `dL/dx = 2(a - target) * (1 - a^2) * w`

Common issues:

- Forgetting the derivative of `tanh`.
- Omitting `dL/da`.
- Treating `b` as if its derivative were `b` instead of `1`.

## Exercise 00.04 — Softmax and loss

A correct answer should include:

- Stable softmax terms for logits `[2.0, 1.0, 0.0]`, such as `[1, e^-1, e^-2]`.
- Approximate probabilities near `[0.665, 0.245, 0.090]`.
- For target class `0`, negative log likelihood `-log(0.665)`, approximately `0.408`.

Allow small rounding differences.

Common issues:

- Treating logits as probabilities directly.
- Forgetting to normalize by the sum of exponentials.
- Using base-10 log instead of natural log.

## Exercise 00.05 — Training-loop narration

A correct answer should explain, in five sentences or fewer:

- `forward` computes predictions/logits from inputs and parameters.
- `loss` compares predictions with targets.
- `backward` computes or accumulates gradients for parameters.
- `step` updates parameters using those gradients.
- `zero_grad` clears accumulated gradients before the next iteration.

Common issues:

- Saying `backward` updates weights directly.
- Forgetting gradient accumulation.
- Omitting targets/loss and describing only inference.

## Exercise 00.06 — Environment check

A correct answer should include:

- The command that was run: `python scripts/smoke_test.py`.
- Whether it completed successfully.
- If it failed, the relevant error and the next debugging step.
- On an Apple Silicon machine, whether PyTorch reports MPS availability or why it does not.

Common issues:

- Reporting only that setup was attempted, without the result.
- Ignoring a failed MPS check on an Apple Silicon machine.
- Pasting a long log without summarizing pass/fail and the key issue.

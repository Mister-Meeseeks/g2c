# Module 00 Practice Set 002 — Shapes and Softmax

Generated because: follow-up practice on tensor shape contracts, logits versus probabilities, and target-class negative log likelihood.

Related lesson: `docs/modules/00-prerequisite-review.md`
Related rubric: `docs/rubrics/module-00.md`

When ready for help or grading, ask:

```text
Can you review practice/module-00/set-002-shapes-softmax.md?
```

## Problem P00.002.01 — Embedding Lookup Shapes

### Prompt

Let `B = 3`, `T = 5`, `C = 12`, and `V = 200`.

Token IDs have shape `(B, T)`. An embedding table has shape `(V, C)`.

1. What is the shape after embedding lookup?
2. In one sentence, explain what the embedding lookup does to each token ID.

### Help request / hint request


### Student answer


### Notes / uncertainty


### Agent feedback


## Problem P00.002.02 — Projection to Vocabulary Logits

### Prompt

You have token representations with shape `(B, T, C) = (2, 4, 8)`.

A final projection matrix has shape `(C, V) = (8, 50)`.

1. What is the logits shape after applying the projection at every token position?
2. Are these logits probabilities? Explain briefly.

### Help request / hint request


### Student answer


### Notes / uncertainty


### Agent feedback


## Problem P00.002.03 — Matmul Shape Contract

### Prompt

For each pair, state whether the matrix multiplication is valid. If it is valid, give the output shape. If it is not valid, say which dimensions fail to match.

1. `(6, 4) @ (4, 10)`
2. `(6, 4) @ (10, 4)`
3. `(2, 3, 5) @ (5, 7)`

### Help request / hint request


### Student answer


### Notes / uncertainty


### Agent feedback


## Problem P00.002.04 — Stable Softmax Terms

### Prompt

Given logits `[4.0, 2.0, 1.0]`:

1. Subtract the maximum logit and write the stabilized logits.
2. Write the unnormalized exponential terms after stabilization.
3. Explain why subtracting the maximum does not change the final probabilities.

You do not need to compute decimal probabilities for this problem.

### Help request / hint request


### Student answer


### Notes / uncertainty


### Agent feedback


## Problem P00.002.05 — Softmax Probabilities and Target Loss

### Prompt

Given logits `[1.0, 0.0, -1.0]` and target class `1`:

1. Compute the approximate softmax probabilities.
2. Compute the negative log likelihood for the target class only.
3. In one sentence, explain why you should not report the negative log likelihood for all classes here.

Round decimals to three places.

### Help request / hint request


### Student answer


### Notes / uncertainty


### Agent feedback


## Problem P00.002.06 — Sequence Logits to Loss Shape

### Prompt

A tiny language model receives input token IDs with shape `(B, T) = (4, 6)`.

After embedding lookup, the representation size is `C = 16`. The vocabulary size is `V = 100`.

1. What is the embedded shape?
2. What is the logits shape?
3. If each token position has one target token ID, what is the target shape?
4. Cross-entropy will produce one loss per token position before averaging. What is that per-token loss shape?

### Help request / hint request


### Student answer


### Notes / uncertainty


### Agent feedback


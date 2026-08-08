# Rubric — Beyond: Multimodal language models

Student answers live in `notebooks/solutions/multimodal-mnist.ipynb`
(fall back to `notebooks/clean/multimodal-mnist.ipynb`). Grade each
submitted `Question:` / `Answer:` independently; skip blank answers.

## Q1 — The token cost of resolution

- **Correct**: larger patches are cheaper because they produce fewer visual
  tokens but compress more pixels before attention can operate; smaller
  patches preserve finer spatial detail while increasing sequence length,
  KV-cache use, and full-attention cost. Connects this trade to production
  image token budgets without claiming raw patch count is every VLM's billing
  formula.
- **Partially correct**: gives only one side of the trade, or says “more
  tokens cost more” without identifying where detail is discarded.

## Q2 — Generated captions versus the Module 03 MLP

- **Correct**: reports both measured accuracies and discusses actual sample
  captions. Explains that the caption model learns image features, the
  placeholder/splice convention, caption syntax, and autoregressive routing,
  while the MLP directly optimizes a ten-class decision. Treats the result as
  a comparison of these models and objectives at this scale—not a general law
  about transformers or image classifiers.
- **Mostly correct**: reports both numbers and a plausible explanation but
  does not distinguish generated evaluation from teacher forcing.
- **Needs revision**: substitutes a remembered MLP benchmark for the
  student's measured Module 03 result, or scores the true digit slot with the
  caption prefix supplied and calls it generation.

## Q3 — The shuffled-patch result

- **Correct**: reports both generated-caption accuracies and the observed
  change without assuming its direction or size. Explains that one fixed
  permutation preserves patch content and stable slot identity while changing
  the row-major adjacency prior; distinguishes this from a fresh permutation
  per example, which would remove stable positional meaning.
- **Partially correct**: reports the result without explaining what the fixed
  permutation preserved, or claims attention is inherently order-invariant
  despite learned position embeddings.

## Q4 — From raw patches to a production VLM

- **Correct**: preserves the central interface idea—visual features are mapped
  to language-model width and enter a shared context where text loss can train
  across modalities. Names at least three production additions with their
  jobs, such as resize/tiling, a vision tower for semantic features, a
  projector for width/alignment, a resampler for token-budget control, large
  paired datasets, or staged/joint optimization. States that “native
  multimodal” is not a standardized claim that no vision encoder exists.
- **Partially correct**: lists components without explaining their jobs, or
  treats a pretrained tower as merely a faster version of `Linear(49, D)`.
- **Needs revision**: claims this raw-pixel projector reproduces the complete
  production architecture.

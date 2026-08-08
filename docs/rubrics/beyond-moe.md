# Rubric — Beyond: Mixture of experts

Student answers live in `notebooks/solutions/moe-mixture-of-experts.ipynb`
(fall back to `notebooks/clean/moe-mixture-of-experts.ipynb` if no working
copy exists). Grade each submitted `Question:` / `Answer:` independently;
skip blank answers.

## Q1 — Why renormalize the top-k weights

- **Correct**: identifies that without renormalization the output's
  *magnitude* depends on how much probability mass fell outside the
  top-k — i.e., on router confidence — so the layer's output scale
  varies token-to-token; notes shape is unchanged, which is why no
  shape test catches it.
- **Partially correct**: says "outputs get smaller" without connecting
  it to router confidence, or claims it breaks correctness outright.
- **Needs revision**: says renormalization is needed for gradients to
  flow (it isn't — gradients flow either way).

## Q2 — Which number the API price tracks

- **Correct**: per-token compute usually tracks *active* parameters far
  more closely than total parameters, while storage follows total.
  Notes that API price and latency also depend on routing overhead,
  memory movement, batching, hardware, and provider policy. A modest
  val-loss gap is acceptable if attributed to scale (small model and
  short run — capacity may not be binding yet) or noisy routing.
- **Partially correct**: right number, no explanation of the modest gap.

## Q3 — The balance-loss sweep

- **Correct**: reports the observed zero/default/100× utilization
  patterns without claiming stronger specialization or collapse than
  the plots show. Explains why the sweep uses `k=2`: its relative
  combination weights let the language-modeling loss train the router,
  whereas renormalized `k=1` always has a selected weight of 1 and gets
  no useful task gradient. Explains both forces in the `k=2` sweep:
  without the auxiliary loss a slightly favored expert can receive more
  weight and task gradient, which can compound; at 100× the balancing
  objective can dominate the language loss and force nearly uniform
  routing, leaving less room for useful specialization.
- **Mostly correct**: explains both mechanisms but reports only two of
  the three runs.
- **Partially correct**: describes a collapsed or uniform end state
  without the mechanism that produced it, or misses the `k=1` versus
  `k=2` distinction.

## Q4 (optional, specialization probe)

- **Correct**: names an interpretable decoded-token bucket if one
  appeared and explains why per-token routing can specialize by token
  role or local context. Any honest "my buckets look noisy" with a
  plausible scale/data explanation is equally acceptable — grade the
  reasoning, not the luck.

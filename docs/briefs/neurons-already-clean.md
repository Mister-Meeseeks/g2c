# G2C Brief — "Neural Networks Are Already Clean": an argument about what neurons are

> **Question this Brief answers:** *When a neuron seems to mean two unrelated things, is the network packing them together, or are we reading it wrong?*

Most Briefs decode a model release. This one decodes an *argument*. On October
15, 2026, Nick Cammarata of Goodfire published a long essay,
[*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/). It
says that one of the founding assumptions of modern mechanistic
interpretability was a misreading. The assumption is that neurons arbitrarily
pack unrelated concepts together, so you must decompose a network into
something else before you can understand it. The essay claims that once you
fix the microscope, most neurons in modern models, up to trillions of
parameters, turn out to have one recognizable job.

That is a big claim, made by someone who helped build the view it now
disputes. It is also a claim a g2c graduate is unusually well equipped to read.
Every object the essay talks about is something you have built: hidden units
([Module 03](../modules/03-nn.md)), the feed-forward sublayer and residual
stream ([Module 09](../modules/09-transformer-block.md)), attention
([Module 07](../modules/07-attention.md)), and gradients you can push all the
way back to the input ([Module 01](../modules/01-autodiff.md)).

This Brief explains the debate the essay enters, lays out its argument and
evidence, and is explicit about what is established, what is preliminary, and
where informed readers will push back. It does not try to settle the question.

---

## Snapshot

| | |
|---|---|
| **Source** | [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/), Nick Cammarata, Goodfire. The language-model section was done with Curt Tigges. A [plain-text version](https://already-clean.goodfire.com/essay.md) is linked from the page. |
| **Published** | October 15, 2026 |
| **Kind of document** | An informal research essay that calls itself "a working hypothesis." It is not peer-reviewed and has no methods section. |
| **Companion paper** | A "forthcoming technical paper" on language-model neurons is referenced throughout. Not yet published. |
| **Models studied** | InceptionV1, three ConvNeXts the author trained (92K / 7.06M / 28.6M parameters), DINOv2, GPT-2 XL, Qwen3-32B, Qwen3.6 models, and Kimi K3 |
| **Last verified** | October 17, 2026, against the page as last modified that day |

**How to read the labels here.** A research essay is not a model release, so
the [standard Brief labels](index.md) shift slightly in meaning:

- **Reported:** an observation, measurement, or number the essay presents.
  Most of these come from experiments that are not yet published in
  reproducible form. Some can be inspected in the essay's interactive
  explorers.
- **Derived:** arithmetic this Brief computes from disclosed values, with the
  working shown.
- **G2C interpretation:** this Brief's reading, connection, or critique.
- **Not disclosed:** something the essay doesn't provide that a stronger
  conclusion would need. For an essay, that usually means protocols,
  thresholds, or a full dataset.

---

## The vocabulary, mapped to things you built

Interpretability has its own dialect. Here is the minimum you need, translated
into course terms.

| Term | What it means | Where you've seen it |
|---|---|---|
| **Neuron** | One hidden unit of an MLP: one coordinate of the vector *after* the nonlinearity. A transformer's neurons are the `4×d` hidden units of each FFN. | Module 03 hidden layer; Module 09 FFN |
| **Pole** | A neuron's positive or negative side. The essay treats both as meaningful. | Any activation that can take both signs |
| **Monosemantic / polysemantic** | A neuron that responds to one recognizable thing, versus several apparently unrelated things. | — |
| **Residual stream** | The running `d`-dimensional vector each block reads from and adds to. | Module 09's `x = x + sublayer(norm(x))` |
| **Ablation** | Set a neuron (or head) to zero, or to its mean, and see what changes downstream. | Editing a tensor in a forward pass |
| **Steering** | Add to a neuron's activation and see what the model writes. | Same, with a nonzero edit |
| **Feature visualization** | Optimize an *input image* by gradient ascent to maximize one neuron. | Module 01 gradients, taken with respect to the input instead of the weights |
| **Dictionary / SAE** | A sparse autoencoder: a *second* network trained to reconstruct a layer's activations as sparse combinations of many more directions than the layer has neurons. Its units are called "features." | Module 03's MLP, trained on activations instead of data |
| **Attribution graph** | A graph of which units caused which, for one specific input and output. | Backprop's chain rule, recorded per example |

The word **superposition** is the one to be careful with, because it is used
in two ways:

1. **"More features than dimensions."** A layer represents more meaningful
   patterns than it has neurons, by using directions and combinations. This
   is the definition from Anthropic's
   [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html).
2. **Arbitrary packing.** A network that runs out of room stops grouping
   *similar* things and starts folding *unrelated* things into the same
   neurons, because they rarely co-occur and so rarely interfere.

**Reported.** The essay argues against sense 2 only. It explicitly accepts
sense 1, which it regards as composition and as good news: a car neuron built
from wheel and window neurons means the layer below "represents" a car
without having a car neuron. Much of the apparent disagreement between this
essay and the work it criticizes dissolves once you keep these two senses
apart. What remains is the argument about sense 2.

---

## The story the field told

To see why the essay matters, you need the history it is revising. Here it is,
compressed.

**2017–2020: neurons look readable early, unreadable late.** In vision models,
especially InceptionV1, the field's main model organism, early-layer neurons
were strikingly clean: curve detectors, edge detectors, high-low frequency
detectors. Late layers were not. The most famous example was neuron **4e:55**,
which responded to cat faces *and* the fronts of cars. The influential
[*Zoom In*](https://distill.pub/2020/circuits/zoom-in/) paper, which Cammarata
co-authored, used it and a "car feature spreading into dog neurons" circuit as
its two headline examples of polysemanticity and superposition.

**2022–2023: superposition becomes the default explanation.** *Toy Models of
Superposition* showed that small networks forced to reconstruct many sparse
features through a narrow bottleneck do pack them. Language-model neurons also
looked messy. **Reported (citing those papers).** Anthropic's
[*Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
gave the median neuron of its one-layer model a 0 out of 14 on its
interpretability rubric. OpenAI's
[automated-interpretability project](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html),
which Cammarata started, found about 1.7% of GPT-2 XL's neurons interpretable
by its most generous measure.

**The consequence was a change of method.** If neurons are knots, studying
them directly is hopeless. You first train a decomposition, typically a sparse
autoencoder or transcoder, to produce a cleaner basis, and study that instead.
That has been the dominant research program since:

```text
direct study                         decomposition first
─────────────                        ───────────────────
network ──► read neurons,           network ──► train SAE / transcoder ──► read its features
            circuits, data flow                 (a second model that approximates the first)
```

The essay's thesis is that the left-hand column was abandoned on bad evidence.

---

## The argument in four steps

**Reported.** The essay states its own argument this way:

1. Networks use **one learning principle**, generalization, not two
   (generalization plus arbitrary packing). So nothing blocks ordinary science
   on the network itself.
2. With modern models and tools, that science goes much better than most
   people expect.
3. Understanding **compounds**. Each explained neuron constrains its
   neighbors, like a crossword filling in.
4. *(Flagged by the essay as "highly speculative.")* This may be most of what
   interpretability needs, and the problem may be far more tractable, possibly
   soon, than assumed.

The engine of the argument is a list of reasons a neuron might *look*
polysemantic. Four of them are mundane and testable. Only one requires a new
learning principle:

| Cause | How you'd test it |
|---|---|
| **Bad tools.** The microscope is miscalibrated. | Improve the tool and see whether the neuron resolves. |
| **Bad explanations.** The pattern is real but described wrongly, e.g. as a list of nouns. | Find a shared pattern that predicts new cases. |
| **Bad learning.** The architecture or training made the layer learn poorly. | Retrain with a better architecture and compare. |
| **Recursive generalization.** The neuron grew offshoots into neighboring territory. | Watch it grow across training checkpoints. |
| **Superposition (sense 2).** Unrelated things are packed together. | No direct test: its only symptom is "doesn't make sense." |

**Reported.** The essay's methodological point is that superposition was
treated as the *null* hypothesis when it should have been the *last* one,
because it is the most radical claim and the hardest to falsify. It quotes
Anthropic's own
[*Softmax Linear Units*](https://transformer-circuits.pub/2022/solu/index.html)
paper (2022) saying superposition "is often treated as the default hypothesis
simply because it seems intuitively more compelling than other explanations,
but there is little evidence."

**G2C interpretation.** This table is the most portable idea in the essay. It
is a checklist for any unexplained thing in a model you train, not just
neurons.

---

## The evidence, case by case

The essay's evidence comes in two very different grades. In **vision**, the
author trained the models, controlled the experiments, and published explorable
microscopes. In **language**, the work is preliminary and the detailed methods
are deferred to the companion paper. The cases below are grouped that way.

### Vision: re-reading the canonical examples

**The cat-and-car neuron is a V detector.** **Reported.** Asked "what one thing
could this be?", the author saw a **V shape**: the point of a cat's face, the
V of a car's grille. The neuron's other strong connections are to V-shaped
flames, V-shaped fireboat water jets, and fox faces. Its *negative* pole is an
**O**: round where the V is pointed. The original reading, "OR(cat, car)," was
a pattern described as a list of nouns.

> **G2C interpretation.** This is the essay's most useful habit for a working
> reader. Describe a neuron as a *distinction it draws for downstream units*
> (V versus O), not as a set of things it "detects." The negative pole is
> often the fastest check on whether a grouping is real.

**The original "superposition circuit" was pose, plus a reward-hacked tool.**
**Reported.** The circuit that appeared to show a car feature hiding among dog
neurons turned out to involve **generalized pose neurons**. InceptionV1 has
more than a hundred of them, each grouping cars, dogs, and other objects by
the way they face the camera. The dogs appeared partly because **feature
visualization was reward hacked**. Gradient ascent on the input takes the
easiest path, and in a model trained on ImageNet's many dog classes, that path
falls into a "dog basin" and never escapes. Nudging the optimizer toward the
routes real top images take made two of the four "dog" neurons render as car
neurons and two as shared car-and-dog pose detectors.

> **G2C interpretation.** You know this failure from the optimizer side. An
> objective optimized by gradient descent finds the cheapest route to a high
> score, whether or not it is the route you meant. Here the objective is
> "make this neuron fire" and the parameters are pixels. Module 01's autodiff
> will compute that gradient for you. The lesson is that an interpretability
> tool is itself an optimization, and it can be gamed by the model it
> inspects.

**InceptionV1's last layer was a learning failure, not a packing strategy.**
**Reported.** Estimates of how many final-layer neurons are easily
interpretable:

```text
InceptionV1 (2014 checkpoint), 2017-era tools     ~5%     (others put it near 0%)
same checkpoint, better tools                    ~50%
same architecture, retrained with modern
  regularization                                 ~70%
ConvNeXt, same parameter count and data          ~99%    (explorable on the essay's site)
```

The jump from the third line to the fourth comes from architecture alone:
residual connections, LayerNorm, and a few other changes. They were introduced
to help deep networks *learn*, not to make them readable.

> **G2C interpretation.** Those are exactly the two pieces Module 09 has you
> add to the transformer block, for the same reason. The essay's claim is that
> they buy interpretability as a side effect, and that a model without a
> residual path forces its late layers to carry too much. Read alongside
> [Module 12](../modules/12-scaling.md), this is a claim that "learns better"
> and "reads better" are correlated.

**Squeeze a layer and neurons get coarser, not mixed.** This is the
experiment most directly aimed at superposition's prediction. **Reported.**
InceptionV1 was retrained with layer `mixed3b` cut from 480 channels to **10**,
with no bypass. If cramped layers pack unrelated features, this 48× squeeze
should produce overlapping mixtures. Instead, the 10 neurons stayed
interpretable but broad: one "curves and loops" neuron in place of ten curve
detectors, two circle neurons in place of several. The model still reached
58.35% top-1 accuracy. An SAE trained on the squeezed layer did **not**
recover clean curve detectors. Its nearest units were broad and bunched. The
same pattern held when a medium ConvNeXt's final layer was cut from 1,024
neurons to 13.

**Dictionaries mostly rediscover neurons.** **Reported.** On the ConvNeXt
layers studied:

- 88% of the SAE's activity on the last layer is near-copies of single neurons.
- Removing every dictionary direction that matches no neuron leaves accuracy
  unchanged.
- Searching over rotations of a layer for the basis where a few units carry
  the most activity lands back on the neuron basis, in every block.

> **G2C interpretation.** The last result has a clean explanation from
> Module 03. An elementwise nonlinearity like ReLU or GELU treats each
> coordinate separately, so the neuron basis is *privileged*. Rotate the
> hidden layer and the nonlinearity no longer commutes with it. That gives
> neurons a reason to align with meaningful directions. The residual stream
> has no elementwise nonlinearity, so it has no privileged basis, and the
> essay treats it separately.

**Memorization rides existing circuits.** **Reported.** Teaching the 7M-parameter
ConvNeXt that one umbrella photo is a "bee" took nine steps. The top guess went
umbrella → maypole → bee, and the change worked by switching off umbrella
neurons and switching on *existing bee neurons*, almost entirely in the last
three blocks. No isolated "this exact image means bee" unit formed.

**Weak activations are mostly fine distinctions.** **Reported.** Keeping only
about the strongest quarter of positive activations in InceptionV1's final
layer preserves 69.47% accuracy, versus 70.08% with everything. The answers
that flipped were mostly near-ties the model was already unsure about.

### Language: preliminary, but pointed

**Reported.** The language-model results are explicitly preliminary, and the
headline numbers come with caveats about thresholds. They are:

- **Qwen3-32B audit:** about **90%** of neuron poles have one main job (often with
  small related offshoots), about **7%** have two related jobs, and about
  **2–4%** are not yet understood. Dense and MoE models look similar. The
  essay stresses that the percentages depend on where you set thresholds for
  "one thing" and how far into weak activations you look.
- **GPT-2 XL, revisited:** the same model that scored under 2% in 2023. With
  500 quickly generated "neuron papers," about **70%** now look like roughly one
  thing, about **5%** look pristine, and the rest do a few broad things (up to
  about six, mostly two).

The individual examples are more instructive than the totals:

- **The "pr" neuron.** It fired on "pr", on Base64 URL strings, and on some
  Bengali. Decode the URLs and transliterate the Bengali: they are "pr" words.
- **The "comma" neuron that tracks Colorado.** Automated labeling called it a
  polysemantic comma neuron. It fires on commas *after* Colorado has come up,
  brightening as the context builds. Steering it makes the model want to go
  hiking. **Reported.** Attention tends to collect running summaries on
  punctuation (the essay calls this the *summary motif*), so neurons that
  track a context often fire on commas rather than on the content words.
- **Towards Monosemanticity's opening example.** That paper introduced a
  neuron firing on "academic citations, English dialogue, HTTP requests, and
  Korean text." The essay proposes it is a **"don't say a dot"** neuron. Most
  examples fire on or near a period, and eight of its ten strongest negative
  output connections begin with one. The essay concedes that without the
  original model it can't narrow this further.
- **Santa lives on the negative pole.** In one Qwen3-32B attribution graph,
  removing the 8 neurons whose *negative* side encodes Santa lore drops "Santa"
  from near-certain to 6%. Removing the 20 neurons actually labeled
  Christmas or Santa only drops it to 76%.

**Explanations that can lose.** The obvious objection is that a motivated
reader can explain anything. **Reported.** For 183 neurons with positive-side
explanations, an agent read *only* the explanation and wrote five new texts that
should fire the neuron and five near misses that shouldn't. The new texts fired
strongly 76% of the time and the near misses 11%. The explanation won for 164
neurons, tied for 15, and lost for 4. Then the agent wrote texts about the most
distant concept it could invent, the kind of thing a packing network would
store in the same neuron. Those fired strongly about 1% of the time, and 179 of
the 183 neurons never fired on them at all. In a separate "blind maximum" test,
a sentence written from the explanation alone often fired a neuron harder than
anything found in 10 billion tokens of corpus.

**Experts are specialized too.** **Reported.** In MoE models, experts get
narrower with depth, much as neurons do. The essay argues that MoE is almost
an *anti*-packing architecture: a small linear router would have to recognize
ten thousand unrelated triggers to send them all to one packed expert. See
[Beyond: mixture of experts](../beyond/moe.md) for the routing mechanics.

### A tool bug hiding inside your BaseLM's activation function

One finding is pure architecture, and you can verify the arithmetic yourself.

Modern LLMs, including the Llama-architecture BaseLM you load in
[Module 12](../modules/12-scaling.md), use **SwiGLU** feed-forward layers rather
than Module 09's GELU MLP. Each neuron multiplies two projections of the
input:

```text
neuron activation  a = SiLU(g) · u          g = gate pre-activation,  u = "up" (value) projection
SiLU(g) = g · σ(g)
```

**Derived.** SiLU is not zero for negative inputs. It dips to a minimum of
about **−0.278** at `g ≈ −1.28`. So if `g < 0` *and* `u < 0`, the product is
**positive**. A neuron can produce a positive activation for two unrelated
reasons:

```text
          u > 0                 u < 0
g > 0     + × + → strong +      + × − → −
g < 0     − × + → small −       − × − → small +    ← the "double-negative" regime
```

Because `|SiLU(g)|` is capped at about 0.28 when `g < 0`, the double-negative
contributions are usually small. They land in the *middle* of a neuron's
activation range, which is exactly where a top-k labeling tool looks once the
strongest examples run out.

**Reported.** In one "enemy" neuron, 131 of 133 responses between the 75th and
99th percentiles were `− × −`, while all 93 responses at the 99.99th percentile
and above were `+ × +`. For about **4%** of Qwen3-32B neurons, this echo fills
the *whole* top of the activation list. The positive double-negative examples
tend to resemble the neuron's *negative* pole, so a tool that reads only the
product sees two unrelated meanings. Ablating the regime network-wide raised
most benchmarks but lowered some reasoning ones. The essay calls on
architecture designers to look into it.

> **G2C interpretation.** This is the essay's thesis in miniature. An apparent
> mixture turns out to be one neuron seen through an instrument (sort by the
> product) that conflates two sign regimes. You have the background to see it
> from the activation function's graph alone.

---

## The proposed mechanism: recursive generalization and circuit ecology

If neurons aren't packing, what explains the residue: the neurons that really
do fire on a few different things?

**Reported: recursive generalization.** A neuron grows like a branch: it
generalizes along one axis, then sprouts offshoots from the new territory. The
canonical example is a language neuron that started on the letters "iy" (as in
"Miyazaki"), grew to "diy," and then to "do it yourself," which contains no
"iy" at all. Each step is local, but the endpoint looks polysemantic if you
only see the finished neuron.

```text
          "iy" ──► "Miyazaki"
            │
            └──► "diy" ──► "do it yourself"     (no "iy" left)
```

**Reported.** Retraining InceptionV1 from scratch and watching its messiest
late-layer neurons form showed this pattern every time. One neuron started on
rapeseed fields and reached early toward balls and fruit. It then kept
spreading into dogs, and then broccoli, until the fields were nearly gone. In a
ConvNeXt trained on the same data, the equivalent neuron reached for balls and
fruit, let them go, and stayed on the fields. The essay credits the idea to a subsection of its own earlier
[*Multimodal Neurons*](https://distill.pub/2021/multimodal-neurons/) paper,
which called these "conjoined features."

**Reported: circuit ecology.** The essay also proposes a reason why
generalization keeps winning, and labels it a guess. Circuits *race* to capture
the remaining error. A circuit that fires broadly, on "green grass" rather than
"pixel 322 is this exact color," receives gradient on more of each example, so
it grows faster and starves narrower competitors. That echoes
[gradient starvation](https://arxiv.org/abs/2011.09468) and
[coherent gradients](https://arxiv.org/abs/2002.10657). Once a "car region"
exists, the next car-like thing is cheaper to learn *there* than anywhere else.
On this view, arbitrary packing loses for the same reason: an obscure
combination of unrelated neurons is a poor place to build, with weak
gradient paths and nothing useful to inherit.

**Reported.** Two experiments support the "generalize from step one" part. A
network trained on a single Labrador photo learned mostly "dark and green,"
not "dog." A network trained on one photo each of ten classes labeled about 1
in 5 new photos correctly, twice chance, mostly by color. The essay notes the
exception: in [grokking](https://arxiv.org/abs/2201.02177) on modular
arithmetic, memorization wins first, though the generalizing circuit grows
underneath.

> **G2C interpretation.** This is a claim about *training dynamics*, the
> subject of [Module 03B](../modules/03b-training.md). The old superposition
> picture was a *storage* argument: N features, d dimensions, how do they fit?
> The essay replaces it with a *learning-speed* argument: which structure
> captures the error first? That reframing is testable on small models, and
> it makes a falsifiable prediction the essay states outright. The neurons
> and heads that carry a rare fact should be *related* to that fact, not
> unrelated ones chosen because they seldom collide.

---

## What the essay does and does not claim

**It does claim** that the canonical evidence for arbitrary packing was
misread. It claims the vision models it studied show no packing even under
direct pressure, that modern language-model neurons are mostly legible, and
that the network's own mechanisms are a sound place to start studying it.

**It does not claim:**

- **That every neuron matches a human concept.** The essay says the opposite
  would be bad news. Networks find concepts, like high-low frequency
  detectors, that people hadn't named.
- **That a layer represents no more patterns than it has neurons.** It accepts
  "more features than dimensions" as composition.
- **That its percentages are precise.** It calls them "rough pointers" and
  spends a footnote on how thresholds move them.
- **To understand attention.** Attention is "mostly outside the scope." The
  essay offers a tentative estimate that global per-head names would add
  about 5% explanatory value on top of neurons plus local attention flow.
- **To have ruled out continuous packing in the residual stream.** It says that
  "hasn't been a bottleneck," and that crushing the stream 8× made maps
  messier but not cleverly interleaved. It does not claim a proof.
- **Certainty.** It puts itself at "maybe 85% sure," and frames that as 85%
  that nearly *everything* comes from generalization, not that 85% of neurons
  do.

**Reported.** The essay also argues for a style: **exploratory**
interpretability, which starts from a question and zooms between neurons,
circuits, and data like a map, over **enumerative** interpretability, which
tries to produce a complete list of features to audit. It favors "tethered
abstraction": high-level maps that can always be followed back to specific
neurons. Dictionaries and probes are still useful, it says, but "at the end,
not the start."

---

## Where informed readers will push back

**G2C interpretation.** This section is the course's reading, not the essay's
and not a survey of published responses. A careful reader should weigh these:

1. **Most of the language evidence isn't checkable yet.** The vision results
   come with trained models and explorable microscopes. The language numbers
   (90/7/2–4, the GPT-2 XL revisit, the near-miss tests) depend on a companion
   paper that hasn't been published. **Not disclosed:** neuron-sampling
   procedure, labeling thresholds, the corpus behind "strongest activation,"
   and the agents' instructions.
2. **Agents wrote the explanations, and agents graded them.** The near-miss
   and blind-maximum tests are a real improvement on simulation scores,
   because they let an explanation *lose*. But the explanations were
   generated by agents, and the test texts were written by agents too. The
   author describes inspecting thousands of attribution graphs and hundreds of
   weak activations by hand. How that human review fed into the headline
   percentages isn't specified.
3. **Re-reading famous examples is a narrow test.** Showing that 4e:55 is a V
   detector, and that the 2019 circuit was pose, removes the field's founding
   examples. It does not by itself show that the large-scale SAE results, such
   as [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
   on Claude 3 Sonnet, are mostly rediscovering neurons. The essay's direct
   dictionary comparisons are on ConvNeXt layers. Its language comparison is
   one attribution graph re-done with neurons.
4. **Part of the disagreement is about words.** The essay narrows
   "superposition" to arbitrary packing, and it says it does not dispute *Toy
   Models'* experiments in their own setting. A reader from that tradition may
   reply that their claim was always about directions and capacity, which the
   essay grants. The empirical question that survives is which world *real*
   networks live in: one where almost everything is related to something, or
   one with many facts that are related to nothing.
5. **"Explanation of the gaps" cuts both ways.** The essay accuses the field of
   treating "doesn't make sense" as evidence for packing. It acknowledges the
   mirror risk: assuming every neuron *must* have a sensible story. Its
   defense is that the other causes are each testable. That is right in
   principle, but it means the strength of the conclusion depends on how
   many neurons have actually been tested, which returns to point 1.

None of these refutes the essay. Together they mark where the burden of
evidence now sits: on the companion paper.

---

## The laptop boundary, and what you can check yourself

This is where a research Brief differs from a model Brief. You cannot run
Kimi K3. But the essay's central object, a single neuron and the examples
that drive it, works at any scale.

- **Browse the essay's microscopes.** The ConvNeXt explorers
  ([medium](https://already-clean.goodfire.com/microscope/medium.html),
  [large](https://already-clean.goodfire.com/microscope/large.html)) and the
  [early-training map](https://already-clean.goodfire.com/training-map.html)
  let you check the vision claims directly in a browser.
- **Your BaseLM has real SwiGLU neurons.** **Derived.** The course's default
  BaseLM, SmolLM-360M, has 32 layers with 2,560 SwiGLU hidden units each:
  `32 × 2,560 = 81,920` neurons. It is small enough to run on any course
  machine. The course's BaseLM adapter exposes the underlying Hugging Face
  model as `.inner`, so a forward hook on one layer's MLP can record `g` and `u`
  separately. You can then sort a neuron's top examples by sign regime and
  see whether the double-negative echo shows up. Here is a sketch, not an
  assignment:

  ```python
  mlp = model.inner.model.layers[L].mlp           # Llama MLP: gate_proj, up_proj, down_proj
  def record(module, inputs, output):
      x = inputs[0]
      g, u = mlp.gate_proj(x)[..., N], mlp.up_proj(x)[..., N]
      log.append((g.detach(), u.detach()))        # regime = (sign(g), sign(u)) per token
  handle = mlp.register_forward_hook(record)
  ```

- **Your TinyLLM is a test of the essay's scale claim.** Module 09's FFN uses
  GELU and the model is tiny. **Reported.** The essay predicts that small or
  simplified models have *broader* neurons with more generalization residue,
  and that bigger models learn narrower neurons that are easier to recognize.
  If your own model's neurons look mixed, that is what the essay expects,
  not evidence against it. The interesting question is whether the mixtures
  are *local*, like "iy" → "diy," or arbitrary.

None of this is required. It is the easiest way to form your own opinion about
an argument whose core evidence is "look closely."

---

## Coverage map

| Concept in the essay | Label | Where to anchor it |
|---|---|---|
| Neurons as hidden units; elementwise nonlinearity | **Built in g2c** | [Module 03](../modules/03-nn.md) |
| Gradients with respect to inputs (feature visualization) | **Conceptual bridge** | [Module 01](../modules/01-autodiff.md): same autodiff, different leaf |
| FFN sublayer, residual stream, LayerNorm | **Built in g2c** | [Module 09](../modules/09-transformer-block.md) |
| Attention placing information (summary motif) | **Conceptual bridge** | Modules [07](../modules/07-attention.md)–[08](../modules/08-multi-head-attention.md) |
| SwiGLU and its sign regimes | **Conceptual bridge** | Module 09 (GELU), [Module 12](../modules/12-scaling.md) BaseLM (SwiGLU) |
| Training dynamics, gradient competition | **Conceptual bridge** | [Module 03B](../modules/03b-training.md) |
| Ablation and steering via forward hooks | **Conceptual bridge** | Any forward pass you've written |
| MoE expert specialization | **Conceptual bridge** | [Beyond: MoE](../beyond/moe.md) |
| Sparse autoencoders / dictionary learning | **Not yet covered** | — |
| Attribution graphs and transcoders | **Not yet covered** | — |
| Convolutional vision models (InceptionV1, ConvNeXt) | **Not yet covered** | [Beyond: multimodal](../beyond/multimodal.md) covers ViTs, not CNNs |

---

## A reading route

The essay is long, about 65 minutes. If you have less time:

1. **The Short Version.** The whole argument in about ten minutes.
2. **Cars and Animals** and **A History of Microscope Errors.** The re-reading
   of the canonical examples, with the figures.
3. **Superposition of the Gaps.** The five-causes argument.
4. **What Neurons Do Under Pressure** and **What Dictionaries Learn.** The
   experiments most directly aimed at packing.
5. **Exploring Language Models.** Read it knowing it is the preliminary half.
6. **Appendix: A Note on Features and Definitions.** Read this before arguing
   about the essay with anyone. Most disagreements are here.
7. **Appendix: Separating the Sign Regimes**, alongside this Brief's SwiGLU
   section.

Skip "Neuroscience, and Why I Care" unless you're interested in the author's
motivation. It is personal rather than evidential.

## After this Brief, you should be able to

- Distinguish the two meanings of "superposition," and say which one the
  essay disputes.
- Explain why the cat-and-car neuron was re-read as a V detector, and what the
  negative pole contributed.
- Explain how feature visualization can be reward hacked, in terms of
  gradient ascent on inputs.
- List the five candidate causes of apparent polysemanticity, and how each
  could be tested.
- Derive why SwiGLU's double-negative regime produces small positive
  activations that can mislead a top-k labeling tool.
- State recursive generalization and circuit ecology in one sentence each, and
  name a prediction that would falsify them.
- Say which of the essay's claims are well supported now, which await the
  companion paper, and where you would look first to check.

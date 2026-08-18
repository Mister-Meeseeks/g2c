# Beyond — Multimodal language models

> **Question this module answers:** *How does a next-token model read a picture?*

<!-- TODO(hero pipeline): asset not yet generated -->
![An MNIST digit sliced into sixteen patches that enter the transformer's token stream alongside word embeddings, with the loss mask lit only over the text positions.](multimodal/BeyondMultimodal-Hero.png)

The transformer you built in Module 09 never knew it was processing *text* — it processes vectors. This module makes that concrete: you'll slice MNIST digits into patches, project them into the residual stream next to word embeddings, and train a StoryLM-1M-class backbone from scratch to generate captions with the same next-token loss it has always used. The result is deliberately tiny. It teaches the modality-to-residual-stream interface while making the gap to a production vision-language model explicit.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course. Come here in any order, whenever a model card or paper names the idea and you want the load-bearing version — built, trained, and broken on your own machine.

---
## Before you start

* *Review*
	* [05-embeddings](../modules/05-embeddings.md) for the token-to-vector step this module generalizes
	* [09-transformer-block](../modules/09-transformer-block.md) for the residual stream the patches will enter
	* [13-sft](../modules/13-sft.md) for loss masking — it returns here with a new justification
	* [03-nn](../modules/03-nn.md) for the MNIST MLP classifier, which becomes this module's baseline
* *Finish*
	* `g2c/transformer` ([09-transformer-block](../modules/09-transformer-block.md))
	* `g2c/sft` ([13-sft](../modules/13-sft.md)) — the masked cross-entropy is reused directly
	* `g2c/training` ([03b-training](../modules/03b-training.md))
* *Run*
	* `./notebook.sh multimodal`; torchvision downloads MNIST on first use if it is missing
	* `G2C_APPLY_SOLUTIONS=01-13 ./notebook.sh multimodal` instead of the plain launch if you're entering without your own implementations

---
## Where this fits in

In Module 04 we turned raw text into discrete tokens. In Module 05 we turned those tokens into vectors, establishing the move that makes language models possible. From that point on the architecture is geometry, not text processing. In Module 09 we built transformers to work with text, but it never explicilty "knew" it was working with text. Transformers remix vectors with no opinion about what they represent. (Remember how LLMs can't count the r's in strawberry because they're blind to letters.)

That indifference is the secret to multimodal models. Transformers will work with *any* input — an image, an audio clip, a video frame — that you can turn into a sequence of vectors. Embed those vectors into the same dimension as language vectors, and the transformer will attend to them the way it attends to text. Distinctions between modalities get left at the door of the embedding layer.

Model cards often blur two separate design choices:

* **Visual frontend.** Raw patches can be projected directly, or a vision encoder can turn them into richer features before a projector or resampler maps them to width `D`.
* **Training regime.** The visual frontend and language model may be pretrained separately and joined later, or trained jointly on multimodal data from early in training. Components can be frozen, partially tuned, or updated end to end.

“Native multimodal” is not a precise promise that raw patches enter the decoder without a vision tower. Treat it as a cue to inspect both axes. This module chooses the simplest corner — raw MNIST patches, one projector, joint training from scratch — because it isolates the interface using parts you already built.

## The big idea

The *visual frontend* is a pipeline that converts images (or other media) into a sequence of vectors with matched embedding dimension `D`. From the transformer's perspective, the specific implementation of the visual frontend does not matter. 

For images, input typically starts as small patches or pixels. The visual frontend uses a *vision tower* to encode each patch element into a *visual vector*. Depending on the tower, these can encode features ranging from edge detection to "this is an eye on a face" to OCR info. The stream of visual vectors then feeds into a *projector*, which is responsible for converting to a sequence of embedding vectors that compatible with the transformer. 

If the dimensionality already matches, a projector can be as trivial as "pass the visual vectors one for one to the transformer". But projectors might be significantly more complex and sample, shuffle, compress or mix information across the sequence. 

The visual frontend may be pretrained separately, or it may be trained jointly with the transformer using multimodal data from the corpus. This module chooses the simplest approach — raw MNIST patches, one projector, joint training from scratch — because it isolates the interface using parts you already built.

The whole pipeline, end to end:

```
   28×28 MNIST digit
        │  patchify: 4×4 grid of 7×7 patches
        ▼
   16 patches × 49 pixels                     "This is a 7."
        │  Linear(49 → D)                          │  token embedding (Module 05)
        ▼                                          ▼
   16 patch vectors (D,)                      text vectors (D,)
        │                                          │
        └────────────── one sequence ──────────────┘
                             │
                             ▼
              [p₁ p₂ … p₁₆  This  is  a  7  .]
                             │
                             ▼
                  TransformerLM (Module 09, unchanged)
                             │
                             ▼
              next-token loss on TEXT positions only
```

Three things to notice, because they carry the whole module:

1. **The transformer is unchanged.** Same blocks, same attention, same residual stream. The only new learned component in this toy is one linear projection (`49 → D`); the existing position table covers patch slots too.
2. **The objective in this module is unchanged.** Caption training is next-token prediction over the mixed sequence. Production pretraining may add contrastive or other auxiliary objectives, but the decoder can still learn from ordinary token loss.
3. **The loss mask does the modal bookkeeping.** Image positions are inputs, never targets — the model attends *to* patches but is never asked to *predict* them.

### Patches are the tokenizer for images

Module 04 solved "text is continuous, models need discrete chunks" with BPE. Images pose the reverse problem — they're a dense grid with no natural vocabulary — and the field's answer (from ViT) is almost embarrassingly simple: cut the grid into fixed-size squares, flatten each square into a vector, and apply a learned linear projection.

```
   BPE (Module 04):    characters → merge rules → token ids → embedding table lookup
   Patchify (here):    pixels     → fixed grid  → patch vecs → linear projection
```

The projection plays the embedding table's role, with one structural difference: there's no discrete vocabulary. A patch doesn't get looked up; it gets *transformed*. Two nearly-identical patches land at nearly-identical vectors — image "tokens" live on a continuum. This is also why the loss mask isn't optional: cross-entropy needs a discrete target, and patches aren't discrete. (Models that *generate* images have to solve exactly this — see "What we don't cover.")

### Position has two dimensions now

A flattened patch sequence has row-major order: patch 5 is "row 1, column 1," but the model only sees "position 5." Module 09's learned position embeddings extend without modification — patches get position slots like any other token. Exercise 4 compares row-major order with one fixed random permutation. Consistency preserves the identity of each slot; the experiment measures how much the row-major spatial prior helped this model rather than promising a particular accuracy drop.

### The production-system gap

Our visual frontend is `Linear(49 → D)` over raw seven-by-seven pixel patches. It must learn edges, strokes, digit identity, and alignment with caption tokens from 60,000 simple grayscale examples. MNIST makes that plausible; natural images do not.

A production understanding system more often looks like this:

```
high-resolution image
    → resize / tile / normalize
    → large vision tower
    → semantic patch features
    → projector or token resampler
    → language-model-width vectors
    → shared transformer context
```

The vision tower supplies features already sensitive to shapes, objects, text, and spatial structure. A projector aligns widths; a resampler may reduce thousands of visual features to a controlled token budget. Large paired corpora teach alignment, and some systems train the tower and language backbone jointly while others freeze or stage them. What transfers from this module is the interface — visual vectors enter the residual stream and text loss can train across that boundary. What does **not** transfer is the claim that one raw-pixel linear layer is enough for production vision.

## Concepts to internalize

- **Modality dies at the embedding layer.** Past the projection, patches and words are citizens of the same residual stream, mixed by the same attention.
- **Captioning can use the same objective.** This module trains with next-token prediction over a mixed sequence; other multimodal stages may add auxiliary objectives.
- **Image tokens are inputs, not targets.** The loss mask carries the asymmetry — Module 13's mechanism, new rationale.
- **Patchify is tokenization for grids.** Fixed squares plus a linear projection; the "vocabulary" is continuous.
- **The visual frontend and training regime are separate choices.** A model can be jointly trained from scratch and still use a vision tower; a projector-based system can update every component end to end.
- **Model-card literacy:** inspect the vision tower, projector/resampler, visual token count, frozen/tuned components, and training stages instead of inferring an architecture from “native multimodal.”

### What we don't cover

- **Image generation.** Emitting images requires a discrete visual vocabulary (VQ tokenizers) or a diffusion head — a genuinely different output machinery. This module is understanding-only.
- **Contrastive pretraining itself.** We describe what CLIP-style encoders provide; training one is a batch-size-hungry exercise that fights the laptop constraint.
- **Audio, video, and streaming.** Different token rates, synchronization, and chunking — the Omni-class problems. The representation story is the same; the engineering is its own topic.
- **Resolution tiling and token budgets.** Production VLMs slice large images into tiles and spend hundreds of tokens per image. Important cost machinery, no new concepts.

---
## What you'll build

Package: `g2c/multimodal/`

```python
def patchify(images, patch_size): ...                       # SCAFFOLDED
    # (B, H, W) → (B, num_patches, patch_size²), row-major,
    # pure reshape/permute arithmetic — no learned parameters

class PatchEmbedding(Module):
    patch_size: int
    embedding_dim: int
    proj: Linear                          # (patch_size², D)

    def parameters(self): ...                               # implemented
    def forward(self, images): ...                          # SCAFFOLDED
        # patchify, project, return (B, num_patches, D)

class MultimodalLM(Module):
    lm: TransformerLM                     # Module 09's model, unmodified
    patch_embed: PatchEmbedding
    image_token_id: int                   # reserved <img> placeholder

    def parameters(self): ...                               # implemented
    def forward(self, token_ids, images): ...               # SCAFFOLDED
        # embed text, splice patch embeddings at each <img>
        # placeholder, run the transformer over the mixed sequence

def build_caption_batch(images, labels, patch_size): ...    # implemented
    # renders "<img>×N This is a 7 . <end>" with the loss mask zeroed
    # over the image span — reuses g2c/sft's masked collation
```

Total scaffolded code: roughly 35 lines across three functions. The splice in `MultimodalLM.forward` is the one genuinely fiddly part — index arithmetic aligning patch vectors, text embeddings, positions, and the mask — and its tests are correspondingly specific.

## How to run the tests

Tests live in `tests/test_multimodal.py`. Initial state: 2 passed (the provided caption vocab and batch builder), 11 failed.

```bash
source .venv/bin/activate

pytest tests/test_multimodal.py                # all module tests
pytest tests/test_multimodal.py -x             # stop at first failure (recommended)
pytest tests/test_multimodal.py -k patchify    # patch arithmetic only
pytest tests/test_multimodal.py -k splice      # sequence-assembly tests only
pytest tests/test_multimodal.py -v             # verbose
```

The anchor test: `test_patchify_roundtrip` asserts patches reassemble into the original image exactly — patchify is pure indexing, and if the round trip is lossless, the fiddliest arithmetic in the module is verified before any training starts. `test_splice_preserves_text_positions` then pins the seam where most real bugs live.

## Exercises

Open the working notebook with `./notebook.sh multimodal`, write your answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading when you're ready. Partial submissions are fine because blank answers are skipped.

To launch it:

```bash
./notebook.sh multimodal
```

If at any point you want to archive the work in your current notebook and restart fresh:

```bash
./notebook.sh multimodal --fresh
```

1. **Patchify and look.** Slice digits at `patch_size ∈ {4, 7, 14}`, visualize the grids, verify the round trip. Note the sequence-length cost of each choice — this is the "an image costs N tokens" line item, held in your hand.
2. **Train the caption model.** Train a four-layer, `D=128`, four-head, `hidden=512` backbone from scratch on `"<img>×16 This is a 7 . <end>"`. Watch masked train/validation loss and generate captions for held-out digits autoregressively.
3. **Score generated captions.** Parse the first digit token from each generated caption and compute MNIST test accuracy, then compare it with the result you recorded for Module 03's MLP. Both numbers must be measured; neither direction is guaranteed by the exercise.
4. **Shuffle the patches.** Retrain from the same initialization and batch order with one fixed random patch order. Compare generated-caption accuracy and explain the measured effect.
5. **Bridge the toy to production.** Identify what the shared-vector interface teaches and what production vision towers, projectors/resamplers, resolution pipelines, and large-scale multimodal training add. Explain why “native multimodal” does not imply “no vision encoder.”
6. **Two images, one sequence (optional).** Verify that two image-placeholder spans splice into one context. Treat this as an interface smoke test; training and evaluating genuine two-image binding is an extension.

## Pitfalls to expect

- **Splice misalignment.** Off-by-one between the patch span and the text that follows it shifts every downstream position — loss falls anyway (the model adapts to the garbled layout), accuracy craters. The splice tests exist because this failure is silent in the loss curve.
- **Loss computed over patch positions.** Cross-entropy against a continuous input isn't meaningful; depending on your indexing this either crashes or quietly trains the model to "predict" placeholder ids. The mask must zero the whole image span.
- **Normalizing pixels twice — or not at all.** Raw 0–255 patches into a `Linear` produce giant activations and an ugly first hundred steps. Normalize once, in `build_caption_batch`, and nowhere else.
- **`max_seq_len` too small.** Sixteen patches plus a caption fits; two images plus a longer caption may not. Check before Exercise 6, not during.
- **Position table too small for the patch count.** At `patch_size=4` an image is 49 patches, not 16 — the position embedding table must cover the longest mixed sequence you build.
- **Calling teacher-forced digit logits “generated captions.”** Exercise 3 begins from image placeholders and feeds predictions back autoregressively. Reading the digit target position while the true prefix is supplied measures an easier task.
- **Comparing against the MLP unfairly.** Use the same canonical MNIST train/test split and your measured Module 03 result. Do not replace it with a remembered benchmark number.

## M-series notes

- **The required path trains two StoryLM-1M-class models from scratch.** The ordered captioner and shuffled-patch ablation are both short-context MNIST runs, but generated-caption evaluation also performs several autoregressive passes over the full test set. MPS is recommended.
- **Patchify on CPU is fine.** It's pure indexing; don't reach for the GPU until training starts.
- **Memory is a non-issue** — the largest object in the module is the MNIST tensor itself.

---
## Reading

Primary:

- **Dosovitskiy, Beyer, Kolesnikov et al., "An Image is Worth 16x16 Words" (ViT, 2020).** The patchify-and-project move, at scale, with the "transformers don't care about modality" result that started everything. §3.1 is this module's `PatchEmbedding`.
- **Radford, Kim, Hallacy et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021).** Where adapter-recipe semantics come from. Read §2 for the contrastive objective; the rest is evaluation.
- **Liu, Li, Wu, Lee, "Visual Instruction Tuning" (LLaVA, 2023).** The adapter recipe in its cleanest form — frozen CLIP, one projector, instruction data. After this module, the architecture diagram reads in one glance.

Secondary:

- **Adept, "Fuyu-8B" (2023).** A direct-patch decoder example with no separate vision tower — the closest published relative of this module's visual frontend.
- **Alayrac, Donahue, Luc et al., "Flamingo" (2022).** A useful reference for interleaved image-text sequences, the interface explored by Exercise 6.
- **Meta, "Chameleon: Mixed-Modal Early-Fusion Foundation Models" (2024).** Early fusion taken seriously at scale, including the training-stability costs the adapter recipe avoids.

Optional:

- **Current multimodal technical reports.** Read the visual-encoder, projector/resampler, token-budget, and training-stage sections separately; “native” alone does not answer any of those architectural questions.

## Deliverable checklist

- [ ] All tests in `tests/test_multimodal.py` pass — the patchify round trip and splice-alignment tests especially.
- [ ] Notebook: from-scratch StoryLM-1M-class caption model trained, with held-out generated captions and generated-caption digit accuracy compared against your measured Module 03 MLP result.
- [ ] Shuffled-patch ablation run, with a written sentence on what it showed.
- [ ] You can explain — out loud, without notes — why the transformer needs no architectural change to accept images, and where in the pipeline modality actually disappears.
- [ ] You can explain — out loud, without notes — why image positions are masked out of the loss, and what a model that *generates* images has to add.
- [ ] You can explain — out loud, without notes — what this raw-patch projector teaches, what a production vision tower/projector/resampler adds, and why “native multimodal” does not settle that architecture by itself.

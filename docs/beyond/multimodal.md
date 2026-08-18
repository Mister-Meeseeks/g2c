# Beyond — Multimodal models

> **Question this module answers:** *How does a next-token model read a picture?*

<!-- TODO(hero pipeline): asset not yet generated -->
![An MNIST digit sliced into sixteen patches that enter the transformer's token stream alongside word embeddings, with the loss mask lit only over the text positions.](multimodal/BeyondMultimodal-Hero.png)

We've spent the entire course learning how to model text. This week we discover that the same machinery can be used across images, audio, and video. The key insight is the LLMs natively work with vectors not words. If we have a way to turn media into semantically meaningful vectors, then we have a way to integrate it directly into an LLM.

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

In Module 04 we turned raw text into discrete tokens. In Module 05 we turned those tokens into vectors, establishing the move that makes language models possible. From that point on, the architecture operates on geometry rather than raw text. Module 09's transformer blocks mix vectors without containing text-specific attention or FFN machinery.

That architectural indifference is the opening for *multimodal models*. A transformer can process an image, audio clip, or video frame once a modality-specific frontend converts it into a sequence of vectors with the model's width. The operations after that interface are shared, but modality information does not disappear: the vectors still encode where they came from, and the model must learn how visual and textual representations differ and interact.

## The big idea

The *visual frontend* is a pipeline that converts images (or other media) into a sequence of vectors with matched embedding dimension `D`. The transformer-facing shape contract is simple, but the frontend's implementation matters enormously to what visual information those vectors contain.

For images, input typically starts as pixels divided into patches. A *vision tower* contextualizes those patches into visual features that may encode edges, shapes, objects, or text. A *projector* then maps the vision feature width and representation space into vectors compatible with the language model.

If the dimensions and representation spaces already align, the projector can be an identity mapping; more often it is a learned linear layer or MLP. A separate *resampler* or more general connector may mix or compress the visual sequence to control how many tokens enter the language model. Papers sometimes use these names loosely, so inspect both the width mapping and the token-count transformation.

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
       [<img> p₁ p₂ … p₁₆ </img> This is a 7 .]
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
3. **The loss mask does the modal bookkeeping.** The image span is input-only — the model attends *to* patch vectors but receives supervision only on caption targets.

### Patches are the tokenizer for images

Module 04 turned a variable-length symbol stream into a manageable discrete vocabulary with BPE. Images pose a different problem: they are dense grids with no natural patch vocabulary. ViT's answer is almost embarrassingly simple: cut the grid into fixed-size squares, flatten each square into a vector, and apply a learned linear projection.

```
   BPE (Module 04):    characters → merge rules → token ids → embedding table lookup
   Patchify (here):    pixels     → fixed grid  → patch vecs → linear projection
```

The projection plays the embedding table's role, with one structural difference: there is no discrete patch vocabulary. A patch does not get looked up; it gets *transformed*. Two nearly identical patches can land at nearby vectors — image "tokens" live on a continuum. The batch uses discrete `<image_patch>` ids only to reserve splice positions; their embeddings are overwritten by the continuous patch vectors before attention. Cross-entropy could still mechanically reward predicting those placeholder ids, so the loss mask excludes that meaningless surrogate objective. Patches provide conditioning inputs, while caption tokens provide supervised targets. Models that *generate* images need an actual image-output representation — see "What we don't cover."

### A 2D grid becomes a 1D sequence

A flattened patch sequence has row-major order: patch 5 is "row 1, column 1," but the model only sees "position 5." Module 09's learned one-dimensional position embeddings extend without modification — patches get sequence slots like any other token, not explicit row and column coordinates. Exercise 4 compares row-major order with one fixed random permutation.

The permutation preserves every pixel and a stable patch-to-slot mapping, so it does not remove spatial information in the way a fresh permutation per example would. It is an exploratory test of sensitivity to scan order and optimization, not a clean measurement of a built-in two-dimensional spatial prior. The causal mask also makes sequence order asymmetric because later patch states can incorporate earlier patches, but not vice versa.

That attention geometry is another part of the toy-to-production gap. This decoder processes the flattened image with causal, one-dimensional attention. A typical vision tower instead lets patches attend bidirectionally within the image and supplies explicit two-dimensional position information before its features reach the language model. The tower therefore contributes not only better feature extraction, but also a representation shaped around image geometry.

### Image boundaries and identity

This module represents one image as `<img> <image_patch>×N </img>`. Only the `<image_patch>` embeddings are overwritten; the start and end tokens survive as learned boundary signals. That makes two adjacent images structurally distinct even when they have different patch counts. 

Production systems vary—some expand one `<image>` token internally, while others add delimiters, modality/type embeddings, image identifiers, or extra position metadata. This is a clear convention, not a universal standard. Boundaries expose the intended grouping, but they do not guarantee that the model binds the right caption claim to the right image; training and evaluation must still teach and test that behavior. 

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

There are also two common ways to connect those visual features to the language model:

```
early fusion (this module):  [visual vectors | text vectors] → shared decoder

cross-attention:             visual vectors → separate visual memory
                             text states ───→ cross-attend to that memory
```

Early fusion spends ordinary context positions on visual vectors and lets self-attention mix both modalities. Cross-attention keeps a separate visual stream and adds layers through which text states read it. Neither topology is implied by the phrase “multimodal”; model cards must say where fusion occurs.

## Concepts to internalize

- **Modality-specific processing ends at a shared interface.** Past the projection, patches and words occupy the same residual stream and are mixed by the same attention, while their vectors still retain modality information the model can use.
- **Captioning can use the same objective.** This module trains with next-token prediction over a mixed sequence; other multimodal stages may add auxiliary objectives.
- **Image tokens are inputs, not targets.** The loss mask carries the asymmetry — Module 13's mechanism, new rationale.
- **Patchify is tokenization for grids.** Fixed squares plus a linear projection; the "vocabulary" is continuous.
- **The visual frontend and training regime are separate choices.** A model can be jointly trained from scratch and still use a vision tower; a projector-based system can update every component end to end.
- **Model-card literacy:** inspect the vision tower, attention geometry, projector/resampler, visual token count, fusion topology, boundary signals, frozen/tuned components, and training stages instead of inferring an architecture from “native multimodal.”

### What we don't cover

- **Image generation.** Emitting images requires a discrete visual vocabulary (VQ tokenizers) or a diffusion head — a genuinely different output machinery. This module is understanding-only.
- **Contrastive pretraining itself.** We describe what CLIP-style encoders provide; training one is a batch-size-hungry exercise that fights the laptop constraint.
- **Audio, video, and streaming.** Different token rates, synchronization, and chunking — the Omni-class problems. The representation story is the same; the engineering is its own topic.
- **Resolution tiling and token budgets.** Production VLMs slice large images into tiles and spend hundreds of tokens per image. Important cost machinery, no new concepts.
- **A full multimodal evaluation.** Caption accuracy tests this toy's digit recognition and generation path, not grounding, OCR, spatial reasoning, hallucination, or multi-image binding. Production systems need targeted probes for each capability and failure mode.

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
    patch_token_id: int                   # reserved <image_patch> slot

    def parameters(self): ...                               # implemented
    def forward(self, token_ids, images): ...               # SCAFFOLDED
        # embed every token, splice patch vectors at <image_patch>
        # slots, retain <img>/</img>, then run the transformer

def build_caption_batch(images, labels, patch_size): ...    # implemented
    # renders "<img> <image_patch>×N </img> This is a 7 . <end>"
    # with the loss mask zeroed over the image span
```

Total scaffolded code: roughly 35 lines across three functions. The splice in `MultimodalLM.forward` is the one genuinely fiddly part — index arithmetic aligning patch vectors, text embeddings, positions, and the mask — and its tests are correspondingly specific.

## How to run the tests

Tests live in `tests/test_multimodal.py`. Initial state: 2 passed (the provided caption vocab and batch builder), 12 failed.

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
2. **Train the caption model.** Train a four-layer, `D=128`, four-head, `hidden=512` backbone from scratch on `"<img> <image_patch>×16 </img> This is a 7 . <end>"`. Watch masked train/validation loss and generate captions for held-out digits autoregressively.
3. **Score generated captions.** Parse the first digit token from each generated caption and compute MNIST test accuracy, then compare it with the result you recorded for Module 03's MLP. Both numbers must be measured; neither direction is guaranteed by the exercise.
4. **Permute the scan order.** Retrain from the same initialization and batch order with one fixed random patch order. Compare generated-caption accuracy, then explain what the permutation preserved and why a single run measures sensitivity to sequence order and optimization rather than isolating a two-dimensional spatial prior.
5. **Bridge the toy to production.** Identify what the shared-vector interface teaches and what production vision towers, projectors/resamplers, resolution pipelines, and large-scale multimodal training add. Explain why “native multimodal” does not imply “no vision encoder.”
6. **Two images, one sequence (optional).** Verify that two explicitly bounded image spans splice into one context. Inspect the retained `<img>` and `</img>` tokens, then explain why boundaries expose grouping without proving genuine two-image binding. Training and evaluating that behavior is an extension.

## Pitfalls to expect

- **Splice misalignment.** Off-by-one between the patch span and the text that follows it shifts every downstream position — loss falls anyway (the model adapts to the garbled layout), accuracy craters. The splice tests exist because this failure is silent in the loss curve.
- **Loss computed over the image span.** Shifted targets there are structural `<img>`, `<image_patch>`, and `</img>` ids, so cross-entropy does not crash—it quietly rewards predicting sequence scaffolding instead of learning the caption task. The mask must zero the whole image span so patches and boundaries are conditioning inputs rather than supervised targets.
- **Normalizing pixels twice — or not at all.** Raw 0–255 patches into a `Linear` produce giant activations and an ugly first hundred steps. Normalize once, in `build_caption_batch`, and nowhere else.
- **`max_seq_len` too small.** Sixteen patches plus a caption fits; two images plus a longer caption may not. Check before Exercise 6, not during.
- **Position table too small for the patch count.** At `patch_size=4` an image is 49 patches, not 16 — the position embedding table must cover the longest mixed sequence you build.
- **Calling teacher-forced digit logits “generated captions.”** Exercise 3 begins from image placeholders and feeds predictions back autoregressively. Reading the digit target position while the true prefix is supplied measures an easier task.
- **Comparing against the MLP unfairly.** Use the same canonical MNIST train/test split and your measured Module 03 result. Do not replace it with a remembered benchmark number.

## M-series notes

- **The required path trains two StoryLM-1M-class models from scratch.** The row-major captioner and fixed scan-order permutation are both short-context MNIST runs, but generated-caption evaluation also performs several autoregressive passes over the full test set. MPS is recommended.
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
- [ ] Fixed scan-order permutation run, with a written sentence on what it preserved and what the observed difference can and cannot establish.
- [ ] You can explain — out loud, without notes — why the transformer blocks need no modality-specific architectural change to accept images, where modality-specific preprocessing ends, and why modality information itself does not disappear.
- [ ] You can explain — out loud, without notes — why image positions are masked out of the loss, and what a model that *generates* images has to add.
- [ ] You can explain — out loud, without notes — what this raw-patch projector teaches, what a production vision tower/projector/resampler adds, and why “native multimodal” does not settle that architecture by itself.

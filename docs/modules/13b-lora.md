# Module 13B — LoRA

> **Question this module answers:** *How do you fine-tune a model whose optimizer won't fit on your machine?*

<!-- TODO(hero): docs/modules/13b-lora/Module13B-Hero.png — frozen weight matrix with a
     thin trainable A/B bypass path, and an adapter file being carried away.
     Add the image reference + alt text here when the asset lands. -->

LoRA stands for *low ranked adapation*. Behaviorally it is almost identical to the full SFT approach we learned last week. However it uses linear algebra to reduce the effective training parameter set to be substantially smaller. This is possible because of the empirical properties of fine-tuning. All that means is we get the same results at much lower hardware requirements. A must for any practitioner working with local models

---
## Before you start

* *Review*
	* [13-sft](13-sft.md) — this module reruns its pipeline with a different parameterization
	* [03-nn](03-nn.md) — the initialization argument returns here with a twist
* *Finish*
	* Module 13 end to end: `tests/test_sft.py` passing and your hand-authored dataset saved at `data/work/module13/instructions.json` — Module 13B reuses it byte for byte
* *Run*
	* `./baselm.sh` if BaseLM isn't already set up

---
## Where this fits in

In Module 13 we learned how to finetune a model's behavior using SFT, a small high quality dataset, and the same gradient descent machinery we used for pre-training. It was effective, but despite what's effectively a "small nudge" to the model, it was as memory intensive as full pretraining.

Do the memory arithmetic for full fine-tuning:

```
   weights          1 float per parameter
   gradients        1 float per parameter
   AdamW m and v    2 floats per parameter
   ─────────────────────────────────────────
                    4 floats per parameter — before a single activation
```

At 362M parameters that's about 5.8 GB. At 3B parameters it grows to 48 GB. The upshot is that the memory requirement to train a model is orders of magnitude higher than to use a model, and that includes full SFT. Which also means that for a given hardware constraint (like our local Macbook), models that you'll regularly run will be out of reach for local finetuning.

It should be noted that this module is optional. Nothing later in the course depends on the lesson developed here. But this course, whose identity is "everything on your Mac", owes you this module.

## The big idea

Take a look back at the memory arithmetic. Three of those four tenants — gradients, `m`, `v` — only exist for parameters under optimization. Shrink the trainable set a thousandfold, and they shrink along with it. 

That's the whole pitch. *Low rank adaptation* (LoRA) is not a new objective, a new loss, or a new trainer. It is a surgical answer to one question: *what is the smallest set of parameters that can carry a fine-tune?* The empirical answer is surprisingly small. It's why LoRA became the default for fine-tuning especially when resources are constrained.

How do we actually shrink the trainable parameter set while still building on top of the full base model? We use basic linear algebra to take a *low dimensional projection* of the full parameter set. Because of the nature of high-dimensional geometry, almost any randomly initialized projection matrix will still be trainable. 

The actual matrix mechanics: Freeze the pretrained weight `W`. Then train a low-rank correction on top of it:

```
                x ──────────────► W (frozen) ──────► + ──► y
                │                                    ▲
                └──► A (in,r) ──► B (r,out) ──► × α/r
                      random         zero
                     trainable     trainable
```

The delta `A @ B` has the same shape as `W`, but is built from two skinny low-dimensional matrices. That reduces the effective dimensionality of the training set to `r * (in + out)` instead of `in * out` . At rank 8 on a 960-wide projection, that reduces 921,600 parameters down to 15,360. The rank knob makes the trade explicit.

### Why low rank is enough

The observation underpinning LoRA has a name: *intrinsic dimensionality* (Aghajanyan et al., 2020). Pretraining does the hard, high-dimensional work. Fine-tuning is just a small course correction. And small corrections empirically live in low-dimensional subspaces. Hu et al. measured it directly: fine-tuning deltas on large transformers are approximately low-rank, and constraining them to a rank of 4-16 barely cost any behavioral quality. Module 13 gave you the same fact from the data side. SFT teaches a *format*, and 50 examples suffice because a format is a small thing to learn. LoRA is the same lesson applied to weights instead of data. 

### The initialization asymmetry

The only new parameters that need to be initialized with LoRA are the `A` and `B` projection matrices. The weights of the network remain fixed at their pretrained values. For the two matrices we initialize with:

* `A` — Random using the same `Uniform(-1/√fan_in, 1/√fan_in)` used for neural network weights in Module 03.
* `B` — starts at exactly zero.

Two consequences to this initialization. First, at step zero the adapter has exactly zero impact on the neural network weights, and therefore the injected model starts bit-identical to the base model. Fine-runing starts from exactly the pretrained model,  not near it. 

Second, `A` starts with zero gradient at step one. Its backprop flows through `B` and `B` starts at zero. However after step one `B` moves from zero, and now `A` has a nonzero gradient. Both matrices learn after step two. But this is why at least one the matrices has to start non-zero. 

### Merge, unmerge, and the adapter as a file

Because the delta is just a matrix, it can be *folded in*: 

```
W += (A @ B) · α/r
```

This `merge()` operation makes the adapted layer cost exactly one matmul. `unmerge()` subtracts it back out. Since the base never trains, the durable output of a LoRA run is just the low rank A/B matrices, a few megabytes riding on top of gigabytes. Ship the adapter, not the full model — one base, many adapters, swap per task. This substantially reduces the storage and I/O costs of having many individualized adapters on a single machine.  

The un-merged workflow also buys something full rank SFT couldn't offer at any setting: **zero forgetting by construction**. Format forgetting, catastrophic forgetting, and the rest of Module 13's failure zoo happens *inside* the weights. Low rank adaptation leaves the vast majority of the weight entropy undisturbed, and therefore 

### What LoRA does *not* save

To be precise, because the misconception is near-universal: LoRA saves **memory**, not (much) **time**. The forward pass still runs the full frozen model; the backward pass still propagates through every layer to reach the adapters. What scales down is the gradient storage, the optimizer state, and the weight-gradient work for frozen layers — not the FLOPs of the network itself. Your wall-clock per step in the notebook will land near Module 13's. That is correct behavior, not a bug.

## Concepts to internalize

- **`requires_grad` is the valve on the gradient flow you built in Module 01.** A frozen parameter isn't guarded by the optimizer — it's invisible to autograd: no gradient tensor, no `m`, no `v`. Every byte LoRA saves traces back to this one flag.
- **Wrapping and freezing are separate decisions.** `inject_lora` adds adapters; `mark_only_lora_trainable` freezes the base. Doing the first without the second trains the whole model with adapters bolted on — and the loss goes down either way. Only the parameter counts tell the truth.
- **The optimizer's memory is sized by what `parameters()` exposes.** The course `AdamW` allocates `m`/`v` eagerly for every tensor it is handed. `LoRAModel` exists to hand it only the adapters — the course `Module` contract ("`parameters()` returns all *trainable* parameters") applied to a torch tree.
- **The no-op start is exact, and one-sided.** `B = 0` makes the delta zero; `A` random keeps the gradient path alive. Zero/random, not zero/zero, not random/random.
- **Merging changes cost, never function.** Merged and unmerged compute the same map (up to float reassociation — the notebook makes you find the `1e-6`); only the number of matmuls differs.
- **An adapter file is a contract.** Same base weights, same target names, same rank — or the load must fail loudly. `load_lora_state_dict` is strict for the same reason Module 13's chat template is: a silent near-match behaves wrong invisibly.

### What we don't cover

- **QLoRA.** LoRA over a 4-bit quantized frozen base — the recipe that puts 7B+ fine-tuning on consumer hardware. Same adapters, harder numerics; read Dettmers et al. after this module and it will feel familiar.
- **The variant zoo (DoRA, AdaLoRA, rsLoRA, …).** Refinements of where the delta lives and how it's scaled. Skim after you've built the original; none change the core mechanism.
- **LoRA on your course TransformerLM.** `g2c/lora` targets `torch.nn.Linear` trees because that is what BaseLM (and every HF model) is made of — and because your 30M course model doesn't *need* LoRA; full fine-tuning is already cheap there. The concepts transfer one-to-one; only the tree surgery differs.
- **Multi-adapter serving.** Hot-swapping adapters per request over one shared base — the production payoff of "ship the adapter." A systems concern, out of scope at course scale.

---
## What you'll build

Package: `g2c/lora/`

```python
class LoRALinear(torch.nn.Module):
    # wraps a torch.nn.Linear; A random, B zero; scaling = alpha/rank
    def forward(self, x) -> Tensor:                       # SCAFFOLDED
    def merge(self) -> None:                              # SCAFFOLDED
    def unmerge(self) -> None:                            # SCAFFOLDED


def inject_lora(model, target_names, *, rank, alpha=None) -> list[str]:
                                                          # implemented
def mark_only_lora_trainable(model) -> tuple[int, int]:   # SCAFFOLDED
def count_parameters(model) -> tuple[int, int]:           # implemented

def lora_state_dict(model) -> dict[str, Tensor]:          # implemented
def load_lora_state_dict(model, state) -> None:           # implemented

class LoRAModel(g2c.nn.Module):                           # implemented
    # trainer-facing view: parameters() -> only what training can move
```

One deliberate house-style exception, called out because you'll notice it: `LoRALinear` subclasses `torch.nn.Module`, not `g2c.nn.Module`. The model being adapted is a Hugging Face torch tree, and the adapter must live inside that tree — moved by its `.to()`, found by its `named_parameters()`. The adapter speaks the host's dialect; `LoRAModel` translates back to the course's.

Total scaffolded code: roughly 25 lines across four locations.

## How to run the tests

Tests live in `tests/test_lora.py`. Initial state: 10 passed, 11 failed. Everything runs on tiny synthetic models — no BaseLM download needed, and the fixtures use GQA-style non-square projections on purpose: a transposed merge that survives square layers fails loudly here.

```bash
source .venv/bin/activate

pytest tests/test_lora.py                     # all module-13B tests
pytest tests/test_lora.py -x                  # stop at first failure
pytest tests/test_lora.py -k forward          # the delta path only
pytest tests/test_lora.py -k "merge or unmerge"   # fold/unfold semantics
pytest tests/test_lora.py -k trainable        # the freeze and its guarantees
pytest tests/test_lora.py -v                  # verbose
```

## Exercises

To launch the exercise notebook run:

```bash
./notebook.sh 13b
```

If at any point you want to archive the work in your current notebook and restart fresh:

```bash
./notebook.sh 13b --fresh
```

Write your answers in the `Question:` / `Answer:` cells and ask a coding agent for hints or grading; partial submissions are fine — blank answers are skipped, not counted wrong.

1. **Count before you build.** Derive the rank-8 trainable-parameter count on paper — including the GQA wrinkle — then verify against `count_parameters`.
2. **The exact no-op.** Prove the injected, untrained model is bit-identical to the base. Then explain, via the chain rule, why `B` starts at zero and `A` doesn't.
3. **Train the adapter.** Rerun Module 13's SFT on your own dataset with ~0.2% of the parameters, and compare against your full-SFT run.
4. **Merge for deployment.** Fold the delta in, verify the function didn't move, and time unmerged vs merged vs base forwards.
5. **Unmerge: the eject button.** Round-trip the weights and account for the float residue — then argue why the pristine-base-plus-adapter-file workflow makes zero forgetting structural.
6. **Ship the adapter, not the model.** Save the state dict, load it into a fresh BaseLM, and watch the behavior arrive with a file ~1000× smaller than the model.
7. **Rank sweep (optional).** Rank 1 vs rank 8 on the same task — a firsthand measurement of how low-rank a format shift really is.

## Pitfalls to expect

- **Injecting without freezing.** The headline pitfall, and it's silent: training "works," loss falls, samples improve — and you fine-tuned all 362M with adapters along for the ride. Check `count_parameters` *after* `mark_only_lora_trainable`, every time.
- **Handing `SFTTrainer` the raw model instead of `LoRAModel`.** The course `AdamW` allocates `m`/`v` eagerly for every parameter it receives — frozen or not. Skip the wrapper and the optimizer quietly allocates ~2.9 GB of state for a model that cannot move.
- **The transpose in `merge`.** `torch.nn.Linear` stores `weight` as `(out, in)`; `A @ B` is `(in, out)`. On square `q_proj` a missing `.T` *runs and is wrong*; on the non-square `v_proj` it crashes. The test fixtures are non-square for exactly this reason.
- **Double-merging.** `merge()` on a merged layer must be a no-op, not a second add. Same for `unmerge()`. The `merged` flag is the guard; the tests pin it.
- **Training while merged.** After `merge()`, the delta lives inside the frozen base weight — gradient can no longer reach `A`/`B` meaningfully, and the base is no longer pristine. Merge is for inference; unmerge before touching the trainer again.
- **Dropping `alpha / rank`.** Forget the scaling and rank changes silently rescale your delta — a rank sweep then confounds capacity with step size.

## M-series notes

Wall-clock per SFT step lands near Module 13's — LoRA cuts memory, not FLOPs (see "What LoRA does *not* save"). What changes is the memory ledger, float32 on the 362M BaseLM:

```
   ┌─────────────────────────┬──────────────┬──────────────┐
   │ tenant                  │ full SFT     │ LoRA r=8     │
   ├─────────────────────────┼──────────────┼──────────────┤
   │ weights                 │   ~1.45 GB   │   ~1.45 GB   │
   │ gradients               │   ~1.45 GB   │    ~3 MB     │
   │ AdamW m + v             │   ~2.90 GB   │    ~7 MB     │
   ├─────────────────────────┼──────────────┼──────────────┤
   │ total (pre-activations) │ ~5.8 GB      │   ~1.5 GB    │
   └─────────────────────────┴──────────────┴──────────────┘
```

- On 8–16 GB machines this is the difference between "swapping" and "comfortable" — and at 1–3B it is the difference between impossible and routine.
- **Exercise 6 briefly holds a third copy of BaseLM** (~1.5 GB) while proving the adapter transplant. On 8 GB machines, restart the kernel before running it standalone.
- The optional rank sweep loads fresh BaseLM copies sequentially and deletes them between runs; expect a few minutes per rank at 150 steps.

---
## Reading

Primary:

- **Hu, Shen, Wallis et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).** The paper you just implemented. §4.1 for which weight matrices to adapt (their answer: query and value — the one you used); §7 for the rank ablation your Exercise 7 miniaturizes.
- **Aghajanyan, Zettlemoyer, Gupta, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" (2020).** The *why* underneath LoRA: fine-tuning objectives can be satisfied in surprisingly low-dimensional reparameterizations, and the dimension shrinks as the base model grows.
- **Dettmers, Pagnoni, Holtzman, Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs" (2023).** LoRA over a 4-bit frozen base — the recipe behind essentially every "fine-tune a 7B on your laptop" guide. Read it as this module plus quantization.

Secondary:

- **The Hugging Face PEFT library documentation.** The production implementation of everything in `g2c/lora` — `target_modules`, `merge_and_unload`, adapter files. After this module it reads as an API tour of code you've already written.
- **Raschka, "Practical Tips for Finetuning LLMs Using LoRA" (2023).** The empirical hyperparameter picture: rank, alpha, which layers, learning rates — a working engineer's ablation notes.
- **Liu, Wang, Yin et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024).** The strongest of the variants; decomposes the update into magnitude and direction. Representative of where the adapter literature went next.

## Deliverable checklist

- [ ] All tests in `tests/test_lora.py` pass.
- [ ] The no-op check in the notebook printed a max logit difference of exactly `0.0`.
- [ ] A trained adapter saved at `data/work/module13b/lora-adapter.pt`, and you looked at its file size next to the base model's.
- [ ] Your Module 13 base artifact is untouched — the fine-tune lives entirely in the adapter file.
- [ ] You can explain — out loud, without notes — why `B` starts at zero, `A` starts random, and what would go wrong with zero/zero.
- [ ] You can explain — out loud, without notes — which of the four memory tenants (weights, activations, gradients, optimizer state) LoRA shrinks, and why the wall-clock per step barely changes.
- [ ] You can explain — out loud, without notes — what must match between saver and loader for an adapter file to work, and why `load_lora_state_dict` refuses a near-match instead of loading the intersection.

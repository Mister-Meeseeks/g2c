# Syllabus — From Gradients to ChatGPT

![Cover illustration: a circus-themed map of the course. Numbered booths under a big top represent each of the twenty modules — pretraining, tokenization, embeddings, gradient descent, self-attention, multi-head attention, the transformer (a central tower of "Add & Norm / Feed Forward / Multi-Head Attention" floors), sampling, SFT, DPO, RAG, tools, the agent, the eval inspector, and an "inference booth" robot at the bottom. Banners read "A tiny LLM stack from first principles," "Data. Compute. Curiosity. That's all you need," and "Built step by step under the big top."](CourseCover.png)

A 20-week self-study course, preceded by a fast prerequisite review, building a tiny LLM stack from scalar autograd up through a working chat assistant. Each week builds on the previous one. The codebase grows in `g2c/` as a single Python package; later modules import earlier ones. 

## How to read this syllabus

Each week has the same structure:

- **Question** — the one-sentence motivation, in the style of the brainstorm.
- **Goal** — what you'll have built by the end.
- **Concepts** — the ideas to internalize.
- **Build** — what code lands in `g2c/<topic>/`.
- **Exercises** — concrete tasks to do.
- **Deliverable** — what "done" looks like.
- **Reading** — primary references.
- **M-series notes** — practical compute notes when relevant.

A "week" is one week of effort at the level of a rigorous elite-college course. Real calendar time will vary.

Module 0 is the exception: it is a fast review, not a full course week. Use it to put the necessary math, ML, PyTorch, and repo mechanics back in working memory before starting Module 1.

## Prerequisites

- Comfortable Python and software engineering hygiene (testing, packaging, debugging).
- Calculus through partial derivatives and the chain rule.
- Linear algebra: matrix-vector products, inner products, basis transformations.
- Classical ML literacy (regression, classification, train/val/test, overfitting).
- Limited prior deep-learning experience is expected — DL concepts are introduced as new material.

If these are familiar but rusty, start with [Module 0: Prerequisite review](modules/00-prerequisite-review.md). If several are new, treat Module 0 as a map for a longer fundamentals pass before beginning the course.

## Stack

- Python 3.11+
- PyTorch with the MPS backend (primary)
- Jupyter for exploration (`notebooks/`)
- Ollama for serving pretrained open models in the capstone

## Ground rules

- **From-scratch through the architecture (weeks 1–12).** When the topic of the week is the thing — autograd, attention, the transformer block — build it. Don't import a high-level abstraction that does the work for you.
- **Use a prertrained base model through behavior shaping (weeks 13–15).** Scaling experiments, SFT, DPO, and eval should support your self-trained TinyLLM. If those artifacts are too weak for the post-training lesson, use the default small pretrained BaseLM path.
- **Pivot to a pretrained open model for the assistant phase (weeks 16–20).** RAG, tools, agents, and the capstone use a local pretrained instruct model, called ProdLM in the course, so the system is actually usable.
- **Pedagogy beats performance.** Code should be legible. Optimization is a separate concern.
- **Tiny everything.** Tiny corpora, tiny models. The whole course thesis is that the tiny version teaches the idea.
- **Tracks are artifact choices, not separate courses.** Use the default models and datasets. Or experiment with larger or smaller artifacts to see the difference. The conceptual path is the same regardless. See [Course Tracks and Artifacts](tracks.md).

## Phase overview

| Phase              | Weeks           | Theme                                                            |
| ------------------ | --------------- | ---------------------------------------------------------------- |
| 0 — Review         | 0               | Fast prerequisite refresh                                        |
| I — Learning       | 1–3 + draft 03B | Scalar autograd → tensors → first neural net → training dynamics |
| II — Language      | 4–6             | Tokenization → embeddings/positions → next-token prediction      |
| III — Transformers | 7–10 + draft 09B | Attention → multi-head → block → pretraining → first LLM milestone |
| IV — Behavior      | 11–15           | Sampling → scaling → SFT → DPO → eval                            |
| V — Assistants     | 16–20           | Pretrained inference → RAG → tools → agents → capstone           |

---

## Module 0 — Prerequisite review — [module ↗](modules/00-prerequisite-review.md)

- **Question.** What do I need back in cache before building the stack?
- **Goal.** Refresh the exact prerequisite surface the course depends on: shapes, matmul, derivatives, chain rule, logits, softmax, cross-entropy, train/validation workflow, PyTorch tensors, and repo mechanics.
- **Concepts.** The training loop as the anchor; tensors and shape contracts; local derivatives and reverse-mode backprop; logits as unnormalized scores; loss as the scalar training signal; mini-batch optimization; `pytest -x` as the implementation workflow.
- **Build.** No package code. This is a readiness review and environment check.
- **Exercises.**
  - Trace tensor shapes through embedding lookup and a projection to vocabulary logits.
  - Compute a small matrix multiply by hand.
  - Derive a one-neuron gradient with the chain rule.
  - Compute a softmax and negative log likelihood from small logits.
  - Run the repo smoke test.
- **Deliverable.** A short readiness artifact — notes, a scratch notebook, or course journal page — proving you can trace one training step end-to-end.
- **Reading.** 3Blue1Brown linear algebra and backprop refreshers; PyTorch "Tensors" tutorial; Karpathy's micrograd lecture as the bridge into Module 1.
- **M-series notes.** Environment setup only. MPS becomes operationally relevant in Module 2.

## Phase I — Learning

### Week 1 — Scalar autodiff — [module ↗](modules/01-autodiff.md)

- **Question.** How does the model learn?
- **Goal.** Build a scalar-valued automatic differentiation engine from first principles.
- **Concepts.** Computational graph, forward pass, chain rule, reverse-mode differentiation, topological sort, gradient accumulation, the local-derivative-at-each-node abstraction.
- **Build.** `g2c/autodiff/` — a `Value` class supporting `+`, `*`, `-`, `/`, `**`, `exp`, `log`, `tanh`, `relu`, with `.backward()` traversing the graph in topological order.
- **Exercises.**
  - Implement gradient checking against finite differences.
  - Trace a small expression by hand; verify against the engine.
  - Build a 2-input neuron (linear + tanh) and train it on XOR using only your `Value` class.
- **Deliverable.** Engine + a notebook that trains an XOR-style MLP using `Value` only — no PyTorch.
- **Reading.** Karpathy, *micrograd* repo and "The spelled-out intro to neural networks and backpropagation: building micrograd" (YouTube).
- **M-series notes.** Pure CPU; runs in seconds.

### Week 2 — Tensors and matmul — [module ↗](modules/02-tensors.md)

- **Question.** How do we scale computation from single numbers to whole layers?
- **Goal.** Move from scalar autograd to vectorized tensor operations and develop intuition for why GPU/MPS-class hardware matters.
- **Concepts.** Vectors, matrices, broadcasting, batching, einsum, the FLOP cost of matmul, why deep learning workloads are matmul-dominated.
- **Build.** `g2c/tensors/` — a thin teaching wrapper that demonstrates batched matmul and broadcasting; port a couple of week-1 ops to operate on `torch.Tensor` (without using `torch.autograd`) to feel the speedup.
- **Exercises.**
  - Implement matmul three ways (nested loops, NumPy, PyTorch on MPS); benchmark across sizes.
  - Implement broadcasting from scratch on a toy class.
  - Reproduce a linear-layer + softmax forward pass with explicit shape annotations.
- **Deliverable.** A benchmark notebook plotting matmul cost across CPU loops, NumPy, torch CPU, torch MPS at sizes from 32 to 4096.
- **Reading.** PyTorch broadcasting docs; Karpathy "Neural Networks: Zero to Hero" lectures 2–3; Parr & Howard, "The Matrix Calculus You Need For Deep Learning."
- **M-series notes.** First MPS use of the course. Verify the backend works; expect occasional ops to fall back to CPU.

### Week 3 — First neural net — [module ↗](modules/03-nn.md)

- **Question.** How do numbers approximate functions?
- **Goal.** Train a small MLP from scratch with a clean training loop.
- **Concepts.** Linear layer, activations (ReLU, tanh, sigmoid), softmax, cross-entropy, gradient descent, mini-batching, train/val split, overfitting, weight decay.
- **Build.** `g2c/nn/` — `Linear`, `ReLU`, `Sequential`, `CrossEntropyLoss`, and an `SGD` optimizer hand-written. Use PyTorch tensors as the substrate but do not use `torch.nn.Linear` etc.
- **Exercises.**
  - Fit y = 3x + 2 with a single linear layer.
  - Classify points on the moons / circles toy datasets.
  - Train a 2-layer MLP on MNIST. Track train + val loss; observe overfitting; add weight decay.
- **Deliverable.** MNIST MLP at ≥95% test accuracy, with the run logged in `notebooks/03-mlp-mnist.ipynb`.
- **Reading.** 3Blue1Brown "Neural Networks" series; Goodfellow ch. 6; Karpathy lecture 4.
- **M-series notes.** MNIST trains in minutes on MPS.

### Week 3B — Training — [module ↗](modules/03b-training.md)

- **Question.** Why does the same network sometimes learn, stall, or explode?
- **Goal.** Make learning rate, optimizer choice, gradient clipping, schedules, and train/validation diagnostics feel like understandable tools rather than magic knobs.
- **Concepts.** Learning-rate scale; SGD vs AdamW; first and second optimizer moments; bias correction; decoupled weight decay; global gradient clipping; warmup/cosine decay; reading train/validation curves.
- **Build.** `g2c.training.AdamW`; `g2c.training.clip_grad_norm_`; `g2c.training.cosine_with_warmup`; trainer optimizer selection for later pretraining runs.
- **Exercises.**
  - Sweep learning rates on the same tiny MLP.
  - Compute the first AdamW update by hand.
  - Implement `AdamW.step`.
  - Compare SGD and AdamW on identical data/model/seed.
  - Demonstrate global-norm gradient clipping.
  - Plot warmup + cosine decay and diagnose train/validation curves.
- **Deliverable.** A notebook showing the optimizer and schedule comparisons, plus passing focused tests for AdamW, clipping, and cosine warmup.
- **Reading.** Kingma and Ba, "Adam"; Loshchilov and Hutter, "Decoupled Weight Decay Regularization"; Karpathy nanoGPT optimizer setup.
- **M-series notes.** CPU is enough; MPS is optional for larger sweeps.

---

## Phase II — Language

### Week 4 — Tokenization — [module ↗](modules/04-tokenizer.md)

- **Question.** How does text become model input?
- **Goal.** Implement byte-pair encoding from scratch.
- **Concepts.** Char vs word vs subword tokenization; vocab-size tradeoffs; the BPE merge algorithm; byte-level pre-tokenization (GPT-2 style); special tokens; lossless round-tripping.
- **Build.** `g2c/tokenizer/` — `BPETokenizer` with `train`, `encode`, `decode`. Train on a small corpus (TinyShakespeare or a Project Gutenberg slice).
- **Exercises.**
  - Train BPE at vocab sizes 256, 1k, 8k. Compare token counts on the same passage.
  - Implement byte-level pre-tokenization.
  - Verify lossless `encode → decode` on held-out strings.
- **Deliverable.** Tokenizer package + a notebook visualizing how the same passage tokenizes at different vocab sizes.
- **Reading.** Karpathy, "Let's build the GPT Tokenizer"; Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE); GPT-2 paper §2.2.
- **M-series notes.** CPU-bound; trains in seconds to minutes.

### Week 5 — Embeddings and positions — [module ↗](modules/05-embeddings.md)

- **Question.** How do discrete symbols become meaning-like vectors, and how does order get in?
- **Goal.** Implement learned token embeddings and several positional encoding schemes.
- **Concepts.** Embedding lookup as a learned table; learned vs fixed positional encodings; sinusoidal positions; rotary positional embeddings (RoPE); the "bag of tokens" failure without position info.
- **Build.** `g2c/embeddings/` — `TokenEmbedding`, `LearnedPositionalEmbedding`, `SinusoidalPositionalEmbedding`, and a `RotaryEmbedding` you'll wire in during week 7+.
- **Exercises.**
  - Train a tiny embedding model on word co-occurrence; visualize via t-SNE or UMAP.
  - Replicate `king - man + woman ≈ queen`-style structure on pretrained GloVe vectors; attempt the same on your tiny model.
  - Implement and unit-test sinusoidal vs learned encodings on identical inputs.
- **Deliverable.** Embedding visualizations + tested implementations of three positional schemes.
- **Reading.** Mikolov et al. "Efficient Estimation of Word Representations" (word2vec); Vaswani et al. §3.5; Su et al. "RoFormer" (RoPE — skim).
- **M-series notes.** Embedding tables of 8k × 256 are tiny.

### Week 6 — Next-token prediction — [module ↗](modules/06-language-models.md)

- **Question.** What is the actual training objective of a language model?
- **Goal.** Train the simplest possible language models on next-token prediction, before introducing transformers.
- **Concepts.** Autoregressive modeling, cross-entropy as next-token loss, perplexity, n-gram baselines, the leap from counts to neural LMs.
- **Build.** `g2c/lm/` — a counts-based bigram model, a neural bigram model, and a Bengio-style trigram MLP language model, plus a training loop.
- **Exercises.**
  - Train a counts bigram and a neural bigram on the same corpus; compare perplexity.
  - Train the trigram MLP LM on TinyShakespeare; sample.
  - Plot validation perplexity at training checkpoints.
- **Deliverable.** Three LMs with a perplexity comparison table and sampled text from each.
- **Reading.** Karpathy, "makemore" series parts 1–3; Bengio et al. 2003, "A Neural Probabilistic Language Model."
- **M-series notes.** All run in minutes.

---

## Phase III — Transformers

### Week 7 — Self-attention — [module ↗](modules/07-attention.md)

- **Question.** How do tokens communicate?
- **Goal.** Build single-head self-attention from scratch.
- **Concepts.** Queries, keys, values; attention scores; softmax over context; weighted value mixing; causal masking; the QKᵀ/√d scaling factor.
- **Build.** `g2c/attention/SelfAttention` — explicit Q, K, V projections, mask, softmax, output projection. No `torch.nn.MultiheadAttention`.
- **Exercises.**
  - Implement attention with explicit shape annotations; verify against a hand computation on a 3-token toy sequence.
  - Visualize the attention pattern on the canonical "the animal didn't cross the street because it was too tired/wide" example.
  - Confirm that without the causal mask, next-token loss collapses to ~0 (the model trivially cheats).
- **Deliverable.** Self-attention layer + an attention-map visualization notebook.
- **Reading.** Vaswani et al. §3.2; Karpathy, "Let's build GPT: from scratch" (attention section); Alammar, "The Illustrated Transformer."
- **M-series notes.** Tiny.

### Week 8 — Multi-head attention — [module ↗](modules/08-multi-head-attention.md)

- **Question.** Why split attention into multiple heads?
- **Goal.** Implement multi-head attention efficiently (via reshaping, not N independent linear layers).
- **Concepts.** Splitting embedding dim into heads; parallel attention computations; head concatenation + output projection; what different heads tend to specialize in (induction heads, syntactic heads, copy heads).
- **Build.** `g2c/attention/MultiHeadAttention`.
- **Exercises.**
  - Train a small LM with 1, 4, and 8 heads at fixed total dim; compare loss curves.
  - Visualize per-head attention patterns on a trained model; identify any interpretable heads.
- **Deliverable.** Multi-head attention + a per-head visualization notebook.
- **Reading.** Vaswani §3.2.2; Elhage et al., "A Mathematical Framework for Transformer Circuits" (introductory sections); Olsson et al., "In-context Learning and Induction Heads."
- **M-series notes.** Still tiny.

### Week 9 — The transformer block — [module ↗](modules/09-transformer-block.md)

- **Question.** How do we compose attention and per-token computation into a reusable unit?
- **Goal.** Assemble the canonical transformer block: pre-norm, MHA, residual, MLP, residual.
- **Concepts.** Layer normalization; pre-norm vs post-norm and why pre-norm trains more stably; residual connections as a "communication bus"; the position-wise FFN (with GELU); stacking blocks.
- **Build.** `g2c/transformer/Block`, `g2c/transformer/TransformerLM` — embed, stack N blocks, project to logits.
- **Exercises.**
  - Implement post-norm and pre-norm variants; train both at small scale; compare stability.
  - Strip residual connections; confirm training fails.
  - Strip layer norm; confirm training fails or destabilizes.
- **Deliverable.** `TransformerLM` ready to train at scale (week 10).
- **Reading.** Vaswani §3; Xiong et al., "On Layer Normalization in the Transformer Architecture"; Anthropic Transformer Circuits Thread (intro post).
- **M-series notes.** Still tiny.

### Week 9B — Pretraining — [module ↗](modules/09b-pretraining.md)

- **Question.** How do we turn a text corpus and a TransformerLM into supervised training data?
- **Goal.** Make corpus splitting, `(B, T)` language-model batches, multi-position targets, LM cross-entropy, and the `log(V)` baseline explicit before the first full pretraining run.
- **Concepts.** Contiguous token streams; train/validation split; shifted input/target windows; causal mask as the reason every position can be supervised; flattening `(B, T, V)` logits into ordinary cross-entropy; `log(V)` as a step-0 sanity baseline.
- **Build.** `g2c/pretraining/data.py` with `split_token_stream` and `get_lm_batch`; `g2c/pretraining/loss.py` with `lm_cross_entropy`.
- **Exercises.**
  - Shift toy token streams by hand.
  - Inspect sampled `(B, T)` batches from a known sequence.
  - Implement `lm_cross_entropy` and verify every position contributes.
  - Compare random model loss to `log(V)`.
- **Deliverable.** Passing `tests/test_pretraining_setup.py` and a notebook that traces one corpus through token IDs, train/val split, batch sampling, logits, and scalar loss.
- **Reading.** Karpathy nanoGPT data loader and loss computation; "Let's reproduce GPT-2" data-loading sections.
- **M-series notes.** CPU-light. This is setup, not a serious training run.

### Milestone Week 10 — TinyLLM — [module ↗](modules/10-tinyllm.md)

- **Question.** What changes when the transformer block becomes a trained language model?
- **Goal.** Pretrain a small transformer LM on a real corpus.
- **Concepts.** Trainer orchestration; the exact training-step order; validation history; learning-rate and gradient-norm logging; checkpoint/run artifacts; qualitative sampling as a noisy but useful inspection tool.
- **Build.** `g2c/pretraining/trainer.py` — `Trainer` with the full loop using Module 09B's data/loss helpers and Module 03B's optimizer controls.
- **Exercises.**
  - Implement `Trainer.train_step`.
  - Train a ~1M-param model on TinyShakespeare; sample text every N steps and watch it learn.
  - Plot train/validation loss, learning rate, and gradient norm.
  - Run one controlled scale-up and compare loss and samples.
- **Deliverable.** A trained tiny-GPT checkpoint that produces locally coherent text. Save weights + training logs.
- **Reading.** Karpathy nanoGPT repo; "Let's reproduce GPT-2 (124M)"; Kaplan et al., "Scaling Laws for Neural Language Models"; Hoffmann et al., "Chinchilla" (skim for intuition).
- **M-series notes.** 1M params trains in minutes; 10M in hours. 16GB unified memory is the floor; 32GB is more comfortable.

---

## Phase IV — Behavior

### Week 11 — Sampling and decoding — [module ↗](modules/11-sampling.md)

- **Question.** How does a probability distribution over tokens become actual text?
- **Goal.** Implement and compare decoding strategies on your trained model.
- **Concepts.** Greedy decoding; temperature; top-k; top-p (nucleus); repetition penalty; beam search (skim); stop tokens; the diversity-vs-quality tradeoff.
- **Build.** `g2c/sampling/` — a `generate()` function with all strategies controllable via parameters.
- **Exercises.**
  - Generate samples from your week-10 model at temperatures 0.1, 0.7, 1.0, 1.5; compare qualitatively.
  - Implement and compare top-k vs top-p; observe their effect on sample diversity.
  - Build an interactive CLI playground for sampling parameter exploration.
- **Deliverable.** `g2c/sampling/` + a side-by-side comparison notebook.
- **Reading.** Holtzman et al., "The Curious Case of Neural Text Degeneration" (top-p paper); Fan et al., "Hierarchical Neural Story Generation" (top-k).
- **M-series notes.** Inference-only — fast.

### Week 12 — Scaling experiments — [module ↗](modules/12-scaling.md)

- **Question.** What gets better with size, and how cleanly does it scale?
- **Goal.** Empirically measure how a few capabilities scale within MacBook range.
- **Concepts.** Parameter scaling; compute-optimal training (Chinchilla); emergent capabilities; the difference between smooth and threshold-looking improvements; the "evaluation matters" caveat.
- **Build.** No new package code — a TinyStories-only StoryLM comparison, with a small `StoryLM-1M` anchor trained here if missing.
- **Exercises.**
  - Compare `StoryLM-1M`, `StoryLM-5M`, and `StoryLM-30M` on the same TinyStories tokenizer/corpus family.
  - Plot validation loss/perplexity vs params and inspect the compute spent by each run.
  - Read samples and next-token distributions to identify qualitative differences across scale.
- **Deliverable.** A scaling-experiments notebook with plots and qualitative samples.
- **Reading.** Kaplan et al. 2020; Hoffmann et al. "Chinchilla"; Wei et al. "Emergent Abilities of Large Language Models" (and the BIG-bench debate paper that followed).
- **M-series notes.** Reuse `StoryLM-5M` and `StoryLM-30M` from Module 10; Module 12 should usually train only the small `StoryLM-1M` anchor.

### Week 13 — Instruction tuning (SFT) — [module ↗](modules/13-sft.md)

- **Question.** Why does the model follow requests rather than just continuing text?
- **Goal.** Convert your base LM into an instruction-following one via supervised fine-tuning.
- **Concepts.** Base model vs instruction model; SFT data format and chat templates; loss masking on user vs assistant turns; format collapse; the "data quality > data quantity" finding.
- **Build.** `g2c/sft/` — chat-template formatting + an SFT training script that reuses the week-10 Trainer.
- **Exercises.**
  - Curate or hand-author 50–500 instruction → response pairs (an Alpaca-style mini-dataset, or a topical one).
  - Fine-tune your week-10 / week-12 checkpoint on it.
  - Compare base vs SFT outputs on held-out instructions; note where the model follows and where it fails.
- **Deliverable.** SFT'd checkpoint + a base-vs-SFT comparison.
- **Reading.** Ouyang et al., "InstructGPT"; the Stanford Alpaca blog post; the LIMA paper ("Less Is More for Alignment").
- **M-series notes.** Tiny SFT trains in minutes. Output quality will be visibly toy — that's the point.

### Week 14 — Preference tuning (DPO) — [module ↗](modules/14-dpo.md)

- **Question.** Why is the model helpful, polite, or stylistically consistent?
- **Goal.** Improve the SFT model with a preference-based objective.
- **Concepts.** Reward modeling vs direct preference optimization; the RLHF pipeline conceptually; why DPO sidesteps explicit RL; KL regularization (the β coefficient); preference dataset construction.
- **Build.** `g2c/dpo/` — DPO loss + training loop. Reference model frozen; policy trained.
- **Exercises.**
  - Construct 50–200 (prompt, chosen, rejected) preference pairs by hand or via a stronger model's judgments.
  - Run DPO; compare DPO model vs SFT model on held-out prompts.
  - Sweep β; observe behavior at extremes (too low: model drifts; too high: no learning).
- **Deliverable.** DPO'd checkpoint + a comparison.
- **Reading.** Rafailov et al., "Direct Preference Optimization"; Ouyang et al. (RLHF, for context); Christiano et al., "Deep RL from Human Preferences" (skim).
- **M-series notes.** DPO is more memory-hungry than SFT (two model copies in memory). 32GB+ helps materially.

### Week 15 — Hallucination and evaluation — [module ↗](modules/15-evaluation.md)

- **Question.** Why does the model confidently invent things, and how do we measure it?
- **Goal.** Build a small eval harness; characterize your model's failure modes precisely.
- **Concepts.** Fluency vs truth as separate objectives; calibration; evaluation methods (perplexity, multiple choice, generation scoring, model-graded eval); hallucination categories; the limits of small-model reasoning.
- **Build.** `g2c/eval/` — a tiny eval harness with a few task types: factual QA, multiple choice, simple arithmetic, instruction following.
- **Exercises.**
  - Probe your week-13/14 model with adversarial, out-of-distribution, and impossible prompts. Catalog failure modes.
  - Implement a multiple-choice eval on a small subset of MMLU or a hand-built one.
  - Measure log-probability calibration on a small QA set.
- **Deliverable.** Eval harness + a written characterization of your model's typical failure modes.
- **Reading.** Ji et al., "Survey of Hallucination in Natural Language Generation"; HELM (Liang et al., skim); Zheng et al., "Judging LLM-as-a-Judge."
- **M-series notes.** Inference + analysis; comfortable.

---

## Phase V — Assistants

### Week 16 — Local pretrained models and inference — [module ↗](modules/16-inference.md)

- **Question.** How do we get from "I built it" to "I can use it"?
- **Goal.** Set up a stronger pretrained open model running locally and understand the inference stack.
- **Concepts.** Quantization (int8, int4, GGUF, MLX formats); KV cache; speculative decoding (skim); inference servers (Ollama, llama.cpp, MLX); why your from-scratch model is too small to be useful for the rest of the course.
- **Build.** `g2c/inference/` — a uniform Python interface that wraps either your trained model or a locally-served pretrained model (Ollama or MLX).
- **Exercises.**
  - Run a 7–8B quantized model (Llama 3 8B or Qwen 2.5 7B) via Ollama; measure tokens/sec.
  - Run the same model via MLX; compare throughput.
  - Run identical prompts across your tiny model and the pretrained one; characterize the gap.
- **Deliverable.** Unified inference interface + benchmark notebook.
- **Reading.** GGUF / llama.cpp docs; MLX examples repo; Ollama docs; Dettmers, "LLM.int8()" (skim).
- **M-series notes.** Unified-memory size really starts to matter. 16GB → comfortable up to 7B at 4-bit; 32GB+ → easier headroom; 64GB → comfortable with 13B-class.

### Week 17 — Retrieval-augmented generation — [module ↗](modules/17-rag.md)

- **Question.** How does the model use external knowledge it doesn't have memorized?
- **Goal.** Build a working RAG pipeline.
- **Concepts.** Document chunking; embedding models; vector stores; cosine similarity; hybrid retrieval (BM25 + dense); prompt construction with retrieved context; citation formatting; chunking and retrieval failure modes.
- **Build.** `g2c/rag/` — chunker, embedder (sentence-transformers or a local embedding model), vector store (SQLite + numpy, or Chroma / LanceDB), retriever, prompt assembler.
- **Exercises.**
  - Index a local corpus (a few books, your own notes, or the course's `docs/`).
  - Build a Q&A interface that generates with citations.
  - Probe failure modes: questions answerable only by retrieval, questions with no relevant doc, ambiguous questions.
- **Deliverable.** A RAG chatbot over your local docs.
- **Reading.** Lewis et al., "Retrieval-Augmented Generation"; Karpukhin et al., "Dense Passage Retrieval"; Anthropic / Llamaindex blog posts on retrieval evaluation.
- **M-series notes.** Comfortable. Embedding generation can be MLX-accelerated.

### Week 18 — Tool use — [module ↗](modules/18-tools.md)

- **Question.** How does the model act outside itself?
- **Goal.** Give the model a registry of structured tools and a dispatch loop.
- **Concepts.** Tool schemas (JSON); function-calling fine-tuning vs prompt-engineered tool use; the parse → execute → feedback loop; tool selection failures; error handling and retry.
- **Build.** `g2c/tools/` — tool registry, JSON-schema dispatch, real implementations of `calculator`, `read_file`, `web_search` (or a stub), `run_python`.
- **Exercises.**
  - Get the model to reliably call `calculator` for arithmetic that the model gets wrong on its own.
  - Wire `read_file` + `run_python` and have the model solve a small data task end-to-end.
  - Stress-test malformed outputs; implement recovery.
- **Deliverable.** Tool-using assistant + a few solved end-to-end tasks.
- **Reading.** Schick et al., "Toolformer"; Anthropic tool use docs; OpenAI function-calling docs; Patil et al., "Gorilla" (skim).
- **M-series notes.** All local; quality depends on the inference model's instruction-following.

### Week 19 — Agent loops — [module ↗](modules/19-agent.md)

- **Question.** How does the model pursue multi-step goals?
- **Goal.** Build a minimal agent loop with state, planning, tool use, and stopping criteria.
- **Concepts.** ReAct-style observe / think / act loops; scratchpad memory; planning vs reactive control; error recovery; preventing infinite loops; evaluating agentic systems.
- **Build.** `g2c/agent/` — agent runtime with conversation history, tool dispatch from week 18, a scratchpad, and max-steps + stopping rules.
- **Exercises.**
  - Have the agent research a topic across your RAG corpus and write a summary.
  - Have it complete a multi-file task: read X, transform Y, write Z.
  - Identify three failure modes and add a guard for each.
- **Deliverable.** An agent that solves at least one nontrivial multi-step task end-to-end.
- **Reading.** Yao et al., "ReAct"; Anthropic, "Building Effective Agents"; Park et al., "Generative Agents" (skim); SWE-bench / GAIA papers (skim).
- **M-series notes.** Loop length × per-step inference cost — keep contexts tight.

### Week 20 — Capstone: a tiny ChatGPT — [module ↗](modules/20-capstone.md)

- **Question.** Can I integrate everything?
- **Goal.** Ship a working assistant that you understand end-to-end.
- **Build.** `g2c/assistant/` — combines tokenizer, inference (your tiny model + a local pretrained model, swappable), sampling controls, RAG, tools, agent loop, conversation memory, and a basic chat UI (CLI or simple web).
- **Required features.**
  - Chat with conversation history
  - Configurable sampling controls
  - RAG over a chosen corpus
  - At least three working tools
  - Multi-step task completion
  - One eval suite that gates "does this still work" across iterations
- **Deliverable.** A reproducible assistant + a written post-mortem covering what each layer does, what its limitations are, and where the from-scratch model stops being viable. The post-mortem is the actual learning outcome.
- **Reading.** Re-read the brainstorm; revisit any modules whose lessons want reinforcement.
- **M-series notes.** Use the strongest local model you can comfortably run as the agent backbone.

---

## What you should be able to say at the end

> An LLM is a learned program for transforming token sequences. Training creates the program. Attention lets information move around inside the sequence. The transformer stack repeatedly refines representations. The output is a probability distribution over the next token. Everything else — chat behavior, reasoning, tool use, retrieval, agents — is built around that core. I know what every part is doing. The full system is huge, but it is not ontologically mysterious.

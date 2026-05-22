# From Gradients to ChatGPT

![Cover illustration: a circus-themed map of the course. Numbered booths under a big top represent each of the twenty modules — pretraining, tokenization, embeddings, gradient descent, self-attention, multi-head attention, the transformer (a central tower of "Add & Norm / Feed Forward / Multi-Head Attention" floors), sampling, SFT, DPO, RAG, tools, the agent, the eval inspector, and an "inference booth" robot at the bottom. Banners read "A tiny LLM stack from first principles," "Data. Compute. Curiosity. That's all you need," and "Built step by step under the big top."](docs/CourseCover.png)

A self-study course on the core building blocks of LLMs, modeled after *From NAND to Tetris*.

This repo contains both the instructional material and the student work product. The codebase grows layer-by-layer from scalar autodiff up through a working chat assistant — each module's deliverable becomes a building block for the next.

The hard constraint: **all tasks are runnable locally on an M-series MacBook**, without access to expensive cloud environments or GPUs.

## Quickstart 

This repo contains everything needed for the course exercise and student projects. Lecture notes are both in repo docs and hosted on the web at [the course site](https://mister-meeseeks.github.io/g2c/). 

```bash
git clone https://github.com/Mister-Meeseeks/g2c
cd g2c/
./setup.sh
```

## Contents

Start with the [syllabus](docs/syllabus.md) for the full 20-week arc

| #  | Module                                                          | Phase                |
| -- | --------------------------------------------------------------- | -------------------- |
| 00 | [Prerequisite review](docs/modules/00-prerequisite-review.md)   | 0 — Review           |
| 01 | [Scalar autodiff](docs/modules/01-autodiff.md)                  | I — Foundations      |
| 02 | [Tensors and matmul](docs/modules/02-tensors.md)                | I — Foundations      |
| 03 | [A first neural network](docs/modules/03-nn.md)                 | I — Foundations      |
| 03B | [Training](docs/modules/03b-training.md) | I — Foundations |
| 04 | [Tokenization](docs/modules/04-tokenizer.md)                    | II — Language        |
| 05 | [Embeddings and positions](docs/modules/05-embeddings.md)       | II — Language        |
| 06 | [Next-token prediction](docs/modules/06-language-models.md)     | II — Language        |
| 07 | [Self-attention](docs/modules/07-attention.md)                  | III — The transformer |
| 08 | [Multi-head attention](docs/modules/08-multi-head-attention.md) | III — The transformer |
| 09 | [The transformer block](docs/modules/09-transformer-block.md)   | III — The transformer |
| 09B | [Pretraining](docs/modules/09b-pretraining.md)              | III — The transformer |
| 10 | [Milestone: TinyLLM](docs/modules/10-tinyllm.md) | III — The transformer |
| 11 | [Sampling and decoding](docs/modules/11-sampling.md)            | IV — Behavior shaping |
| 12 | [Scaling experiments](docs/modules/12-scaling.md)               | IV — Behavior shaping |
| 13 | [Instruction tuning (SFT)](docs/modules/13-sft.md)              | IV — Behavior shaping |
| 14 | [Preference tuning (DPO)](docs/modules/14-dpo.md)               | IV — Behavior shaping |
| 15 | [Hallucination and evaluation](docs/modules/15-evaluation.md)   | IV — Behavior shaping |
| 16 | [Local pretrained models and inference](docs/modules/16-inference.md) | V — Assistant systems |
| 17 | [Retrieval-augmented generation](docs/modules/17-rag.md)         | V — Assistant systems |
| 18 | [Tool use](docs/modules/18-tools.md)                            | V — Assistant systems |
| 19 | [Agent loops](docs/modules/19-agent.md)                         | V — Assistant systems |
| 20 | [Capstone: a tiny ChatGPT](docs/modules/20-capstone.md)         | V — Assistant systems |

## Course structure

Each module is one "week" of effort in terms of the pacing one would find at a high level college course. Calendar pace is whatever it ends up being. Every module follows the same format:

* **Lecture notes**. Short and focused introduction to the concepts. Includes additional reading for those who want to go deeper. 

*  **`g2c/` package**. After each module, students implement a new python package in `g2c/`. Boilerplate and plumbing are pre-built. Test with the included pytest suite. 

* **Exercise notebook**. Every module has a Jupyter notebook with an exercise set based on the module's lesson. The notebooks rely on the `g2c/` packages from that module.

The focus of the course is on *building*. The lecture notes are generally short and oriented around the concepts the student needs to know to finish the module deliverables. Optional additional reading is included in each module for students who want to go deeper on any topic.

The `g2c/` package grows through each week of the course. Every module's package is consumed by later modules in some way. To complete the course, students must implement all the module packages, until it comes together in the capstone project. Each module has a test suite to help students students validate their implementation. Iterate until tests come back green. 

The course is agent friendly. Students should use their favorite coding agent (like Claude or Codex) to answer questions, give hints, grade problems, etc. The `AGENTS.md`/`CLAUDE.md` is setup so coding agents launched in the repo will naturally act like teacher assistants. 

## System requirements

The course is designed for M-series Macs, but not every student needs the same download size, model size, or training time. The track system is artifact-based: prepare as much data as you want, and notebooks should load the strongest local artifact they can find.

The notebook launcher and course setup scripts will use sensible but relatively small defaults for models and datasets. The exercises and projects are designed to work well with the default options. However students are encouraged to, and the course supports, exploring different sized models and datasets.

Minimum system recommendations:
* M2 Apple chip or better
* 16GB+ memory
* 20GB+ free storage

Minimum system requirements (will likely have to use smaller models than defaults):
* Any M- or A- series Apple chip
* 8GB memory
* 5GB free stroage

For sizing models and datasets or going outside the standard defaults see [Course Tracks and Artifacts](docs/tracks.md) 

## Repository layout

```
docs/
  syllabus.md           # the detailed syllabus
  modules/
    00-prerequisite-review.md
    01-autodiff.md      # lesson, motivation, exercises, deliverable spec
    01-autodiff/        # assets for module 01 (diagrams, supplementary files)
    02-tensors.md
    02-tensors/
    ...
g2c/                    # the work-product Python package 
  autodiff/             # module 1
  tensors/              # module 2
  nn/                   # module 3
  ...
  solutions/            # mirror tree with worked implementations
  notebook_extras/      # non-pedagogical notebook helpers (matplotlib, etc.)
notebooks/
  clean/                # canonical pristine notebooks
  solutions/            # working notebook copies and solved notebooks
artifacts/              # saved models and durable outputs
data/                   # corpora, datasets, caches
scripts/                # utility scripts
tests/                  # tests across modules
```

The split is intentional: `docs/` is what a student reads, `g2c/` is what the student builds. They evolve together.

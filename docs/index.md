# From Gradients to ChatGPT

![Cover illustration: a circus-themed map of the course. Numbered booths under a big top represent each of the twenty modules — pretraining, tokenization, embeddings, gradient descent, self-attention, multi-head attention, the transformer, sampling, SFT, DPO, RAG, tools, the agent, the eval inspector, and an "inference booth" robot at the bottom.](CourseCover.png)

A 20-week self-study course building a tiny LLM stack from scalar autograd up through a working chat assistant. Modeled after *From NAND to Tetris*: the codebase grows layer-by-layer, and every block of the stack is something you write yourself.

The hard constraint: **everything runs on an M-series MacBook** — no cloud GPUs, no paid compute.

## Get started

- Read the [syllabus](syllabus.md) for the full 20-week arc.
- Pick a [track](tracks.md) (Tiny / Standard / Full) sized to your machine.
- Start with [Module 0: Prerequisite review](modules/00-prerequisite-review.md), or jump straight into [Module 1: Scalar autodiff](modules/01-autodiff.md).

## How the course works

Each module has a lesson page (here on the site) and a deliverable that lives in the [course repo](https://github.com/your-org/g2c). To do the exercises you clone the repo, run `setup.sh`, and work through the notebook for each module. The lesson pages are the readable front door; the repo is where the code lives.

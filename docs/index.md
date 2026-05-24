# From Gradients to ChatGPT

![Cover illustration: a circus-themed map of the course. Numbered booths under a big top represent each of the twenty modules — pretraining, tokenization, embeddings, gradient descent, self-attention, multi-head attention, the transformer, sampling, SFT, DPO, RAG, tools, the agent, the eval inspector, and an "inference booth" robot at the bottom.](CourseCover.png)

A 20-week self-study course building a tiny LLM stack from scalar autograd up through working chat assistant. Modeled after *From NAND to Tetris*: the codebase grows layer-by-layer, and every block of the stack is something you write yourself.

The hard constraint: **everything runs on an M-series MacBook** — no cloud GPUs, no paid compute.

## Why this course

Millions interact with LLMs on a daily basis. But few bother to understand how these systems actually work. How does an LLM understand language? What is a "model" and how does it learn? How does a chat assistant "know" how to answer a question? How does matrix multipilcation produce intelligent behavior? In this course we answer these questions from first principles.

Our goal is to go *below the API*. To understand every part of the LLM stack, end to end. You start with basic autodiff, and over twenty modules grow it: tensors, a neural net, a tokenizer, embeddings, attention, the transformer, pretraining, sampling, SFT, DPO, evaluation, RAG, tool use, an agent loop. Until the final module is a chat assistant running on a model you trained yourself, with tools and retrieval. All on your laptop. All built by you. 

The scaffolding is real but small: a tokenizer that takes ten minutes to train, a transformer with a few million parameters, a corpus that fits in RAM. Tiny is deliberate. Once you've built every layer once at toy scale, the production-scale versions stop being magic.

## What's in it

Twenty modules plus a fast prerequisite review, organized in five phases. Each module is roughly one week of effort at the level of a rigorous elite-college course.

| #   | Module                                                              | Phase                  |
| --- | ------------------------------------------------------------------- | ---------------------- |
| 00  | [Prerequisite review](modules/00-prerequisite-review.md)            | 0 — Review             |
| 01  | [Scalar autodiff](modules/01-autodiff.md)                           | I — Foundations        |
| 02  | [Tensors and matmul](modules/02-tensors.md)                         | I — Foundations        |
| 03  | [A first neural network](modules/03-nn.md)                          | I — Foundations        |
| 03B | [Training](modules/03b-training.md)                                 | I — Foundations        |
| 04  | [Tokenization](modules/04-tokenizer.md)                             | II — Language          |
| 05  | [Embeddings and positions](modules/05-embeddings.md)                | II — Language          |
| 06  | [Next-token prediction](modules/06-language-models.md)              | II — Language          |
| 07  | [Self-attention](modules/07-attention.md)                           | III — The transformer  |
| 08  | [Multi-head attention](modules/08-multi-head-attention.md)          | III — The transformer  |
| 09  | [The transformer block](modules/09-transformer-block.md)            | III — The transformer  |
| 09B | [Pretraining](modules/09b-pretraining.md)                           | III — The transformer  |
| 10  | [Milestone: TinyLLM](modules/10-tinyllm.md)                         | III — The transformer  |
| 11  | [Sampling and decoding](modules/11-sampling.md)                     | IV — Behavior shaping  |
| 12  | [Scaling experiments](modules/12-scaling.md)                        | IV — Behavior shaping  |
| 13  | [Instruction tuning (SFT)](modules/13-sft.md)                       | IV — Behavior shaping  |
| 14  | [Preference tuning (DPO)](modules/14-dpo.md)                        | IV — Behavior shaping  |
| 15  | [Hallucination and evaluation](modules/15-evaluation.md)            | IV — Behavior shaping  |
| 16  | [Local pretrained models and inference](modules/16-inference.md)    | V — Assistant systems  |
| 17  | [Retrieval-augmented generation](modules/17-rag.md)                 | V — Assistant systems  |
| 18  | [Tool use](modules/18-tools.md)                                     | V — Assistant systems  |
| 19  | [Agent loops](modules/19-agent.md)                                  | V — Assistant systems  |
| 20  | [Capstone: a tiny ChatGPT](modules/20-capstone.md)                  | V — Assistant systems  |

The [syllabus](syllabus.md) lays out each phase in detail and gives the full motivation for the ordering.

## Who it's for

You'll get the most out of this if you're comfortable with Python, undergraduate calculus (chain rule, gradients), and basic linear algebra. You don't need prior deep learning experience — Module 0 covers the prerequisites and Modules 1–3 build the math substrate from scratch. You should be willing to read a paper now and then, and be willing to debug your own code without a framework hiding the failure mode.

If you've watched Karpathy's videos and wished for a structured curriculum with exercises, deliverables, and tests, this is that.

## How the course works

Each module in the course has a lesson page and a set of deliverables combining a coding project and problem sets. The lesson pages are hosted on this site, as well as available as markdown in the [course repo](https://github.com/Mister-Meeseeks/g2c). 

Each week, students will read the module lesson page. They'll then implement a new package covering that weeks topic inside the `g2c/` python pacakge. Finally they'll complete a set of student exercises in a Jupyter notebook using the code they wrote that week. The lesson pages are the readable front door; the repo is where the code lives.

## Get started

- Read the [syllabus](syllabus.md) for the full 20-week arc.
- Clone the [repository](https://github.com/Mister-Meeseeks/g2c) and follow the README quickstart to setup on your machine
- Start with [Module 0: Prerequisite review](modules/00-prerequisite-review.md), or jump straight into [Module 1: Scalar autodiff](modules/01-autodiff.md).

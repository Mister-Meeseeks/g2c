# From Gradients to ChatGPT

![Cover illustration: a circus-themed map of the course. Numbered booths under a big top represent each of the twenty modules — pretraining, tokenization, embeddings, gradient descent, self-attention, multi-head attention, the transformer (a central tower of "Add & Norm / Feed Forward / Multi-Head Attention" floors), sampling, SFT, DPO, RAG, tools, the agent, the eval inspector, and an "inference booth" robot at the bottom. Banners read "A tiny LLM stack from first principles," "Data. Compute. Curiosity. That's all you need," and "Built step by step under the big top."](docs/CourseCover.png)

A self-study course on the core building blocks of LLMs, modeled after *From NAND to Tetris*.

This repo contains both the instructional material and the student work product. The codebase grows layer-by-layer from scalar autodiff up through a working chat assistant — each module's deliverable becomes a building block for the next.

The hard constraint: **all tasks are runnable locally on an M-series MacBook**, without access to expensive cloud environments or GPUs.

## Quickstart 

This repo contains everything needed for the course exercises and student projects. Lecture notes are both in repo docs and hosted on the web at [the course site](https://mister-meeseeks.github.io/g2c/). 

Start by [forking this repo](https://github.com/Mister-Meeseeks/g2c/fork). The course is months of your own work, and a fork gives that work a home on GitHub — backup, history, and one-click course updates. (A plain `git clone` of this repo works too; you just won't have anywhere to push.)

```bash
git clone https://github.com/<your-username>/g2c
cd g2c/
./setup.sh
```

**Prerequisites.** `setup.sh` checks for (but does not install) three system tools:

* **Python 3.11+** — `brew install python@3.11`
* **[uv](https://docs.astral.sh/uv/)** — `brew install uv`
* **[Ollama](https://ollama.com/download)** — `brew install ollama`. Only needed from Module 16 on; fine to defer.

Everything else (PyTorch, Jupyter, the `g2c` package itself) is installed into a project-local `.venv` by the script.

Then open the first module's notebook:

```bash
./notebook.sh 00
```

## Contents

Start with the [syllabus](docs/syllabus.md) for the full arc across both parts.

### Part I — From gradients to a language model

Build the model itself. You end with a language model whose every layer you wrote, generating readable text on your laptop.

| #  | Module                                                          | Group                |
| -- | --------------------------------------------------------------- | -------------------- |
| 00 | [Prerequisite review](docs/modules/00-prerequisite-review.md)   | Prerequisite review  |
| 01 | [Scalar autodiff](docs/modules/01-autodiff.md)                  | Foundations          |
| 02 | [Tensors and matmul](docs/modules/02-tensors.md)                | Foundations          |
| 03 | [A first neural network](docs/modules/03-nn.md)                 | Foundations          |
| 03B | [Training](docs/modules/03b-training.md)                       | Foundations          |
| 04 | [Tokenization](docs/modules/04-tokenizer.md)                    | Language             |
| 05 | [Embeddings and positions](docs/modules/05-embeddings.md)       | Language             |
| 06 | [Next-token prediction](docs/modules/06-language-models.md)     | Language             |
| 07 | [Self-attention](docs/modules/07-attention.md)                  | The transformer      |
| 08 | [Multi-head attention](docs/modules/08-multi-head-attention.md) | The transformer      |
| 09 | [The transformer block](docs/modules/09-transformer-block.md)   | The transformer      |
| 09B | [Pretraining](docs/modules/09b-pretraining.md)                 | The transformer      |
| 10 | [Milestone: TinyLLM](docs/modules/10-tinyllm.md)                | The transformer      |
| 11 | [Sampling and decoding](docs/modules/11-sampling.md)            | The transformer      |

**Part I ends here, and finishing it is a real accomplishment.** You will have built a working language model from scalar derivatives up, with no black boxes in the path.

### Part II — From a language model to ChatGPT

Build the system around a model. A different subject from Part I, not a harder one: less derivation, more systems engineering.

| #  | Module                                                          | Group                |
| -- | --------------------------------------------------------------- | -------------------- |
| 12 | [Scaling experiments](docs/modules/12-scaling.md)               | Behavior shaping     |
| 13 | [Instruction tuning (SFT)](docs/modules/13-sft.md)              | Behavior shaping     |
| 13B | [LoRA](docs/modules/13b-lora.md)                                | Behavior shaping     |
| 14 | [Preference tuning (DPO)](docs/modules/14-dpo.md)               | Behavior shaping     |
| 15 | [Hallucination and evaluation](docs/modules/15-evaluation.md)   | Behavior shaping     |
| 16 | [Local pretrained models and inference](docs/modules/16-inference.md) | Assistant systems |
| 16B | [Synthetic data](docs/modules/16b-synthetic-data.md)            | Assistant systems   |
| 17 | [Retrieval-augmented generation](docs/modules/17-rag.md)         | Assistant systems   |
| 18 | [Tool use](docs/modules/18-tools.md)                            | Assistant systems    |
| 19 | [Agent loops](docs/modules/19-agent.md)                         | Assistant systems    |
| 20 | [Capstone: a tiny ChatGPT](docs/modules/20-capstone.md)         | Assistant systems    |

Part II can also be entered directly if you already know the fundamentals and want the systems material — see [Module 12](docs/modules/12-scaling.md).

## Course structure

Each module is roughly one week of effort at the level of a rigorous elite-college course. Calendar pace is whatever it ends up being. Every module follows the same format:

* **Lecture notes**. A short, focused introduction to the concepts. Includes additional reading for those who want to go deeper. 

*  **`g2c/` package**. After each module, students implement a new Python package in `g2c/`. Boilerplate and plumbing are pre-built. Test with the included pytest suite. 

* **Exercise notebook**. Every module has a Jupyter notebook with an exercise set based on the module's lesson. The notebooks rely on the `g2c/` packages from that module.

The focus of the course is on *building*. The lecture notes are generally short and oriented around the concepts the student needs to know to finish the module deliverables. Optional additional reading is included in each module for students who want to go deeper on any topic.

The `g2c/` package grows through each week of the course. Every module's package is consumed by later modules in some way. Part I culminates in a language model you wrote every layer of; Part II carries the same codebase through to the capstone assistant. Each module has a test suite to help students validate their implementation. Iterate until tests come back green. 

The course is agent friendly. Students should use their favorite coding agent (like Claude or Codex) to answer questions, give hints, grade problems, etc. The `AGENTS.md`/`CLAUDE.md` is set up so coding agents launched in the repo will naturally act like teacher assistants. 

### The answer key is public, on purpose

The worked implementations (`g2c/solutions/`) and the grading rubrics (`docs/rubrics/`, a growing set) ship in this repo, in the open. That's deliberate. This is a self-study course: there is no transcript, no credential, and no proctor — a student who pastes in the answers is only cheating themself out of the thing they came for. And the open answer key is what makes the course work: it lets maintainers and coding agents verify every module end-to-end, it powers `./notebook.sh NN --solutions`, which runs a notebook against the reference implementations, and it gives a stuck student a way to get unstuck without abandoning the course. Treat it like the answers in the back of a math textbook: reach for it after you've fought with the problem, not before.

### Getting unstuck without starting over

Every module imports the one before it, so a bug in Module 07 can surface as garbage output in Module 12. When that happens you shouldn't have to choose between debugging backwards through five modules and throwing away everything you wrote. Hand back only the modules you're not working on:

```bash
./notebook.sh 12 --solutions=01-07   # reference code for 01-07, your own from 08 on
```

The same selector works for tests and for Python:

```bash
G2C_APPLY_SOLUTIONS=01-07 pytest     # a range
G2C_APPLY_SOLUTIONS=07,09b pytest    # specific modules
G2C_APPLY_SOLUTIONS=attention pytest # a whole topic
G2C_APPLY_SOLUTIONS=1 pytest         # everything
```

This is the intended way to keep moving when one module has you stuck. Your work stands everywhere you didn't name.

## Keeping your copy up to date

The course keeps improving after you start it — lesson fixes, better tests, corrected reference implementations. Updates are safe to take mid-course, no matter how much you've written:

```bash
# In a fork: click "Sync fork" on your fork's GitHub page, then
git pull
# In a plain clone: just git pull
```

That safety is a deliberate contract, not luck: **course updates never modify the files you edit.** Your work lives in the `g2c/` scaffold files and your working notebooks (which aren't tracked at all); post-release fixes land in docs, tests, rubrics, and the solutions mirror. CI enforces the contract (`scripts/check_scaffold_freeze.sh`), and in the rare case a scaffold file itself must change, the exception is recorded in `.github/scaffold-freeze` with a note on how to patch an already-edited copy — so if a pull ever does conflict, that file tells you what happened.

One default worth knowing: your written answers in `notebooks/solutions/` are gitignored, so that this repo never ships a solved notebook. In your own fork that protection serves no one — if you want your notebook answers versioned and backed up with the rest of your work, delete `notebooks/solutions/.gitignore` and commit them like anything else.

## System requirements

The course is designed for M-series Macs, but not every student needs the same download size, model size, or training time. The track system is artifact-based: prepare as much data as you want, and notebooks should load the strongest local artifact they can find.

The notebook launcher and course setup scripts will use sensible but relatively small defaults for models and datasets. The exercises and projects are designed to work well with the default options. However, students are encouraged to explore different-sized models and datasets, and the course supports it.

Minimum system recommendations:
* M2 Apple chip or better
* 16GB+ memory
* 20GB+ free storage

Minimum system requirements (will likely have to use smaller models than defaults):
* Any M- or A- series Apple chip
* 8GB memory
* 5GB free storage

**Not on an M-series Mac?** The course is developed, tested, and supported only on Apple Silicon. The code is ordinary PyTorch, so Part I will *probably* run on Linux — CUDA or even CPU — with small device-string changes, but no lesson, test, or wall-clock claim has been validated there, and the M-series notes assume unified memory. Intel Macs fall back to CPU-only PyTorch, which turns the training modules from minutes into hours. Windows is unsupported outright (the course scripts are bash; WSL2 puts you in the untested-Linux camp). You're welcome to adapt the course to other hardware — genuinely — but you'll be off the map, and problems that only reproduce off Apple Silicon aren't something we can debug for you.

For sizing models and datasets, or going outside the standard defaults, see [Course Tracks and Artifacts](docs/tracks.md). 

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

## Feedback

If the course loses you somewhere, please [say where](https://github.com/Mister-Meeseeks/g2c/issues/new?template=01-stuck.yml). You don't need to have solved it first, and "I lost the thread somewhere in the backward pass" is a perfectly good report.

Nothing in this repo phones home — it all runs locally, and it always will. The consequence is that a module which quietly loses half the people who reach it looks exactly like a module nobody struggled with. Stuck reports are the only way that difference becomes visible.

For questions, "is this normal?" checks, and study-group organizing, use [GitHub Discussions](https://github.com/Mister-Meeseeks/g2c/discussions). Bug reports, corrections, and pull requests are all welcome — see [CONTRIBUTING.md](CONTRIBUTING.md), which also sets honest expectations about response times. Wondering why a topic isn't covered? The [roadmap](docs/roadmap.md) lists what's planned and what's deliberately out.

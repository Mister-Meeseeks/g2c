# Course Tracks and Artifacts

The course has one conceptual path, but several local-compute tracks. A track is
not a permanent identity. It is just how much data and how many reusable
artifacts you choose to prepare on this machine.

The repo should mostly manage this for you:

```bash
.venv/bin/python scripts/artifact_status.py
.venv/bin/python scripts/artifact_status.py --module 10
```

Notebooks and later modules should load the strongest compatible local artifact
they can find, print what they selected, and fall back gracefully when a larger
artifact is missing.

## What You Decide

The conceptual path is identical across tracks. Four independent choices set up
your local experience:

| Decision | Command | Tiny | Standard | Stretch |
|---|---|---|---|---|
| Dataset footprint | `./datasets.sh` | `--tiny` | `--small` | (no flag) |
| Module 10 model | Module 10 notebook | `ShakespeareLM-1M`, `StoryLM-5M` | `StoryLM-30M`, `TinyLLM-30M` | `TinyLLM-100M` |
| BaseLM (Modules 13-16) | `./baselm.sh --model-id` | small HF base | small HF base | small HF base |
| ProdLM (Modules 16-20) | `./prodlm.sh --model-id` | 1.5B-3B instruct | 7B-8B instruct | 14B-class instruct |

BaseLM is a per-need fallback rather than a hardware tier — pick any small
open-weight base model that runs comfortably on your machine.

## Tracks

| Track | Command | Intended use | Downloads | Free disk target | Heavy one-time work |
|---|---|---|---:|---:|---|
| Tiny | `./datasets.sh --tiny` | Fastest path, weaker hardware, quick Module 10 artifact | about 100MB plus artifacts | 5-10GB | TinyStories sample, StoryTokenizer, tokenized sample |
| Standard | `./datasets.sh --small` | Recommended local course experience | several GB | 20-40GB | GloVe, full TinyStories, small G2C corpus, tokenizers, tokenized corpora |
| Full | `./datasets.sh` | Stretch path for stronger machines and longer runs | 10GB+ | 40-80GB | full G2C corpus, larger tokenized corpus artifact |

`./datasets.sh all` is the same as `./datasets.sh`.

Individual targets are still available:

```bash
./datasets.sh glove
./datasets.sh tinystories
./datasets.sh g2c-corpus-small
./datasets.sh g2c-corpus-full
```

BaseLM and ProdLM are prepared separately because they are model artifacts, not
dataset tracks:

```bash
./baselm.sh --model-id <hf-model-id>
./prodlm.sh --model-id <ollama-tag>
```

Both scripts have sensible defaults if you omit `--model-id`. Pick any small
HF causal LM for BaseLM and any local instruct model for ProdLM that fits your
machine (see the design doc for hardware-tier guidance).

All dataset commands are intended to be idempotent. Rerunning should skip
completed downloads and completed artifacts.

## What Each Track Builds

| Track | Raw data | Tokenizers | Tokenized corpora |
|---|---|---|---|
| Tiny | TinyStories 100MB sample | `StoryTokenizer` | `StoryLM-tinystories-100MB-v4096` |
| Standard | GloVe, full TinyStories, small G2C Corpus v1 | `StoryTokenizer`, `G2CTokenizer` | `StoryLM-tinystories-full-v4096`, `TinyLLM-g2c-small-v8192` |
| Full | GloVe, full TinyStories, full G2C Corpus v1 | `StoryTokenizer`, `G2CTokenizer` | `StoryLM-tinystories-full-v4096`, `TinyLLM-g2c-full-v8192` |

The normal `./setup.sh` path stays small. It prepares the Python environment,
TinyShakespeare, and the `ShakespeareTokenizer` artifact used by the first
Module 10 smoke run.

A note on the `vNNNN` suffix: a tokenizer artifact stores its trained vocab as
the *maximum* usable size. Downstream artifacts (tokenized corpora, models)
record the slice they actually use through the `vNNNN` suffix and the manifest
field `vocab_size` / `effective_vocab_size`. A BPE tokenizer truncated to its
first N merges is still a valid tokenizer, so one trained `StoryTokenizer` can
back multiple smaller-vocab corpora and models. If `StoryTokenizer` reports a
trained vocab of 8192 but `StoryLM-tinystories-full-v4096` and `StoryLM` both
pin `vocab_size: 4096`, that is the system working as intended.

## Artifact Roles

| Role | Meaning |
|---|---|
| `ShakespeareLM` | Tiny baseline model from Module 10. Useful for proving the loop works, not for quality. |
| `StoryLM` | TinyStories-trained model. More coherent stories, still not a general assistant. |
| `TinyLLM` | Broader G2C-corpus model trained from scratch. Best self-trained candidate for assistant-shaped experiments. |
| `BaseLM` | Small external pretrained base model for Modules 13-16 when self-trained models are too weak. The role is model-agnostic -- run `./baselm.sh --model-id <hf-model-id>` to bind it to any small HF causal LM that fits your machine. |
| `ProdLM` | Local pretrained instruct model for Modules 16-20. Usually served through Ollama, llama.cpp, or another local runtime. |
| `<base>-SFT` / `<base>-DPO` | Derived artifacts produced by Modules 13 and 14. The base is whichever of `StoryLM-<N>`, `TinyLLM-<N>`, or `BaseLM` was fine-tuned; the suffix records the last training pass. DPO artifacts are typically layered on top of the corresponding `-SFT` artifact, which is recorded in the manifest's `base_artifact` field. |

### Naming and aliasing

Self-trained artifacts always include an explicit parameter count in their
on-disk name: `ShakespeareLM-1M`, `StoryLM-5M`, `StoryLM-30M`, `TinyLLM-30M`,
`TinyLLM-30M-SFT`. The number is approximate parameter count rounded to ~2
significant figures, following industry convention (Qwen-9B, Llama-3-8B).
`BaseLM` and `ProdLM` carry no size suffix because we don't train them.
`ProdLM` also takes no stage suffix because we don't post-train it.

A name with the size omitted (`StoryLM`, `StoryLM-SFT`, `TinyLLM-DPO`) is an
**alias** that resolves at load time to the largest available artifact in that
family at that stage. Aliases are never written to disk -- only canonical full
names are saved.

Modules 10-15 should support both self-trained artifacts and the BaseLM
fallback. Modules 16-18 should assume ProdLM for the main assistant-system
experience, while keeping TinyLLM or StoryLM useful for comparison when present.
Modules 19-20 are ProdLM-only; tiny self-trained models cannot reliably drive
ReAct or multi-turn assistant loops, so they are not currently exposed there.

## Time Costs

Exact times vary with Mac generation, RAM, network, and whether MPS is available.
Use these as planning ranges, not promises.

| Step | Typical track | Time shape | Cached output |
|---|---|---|---|
| `./setup.sh` | all | minutes | `.venv/`, TinyShakespeare, `ShakespeareTokenizer` |
| GloVe download/extract | Standard/Full | minutes, network-bound | `data/embeddings/glove.6B.50d.txt` |
| TinyStories sample | Tiny | minutes, network-bound | compressed 100MB shards |
| Full TinyStories | Standard/Full | several minutes to tens of minutes | compressed 100MB shards |
| G2C corpus small | Standard | tens of minutes-ish | `data/datasets/g2c-corpus-v1-small/` |
| G2C corpus full | Full | long: network plus processing | `data/datasets/g2c-corpus-v1/` |
| StoryTokenizer | Tiny/Standard/Full | minutes | `artifacts/tokenizers/StoryTokenizer/` |
| G2CTokenizer | Standard/Full | minutes to tens of minutes | `artifacts/tokenizers/G2CTokenizer/` |
| Tokenized TinyStories | Tiny/Standard/Full | minutes | `data/cache/token-corpus/StoryLM-*` |
| Tokenized G2C corpus | Standard/Full | minutes to tens of minutes | `data/cache/token-corpus/TinyLLM-*` |
| StoryLM 5M/30M training | Module 10 | minutes to about hour-class | checkpoints and model artifacts |
| TinyLLM 30M-100M training | Module 10/12 | multi-hour or overnight | checkpoints and model artifacts |
| BaseLM fetch | Modules 13-16 | model-size and network dependent | `./baselm.sh`, HF cache under `data/cache/baselm/` |
| ProdLM fetch | Modules 16-20 | model-size and network dependent | `./prodlm.sh`, external Ollama model cache |

Downloads and tokenized corpora are one-time setup costs. Training runs are the
recurring cost. Long training sections should checkpoint so you can interrupt,
inspect, sample, and continue.

## Module Expectations

| Modules | Requirement pattern |
|---|---|
| 01-03B | Normal setup only. |
| 04 | Conceptual tokenizer exercises run small; artifact mini-milestone uses the dataset track you prepared. |
| 05 | Optional GloVe download for pretrained vector exercises. |
| 06-09B | Mostly lightweight. Use TinyShakespeare and small synthetic corpora. |
| 10 | Main artifact fork: ShakespeareLM baseline, StoryLM, and optional TinyLLM. |
| 11 | Uses the strongest saved Module 10 model it can find. |
| 12 | Scaling lab. Extends Module 10 and can stay small or go stretch. |
| 13-15 | Prefer a capable self-trained TinyLLM when available; otherwise run `./baselm.sh` and use BaseLM. |
| 16-18 | Use ProdLM for the main assistant path. The strongest self-trained artifact can also be loaded for comparison. |
| 19-20 | ProdLM only. The deterministic exercises use a `FakeBackend`; live cells require ProdLM. Self-trained models are not currently exposed here because they do not reliably follow ReAct or multi-turn assistant formats. |

## Working Rule

Do not worry about staying inside one track forever. Start small, then add more
artifacts when you want better outputs:

```bash
./datasets.sh --tiny
.venv/bin/python scripts/artifact_status.py --module 10

# Later, if you want stronger artifacts:
./datasets.sh --small
.venv/bin/python scripts/artifact_status.py --module 10
```

The course should follow the artifacts you have, not make you manually track
which path you chose.

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

## Artifact Roles

| Role | Meaning |
|---|---|
| `ShakespeareLM` | Tiny baseline model from Module 10. Useful for proving the loop works, not for quality. |
| `StoryLM` | TinyStories-trained model. More coherent stories, still not a general assistant. |
| `TinyLLM` | Broader G2C-corpus model trained from scratch. Best self-trained candidate for assistant-shaped experiments. |
| `BaseLM` | Small external pretrained base model for Modules 13-15 when self-trained models are too weak. Qwen-0.6B is the current reference candidate, but the role is not tied to Qwen. |
| `ProdLM` | Local pretrained instruct model for Modules 16-20. Usually served through Ollama, llama.cpp, or another local runtime. |

Modules 10-15 should support both self-trained artifacts and the BaseLM
fallback. Modules 16-20 should assume ProdLM for the main assistant-system
experience, while keeping TinyLLM or StoryLM useful for comparison when present.

## Time Costs

Exact times vary with Mac generation, RAM, network, and whether MPS is available.
Use these as planning ranges, not promises.

| Step | Typical track | Time shape | Cached output |
|---|---|---|---|
| `./setup.sh` | all | minutes | `.venv/`, TinyShakespeare, `ShakespeareTokenizer` |
| GloVe download/extract | Standard/Full | minutes, network-bound | `data/glove.6B.50d.txt` |
| TinyStories sample | Tiny | minutes, network-bound | compressed 100MB shards |
| Full TinyStories | Standard/Full | several minutes to tens of minutes | compressed 100MB shards |
| G2C corpus small | Standard | tens of minutes-ish | `data/g2c-corpus-v1-small/` |
| G2C corpus full | Full | long: network plus processing | `data/g2c-corpus-v1/` |
| StoryTokenizer | Tiny/Standard/Full | minutes | `artifacts/tokenizers/StoryTokenizer/` |
| G2CTokenizer | Standard/Full | minutes to tens of minutes | `artifacts/tokenizers/G2CTokenizer/` |
| Tokenized TinyStories | Tiny/Standard/Full | minutes | `artifacts/tokenized-corpora/StoryLM-*` |
| Tokenized G2C corpus | Standard/Full | minutes to tens of minutes | `artifacts/tokenized-corpora/TinyLLM-*` |
| StoryLM 5M/30M training | Module 10 | minutes to about hour-class | checkpoints and model artifacts |
| TinyLLM 30M-100M training | Module 10/12 | multi-hour or overnight | checkpoints and model artifacts |
| BaseLM fetch | Modules 13-15 | model-size and network dependent | external model cache |
| ProdLM fetch | Modules 16-20 | model-size and network dependent | external model cache |

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
| 13-15 | Prefer a capable self-trained TinyLLM when available; otherwise use BaseLM. |
| 16-20 | Use ProdLM for the main assistant path. Self-trained models are comparison backends. |

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

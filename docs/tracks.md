# Course Tracks and Artifacts

This course is designed to be flexible across a wide range of system capabilities. The one hard constraint is that most of the exercises and libraries will only work on an M-Series Mac laptop or desktop. 

Everything runs on one conceptual path, but there are several local-compute tracks, depending on your specific machine and its CPU/GPU, memory and storage. Students are encouraged to experiment with different sized models, training runs, and datasets. However the course loaders will choose sensible defaults, designed to produce good results for relatively small resource requirements.

The repo includes a helpful system probe script that will assess which size models best fit your local machine:

```bash
./sysprobe.sh
```

You can use the output from that to help you decide on sizing models and datasets. If that clears your system for the *standard track*, then you can most likely use the  defaults without issue.

## System Requirements

The course is designed to be flexible across a wide range of system capabilities. Students are encouraged to experiment with different sized models, training runs, and datasets, because o

The repo includes a helpful system probe script that will assess which size models best fit your local machine:

```bash
./sysprobe.sh
```

Each module's notebook is set up with a lazy loader that automatically loads any dataset or model as needed. The notebooks are setup to to let the user specify, but by default use the strongest compatible local dataset or model that's available, and fall back gracefully when a larger artifact is missing.

Assuming you meet the following specs:
* M2 Chip or better
* 24GB of memory or higher
* 30GB of storage

Then you're probably 
## What You Decide

Before getting started, it's helpful to run the course included system probe script, which will assess the compute and memory capabilities of your Macbook.

```
./sysprobe.sh
```

If you're clear for standard track, then the simplest approach is to l One is to just let the `./notebook.sh` launcher script lazily load any models or datasets as needed. The notebook launcher will default to the standard 

The conceptual path is identical across tracks. Four independent choices set up
your local experience:

| Decision                   | Command            | Tiny                             | Standard                     | Stretch            |
| -------------------------- | ------------------ | -------------------------------- | ---------------------------- | ------------------ |
| Dataset footprint          | `./datasets.sh`    | `--tiny`                         | `--small`                    | (no flag)          |
| TinyLLM<br>(Modules 10-12) | Module 10 notebook | `ShakespeareLM-1M`, `StoryLM-5M` | `StoryLM-30M`, `TinyLLM-30M` | `TinyLLM-100M`     |
| BaseLM (Modules 13-16)     | `./baselm.sh`      | 125M base                        | 350M base                    | 600M base          |
| ProdLM (Modules 16-20)     | `./prodlm.sh`      | 1.5B-3B instruct                 | 7B-8B instruct               | 14B-class instruct |

## Tracks

| Track    | Command                 | Intended use                                            |                  Downloads | Free disk target | Heavy one-time work                                                      |
| -------- | ----------------------- | ------------------------------------------------------- | -------------------------: | ---------------: | ------------------------------------------------------------------------ |
| Tiny     | `./datasets.sh --tiny`  | Fastest path, weaker hardware, quick Module 10 artifact | about 100MB plus artifacts |           5-10GB | TinyStories sample, StoryTokenizer, tokenized sample                     |
| Standard | `./datasets.sh --small` | Recommended local course experience                     |                 several GB |          20-40GB | GloVe, full TinyStories, small G2C corpus, tokenizers, tokenized corpora |
| Full     | `./datasets.sh`         | Stretch path for stronger machines and longer runs      |                      10GB+ |          40-80GB | full G2C corpus, larger tokenized corpus artifact                        |

Datasets are loaded with

```bash
./datasets.sh --tiny    # Tiny dataset track, a few hundred MBs
./datasets.sh --small   # Small dataset track, several GBs
./datasets.sh.          # Standard dataset track, around 10GB
```

BaseLM and ProdLM are prepared separately because they are model artifacts, not
dataset tracks:

```bash
./baselm.sh <[OPTIONAL] hf-model-id>
./prodlm.sh <[OPTIONAL] ollama-tag>
```

Both scripts have sensible defaults if you omit the model tag. Pick any small LM for BaseLM and any local instruct model for ProdLM that fits your machine (see the design doc for hardware-tier guidance).

All `./datasets.sh` commands are intended to be idempotent. Rerunning should skip previously completed downloads and completed artifacts.

| Track    | Raw data                                     | Tokenizers                       |
| -------- | -------------------------------------------- | -------------------------------- |
| Tiny     | TinyStories 100MB sample                     | `StoryTokenizer`                 |
| Standard | GloVe, full TinyStories, small G2C Corpus v1 | `StoryTokenizer`, `G2CTokenizer` |
| Full     | GloVe, full TinyStories, full G2C Corpus v1  | `StoryTokenizer`, `G2CTokenizer` |

The normal `./setup.sh` path stays small. It prepares the Python environment,
TinyShakespeare, and the `ShakespeareTokenizer` artifact.

## Artifact Roles

| Role            | Meaning                                                                                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ShakespeareLM` | Tiny baseline model. Useful for proving the loop works, not for quality.                                                                                              |
| `StoryLM`       | TinyStories-trained model. More coherent stories, still not a general assistant.                                                                                      |
| `TinyLLM`       | Broader G2C-corpus model trained from scratch. Best self-trained candidate for assistant-shaped experiments.                                                          |
| `BaseLM`        | Small external pretrained base model for Modules 13-16                                                                                                                |
| `ProdLM`        | Local pretrained instruct model for Modules 16-20. Served on Ollama.                                                                                                  |
| `<base>-SFT`    | Derived artifacts produced by Module 13. The base is whichever model was fine-tuned.                                                                                  |
| `<base>-DPO`    | Derived DPO model produced in Module 14. The base is whichever model was fine-tuned. DPO artifacts are typically layered on top of the corresponding `-SFT` artifact. |

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

Downloads and tokenized corpora are one-time setup costs. Training runs are the recurring cost. Long training runs are setup to checkpoint so you can interrupt, inspect, sample, and continue.

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

Do not worry about staying inside one track forever. Start small, then add more artifacts when you want better outputs. All of the setup scripts are designed to be idempotent and one button installs and re-installs.

```bash
./datasets.sh --small
./prodLM.sh llama3.2:1b

# Later, if you want stronger artifacts:
./datasets.sh
./prodLM.sh qwen3:4b
```

The course will follow the artifacts you have, not make you manually track which path you chose.

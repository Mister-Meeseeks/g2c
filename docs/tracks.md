# Course Tracks and Artifacts

This course is designed to be flexible across a wide range of system capabilities. The one hard constraint is that most of the exercises and libraries will only work on an M-series Mac laptop or desktop. 

Everything runs on one conceptual path, but there are several local-compute tracks, depending on your specific machine and its CPU/GPU, memory and storage. Students are encouraged to experiment with different-sized models, training runs, and datasets. However, the course loaders will choose sensible defaults, designed to produce good results for relatively small resource requirements.

## System Probe

The repo includes a helpful system probe script that will assess which size models best fit your local machine:

```bash
./sysprobe.sh
```

You can use the output from that to help you decide on sizing models and datasets. If that clears your system for the *standard track*, then you can most likely use the defaults without issue.

## What You Decide

There are two reasons to deviate from the course defaults. One is that your system does not have the resources to support the standard track choices. The other is that you want to experiment with different or more powerful models, which is highly encouraged.

You'll need to make four independent choices across possible tracks. The conceptual path is identical across tracks. Any can be deferred and will fall back to course defaults. The track choices are below:

| Decision                   | Command            | Tiny                             | Standard                     | Stretch            |
| -------------------------- | ------------------ | -------------------------------- | ---------------------------- | ------------------ |
| Dataset footprint          | `./datasets.sh`    | `--tiny`                         | `--small`                    | (no flag)          |
| TinyLLM<br>(Modules 10-12) | Module 10 notebook | `ShakespeareLM-1M`, `StoryLM-5M` | `StoryLM-30M`, `TinyLLM-30M` | `TinyLLM-100M`     |
| BaseLM (Modules 13-16)     | `./baselm.sh`      | 125M base                        | 350M base                    | 600M base          |
| ProdLM (Modules 16-20)     | `./prodlm.sh`      | 1.5B-3B instruct                 | 7B-8B instruct               | 14B-class instruct |

The recommendations for each track are:

| Track    | Target                                                  | Chip                         | Memory |   Downloads | Free disk |
| -------- | ------------------------------------------------------- | ---------------------------- | ------ | ----------: | --------: |
| Tiny     | Fastest path, weaker hardware, quick Module 10 artifact | Any Apple chip               | 8GB    | about 100MB |    5-10GB |
| Standard | Recommended local course experience                     | M2 or better                 | 24GB   |  several GB |   20-40GB |
| Stretch  | Stretch path for stronger machines and longer runs      | M3 or better (16+ GPU cores) | 64GB   |       10GB+ |   40-80GB |

## Datasets

Datasets are loaded with

```bash
./datasets.sh --tiny    # Tiny dataset track, a few hundred MBs
./datasets.sh --small   # Small dataset track, several GBs
./datasets.sh          # Standard dataset track, around 10GB
```

All `./datasets.sh` commands are intended to be idempotent. Rerunning should skip previously completed downloads and derived artifacts.

| Track    | Raw data                                     | Tokenizers                       |
| -------- | -------------------------------------------- | -------------------------------- |
| Tiny     | TinyStories 100MB sample                     | `StoryTokenizer`                 |
| Standard | GloVe, full TinyStories, small G2C Corpus v1 | `StoryTokenizer`, `G2CTokenizer` |
| Full     | GloVe, full TinyStories, full G2C Corpus v1  | `StoryTokenizer`, `G2CTokenizer` |
The normal `./setup.sh` path handles the smallest datasets. It prepares the Python environment, TinyShakespeare, and the `ShakespeareTokenizer` artifact.

## External models

BaseLM and ProdLM are prepared separately because they are model artifacts, not
dataset tracks:

```bash
./baselm.sh <hf-model-id> [OPTIONAL]
./prodlm.sh <ollama-tag> [OPTIONAL]
```

Both scripts have sensible defaults if you omit the optional model tag. Pick any small base LM for BaseLM and any local instruct model for ProdLM that fits your machine. You can use `sysprobe.sh` to get a list of suggestions, or use any valid Hugging Face or Ollama tag:

```bash
./sysprobe.sh    # Run sysprobe checks against suggested candidate set
./sysprobe.sh --baselm-model  Qwen/Qwen3-0.6B-Base  # Eval for BaseLM
./sysprobe.sh --prodlm-model qwen3.5:9b             # Eval for ProdLM
```

You can use the scripts to download multiple models. The scripts are idempotent and cache the results, so you can run them multiple times. The last tag the script is run with becomes the canonical BaseLM or ProdLM model:

```bash
./prodlm.sh llama3.2:3b    # Downloads llama, sets it as ProdLM
./prodlm.sh qwen2.5:3b     # Downloads qwen, ProdLM now points to qwen
./prodlm.sh llama3.2:3b    # Qwen still cached, ProdLM now points to llama
```

The notebooks all allow you to specify any valid Hugging Face (BaseLM) or Ollama (ProdLM) model that you have previously downloaded. Re-running the same notebook with different models is a very good learning experience.

Note that ProdLM and BaseLM have different system requirements for equivalent sized models. BaseLM is fine-tuned locally, whereas ProdLM is only used for Ollama inference. Therefore the ceiling for BaseLM is substantially lower than ProdLM's. For example, the course defaults use a 360M param model for BaseLM and a 2B model for ProdLM.

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

Downloads and tokenized corpora are one-time setup costs. Training runs are the recurring cost. Long training runs are set up to checkpoint so you can interrupt, inspect, sample, and continue.

## Module Expectations

| Modules | Requirement pattern                                                                                                                                                                                                    |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01-03B  | Normal setup only.                                                                                                                                                                                                     |
| 04      | Conceptual tokenizer exercises run small; artifact mini-milestone uses the dataset track you prepared.                                                                                                                 |
| 05      | Optional GloVe download for pretrained vector exercises.                                                                                                                                                               |
| 06-09B  | Mostly lightweight. Use TinyShakespeare and small synthetic corpora.                                                                                                                                                   |
| 10      | Main artifact fork: ShakespeareLM baseline, StoryLM, and optional TinyLLM.                                                                                                                                             |
| 11      | Uses the strongest saved Module 10 model it can find.                                                                                                                                                                  |
| 12      | Scaling lab. Extends Module 10 and can stay small or go stretch.                                                                                                                                                       |
| 13-15   | Prefer a capable self-trained TinyLLM when available; otherwise run `./baselm.sh` and use BaseLM.                                                                                                                      |
| 16-18   | Use ProdLM for the main assistant path. The strongest self-trained artifact can also be loaded for comparison.                                                                                                         |
| 19-20   | ProdLM is the default live backend; deterministic exercises use a `FakeBackend`. Self-trained artifacts can be swapped in via `MODEL_SELECTION`, but expect ReAct and multi-turn formats to break down.                |

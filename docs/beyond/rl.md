# Beyond — Reinforcement learning for LLMs

> **Question this module answers:** *How does a model get better at problems where you can check the answer but not write it?*

<!-- TODO(hero pipeline): asset not yet generated -->
![The GRPO loop drawn as a circle: one prompt fans out to K sampled completions, a programmatic verifier scores each one, scores become group-relative advantages, and the policy update pushes up above-average completions while a KL leash anchors the model to its frozen reference.](rl/BeyondRL-Hero.png)

Module 13 taught the model behavior by imitation: here is the answer, predict it. Module 14 nudged preferences from a fixed dataset of chosen-vs-rejected pairs. Both learn from data that existed before training started. This module closes a different loop — the model generates its own attempts, a program scores them, and the scores become the training signal. Verifiable-reward RL is now an important part of training reasoning and tool-using systems. You'll build its smallest legible version on a laptop: fresh LoRA adapters over BaseLM, toy tasks with exact graders, and the group-relative core of GRPO.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course. Come here in any order, whenever a model card or paper names the idea and you want the load-bearing version — built, trained, and broken on your own machine.

---
## Before you start

* *Review*
	* [13-sft](../modules/13-sft.md) for the chat template and the masked-loss discipline — both return here
	* [14-dpo](../modules/14-dpo.md) for the reference model and the KL intuition
	* [11-sampling](../modules/11-sampling.md) for the generation loop that produces the training data this time
* *Finish*
	* `g2c/sft` ([13-sft](../modules/13-sft.md))
	* `g2c/sampling` ([11-sampling](../modules/11-sampling.md))
	* `g2c/training` ([03b-training](../modules/03b-training.md))
	* `g2c/lora` ([13b-lora](../modules/13b-lora.md)) — LoRA is the required training path in this module
* *Run*
	* `./baselm.sh` to set up the BaseLM artifact this module fine-tunes
	* `G2C_APPLY_SOLUTIONS=01-13b ./notebook.sh rl` instead of the plain launch if you're entering without your own implementations

---
## Where this fits in

Every training signal in the course so far has been **supervised**: the data contained the right answer, token by token, and the loss measured distance from it. That works exactly as far as someone can write the answers down. It fails for the problems we most want models to solve — long derivations, working code, multi-step tool use — where checking an answer is easy but authoring the ideal token sequence is not, and where the model's *own* best path to an answer may look nothing like a human demonstration.

Reinforcement learning inverts the data flow. Instead of training toward answers written in advance, the model **samples** attempts, a **verifier** scores them after the fact, and the update makes high-scoring attempts more likely. The demonstrations are gone; only a grading function remains.

DeepSeekMath introduced GRPO for mathematical reasoning, and DeepSeek-R1 made verifiable-reward RL a central part of its reasoning recipe. Module 14 stopped at DPO by design — offline, no sampling loop, no programmatic verifier. This module adds the online loop at a scale where you can inspect every rollout and every reward.

## The big idea

The whole loop, which you'll implement piece by piece:

```
        prompt
          │
          ▼
   sample K completions          Module 11's sampler, temperature > 0
          │                      (the model writes its own training data)
          ▼
   verify each one               r_i = a program's score — not a label,
          │                      not a human, not another model
          ▼
   group-relative advantages     A_i = (r_i − mean(r)) / std(r)
          │                      ("was this attempt better than my
          ▼                        other attempts at the same prompt?")
   policy update                 push up log p(completion_i) where A_i > 0,
          │                      down where A_i < 0 — plus a KL leash
          │                      to the frozen reference model
          └───────────► repeat, always with FRESH samples
```

Four ideas carry it: rewards instead of labels, the policy gradient, the group as a baseline, and the KL leash. Each gets a section; each is small.

### What this module means by GRPO

The implementation is intentionally the **group-relative REINFORCE core** of GRPO: fresh samples from the current policy, group-normalized rewards, one policy-gradient update, and a KL penalty to a frozen reference. Full GRPO recipes also use an old-policy likelihood ratio, clipping, and often multiple optimization passes over a rollout batch. Those stability mechanisms matter at scale, but they would hide the estimator this module is trying to expose. The notebook and code therefore call this a *simplified GRPO loop* rather than claiming to reproduce a production trainer.

### Rewards instead of labels

A supervised example specifies ~every token. A reward specifies one number per *attempt*:

```
   SFT example:    "What is 23+58?"  →  "23+58 = 81"      (every token supervised)
   RL episode:     "What is 23+58?"  →  model writes whatever it wants;
                                        verifier returns 1.0 or 0.0
```

This module uses **verifiable rewards** — the scorer is a short program: parse the final answer and compare it with the true sum, or check that the output contains valid JSON with the required key. No learned reward model and no human preference data. That restriction keeps the module's failures legible: when training goes weird, the reward function is twenty lines you can read.

The deeper point, which Exercise 4 turns into an experience: **the verifier is a specification, and the model is a specification-gap-finding machine.** You are not scoring what you meant; you are scoring what you wrote.

### The policy gradient in one line

How do you differentiate "make good attempts more likely" when sampling isn't differentiable? The REINFORCE identity:

```
   ∇ E[reward]  =  E[ reward · ∇ log p(completion) ]
```

Read it right-to-left: take the gradient that would make this completion more likely (`∇ log p` — a quantity you already know how to compute; it's the SFT gradient of the completion), and scale it by how good the completion turned out to be. Good attempt → pull toward it, exactly like an SFT step on self-generated data. Bad attempt → push away. No gradient ever flows through the sampler; the reward is a scalar multiplier on a supervised-shaped gradient.

Used raw, this estimator is uselessly noisy — if every reward is positive, *everything* gets pulled up, just by different amounts, and the signal drowns. The fix is a **baseline**: subtract "how well do I usually do here" so that only *better-than-usual* is reinforced.

### GRPO: the group is the baseline

PPO-style RLHF commonly learns that baseline with a second neural network (a value model) and uses a clipped surrogate for conservative updates. GRPO replaces the learned value baseline with relative scores from `K` completions sampled for the *same prompt*:

```
   rewards for one prompt's group:   r = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

   A_i = (r_i − mean(r)) / std(r)   →   correct attempts get A > 0,
                                        failed attempts get A < 0,
                                        centered and scaled per group
```

The advantage `A_i` asks: *was this attempt better than the model's other attempts at the same prompt?* A hard prompt where one of eight succeeds and an easy prompt where seven of eight succeed both create within-prompt contrast. No value network is needed. A reward model is a separate choice: this module avoids one because its tasks have programmatic verifiers.

One degenerate case is load-bearing: if all K rewards are equal (all failed, or all succeeded), the group carries no information — `std = 0`, and the advantage is undefined. The correct move is to skip that group, not to divide by zero. The tests pin this.

The per-token loss for a completion is then `−A_i · log p(token)` summed **over completion tokens only** — the prompt tokens are the user's, not the model's. This is Module 13's loss mask wearing a new hat, with the same off-by-one seam.

### The KL leash

Reward optimization can trade away useful behavior that the verifier does not score. A narrow "final answer is correct" reward, for example, gives no direct credit for readable explanations or general language quality. A standard counterweight adds a penalty for drifting from a **frozen reference model** (the pre-RL checkpoint — Module 14's reference model, playing the same role for the same reason):

```
   loss = − A_i · log p_θ(completion)  +  β · KL( p_θ ‖ p_ref )
```

`β` sets the leash length: too tight can suppress learning; too loose permits more drift. The reference model anchors the behavior learned before RL while the reward supplies the new direction. The optional no-KL run measures whether a short toy run shows visible drift—it does not promise collapse on schedule.

### Reward hacking

Reward hacking is not the model cheating — it is the model finding behavior that scores well under the objective you actually wrote:

```
   You meant:                          You wrote:                      Model learns:
   "solve the addition problem"        reward if the true sum          emit twenty numbers;
                                       appears anywhere in output      one of them is right

   "answer concisely"                  reward inversely                emit nothing
                                       proportional to length
```

A flawed reward can rise while the intended behavior does not. Exercise 4 therefore reports the sloppy verifier and the honest verifier on held-out prompts, then inspects samples. A short run may discover number spraying, a different loophole, or no exploit at all. The discipline is stable across outcomes: **in RL, the reward curve is a claim; independent evaluation and samples are the evidence.**

### On-policy, and why DPO was the offline cousin

The plain estimator above assumes samples from the **current** model. Reusing older rollouts requires importance ratios and the controls that this simplified implementation omits. This is why the notebook resamples every step and why generation dominates its wall-clock.

It also places Module 14 precisely: DPO learns offline from a *fixed* set of preference pairs, with no fresh rollout-and-verifier loop. It is cheaper and more stable, but bounded by the comparisons in that dataset. This module pays the sampling cost to generate new attempts under the current policy. DPO and online RL are different objectives, not algebraic versions of one another, but the online/offline contrast explains much of their practical tradeoff.

## Concepts to internalize

- **RL trains against a grader, not a corpus.** Anything checkable is trainable — that inversion is the capability unlock and the safety hazard in one move.
- **The policy gradient is SFT on self-generated data, scaled by advantage.** `∇ log p` is a gradient you've computed since Module 13; the reward decides its sign and size.
- **The group baseline is the "relative to what?"** Raw rewards reinforce everything; group-centered rewards reinforce only better-than-usual. Degenerate groups (all same reward) carry zero signal — skip them.
- **The KL leash limits unscored drift.** The reference checkpoint anchors prior behavior while the reward pulls on the measured behavior.
- **Reward hacking is the objective working as written.** The reward curve is a claim; the samples are the evidence. Read them every run.
- **On-policy means fresh samples every step.** Generation cost is not overhead — it *is* the method. DPO is what removing it buys and costs.

### What we don't cover

- **The full clipped GRPO surrogate.** Old-policy likelihood ratios, clipping, multiple optimizer passes, and token-level variants are production stability machinery. This module implements the group-relative REINFORCE core so the estimator remains visible.
- **PPO's value-learning machinery.** GAE and learned value networks are important lineage, but the group baseline lets this module omit them.
- **Learned reward models and full RLHF.** Training a reward model from human preferences is its own pipeline with its own failure modes (the reward model is *also* hackable). Programmatic verifiers let this module omit that component.
- **Agentic / long-horizon RL.** The frontier version runs this same loop inside tool-using environments over thousands-of-token trajectories with checkpointed sandboxes. The loop you build is the same shape; the environment infrastructure is not laptop-shaped and is exactly what the frontier reports spend their pages on.
- **Test-time reasoning budgets.** Effort levels, budget forcing, and "thinking" modes are a decoding-and-training co-design built on top of RL'd models; they need this module but don't fit in it.

---
## What you'll build

Package: `g2c/rl/`

```python
def arithmetic_choice_task(rng) -> Task                  # implemented
def format_task(rng) -> Task                             # implemented
    # task families calibrated to produce mixed groups under BaseLM

def verify_arithmetic(example, completion) -> float       # implemented
def verify_format(example, completion) -> float           # implemented
    # each ~20 lines: parse, check, return 0.0 or 1.0

def sample_group(model, tokenizer, prompt, k, ...) -> GroupSample    # implemented
    # wraps g2c/sampling: K completions at temperature > 0,
    # returns token ids with the prompt/completion boundary marked

def group_advantages(rewards) -> Tensor:                              # SCAFFOLDED
    # (K,) rewards → (K,) advantages: center by group mean, scale by
    # group std; a degenerate group (std == 0) returns all zeros

def completion_log_prob(model, ids, prompt_len) -> Tensor:            # SCAFFOLDED
    # sum of token log-probs over the COMPLETION span only —
    # Module 13's mask seam, one more time

def grpo_loss(logp, ref_logp, advantages, kl_coef) -> Tensor:         # SCAFFOLDED
    # −(A · logp).mean() + kl_coef · KL estimate from (logp, ref_logp)

class GRPOTrainer:
    def train_step(self) -> dict[str, float]                          # SCAFFOLDED
        # sample groups → verify → advantages → loss → optimizer step
        # reports reward, KL, sampled entropy, and degenerate skips
    def train(self, ...) -> dict[str, list]                           # implemented
    def evaluate(self, prompts, verifier=None) -> float               # implemented
        # greedy held-out generated pass rate; verifier override audits
        # a training reward against the intended metric
```

Total scaffolded code: roughly 50 lines across four locations. The trainer reuses `g2c/training`'s optimizer plumbing. The notebook injects rank-8 LoRA adapters into the policy, freezes every base parameter, and loads a second untouched BaseLM as the reference. Each required experiment gets a fresh pair so format training cannot leak into arithmetic and the sloppy verifier is the only changed variable in its comparison.

## How to run the tests

Tests live in `tests/test_rl.py`. Initial state: 7 passed (the provided verifiers, task builders, sampler validation, and trainer argument validation), 14 failed.

```bash
source .venv/bin/activate

pytest tests/test_rl.py                    # all RL tests
pytest tests/test_rl.py -x                 # stop at first failure (recommended)
pytest tests/test_rl.py -k advantages      # group-baseline tests only
pytest tests/test_rl.py -k log_prob        # completion-masking tests only
pytest tests/test_rl.py -k grpo_loss       # loss tests only
pytest tests/test_rl.py -v                 # verbose
```

The tests run against a tiny stub model, not BaseLM — they pin the math (zero-mean advantages, degenerate-group handling, prompt tokens excluded from `completion_log_prob`, loss sign conventions) so that when the notebook's real run misbehaves, you can trust the pieces and debug the loop.

## Exercises

Open the working notebook with `./notebook.sh rl`, write your answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading when you're ready. Partial submissions are fine because blank answers are skipped.

To launch it:

```bash
./notebook.sh rl
```

If at any point you want to archive the work in your current notebook and restart fresh:

```bash
./notebook.sh rl --fresh
```

1. **Build the LoRA policy and baseline it.** Inject rank-8 adapters into BaseLM's query/value projections, freeze the base, print the trainable fraction, and measure greedy format pass rate on prompts excluded from training. The format prompt supplies the opening `{"answer":` prefix: this keeps the binary verifier strict while making success discoverable by a small group sampled from a base model.
2. **Smoke-test the loop on format.** Run ten simplified-GRPO updates against `verify_format`, which checks whether the model completed that prefix into a valid object. Report held-out generated pass rate before and after; interpret training reward and degenerate-group skips as diagnostics rather than the result.
3. **Main run: arithmetic choice.** Release the format models and start from a fresh BaseLM/LoRA pair. Each two-digit addition prompt supplies two numeric options, and the model must emit the correct one. Free-response arithmetic made nearly every sampled group all-wrong in calibration; two options preserve a checkable arithmetic decision while producing enough successes and failures for group-relative learning. Report held-out generated pass rate and inspect reward, KL, sampled entropy, skips, and completions. Improvement is empirical, not promised.
4. **Audit a sloppy verifier.** Start from the same fresh initialization and change only the reward to `verify_arithmetic_sloppy`. Report both sloppy and honest held-out pass rates and cite generated samples. Document the exploit found—or the fact that this finite run did not discover one—and patch the specification gap.
5. **Remove the KL leash (optional).** Run a fresh arithmetic comparison with `kl_coef=0`. Check held-out behavior, KL, sampled entropy, and generations; do not assume a short run must visibly collapse.
6. **Group-size sweep (optional).** Compare `K ∈ {2, 8, 16}` at a fixed total rollout budget. Measure reward variance and degenerate-group frequency instead of assuming a universal knee.

## Pitfalls to expect

- **Log-prob over the prompt tokens.** The completion mask is this module's version of Module 13's off-by-one seam. Symptom: training "works" but drifts strangely — you're reinforcing the model's opinion of the *question*.
- **Dividing by a degenerate group's zero std.** All-correct or all-wrong groups produce NaN advantages that poison the step. Skip them (and log how often — on a too-easy task, that count quietly approaches 100% and learning stops for lack of signal).
- **Reading missing diagnostics as zero drift.** Skipped groups do not run policy/reference rescoring, so their KL and sampled-entropy entries are `NaN` and appear as plot gaps. Zero would be a false measurement.
- **Sampling at temperature 0.** All K completions identical → every group degenerate → zero gradient forever. Exploration is not optional; it's where the signal comes from.
- **A reference model that isn't frozen.** If `p_ref` tracks `p_θ`, the KL term measures nothing and the leash is decorative. The reference is loaded once and never updated.
- **Trusting the reward curve.** A training reward measures the verifier on sampled training attempts. The result is greedy generated pass rate on held-out prompts, scored by the intended verifier, plus the samples themselves.
- **Reusing a trained policy for the next comparison.** If the arithmetic run starts from the format-tuned policy, or the sloppy run starts from the arithmetic-tuned policy, more than one variable changed. The notebook releases each pair and reloads fresh zero-delta adapters.
- **Optimizing every BaseLM parameter.** The required path wraps the frozen, LoRA-injected policy in `LoRAModel`; handing the raw unfrozen BaseLM to the course AdamW allocates full-model optimizer state.

## M-series notes

- **Generation dominates the wall-clock.** Every step samples `K` completions and this teaching implementation deliberately uses Module 11's readable, uncached decoding loop. The required format and arithmetic-choice completions are capped at 12 and 8 tokens respectively. Training reward can therefore improve in fewer optimizer steps while still taking longer than an SFT run with many more steps.
- **Memory: two frozen bases, tiny optimizer.** The LoRA policy still stores and executes all BaseLM weights, and the untouched reference is a second full forward-only model. Only the adapters receive gradients and AdamW state. The notebook releases one pair before loading the next so the independent experiments do not accumulate copies.
- **Use MPS when available.** Rollout generation and recomputed completion log-probabilities both run the model repeatedly. Close other notebook kernels before starting a required run.
- **CI note.** Nothing in this module runs in per-push CI; like Modules 09B+, the training exercises are covered by pre-release dry runs.

---
## Reading

Primary:

- **Shao, Wang, Zhu et al., ["DeepSeekMath" (2024), §4](https://arxiv.org/abs/2402.03300).** The source of GRPO. Compare its old-policy ratio and clipped objective with the group-relative REINFORCE core implemented here.
- **DeepSeek-AI, ["DeepSeek-R1" (2025)](https://arxiv.org/abs/2501.12948).** Verifiable-reward RL for reasoning at a scale far beyond this module. Read §2 for the training stages and keep the difference between a reported large-scale result and this toy experiment explicit.
- **Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992).** REINFORCE. Skim for the identity and the baseline argument — thirty years ahead of its use case.

Secondary:

- **Ouyang, Wu, Jiang et al., "Training language models to follow instructions with human feedback" (2022).** The RLHF pipeline — SFT you built in 13, plus the learned reward model this module deliberately omits. Read to see what verifiable rewards let you skip.
- **Schulman, Wolski, Dhariwal et al., "Proximal Policy Optimization" (2017).** The lineage GRPO simplified. Read the clipping intuition; skip the implementation.
- **Ahmadian, Cremer, Gallé et al., "Back to Basics: Revisiting REINFORCE-Style Optimization for RLHF" (2024).** Independent evidence that the simple estimator, done carefully, competes with PPO — the same simplification pressure GRPO represents.
- **Current model technical reports.** Read their rollout-generation, verifier, clipping/importance-ratio, and distributed-training sections separately. Use this module to identify the loop's parts without assuming every report uses one canonical GRPO recipe.

Optional:

- **Skalse, Howe, Krasheninnikov, Krueger, "Defining and Characterizing Reward Hacking" (2022).** The formal treatment of Exercise 4.
- **Rafailov, Sharma, Mitchell et al., "Direct Preference Optimization" (2023).** Reread after this module — DPO's "your language model is secretly a reward model" argument lands differently once you've run the loop it removes.

## Deliverable checklist

- [ ] All tests in `tests/test_rl.py` pass.
- [ ] Notebook: LoRA trainable-parameter count; held-out generated pass rates for the format smoke test and main arithmetic run; reward, KL, sampled-entropy, and skip diagnostics; generated samples.
- [ ] Sloppy-verifier experiment scored by both sloppy and honest held-out verifiers, with a written account of what the finite run found or failed to find and how to close the specification gap.
- [ ] You can explain — out loud, without notes — why RL needs only a verifier where SFT needs an answer, and what that inversion unlocks.
- [ ] You can explain — out loud, without notes — what the group baseline does to the REINFORCE estimator, and why a degenerate group teaches nothing.
- [ ] You can explain — out loud, without notes — what the KL leash limits, what this simplified loop omits from full GRPO, and how Module 14's offline DPO differs from fresh on-policy sampling.

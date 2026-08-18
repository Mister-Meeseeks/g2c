# Beyond — Reinforcement learning

> **Question this module answers:** *How does a model get better at problems where you can check the answer but not write it?*

<!-- TODO(hero pipeline): asset not yet generated -->
![The GRPO loop drawn as a circle: one prompt fans out to K sampled completions, a programmatic verifier scores each one, scores become group-relative advantages, and the policy update pushes up above-average completions while a KL leash anchors the model to its frozen reference.](rl/BeyondRL-Hero.png)

In previous modules we learned how to post-train a model based on hand written or synthetic answers. But in many cases, especially with agents and tools, we don't care about specific answers as much as objective outcomes. 

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

Every training signal in the course so far has been *supervised*. The data already contained the right answer. The loss merely measured distance from token to token. That works exactly as far as someone can write the answers down. 

It fails for the problems we most want models to solve — long derivations, working code, multi-step tool use — where checking an answer is easy but authoring the ideal token sequence is not. When we care more about function than form, this is also where the model's *own* best path to an answer may look nothing like a human demonstration.

*Reinforcement learning (RL)* inverts this data flow. Instead of training against pre-formulated answers, the model outputs repeatedly samples its own attempt. An external *scorer* returns a numeric *reward* for each attempt. Training updates the weights to make high-scoring attempts more likely. 

In previous eras RL primarily referred to *reinforcement learning from human feedback (RLHF)*. Because of issues with reward hacking on subjecting judgement, RLHF was largely replaced with DPO, which we covered in Module 14. DPO is offline, no sampling loop, no explicit scorer. Explicitly avoiding reinforcement learning by design. 

In most LLM contexts today, and for the rest of the module, RL will refer to *reinfocement learning from verifiable rewards (RLVR)*. RLVR uses a *programmatic* verifier, which scores rewards based off deterministic logic rather than subjective judgement. With RLVR, the demo examples are gone and so is the human judge, only a grader function remains. 

## The big idea

The core of reinforcement learning is a *policy* object which updates with training. In the case of LLMs, the policy is the model itself. Repeated *rollouts* generate *trajectories* of output completions from pre-specified prompts. The *verifier* is a deterministic program that returns a reward based off the trajectory.

The job of the *training algorithm* is to convert rewards into policy updates. For LLMs specifically, which always train based off gradient descent, the operative decision is how to convert rewards to *advantages* which are applied to gradient updates. 

The central way this is done today, and what we'll spend the rest of the module covering, is *group relative policy optimization (GRPO)*. Instead of trying to model or fit activations, GRPO simply asks "how did this rollout do compared to all previous attempts on the prompt?"

The whole loop, which you'll implement piece by piece:

```
        prompt
          │
          ▼
   sample K completions          Module 11's sampler, temperature = 1
          │                           (the model writes its own training data)
          ▼
   verify each one               r_i = a 
          |                           (a program's score — not a label,
          │                            not a human, not another model)
          ▼
   group-relative advantages     A_i = (r_i − mean(r)) / std(r)
          │                        ("was this attempt better than my
          |                          other attempts at the same prompt?")
          ▼                     
   policy update                 push log p(completion_i):  
          |                        up   + KL leash   where A_i > 0
          |                        down + KL leash   where A_i < 0
          │                      
          └───────────► repeat, always with FRESH samples
```

Four ideas carry it: 1) rewards instead of labels; 2) the policy gradient; 3) the group as a baseline; and 4) the KL leash (the same regularizer we covered in Module 14). Each gets a section, but each is small and simple in isolation.

This implementation is stripped down to the core of GRPO. Full GRPO recipes also tend to use old-policy likelihood ratios, clippings, and often multiple optimization passes across a rollout batch. Those stability mechanisms matter at scale, but complicate the core machinery behind RL. The notebook and code therefore call this a *simplified GRPO loop* rather than claiming to reproduce a production trainer.

### Rewards instead of labels

A supervised example specifies every token in the answer to each example. A reward specifies one number per *attempt*:

```
   SFT example:    "What is 23+58?"  →  "23+58 = 81"
   RL episode:     "What is 23+58?"  →  model writes whatever it wants;
                                        verifier returns 1.0 or 0.0
```

Reinforcement learning relies on *verifiable rewards*. That means deterministic *scoring programs*. This scorer in this module's notebook is a simple program: check the output contains valid JSON, look for the the required key, parse the final answer, then compare to the true sum. No learned reward model and no human preference data. That restriction keeps the module's failures legible: when training goes weird, the reward function is twenty lines you can read.

The deeper point: **the verifier is a specification, and the model is a specification-gap-finding machine.** You are not scoring what you meant; you are scoring what you spec'd.

The arithmetic verifier is an *outcome reward*: it scores the completed attempt but says nothing about how intermediate choices helped. A *process reward* by contrast scores selected intermediate reasoning steps, tool actions, or state transitions. This exposes denser credit on long trajectories. That can ease credit assignment and create a more recoverable learning signal, but only if those intermediate judgments are themselves reliable. A flawed process grader can create "traps" that optimize against the wrong specification. This module stays with outcome rewards.

### The policy gradient in one line

How do you differentiate "make good attempts" when sampling isn't differentiable? That's The REINFORCE identity:

```
   ∇ E[reward]  =  E[ reward · ∇ log p(completion) ]
```

Read it right-to-left: take the gradient that would make this completion more likely (`∇ log p` — a quantity you already know how to compute from Module 13), and scale it by this completion's reward. Good attempt → pull toward it, exactly like an SFT step on self-generated data. Bad attempt → push away. No gradient ever flows through the sampler; the reward is a scalar multiplier on a supervised-shaped gradient.

Raw REINFORCE is valid but often high-variance. A positive reward on every attempt makes it hard to distinguish the behavior that unusually useful for this prompt. An action-independent **baseline** can reduce that variance without changing the expected policy gradient. GRPO turns that idea into a practical estimator by centering and scaling the sampled rewards together. The resulting sampled update reinforces better-than-group-average attempts while suppressing worse ones.

### GRPO: the group is the baseline

Prior to GRPO, the most common RL algorithm was PPO. This works by using a second neural network (a value model) to learn expected *baselines*. Policy advantages are calculated as realized rewards minus expected baselines. This rewards not just high scores, but high scores in places where low scores are the norm. GRPO replaces the value model with the just the average score for other samples on the same prompt:

```
   rewards for one prompt's group:   r = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

   A_i = (r_i − mean(r)) / std(r)   →   correct attempts get A > 0,
                                        failed attempts get A < 0,
                                        centered and scaled per group
```

The advantage `A_i` asks: *was this attempt better than other attempts on the same prompt?* A hard prompt where one of eight succeeds and an easy prompt where seven of eight succeed both create within-prompt contrast. No value network is needed.

One degenerate case is load-bearing. If all K rewards are equal (all failed, or all succeeded), the group carries no information. `std = 0`, the advantage is undefined, and no learning signal exists. The correct move is to skip that group.

This exposes the *exploration boundary* of verifier-based RL. A verifier is only useful when the current policy actually sees different reward outcomes. RL cannot reinforce a solution that never appears in its rollouts. Prompting, task difficulty, curriculum, sampling, and group size therefore all factor into whether a verifier produces a learning signal at all. 

To complete the training gradient, the policy activation has to be translated back into per-token loss. This is just `−A_i · log p(token)` summed **over completion tokens only**. The same loss mask we learned in Module 13.

### The KL leash

Reward optimization can trade away useful behavior that's not explicitly scored. A narrow "final answer is correct" reward, for example, gives no direct credit for readable explanations or general language quality. A standard counterweight is to add a penalty for drifting from the **frozen reference model**:

```
   loss = − A_i · log p_θ(completion)  +  β · KL( p_θ ‖ p_ref )
```

`β` sets the leash length. Too tight can suppress learning. Too loose permits drift. The reference model anchors the previously learned behavior. 

Our implementation applies the nonnegative KL estimator to **summed** completion log-probabilities. That makes it a trajectory-level, length-sensitive teaching approximation. Production GRPO more commonly computes the estimate per token and then chooses how to aggregate across response tokens. That choice is part of the response-length story.

It's important to be aware that KL and entropy answer different diagnostic questions. **KL** asks how far the policy moved from the frozen reference. **Sampled entropy** estimates how surprising the sampled completion tokens look under the current policy. A policy can become more concentrated without moving far from an already-concentrated reference, so neither metric substitutes for the other. Some trainers add an entropy bonus to resist collapse. This module observes entropy but does not optimize it directly.

### Reward hacking

Reward hacking is not the model cheating — it is the model finding behavior that scores well under the objective you actually wrote. A flawed reward can rise while the intended behavior does not

If you test if the answer to an arithmetic question appears anywhere in the output, the model might learn to emit hundreds of numbers to maximize its chances. If you reward concise answers, the model might output empty strings. If you reward an answer that's known to exists, the model might break out of its sandbox to search the Internet. Sometimes reward hacking can become literal hacking. 

The lesson in RL is central: the reward curve is just a claim. The real evidence is always independent evaluation.

### RLVR in Agent Environments

In the modern period, the behavior we want from LLMs is increasingly agent based. As we learned in Module 19, models can be adapted to not only answer prompts, but to act as *agents* that with an external environment.

This is the same machinery as vanilla RLVR. The fundamental shift is *environment RL* is where the verifier cares scores the agent environment, rather than the correctness of a response. For example in vanilla RLVR, we might pose a math problem, generate 8 solutions, and grade responses for accuracy. But in environment RL, we might run an agent 8 times in a test environment, then check whether all the unit tests pass. 

The biggest practical challenge for agentic workloads is that the trajectories tend to be multi-turn, highly tool dependent and extremely long. For example a trajectory for a coding agent could look like:

```
Task: fix bug

LLM: inspect file A
ENV: <contents>

LLM: search for Foo
ENV: <search results>

LLM: edit file
ENV: patch applied

LLM: run tests
ENV: 4 failures

LLM: edit again
ENV: patch applied

LLM: run tests
ENV: all pass

reward = 1
```

The **entire interaction history** is now the rollout. That's why papers often distinguish agentic RL from ordinary single-turn RLVR even when the final reward is perfectly verifiable. Software-agent work, for example, uses unit tests as rewards but finds naive RLVR difficult because successful terminal rewards are extremely sparse over long multi-step trajectories. The interesting question is what counts as the trajectory whose log probability you're optimizing.

There might be 50 decisions over half an hour of simulated interaction. Perhaps action #7 was brilliant, action #19 was useless, action #31 nearly ruined the solution, and action #44 rescued it. Yet terminal reward gives you `R=1.0` for everything.

That's why current agent RL research spends so much effort on denser rewards, process rewards, subgoal rewards, trajectory segmentation, search, curriculum/guidance, and improved credit assignment. Recent work explicitly describes sparse outcome rewards as a key obstacle for multi-turn agents.

### On-policy, and why DPO was the offline cousin

The plain estimator above assumes samples from the **same current policy whose log-probabilities appear in the update**. Reusing older rollouts requires importance ratios and the controls that this simplified implementation omits. Changing sampling temperature also changes the behavior policy; this module fixes it at `1.0` so untempered generation and rescoring match. The notebook resamples every step, and generation therefore dominates its wall-clock.

It also places Module 14 precisely: DPO learns offline from a *fixed* set of preference pairs, with no fresh rollout-and-verifier loop. It is cheaper and more stable, but bounded by the comparisons in that dataset. This module pays the sampling cost to generate new attempts under the current policy. DPO and online RL are different objectives, not algebraic versions of one another, but the online/offline contrast explains much of their practical tradeoff.

## Concepts to internalize

- **RL trains against a grader, not a corpus.** Checkable behavior is potentially trainable when exploration produces reward variation — that inversion is the capability unlock and the safety hazard in one move.
- **The policy gradient is SFT on self-generated data, scaled by advantage.** `∇ log p` is a gradient you've computed since Module 13; the reward decides its sign and size.
- **The group baseline is the "relative to what?"** Centering rewards reduces variance and turns the sampled update into better-versus-worse contrast. Degenerate groups (all same reward) carry zero relative signal — skip them.
- **Outcome rewards leave credit assignment to exploration.** Process rewards can make the signal denser, but require trustworthy intermediate judgments.
- **The KL leash limits unscored drift.** The reference checkpoint anchors prior behavior while the reward pulls on the measured behavior.
- **KL and entropy are not interchangeable.** One measures drift from a reference; the other measures concentration of the current policy.
- **Reward hacking is the objective working as written.** The reward curve is a claim; the samples are the evidence. Read them every run.
- **On-policy means fresh samples every step.** Generation cost is not overhead — it *is* the method. DPO is what removing it buys and costs.

### What we don't cover

- **The full clipped GRPO surrogate.** Old-policy likelihood ratios, clipping, multiple optimizer passes, and production token-aggregation choices add stability and reuse rollout batches. This module implements the group-relative REINFORCE core so the estimator remains visible.
- **PPO's value-learning machinery.** GAE and learned value networks are important lineage, but the group baseline lets this module omit them.
- **Learned reward models and full RLHF.** Training a reward model from human preferences is its own pipeline with its own failure modes (the reward model is *also* hackable). Programmatic verifiers let this module omit that component.
- **Process supervision.** Step-level rewards can improve credit assignment on long solutions, but building a reliable process grader is a separate data and evaluation problem.
- **Response-length corrections.** Summing versus averaging token objectives changes how response length affects an update, and GRPO variants make different normalization choices. The short capped completions here keep that issue visible in the trajectory-level KL without turning it into another implementation branch.
- **Entropy regularization.** The trainer reports sampled entropy as a diagnostic but does not add an entropy bonus or adapt sampling pressure during training.
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
    # wraps g2c/sampling: K stochastic completions at temperature 1.0,
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
        # sample group → verify → advantages → update if informative
        # reports reward, KL, sampled entropy, and degenerate skips
    def train(self, ...) -> dict[str, list]                           # implemented
    def evaluate(self, prompts, verifier=None) -> float               # implemented
        # greedy held-out generated pass rate; verifier override audits
        # a training reward against the intended metric
```

Total scaffolded code: roughly 50 lines across four locations. The trainer reuses `g2c/training`'s optimizer plumbing. The notebook injects rank-8 LoRA adapters into the policy, freezes every base parameter, and loads a second untouched BaseLM as the reference. Each required experiment gets a fresh pair so format training cannot leak into arithmetic and the sloppy verifier is the only changed variable in its comparison.

## How to run the tests

Tests live in `tests/test_rl.py`. Initial state: 8 passed (the provided verifiers, task builders, sampler validation, and trainer argument validation), 14 failed.

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

1. **Build the LoRA policy and baseline it.** Inject rank-8 adapters into BaseLM's query/value projections, freeze the base, print the trainable fraction, and measure greedy format pass rate on prompts excluded from training. The format prompt supplies the opening `{"answer":` prefix, making both successful and failed completions discoverable by a small group sampled from a base model. The format-only verifier accepts a valid first object even if text follows it; that narrow contract is intentional and should not be mistaken for full structured-output validation.
2. **Smoke-test the loop on format.** Run ten simplified-GRPO rollout steps against `verify_format`, which checks whether the model completed that prefix into a valid object. Degenerate groups count as rollout steps but not optimizer updates, so report both attempts and skips. Report held-out generated pass rate before and after; interpret training reward and degenerate-group skips as diagnostics rather than the result.
3. **Main run: arithmetic choice.** Release the format models and start from a fresh BaseLM/LoRA pair. Each two-digit addition prompt supplies two numeric options, and the model must emit the correct one. Free-response arithmetic made nearly every sampled group all-wrong in calibration; two options preserve a checkable arithmetic decision while producing enough successes and failures for group-relative learning. Report held-out generated pass rate and inspect reward, KL, sampled entropy, skips, and completions. Improvement is empirical, not promised.
4. **Audit a sloppy verifier.** Reset to the same seeded BaseLM/LoRA initialization and change only the reward to `verify_arithmetic_sloppy`. Report both sloppy and final-answer held-out pass rates and cite greedy held-out generations from the evaluated policy. Document the exploit found—or the fact that this finite run did not discover one—and patch the specification gap. The final-answer verifier scores the last emitted integer; it tests answer correctness, not strict compliance with the prompt's "only a number" wording.
5. **Remove the KL leash (optional).** Run a fresh arithmetic comparison with `kl_coef=0`. Check held-out behavior, KL, sampled entropy, and generations; do not assume a short run must visibly collapse.
6. **Group-size sweep (optional).** Compare `K ∈ {2, 8, 16}` at a fixed total rollout budget. Measure reward variance and degenerate-group frequency instead of assuming a universal knee.

## Pitfalls to expect

- **Log-prob over the prompt tokens.** The completion mask is this module's version of Module 13's off-by-one seam. Symptom: training "works" but drifts strangely — you're reinforcing the model's opinion of the *question*.
- **Dividing by a degenerate group's zero std.** All-correct or all-wrong groups produce NaN advantages that poison the step. Skip them (and log how often — on a too-easy task, that count quietly approaches 100% and learning stops for lack of signal).
- **Reading missing diagnostics as zero drift.** Skipped groups do not run policy/reference rescoring, so their KL and sampled-entropy entries are `NaN` and appear as plot gaps. Zero would be a false measurement.
- **Changing rollout temperature without changing the scorer.** This simplified trainer fixes `temperature=1.0` so generation and recomputed log-probabilities describe the same policy. Greedy decoding gives no exploration; another non-unit temperature would require scoring that tempered behavior policy or adding an off-policy correction.
- **A reference model that isn't frozen.** If `p_ref` tracks `p_θ`, the KL term measures nothing and the leash is decorative. The reference is loaded once and never updated.
- **Trusting the reward curve.** A training reward measures the verifier on sampled training attempts. The result is greedy generated pass rate on held-out prompts, scored by the intended verifier, plus the samples themselves.
- **Changing initialization between verifier comparisons.** If the sloppy run starts from the trained arithmetic policy or from a different random LoRA basis, more than one variable changed. The notebook reloads the base and resets the adapter seed so both arithmetic trainers begin from the same zero-delta parameters.
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
- [ ] Sloppy-verifier experiment scored by both sloppy and final-answer held-out verifiers, with a written account of what the finite run found or failed to find and how to close the specification gap.
- [ ] You can explain — out loud, without notes — why RL needs only a verifier where SFT needs an answer, and what that inversion unlocks.
- [ ] You can explain — out loud, without notes — what the group baseline does to the REINFORCE estimator, and why a degenerate group teaches nothing.
- [ ] You can explain — out loud, without notes — what the KL leash limits, what this simplified loop omits from full GRPO, and how Module 14's offline DPO differs from fresh on-policy sampling.

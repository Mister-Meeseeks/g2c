# Module 03B — Training

> **Question this module answers:** *Why does the same network sometimes learn, stall, or explode?*

![A training dashboard: learning rate controls step size, AdamW adapts that step per parameter, gradient clipping caps rare spikes, and train/validation curves diagnose whether the run is learning, overfitting, or stalled.](03b-training/Module03B-Hero.png)

*The model architecture is only half the story. The training recipe controls whether gradient descent turns that architecture into a useful function. This module names the knobs that otherwise feel like magic when they reappear in pretraining.*

## Prerequisites

### Math

- **Gradient descent from Module 03.** You should be comfortable with `param -= lr * grad`.
- **Elementwise tensor arithmetic.** AdamW's state is just tensors with the same shape as each parameter.
- **Basic plots.** You will read loss curves and learning-rate curves.

### Computer Science

- **Stateful objects.** Optimizers carry state between steps. `SGD` barely has any; `AdamW` has two buffers per parameter.
- **Interfaces.** Module 10's trainer only needs `zero_grad`, mutable `lr`, and `step`. Different optimizers can plug into the same loop if they satisfy that interface.

### Programming

- **PyTorch autograd.** You still use `.backward()` to populate gradients.
- **No high-level optimizer imports for the implementation.** Do not use `torch.optim.AdamW` to implement `g2c.training.AdamW`. Comparing against it in an experiment is fine after your own version works.

### What You Can Skip

- Formal convergence proofs.
- The optimizer zoo. We only need SGD and AdamW.
- BatchNorm and advanced regularization. They matter in deep learning generally, but LayerNorm and AdamW matter more for this LLM stack. Dropout is introduced conceptually below, but you will not implement it in this module.

## Why This Module Exists

Module 03 taught the basic loop:

```text
forward -> loss -> backward -> optimizer.step()
```

That loop is correct, but it does not tell you how to choose the knobs. A too-small learning rate makes a correct model crawl. A too-large learning rate can destroy a run. SGD uses one global step size for every parameter, which is often enough for small MLPs but often stalls on transformers. Deep models also occasionally produce gradient spikes; clipping prevents one bad batch from wrecking the run. Long runs usually need a learning-rate schedule so early steps can move fast and late steps can settle.

The point of this module is not to turn training into a cookbook. The point is to give you a small diagnostic language:

```text
loss explodes        -> lr probably too high, or gradients spiked
loss barely moves    -> lr too low, optimizer weak, or model/data mismatch
train low, val high  -> overfitting
train and val high   -> underfit or under-optimized
train and val flat   -> stalled optimization
```

When Module 10 pretrains a tiny GPT, these ideas should already be familiar. The payoff module should feel like assembly, not like learning five training tricks at once.

## The Big Idea

The learning rate is not "how much the model learns." It is the scale of the parameter update:

```text
update size ~= learning_rate * gradient scale
```

That means the same `lr` can be too small for one parameter and too large for another. A token embedding row that appears rarely may receive tiny gradients. An output-head column for a common token may receive large gradients. LayerNorm scale parameters, attention projections, and feed-forward weights all live on different gradient scales.

SGD uses one global `lr`:

```text
param <- param - lr * grad
```

AdamW keeps running statistics so each parameter gets a better-scaled update:

```text
m <- beta1 * m + (1 - beta1) * grad
v <- beta2 * v + (1 - beta2) * grad^2

m_hat <- m / (1 - beta1^step)
v_hat <- v / (1 - beta2^step)

param <- param * (1 - lr * weight_decay)
param <- param - lr * m_hat / (sqrt(v_hat) + eps)
```

The result is still gradient descent. AdamW does not change the model or the objective. It changes how the update is scaled.

### Learning Rates

The learning rate is the knob that turns gradient direction into parameter motion. In the plain SGD case, the relationship is visible:

```text
update = -lr * grad
```

If `lr` is too small, the loss may be moving in the right direction but too slowly for your compute budget. If `lr` is too large, the update jumps past useful regions of parameter space and loss can spike or become `nan`. Most "what learning rate should I use?" questions are really asking: *what update scale is reasonable for this model, optimizer, batch size, and data?*

There is no universal answer, so the practical move is a small sweep. Try a few values spaced by powers of 3 or 10, plot the curves, and keep the largest learning rate that trains smoothly. That habit matters more than memorizing one magic constant.

### AdamW's Effective Step Size

AdamW still has a global `lr`, but the actual update is adapted per tensor element:

```text
param <- param - lr * m_hat / (sqrt(v_hat) + eps)
```

The denominator is why AdamW can tolerate transformer training better than raw SGD. If one parameter has historically large gradients, `sqrt(v_hat)` is large and the effective step for that parameter shrinks. If another parameter has small gradients, it is not forced to share the exact same raw scale. The global `lr` still matters, but AdamW makes it less brittle.

Weight decay is separate:

```text
param <- param * (1 - lr * weight_decay)
```

That direct shrink is the "W" in AdamW. Do not fold it into the gradient update.

### Gradient Clipping

Gradient clipping handles rare bad steps. First compute the global norm across every parameter gradient:

```text
total_norm = sqrt(sum(||p.grad||^2 for every parameter))
```

If `total_norm` exceeds `max_norm`, multiply every gradient by the same scalar:

```text
scale = max_norm / total_norm
p.grad <- p.grad * scale
```

This preserves the overall gradient direction and only shortens the step. It is not the same as clipping each parameter independently, and it is not a replacement for a reasonable learning rate. If clipping fires constantly, your run is telling you the requested updates are too aggressive.

### Warmup And Cosine Decay

A schedule changes the learning rate over training. The standard small-LM recipe has two phases:

```text
warmup:        lr rises linearly to max_lr
cosine decay:  lr falls smoothly from max_lr to min_lr
```

Warmup protects the first steps, when the model is random and gradients can be poorly scaled. Cosine decay then lets the run take larger steps early and smaller steps late. In this repo's convention, if `warmup_steps > 0`, step 0 uses `max_lr / warmup_steps`, and the last warmup step reaches exactly `max_lr`.

This is not deep magic. It is arithmetic on the step counter. But once it is named and implemented, Module 10 can focus on the language-modeling loop instead of reintroducing schedules during the payoff week.

### Reading Curves

Training curves are your feedback loop:

- **Train and validation both fall smoothly.** The run is healthy.
- **Train falls, validation rises.** The model is overfitting or the validation split is too small/noisy.
- **Both stay high and flat.** The model is under-optimized, the learning rate is wrong, or the model/data pair is too weak.
- **Loss spikes or becomes `nan`.** Lower the learning rate, check gradient norms, and make sure clipping is wired before `optimizer.step`.

Curve reading is the bridge from "I know the mechanics" to "I can debug a training run."

### Regularization And Dropout

Optimization asks whether the model can lower the training loss. Regularization asks whether the model is learning something that transfers beyond the exact examples it saw. The classic warning sign is a widening train/validation gap: train loss keeps improving while validation loss stalls or gets worse.

Regularization is a family of responses to that pattern. Weight decay gently prefers smaller weights. More data gives the model fewer chances to memorize quirks. Early stopping keeps the checkpoint from the best validation point instead of the final training step. Data cleaning and deduplication prevent the model from seeing the same examples so often that memorization becomes the easy path.

Dropout is another regularizer. During training, it randomly zeros some activations, forcing the network not to rely too heavily on any one feature path. At evaluation time, dropout is disabled so predictions are deterministic. Correct dropout also has scaling details so the expected activation size stays comparable between train and eval.

We are not implementing dropout in this course path because it is not the bottleneck for the tiny GPT stack. It adds a useful but separate concept cluster: stochastic forward passes, train/eval mode, RNG control, and activation scaling. For GPT-style decoder-only pretraining today, dropout is often small or zero, while AdamW, learning-rate schedules, clipping, data quality, and validation monitoring are more central. So dropout is worth knowing by name, but not worth spending a build week on here.

## Concepts To Internalize

- **Learning rate controls update scale.** If loss explodes, lower it. If loss crawls and gradients are finite, raise it.
- **SGD has one global scale.** Every parameter sees the same nominal `lr`.
- **Momentum smooths direction.** AdamW's `m` is a moving average of gradients.
- **Second moment scales the step.** AdamW's `v` tracks squared gradients; large historical gradients shrink the effective step.
- **Bias correction matters early.** At step 1, `m` and `v` are biased toward zero. Dividing by `1 - beta^step` fixes that.
- **AdamW weight decay is decoupled.** Shrink the parameter directly. Do not add `weight_decay * param` to the gradient as SGD does.
- **Gradient clipping is a guardrail.** It rescales a too-large global gradient vector; it does not replace a reasonable learning rate.
- **Schedules are part of the run.** Warmup avoids blasting random-initialized weights; cosine decay lowers the step size as the run settles.
- **Curves are diagnostics.** Train/validation loss curves are how you decide what to change next.
- **Regularization targets generalization.** Dropout is one regularizer, but this course leans on weight decay, data, and validation curves for the tiny LLM path.

## Scaffolding And How To Run The Tests

This module touches two packages:

- `g2c/training/optim.py` — `AdamW` lives here because Module 03B owns adaptive optimization.
- `g2c/training/clip.py` and `g2c/training/schedule.py` — gradient clipping and warmup/cosine scheduling live beside `AdamW`.

Run the focused tests:

```bash
source .venv/bin/activate

pytest tests/test_training.py -x
```

Suggested implementation order:

1. **`AdamW.step`** in `g2c/training/optim.py`.
2. **`clip_grad_norm_`** in `g2c/training/clip.py`.
3. **`cosine_with_warmup`** in `g2c/training/schedule.py`.
4. **Notebook experiments** comparing bad LR, good LR, SGD, AdamW, clipping, and schedules.

## What You'll Build

### `AdamW`

The constructor is implemented. It stores:

```python
self.params
self.lr
self.beta1, self.beta2
self.eps
self.weight_decay
self.m              # first moments, one tensor per param
self.v              # second moments, one tensor per param
self.step_count
```

You implement `step()`.

### `clip_grad_norm_`

Compute the global L2 norm over all populated gradients:

```text
total_norm = sqrt(sum(||p.grad||^2 for every parameter))
```

If `total_norm > max_norm`, multiply every gradient by `max_norm / total_norm`.

### `cosine_with_warmup`

Return the scheduled learning rate at a given step:

```text
warmup phase: lr ramps linearly up to max_lr
cosine phase: lr decays smoothly down to min_lr
```

### Optimizer Plug-In Boundary

Module 10's trainer can choose either optimizer:

```python
Trainer(..., optimizer="sgd")
Trainer(..., optimizer="adamw")
```

The training step stays the same because both optimizers expose the same small interface.

## Exercises

Enter questions or answers in [`answers/module-03b.md`](../../answers/module-03b.md) for agent help and grading. You can ask for a hint, answer one question, answer a subset, or answer all of them; blank answer sections are skipped rather than counted wrong.

Set up or resume the working notebook:

```bash
.venv/bin/python scripts/open_notebook.py 03b
```

1. **Learning-rate sweep.** Train the same tiny MLP at `lr ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1.0}`. Plot final train loss and describe the pattern. Which runs crawl, which learn, and which diverge?

2. **AdamW by hand.** For a scalar parameter `p = 1.0`, gradient `g = 0.1`, `lr = 0.001`, `betas = (0.9, 0.999)`, `eps = 1e-8`, and no weight decay, compute the first AdamW update by hand. Explain why bias correction makes the first update roughly `0.001`.

3. **Implement `AdamW.step`.** Run `pytest tests/test_training.py -k adamw -x` until the AdamW tests pass.

4. **SGD vs AdamW on the same model.** Train the same MLP with SGD and AdamW. Keep model, data, seed, and number of steps fixed. Plot both loss curves. Which optimizer is less sensitive to the initial LR choice?

5. **Gradient clipping demo.** Create two fake parameters with gradients `[3]` and `[4]`. Clip to `max_norm=1.0`. Explain why both gradients are multiplied by `1/5`, not clipped independently. Run `pytest tests/test_training.py -k clip_grad_norm -x`.

6. **Warmup/cosine schedule.** Plot `cosine_with_warmup` for `warmup_steps=100`, `max_steps=1000`, `max_lr=3e-4`, `min_lr=3e-5`. Explain what happens at step 0, step 99, step 100, halfway through decay, and at the end. Run `pytest tests/test_training.py -k cosine_with_warmup -x`.

7. **Curve diagnosis.** Given three train/validation plots — both high and flat, train low / val high, and both decreasing smoothly — write what you would try next for each.

## Pitfalls To Expect

- **Confusing Adam and AdamW.** Adam's original L2 penalty adds `weight_decay * param` to the gradient. AdamW decays the parameter directly. The distinction matters.
- **Incrementing `step_count` per parameter.** The step counter advances once per optimizer step.
- **Forgetting bias correction.** Early AdamW updates become incorrectly small or oddly scaled.
- **Letting optimizer updates enter autograd.** Wrap `step()` in `torch.no_grad()`.
- **Clipping each parameter independently.** Clipping is global. Preserve the direction of the full gradient vector.
- **Treating clipping as a tuning substitute.** If clipping fires constantly, the learning rate is probably too high.
- **Reading only training loss.** A model can improve train loss while getting worse on validation data.

## Reading

Primary:

- **Kingma and Ba, "Adam: A Method for Stochastic Optimization" (2014).** The original Adam paper. Focus on the update rule and bias correction.
- **Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (2017).** The AdamW paper. Focus on why weight decay should be decoupled from the adaptive gradient update.
- **Karpathy, nanoGPT `configure_optimizers` and training loop.** Read after implementing AdamW; it will look much less magical.

Secondary:

- **Goodfellow, Bengio, Courville, *Deep Learning*, optimization chapter.** Useful for vocabulary around momentum, conditioning, and learning-rate schedules.
- **Goyal et al., "Accurate, Large Minibatch SGD" (2017).** Skim for the warmup idea.

## Deliverable Checklist

- [ ] `AdamW.step` passes `pytest tests/test_training.py -k adamw`.
- [ ] `clip_grad_norm_` and `cosine_with_warmup` pass `pytest tests/test_training.py`.
- [ ] The notebook includes an LR sweep plot.
- [ ] The notebook includes an SGD vs AdamW comparison on the same model.
- [ ] The notebook includes a gradient clipping demonstration.
- [ ] The notebook includes a warmup/cosine schedule plot.
- [ ] You can explain why Module 10 will use AdamW for the serious training run without changing the model architecture or loss.

## M-Series Notes

Everything in this module should run comfortably on CPU. MPS is useful for larger notebook experiments, but the point here is diagnosis, not throughput. Keep the experiments small enough that you can rerun them many times while changing one knob at a time.

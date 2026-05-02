# Module 00 Answers

Student-owned workspace for Module 00 written exercises.

Use `Help request / hint request` when you are stuck and want tutoring before grading. Use `Student answer` when you want an answer graded. Blank sections are fine; the grader should skip them rather than treating them as wrong. Leave the rubric alone, and when you want feedback or a hint, ask an agent:

```text
Can you review my module 0 answers?
```

## Exercise 00.01 — Shape trace

### Help request / hint request


### Student answer
After embedding lookup the shape is (B,T,C) this is (4,8,16) because each token gets projected to a vector of length C. The logits have shape (B,T,V) this is (4,8,1000) because each token has a logic across every possible token. 

### Notes / uncertainty


## Exercise 00.02 — Matmul by hand

### Help request / hint request


### Student answer

| 220   280 |
| 490   640 |

### Notes / uncertainty


## Exercise 00.03 — Backprop by hand

### Help request / hint request


### Student answer
dL/db = (2a - 2 * target) * (1 - a^2) 
dL/dx = w * (2a - 2 * target) * (1 - a^2)
dL/dw = x * (2a - 2 * target) * (1 - a^2)
### Notes / uncertainty


## Exercise 00.04 — Softmax and loss

### Help request / hint request


### Student answer
Probabilities: 0.665, 0.245, 0.09
Neg log likelihod: 0.4, 1.4, 2.4

### Notes / uncertainty
Values are rounded

## Exercise 00.05 — Training-loop narration

### Help request / hint request


### Student answer
In the loop logits are generated in the forward. Then a loss score for the entire batch is calculated. Then backprop is used to derive the gradient of the loss against each parameter. The gradient is multiplier by a step size along each parameter dimension. This step is added back to the previous parameter values. And then the loop repeats 

### Notes / uncertainty


## Exercise 00.06 — Environment check

### Help request / hint request


### Student answer
Environmnet works. MPS is installed

### Notes / uncertainty

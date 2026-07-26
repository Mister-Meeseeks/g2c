# Module 04 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/04-tokenizer.ipynb`, falling back to `notebooks/clean/04-tokenizer.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

Note: the Exercise 2 questions sit below a `tiny_corpus = "the the the"` line in their cell — grade every `Question:` string, not just cells that open with one.

## Exercise 04.01 — Next failing test

Run-dependent — grade the reasoning shape.

A correct answer should include:

- A concrete test name from `pytest tests/test_tokenizer.py -x` at the time they ran it, mapped to the implementation it points at (`_get_pair_counts`, `_merge`, `train_step`, `encode`, or `decode` in `g2c/tokenizer/bpe.py`). "All passing" is a complete answer.

Common issues:

- Naming a file with no test, or a test with no target function.

## Exercise 04.02 — Pair counts for `[1, 2, 1, 2, 3]`

A correct answer should include:

- `{(1, 2): 2, (2, 1): 1, (2, 3): 1}`.

Common issues:

- Counting unordered pairs (merging `(1, 2)` and `(2, 1)` into one bucket) — pairs are ordered.
- Including a wrap-around pair `(3, 1)` or counting non-adjacent co-occurrences.

## Exercise 04.03 — Why n − 1 adjacent pairs

A correct answer should include:

- Each pair starts at an index `i` with a successor at `i + 1`; valid starts are `0 … n − 2`, so there are `n − 1` pairs. Equivalently: the last element has no successor.

Common issues:

- Off-by-one hand-waving ("fencepost") without identifying which element lacks a partner.

## Exercise 04.04 — `_merge([1, 1, 1], (1, 1), 99)`

A correct answer should include:

- `[99, 1]`, not `[99, 99]`.
- Why: merging is left-to-right and non-overlapping — the first match consumes indices 0 and 1 (the middle `1` is used up), leaving a lone trailing `1` with no partner.

Common issues:

- `[99, 99]` from letting the middle element participate in two matches.
- Right answer, no consumption argument.

## Exercise 04.05 — Most frequent pair in `"the the the"`

A correct answer should include:

- `(116, 104)` = `'t','h'`, count 3 — the pair the worked example merges first.
- (Bonus, worth crediting) `('h','e')` also occurs 3 times; `'th'` wins the tie by first-encountered order in the counts dict.

Common issues:

- Counting `('e', ' ')` or `(' ', 't')` as 3 (they occur only twice — there are two spaces).
- Answering with a character pair but the wrong count.

## Exercise 04.06 — Merges for target vocab 259

A correct answer should include:

- 3 merges: a fresh byte-only tokenizer starts at vocab 256, and each merge adds exactly one entry, so `259 − 256 = 3`.

Common issues:

- Answering 259 (confusing target vocab with merge count).
- Subtracting from 255 instead of 256.

## Exercise 04.07 — Why new IDs start at 256

A correct answer should include:

- IDs 0–255 are reserved for the 256 single-byte base tokens; the first minted merge takes the next free integer, 256.
- (Implicitly or explicitly) this keeps every byte permanently in the vocab, which is what guarantees no-OOV encoding.

Common issues:

- Saying 256 is arbitrary or "just a convention" without the byte-base-vocab reason.

## Exercise 04.08 — Why byte-level handles unseen text

A correct answer should include:

- Every possible byte value 0–255 is in the base vocab, and any UTF-8 string is a byte sequence — so worst case, unfamiliar text falls back to raw single-byte tokens. Nothing is ever out-of-vocabulary.

Common issues:

- Claiming the tokenizer generalizes because merges "transfer" — the guarantee is the byte fallback, not the merges.
- Confusing "can encode it" with "encodes it efficiently" (unseen text just gets more, smaller tokens).

## Exercise 04.09 — Why lowest-new-ID merges win at encode

A correct answer should include:

- Lower ID = learned earlier = more frequent pattern; encode must replay merges in training order so new text tokenizes the same way the training corpus did.
- If a later (higher-ID) merge applied first, it could consume bytes an earlier merge needed, producing token sequences the model/vocab never saw during training.

Common issues:

- "Lower IDs are more frequent" without the replay-determinism consequence.
- Confusing encode-time priority with train-time most-frequent-pair selection (related but distinct loops).

## Exercise 04.10 — What makes decode lossless

A correct answer should include:

- Every vocab entry maps an ID to a fixed byte sequence: base IDs are single bytes, and each learned ID's bytes are exactly the concatenation of its parents' bytes.
- So concatenating `vocab[id]` over the sequence reconstructs the exact original byte stream, which UTF-8-decodes back to the original text. It's structural, not tested-per-input.

Common issues:

- Attributing losslessness to encode being deterministic (that's consistency, not invertibility).
- Missing the concatenation-of-parents invariant that makes learned tokens exact.

## Exercise 04.11 — Why `<|assistant|>` must be atomic

A correct answer should include:

- It's a control/interface marker, not prose: the model and runtime key on its exact single ID (chat-role boundaries, supervision masks, stopping in Module 13+).
- Split into ordinary BPE pieces its tokenization could vary with context, and the model couldn't reliably learn (or the runtime reliably detect) the marker; ordinary text that happens to contain the string must not be confusable with the real control token.

Common issues:

- "It's shorter as one token" — compression is not the reason; the interface contract is.
- Not connecting atomicity to any downstream consumer (chat format, masking, stopping).

## Exercise 04.12 — First learned merge ID with 8 special tokens

A correct answer should include:

- 264: bytes take 0–255, the 8 reserved specials take 256–263, so learned merges start at `base_vocab_size = 256 + 8 = 264`.

Common issues:

- Answering 256 (forgetting the reserved block).
- Placing specials after the learned merges instead of between bytes and merges.

## Exercise 04.13 — Why cross-whitespace merges are less useful

A correct answer should include:

- A merge spanning a word boundary (e.g., ending of one word + space + start of the next) is tied to one specific word *sequence*, so it recurs far less often and generalizes poorly.
- Within-word subwords (`"the"`, `"ing"`) are reusable across many contexts; word boundaries are where reusable frequency lives.

Common issues:

- Saying cross-whitespace merges are impossible rather than merely low-value (plain BPE will happily learn `"e t"`-style tokens without pre-tokenization).
- No generalization/reuse argument.

## Exercise 04.14 — What to compare around pre-tokenization

A correct answer should include:

- At least two concrete before/after comparisons, e.g.: the top pair counts (do any cross a boundary?), the learned vocab contents (cross-word tokens vs. within-word subwords), and compression (token count on a fixed passage at the same vocab size).

Common issues:

- "See if it's better" with no measurable quantity.
- Comparing at different vocab sizes, which confounds the compression comparison.

## Exercise 04.15 — Token count vs. vocab size

A correct answer should include:

- Token count for the fixed passage *decreases* as vocab grows (chars/token rises) — more learned merges means more text captured per token.

Common issues:

- Predicting the opposite direction, or predicting linear improvement (the next question is about why it flattens).

## Exercise 04.16 — Why diminishing returns

A correct answer should include:

- Merges are minted in frequency order: the earliest merges capture the most common patterns and save the most occurrences; later merges are progressively rarer and each fires fewer times on the passage.
- (Implicitly or explicitly) token/pattern frequencies are heavy-tailed (Zipf-like), so the wins shrink fast.

Common issues:

- Attributing the flattening to the passage being short rather than to the frequency distribution.
- No connection to merge ordering.

## Exercise 04.17 — The large-vocab tradeoff

A correct answer should include:

- Costs: a bigger embedding table and output projection/softmax (more parameters and compute per prediction), and rare tokens each get fewer training occurrences, leaving their embeddings undertrained.
- Benefit being traded: shorter sequences (more text per context window). Real LLMs settle in the 32k–200k range.

Common issues:

- Only naming the parameter cost, missing the rare-token/data-sparsity cost (or vice versa).
- Claiming a large vocab breaks correctness rather than efficiency/statistical strength.

## Exercise 04.18 — Why low learned IDs are common patterns

A correct answer should include:

- Training mints IDs in order of pair frequency: each step merges the currently most frequent pair, so the first learned IDs (just above the base vocab) captured the corpus's most common patterns.

Common issues:

- Saying low IDs are common "because they're short" — length correlates, but the ordering mechanism is frequency at mint time.

## Exercise 04.19 — Corpus-specific vs. generally useful tokens

A correct answer should include:

- A corpus-specific token is one tied to this corpus's particular content — e.g., a Shakespeare character name or stage-direction fragment (`"ROMEO"`, `"First Citizen"`) — useful here, useless on general text.
- Contrast with generally useful subwords like `"the"`, `"ing"`, `" of"` that recur in any English corpus.

Common issues:

- Defining corpus-specific as "long" — length is a symptom; the criterion is whether the pattern's frequency transfers to other corpora.

## Exercise 04.20 — Two early learned tokens

Run-dependent — grade against the student's own printed vocab window.

A correct answer should include:

- Two actual tokens from their `first learned tokens` output (typically short, high-frequency pieces: `"th"`, `"he"`, `" t"`, `"e "`, `" the"`-style fragments, often with a leading space).
- A frequency argument for each — common digraphs, function words, or space-plus-word starts in English/Shakespeare.

Common issues:

- Naming plausible tokens not actually in their output.
- No explanation of *why* the named pieces are frequent.

## Exercise 04.21 — One late learned token

Run-dependent — grade against the student's own printed vocab window.

A correct answer should include:

- An actual token from their `last learned tokens` output (typically long and Shakespeare-specific — a character name or multi-word fragment).
- What makes it specific: longer byte sequence, far fewer occurrences, minted only after all the commoner patterns were exhausted.

Common issues:

- Citing an early-style short token as "late."
- Describing it as rare without tying rarity to its late mint order.

## Exercise 04.22 — Why KING is one token but King is not

A correct answer should include:

- BPE mints tokens for whatever byte sequences are frequent in *this* corpus; Shakespeare's stage directions and speaker headers write names in capitals, so `KING` earns a merge chain while `King` stays fragmented.
- A web-trained tokenizer sees title-case far more often than all-caps, so the relationship inverts — the corpus, not the language, decides which surface form becomes "one word."

Common issues:

- Explaining it as a property of English capitalization rather than corpus statistics.
- Not noticing that the case variants get *unrelated* IDs — nothing links `lord` and `Lord` in the model's input representation.

## Exercise 04.23 — Why letter-counting is hard

A correct answer should include:

- The model receives token IDs, not characters; `strawberry` arrives as opaque symbols like `st / raw / ber / ry`, and no letter-level structure is visible in the input.
- Answering "how many r's" requires the model to have memorized each token's spelling as a fact, then aggregate across pieces — two error-prone steps for something trivial at character level.

Common issues:

- "The model is bad at counting" without the representation argument.
- Claiming the tokenizer *could not* encode letters (it can — byte fallback — but frequent words never arrive that way).

## Exercise 04.24 — Why production tokenizers force digit-splitting

A correct answer should include:

- Web-trained BPE learns merges for *frequent* numbers only, so `1234` might be one token while `1235` is two — arithmetic gets inconsistent building blocks and place-value structure is destroyed unpredictably.
- Forced digit-splitting makes every number decompose the same way, giving the model a uniform representation to learn arithmetic over.

Common issues:

- Saying digit-splitting compresses better (it compresses *worse*; the tradeoff is bought for consistency).
- Missing that TinyShakespeare digit-splits by accident (numerals are rare), not by design.

## Exercise 04.25 — Never-fired tokens and the glitch mechanism

A correct answer should include:

- Embedding rows for tokens absent from training data receive (almost) no gradient updates, so they remain near their random initialization — Module 03's "Where weights start" describes exactly what such a row is: noise at the right scale.
- At inference, that near-random vector is processed by layers trained on real distributions — the output is confidently structured nonsense, which is the mechanism behind glitch tokens like `SolidGoldMagikarp` (frequent in the tokenizer's training data, scrubbed from the model's).

Common issues:

- Treating glitch tokens as a bug in the tokenizer code rather than a tokenizer/model training-data mismatch.
- Saying the model "doesn't know" the token — it's stronger than that: the embedding was never trained, so downstream computation is on noise, not on a poorly-learned meaning.

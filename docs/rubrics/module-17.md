# Module 17 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/17-rag.ipynb`, falling back to `notebooks/clean/17-rag.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

Live-answer cells run against the student's chosen backend and corpus, so outputs vary. Grade cited retrieval results and prompts from their run and the correctness of the pipeline-stage reasoning.

## Exercise 17.01 — Chunk-size tradeoffs and overlap

A correct answer should include:

- Gain as `chunk_size` grows: fewer, more self-contained chunks with more usable context per hit. Loss: the embedding averages over more content, so vectors go generic and one distinctive fact gets diluted — retrieval misses specifics (and each hit drags more irrelevant text into the prompt).
- The overlap failure: with zero overlap, an answer-bearing sentence that crosses a chunk boundary is split so *neither* chunk contains it whole; overlap duplicates the boundary region into both chunks so the sentence survives intact in one of them.

Common issues:

- Framing the sweep as only "more chunks vs fewer chunks" without the embedding-dilution half.
- Explaining overlap as deduplication, compute savings, or "more context" rather than boundary insurance.
- Stride confusion: the step is `chunk_size - overlap`, not `overlap`.

## Exercise 17.02 — What HashEmbedder measures

A correct answer should include:

- Scores cited from their table, noting the statement-vs-question pair scores comparably to (usually higher than) the statement-vs-paraphrase pair, because "Madrid is the capital of Spain." and "What is the capital of Spain?" share the long literal character run "is the capital of Spain", while the paraphrase shares meaning but fewer contiguous n-grams.
- The conclusion: HashEmbedder measures surface character-n-gram overlap, not meaning — shared substrings cluster, paraphrases do not.

Common issues:

- Expecting the paraphrase to score highest and reporting the observed ordering as a bug.
- Describing HashEmbedder as capturing semantics.
- Not citing the actual cosine values from the table.

## Exercise 17.03 — Prompt with vs without context

A correct answer should include:

- With chunks: the numbered `[i] (source: ...)` context blocks ground the model in specific evidence, steer it toward answering from the context, and make citations checkable.
- Empty case: no evidence stands between the model and answering from parametric memory, so the trailing "I don't know" instruction is the only thing prompting abstention — and empty/insufficient retrieval is exactly when hallucination risk peaks.

Common issues:

- Claiming the empty-retrieval prompt *prevents* answering — the guard is an instruction the model usually follows, not a mechanism; models can still hallucinate past it.
- Missing that the instruction line is the single highest-leverage anti-hallucination lever in the template.
- Not comparing the two actually-printed prompts.

## Exercise 17.04 — Hash vs semantic retrieval

A correct answer should include:

- Where the semantic retriever clearly won, cited by retrieved sources/scores: questions phrased abstractly relative to the source wording (e.g., "What part of the course lets the assistant use information outside its weights?" shares almost no tokens with the RAG chunk), where the hash retriever needs literal n-gram overlap and misses.
- The mechanism: hash retrieval requires lexical overlap between question and chunk; the neural embedder maps paraphrases of the same idea to nearby vectors. Hash stays competitive when the question reuses the document's own words (e.g., the inference-optimization question).
- If only the hash retriever ran: the question with the least lexical overlap with its answer chunk named as the worst case, with that reasoning.

Common issues:

- No citation of which chunks each retriever actually surfaced.
- Explaining the gap as "the Ollama model is bigger/smarter" rather than lexical vs semantic matching.
- Concluding hash embedding is useless — it is fine for keyword-heavy queries.

## Exercise 17.05 — The unanswerable probe

A correct answer should include:

- The observed behavior — refused, hedged, or invented — with the model's actual answer quoted or summarized.
- The retrieval-score reading: whether the top scores for the Pluto question were low/flat compared to the answerable probes. A low best score is the advance warning that no relevant chunk exists — the signal you would check before trusting the generated answer.

Common issues:

- Crediting a refusal to the model's self-knowledge when the prompt's "I don't know" guard is doing the work.
- Treating an absolute score threshold as meaningful across embedders — only relative comparison within one embedder is.
- Not citing the scores from the probe table at all.

## Exercise 17.06 — Snapshot without embedder config

A correct answer should include:

- The silent failure: reload with a *different embedder at the same dim* (different model, seed, or ngram_range) loads and searches without error, but query vectors live in a different space than the stored vectors — retrieval returns garbage with no exception. Different `dim` at least fails loudly on shape mismatch.
- The guard: persist embedder identity (class/model id, dim, ngram_range, seed — or a config fingerprint) in the snapshot and validate it on load, refusing or triggering a rebuild on mismatch.

Common issues:

- Only naming the loud dim-mismatch case and missing the silent same-dim case, which is the dangerous one.
- Not recognizing that swapping embedders always requires re-embedding the corpus, not just re-pointing the query.
- A guard that "stores the embedder" wholesale rather than recording and checking its configuration.

## Exercise 17.07 — Weakest stage

A correct answer should include:

- One stage named with cited failures from their runs. With the hash embedder, embedding/retrieval misses typically dominate; generation hallucination on weak or empty context is the other common winner.
- An improvement that targets that stage: embedding → semantic embedder; retrieval → hybrid or reranking; chunking → natural-boundary/recursive splitting; prompt → stricter guard or better citation instructions; generation → stronger backend.

Common issues:

- Blaming the generator for a retrieval miss — if the wrong chunk was in the prompt, the model answering faithfully from wrong evidence is upstream failure.
- An improvement that does not address the stage named as weakest.
- A verdict with no cited failure from the notebook runs.

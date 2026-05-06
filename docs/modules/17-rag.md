# Module 17 — Retrieval-augmented generation

> **Question this module answers:** *How does the model use external knowledge it doesn't have memorized?*

![Module 17 on one page: a four-panel circus map of the RAG pipeline. PANEL 1 (top-left, "INDEX TIME"): a stack of source documents on the left — `notes.md`, `16-inference.md`, `attention.md` — feeding into a `chunk_text(...)` machine that emits a row of overlapping rectangles labeled `[1] start=0 end=1500`, `[2] start=1350 end=2850`, etc. Each chunk flows into a `HashEmbedder` / `OllamaEmbedder` box, which spits out a `(d=512)` vector with bars representing coordinate magnitudes. The vectors stack into a `NumpyVectorStore` (a tall (N, d) matrix). PANEL 2 (top-right, "QUERY TIME"): a user typing "What is the capital of Spain?" The same embedder embeds the question to a `(d,)` vector, which dots against every row of the store; an `argpartition(-k)` tape spits out the top-3 chunks, ranked, with cosine similarities `0.78`, `0.62`, `0.41`. PANEL 3 (bottom-left, "PROMPT TIME"): an `assemble_rag_prompt` machine takes the question + top-k chunks and builds a numbered-citation prompt: `"You are a helpful assistant... Context: [1] (source: es.md) Madrid is... [2] ... Question: What is the capital of Spain? If the context doesn't contain the answer, say I don't know."` Two arrows: one to a `LocalTransformerBackend` (your tiny model from Modules 1–14), one to an `OllamaBackend` (Llama 3.2 3B from Module 16). PANEL 4 (bottom-right, "ANSWER TIME"): the `RAGPipeline` returns a `RAGAnswer` containing the model's response ("Madrid is the capital, per [1]."), the retrieved chunks (so the user can verify), the assembled prompt (so a debugger can see what the model actually saw), and the underlying `InferenceResult` (latency, token counts). A right-edge sidebar lists key concepts: chunking strategies (size, overlap, semantic vs character), embedder taxonomy (sparse/dense/hybrid), cosine similarity as dot-of-unit-vectors, citation formatting, "I don't know" guards, and the Lewis et al. 2020 lineage. Bottom caption: "Modules 1–16 taught the model how to read and answer; Module 17 teaches it how to LOOK THINGS UP."](17-rag/Module17-Hero.png)

*The whole module on one page. RAG is the smallest possible architecture for "give the model access to information it doesn't have memorized." Chunk the corpus, embed each chunk, store the vectors, embed the query, retrieve the top-k by cosine similarity, splice them into a citation-formatted prompt, send to the inference backend. None of the components are individually deep — but the wiring is the lesson, because RAG is the substrate Module 18's tools and Module 19's agent loop both build on.*

## Prerequisites

Module 17 opens the second half of Phase V (assistant systems). Module 16 built the unified `Backend` interface; this module is the first downstream piece that uses it. Every component you build here — the chunker, the embedder, the vector store, the retriever, the prompt assembler — is going to keep showing up: the agent loop in Module 19 retrieves over conversation history; the capstone in Module 20 retrieves over your own notes.

This module is short on math. There's no new training loss, no new architecture, no gradient. The whole content is:

- A taxonomy of "how to turn text into vectors" (sparse vs dense, hashing vs neural, the cosine-similarity convention).
- The tradeoffs in chunk size and overlap.
- The pattern for splicing retrieved context into a prompt without breaking the model's instruction-following.
- A small Python interface that downstream modules can import without knowing whether the embedder is hash-based or neural.

The only code you write is the pipeline itself.

### Math

Three small ideas worth having in your head:

- **Cosine similarity reduces to a dot product on unit vectors.** For vectors `u, v` with `||u|| = ||v|| = 1`, `cos(u, v) = u · v`. Every embedder in this module L2-normalizes its rows; the vector store therefore searches with a plain dot product. If you ever drop the normalization on either side, the *ranking* still works (norms are positive scalars and don't change argmax order) but the absolute scores become uninterpretable.
- **Memory cost of a vector index.** A corpus of `N` chunks at dimension `d`, stored in float32, takes `N · d · 4` bytes. A real-world index of 10k chunks at 768-dim is 30 MB — fine. 1M chunks at 1024-dim is 4 GB — already worth thinking about quantizing the index. We don't optimize for this; for course-scale corpora the naive numpy array is fast enough.
- **Top-k via argpartition is `O(N)`, not `O(N log N)`.** `np.argpartition(scores, -k)` is the right call when `N` is large and `k` is small. The implementation is "quickselect with k pivots" — it finds the top-k indices in linear time without sorting the whole array. For tiny corpora where `k ≈ N`, just sort. The `NumpyVectorStore.search` recipe branches on this.

### Computer science

- **Chunking strategies.** Three families:
    - **Character / token sliding window.** Fixed-size windows with overlap. What this module implements. Easy to implement, dependency-free, robust to messy text. Loses paragraph and sentence structure.
    - **Recursive character splitting.** Try to split on `"\n\n"`, fall back to `"\n"`, fall back to `" "`, fall back to characters. Preserves natural breaks. LangChain's default. A small but real upgrade over the sliding window.
    - **Semantic chunking.** Embed sentences, group consecutive sentences whose embeddings are similar enough that they "belong together." Best quality; slowest; requires an embedder. Used by some agentic systems for indexing long technical documents.

- **Embedder taxonomy.** Three kinds, in order of pedagogical and practical importance:
    - **Sparse / lexical.** TF-IDF, BM25, our `HashEmbedder`. The vector is mostly zeros; non-zero entries correspond to terms (or n-grams, or hash buckets). Fast, deterministic, no model required. Captures lexical overlap, no semantics. Good for "find passages mentioning the same words" — bad for "find passages on the same topic."
    - **Dense / neural.** A model — typically a transformer encoder — turns each text into a fixed-dim float vector. `nomic-embed-text` (768-dim), `mxbai-embed-large` (1024-dim), `text-embedding-3-small` (1536-dim). Captures topic, paraphrase, and analogies. Requires a model. The substrate of every modern RAG system.
    - **Hybrid.** Sparse + dense scores combined (typically by reciprocal-rank fusion). Robust across query types — keyword-heavy queries that dense embedders mishandle ("what does CVE-2024-3094 do?") still hit when sparse retrieval matches the literal token.

- **Vector stores.** Two axes — what's indexed (flat vs ANN) and where it lives (in-memory vs on-disk):
    - **Flat in-memory.** What `NumpyVectorStore` is. Exact search, `O(N · d)` per query. Fast up to ~100k chunks; eats more memory past that.
    - **HNSW** (hierarchical navigable small world). Approximate search, sub-linear in `N`. Used by FAISS, hnswlib, Chroma's default backend. Builds a graph at index time; traverses it greedily at query time. ~99% recall at 10–100× speedup over flat for million-scale corpora.
    - **IVF-PQ** (inverted file with product quantization). Approximate search with quantized vectors. Used by Faiss for the largest corpora. Shrinks the index by a factor of 8–32× at the cost of some recall.
    - **Hosted services** (Pinecone, Weaviate, LanceDB, Qdrant). Same algorithms as above, behind an HTTP API, with persistence + replication. We don't use any of these; the in-memory store is enough.

- **Citation formatting in prompts.** Three competing conventions:
    - **Numbered brackets** (`[1]`, `[2]`). What this module uses. Compact, easy for humans to scan, easy for downstream parsers to extract.
    - **XML tags** (`<source id="1">...</source>`). Anthropic's recommended format for Claude. Verbose; the model ignores it less often.
    - **Markdown footnotes** (`[^1]`). Native to markdown rendering; rarer in LLM contexts.
    Whichever you pick, *be consistent*. The reason `[1]` works is that the model has seen millions of bracketed citations during pretraining and knows what to do with them.

- **The "I don't know" guard.** Every RAG prompt should end with an instruction telling the model what to do when the context is insufficient. Without it, the model hallucinates rather than abstaining — because pretraining rewarded fluent continuation, not honesty about the limits of its evidence. This single instruction is responsible for a non-trivial fraction of measurable RAG quality on factual benchmarks.

### Programming

- **`numpy` for the math.** `np.linalg.norm`, `@` (matrix multiply), `np.argpartition`, `np.argsort`. The four functions cover everything in this module.
- **`hashlib.blake2b`** for stable hashing. Python's built-in `hash()` is salted differently per process — different runs of the same code produce different hashes — so it's unusable for embeddings. BLAKE2b is fast, stdlib, and stable across runs.
- **`abc.ABC` + `@abstractmethod`** for the `Embedder` interface. Same pattern as `Backend` from Module 16.
- **`@dataclass(frozen=True)`** for `Chunk`, `RetrievedChunk`, and `RAGPrompt`. Values, not handles.
- **`urllib.request`** for the `OllamaEmbedder`'s HTTP call. Same recipe as Module 16's `OllamaBackend`. The `urlopen` injection point lets the tests run without a real Ollama daemon.

### What you can skip

- **Implementing HNSW or any other ANN index.** Production scale wants this; course scale does not. A few-hundred-chunk corpus searches in microseconds against a flat index. Building HNSW from scratch is a 200-line project and a different lesson.
- **Implementing a sentence-transformer-style dense embedder.** Training a contrastive bi-encoder is its own multi-week curriculum. We use `OllamaEmbedder` to plug into a pretrained one (`nomic-embed-text` is 137M parameters, trained by NomicAI on ~250M sentence pairs). Treat the embedder as a black box that turns text into a 768-vector.
- **BM25 / hybrid retrieval.** BM25 is a 1990s-era lexical scorer that consistently improves on dense-only retrieval for keyword-heavy queries. We don't build it; the `HashEmbedder` covers the lexical-side intuition without the BM25 IDF / saturation math. Exercise 6 walks through adding BM25 as a hybrid retriever.
- **Re-ranking with a cross-encoder.** Production RAG pipelines often retrieve top-k=50 with a cheap dense embedder and then re-rank top-k=5 with an expensive cross-encoder (`bge-reranker-v2-m3` etc.). Adds ~100 ms per query and 1–3% retrieval accuracy. We skip — the re-ranker is its own model, and the lesson is the pipeline shape.
- **Persistence.** A real index lives on disk so you don't re-embed the corpus on every restart. `NumpyVectorStore` is in-memory only; `np.save` / `np.load` on `store.vectors` plus pickle for `store.chunks` is the obvious one-screen solution. Exercise 4 walks through it.
- **Async embedding.** Embedding 10k chunks against a remote service one-at-a-time takes minutes. Real pipelines batch via async. We don't — the lesson is the math, not the I/O.

## Why we start here

Modules 1–16 built a model and made it usable. The tiny model from Module 14 doesn't know facts — Module 16 fixed that by pivoting to a real pretrained model behind a unified `Backend`. But even a 7B-class model has gaps:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  WHAT EVEN A 7B MODEL DOESN'T KNOW                                    │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   • Anything not in its training data:                                │
   │       - your private notes, your company's wiki, last week's news    │
   │       - documents written after the model's training cutoff           │
   │       - long-tail factual claims it saw 0–3 times in pretraining     │
   │                                                                       │
   │   • Anything it memorized BADLY:                                      │
   │       - fine details of a specific paper it saw once                  │
   │       - exact phrasings, exact dates, exact numbers                   │
   │       - the line of code in `g2c/sampling/generate.py` line 47        │
   │                                                                       │
   │   • Anything it would need to update mid-conversation:                │
   │       - "given this document, answer X" workflows                    │
   │       - "the user just told me their cat's name, remember it"        │
   │                                                                       │
   │   These are the gaps RAG fills. Not by training the model on more     │
   │   data — by retrieving the relevant text at QUERY TIME and putting    │
   │   it into the prompt. The model doesn't memorize new facts; it gets   │
   │   to read a relevant passage right before it answers.                 │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The mechanism is simple: turn the user's question into a vector, search a pre-indexed library of text chunks for the closest matches, paste those chunks into the model's prompt, then ask the model to answer using only what's in front of it. The model doesn't learn anything new — it just gets handed the right page of the book.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  THE PIPELINE — TWO PHASES                                            │
   └──────────────────────────────────────────────────────────────────────┘

      ┌────────────────────────────────┐
      │  INDEX TIME (once per corpus)   │
      │                                 │
      │   docs/  →  chunk_text  →       │
      │     [Chunk, Chunk, Chunk, ...] →│
      │       embedder.embed   →        │
      │         (N, d) vectors  →        │
      │           store.add             │
      │                                 │
      │   Cost: O(N) embedding calls.   │
      │   Done once; persisted.         │
      └─────────────┬───────────────────┘
                    ▼
      ┌────────────────────────────────┐
      │  QUERY TIME (once per question) │
      │                                 │
      │   question   →                  │
      │     embedder.embed → (1, d)     │
      │       store.search →             │
      │         top-k (Chunk, score)     │
      │           assemble_rag_prompt → │
      │             prompt              │
      │               backend.complete →│
      │                 RAGAnswer       │
      │                                 │
      │   Cost: 1 embed + 1 search +    │
      │         1 generate. ~hundreds   │
      │         of ms total.            │
      └────────────────────────────────┘
```

Phase 1 happens once. Phase 2 happens on every user question. The asymmetry is the point: indexing is the expensive amortized step that makes querying cheap.

## The big idea

### Chunking — what to retrieve

A document is too big to retrieve as a unit (a 50-page paper is one "topic" but you only want the paragraph that actually answers the question). A sentence is too small (the answer often spans 2–3 sentences). The chunk is the right intermediate — small enough to be specific, big enough to be self-contained.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   CHUNK SIZE TRADEOFFS                                                │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Size       Pros                          Cons                       │
   │   ────       ────                          ────                       │
   │   100 char   Highly specific embedding;    Answers spanning 2 chunks  │
   │              one chunk = one fact          require fetching both;     │
   │                                            no inter-sentence context  │
   │                                                                       │
   │   500 char   Balanced; 1–3 sentences       Generic-feeling vectors    │
   │              of context                                                │
   │                                                                       │
   │   1500 char  Self-contained paragraphs;    Embedding "averages" over  │
   │              good for "what does this      topic; one distinctive    │
   │              section say"                  sentence gets diluted     │
   │                                                                       │
   │   3000+ char Whole sections preserved      Embedding is a coarse     │
   │              for the model to read         summary; retrieval misses │
   │                                            specific facts            │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

Overlap exists for exactly one reason: an answer-bearing sentence that crosses a chunk boundary appears in BOTH chunks rather than being split between them. Without overlap, the chunker's split points become noise — moving a paragraph break a few characters drops the answer's recall.

```
   No overlap, answer split at the boundary:
     │ ... background context | The capital of Spain  │ is Madrid. More text ...│
     └────── chunk 1 ─────────┴────────── chunk 2 ──────┴──────── chunk 3 ──────┘
     Embedding chunk 2 has "The capital of Spain" but not "Madrid".
     Embedding chunk 3 has "Madrid" but not "The capital of Spain".
     The Spain-capital question retrieves NEITHER chunk well.

   With ~20% overlap:
     │ ... background context | The capital of Spain ...│
     └────── chunk 1 ─────────┴────────── chunk 2 ──────┘
                                        │ ...al of Spain is Madrid. More text ...│
                                        └──────── chunk 3 ──────────────────────┘
     Chunk 3 contains the full "...of Spain is Madrid" — retrieval works.
```

This module's chunker is the simplest possible: fixed-size sliding window with overlap. It ignores sentence and paragraph boundaries. A real chunker prefers natural breaks; the `chunk_text` recipe is intentionally minimal so the lesson is the math.

### Embedding — turning text into coordinates

![Embedding space and cosine search. Left half: a 2D scatter of chunk embeddings color-coded by cluster — Spain/Cities (green), Python/Code (blue), Cooking (orange), Machine Learning (purple). Texts about the same topic land in the same neighborhood. The user's query "What is the capital of Spain?" embeds to a point near the Spain/Cities cluster, marked with a star. A "cosine similarity intuition" panel pins the math: for L2-normalized vectors, `cos(u, v) = u · v` — small angle → high similarity → score near 1; orthogonal → score near 0; opposite → score near -1. Right half: the search algorithm in five steps. Step 1 — embed the query. Step 2 — dot the query against every row of the (N, d) store matrix to get N similarities. Step 3 — `np.argpartition(scores, -k)` finds the top-k indices in O(N) without sorting the full array. Step 4 — sort just those k indices by descending score. Step 5 — return the top-k chunks plus their similarity scores. A "key takeaway" panel: cosine similarity finds the chunks whose meaning is most similar to the query — not the chunks that share words.](17-rag/Module17-Embedding.png)

*The geometric picture for `NumpyVectorStore.search`. The five-step recipe IS the implementation: embed query, dot against store, argpartition for top-k, argsort the k indices, return chunks + scores. The test `test_search_returns_descending_order` pins the sort direction; reading off this image, the "most similar first" convention matches the natural reading direction of the panel.*

An embedding is a function from `string → R^d`. Two strings are "similar" if their embeddings have a small angle between them. The choice of embedding function determines what "similar" means:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   EMBEDDER COMPARISON                                                 │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   HashEmbedder:                                                       │
   │     "Madrid"            ≈  "Madrid"            (identical, sim=1)    │
   │     "Madrid"            ≈  "madrid"            (lowercase, sim=1)     │
   │     "Madrid"            ≈  "Madrideño"         (substring, sim=0.4)   │
   │     "Madrid"            ≈  "the capital city"  (no overlap, sim≈0)    │
   │     "running"           ≈  "runs"              (shared "run", sim=0.3)│
   │                                                                       │
   │   OllamaEmbedder (nomic-embed-text):                                  │
   │     "Madrid"            ≈  "the capital of Spain"   (semantic, sim=0.5)│
   │     "running"           ≈  "jogging"                 (paraphrase, sim=0.6)│
   │     "the cat sat"       ≈  "the dog stood"          (similar shape, sim=0.4)│
   │     "the cat sat"       ≈  "running on a treadmill" (different topic, sim=0.05)│
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

`HashEmbedder` is what you can build in a hundred lines of stdlib. It captures lexical signal — strings sharing tokens / n-grams cluster together. It's enough to make the pipeline work for tests, and enough to feel the limits: a question phrased differently from the source document doesn't retrieve. `OllamaEmbedder` (or any neural embedder) captures semantics — the same idea expressed in different words still clusters. **For real retrieval over heterogeneous queries, you want a neural embedder.** The `HashEmbedder` exists because (1) it teaches what an embedding even is, and (2) a fully-stdlib pipeline is testable without an external service.

### Retrieval — the math is one line

Given a query embedding `q` and a stored matrix of chunk embeddings `V` (shape `(N, d)`), all of which are L2-normalized:

```
   similarities = V @ q         # shape (N,) — one cosine per chunk

   top_k_indices = argpartition(similarities, -k)[-k:]    # O(N)
   top_k_sorted  = top_k_indices[argsort(-similarities[top_k_indices])]

   results = [(chunks[i], similarities[i]) for i in top_k_sorted]
```

That's the whole search algorithm for a flat index. ~five lines of numpy. The performance work — HNSW, IVF, GPU indexes — replaces the `V @ q` line with something sub-linear in `N`. For our scale, `V @ q` is fast.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   FLAT vs ANN — WHEN TO CARE                                          │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     Corpus size    Flat search    HNSW    Notes                      │
   │     ──────────     ───────────    ────    ─────                       │
   │     100            < 1 ms         < 1 ms  flat is simpler             │
   │     10k            ~5 ms          ~1 ms   flat is fine                │
   │     100k           ~50 ms         ~2 ms   HNSW starts winning         │
   │     1M             ~500 ms        ~3 ms   flat is too slow            │
   │     10M            ~5 s           ~5 ms   ANN required                │
   │                                                                       │
   │   "Module 17 corpus" — your local docs/ — is in the 100–1k chunk      │
   │   range. Flat search is microseconds. Worry about ANN only when        │
   │   you've outgrown the laptop.                                         │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

### Prompt assembly — the format matters

![Prompt assembly — turn retrieved chunks into a useful prompt. Top-left: input from the retriever — three top-k chunks with scores 0.84, 0.78, 0.62 and source labels (`design.md`, `notes.md`, `research.pdf`). Center: the assembled RAG prompt as a single concatenated string with four sections — system/context header (a helpful-assistant persona); context (the three retrieved chunks, each prefaced with `[1]`, `[2]`, `[3]` plus a `(source: ...)` label); the user's question; trailing instructions ("Use only the provided context to answer," "Cite your sources using bracketed numbers," "If the answer is not in the context, say I don't know"). Top-right: the model's output — a grounded answer that quotes specific chunks by number ("We recommend ~20% chunk overlap [1]. Overlap prevents answer-bearing sentences from being split..."). Bottom: a "common mistakes to avoid" row (no citations, too much context, unlabeled sources, missing instructions) and a "good defaults" row (chunk size 500–800 chars, k=3–6, citations [1] [2] [3], the "I don't know" guard). A "the takeaway" panel: prompt assembly is the bridge between retrieval and generation; good structure plus clear instructions plus citations equals answers you can trust.](17-rag/Module17-Prompt.png)

*The reference card for `assemble_rag_prompt`. The four-section template (system → context → question → instructions) plus the 1-based citation scheme is what the test `test_assemble_rag_prompt_uses_one_based_citations` pins. Exercise 5 (probing failure modes on unanswerable questions) measures how well the "I don't know" guard at the bottom of the prompt actually triggers refusal — it's the single highest-leverage line in the whole template.*

Once the retriever has handed you `[chunk_1, chunk_2, ..., chunk_k]`, the question is how to splice them into the model's input. The default template:

```
   {DEFAULT_SYSTEM}

   Context:
   [1] (source: {chunks[0].source})
   {chunks[0].text}

   [2] (source: {chunks[1].source})
   {chunks[1].text}

   ...

   Question: {question}

   {DEFAULT_INSTRUCTION}
```

Three things this format is doing:

1. **The numbered brackets `[i]`** give the model and any downstream parser a way to refer to a specific chunk. The model often produces answers like `"As described in [1], Madrid is the capital."` — verifiable.
2. **The `(source: ...)` label** lets the model name the document by its filename or URL when answering. Without it, citations can only be `[1]` — less useful.
3. **The trailing `instruction`** is the "I don't know" guard. Without it, models hallucinate when context is insufficient. With it, instruction-tuned models will (more often than not) actually abstain.

Note what's NOT in the template: chat-template markers like `<|user|>...<|assistant|>`, role tags, system prefixes like `system: `. Those belong to the chat template, which lives outside the RAG layer. The RAG prompt assembler produces the *body* of the user turn; whoever's calling the backend wraps it in the chat template their backend expects.

### The unified RAG interface

Three classes plus one helper, all in `g2c/rag/`:

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │   g2c/rag/  PUBLIC API                                                │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Chunk(text, source, start, end, metadata)                           │
   │     a slice of source document                                        │
   │                                                                       │
   │   chunk_text(text, *, source, chunk_size, chunk_overlap, metadata)    │
   │     -> list[Chunk]                                                    │
   │                                                                       │
   │   Embedder (abstract)                                                 │
   │     .dim          → int                                               │
   │     .embed(texts) → (n, dim) float32, L2-normalized rows              │
   │                                                                       │
   │   HashEmbedder(*, dim, ngram_range, seed)                             │
   │     stdlib character-n-gram hashing                                   │
   │                                                                       │
   │   OllamaEmbedder(model_id, *, dim, base_url, timeout, urlopen)        │
   │     wraps Ollama's /api/embeddings                                    │
   │                                                                       │
   │   NumpyVectorStore(*, dim)                                            │
   │     .add(chunks, vectors)                                             │
   │     .search(query, *, k) → list[(Chunk, score)]                       │
   │                                                                       │
   │   DenseRetriever(embedder, store)                                     │
   │     .retrieve(query, *, k) → list[RetrievedChunk]                     │
   │                                                                       │
   │   assemble_rag_prompt(question, chunks, *, system, instruction)       │
   │     -> RAGPrompt                                                      │
   │                                                                       │
   │   RAGPipeline(retriever, backend, *, system, instruction)             │
   │     .answer(question, *, k, max_new_tokens, ...) → RAGAnswer          │
   │                                                                       │
   └─────────────────────────────────────────────────────────────────────┘
```

Total scaffolded code: roughly 60 lines spread across four method bodies (`chunk_text`, `HashEmbedder.embed`, `NumpyVectorStore.search`, `assemble_rag_prompt`). Everything else — including `OllamaEmbedder.embed`, `DenseRetriever.retrieve`, and `RAGPipeline.answer` — is pre-implemented because the wiring isn't the lesson; the components are.

## Concepts to internalize

- **The vector is not the meaning, it's a *coordinate for* the meaning.** Two strings whose embeddings are close are similar *in the geometry of that embedder*. A different embedder gives you different geometry. There's no "ground truth" embedding space — only embedders that are useful for specific retrieval tasks.
- **Retrieval quality dominates RAG quality.** A 7B model handed the right chunk answers correctly; the same model handed the wrong chunk hallucinates fluently. Improving retrieval (better chunker, better embedder, hybrid scoring, re-ranking) buys more than improving the model. This is why production RAG stacks invest more in retrieval than in generation.
- **The "I don't know" guard is the single highest-leverage line in the prompt.** Models that have it abstain on insufficient context; models that don't, hallucinate. One sentence of prompting is worth several percent of measurable accuracy on factual benchmarks.
- **Chunks should be self-contained.** A chunk that ends mid-sentence forces the model to hallucinate the missing context. A chunk that ends at a sentence boundary doesn't. Real chunkers prefer natural breaks for this reason; the sliding-window chunker we build here is a starting point, not a destination.
- **Citations are alignment, not formatting.** The point of `[1] (source: foo.md)` isn't pretty output — it's that a downstream system (or a human reviewer) can verify whether the model's claim is actually supported by chunk `[1]`. Citation-without-verification is just decoration.
- **The interface is the substrate.** Module 19's agent loop will call `pipeline.answer(question, k=5)`. The pipeline doesn't care whether the embedder is `HashEmbedder`, `OllamaEmbedder`, or some future `MLXEmbedder` — same `embed(texts)` contract, same downstream behavior. This is the pattern Module 16 established with `Backend`; Module 17 extends it to `Embedder`.
- **In-memory is enough for course scale.** A flat numpy array searches a few-thousand-chunk corpus in microseconds. ANN indexes solve a problem we don't have. Build the simple thing; upgrade only when you've measured that you need to.
- **Embedders and chat models are different beasts.** A chat model `complete(prompt) → completion` expects to generate. An embedder `embed(text) → vector` expects to *score* — no decoder, no sampling, no autoregression. Confusing them (e.g., trying to use a chat model as an embedder) is a classic early mistake. They share the same family of architectures (transformers) but diverge entirely in usage.

## Scaffolding and how to run the tests

This module ships seven files in `g2c/rag/`:

- **`chunk.py`** — the `Chunk` dataclass plus `chunk_text`. Boilerplate (the dataclass): implemented. Logic (`chunk_text`): **scaffolded.**
- **`embed.py`** — `Embedder` ABC, `HashEmbedder` (boilerplate + **scaffolded** `embed`), `OllamaEmbedder` (fully implemented), `OllamaEmbedError`.
- **`store.py`** — `NumpyVectorStore` (boilerplate + **scaffolded** `search`) plus the `cosine_similarity` helper.
- **`retrieve.py`** — `DenseRetriever` and `RetrievedChunk`. Fully implemented.
- **`prompt.py`** — `RAGPrompt`, `assemble_rag_prompt` (**scaffolded**), default templates.
- **`pipeline.py`** — `RAGPipeline` and `RAGAnswer`. Fully implemented.
- **`__init__.py`** — public exports.

Tests live in `tests/test_rag.py`. Initial state on `main`: 73 tests pass (boilerplate + the components that are fully implemented). 69 tests fail with `NotImplementedError` until you fill in the four scaffolded methods.

```bash
pytest tests/test_rag.py                          # all module-17 tests
pytest tests/test_rag.py -x                       # stop at first failure
pytest tests/test_rag.py -k Chunk                 # chunker tests
pytest tests/test_rag.py -k HashEmbedder          # hash embedder tests
pytest tests/test_rag.py -k Search                # vector store search tests
pytest tests/test_rag.py -k Prompt                # prompt assembly tests
pytest tests/test_rag.py -k Pipeline              # end-to-end pipeline tests
pytest tests/test_rag.py -k Integration           # full-pipeline smoke
pytest tests/test_rag.py -v                       # verbose
```

Implementation order — four independent steps:

  1. **`chunk_text`** in `g2c/rag/chunk.py`. Sliding window with overlap. Pure string slicing — no numpy, no HTTP, no model. Easiest place to start. Turns green: `TestChunkText`, `TestChunkTextEdgeCases`.

  2. **`HashEmbedder.embed`** in `g2c/rag/embed.py`. Hash character n-grams into bucket counts; L2-normalize each row. Tests don't require any external service. Turns green: `TestHashEmbedderEmbed`.

  3. **`NumpyVectorStore.search`** in `g2c/rag/store.py`. Dot product + top-k via `np.argpartition` / `np.argsort`. Turns green: `TestNumpyVectorStoreSearch`, `TestVectorStoreSearchEdgeCases`.

  4. **`assemble_rag_prompt`** in `g2c/rag/prompt.py`. Splice question + chunks + system + instruction into a numbered-citation prompt. Pure string construction. Turns green: `TestAssembleRAGPrompt`, `TestAssembleRAGPromptEdgeCases`.

The four steps are independent — work in any order. The integration tests (`TestIntegrationSmoke`, `TestRAGPipelineAnswer`) and the retriever tests (`TestDenseRetrieve`) turn green automatically once all four are filled in, because they exercise the wiring around the components.

Headline tests to watch:

- **`test_chunk_text_overlap_correct`** — pins the stride math: each chunk starts `chunk_size - chunk_overlap` characters past the previous. Off-by-one here breaks every retrieval downstream.
- **`test_chunk_text_covers_full_document`** — the chunker must reach the end. A bug that stops one stride short silently loses the last paragraph of every document.
- **`test_hash_embedder_rows_are_unit_norm`** — the L2-normalization contract. Without it, `NumpyVectorStore.search`'s dot-product trick computes the wrong similarity.
- **`test_search_returns_descending_order`** — top-1 is the most similar. A reversed sort silently degrades retrieval to anti-retrieval.
- **`test_assemble_rag_prompt_uses_one_based_citations`** — humans cite `[1]`, not `[0]`. Forgetting `start=1` confuses the model AND any citation parser.
- **`test_pipeline_propagates_retrieved_chunks_to_prompt`** — wire-bug catcher: the chunks the retriever returned MUST be the chunks the prompt cites. A confused implementation could embed the question, retrieve, then assemble a prompt against a different chunk list.

## What you'll build

Package: `g2c/rag/`

```python
# chunk.py
@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)        # implemented

def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:                                                  # SCAFFOLDED
    ...


# embed.py
class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...                                      # implemented
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...           # implemented

class HashEmbedder(Embedder):
    def __init__(self, *, dim: int = 512,                          # implemented
                 ngram_range: tuple[int, int] = (3, 5),
                 seed: int = 0) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray:               # SCAFFOLDED
        ...

class OllamaEmbedder(Embedder):
    def __init__(self, model_id="nomic-embed-text", *, dim=768,    # implemented
                 base_url=DEFAULT_OLLAMA_URL, timeout=120.0,
                 urlopen=None) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...           # implemented


# store.py
class NumpyVectorStore:
    def __init__(self, *, dim: int) -> None: ...                   # implemented
    def add(self, chunks, vectors) -> None: ...                    # implemented
    def search(self, query, *, k=5) -> list[tuple[Chunk, float]]:  # SCAFFOLDED
        ...

def cosine_similarity(a, b) -> float: ...                          # implemented


# retrieve.py
@dataclass(frozen=True)
class RetrievedChunk:                                              # implemented
    chunk: Chunk
    score: float
    rank: int

class DenseRetriever:                                              # implemented
    def __init__(self, embedder, store): ...
    def retrieve(self, query, *, k=5) -> list[RetrievedChunk]: ...


# prompt.py
@dataclass(frozen=True)
class RAGPrompt:                                                   # implemented
    text: str
    question: str
    chunks: tuple[Chunk, ...]

def assemble_rag_prompt(question, chunks, *,                       # SCAFFOLDED
                        system=DEFAULT_SYSTEM,
                        instruction=DEFAULT_INSTRUCTION,
                        ) -> RAGPrompt:
    ...


# pipeline.py
@dataclass
class RAGAnswer:                                                   # implemented
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    prompt: RAGPrompt
    inference: InferenceResult
    metadata: dict[str, Any] = field(default_factory=dict)

class RAGPipeline:                                                 # implemented
    def __init__(self, retriever, backend, *,
                 system=DEFAULT_SYSTEM,
                 instruction=DEFAULT_INSTRUCTION) -> None: ...
    def answer(self, question, *, k=5, max_new_tokens=256,
               temperature=0.2, top_k=None, top_p=None,
               ) -> RAGAnswer: ...
```

Total scaffolded code: roughly 60 lines across four function bodies. Everything else is pre-implemented because the lesson is the math (chunking, embedding, cosine similarity, prompt construction), not the orchestration.

## Exercises

These exercises require Ollama running with both an inference model AND an embedding model:

```bash
ollama pull llama3.2:3b           # the chat model from Module 16
ollama pull nomic-embed-text       # the embedding model
ollama serve                       # if not already running
```

The exercises are written assuming `llama3.2:3b` for inference and `nomic-embed-text` for embeddings; substitute as needed for your hardware budget.

1. **Index your own notes.** Pick a corpus you actually care about — your `docs/` directory, your meeting notes, your bookmarked links exported as text, the course's `docs/modules/` directory. Write a small script that walks the directory, calls `chunk_text` on each `.md` file, and indexes everything into a `NumpyVectorStore` using `OllamaEmbedder`. Report:
    - Total documents.
    - Total chunks (and chunks-per-document distribution).
    - Total embedding wall time and tokens-per-second equivalent (via `time.perf_counter` around the indexing loop).

   Then run a few hand-authored questions and inspect the top-3 retrieved chunks for each. Are the results obviously relevant? Where does retrieval fail?

2. **Compare HashEmbedder vs OllamaEmbedder on the same corpus.** Index your corpus twice — once with `HashEmbedder(dim=512)`, once with `OllamaEmbedder(dim=768)`. For 10 questions, retrieve top-3 from each and tabulate:
    - Where does HashEmbedder match OllamaEmbedder?
    - Where does it differ? (Lexical-overlap questions where both should agree; paraphrase questions where the hash embedder fails.)
    - On a question whose phrasing differs from the source document, does the hash embedder retrieve the right chunk at all?

   This exercise is the empirical demonstration of "lexical vs semantic" — feel the gap.

3. **Build the end-to-end RAG chatbot.** Wire your `OllamaEmbedder` + `NumpyVectorStore` (loaded from Exercise 1) + `DenseRetriever` + `OllamaBackend("llama3.2:3b")` into a `RAGPipeline`. Build a tiny CLI:

   ```python
   while True:
       q = input("? ")
       if not q.strip():
           break
       ans = pipeline.answer(q, k=5, max_new_tokens=256, temperature=0.2)
       print(ans.answer)
       print()
       print("Sources:")
       for r in ans.retrieved:
           print(f"  [{r.rank}] (score={r.score:.3f}) {r.chunk.source}")
   ```

   Run a 10-question session. Try: questions answerable by your corpus, questions that aren't, questions about facts your corpus contains but phrased very differently. Save the transcript.

4. **Persist the index.** `NumpyVectorStore` is in-memory — a restart drops everything. Write `save_store(store, path)` and `load_store(path)` helpers. Use `np.save` for `store.vectors` and pickle (or `json` + a Chunk-rehydrator) for `store.chunks`. Add tests for round-tripping. Now your indexing is amortized across runs: re-embedding the corpus is the slow step, and you only pay it when the corpus changes.

5. **Probe failure modes.** Construct three categories of question:
    - **Answerable from the corpus.** Should retrieve the right chunk and produce a grounded answer.
    - **Answerable but phrased adversarially.** "What's the city in Spain that has soccer teams?" when the source says "Madrid is the capital of Spain and home to Real Madrid." A dense embedder should retrieve; a hash embedder might not.
    - **Not answerable from the corpus.** "What's the population of Pluto?" The model should say "I don't know based on the provided context" — but might hallucinate.

   Build a 30-question test set across these three buckets. Report the model's behavior in each (correct / partially correct / hallucinated / refused). The "hallucination on unanswerable questions" rate is your RAG pipeline's calibration. Lower is better.

6. **Hybrid retrieval (BM25 + dense).** Implement a tiny BM25 scorer (the function is ~30 lines: term frequencies, IDFs, document-length normalization) and combine its scores with `DenseRetriever`'s via reciprocal-rank fusion (`score = 1/(60+rank_dense) + 1/(60+rank_bm25)`). Re-run Exercise 5's question set and compare hybrid retrieval to dense-only. On keyword-heavy queries (named entities, code identifiers, exact numbers), hybrid should noticeably win.

7. **Re-rank with a cross-encoder.** Pull a small cross-encoder via Ollama or HuggingFace (`bge-reranker-v2-m3` is a popular option, ~560 MB). Retrieve top-20 with the dense embedder, then re-rank top-5 with the cross-encoder. Wrap as a `RerankRetriever(base, reranker)` that subclasses `DenseRetriever`. Measure: how many cross-encoder calls per query, what's the latency cost, and does retrieval accuracy improve on Exercise 5's adversarial bucket?

8. **Smarter chunker.** The `chunk_text` you built is a fixed-size sliding window. Write `chunk_text_recursive(text, *, source, chunk_size, chunk_overlap, separators=["\n\n", "\n", " ", ""])` that recursively tries each separator: split on the strongest available, then if any sub-chunk is still too big, recurse with a weaker separator. This is LangChain's `RecursiveCharacterTextSplitter` algorithm. Compare retrieval quality to the sliding window on a markdown-heavy corpus.

9. **The deliverable: RAG post-mortem.** Write 3–4 paragraphs in `docs/rag-postmortem.md` covering:
    - **What you indexed.** Corpus, chunk size / overlap, embedder, dim, vector count.
    - **What worked.** Question types where retrieval reliably succeeded.
    - **Where it broke.** Question types where retrieval failed; specific examples; whether the cause was chunking, embedding, or model abstention.
    - **What you'd build next.** Hybrid retrieval? Re-ranking? A smarter chunker? A larger embedder? Justify the next investment in 2–3 sentences.

   This is the actual deliverable. The pipeline code is the starting point; the *characterization* of where it works and where it fails is what you keep.

## Pitfalls to expect

- **Forgetting the L2-normalization in `HashEmbedder.embed`.** `NumpyVectorStore.search`'s dot-product trick assumes unit-norm vectors. Without normalization, the *ranking* is preserved (norms are positive scalars; argmax is stable) but the absolute scores become uninterpretable, and the cosine similarity is wrong by a factor of the norms. Failure mode: scores look "high" or "low" with no clear meaning.

- **Off-by-one in the chunker stride.** `start += chunk_size - chunk_overlap` is right. `start += chunk_size` skips the overlap entirely. `start += chunk_overlap` makes near-zero progress and creates a near-infinite list. The test `test_chunk_text_overlap_correct` pins the right value; if it's wrong, stop and re-derive.

- **Forgetting the `if end == len(text): break` exit condition.** Without it, the loop emits a chunk that goes past the end, then advances `start` past `len(text)` and exits — but produces a final chunk whose `end` is `start + chunk_size` (greater than `len(text)`). The Chunk constructor rejects this. Failure mode: the chunker raises on every doc longer than one chunk.

- **0-based vs 1-based citations.** The test pins 1-based (`[1]`, `[2]`, `[3]`). `enumerate(chunks)` defaults to 0-based; you must pass `start=1`. Forgetting silently produces a `[0]` in the prompt — which the model will sometimes parrot ("...as cited in [0]...") and sometimes fix.

- **Confusing prompt index vs retriever rank.** Both are 1-based, both correspond chunk-by-chunk, but they live in different objects. `RetrievedChunk.rank` is the retriever's ordering; the citation index `[i]` is the prompt's ordering. As long as `assemble_rag_prompt` iterates `chunks` in order, the two agree. Don't hand-mix — let the order propagate cleanly.

- **`np.argpartition(scores, -k)` returns UNSORTED indices.** The top-k indices are in the last k positions of the result, but in arbitrary order. You must `argsort` those k entries before returning. Failure mode: the top-k chunks are correct, but their order within the k is meaningless — the "top result" might not be the most similar one. Subtle but real.

- **`np.argsort` defaults to ascending.** For "most similar first," reverse the order: `np.argsort(sims)[::-1]` or `np.argsort(-sims)`. The test `test_search_returns_descending_order` pins this; if it fails, you have anti-retrieval.

- **Embedding the query and the corpus with different embedders.** They must be the same embedder (or at least the same model) — vectors from different embedders aren't comparable. The retriever uses one embedder for both. If you ever swap embedders mid-pipeline, you must re-embed the whole corpus.

- **Calling `OllamaEmbedder` against a chat model tag.** `nomic-embed-text` is an embedding model; `llama3.2:3b` is a chat model. `OllamaEmbedder("llama3.2:3b")` will fail at request time — Ollama returns a 400 because the model doesn't have an embedding head. The error wraps as `OllamaEmbedError`. If you're confused why the embed call is failing, check `ollama list` for an `*-embed-*` tag.

- **Wrong dim in the OllamaEmbedder constructor.** `nomic-embed-text` is 768-dim. If you instantiate `OllamaEmbedder(dim=384)` and then call it, the response will have 768-dim vectors but you'll get an `OllamaEmbedError("returned a 768-dim vector for ...; OllamaEmbedder was configured with dim=384")`. The fix is to look up the model's actual dim and pass it.

- **Empty `chunks` to `assemble_rag_prompt`.** Allowed but pathological — the prompt has `Context:\n\nQuestion: ...` with literally no context. The model is told "answer using ONLY the context" but given none. A well-behaved instruction-tuned model defaults to "I don't know"; many will hallucinate. Decide upstream whether to short-circuit (return "no context found") before assembling.

- **Mixing `Chunk` and `RetrievedChunk` in `assemble_rag_prompt`.** The `_coerce_chunks` helper handles both — caller can pass either. But if you pass something that's NEITHER, you get `TypeError("chunks must contain Chunk or RetrievedChunk, ...")`. Common cause: passing the raw `(chunk, score)` tuples from `NumpyVectorStore.search` directly. Wrap in `RetrievedChunk` first, OR strip to just the chunks.

- **The HashEmbedder produces zero rows for very short strings.** If your `ngram_range` is `(3, 5)` and the input is `"ab"` (2 chars), there are no n-grams of length ≥ 3 — the row stays zero. Cosine similarity against a zero row is 0. Failure mode: extremely short chunks (or chunks of mostly punctuation) retrieve poorly. The fix: lower the `ngram_range` floor, or filter out very short chunks at index time.

- **The chunker's `metadata` dict shared across all chunks.** Without `dict(metadata)` on each Chunk creation, all chunks share a reference to the same dict. A caller mutating `chunks[0].metadata` then sees the change in `chunks[1].metadata`. Defensive copy at chunk-creation time prevents this.

- **`Chunk` is frozen but `Chunk.metadata` is not.** `dataclass(frozen=True)` freezes attribute assignment, not nested objects. `c.text = "x"` raises; `c.metadata["k"] = 1` doesn't. Treat the metadata as read-only after indexing — mutating it desyncs from any persisted state.

- **`pipeline.answer` returns the model's raw completion.** No post-processing, no citation extraction, no fact-checking. If the model hallucinates `"[7]"` (a citation index that doesn't exist among the retrieved chunks), the pipeline doesn't catch it. Citation verification is a separate step — Exercise 7 of Module 19 (agent loops) is one place to put it.

## Reading

Primary:

- **Lewis, Perez, Piktus et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020).** The paper that named "RAG." Read §3 (the model) — the joint training of retriever + generator is more elaborate than what we build, but the framing of "retrieve, then condition the generator" is the durable idea. The empirical results in §4 are still the canonical demonstration.
- **Karpukhin, Oğuz, Min et al., "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP 2020).** The DPR paper — the "use a bi-encoder, embed both queries and passages, retrieve by inner product" recipe that defines modern dense retrieval. Read §3 (training) and §4 (results). Skim §5 (analysis).
- **Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009).** The canonical BM25 reference. Skim §3 (the BM25 formula) — even if you don't implement BM25 in this module, knowing what it does helps you reason about hybrid retrieval. Section 4 walks through the term-frequency-and-saturation intuition.

Secondary:

- **Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (EMNLP 2019).** The paper that established the "fine-tune a transformer with a contrastive loss to produce sentence embeddings" recipe. Most modern embedders (`nomic-embed`, `bge-base`, `mxbai-embed`) descend from this lineage. Read §3 (the architecture) and §4 (the loss function options).
- **Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2018).** The HNSW paper. Skim if you're curious how production vector stores get sub-linear retrieval. Don't implement; this module's flat search is enough for course scale.
- **Anthropic, "Contextual retrieval" blog post (Sep 2024).** A practical exposition of "prepend a chunk-specific summary to each chunk before embedding" as a retrieval-quality lever. Modest engineering, real wins. Read it after Exercise 1 — it's the next thing you'd try.

Optional:

- **Khattab, Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" (SIGIR 2020).** The "late interaction" alternative to bi-encoders — score query and passage at the *token* level, not the sentence level. Higher quality, more compute. The retrieval-quality frontier was set here for several years.
- **Jiang, Lin, Liu et al., "Active Retrieval Augmented Generation" (EMNLP 2023).** The "decide DURING generation whether to retrieve" angle — a step toward Module 19's agent loop. The model can issue mid-generation "retrieve again" requests when it's about to produce a fact-laden span.
- **Asai, Wu, Wang et al., "Self-RAG: Self-Reflective Retrieval-Augmented Generation" (ICLR 2024).** Trains the LLM to emit "retrieve" and "critique" tokens that gate the retrieval calls. Combines RAG with light agentic reasoning. Skim — useful as a pointer to where this is going.

## Deliverable checklist

- [ ] All tests in `tests/test_rag.py` pass: 142 tests, all green.
- [ ] Ollama running with at least one embedding model pulled. `ollama list` shows `nomic-embed-text` (or your chosen embedder).
- [ ] Notebook: `notebooks/17-rag.ipynb`. Indexes a corpus you care about (your `docs/`, your notes, the course material), runs Exercises 1, 2, 3, 5 with output cells visible.
- [ ] **RAG post-mortem** (Exercise 9) in `docs/rag-postmortem.md`. 3–4 paragraphs. The actual deliverable — what you indexed, what worked, what broke, what you'd build next.
- [ ] You can explain — out loud, without notes — why retrieval quality dominates RAG quality, and why improving the retriever buys more than improving the model.
- [ ] You can explain — out loud, without notes — what cosine similarity means as a dot product, and why every embedder in this module L2-normalizes its rows.
- [ ] You can explain — out loud, without notes — what chunk overlap is for, and what fails without it.
- [ ] You can explain — out loud, without notes — why the "I don't know" guard in the prompt is high-leverage, and what the model does without it.

## M-series notes

This module is comfortable on every M-series Mac. Practical considerations:

- **Embedding wall time.** `nomic-embed-text` runs at 30–80 chunks/sec on M-series, depending on Mac config and chunk size. A 1000-chunk corpus indexes in 15–30 seconds. On a 10k-chunk corpus, the indexing budget starts to matter — plan for a few minutes.
- **Vector store memory.** A 10k-chunk corpus at 768-dim float32 is 30 MB. A 100k-chunk corpus is 300 MB. Comfortable on every Mac.
- **Inference still happens via `OllamaBackend`.** All Module 16 caveats apply — first call is slow, steady-state matches the model size, etc. The RAG pipeline doesn't change the inference cost; it just changes the prompt the inference sees.
- **MLX-accelerated embedders.** `mlx-lm` doesn't ship a built-in embedder, but several MLX-converted embedding models are on HuggingFace. The conversion is a one-line `mlx_lm.convert` for most encoder-only architectures. For a corpus you re-embed often, MLX is 2–3× faster than Ollama's GGUF embedder on M-series. Not a deliverable; a worthwhile exercise once you know the embedder model you're keeping.
- **Disk space for embedding models.** `nomic-embed-text` is ~265 MB. `mxbai-embed-large` is ~670 MB. `bge-large` (if you go that route) is ~1.3 GB. Smaller than the chat models from Module 16 — not a constraint.

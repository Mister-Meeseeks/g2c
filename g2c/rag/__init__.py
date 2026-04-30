from .chunk import Chunk, chunk_text
from .embed import (
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_OLLAMA_URL,
    Embedder,
    HashEmbedder,
    OllamaEmbedder,
    OllamaEmbedError,
)
from .pipeline import RAGAnswer, RAGPipeline
from .prompt import DEFAULT_INSTRUCTION, DEFAULT_SYSTEM, RAGPrompt, assemble_rag_prompt
from .retrieve import DenseRetriever, RetrievedChunk
from .store import NumpyVectorStore, cosine_similarity

__all__ = [
    "Chunk",
    "DEFAULT_INSTRUCTION",
    "DEFAULT_OLLAMA_EMBED_MODEL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_SYSTEM",
    "DenseRetriever",
    "Embedder",
    "HashEmbedder",
    "NumpyVectorStore",
    "OllamaEmbedError",
    "OllamaEmbedder",
    "RAGAnswer",
    "RAGPipeline",
    "RAGPrompt",
    "RetrievedChunk",
    "assemble_rag_prompt",
    "chunk_text",
    "cosine_similarity",
]

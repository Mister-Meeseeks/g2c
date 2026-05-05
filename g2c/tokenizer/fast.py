"""Rust-backed BPE helpers for larger tokenizer artifact work.

Module 04 asks students to implement BPE in `bpe.py` because the algorithm is
small enough to understand directly: count adjacent pairs, choose the most
frequent pair, merge, repeat. This file exists for a different reason. Larger
corpora make the pure-Python teaching implementation slow, so these helpers use
Hugging Face `tokenizers`, whose core BPE trainer and encoder are implemented in
optimized Rust.

The important pedagogical point: `train_fast` and `encode_fast` are intended to
be functionally equivalent to the student-written `train` and `encode` methods.
They still populate the same `BPETokenizer.merges` and `BPETokenizer.vocab`
tables. They are infrastructure for reusable artifacts, not a replacement for
learning the algorithm in `bpe.py`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bpe import BPETokenizer


def _require_tokenizers():
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in bad envs.
        raise ModuleNotFoundError(
            "BPETokenizer fast paths require the `tokenizers` package. "
            "Run `./setup.sh` to install project dependencies."
        ) from exc
    return Tokenizer, decoders, models, pre_tokenizers, trainers


def bytes_to_unicode() -> dict[int, str]:
    """Return GPT-2's reversible byte-to-Unicode mapping.

    Hugging Face's byte-level BPE represents raw bytes as visible Unicode
    characters before applying BPE. This mapping lets us convert between that
    representation and the course tokenizer's explicit `bytes` vocab entries.
    """
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {char: byte for byte, char in BYTE_TO_UNICODE.items()}


def bytes_to_token_string(token_bytes: bytes) -> str:
    """Convert raw token bytes to the Unicode alphabet used by byte-level BPE."""
    return "".join(BYTE_TO_UNICODE[byte] for byte in token_bytes)


def token_string_to_bytes(token: str) -> bytes:
    """Convert a byte-level BPE token string back to raw bytes."""
    return bytes(UNICODE_TO_BYTE[char] for char in token)


def _byte_level_tokenizer_from_bpe(tokenizer: BPETokenizer):
    Tokenizer, decoders, models, pre_tokenizers, _ = _require_tokenizers()

    vocab = {bytes_to_token_string(bytes([byte])): byte for byte in range(256)}
    merges: list[tuple[str, str]] = []

    for (left_id, right_id), new_id in sorted(
        tokenizer.merges.items(),
        key=lambda item: item[1],
    ):
        left = bytes_to_token_string(tokenizer.vocab[left_id])
        right = bytes_to_token_string(tokenizer.vocab[right_id])
        token = bytes_to_token_string(tokenizer.vocab[new_id])
        vocab[token] = new_id
        merges.append((left, right))

    model = models.BPE(vocab=vocab, merges=merges, unk_token=None, fuse_unk=False)
    rust_tokenizer = Tokenizer(model)
    rust_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    rust_tokenizer.decoder = decoders.ByteLevel()
    return rust_tokenizer


def encode_fast(tokenizer: BPETokenizer, text: str) -> list[int]:
    """Encode text using the Rust-backed byte-level BPE implementation."""
    if not text:
        return []
    if not tokenizer.merges:
        return list(text.encode("utf-8"))
    rust_tokenizer = _byte_level_tokenizer_from_bpe(tokenizer)
    return rust_tokenizer.encode(text).ids


def train_fast(
    tokenizer: BPETokenizer,
    text: str,
    vocab_size: int,
    *,
    show_progress: bool = True,
    chunk_chars: int = 1_000_000,
) -> list[int]:
    """Train byte-level BPE with the Rust-backed trainer.

    The learned merge table is copied back onto `tokenizer`, preserving the
    course `BPETokenizer` representation. The input text is passed to the Rust
    trainer as a chunked iterator instead of one giant string, which is closer
    to how the library is optimized to consume larger corpora.
    """
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be at least 1")

    Tokenizer, decoders, models, pre_tokenizers, trainers = _require_tokenizers()
    tokenizer.merges = {}
    tokenizer.vocab = {i: bytes([i]) for i in range(256)}
    byte_to_id = {bytes([i]): i for i in range(256)}

    if not text or vocab_size == 256:
        return list(text.encode("utf-8"))

    rust_tokenizer = Tokenizer(models.BPE(unk_token=None, fuse_unk=False))
    rust_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    rust_tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
        special_tokens=[],
    )
    rust_tokenizer.train_from_iterator(
        _iter_text_chunks(text, chunk_chars),
        trainer=trainer,
        length=_num_text_chunks(text, chunk_chars),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _vocab_path, merges_path = rust_tokenizer.model.save(tmpdir, prefix="bpe")
        merge_lines = Path(merges_path).read_text(encoding="utf-8").splitlines()

    new_id = 256
    for line in merge_lines:
        if not line or line.startswith("#"):
            continue
        left, right = line.split(" ", maxsplit=1)
        left_bytes = token_string_to_bytes(left)
        right_bytes = token_string_to_bytes(right)
        left_id = _ensure_vocab_bytes(byte_to_id, left_bytes)
        right_id = _ensure_vocab_bytes(byte_to_id, right_bytes)
        pair = (left_id, right_id)
        if pair in tokenizer.merges:
            continue
        tokenizer.merges[pair] = new_id
        new_bytes = left_bytes + right_bytes
        tokenizer.vocab[new_id] = new_bytes
        byte_to_id[new_bytes] = new_id
        new_id += 1
        if new_id >= vocab_size:
            break

    return encode_fast(tokenizer, text)


def _ensure_vocab_bytes(byte_to_id: dict[bytes, int], token_bytes: bytes) -> int:
    token_id = byte_to_id.get(token_bytes)
    if token_id is not None:
        return token_id
    raise ValueError("fast BPE merge references token bytes not yet in vocab")


def _iter_text_chunks(text: str, chunk_chars: int):
    for start in range(0, len(text), chunk_chars):
        yield text[start : start + chunk_chars]


def _num_text_chunks(text: str, chunk_chars: int) -> int:
    return max(1, (len(text) + chunk_chars - 1) // chunk_chars)

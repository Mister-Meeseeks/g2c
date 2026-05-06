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

import gc
import json
import os
import tempfile
import threading
from collections.abc import Callable
from contextlib import contextmanager
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
MAX_FAST_TRAIN_CHUNK_CHARS = 8_192
FastProgressCallback = Callable[[dict[str, object]], None]


def bytes_to_token_string(token_bytes: bytes) -> str:
    """Convert raw token bytes to the Unicode alphabet used by byte-level BPE."""
    return "".join(BYTE_TO_UNICODE[byte] for byte in token_bytes)


def token_string_to_bytes(token: str) -> bytes:
    """Convert a byte-level BPE token string back to raw bytes."""
    return bytes(UNICODE_TO_BYTE[char] for char in token)


def _byte_level_tokenizer_from_bpe(tokenizer: BPETokenizer):
    Tokenizer, decoders, models, pre_tokenizers, _ = _require_tokenizers()

    vocab = {bytes_to_token_string(bytes([byte])): byte for byte in range(256)}
    for token, token_id in tokenizer.special_to_id.items():
        vocab[token] = token_id
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
        return tokenizer._encode_initial_ids(text)
    rust_tokenizer = _byte_level_tokenizer_from_bpe(tokenizer)
    ids: list[int] = []
    for segment in tokenizer._special_aware_segments(text):
        if isinstance(segment, int):
            ids.append(segment)
        else:
            encoding = rust_tokenizer.encode(segment)
            ids.extend(encoding.ids)
            del encoding
    del rust_tokenizer
    gc.collect()
    return ids


def train_fast(
    tokenizer: BPETokenizer,
    text: str,
    vocab_size: int,
    *,
    show_progress: bool = True,
    chunk_chars: int = MAX_FAST_TRAIN_CHUNK_CHARS,
    encode_training_text: bool = True,
    progress_callback: FastProgressCallback | None = None,
) -> list[int]:
    """Train byte-level BPE with the Rust-backed trainer.

    The learned merge table is copied back onto `tokenizer`, preserving the
    course `BPETokenizer` representation. The input text is passed to the Rust
    trainer as a chunked iterator instead of one giant string. The Rust trainer
    is fast, but very large iterator items can still become pathological, so
    chunk size is capped to keep notebook artifact generation interactive.
    Artifact builders can set `encode_training_text=False` when they only need
    the learned tokenizer tables and will encode a smaller inspection sample.
    """
    if vocab_size < tokenizer.base_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {tokenizer.base_vocab_size} "
            "for this tokenizer's byte + special-token base vocabulary"
        )
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be at least 1")
    effective_chunk_chars = min(chunk_chars, MAX_FAST_TRAIN_CHUNK_CHARS)
    total_chunks = _num_text_chunks(text, effective_chunk_chars)

    Tokenizer, decoders, models, pre_tokenizers, trainers = _require_tokenizers()
    tokenizer._reset_to_base_vocab()
    byte_to_id = {bytes([i]): i for i in range(256)}

    if not text or vocab_size == tokenizer.base_vocab_size:
        return tokenizer._encode_initial_ids(text)

    rust_tokenizer = Tokenizer(models.BPE(unk_token=None, fuse_unk=False))
    rust_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    rust_tokenizer.decoder = decoders.ByteLevel()
    capture_native_progress = progress_callback is not None
    trainer = _make_bpe_trainer(
        trainers,
        pre_tokenizers,
        vocab_size=vocab_size,
        show_progress=show_progress or capture_native_progress,
        progress_format="json" if capture_native_progress else "indicatif",
        special_tokens=tokenizer.special_tokens,
    )
    _emit_progress(
        progress_callback,
        phase="fast_chunks_start",
        chars=len(text),
        chunks=total_chunks,
        chunk_chars=effective_chunk_chars,
    )
    iterator = _iter_text_chunks(
        text,
        effective_chunk_chars,
        progress_callback=progress_callback,
        total_chunks=total_chunks,
        target_vocab_size=vocab_size,
    )
    native_reporter = _NativeProgressReporter(progress_callback)
    with _maybe_capture_fd2_json(native_reporter, enabled=capture_native_progress):
        rust_tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=total_chunks,
        )
    del iterator, native_reporter, trainer
    gc.collect()
    _emit_progress(
        progress_callback,
        phase="fast_chunks_done",
        chunks=total_chunks,
        chars=len(text),
    )

    _emit_progress(
        progress_callback,
        phase="fast_export_start",
        chunks=total_chunks,
        target_vocab_size=vocab_size,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _vocab_path, merges_path = rust_tokenizer.model.save(tmpdir, prefix="bpe")
        merge_lines = Path(merges_path).read_text(encoding="utf-8").splitlines()
    _emit_progress(
        progress_callback,
        phase="fast_export_done",
        merge_count=max(0, len(merge_lines) - 1),
        target_vocab_size=vocab_size,
    )
    del rust_tokenizer
    gc.collect()

    _emit_progress(
        progress_callback,
        phase="fast_import_start",
        merge_count=max(0, len(merge_lines) - 1),
        target_vocab_size=vocab_size,
    )
    _import_merge_lines(tokenizer, byte_to_id, merge_lines)
    _emit_progress(
        progress_callback,
        phase="fast_import_done",
        vocab_size=len(tokenizer.vocab),
        target_vocab_size=vocab_size,
    )
    del merge_lines
    gc.collect()

    if not encode_training_text:
        _emit_progress(
            progress_callback,
            phase="fast_encode_skipped",
            vocab_size=len(tokenizer.vocab),
            target_vocab_size=vocab_size,
        )
        return []

    _emit_progress(
        progress_callback,
        phase="fast_encode_start",
        vocab_size=len(tokenizer.vocab),
        target_vocab_size=vocab_size,
    )
    ids = encode_fast(tokenizer, text)
    _emit_progress(
        progress_callback,
        phase="fast_encode_done",
        vocab_size=len(tokenizer.vocab),
        target_vocab_size=vocab_size,
        token_count=len(ids),
    )
    return ids


def _import_merge_lines(
    tokenizer: BPETokenizer,
    byte_to_id: dict[bytes, int],
    merge_lines: list[str],
) -> None:
    """Import Hugging Face merge lines into the course tokenizer tables.

    Hugging Face merge files are usually topological, but large mixed corpora can
    produce lines that reference a composite token before that token's own merge
    has been imported into our course ID space. Build a byte-level dependency
    map first, then materialize missing operands recursively.
    """
    merge_pairs: list[tuple[bytes, bytes]] = []
    merge_by_output: dict[bytes, tuple[bytes, bytes]] = {}
    for line in merge_lines:
        if not line or line.startswith("#"):
            continue
        left, right = line.split(" ", maxsplit=1)
        left_bytes = token_string_to_bytes(left)
        right_bytes = token_string_to_bytes(right)
        merge_pairs.append((left_bytes, right_bytes))
        merge_by_output.setdefault(left_bytes + right_bytes, (left_bytes, right_bytes))

    next_id = max(tokenizer.vocab) + 1

    def ensure(token_bytes: bytes) -> int:
        nonlocal next_id
        token_id = byte_to_id.get(token_bytes)
        if token_id is not None:
            return token_id

        split = merge_by_output.get(token_bytes)
        if split is None:
            split = _fallback_split(token_bytes, byte_to_id)
        left_bytes, right_bytes = split
        left_id = ensure(left_bytes)
        right_id = ensure(right_bytes)
        pair = (left_id, right_id)

        existing_id = byte_to_id.get(token_bytes)
        if existing_id is not None:
            return existing_id
        if pair in tokenizer.merges:
            token_id = tokenizer.merges[pair]
            byte_to_id[token_bytes] = token_id
            return token_id

        token_id = next_id
        next_id += 1
        tokenizer.merges[pair] = token_id
        tokenizer.vocab[token_id] = token_bytes
        byte_to_id[token_bytes] = token_id
        return token_id

    for left_bytes, right_bytes in merge_pairs:
        ensure(left_bytes + right_bytes)


def _fallback_split(
    token_bytes: bytes,
    byte_to_id: dict[bytes, int],
) -> tuple[bytes, bytes]:
    """Split an unexpected composite token into known byte-level pieces."""
    if len(token_bytes) < 2:
        raise ValueError("fast BPE merge references unknown byte token")
    for split_at in range(len(token_bytes) - 1, 0, -1):
        left = token_bytes[:split_at]
        right = token_bytes[split_at:]
        if left in byte_to_id:
            return left, right
    return token_bytes[:1], token_bytes[1:]


def _make_bpe_trainer(
    trainers,
    pre_tokenizers,
    *,
    vocab_size: int,
    show_progress: bool,
    progress_format: str,
    special_tokens: tuple[str, ...] = (),
):
    kwargs = {
        "vocab_size": vocab_size,
        "min_frequency": 1,
        "initial_alphabet": pre_tokenizers.ByteLevel.alphabet(),
        "show_progress": show_progress,
        "progress_format": progress_format,
        "special_tokens": list(special_tokens),
    }
    try:
        return trainers.BpeTrainer(**kwargs)
    except TypeError:
        # Older `tokenizers` versions do not expose `progress_format`. They
        # still train correctly, but native JSON progress will be unavailable.
        kwargs.pop("progress_format")
        return trainers.BpeTrainer(**kwargs)


def _iter_text_chunks(
    text: str,
    chunk_chars: int,
    *,
    progress_callback: FastProgressCallback | None,
    total_chunks: int,
    target_vocab_size: int,
):
    progress_every = max(1, total_chunks // 80)
    for chunk_index, start in enumerate(range(0, len(text), chunk_chars), start=1):
        if chunk_index == 1 or chunk_index == total_chunks or chunk_index % progress_every == 0:
            _emit_progress(
                progress_callback,
                phase="fast_chunk",
                chunk_index=chunk_index,
                chunks=total_chunks,
                chars_seen=min(len(text), start + chunk_chars),
                chars=len(text),
                chunk_chars=chunk_chars,
            )
        yield text[start : start + chunk_chars]
    _emit_progress(
        progress_callback,
        phase="fast_native_train_start",
        chunks=total_chunks,
        chars=len(text),
        target_vocab_size=target_vocab_size,
    )


class _NativeProgressReporter:
    """Throttle Rust tokenizer JSON progress before calling Python callbacks."""

    def __init__(self, callback: FastProgressCallback | None, max_updates: int = 80) -> None:
        self.callback = callback
        self.max_updates = max_updates
        self.last_bucket_by_stage: dict[str, int] = {}

    def __call__(self, payload: dict[str, object]) -> None:
        if self.callback is None:
            return
        stage = payload.get("stage")
        current = payload.get("current")
        total = payload.get("total")
        if not isinstance(stage, str) or not isinstance(current, int) or not isinstance(total, int):
            return
        if total <= 0:
            return

        bucket = min(self.max_updates, int(current * self.max_updates / total))
        should_emit = (
            current == 0
            or current >= total
            or self.last_bucket_by_stage.get(stage) != bucket
        )
        if not should_emit:
            return

        self.last_bucket_by_stage[stage] = bucket
        _emit_progress(
            self.callback,
            phase="fast_native_progress",
            native_stage=stage,
            current=current,
            total=total,
        )


@contextmanager
def _maybe_capture_fd2_json(callback: Callable[[dict[str, object]], None], *, enabled: bool):
    """Capture native Rust JSON progress written directly to fd 2."""
    if not enabled:
        yield
        return

    old_fd2 = os.dup(2)
    read_fd, write_fd = os.pipe()

    def read_progress() -> None:
        with os.fdopen(read_fd, "rb", closefd=True) as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    os.write(old_fd2, raw_line)
                    continue
                if isinstance(payload, dict):
                    callback(payload)

    reader = threading.Thread(target=read_progress, daemon=True)
    reader.start()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        yield
    finally:
        os.dup2(old_fd2, 2)
        reader.join()
        os.close(old_fd2)


def _num_text_chunks(text: str, chunk_chars: int) -> int:
    return max(1, (len(text) + chunk_chars - 1) // chunk_chars)


def _emit_progress(callback: FastProgressCallback | None, **payload: object) -> None:
    if callback is not None:
        callback(payload)

"""Display + orchestration helpers for the Module 04 tokenizer notebook.

These wrap the durable tokenizer-artifact training in an IPython progress bar
and render the post-training inspection (most-frequent tokens, longest learned
tokens, sample re-encoding). They are not part of the course deliverable.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from g2c.artifacts import (
    TokenizerArtifact,
    TokenizerArtifactConfig,
    train_or_load_tokenizer_artifact,
)
from g2c.tokenizer import BPETokenizer

__all__ = [
    "inspect_tokenizer_artifact",
    "make_artifact_display",
    "run_and_inspect_tokenizer_artifact",
]


def make_artifact_display(label: str):
    """Return a status/progress callback that renders a Markdown progress bar."""
    start = time.perf_counter()
    progress = None

    def show(message: Markdown) -> None:
        nonlocal progress
        if progress is None:
            progress = display(message, display_id=True)
        else:
            progress.update(message)

    def update(info: dict) -> None:
        phase = info.get("phase")
        if phase == "loaded":
            print(f"loaded {label}: {info['token_count']:,} sample token IDs")
        elif phase == "missing_source":
            print(f"Skipping {label}: source {info['source']!r} is not available.")
        elif phase == "fast_start":
            show(
                Markdown(
                    f"{label}: Rust-backed training starting "
                    f"| chars `{info['chars']:,}` "
                    f"| chunks `{info['chunks']:,}` "
                    f"| target vocab `{info['target_vocab_size']:,}`"
                )
            )
        elif phase == "fast_chunks_start":
            show(
                Markdown(
                    f"{label}: preparing text chunks "
                    f"| chars `{info['chars']:,}` "
                    f"| chunks `{info['chunks']:,}`"
                )
            )
        elif phase == "fast_chunk":
            chunk_index = int(info["chunk_index"])
            chunks = int(info["chunks"])
            width = 28
            filled = max(1, min(width, round(width * chunk_index / chunks)))
            bar = "#" * filled + "-" * (width - filled)
            show(
                Markdown(
                    f"{label}: ingesting text chunks `[{bar}]` "
                    f"`{chunk_index:,}/{chunks:,}` "
                    f"| chars `{info['chars_seen']:,}/{info['chars']:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_native_train_start":
            show(
                Markdown(
                    f"{label}: starting Rust tokenizer trainer... "
                    f"| target vocab `{info['target_vocab_size']:,}` "
                    f"| chunks `{info['chunks']:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_native_progress":
            current = int(info["current"])
            total = max(1, int(info["total"]))
            width = 28
            filled = max(1, min(width, round(width * current / total)))
            bar = "#" * filled + "-" * (width - filled)
            show(
                Markdown(
                    f"{label}: {info['native_stage']} `[{bar}]` "
                    f"`{current:,}/{total:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_chunks_done":
            show(
                Markdown(
                    f"{label}: Rust BPE training complete; preparing tokenizer export... "
                    f"| chunks `{info['chunks']:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_export_start":
            show(
                Markdown(
                    f"{label}: exporting Rust tokenizer merges... "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_export_done":
            show(
                Markdown(
                    f"{label}: exported `{info['merge_count']:,}` learned merges; releasing Rust tokenizer... "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_import_start":
            show(
                Markdown(
                    f"{label}: importing merges into the course tokenizer tables... "
                    f"| merges `{info['merge_count']:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_import_done":
            show(
                Markdown(
                    f"{label}: imported course tokenizer vocab `{info['vocab_size']:,}/{info['target_vocab_size']:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_encode_skipped":
            show(
                Markdown(
                    f"{label}: skipping full-corpus encode; tokenizer artifact will save an encoded inspection sample. "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "sample_encode_start":
            show(
                Markdown(
                    f"{label}: encoding inspection sample... "
                    f"| sample chars `{info['chars']:,}` "
                    f"| training chars `{info['training_chars']:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "sample_encode_done":
            show(
                Markdown(
                    f"{label}: encoded inspection sample "
                    f"| chars `{info['chars']:,}` "
                    f"| tokens `{info['token_count']:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_encode_start":
            show(
                Markdown(
                    f"{label}: encoding training text with learned tokenizer... "
                    f"| vocab `{info['vocab_size']:,}/{info['target_vocab_size']:,}` "
                    f"| elapsed `{time.perf_counter() - start:.1f}s`"
                )
            )
        elif phase == "fast_done":
            show(
                Markdown(
                    f"{label}: trained vocab `{info['vocab_size']:,}/{info['target_vocab_size']:,}` "
                    f"| sample tokens `{info['token_count']:,}` "
                    f"| chunks `{info['chunks']:,}` "
                    f"| elapsed `{info['elapsed_seconds']:.1f}s`"
                )
            )
        elif phase == "saved":
            print(f"saved {label}: {info['token_count']:,} sample token IDs")
        elif {"vocab_size", "target_vocab_size", "steps", "tokens", "last_merge_count"}.issubset(info):
            if progress is None:
                show(Markdown(f"{label} tokenizer training starting..."))
            last_token = info.get("last_token_repr")
            if last_token is None:
                last_token = "None"
            last_token = str(last_token).replace("`", "\\`")
            elapsed = info.get("elapsed_seconds", time.perf_counter() - start)
            show(
                Markdown(
                    f"{label}: vocab `{info['vocab_size']:,}/{info['target_vocab_size']:,}` "
                    f"| steps `{info['steps']:,}` "
                    f"| tokens `{info['tokens']:,}` "
                    f"| last token `{last_token}` "
                    f"| last count `{info['last_merge_count']}` "
                    f"| elapsed `{elapsed:.1f}s`"
                )
            )
        elif phase is not None:
            show(Markdown(f"{label}: `{phase}` | elapsed `{time.perf_counter() - start:.1f}s`"))

    return update


def _display_token_bytes(token_bytes: bytes) -> str:
    text = token_bytes.decode("utf-8", errors="backslashreplace")
    return text.replace("\n", "\\n").replace("\t", "\\t")


def _shorten(text: str, width: int = 36) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _deterministic_text_window(text: str, *, chars: int = 360, seed: int = 0) -> str:
    if len(text) <= chars:
        return text
    rng = random.Random(seed)
    start = rng.randrange(0, len(text) - chars)
    return text[start : start + chars]


def _token_strings(tokenizer: BPETokenizer, text: str) -> list[str]:
    return [_display_token_bytes(tokenizer.vocab[token_id]) for token_id in tokenizer.encode_fast(text)]


def _most_frequent_final_tokens(
    tokenizer: BPETokenizer,
    text: str,
    *,
    top_n: int = 20,
    max_chars: int = 200_000,
    learned_only: bool = False,
) -> list[tuple[int, str, int, int]]:
    ids = tokenizer.encode_fast(text[:max_chars])
    min_token_id = tokenizer.base_vocab_size if learned_only else 0
    counts = Counter(token_id for token_id in ids if token_id >= min_token_id)
    rows = []
    for token_id, count in counts.most_common(top_n):
        token_text = _display_token_bytes(tokenizer.vocab[token_id])
        rows.append((token_id, token_text, count, len(tokenizer.vocab[token_id])))
    return rows


def _longest_learned_tokens(tokenizer: BPETokenizer, *, top_n: int = 20) -> list[tuple[int, int, str]]:
    learned_ids = [token_id for token_id in tokenizer.vocab if token_id >= tokenizer.base_vocab_size]
    learned_ids.sort(key=lambda token_id: (len(tokenizer.vocab[token_id]), token_id), reverse=True)
    selected_token_bytes: list[bytes] = []
    rows = []
    for token_id in learned_ids:
        token_bytes = tokenizer.vocab[token_id]
        if any(token_bytes in selected for selected in selected_token_bytes):
            continue
        selected_token_bytes.append(token_bytes)
        rows.append((token_id, len(token_bytes), _display_token_bytes(token_bytes)))
        if len(rows) >= top_n:
            break
    return rows


def _print_rows(title: str, headers: tuple[str, ...], rows: list[tuple]) -> None:
    print("\n" + title)
    print(" | ".join(headers))
    print("-" * 72)
    for row in rows:
        print(" | ".join(str(value) for value in row))


def _plot_frequency_rows(rows: list[tuple[int, str, int, int]], *, title: str) -> None:
    if not rows:
        print("No tokens found in this encoded sample.")
        return
    labels = [_shorten(repr(token), 28) for _, token, _, _ in reversed(rows)]
    counts = [count for _, _, count, _ in reversed(rows)]
    plt.figure(figsize=(8, max(4, len(rows) * 0.28)))
    plt.barh(labels, counts)
    plt.xlabel("count in encoded sample")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Early dividers shift with special tokens (1st/4th/16th/64th *learned* token);
# late dividers are absolute power-of-2 boundaries (vocab_size = 2**k is the
# usual demarcation, so the highest ID at that size is 2**k - 1).
_EARLY_LEARNED_OFFSETS = (0, 3, 15, 63)
_LATE_ABSOLUTE_INDICES = (511, 1023, 2047, 4095, 8191, 16383)


def _vocab_divider_indices(tokenizer: BPETokenizer) -> list[int]:
    base = tokenizer.base_vocab_size
    vocab_size = len(tokenizer.vocab)
    candidates = [base + offset for offset in _EARLY_LEARNED_OFFSETS]
    candidates.extend(_LATE_ABSOLUTE_INDICES)
    seen: set[int] = set()
    indices: list[int] = []
    for idx in candidates:
        if idx < base or idx >= vocab_size or idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    indices.sort()
    return indices


def _plot_vocab_divider_tokens(
    tokenizer: BPETokenizer,
    text: str,
    *,
    title: str,
    max_chars: int = 200_000,
) -> None:
    indices = _vocab_divider_indices(tokenizer)
    if not indices:
        return
    counts = Counter(tokenizer.encode_fast(text[:max_chars]))
    rows = [
        (idx, _display_token_bytes(tokenizer.vocab[idx]), counts.get(idx, 0))
        for idx in indices
    ]
    labels = [f"#{idx}: {_shorten(repr(token), 24)}" for idx, token, _ in reversed(rows)]
    bar_counts = [count for _, _, count in reversed(rows)]
    plt.figure(figsize=(8, max(4, len(rows) * 0.32)))
    plt.barh(labels, bar_counts)
    plt.xlabel("count in encoded sample")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def inspect_tokenizer_artifact(artifact: TokenizerArtifact, *, seed: int = 13) -> None:
    """Print summary stats and plot most-frequent learned tokens."""
    tokenizer = artifact.tokenizer
    text = artifact.text
    if not text:
        print(f"Skipping inspection for {artifact.config.name}: no source text loaded.")
        return

    name = artifact.config.name
    manifest = artifact.manifest
    print("=" * 88)
    print(name)
    print("trainer:", manifest.get("trainer", "unknown"))
    print("vocab size:", manifest["actual_vocab_size"])
    print("sample token count:", manifest["token_count"])
    print("encoded sample chars:", manifest.get("encoded_chars", manifest.get("actual_chars")))
    print("chars/token:", round(manifest["chars_per_token"], 3))

    sample = _deterministic_text_window(text, chars=360, seed=seed)
    print("\nSample text:")
    print(repr(sample))
    print("\nToken strings for sample:")
    print(_token_strings(tokenizer, sample))

    longest_rows = _longest_learned_tokens(tokenizer, top_n=20)
    _print_rows(
        "Longest learned tokens, greedily excluding byte substrings",
        ("token_id", "bytes", "token"),
        [(token_id, length, repr(_shorten(token, 120))) for token_id, length, token in longest_rows],
    )

    learned_frequent_rows = _most_frequent_final_tokens(tokenizer, text, top_n=20, learned_only=True)
    _plot_frequency_rows(learned_frequent_rows, title=f"{name}: most frequent learned final tokens")
    _plot_vocab_divider_tokens(tokenizer, text, title=f"{name}: tokens sampled at vocabulary dividers")


def run_and_inspect_tokenizer_artifact(
    config: TokenizerArtifactConfig,
    *,
    repo_root: Path,
    enabled: bool = True,
    force: bool = False,
) -> TokenizerArtifact | None:
    """Train (or load) the tokenizer artifact and print/plot its inspection."""
    if not enabled:
        print(f"Skipping {config.name}: disabled for this run.")
        return None
    artifact_display = make_artifact_display(config.name)
    artifact = train_or_load_tokenizer_artifact(
        config,
        repo_root=repo_root,
        force=force,
        progress_callback=artifact_display,
        status_callback=artifact_display,
    )
    if artifact is not None:
        inspect_tokenizer_artifact(artifact)
    return artifact

"""A tiny fixed caption vocabulary for the MNIST captioning exercise.

This Beyond module deliberately does NOT depend on the Module 04
tokenizer: the point under study is how images enter the token stream,
not how text gets tokenized. Sixteen word-level tokens cover every
caption the exercises use ("This is a 7 .") plus five structural ids:
padding, retained image-start and image-end boundaries, the image-patch
placeholder, and the end-of-caption marker.

All provided plumbing — nothing here is scaffolded.
"""
from __future__ import annotations

CAPTION_TOKENS: list[str] = [
    "<pad>",
    "<img>",
    "<image_patch>",
    "</img>",
    "<end>",
    "This",
    "is",
    "a",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

_TOKEN_TO_ID = {tok: i for i, tok in enumerate(CAPTION_TOKENS)}

PAD_ID: int = _TOKEN_TO_ID["<pad>"]
IMAGE_START_ID: int = _TOKEN_TO_ID["<img>"]
IMAGE_PATCH_ID: int = _TOKEN_TO_ID["<image_patch>"]
IMAGE_END_ID: int = _TOKEN_TO_ID["</img>"]
END_ID: int = _TOKEN_TO_ID["<end>"]

VOCAB_SIZE: int = len(CAPTION_TOKENS)


def caption_ids(digit: int) -> list[int]:
    """Token ids for `This is a <digit> . <end>`."""
    if not 0 <= digit <= 9:
        raise ValueError(f"digit must be in [0, 9], got {digit}")
    words = ["This", "is", "a", str(digit), ".", "<end>"]
    return [_TOKEN_TO_ID[w] for w in words]


def decode_ids(ids: list[int]) -> str:
    """Human-readable string for a list of caption-vocab ids."""
    return " ".join(
        CAPTION_TOKENS[i] if 0 <= i < VOCAB_SIZE else f"<{i}?>" for i in ids
    )

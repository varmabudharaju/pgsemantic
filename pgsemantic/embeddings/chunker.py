"""Sentence-aware text chunking for long document embedding.

Splits text into overlapping chunks at sentence boundaries.
Uses whitespace tokenization (no external tokenizer dependency).
"""
from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DEFAULT_MAX_CHUNKS = 200


def chunk_text(
    text: str,
    max_tokens: int = 256,
    overlap: int = 64,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Args:
        text: The text to split.
        max_tokens: Maximum words per chunk (whitespace-tokenized).
        overlap: Number of words to overlap between adjacent chunks.
        max_chunks: Safety cap on total chunks per text.

    Returns:
        List of chunk strings. Empty list for empty/whitespace input.
    """
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    sentences = _SENTENCE_RE.split(text)

    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue

        if current_words and len(current_words) + len(sentence_words) > max_tokens:
            chunks.append(" ".join(current_words))
            if len(chunks) >= max_chunks:
                return chunks
            current_words = current_words[-overlap:] if overlap > 0 else []

        current_words.extend(sentence_words)

        while len(current_words) > max_tokens:
            chunks.append(" ".join(current_words[:max_tokens]))
            if len(chunks) >= max_chunks:
                return chunks
            if overlap > 0:
                current_words = current_words[max_tokens - overlap:]
            else:
                current_words = current_words[max_tokens:]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks[:max_chunks]

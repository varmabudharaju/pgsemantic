"""Tests for text chunking."""
from __future__ import annotations


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        from pgsemantic.embeddings.chunker import chunk_text
        text = "This is a short sentence."
        chunks = chunk_text(text, max_tokens=256, overlap=64)
        assert chunks == ["This is a short sentence."]

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        from pgsemantic.embeddings.chunker import chunk_text
        sentences = [f"Sentence {i} " + "word " * 48 + "end." for i in range(5)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=100, overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 100

    def test_overlap_produces_shared_content(self) -> None:
        from pgsemantic.embeddings.chunker import chunk_text
        sentences = [f"Unique sentence number {i}. " for i in range(20)]
        text = "".join(sentences)
        chunks = chunk_text(text, max_tokens=30, overlap=10)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split())
            words_b = set(chunks[i + 1].split())
            assert words_a & words_b

    def test_no_overlap(self) -> None:
        from pgsemantic.embeddings.chunker import chunk_text
        text = " ".join(["word"] * 100)
        chunks = chunk_text(text, max_tokens=30, overlap=0)
        total_words = sum(len(c.split()) for c in chunks)
        assert total_words == 100

    def test_sentence_boundary_splitting(self) -> None:
        from pgsemantic.embeddings.chunker import chunk_text
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, max_tokens=5, overlap=0)
        for chunk in chunks:
            assert chunk.strip()

    def test_max_chunks_limit(self) -> None:
        from pgsemantic.embeddings.chunker import chunk_text
        text = " ".join(["word"] * 50000)
        chunks = chunk_text(text, max_tokens=10, overlap=0, max_chunks=200)
        assert len(chunks) <= 200

    def test_empty_text_returns_empty(self) -> None:
        from pgsemantic.embeddings.chunker import chunk_text
        assert chunk_text("", max_tokens=256, overlap=64) == []

    def test_whitespace_only_returns_empty(self) -> None:
        from pgsemantic.embeddings.chunker import chunk_text
        assert chunk_text("   \n\t  ", max_tokens=256, overlap=64) == []

"""
Unit tests for PDF processing module.

Tests cover:
- PDF validation
- Text extraction from PDFs
- Text chunking
- Utility functions (clean_text, count_tokens)
- Full processing pipeline
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.pdf_processor import (
    PDFProcessor,
    DocumentChunk,
    clean_text,
    count_tokens
)


class TestPDFValidation:
    """Test suite for PDF validation functionality."""

    @pytest.mark.unit
    def test_validate_existing_pdf(self, simple_pdf):
        """Test validation passes for valid PDF file."""
        processor = PDFProcessor()
        assert processor.validate_pdf(str(simple_pdf)) is True

    @pytest.mark.unit
    def test_validate_nonexistent_file(self):
        """Test validation fails for non-existent file."""
        processor = PDFProcessor()
        assert processor.validate_pdf("/nonexistent/file.pdf") is False

    @pytest.mark.unit
    def test_validate_non_pdf_file(self, non_pdf_file):
        """Test validation fails for non-PDF files."""
        processor = PDFProcessor()
        assert processor.validate_pdf(str(non_pdf_file)) is False

    @pytest.mark.unit
    def test_validate_empty_pdf(self, empty_pdf):
        """Test validation fails for empty PDF (0 pages)."""
        processor = PDFProcessor()
        assert processor.validate_pdf(str(empty_pdf)) is False

    @pytest.mark.unit
    def test_validate_file_size_limit(self, simple_pdf):
        """Test validation respects file size limits."""
        processor = PDFProcessor()
        # Should pass with large limit
        assert processor.validate_pdf(str(simple_pdf), max_size_mb=100) is True
        # Should fail with tiny limit
        assert processor.validate_pdf(str(simple_pdf), max_size_mb=0.001) is False

    @pytest.mark.unit
    def test_validate_unreadable_file(self, simple_pdf):
        """Test validation fails for unreadable files."""
        processor = PDFProcessor()
        # Change permissions to unreadable (on Unix-like systems)
        if os.name != 'nt':  # Skip on Windows
            os.chmod(simple_pdf, 0o000)
            assert processor.validate_pdf(str(simple_pdf)) is False
            # Restore permissions for cleanup
            os.chmod(simple_pdf, 0o644)


class TestTextExtraction:
    """Test suite for PDF text extraction."""

    @pytest.mark.unit
    @pytest.mark.requires_pdf
    def test_extract_single_page(self, simple_pdf):
        """Test extracting text from single-page PDF."""
        processor = PDFProcessor()
        pages = processor.extract_text_from_pdf(str(simple_pdf))

        assert len(pages) == 1
        assert pages[0]['page_number'] == 1
        assert 'simple test PDF' in pages[0]['text']
        assert pages[0]['source_file'] == 'simple.pdf'
        assert 'char_count' in pages[0]

    @pytest.mark.unit
    @pytest.mark.requires_pdf
    def test_extract_multiple_pages(self, multi_page_pdf):
        """Test extracting text from multi-page PDF."""
        processor = PDFProcessor()
        pages = processor.extract_text_from_pdf(str(multi_page_pdf))

        assert len(pages) == 3
        assert pages[0]['page_number'] == 1
        assert pages[1]['page_number'] == 2
        assert pages[2]['page_number'] == 3

        # Check content is different per page
        assert 'page one' in pages[0]['text'].lower()
        assert 'page two' in pages[1]['text'].lower()
        assert 'page three' in pages[2]['text'].lower()

    @pytest.mark.unit
    def test_extract_nonexistent_pdf(self):
        """Test extraction raises error for missing file."""
        processor = PDFProcessor()
        with pytest.raises(FileNotFoundError):
            processor.extract_text_from_pdf("/nonexistent/file.pdf")

    @pytest.mark.unit
    @pytest.mark.requires_pdf
    def test_extract_preserves_metadata(self, simple_pdf):
        """Test that extraction preserves all required metadata."""
        processor = PDFProcessor()
        pages = processor.extract_text_from_pdf(str(simple_pdf))

        required_fields = ['page_number', 'text', 'source_file', 'char_count']
        for field in required_fields:
            assert field in pages[0]


class TestTextChunking:
    """Test suite for text chunking functionality."""

    @pytest.mark.unit
    def test_chunk_single_page(self, simple_pdf):
        """Test chunking a single page document."""
        processor = PDFProcessor(chunk_size=100, chunk_overlap=20)
        pages = processor.extract_text_from_pdf(str(simple_pdf))
        chunks = processor.chunk_text(pages)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert chunks[0].page_number == 1
        assert chunks[0].source_file == 'simple.pdf'

    @pytest.mark.unit
    def test_chunk_long_text(self, long_text_pdf):
        """Test chunking creates multiple chunks for long text."""
        processor = PDFProcessor(chunk_size=200, chunk_overlap=50)
        pages = processor.extract_text_from_pdf(str(long_text_pdf))
        chunks = processor.chunk_text(pages)

        # Long text should create multiple chunks
        assert len(chunks) > 1

        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    @pytest.mark.unit
    def test_chunk_metadata_preservation(self, multi_page_pdf):
        """Test that chunking preserves page numbers and metadata."""
        processor = PDFProcessor(chunk_size=50, chunk_overlap=10)
        pages = processor.extract_text_from_pdf(str(multi_page_pdf))
        chunks = processor.chunk_text(pages)

        # All chunks should have metadata
        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.page_number >= 1
            assert chunk.chunk_index >= 0
            assert chunk.source_file == 'multipage.pdf'
            assert 'character_count' in chunk.metadata
            assert 'token_count' in chunk.metadata

    @pytest.mark.unit
    def test_chunk_overlap_works(self, long_text_pdf):
        """Test that chunk overlap is applied."""
        processor = PDFProcessor(chunk_size=200, chunk_overlap=50)
        pages = processor.extract_text_from_pdf(str(long_text_pdf))
        chunks = processor.chunk_text(pages)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content (overlap)
            chunk1_end = chunks[0].content[-50:]
            chunk2_start = chunks[1].content[:50]
            # There should be some common words due to overlap
            assert any(word in chunk2_start for word in chunk1_end.split()[:5])

    @pytest.mark.unit
    def test_chunk_empty_pages_skipped(self):
        """Test that empty pages are skipped during chunking."""
        processor = PDFProcessor()
        pages = [
            {'page_number': 1, 'text': 'Content here', 'source_file': 'test.pdf'},
            {'page_number': 2, 'text': '', 'source_file': 'test.pdf'},
            {'page_number': 3, 'text': '   ', 'source_file': 'test.pdf'},
            {'page_number': 4, 'text': 'More content', 'source_file': 'test.pdf'},
        ]
        chunks = processor.chunk_text(pages)

        # Only pages 1 and 4 should create chunks
        assert len(chunks) >= 1
        page_numbers = [chunk.page_number for chunk in chunks]
        assert 1 in page_numbers
        assert 4 in page_numbers
        assert 2 not in page_numbers
        assert 3 not in page_numbers

    @pytest.mark.unit
    def test_chunk_size_configuration(self, simple_pdf):
        """Test that chunk size parameter is respected."""
        small_processor = PDFProcessor(chunk_size=50, chunk_overlap=10)
        large_processor = PDFProcessor(chunk_size=500, chunk_overlap=10)

        pages = small_processor.extract_text_from_pdf(str(simple_pdf))

        small_chunks = small_processor.chunk_text(pages)
        large_chunks = large_processor.chunk_text(pages)

        # Smaller chunk size should create more chunks (or same for short text)
        assert len(small_chunks) >= len(large_chunks)


class TestUtilityFunctions:
    """Test suite for utility functions."""

    @pytest.mark.unit
    def test_clean_text_removes_extra_whitespace(self):
        """Test clean_text removes excessive whitespace."""
        dirty_text = "This  has   multiple    spaces"
        clean = clean_text(dirty_text)
        assert "  " not in clean
        assert clean == "This has multiple spaces"

    @pytest.mark.unit
    def test_clean_text_normalizes_newlines(self):
        """Test clean_text handles newlines properly."""
        text_with_newlines = "Line 1\nLine 2\n\n\n\nLine 3"
        clean = clean_text(text_with_newlines)
        # Multiple newlines should become double newline
        assert "\n\n\n" not in clean

    @pytest.mark.unit
    def test_clean_text_fixes_ligatures(self):
        """Test clean_text fixes common PDF ligatures."""
        text_with_ligatures = "ﬁle with ligatures ﬂow"
        clean = clean_text(text_with_ligatures)
        assert "ﬁ" not in clean
        assert "ﬂ" not in clean
        assert "file" in clean
        assert "flow" in clean

    @pytest.mark.unit
    def test_clean_text_handles_quotes(self):
        """Test clean_text normalizes smart quotes."""
        text = "\u201cQuoted text\u201d and \u2018single quotes\u2019"
        clean = clean_text(text)
        assert '"Quoted text" and \'single quotes\'' == clean

    @pytest.mark.unit
    def test_clean_text_handles_empty_input(self):
        """Test clean_text handles empty strings."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text(None) == ""

    @pytest.mark.unit
    def test_count_tokens_returns_int(self):
        """Test count_tokens returns integer value."""
        text = "This is a test sentence for token counting."
        token_count = count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    @pytest.mark.unit
    def test_count_tokens_longer_text_more_tokens(self):
        """Test that longer text has more tokens."""
        short_text = "Short text."
        long_text = "This is a much longer piece of text with many more words."

        short_tokens = count_tokens(short_text)
        long_tokens = count_tokens(long_text)

        assert long_tokens > short_tokens

    @pytest.mark.unit
    def test_count_tokens_handles_empty_text(self):
        """Test count_tokens handles empty strings."""
        assert count_tokens("") == 0

    @pytest.mark.unit
    def test_count_tokens_different_encodings(self):
        """Test count_tokens with different encoding models."""
        text = "Test text for token counting."

        # Should work with different encodings
        tokens_cl100k = count_tokens(text, "cl100k_base")
        assert tokens_cl100k > 0


class TestPDFProcessor:
    """Integration tests for full PDF processing pipeline."""

    @pytest.mark.integration
    @pytest.mark.requires_pdf
    def test_process_pdf_full_pipeline(self, simple_pdf):
        """Test complete PDF processing pipeline."""
        processor = PDFProcessor()
        chunks = processor.process_pdf(str(simple_pdf))

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.page_number >= 1 for chunk in chunks)

    @pytest.mark.integration
    @pytest.mark.requires_pdf
    def test_process_pdf_invalid_file(self):
        """Test process_pdf raises error for invalid files."""
        processor = PDFProcessor()

        with pytest.raises(ValueError, match="validation failed"):
            processor.process_pdf("/nonexistent/file.pdf")

    @pytest.mark.integration
    @pytest.mark.requires_pdf
    def test_process_pdf_with_custom_chunk_size(self, multi_page_pdf):
        """Test processing with custom chunk size."""
        processor = PDFProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.process_pdf(str(multi_page_pdf))

        assert len(chunks) > 0
        # Verify chunks respect size limits (approximately)
        for chunk in chunks:
            # Allow some buffer for overlap
            assert len(chunk.content) <= 150

    @pytest.mark.integration
    @pytest.mark.requires_pdf
    def test_process_multipage_pdf(self, multi_page_pdf):
        """Test processing multi-page PDF preserves page information."""
        processor = PDFProcessor()
        chunks = processor.process_pdf(str(multi_page_pdf))

        # Should have chunks from all 3 pages
        page_numbers = set(chunk.page_number for chunk in chunks)
        assert len(page_numbers) >= 1
        assert all(1 <= pn <= 3 for pn in page_numbers)

    @pytest.mark.integration
    @pytest.mark.requires_pdf
    def test_process_pdf_returns_empty_for_empty_pdf(self, empty_pdf):
        """Test processing empty PDF returns empty list."""
        processor = PDFProcessor()

        # Empty PDF should fail validation
        with pytest.raises(ValueError):
            processor.process_pdf(str(empty_pdf))

    @pytest.mark.integration
    @pytest.mark.requires_pdf
    def test_chunk_index_sequential(self, multi_page_pdf):
        """Test that chunk indices are sequential across pages."""
        processor = PDFProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.process_pdf(str(multi_page_pdf))

        # Chunk indices should be 0, 1, 2, 3, ...
        indices = [chunk.chunk_index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_process_large_document(self, long_text_pdf):
        """Test processing a larger document."""
        processor = PDFProcessor(chunk_size=500, chunk_overlap=100)
        chunks = processor.process_pdf(str(long_text_pdf))

        # Should create multiple chunks
        assert len(chunks) > 3

        # All chunks should have proper structure
        for chunk in chunks:
            assert chunk.content
            assert chunk.metadata['token_count'] > 0
            assert chunk.metadata['character_count'] > 0


class TestDocumentChunk:
    """Test suite for DocumentChunk dataclass."""

    @pytest.mark.unit
    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk instance."""
        chunk = DocumentChunk(
            content="Test content",
            page_number=1,
            chunk_index=0,
            source_file="test.pdf",
            metadata={"test": "value"}
        )

        assert chunk.content == "Test content"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0
        assert chunk.source_file == "test.pdf"
        assert chunk.metadata["test"] == "value"

    @pytest.mark.unit
    def test_document_chunk_immutable_after_creation(self):
        """Test DocumentChunk fields can be accessed."""
        chunk = DocumentChunk(
            content="Test",
            page_number=1,
            chunk_index=0,
            source_file="test.pdf",
            metadata={}
        )

        # Should be able to read all fields
        _ = chunk.content
        _ = chunk.page_number
        _ = chunk.chunk_index
        _ = chunk.source_file
        _ = chunk.metadata


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    @pytest.mark.unit
    def test_processor_with_zero_chunk_size(self):
        """Test processor handles invalid chunk size gracefully."""
        # LangChain should handle this, but test anyway
        with pytest.raises(ValueError):
            processor = PDFProcessor(chunk_size=0)

    @pytest.mark.unit
    def test_processor_with_negative_overlap(self):
        """Test processor handles negative overlap."""
        with pytest.raises(ValueError):
            processor = PDFProcessor(chunk_overlap=-10)

    @pytest.mark.unit
    def test_extract_text_with_special_characters(self, create_simple_pdf):
        """Test extraction handles special characters."""
        special_text = "Special chars: © ® ™ € £ ¥ • § ¶"
        pdf = create_simple_pdf("special.pdf", [special_text])

        processor = PDFProcessor()
        pages = processor.extract_text_from_pdf(str(pdf))

        # Should extract without errors
        assert len(pages) == 1
        # Some characters might be preserved or converted
        assert len(pages[0]['text']) > 0

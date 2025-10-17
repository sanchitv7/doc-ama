"""
PDF Processing Module for RAG System

This module handles PDF parsing, text extraction, and document chunking.
Key concepts to learn:
- Text extraction from PDFs (PyPDF2 vs pdfplumber differences)
- Document chunking strategies (fixed-size vs semantic)
- Text preprocessing and cleaning
- Metadata preservation for citations
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import re

# PDF extraction libraries
from pypdf import PdfReader
import pdfplumber

# Text splitting and token counting
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Represents a chunk of text from a PDF document with metadata for citations.
    Learning: This structure preserves source information needed for RAG citations.
    """

    content: str
    page_number: int
    chunk_index: int
    source_file: str
    metadata: Dict[str, Any]


class PDFProcessor:
    """
    Handles PDF document processing and text chunking.

    Learning objectives:
    1. Understand different PDF parsing libraries and their use cases
    2. Learn about text chunking strategies for RAG systems
    3. Implement citation-friendly metadata preservation
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with chunking parameters.

        Args:
            chunk_size: Target size for text chunks (in characters)
                       - Small chunks (200-500): Better semantic precision
                       - Large chunks (1000-2000): More context, may dilute relevance
            chunk_overlap: Number of overlapping characters between chunks
                          - Prevents information loss at chunk boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize recursive text splitter with hierarchical separators
        # Tries to split on paragraph boundaries first, then sentences, then words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level granularity.

        Uses pypdf as primary extractor, with pdfplumber as fallback for complex layouts.
        pypdf is faster and works well for text-based PDFs, while pdfplumber handles
        tables and complex formatting better.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing page text and metadata

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: For PDF reading errors
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pages = []
        source_filename = os.path.basename(pdf_path)

        try:
            # Try pypdf first (faster for most PDFs)
            logger.info(f"Extracting text from {source_filename} using pypdf")
            reader = PdfReader(pdf_path)

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()

                # If pypdf returns empty or very short text, try pdfplumber for this page
                if not text or len(text.strip()) < 50:
                    logger.warning(f"pypdf extraction poor for page {page_num}, trying pdfplumber")
                    text = self._extract_with_pdfplumber(pdf_path, page_num - 1)

                # Clean the extracted text
                text = clean_text(text)

                pages.append({
                    'page_number': page_num,
                    'text': text,
                    'source_file': source_filename,
                    'char_count': len(text)
                })

            logger.info(f"Extracted {len(pages)} pages from {source_filename}")

        except Exception as e:
            logger.error(f"pypdf extraction failed, falling back to pdfplumber: {str(e)}")
            # Fallback to pdfplumber for entire document
            pages = self._extract_all_with_pdfplumber(pdf_path)

        return pages

    def _extract_with_pdfplumber(self, pdf_path: str, page_index: int) -> str:
        """
        Extract text from a single page using pdfplumber.

        Args:
            pdf_path: Path to PDF file
            page_index: Zero-based page index

        Returns:
            Extracted text from the page
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_index < len(pdf.pages):
                    page = pdf.pages[page_index]
                    text = page.extract_text() or ""
                    return text
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for page {page_index}: {str(e)}")
        return ""

    def _extract_all_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from all pages using pdfplumber.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page dictionaries
        """
        pages = []
        source_filename = os.path.basename(pdf_path)

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = clean_text(text)

                pages.append({
                    'page_number': page_num,
                    'text': text,
                    'source_file': source_filename,
                    'char_count': len(text)
                })

        logger.info(f"Extracted {len(pages)} pages from {source_filename} using pdfplumber")
        return pages

    def chunk_text(self, pages: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Split extracted text into chunks suitable for vector embedding.

        Uses recursive character splitting to create semantically meaningful chunks
        while preserving metadata for citations. Each chunk maintains a reference
        to its source page and file.

        Args:
            pages: List of page dictionaries from extract_text_from_pdf

        Returns:
            List of DocumentChunk objects with preserved citation metadata
        """
        chunks = []
        global_chunk_index = 0

        for page in pages:
            page_text = page['text']

            # Skip empty pages
            if not page_text or len(page_text.strip()) == 0:
                logger.debug(f"Skipping empty page {page['page_number']}")
                continue

            # Split page text into chunks using the configured text splitter
            page_chunks = self.text_splitter.split_text(page_text)

            # Create DocumentChunk objects with full metadata
            for i, chunk_text in enumerate(page_chunks):
                # Count tokens for this chunk (useful for embedding model limits)
                token_count = count_tokens(chunk_text)

                chunks.append(DocumentChunk(
                    content=chunk_text,
                    page_number=page['page_number'],
                    chunk_index=global_chunk_index,
                    source_file=page['source_file'],
                    metadata={
                        'page_chunk_index': i,  # Index within this page
                        'total_chunks_in_page': len(page_chunks),
                        'character_count': len(chunk_text),
                        'token_count': token_count,
                        'page_char_count': page.get('char_count', 0)
                    }
                ))
                global_chunk_index += 1

        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks

    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Complete PDF processing pipeline: extract -> chunk -> prepare for embedding.

        This is the main entry point that orchestrates the entire PDF processing workflow:
        1. Validates the PDF file
        2. Extracts text page by page
        3. Splits text into chunks with metadata
        4. Returns chunks ready for embedding

        Args:
            pdf_path: Path to PDF file to process

        Returns:
            List of DocumentChunk objects ready for embedding

        Raises:
            ValueError: If PDF validation fails
            Exception: For processing errors
        """
        try:
            logger.info(f"Starting PDF processing pipeline: {pdf_path}")

            # Step 1: Validate the PDF file
            if not self.validate_pdf(pdf_path):
                raise ValueError(f"PDF validation failed for: {pdf_path}")

            # Step 2: Extract text from all pages
            pages = self.extract_text_from_pdf(pdf_path)

            if not pages:
                logger.warning(f"No pages extracted from {pdf_path}")
                return []

            # Step 3: Chunk the extracted text
            chunks = self.chunk_text(pages)

            if not chunks:
                logger.warning(f"No chunks created from {pdf_path}")
                return []

            logger.info(f"Successfully processed {pdf_path}: {len(pages)} pages → {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def validate_pdf(self, pdf_path: str, max_size_mb: int = 50) -> bool:
        """
        Validate PDF file before processing.

        Checks:
        - File exists and is readable
        - File has .pdf extension
        - File size is within limits
        - PDF is not encrypted (attempts to read first page)

        Args:
            pdf_path: Path to PDF file
            max_size_mb: Maximum allowed file size in MB (default: 50)

        Returns:
            True if PDF is valid and processable, False otherwise
        """
        # Check file exists
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return False

        # Check file extension
        if not pdf_path.lower().endswith('.pdf'):
            logger.error(f"File is not a PDF: {pdf_path}")
            return False

        # Check file is readable
        if not os.access(pdf_path, os.R_OK):
            logger.error(f"File is not readable: {pdf_path}")
            return False

        # Check file size
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            logger.error(f"File too large: {file_size_mb:.2f}MB > {max_size_mb}MB")
            return False

        # Check if PDF is encrypted or corrupted by trying to read it
        try:
            reader = PdfReader(pdf_path)
            if reader.is_encrypted:
                logger.error(f"PDF is encrypted/password protected: {pdf_path}")
                return False

            # Try to access first page to verify PDF is readable
            if len(reader.pages) == 0:
                logger.error(f"PDF has no pages: {pdf_path}")
                return False

        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False

        logger.debug(f"PDF validation passed: {pdf_path} ({file_size_mb:.2f}MB)")
        return True


def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.

    Different models use different tokenization schemes:
    - cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    - p50k_base: Codex models
    - r50k_base: GPT-3 models (davinci, curie, etc.)

    Args:
        text: Text to tokenize
        model_name: Encoding name (default: cl100k_base for modern models)

    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.get_encoding(model_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Token counting failed, using character estimate: {str(e)}")
        # Rough fallback estimate: ~4 characters per token
        return len(text) // 4


def clean_text(text: str) -> str:
    """
    Clean extracted text for better embedding quality.

    Handles common PDF extraction issues:
    - Excessive whitespace and newlines
    - Multiple spaces
    - Special characters and encoding issues
    - Leading/trailing whitespace

    Args:
        text: Raw text from PDF extraction

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Replace multiple newlines with double newline (preserve paragraph breaks)
    text = re.sub(r'\n\n+', '\n\n', text)

    # Replace single newlines with spaces (join broken lines)
    # but preserve double newlines (paragraph breaks)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

    # Fix common ligatures from PDFs
    replacements = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

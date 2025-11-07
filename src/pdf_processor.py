"""
PDF Processing Module for RAG System

This module handles PDF parsing, text extraction, and document chunking.
Key concepts to learn:
- Text extraction from PDFs using pypdf
- Document chunking strategies (fixed-size vs semantic)
- Text preprocessing and cleaning
- Metadata preservation for citations
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# TODO #1: Uncomment these imports as you implement each component
# from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import tiktoken

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
    1. Understand PDF text extraction using pypdf
    2. Learn about text chunking strategies for RAG systems
    3. Implement citation-friendly metadata preservation
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with chunking parameters.

        TODO #2: Learn about optimal chunk sizes for different use cases:
        - Small chunks (200-500): Better semantic precision
        - Large chunks (1000-2000): More context, may dilute relevance
        - Overlap: Prevents information loss at chunk boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # TODO #3: Initialize text splitter
        # Research: RecursiveCharacterTextSplitter vs other splitting strategies
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level granularity.

        TODO #5: Implement text extraction logic
        Learning resources:
        - pypdf documentation: https://pypdf.readthedocs.io/

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing page text and metadata

        TODO #5 Implementation steps:
        1. Validate PDF file exists and is readable
        2. Use pypdf PdfReader to open the PDF
        3. Extract text page by page using page.extract_text()
        4. Preserve page numbers and other metadata
        5. Handle extraction errors gracefully

        Note: If you encounter complex PDFs with poor extraction quality,
        consider adding pdfplumber as a fallback library later.
        """
        pages = []

        # TODO #5: Implement extraction logic here
        # Example structure:
        # for page_num, page in enumerate(pdf_pages):
        #     pages.append({
        #         'page_number': page_num + 1,
        #         'text': extracted_text,
        #         'source_file': os.path.basename(pdf_path)
        #     })

        return pages

    def chunk_text(self, pages: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Split extracted text into chunks suitable for vector embedding.

        TODO #6: Implement chunking logic
        Learning concepts:
        - Token counting with tiktoken
        - Semantic vs syntactic chunking
        - Preserving context across chunk boundaries

        Args:
            pages: List of page dictionaries from extract_text_from_pdf

        Returns:
            List of DocumentChunk objects

        TODO #6 Implementation steps:
        1. Combine or process pages based on chunking strategy
        2. Use text splitter to create chunks
        3. Calculate token counts to stay within embedding limits
        4. Create DocumentChunk objects with proper metadata
        5. Ensure each chunk has citation information
        """
        chunks = []

        # TODO #6: Implement chunking logic
        # for page in pages:
        #     page_chunks = self.text_splitter.split_text(page['text'])
        #     for i, chunk_text in enumerate(page_chunks):
        #         chunks.append(DocumentChunk(
        #             content=chunk_text,
        #             page_number=page['page_number'],
        #             chunk_index=i,
        #             source_file=page['source_file'],
        #             metadata={
        #                 'total_chunks_in_page': len(page_chunks),
        #                 'character_count': len(chunk_text)
        #             }
        #         ))

        return chunks

    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Complete PDF processing pipeline: extract -> chunk -> prepare for embedding.

        TODO #7: Implement full pipeline
        Learning: This is the main entry point that orchestrates the entire process

        Args:
            pdf_path: Path to PDF file to process

        Returns:
            List of DocumentChunk objects ready for embedding

        TODO #7 Implementation:
        1. Validate input file
        2. Extract text from PDF
        3. Chunk the extracted text
        4. Add any additional preprocessing
        5. Return processed chunks
        """
        try:
            # TODO #7: Implement pipeline
            # pages = self.extract_text_from_pdf(pdf_path)
            # chunks = self.chunk_text(pages)
            # return chunks

            logger.info(f"Processing PDF: {pdf_path}")
            # Placeholder return
            return []

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate PDF file before processing.

        TODO #4: Implement validation
        - Check file exists and is readable
        - Verify it's a valid PDF
        - Check file size limits
        - Ensure file is not encrypted/password protected
        """
        # TODO #4: Implement validation logic
        return os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf")


# TODO #9: Additional utility functions to implement
def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text for the specified model.
    Learning: Different models have different tokenization schemes
    """
    # TODO #9: Implement token counting with tiktoken
    pass


def clean_text(text: str) -> str:
    """
    Clean extracted text for better embedding quality.

    TODO #8: Implement text cleaning:
    - Remove excessive whitespace
    - Fix common PDF extraction issues
    - Handle special characters
    - Normalize text encoding
    """
    # TODO #8: Implement text cleaning
    return text.strip()

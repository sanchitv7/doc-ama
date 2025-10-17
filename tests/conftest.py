"""
Pytest configuration and shared fixtures.

This file contains fixtures that are available to all test modules.
"""

import os
import pytest
import tempfile
from pathlib import Path
from pypdf import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_text():
    """Sample text for testing chunking and processing."""
    return """
    This is a sample document for testing the PDF processing functionality.
    It contains multiple paragraphs to test text extraction and chunking.

    The second paragraph discusses the importance of proper text handling
    in RAG systems. Text chunking must preserve semantic meaning while
    staying within token limits for embedding models.

    Finally, the third paragraph emphasizes citation tracking. Every chunk
    must maintain references to its source page and document for proper
    attribution in the generated responses.
    """.strip()


@pytest.fixture
def create_simple_pdf(temp_dir):
    """
    Factory fixture to create simple test PDFs.

    Usage:
        pdf_path = create_simple_pdf("test.pdf", ["Page 1 text", "Page 2 text"])
    """
    def _create_pdf(filename: str, page_texts: list[str]) -> Path:
        pdf_path = temp_dir / filename

        # Create PDF using reportlab
        c = canvas.Canvas(str(pdf_path), pagesize=letter)

        for page_text in page_texts:
            # Add text to page
            text_object = c.beginText(50, 750)
            text_object.setFont("Helvetica", 12)

            # Handle multi-line text
            for line in page_text.split('\n'):
                text_object.textLine(line)

            c.drawText(text_object)
            c.showPage()

        c.save()
        return pdf_path

    return _create_pdf


@pytest.fixture
def simple_pdf(create_simple_pdf):
    """A simple single-page PDF for basic tests."""
    return create_simple_pdf(
        "simple.pdf",
        ["This is a simple test PDF with one page."]
    )


@pytest.fixture
def multi_page_pdf(create_simple_pdf):
    """A multi-page PDF for testing page extraction."""
    pages = [
        "This is page one. It contains some introductory text.",
        "This is page two. It has different content from page one.",
        "This is page three. The final page with concluding remarks."
    ]
    return create_simple_pdf("multipage.pdf", pages)


@pytest.fixture
def long_text_pdf(create_simple_pdf):
    """A PDF with long text to test chunking."""
    long_text = " ".join([
        f"This is sentence number {i}. It is part of a longer document."
        for i in range(100)
    ])
    return create_simple_pdf("longtext.pdf", [long_text])


@pytest.fixture
def empty_pdf(temp_dir):
    """Create an empty PDF (0 pages)."""
    pdf_path = temp_dir / "empty.pdf"
    writer = PdfWriter()
    with open(pdf_path, 'wb') as f:
        writer.write(f)
    return pdf_path


@pytest.fixture
def non_pdf_file(temp_dir):
    """Create a non-PDF file for validation testing."""
    file_path = temp_dir / "not_a_pdf.txt"
    file_path.write_text("This is not a PDF file.")
    return file_path


@pytest.fixture
def large_pdf(create_simple_pdf):
    """Create a large PDF (>50MB) for size limit testing."""
    # Create a PDF with many pages to exceed size limit
    # Note: This is simulated - actual large file creation is resource intensive
    return None  # Tests should check for None and skip if needed


@pytest.fixture(scope="session")
def fixtures_dir():
    """Path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"

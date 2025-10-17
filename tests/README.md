# Test Suite Documentation

## Overview

This directory contains the test suite for the doc-ama RAG PDF Q&A system. Tests follow industry-standard practices using pytest and are organized by module.

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared fixtures and pytest configuration
├── test_pdf_processor.py    # PDF processing tests (37 tests)
├── fixtures/                # Test data and sample files
└── README.md               # This file
```

## Running Tests

**IMPORTANT:** Always activate the virtual environment before running tests!

### Quick Start (Recommended)
```bash
# Use the test runner script (automatically activates venv)
./run_tests.sh -v

# Or activate venv manually first
source .venv/bin/activate
python -m pytest tests/ -v
```

### Run All Tests
```bash
# Option 1: Using test runner script (recommended)
./run_tests.sh tests/ -v

# Option 2: Using uv run (handles venv automatically)
uv run pytest tests/ -v

# Option 3: Manual activation
source .venv/bin/activate
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
./run_tests.sh tests/test_pdf_processor.py -v
# OR
source .venv/bin/activate && python -m pytest tests/test_pdf_processor.py -v
```

### Run with Coverage
```bash
./run_tests.sh tests/ -v --cov=src --cov-report=term-missing
# OR
source .venv/bin/activate && python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run Only Unit Tests
```bash
./run_tests.sh tests/ -v -m unit
```

### Run Only Integration Tests
```bash
./run_tests.sh tests/ -v -m integration
```

### Stop on First Failure
```bash
./run_tests.sh tests/ -v -x
```

## Test Categories

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.unit` - Unit tests for individual functions/methods
- `@pytest.mark.integration` - Integration tests for multiple components
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.requires_pdf` - Tests that require PDF files

## Current Test Coverage

### PDF Processor Module (`test_pdf_processor.py`)

**Status: ✅ 37/37 tests passing | 81% code coverage**

#### Test Suites:

1. **TestPDFValidation** (6 tests)
   - File existence validation
   - PDF format validation
   - File size limits
   - Encryption detection
   - Empty PDF handling
   - File permission checks

2. **TestTextExtraction** (4 tests)
   - Single-page extraction
   - Multi-page extraction
   - Metadata preservation
   - Error handling for missing files

3. **TestTextChunking** (6 tests)
   - Single page chunking
   - Long text chunking
   - Metadata preservation in chunks
   - Chunk overlap verification
   - Empty page skipping
   - Chunk size configuration

4. **TestUtilityFunctions** (8 tests)
   - Text cleaning (whitespace, newlines, ligatures)
   - Quote normalization
   - Empty input handling
   - Token counting accuracy
   - Different encoding support

5. **TestPDFProcessor** (7 tests - Integration)
   - Full processing pipeline
   - Invalid file handling
   - Custom chunk size configuration
   - Multi-page document processing
   - Empty PDF handling
   - Sequential chunk indexing
   - Large document processing

6. **TestDocumentChunk** (2 tests)
   - Dataclass creation
   - Field accessibility

7. **TestEdgeCases** (3 tests)
   - Zero chunk size handling
   - Negative overlap handling
   - Special character extraction

## Fixtures

The `conftest.py` file provides reusable fixtures:

### PDF Fixtures
- `simple_pdf` - Single-page PDF for basic tests
- `multi_page_pdf` - 3-page PDF for pagination tests
- `long_text_pdf` - Long text for chunking tests
- `empty_pdf` - Empty PDF for validation tests
- `create_simple_pdf` - Factory fixture to create custom test PDFs

### Utility Fixtures
- `temp_dir` - Temporary directory for test files
- `sample_text` - Sample text for testing
- `non_pdf_file` - Non-PDF file for validation tests
- `fixtures_dir` - Path to fixtures directory

## Coverage Report

### src/pdf_processor.py: 81% Coverage

**Covered:**
- PDF validation logic
- Text extraction (pypdf)
- Text chunking with LangChain
- Metadata preservation
- Error handling
- Text cleaning utilities
- Token counting

**Not Covered (19%):**
- pdfplumber fallback paths (lines 125-128, 149-151, 163-179)
- Some error logging branches (lines 263-264, 270-271, 322-323, 330-332)
- Token counting fallback (lines 358-361)

**Note:** Missing coverage is primarily in fallback/error paths that are harder to trigger in tests.

## Test Configuration

### pytest.ini
- Minimum Python version: 6.0
- Test discovery pattern: `test_*.py`
- Coverage thresholds and reporting
- Custom markers configuration
- Output formatting options

### Dependencies
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `reportlab` - PDF generation for test fixtures

## Writing New Tests

### Best Practices

1. **Use Descriptive Names**
   ```python
   def test_extract_single_page_preserves_page_number():
       # Clear what is being tested
   ```

2. **Follow AAA Pattern**
   ```python
   def test_something():
       # Arrange
       processor = PDFProcessor()

       # Act
       result = processor.process_pdf("test.pdf")

       # Assert
       assert len(result) > 0
   ```

3. **Use Fixtures for Test Data**
   ```python
   def test_with_fixture(simple_pdf):
       # Fixture provides clean test PDF
       processor = PDFProcessor()
       result = processor.extract_text_from_pdf(str(simple_pdf))
   ```

4. **Mark Tests Appropriately**
   ```python
   @pytest.mark.unit
   @pytest.mark.requires_pdf
   def test_pdf_processing():
       ...
   ```

5. **Test Edge Cases**
   - Empty inputs
   - None values
   - Boundary conditions
   - Error conditions

### Example Test Template

```python
import pytest
from src.your_module import YourClass

class TestYourClass:
    """Test suite for YourClass functionality."""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic use case."""
        # Arrange
        obj = YourClass()

        # Act
        result = obj.method()

        # Assert
        assert result is not None

    @pytest.mark.unit
    def test_error_handling(self):
        """Test error handling."""
        obj = YourClass()

        with pytest.raises(ValueError):
            obj.method_that_should_fail()
```

## Future Tests

### Planned Test Modules

- `test_vector_store.py` - ChromaDB and embedding tests
- `test_llm_client.py` - OpenRouter API integration tests
- `test_rag_system.py` - End-to-end RAG pipeline tests
- `test_gradio_ui.py` - UI component tests

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions snippet
- name: Run Tests
  run: |
    uv run pytest tests/ -v --cov=src --cov-report=xml
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure dependencies are installed
uv sync
uv pip install pytest pytest-cov pytest-mock reportlab pypdf
```

**PDF Generation Fails:**
```bash
# Check reportlab is installed
uv pip install reportlab
```

**Coverage Not Showing:**
```bash
# Run with explicit coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## Test Results Summary

| Module | Tests | Passed | Failed | Coverage |
|--------|-------|--------|--------|----------|
| pdf_processor | 37 | 37 ✅ | 0 | 81% |
| vector_store | - | - | - | 0% (not implemented) |
| llm_client | - | - | - | 0% (not implemented) |
| rag_system | - | - | - | 0% (not implemented) |

**Last Updated:** 2025-10-17
**Total Tests:** 37
**Pass Rate:** 100% ✅

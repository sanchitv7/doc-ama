# CLAUDE.md - AI Assistant Context Guide

## Project Overview

**doc-ama** is a learning-focused RAG (Retrieval-Augmented Generation) PDF Q&A system. This is a scaffolding project designed for learning modern AI application development, specifically RAG systems and agentic coding practices.

**Current Status:** 🚧 Scaffolding phase - Most functionality is stubbed out with detailed TODOs

**Key Philosophy:** Learn by doing - implement features incrementally while understanding the underlying concepts.

## Architecture

```
User → Gradio UI → RAG System → {Vector Store, LLM Client} → Response with Citations
                        ↓
                   PDF Processor
```

### Core Components

1. **PDF Processor** ([src/pdf_processor.py](src/pdf_processor.py))
   - Extracts text from PDF files (pypdf)
   - Chunks text for embedding (using LangChain text splitters)
   - Preserves metadata for citations (page numbers, source files)
   - **Status:** TODOs #1-9 not implemented

2. **Vector Store** ([src/vector_store.py](src/vector_store.py))
   - ChromaDB for persistent vector storage
   - Sentence Transformers for embeddings (default: all-MiniLM-L6-v2)
   - Semantic search with cosine similarity
   - **Status:** All methods stubbed with detailed TODOs

3. **LLM Client** ([src/llm_client.py](src/llm_client.py))
   - OpenRouter API integration (OpenAI-compatible)
   - Prompt engineering for RAG with citations
   - Token counting and cost management
   - **Status:** Placeholder responses, needs implementation

4. **RAG System** ([src/rag_system.py](src/rag_system.py))
   - Orchestrates retrieval + generation pipeline
   - Query analysis and context selection
   - Citation extraction and confidence scoring
   - **Status:** Pipeline structure present, core logic stubbed

5. **Gradio UI** ([src/gradio_ui.py](src/gradio_ui.py))
   - Web interface for document upload and Q&A
   - **Status:** Not yet examined/implemented

6. **Main Application** ([main.py](main.py))
   - Application orchestration and dependency injection
   - Configuration management via environment variables
   - System health validation
   - **Status:** Configuration loading works, component initialization stubbed

## Tech Stack

- **Frontend:** Gradio 4.0+
- **Vector DB:** ChromaDB 0.4+ (persistent storage)
- **Embeddings:** Sentence Transformers (local, no API)
- **LLM:** OpenRouter API (multi-model proxy)
- **PDF Processing:** pypdf + LangChain
- **Python:** 3.13+

## Configuration

Environment variables (`.env` file):

```bash
# Required
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL=meta-llama/llama-3.1-8b-instruct:free

# Optional
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=5
MAX_PDF_SIZE_MB=50
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
GRADIO_SHARE=false
```

**Note:** `.env` file exists and is gitignored. `.env.template` provided for reference.

## Project Structure

```
doc-ama/
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF extraction & chunking
│   ├── vector_store.py       # ChromaDB & embeddings
│   ├── llm_client.py         # OpenRouter integration
│   ├── rag_system.py         # RAG pipeline orchestration
│   └── gradio_ui.py          # Web interface
├── main.py                   # Application entry point
├── requirements.txt          # Pip dependencies
├── pyproject.toml            # UV/Poetry config
├── uv.lock                   # UV lockfile
├── LEARNING_PLAN.md          # 8-week curriculum
├── README.md                 # User-facing docs
└── CLAUDE.md                 # This file
```

## Implementation Status

### ✅ Completed
- Project scaffolding and structure
- Comprehensive documentation (README, LEARNING_PLAN)
- Configuration management system
- Logging setup
- Data class definitions (DocumentChunk, LLMResponse, RAGResponse, Citation)

### 🚧 In Progress / Not Started
- All PDF processing logic
- ChromaDB integration and embedding generation
- OpenRouter API calls (stubs return placeholders)
- RAG pipeline implementation
- Gradio UI
- End-to-end testing

## Development Approach

### When Implementing Features:

1. **Read the TODOs first** - Each module has detailed implementation guidance
2. **Uncomment imports** as you implement components
3. **Follow the learning plan** - [LEARNING_PLAN.md](LEARNING_PLAN.md) has 8-week curriculum
4. **Test incrementally** - Don't try to implement everything at once
5. **Preserve metadata** - Citations require page numbers and source tracking

### Key Design Patterns:

- **Dataclasses for structure:** DocumentChunk, LLMResponse, RAGResponse, Citation
- **Component injection:** RAGSystem receives VectorStore and LLMClient
- **Configuration centralization:** All settings via environment variables
- **Citation-first design:** Metadata flows from PDF → chunks → retrieval → response

### Common Tasks:

**To add PDF processing:**
1. Implement `PDFProcessor.extract_text_from_pdf()` (TODO #5)
2. Implement `PDFProcessor.chunk_text()` (TODO #6)
3. Uncomment imports: pypdf PdfReader, RecursiveCharacterTextSplitter

**To enable vector search:**
1. Initialize ChromaDB client in `VectorStore.__init__`
2. Implement `embed_chunks()` with Sentence Transformers
3. Implement `add_documents()` and `search()` methods
4. Uncomment: chromadb, sentence-transformers imports

**To connect LLM:**
1. Initialize OpenAI client (OpenRouter-compatible) in `OpenRouterClient.__init__`
2. Implement `generate_response()` with proper error handling
3. Implement `create_rag_prompt()` with citation instructions
4. Uncomment: openai, tiktoken imports

## Git Status

```
Current branch: master
Main branch: (not configured)

Modified:
- LEARNING_PLAN.md
- src/pdf_processor.py

Untracked:
- .github/ (new directory)

Recent commits:
- 1d4225e: Update README.md
- b9cc81a: feat: initialize RAG PDF Q&A system
```

## Learning Resources

**Primary Guide:** [LEARNING_PLAN.md](LEARNING_PLAN.md) contains:
- 8-week learning curriculum
- Phase-by-phase implementation roadmap
- Hands-on exercises for each concept
- Debugging and troubleshooting guides
- Evaluation metrics and success criteria

**External Resources:**
- OpenRouter: https://openrouter.ai/docs
- ChromaDB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/

## Important Notes for AI Assistants

### When Helping with This Project:

1. **Respect the learning goal** - Don't implement everything at once. Guide towards incremental understanding.

2. **Explain concepts** - Each TODO references learning objectives. Connect implementation to theory.

3. **Preserve TODOs** - Don't remove instructional comments when implementing. They're teaching aids.

4. **Citation awareness** - This is a citation-focused RAG system. Metadata preservation is critical throughout the pipeline.

5. **Configuration first** - Check `.env` is set up before suggesting code changes that require API keys.

6. **Test data needed** - User will need sample PDFs to test. Suggest using public domain documents.

### Common Pitfalls to Avoid:

- ❌ Implementing all features at once (overwhelming for learner)
- ❌ Removing educational TODOs and comments
- ❌ Skipping configuration validation before testing
- ❌ Forgetting to preserve page numbers in chunking
- ❌ Not handling API errors gracefully
- ❌ Ignoring token limits and costs

### Suggested Implementation Order:

1. **Sprint 1:** Basic pipeline (PDF → embedding → search → placeholder LLM)
2. **Sprint 2:** Real LLM integration + citation system
3. **Sprint 3:** Gradio UI and user experience
4. **Sprint 4:** Optimization and advanced features

See [LEARNING_PLAN.md](LEARNING_PLAN.md) for detailed sprint breakdown.

## Testing Strategy

**Unit Testing:** Not yet implemented
**Integration Testing:** Not yet implemented

**Recommended Manual Testing Flow:**
1. Start with a single, simple PDF (e.g., 5 pages, text-based)
2. Test PDF extraction → verify text quality
3. Test chunking → verify metadata preservation
4. Test embedding + storage → verify ChromaDB persistence
5. Test search → verify similarity ranking
6. Test LLM call → verify API connectivity
7. Test full RAG pipeline → verify citations
8. Test UI → verify end-to-end UX

## Dependencies Status

**Package Manager:** Using `uv` (lockfile present: `uv.lock`)

**Key Dependencies:**
- gradio >= 4.0.0 ✓
- chromadb >= 0.4.0 ✓
- openai >= 1.0.0 ✓ (OpenRouter compatible)
- sentence-transformers >= 2.2.0 ✓
- pypdf >= 3.0.0 ✓
- langchain >= 0.1.0 ✓
- tiktoken >= 0.5.0 ✓

**Installation:**
```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

## Troubleshooting Quick Reference

**If PDF extraction fails:**
- Check file is not encrypted/password-protected
- Verify file path is absolute
- Check PDF is text-based (not scanned)
- Consider adding pdfplumber for complex layouts if needed

**If embeddings fail:**
- Check ChromaDB persist directory is writable
- Verify embedding model downloaded (first run downloads model)
- Monitor memory usage for large documents

**If LLM calls fail:**
- Verify OPENROUTER_API_KEY in `.env`
- Check API key validity: https://openrouter.ai/keys
- Review OpenRouter rate limits and quotas
- Check model availability (free models may have limits)

**If citations missing:**
- Verify page numbers preserved in DocumentChunk
- Check prompt includes citation instructions
- Review LLM response parsing logic

## Next Steps for Implementation

**Immediate priorities:**
1. ✅ Create CLAUDE.md (this file)
2. 🔲 Implement PDF text extraction (TODO #5 in pdf_processor.py)
3. 🔲 Implement text chunking (TODO #6 in pdf_processor.py)
4. 🔲 Set up ChromaDB and embeddings (vector_store.py)
5. 🔲 Test basic retrieval pipeline

**Follow:** [LEARNING_PLAN.md](LEARNING_PLAN.md) Sprint 1 (Days 1-7) for detailed roadmap.

---

**Last Updated:** 2025-09-29
**Project Version:** 0.1.0 (scaffolding)
**Python Version:** 3.13+
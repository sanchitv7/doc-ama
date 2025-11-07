# 🚀 3-Day RAG Implementation Checklist

**Goal:** Complete functional RAG PDF Q&A system in 3 days

---

## 📋 DAY 1: Core Pipeline (6-8 hours)

### Morning Session: PDF Processing (3 hours)

- [ ] **Setup (30 min)**
  - [ ] Activate virtual environment: `source .venv/bin/activate`
  - [ ] Verify `.env` file has `OPENROUTER_API_KEY`
  - [ ] Install dependencies: `uv pip install -r requirements.txt`
  - [ ] Download a sample PDF (3-5 pages) for testing

- [ ] **PDFProcessor Implementation (2 hours)**
  - [ ] File: `src/pdf_processor.py`
  - [ ] Uncomment imports (pypdf, pdfplumber)
  - [ ] Implement `extract_text_from_pdf()` - TODO #5
  - [ ] Implement `chunk_text()` - TODO #6
  - [ ] Implement `process_pdf()` - TODO #7
  - [ ] Test: `python -c "from src.pdf_processor import PDFProcessor; p = PDFProcessor(); chunks = p.process_pdf('test.pdf'); print(f'{len(chunks)} chunks extracted')"`

**Checkpoint:** Can extract and chunk PDF text with page numbers ✓

---

### Afternoon Session: Vector Database (3-4 hours)

- [ ] **VectorStore Implementation (3.5 hours)**
  - [ ] File: `src/vector_store.py`
  - [ ] Uncomment imports (chromadb, SentenceTransformer)
  - [ ] Implement `__init__()` - Initialize ChromaDB client
  - [ ] Implement `_get_or_create_collection()` - Setup collection
  - [ ] Implement `embed_chunks()` - Generate embeddings
  - [ ] Implement `add_documents()` - Store in ChromaDB
  - [ ] Implement `search()` - Similarity search
  - [ ] Test embedding generation with sample chunks
  - [ ] Test search returns relevant results

**Checkpoint:** Can store chunks and retrieve via semantic search ✓

---

### Evening Session: LLM Integration (1-2 hours)

- [ ] **OpenRouterClient Basics (1.5 hours)**
  - [ ] File: `src/llm_client.py`
  - [ ] Uncomment imports (openai)
  - [ ] Implement `__init__()` - Initialize OpenAI client
  - [ ] Implement `validate_api_key()` - Test connection
  - [ ] Implement `generate_response()` - Basic LLM call
  - [ ] Test: Send "Hello, world!" prompt, verify response

**Day 1 Success Criteria:**
- ✅ PDF → Chunks with page numbers
- ✅ Chunks stored in ChromaDB
- ✅ Semantic search works
- ✅ LLM responds to basic prompts

---

## 📋 DAY 2: RAG Pipeline + Citations (6-8 hours)

### Morning Session: RAG Orchestration (3 hours)

- [ ] **RAGSystem Retrieval (2 hours)**
  - [ ] File: `src/rag_system.py`
  - [ ] Implement `retrieve_relevant_chunks()` - Call vector_store.search()
  - [ ] Implement `rank_and_select_context()` - Token management
  - [ ] Test retrieval with sample queries

- [ ] **RAG Prompt Engineering (1 hour)**
  - [ ] File: `src/llm_client.py`
  - [ ] Implement `create_rag_prompt()` - Format context + query
  - [ ] Implement `count_tokens()` - Token counting with tiktoken
  - [ ] Test: Full RAG query (no citations yet)

**Checkpoint:** Can retrieve relevant chunks and generate contextual responses ✓

---

### Afternoon Session: Citation System (3 hours)

- [ ] **Citation Implementation (3 hours)**
  - [ ] File: `src/rag_system.py`
  - [ ] Implement `_format_context_with_citations()` - Add [Source: X, Page: Y]
  - [ ] Implement `_extract_citations()` - Parse citation patterns
  - [ ] Implement `generate_response_with_citations()` - Full pipeline
  - [ ] Implement `_calculate_confidence_score()` - Basic scoring
  - [ ] Test: Verify citations appear in responses
  - [ ] Test: Citations match actual source pages

**Checkpoint:** RAG returns answers with proper source citations ✓

---

### Evening Session: Integration (2 hours)

- [ ] **Main Application Setup (2 hours)**
  - [ ] File: `main.py`
  - [ ] Uncomment all component imports
  - [ ] Implement `initialize_components()` - Initialize all modules
  - [ ] Implement `_validate_system_health()` - Health checks
  - [ ] Test via Python REPL:
    ```python
    from main import RAGApplication
    app = RAGApplication()
    app.initialize_components()
    result = app.rag_system.query("What is this document about?")
    print(result.answer)
    print(result.sources)
    ```

**Day 2 Success Criteria:**
- ✅ Complete RAG pipeline works
- ✅ Responses include citations
- ✅ Can query via code (no UI yet)

---

## 📋 DAY 3: Gradio UI + Polish (6-8 hours)

### Morning Session: Gradio Interface (3-4 hours)

- [ ] **Read Gradio UI Code (30 min)**
  - [ ] File: `src/gradio_ui.py`
  - [ ] Read through existing structure
  - [ ] Check Gradio docs: https://gradio.app/docs/

- [ ] **Implement Gradio UI (3 hours)**
  - [ ] Uncomment Gradio imports
  - [ ] Implement `_create_interface()` - Build UI with tabs
  - [ ] Implement `_handle_file_upload()` - PDF upload handler
  - [ ] Implement `_handle_question()` - Query handler
  - [ ] Test: Launch UI, upload PDF, ask question

**Checkpoint:** Basic web UI works for upload and Q&A ✓

---

### Afternoon Session: Polish & Features (2-3 hours)

- [ ] **UI Polish (2 hours)**
  - [ ] Implement `_refresh_system_status()` - Show DB stats
  - [ ] Add loading indicators for long operations
  - [ ] Format citations nicely in UI
  - [ ] Add error messages for failed uploads

- [ ] **Error Handling (1 hour)**
  - [ ] Add try-catch blocks in all modules
  - [ ] Implement graceful fallbacks
  - [ ] Add user-friendly error messages
  - [ ] Test with corrupted/encrypted PDFs

**Checkpoint:** UI is polished and handles errors gracefully ✓

---

### Evening Session: Testing & Documentation (1-2 hours)

- [ ] **End-to-End Testing (1 hour)**
  - [ ] Test with multiple PDFs (3+)
  - [ ] Test with different PDF types (text, complex layouts)
  - [ ] Test edge cases (empty PDFs, large files)
  - [ ] Test diverse queries (broad, specific, page-based)

- [ ] **Documentation (30 min)**
  - [ ] Update README with actual usage instructions
  - [ ] Document any known issues or quirks
  - [ ] Add screenshots/GIFs of UI (optional)

- [ ] **Demo Preparation (30 min)**
  - [ ] Prepare demo script
  - [ ] Test full demo flow
  - [ ] Note any improvements for future

**Day 3 Success Criteria:**
- ✅ Gradio UI fully functional
- ✅ Can upload PDFs and ask questions
- ✅ Citations display properly
- ✅ Error handling prevents crashes
- ✅ System ready to demo

---

## 🧪 Quick Testing Commands

### Test PDF Processing
```bash
python -c "from src.pdf_processor import PDFProcessor; p = PDFProcessor(); chunks = p.process_pdf('test.pdf'); print(f'Extracted {len(chunks)} chunks'); print(f'Sample: {chunks[0].content[:100]}...')"
```

### Test Vector Store
```python
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore

p = PDFProcessor()
v = VectorStore()
chunks = p.process_pdf('test.pdf')
v.add_documents(chunks)
results = v.search("main topic", n_results=3)
print(f"Found {len(results)} results")
```

### Test LLM Client
```python
from src.llm_client import OpenRouterClient
import os

client = OpenRouterClient(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url=os.getenv('OPENROUTER_BASE_URL'),
    default_model=os.getenv('DEFAULT_MODEL')
)
response = client.generate_response("Hello, world!")
print(response.content)
```

### Test Full RAG Pipeline
```python
from main import RAGApplication

app = RAGApplication()
app.initialize_components()

# Process a PDF
chunks = app.pdf_processor.process_pdf('test.pdf')
app.vector_store.add_documents(chunks)

# Query
result = app.rag_system.query("What are the main topics?")
print("Answer:", result.answer)
print("Sources:", result.sources)
print("Confidence:", result.confidence_score)
```

---

## 🎯 Priority Order for Each TODO

### Critical Path (Must Complete)
1. PDF extraction + chunking
2. Vector store (embed + search)
3. LLM client (generate response)
4. RAG retrieval + prompt creation
5. Citation system
6. Gradio UI (upload + query)

### Nice-to-Have (If Time Permits)
- Query analysis and expansion
- Advanced ranking strategies
- Confidence scoring refinement
- System health monitoring
- Performance optimization

---

## 🚨 Common Pitfalls to Avoid

1. **PDF Processing:** Don't forget to preserve page numbers in metadata
2. **Embeddings:** Normalize embeddings before storing (cosine similarity)
3. **ChromaDB:** Create persistent client, not in-memory
4. **LLM Prompts:** Be explicit about citation format in system prompt
5. **Token Management:** Check token limits before sending to LLM
6. **Error Handling:** Wrap external API calls (OpenRouter) in try-catch
7. **Testing:** Test each component individually before integration

---

## 📚 Quick Reference Links

- **ChromaDB Docs:** https://docs.trychroma.com/
- **Sentence Transformers:** https://www.sbert.net/
- **OpenRouter API:** https://openrouter.ai/docs
- **Gradio Docs:** https://gradio.app/docs/
- **pypdf Docs:** https://pypdf.readthedocs.io/

---

## ✅ Final Checklist

Before marking the project complete:

- [ ] Can upload multiple PDFs via web UI
- [ ] Can ask questions and get answers
- [ ] Answers include source citations with page numbers
- [ ] Citations are accurate (manually verify a few)
- [ ] Error handling prevents crashes
- [ ] System status tab shows collection stats
- [ ] README reflects actual usage
- [ ] `.env` file is properly configured
- [ ] All tests pass (manual testing acceptable)
- [ ] Demo script prepared

**When complete:** You'll have a production-ready RAG system! 🎉

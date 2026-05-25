---

## Day 11 — RAG Pipeline: Ask Your Documents

**Notebook:** `day11_rag_pipeline.ipynb`

### 🎯 Objective
Build a complete Retrieval Augmented Generation (RAG) pipeline
that answers questions from ML lecture notes — grounding the
LLM in real document content to prevent hallucination.

### 🔧 RAG Pipeline Steps
Load PDF → Chunk text → Embed → Store ChromaDB → Retrieve → Generate

### 📊 Pipeline Stats
| Component | Detail |
|-----------|--------|
| Document | ML lecture notes PDF |
| Pages loaded | 15 |
| Chunks created | 87 |
| Chunk size | 500 chars with 50 overlap |
| Embedding model | all-MiniLM-L6-v2 (local, free) |
| Vector database | ChromaDB |
| LLM | Llama3-8b via Groq |
| Questions answered | 5 + 1 out-of-scope test |

### 💡 Key Findings
- RAG successfully grounds LLM answers in document content
- Out-of-scope question correctly returned "not in notes"
  confirming hallucination prevention is working
- Semantic search retrieves relevant chunks even when exact
  keywords do not match the query
- HuggingFace all-MiniLM-L6-v2 produces quality embeddings
  locally with no API key or cost

### 🛠️ Tech Stack
Python · LangChain · ChromaDB · HuggingFace Embeddings
sentence-transformers · Groq API · Llama3 · pypdf · pandas

### 📂 Output Files
| File | Description |
|------|-------------|
| data/day11_rag_results.csv | All Q&A results with metadata |
| chroma_db/ | Persistent vector database |
# GitHub Desktop commit message:
"Day 11: RAG pipeline, ChromaDB, HuggingFace embeddings, Q&A from lecture notes"
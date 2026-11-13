# RAG_Advance_Project

A **Multi-PDF RAG app with sources**, built with Streamlit. Upload one or more PDFs, ask questions, and get an answer that cites the file and page it came from.

RAG (Retrieval-Augmented Generation) = find the right text first, *then* let the AI answer using it. The main app is [rag_app.py](rag_app.py).

## Features

- **Multiple PDFs at once.** Upload several files and search across all of them.
- **Source citations.** Every chunk keeps its real file name and 1-based page number. The answer is followed by a list of the files and pages used.
- **Conflict handling.** The prompt tells the model not to pick a side silently when documents disagree. It must say they disagree and name the file and page for each claim.
- **"I don't know" answers.** The model is told to answer only from the retrieved context.
- **Tunable retrieval.** Sidebar sliders for chunk size, chunk overlap and number of chunks retrieved (k).
- **Inspect retrieval.** A "Show retrieved chunks" expander shows exactly what the model saw.
- **Local embeddings.** Free `all-MiniLM-L6-v2` model, cached across reruns with `st.cache_resource`.

## How it works

```
Upload -> Load -> Split -> Embed -> Store -> Retrieve -> Answer
```

| Step | What happens | Tool used |
|------|--------------|-----------|
| 1. Upload | You add PDFs; each is written to a temp file so it can be read, then the temp file is deleted | `st.file_uploader` |
| 2. Load | Reads each PDF page by page and sets `source` (file name) and `page` (1-based) metadata | `PyPDFLoader` |
| 3. Split | Cuts pages into overlapping chunks; metadata is kept on every chunk | `RecursiveCharacterTextSplitter` |
| 4. Embed | Converts chunks to vectors with a local model | HuggingFace `all-MiniLM-L6-v2` |
| 5. Store | Builds a FAISS index, keeps it in the session and saves it to `faiss_index/` | FAISS |
| 6. Retrieve | Finds the k chunks most similar to the question | `as_retriever(search_kwargs={"k": k})` |
| 7. Answer | Sends the chunks (each prefixed with `[file.pdf - page N]`) and the question to the LLM | `ChatOpenAI`, `temperature=0` |

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your-key-here
   ```

   `.env` is listed in `.gitignore`. Never commit it. The app shows an error and stops if `OPENAI_API_KEY` is not set.

## Run

```powershell
streamlit run rag_app.py
```

1. Upload one or more PDFs.
2. Click **Build index**. You will see how many chunks were indexed.
3. Type a question. The app shows the **Answer**, the **Sources** (file and page), and an expander with the retrieved chunks.

The first run downloads the embedding model, so it takes longer.

## Settings

| Setting | Where | Default | Range | Applies |
|---------|-------|---------|-------|---------|
| Chunk size | Sidebar slider | 800 | 300 to 1500 | When you click **Build index** |
| Chunk overlap | Sidebar slider | 100 | 0 to 300 | When you click **Build index** |
| Chunks to retrieve (k) | Sidebar slider | 5 | 2 to 10 | Immediately, on every question |
| Embedding model | `get_embeddings()` in code | `all-MiniLM-L6-v2` | | |
| LLM | `ChatOpenAI(...)` in code | `gpt-5.4-mini` | | Use any chat model your account supports |
| Index folder | `INDEX_DIR` in code | `faiss_index` | | |

If an answer cites 5 or more distinct pages, the app suggests trying a smaller chunk size.

## Behavior to know about

- **The index is not reloaded on restart.** It is saved to `faiss_index/`, but `rag_app.py` only reads it from the current session. After a page refresh or restart, upload the PDFs and click **Build index** again.
- **Each build replaces the index.** It contains only the files uploaded at that moment, and any previous contents of `faiss_index/` are overwritten.
- **Changing chunk size or overlap needs a rebuild.** Only k takes effect right away.
- **Pages with no extractable text** (for example scanned images) produce no useful chunks. The app does not do OCR.

## Project structure

```
RAG_Advance_Project/
├── rag_app.py                         # Main app: Streamlit multi-PDF RAG with sources
├── RAG_Demo.py                        # Minimal text RAG demo
├── rag_demo_langsmith.py              # Same demo with LangSmith tracing
├── rag_faiss_app.py                   # PDF Q&A in the terminal
├── rag_app1.py                        # Variant of rag_faiss_app.py (different model)
├── llm_document.txt                   # Sample text for the text demos
├── spotify_web_app_architecture.pdf   # Sample PDF for the terminal apps and for testing rag_app.py
├── faiss_index/                       # Saved FAISS index (generated)
├── requirements.txt                   # Python dependencies
├── .env                               # API keys (not committed)
└── README.md
```

## Other scripts

Smaller scripts that show the same pipeline in simpler forms. Run them with `python <script>`.

| Script | Input | What it does |
|--------|-------|--------------|
| [RAG_Demo.py](RAG_Demo.py) | `llm_document.txt` | Answers one hard-coded question. Prints the retrieved text instead if no API key is set. |
| [rag_demo_langsmith.py](rag_demo_langsmith.py) | `llm_document.txt` | Same, with the OpenAI call wrapped by `wrap_openai` so LangSmith traces it. Needs `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`. |
| [rag_faiss_app.py](rag_faiss_app.py) | `spotify_web_app_architecture.pdf` | Terminal Q&A loop (type `exit` to stop). Builds `faiss_index/` on first run, then reloads it. Model `gpt-4.1-mini`. |
| [rag_app1.py](rag_app1.py) | `spotify_web_app_architecture.pdf` | Same as `rag_faiss_app.py` with model `gpt-5.4-mini`. |

**Shared index folder.** `rag_faiss_app.py`, `rag_app1.py` and `rag_app.py` all use `faiss_index/`. Building an index in one replaces the others'. The terminal apps only rebuild when the folder is missing, so delete `faiss_index/` if they start answering from the wrong PDF.

## Dependencies

- `streamlit`: web UI
- `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-openai`, `langchain-text-splitters`: pipeline building blocks
- `sentence-transformers`: local embedding backend
- `faiss-cpu`: local vector store
- `pypdf`: PDF loading
- `openai`: answer generation
- `python-dotenv`: loads `.env`
- `rank-bm25`, `langsmith`: for the BM25 extension and LangSmith tracing

`FAISS.load_local(..., allow_dangerous_deserialization=True)` in the terminal apps unpickles the saved index. Only load indexes you created yourself.

## Ideas for extending

- Reload the saved index at startup so a restart does not require rebuilding
- Append new PDFs to the existing index instead of replacing it
- Add keyword search (BM25) and combine it with vector search (hybrid retrieval)
- Add a reranking step on the retrieved chunks
- Add LangSmith tracing to the Streamlit app

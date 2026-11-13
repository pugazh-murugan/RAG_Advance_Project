# RAG_Advance_Project

A minimal, beginner-friendly **Retrieval-Augmented Generation (RAG)** demo in Python.

RAG = find the right text first, *then* let the AI answer using it. This project runs the full pipeline in a single script, [RAG_Demo.py](RAG_Demo.py).

## How it works

```
Load -> Split -> Embed -> Store -> Retrieve -> Answer
```

| Step | What happens | Tool used |
|------|--------------|-----------|
| 1. Load | Reads `llm_document.txt` | LangChain `TextLoader` |
| 2. Split | Cuts the text into chunks (500 chars, 50 overlap) | `RecursiveCharacterTextSplitter` |
| 3. Embed | Converts chunks to vectors with a free, local model | HuggingFace `all-MiniLM-L6-v2` |
| 4. Store | Saves the vectors in an in-memory index | FAISS |
| 5. Retrieve | Finds the 3 chunks most similar to the question | `similarity_search(k=3)` |
| 6. Answer | Sends the chunks and the question to an LLM, told to use *only* that text | OpenAI chat completions |

Step 6 is optional. If no `OPENAI_API_KEY` is set, the script prints the retrieved text instead of generating an answer.

## Project structure

```
RAG_Advance_Project/
├── RAG_Demo.py        # The whole RAG pipeline
├── requirements.txt   # Python dependencies
├── llm_document.txt   # Source document to query (you provide this)
├── .env               # API keys (not committed)
└── README.md
```

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   pip install python-dotenv
   ```

   `python-dotenv` is imported by the script but is not yet listed in `requirements.txt`, so install it separately or add it to that file.

3. Add a source document named `llm_document.txt` in the project root. It can be any plain-text (UTF-8) file.

4. (Optional) For LLM-generated answers, create a `.env` file:

   ```
   OPENAI_API_KEY=your-key-here
   ```

   Never commit this file.

## Run

```powershell
python RAG_Demo.py
```

The first run downloads the embedding model, so it takes longer.

- **With an API key:** prints `QUESTION:` followed by `ANSWER:`.
- **Without an API key:** prints the question and the retrieved chunks.

To ask something else, change the `question` variable near the top of [RAG_Demo.py](RAG_Demo.py). The default is `"What is a token?"`.

## Configuration

All settings are set in [RAG_Demo.py](RAG_Demo.py):

| Setting | Default | Notes |
|---------|---------|-------|
| `question` | `"What is a token?"` | The query to answer |
| `chunk_size` / `chunk_overlap` | `500` / `50` | Chunking granularity |
| Embedding model | `all-MiniLM-L6-v2` | Runs locally and is free |
| `k` | `3` | Number of chunks retrieved |
| LLM model | `gpt-5.4-mini` | Change to any model your OpenAI account supports |

## Dependencies

- `langchain`, `langchain-community`, `langchain-huggingface`: pipeline building blocks
- `sentence-transformers`: local embedding backend
- `faiss-cpu`: local vector store
- `openai`: optional, for answer generation
- `rank-bm25`, `langsmith`: optional, listed for planned extensions (BM25 and rerank, LangSmith tracing)
- `python-dotenv`: loads `.env` (install separately, see Setup)

## Ideas for extending

- Add keyword search (BM25) and combine it with vector search (hybrid retrieval)
- Add a reranking step on the retrieved chunks
- Trace runs with LangSmith
- Persist the FAISS index to disk instead of rebuilding it on every run

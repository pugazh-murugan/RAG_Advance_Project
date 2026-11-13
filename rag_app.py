"""
Multi-PDF RAG app with sources.
Run:  streamlit run rag_app.py
Needs OPENAI_API_KEY in the environment (or in a .env file).
"""
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
INDEX_DIR = "faiss_index"  # also saved locally on disk

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

st.set_page_config(page_title="Multi-PDF RAG", page_icon="📚")
st.title("📚 Multi-PDF RAG with Sources")

if not os.getenv("OPENAI_API_KEY"):
    st.error("Set the OPENAI_API_KEY environment variable (or put it in a .env file).")
    st.stop()

# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, 50)
    top_k = st.slider("Chunks to retrieve (k)", 2, 10, 5)

# ---------- 1. Upload + index ----------
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if files and st.button("Build index"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    with st.spinner("Reading and embedding PDFs..."):
        for f in files:
            # save upload to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            pages = PyPDFLoader(tmp_path).load()
            os.remove(tmp_path)

            for p in pages:
                p.metadata["source"] = f.name               # real file name, not temp path
                p.metadata["page"] = p.metadata.get("page", 0) + 1  # 1-based page number
            chunks.extend(splitter.split_documents(pages))  # metadata survives splitting

        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(INDEX_DIR)
        st.session_state.store = store
        st.session_state.files = [f.name for f in files]

    st.success(f"Indexed {len(chunks)} chunks from {len(files)} PDF(s)")

if "files" in st.session_state:
    st.caption("Uploaded: " + ", ".join(st.session_state.files))

# ---------- 2. Ask questions ----------
PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. Each context block starts
with its source in brackets, like [file.pdf - page 3].

Rules:
- If the answer is not in the context, say you don't know.
- If different documents give conflicting answers, do NOT pick one silently.
  Say clearly that the documents disagree and state what each one says,
  naming the file and page.

Context:
{context}

Question: {question}"""
)
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

question = st.text_input("Ask a question about your documents")

if question:
    if "store" not in st.session_state:
        st.warning("Upload PDFs and click 'Build index' first.")
        st.stop()

    docs = st.session_state.store.as_retriever(search_kwargs={"k": top_k}).invoke(question)

    context = "\n\n".join(
        f"[{d.metadata['source']} - page {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )
    with st.spinner("Thinking..."):
        answer = (PROMPT | llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )

    st.subheader("Answer")
    st.write(answer)

    # unique (file, page) pairs, in order
    sources = sorted({(d.metadata["source"], d.metadata["page"]) for d in docs})
    st.subheader("Sources")
    st.write(" | ".join(f"{s} - page {p}" for s, p in sources))
    if len(sources) >= 5:
        st.info("This answer cites many different pages - try a smaller chunk size.")

    with st.expander("Show retrieved chunks"):
        for d in docs:
            st.caption(f"{d.metadata['source']} - page {d.metadata['page']}")
            st.write(d.page_content)
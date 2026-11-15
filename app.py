import os, tempfile, time, urllib.request
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types

st.set_page_config(page_title="RAG Study Assistant", page_icon="📚")
st.title("📚 RAG Study Assistant")
st.caption("Ask questions about PDFs. Answers come with page citations.")

SAMPLES = {
    "rag_paper.pdf": "https://arxiv.org/pdf/2005.11401",
    "attention_paper.pdf": "https://arxiv.org/pdf/1706.03762",
}
MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]


@st.cache_data(show_spinner="Downloading sample papers...")
def load_samples():
    out = []
    for name, url in SAMPLES.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out.append((name, r.read()))
    return tuple(out)


@st.cache_resource(show_spinner="Reading and indexing the PDFs...")
def build_index(files):
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = []
    for name, data in files:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        pages = PyPDFLoader(path).load()
        os.remove(path)
        for p in pages:
            p.metadata["source"] = name
        docs.extend(pages)
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
    if not chunks:
        raise ValueError("No readable text found. Scanned PDFs (images) are not supported.")
    return FAISS.from_documents(chunks, embeddings), len(docs), len(chunks)


def build_prompt(question, docs):
    context = "\n\n".join(
        f"[Source: {d.metadata['source']}, page {d.metadata['page'] + 1}]\n{d.page_content}"
        for d in docs
    )
    return f"""You are a study assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the documents."
Mention the source file and page number for the facts you use.

Context:
{context}

Question: {question}

Answer:"""


def generate_with_retry(api_key, prompt, retries=2):
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30000))
    for model in MODELS:
        for attempt in range(retries):
            try:
                return client.models.generate_content(model=model, contents=prompt).text or "(empty response)"
            except Exception as e:
                msg = str(e)
                if "API key" in msg or "API_KEY" in msg:
                    return "That API key was not accepted. Please check it in the sidebar."
                if any(x in msg for x in ["503", "429", "UNAVAILABLE", "timed out", "Timeout"]):
                    time.sleep(2 ** attempt * 2)
                else:
                    break
    return "The AI service is busy right now. Please try again in a minute."


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Setup")
    user_key = st.text_input("Gemini API key", type="password",
                             help="Free key from aistudio.google.com/apikey")
    uploaded = st.file_uploader("Upload your own PDFs (optional)", type="pdf",
                                accept_multiple_files=True)
    k = st.slider("Chunks to retrieve (k)", 2, 10, 6)

api_key = user_key
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

# ---------- Documents ----------
if uploaded:
    files = tuple((f.name, f.getvalue()) for f in uploaded)
else:
    files = load_samples()
    st.info("Using 2 sample papers (RAG and 'Attention Is All You Need'). "
            "Upload your own PDFs in the sidebar to chat with them instead.")

try:
    db, n_pages, n_chunks = build_index(files)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Indexed {n_pages} pages into {n_chunks} chunks")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources used"):
                for s in m["sources"]:
                    st.text(s)

question = st.chat_input("Ask a question about the documents...")
if question:
    if not api_key:
        st.warning("Add your free Gemini API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            docs = db.similarity_search(question, k=k)
            answer = generate_with_retry(api_key, build_prompt(question, docs))
        st.markdown(answer)
        sources = [f"{d.metadata['source']}, page {d.metadata['page'] + 1}: {d.page_content[:200]}..."
                   for d in docs]
        with st.expander("Sources used"):
            for s in sources:
                st.text(s)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

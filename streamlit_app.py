"""
streamlit_app.py
----------------
Browser front-end for the Enterprise RAG system.
Wraps the existing ingest() / query_pipeline() functions from app.py.

Run with:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent))

from app import ingest, query_pipeline, DATA_DIR, EMBED_MODEL, GEN_MODEL, TOP_K
from validator.json_validator import ValidationError

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise RAG — Ollama",
    page_icon="🔍",
    layout="centered",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    embed_model = st.text_input("Embedding model", value=EMBED_MODEL)
    gen_model   = st.text_input("Generation model", value=GEN_MODEL)
    top_k       = st.slider("Top-k passages", min_value=1, max_value=10, value=TOP_K)
    st.markdown("---")
    st.code("ollama serve", language="bash")
    st.caption("Ollama must be running locally.")

# ── Corpus loading (cached so it only runs once) ───────────────────────────────

@st.cache_resource(show_spinner="Loading and embedding document corpus…")
def load_corpus(data_dir: str, embed_model_key: str):
    """Ingests all .txt files and returns (chunks, metadata, embeddings)."""
    return ingest(
        data_dir    = Path(data_dir),
        embed_model = embed_model_key,
    )

# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🔍 Enterprise RAG System")
st.caption("Local Retrieval-Augmented Generation powered by Ollama · No cloud APIs")

# Load corpus — show friendly error if Ollama is down or data dir is empty
try:
    chunks, metadata, corpus_embeddings = load_corpus(str(DATA_DIR), embed_model)
except FileNotFoundError as exc:
    st.error(f"**Data directory error:** {exc}")
    st.stop()
except ConnectionError as exc:
    host = str(exc).split("'")[1] if "'" in str(exc) else "http://localhost:11434"
    st.error(
        f"**Ollama is not reachable at `{host}`**\n\n"
        "→ Make sure Ollama is running: `ollama serve`\n\n"
        f"→ Pull the embedding model: `ollama pull {embed_model}`"
    )
    st.stop()

# Show corpus summary
doc_names = sorted({m["source"] for m in metadata})
with st.expander(f"📂 Corpus — {len(doc_names)} document(s), {len(chunks)} chunks", expanded=False):
    for name in doc_names:
        count = sum(1 for m in metadata if m["source"] == name)
        st.markdown(f"- **{name}** — {count} chunks")

# Query input
query = st.text_input(
    "Ask a question",
    placeholder="What is the approval threshold for capital expenditures?",
)

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question before clicking Ask.")
    else:
        with st.spinner("Retrieving and generating…"):
            try:
                response = query_pipeline(
                    query             = query,
                    chunks            = chunks,
                    metadata          = metadata,
                    corpus_embeddings = corpus_embeddings,
                    gen_model         = gen_model,
                    embed_model       = embed_model,
                    top_k             = top_k,
                )
            except ConnectionError as exc:
                host = str(exc).split("'")[1] if "'" in str(exc) else "http://localhost:11434"
                st.error(
                    f"**Ollama is not reachable at `{host}`**\n\n"
                    f"→ Pull the generation model: `ollama pull {gen_model}`"
                )
                st.stop()
            except ValidationError as exc:
                st.error(f"**Validation error:** {exc}")
                st.stop()

        # ── Answer ──────────────────────────────────────────────────────
        st.subheader("Answer")
        st.write(response["answer"])

        # ── Source passages ──────────────────────────────────────────────
        st.subheader("Retrieved Sources")
        for i, src in enumerate(response["sources"], start=1):
            label = f"📄 Source {i} — `{src['source']}`"
            with st.expander(label, expanded=(i == 1)):
                st.caption(src["text"])

        # ── Model used ───────────────────────────────────────────────────
        st.caption(f"Model: `{response['model']}`")

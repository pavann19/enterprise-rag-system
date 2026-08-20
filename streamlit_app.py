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

from app import (
    CACHE_DIR,
    DATA_DIR,
    EMBED_BACKEND,
    EMBED_MODEL,
    GEN_BACKEND,
    GEN_MODEL,
    TOP_K,
    VECTOR_BACKEND,
    query_pipeline,
)
from rag.ingestion import ingest
from validator.json_validator import ValidationError

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🔍",
    layout="centered",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption(f"Embedding backend: `{EMBED_BACKEND}`  ·  Generation backend: `{GEN_BACKEND}`")
    embed_model = st.text_input("Embedding model", value=EMBED_MODEL)
    gen_model = st.text_input("Generation model", value=GEN_MODEL)
    top_k = st.slider("Top-k passages", min_value=1, max_value=10, value=TOP_K)
    st.markdown("---")
    if EMBED_BACKEND == "ollama" or GEN_BACKEND == "ollama":
        st.code("ollama serve", language="bash")
        st.caption("Ollama must be running locally for the ollama-backed steps above.")
    if GEN_BACKEND == "anthropic":
        st.caption("Generation uses the Claude API — requires ANTHROPIC_API_KEY.")
    if GEN_BACKEND == "groq":
        st.caption("Generation uses the Groq API — requires GROQ_API_KEY.")

# ── Corpus loading (cached so it only runs once) ───────────────────────────────


@st.cache_resource(show_spinner="Loading and embedding document corpus…")
def load_corpus(data_dir: str, embed_model_key: str, embed_backend_key: str):
    """Ingests all .txt files and returns (chunks, metadata, vector_store)."""
    return ingest(
        data_dir=Path(data_dir),
        embed_model=embed_model_key,
        embed_backend=embed_backend_key,
        backend=VECTOR_BACKEND,
        cache_dir=CACHE_DIR,
    )


# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🔍 Enterprise RAG System")
st.caption(f"Retrieval-Augmented Generation · embed={EMBED_BACKEND} · generate={GEN_BACKEND}")

# Load corpus — show a friendly error if the embedding backend is unreachable
# or misconfigured, or the data dir is empty
try:
    chunks, metadata, vector_store = load_corpus(str(DATA_DIR), embed_model, EMBED_BACKEND)
except FileNotFoundError as exc:
    st.error(f"**Data directory error:** {exc}")
    st.stop()
except ConnectionError as exc:
    if EMBED_BACKEND == "ollama":
        host = str(exc).split("'")[1] if "'" in str(exc) else "http://localhost:11434"
        st.error(
            f"**Ollama is not reachable at `{host}`**\n\n"
            "→ Make sure Ollama is running: `ollama serve`\n\n"
            f"→ Pull the embedding model: `ollama pull {embed_model}`"
        )
    else:
        st.error(f"**Embedding backend unreachable:** {exc}")
    st.stop()
except ImportError as exc:
    st.error(f"**Missing dependency for embed_backend='{EMBED_BACKEND}':** {exc}")
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
                    query=query,
                    chunks=chunks,
                    metadata=metadata,
                    vector_store=vector_store,
                    gen_model=gen_model,
                    gen_backend=GEN_BACKEND,
                    embed_model=embed_model,
                    embed_backend=EMBED_BACKEND,
                    top_k=top_k,
                )
            except ConnectionError as exc:
                if GEN_BACKEND == "ollama":
                    host = str(exc).split("'")[1] if "'" in str(exc) else "http://localhost:11434"
                    st.error(
                        f"**Ollama is not reachable at `{host}`**\n\n"
                        f"→ Pull the generation model: `ollama pull {gen_model}`"
                    )
                else:
                    st.error(f"**Generation backend unreachable:** {exc}")
                st.stop()
            except (ImportError, RuntimeError) as exc:
                st.error(f"**Generation backend misconfigured:** {exc}")
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

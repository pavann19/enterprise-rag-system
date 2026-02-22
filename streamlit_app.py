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

from app import ingest, query_pipeline, EMBED_MODEL, GEN_MODEL, TOP_K
from validator.json_validator import ValidationError

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG — Ollama",
    page_icon="🔍",
    layout="centered",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 Enterprise RAG System")
st.caption("Local Retrieval-Augmented Generation powered by Ollama · No cloud APIs")

st.divider()

# ── Sidebar: configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    embed_model = st.text_input(
        "Embedding model",
        value=EMBED_MODEL,
        help="Must be pulled via: ollama pull <model>",
    )
    gen_model = st.text_input(
        "Generation model",
        value=GEN_MODEL,
        help="Must be pulled via: ollama pull <model>",
    )
    top_k = st.slider("Top-k passages", min_value=1, max_value=6, value=TOP_K)

    st.divider()
    st.markdown(
        "**Requires Ollama running locally.**\n\n"
        "```\nollama serve\n```"
    )

# ── Corpus ingestion (cached per model) ───────────────────────────────────────
@st.cache_resource(show_spinner="Embedding knowledge base…")
def get_corpus(model: str):
    """Loads and embeds the knowledge base once; cached between queries."""
    return ingest(embed_model=model)


# ── Main: query input ──────────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question",
    placeholder="What are the four stages of a RAG pipeline?",
    label_visibility="visible",
)

run = st.button("Ask", type="primary", use_container_width=True)

# ── Pipeline execution ─────────────────────────────────────────────────────────
if run and question.strip():

    # Step 1 — Ingestion (cached)
    try:
        chunks, corpus_embeddings = get_corpus(embed_model)
    except ConnectionError as exc:
        st.error(f"**Ollama connection error during ingestion**\n\n{exc}")
        st.stop()
    except FileNotFoundError as exc:
        st.error(f"**Knowledge base not found**\n\n{exc}")
        st.stop()

    # Step 2 — Retrieval + Generation
    with st.spinner("Retrieving and generating…"):
        try:
            response = query_pipeline(
                question=question,
                chunks=chunks,
                corpus_embeddings=corpus_embeddings,
                embed_model=embed_model,
                gen_model=gen_model,
                top_k=top_k,
            )
        except ConnectionError as exc:
            st.error(f"**Ollama connection error during query**\n\n{exc}")
            st.stop()
        except ValidationError as exc:
            st.error(f"**Output validation failed**\n\n{exc}")
            st.stop()

    # ── Display results ────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.success(response["answer"])

    with st.expander(f"📄 Retrieved passages (top-{top_k})", expanded=False):
        for i, passage in enumerate(response["sources"], start=1):
            st.markdown(f"**Passage {i}**")
            st.text(passage)
            if i < len(response["sources"]):
                st.divider()

    with st.expander("🔧 Structured Response (JSON)", expanded=False):
        st.json(response)

elif run and not question.strip():
    st.warning("Please enter a question before clicking Ask.")

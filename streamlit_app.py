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
)
from rag.embedder import embed_texts
from rag.generator import generate_answer_stream
from rag.ingestion import ingest
from rag.reranker import RERANK_ENABLED, rerank
from rag.retriever import retrieve

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🔍",
    layout="centered",
)

# ── Visual system ────────────────────────────────────────────────────────────
# A single injected stylesheet, not scattered inline styles — one place to
# change the palette/type scale. System-font stack (no web-font fetch, so
# this still renders correctly offline / on a locked-down network) with a
# restrained accent color and consistent spacing/radius tokens rather than
# per-element ad-hoc values.

st.markdown(
    """
    <style>
    /*
     * Streamlit doesn't expose its active theme as CSS custom properties in
     * this version, and its default theme setting itself follows the OS
     * light/dark preference — so prefers-color-scheme is the accurate
     * signal here, not a fallback guess.
     */
    :root {
        --accent: #0a84ff;
        --accent-soft: rgba(10, 132, 255, 0.12);
        --ink: #1d1d1f;
        --ink-soft: rgba(29, 29, 31, 0.62);
        --hairline: rgba(29, 29, 31, 0.14);
        --surface: rgba(0, 0, 0, 0.025);
        --radius: 14px;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --accent: #409cff;
            --accent-soft: rgba(64, 156, 255, 0.16);
            --ink: #f5f5f7;
            --ink-soft: rgba(245, 245, 247, 0.64);
            --hairline: rgba(245, 245, 247, 0.14);
            --surface: rgba(255, 255, 255, 0.045);
        }
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
            Inter, Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        max-width: 760px;
        padding-top: 2.75rem;
        padding-bottom: 4rem;
    }

    /* Header */
    .rag-eyebrow {
        color: var(--accent);
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .rag-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin: 0 0 0.3rem 0;
    }
    .rag-subtitle {
        color: var(--ink-soft);
        font-size: 1rem;
        margin-bottom: 1.75rem;
    }
    .rag-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        margin-right: 0.4rem;
    }

    /* Cards */
    .rag-card {
        border: 1px solid var(--hairline);
        border-radius: var(--radius);
        padding: 1.4rem 1.5rem;
        background: var(--surface);
        margin-bottom: 1.25rem;
    }
    .rag-card h4 {
        margin-top: 0;
        margin-bottom: 0.65rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--ink);
    }

    /* Inputs */
    .stTextInput input {
        border-radius: 10px !important;
        border: 1px solid var(--hairline) !important;
        padding: 0.65rem 0.9rem !important;
    }
    .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    /* Primary button — Apple-style pill, no shouty gradient */
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border: none;
        border-radius: 999px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        box-shadow: none;
        transition: opacity 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent);
        opacity: 0.88;
    }

    /* Expanders as quiet list rows, not boxed accordions */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        border-radius: 10px !important;
        font-size: 0.92rem !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--hairline) !important;
        border-radius: 10px !important;
    }

    hr { border-color: var(--hairline); }

    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Configuration")
    st.caption(f"Embedding · `{EMBED_BACKEND}`  ·  Generation · `{GEN_BACKEND}`")
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
    if RERANK_ENABLED:
        st.caption("Cross-encoder re-ranking is enabled.")

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


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="rag-eyebrow">Retrieval-Augmented Generation</div>', unsafe_allow_html=True)
st.markdown('<div class="rag-title">Enterprise RAG</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="rag-subtitle">
        <span class="rag-pill">embed · {EMBED_BACKEND}</span>
        <span class="rag-pill">generate · {GEN_BACKEND}</span>
        Ask a question grounded strictly in your own documents.
    </div>
    """,
    unsafe_allow_html=True,
)

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

st.write("")

# Query input
query = st.text_input(
    "Ask a question",
    placeholder="What is the approval threshold for capital expenditures?",
    label_visibility="collapsed",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question before clicking Ask.")
    else:
        try:
            with st.spinner("Retrieving passages…"):
                query_embedding = embed_texts([query], model=embed_model, backend=EMBED_BACKEND)[0]
                results = retrieve(
                    query_embedding=query_embedding,
                    vector_store=vector_store,
                    chunks=chunks,
                    metadata=metadata,
                    top_k=top_k,
                )
                if RERANK_ENABLED:
                    results = rerank(query, results)
                passages = [r["text"] for r in results]

            # ── Answer — streamed token-by-token as it's generated ───────
            st.markdown("#### Answer")
            with st.container():
                st.markdown('<div class="rag-card">', unsafe_allow_html=True)
                answer = st.write_stream(
                    generate_answer_stream(
                        query=query,
                        passages=passages,
                        model=gen_model,
                        backend=GEN_BACKEND,
                    )
                )
                st.markdown("</div>", unsafe_allow_html=True)
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

        if not answer:
            st.warning("The model returned an empty response.")

        # ── Source passages ──────────────────────────────────────────────
        st.markdown("#### Sources")
        for i, r in enumerate(results, start=1):
            label = f"📄  {r['source']}"
            with st.expander(label, expanded=(i == 1)):
                st.caption(r["text"])

        # ── Model used ───────────────────────────────────────────────────
        st.caption(f"Model · `{gen_model}`")

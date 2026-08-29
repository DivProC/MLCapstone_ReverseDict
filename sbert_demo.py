"""
Reverse Dictionary — SBERT Demo (Streamlit)
---------------------------------------------
Uses the sentence-transformers/all-MiniLM-L6-v2 model.
Encodes a definition query and ranks all 108,839 candidate words
by cosine similarity against their pre-built index embeddings.

Run from the project root:
    streamlit run sbert_demo.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT   = Path(__file__).resolve().parent
_DATA      = _PROJECT / "data" / "processed"
_PROCESSED = _DATA / "opted_preprocessed.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading SBERT and building word index — first run only (~2 min)…")
def load_everything():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    df = pd.read_csv(_PROCESSED, dtype=str, keep_default_na=False)
    df = df[df["is_unresolved_cross_reference"] != "True"]

    # Collect all definitions per word
    word_defs: dict[str, list[str]] = defaultdict(list)
    word_order: list[str] = []
    for _, row in df.iterrows():
        w = row["word_norm"]
        if w not in word_defs:
            word_order.append(w)
        word_defs[w].append(row["definition_model_input"])

    # Flatten to a single list for batched encoding
    flat_texts: list[str] = []
    flat_word_indices: list[int] = []
    for i, word in enumerate(word_order):
        for defn in word_defs[word]:
            flat_texts.append(defn)
            flat_word_indices.append(i)

    # Encode everything in one batched pass
    embeddings = model.encode(
        flat_texts,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Average-pool per word and re-normalise
    n_words = len(word_order)
    dim = embeddings.shape[1]
    word_embs = np.zeros((n_words, dim), dtype=np.float32)
    counts = np.zeros(n_words, dtype=np.float32)
    for i, emb in zip(flat_word_indices, embeddings):
        word_embs[i] += emb
        counts[i] += 1
    counts = counts.clip(min=1)[:, None]
    word_embs /= counts

    # Final L2 normalisation
    norms = np.linalg.norm(word_embs, axis=1, keepdims=True).clip(min=1e-9)
    word_index = (word_embs / norms).astype(np.float32)

    return model, word_order, word_index


def predict(query: str, top_k: int, model, word_order, word_index):
    query_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)                  # (1, dim)

    scores = (word_index @ query_emb.T).squeeze(1)   # (n_words,)
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        {"word": word_order[i], "score": float(scores[i])}
        for i in top_indices
    ]


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Reverse Dictionary", page_icon="📖", layout="centered")

st.title("📖 Reverse Dictionary")
st.caption(
    "Type a definition and the model will predict the word. "
    "Powered by SBERT (all-MiniLM-L6-v2) trained on 108,839 candidate words."
)

st.divider()

top_k = st.slider("Number of results to show", min_value=1, max_value=10, value=5)

definition = st.text_area(
    "Enter a definition",
    placeholder="e.g.  the fear of heights",
    height=110,
)

if st.button("🔍  Find word", type="primary"):
    query = definition.strip()
    if not query:
        st.warning("Please enter a definition first.")
    else:
        model, word_order, word_index = load_everything()
        with st.spinner("Searching…"):
            results = predict(query, top_k, model, word_order, word_index)

        if not results:
            st.error("Something went wrong — try again.")
        else:
            st.subheader("Top predictions")
            for rank, r in enumerate(results, start=1):
                pct = int((r["score"] + 1) / 2 * 100)
                col_rank, col_word, col_bar = st.columns([0.08, 0.35, 0.57])
                col_rank.markdown(f"**{rank}**")
                col_word.markdown(f"**{r['word']}**")
                col_bar.progress(pct, text=f"{r['score']:.3f} cosine similarity")

st.divider()
st.caption("SBERT · all-MiniLM-L6-v2 · OPTED dataset · val R@10 = 23.7%  |  108,839 candidate words")

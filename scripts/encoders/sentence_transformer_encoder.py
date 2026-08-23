"""Sentence-Transformer phrase encoder."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence_transformers is not installed. Install it before running "
                "the Sentence-BERT encoder path."
            ) from exc

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)

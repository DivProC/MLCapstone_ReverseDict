"""Encoder registry for the shared reverse-dictionary pipeline."""

from __future__ import annotations

from .bert_cls import BertClsEncoder
from .sentence_transformer_encoder import SentenceTransformerEncoder


ENCODER_CHOICES = ("sentence-transformer", "bert-cls")

DEFAULT_MODEL_BY_ENCODER = {
    "sentence-transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "bert-cls": "bert-base-uncased",
}


def make_encoder(
    encoder_name: str,
    model_name: str,
    device: str | None = None,
):
    if encoder_name == "sentence-transformer":
        return SentenceTransformerEncoder(model_name=model_name, device=device)
    if encoder_name == "bert-cls":
        return BertClsEncoder(model_name=model_name, device=device)

    raise ValueError(f"Unknown encoder: {encoder_name}")

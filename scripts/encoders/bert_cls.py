"""BERT CLS phrase encoder.

BERT reads the full definition. We then use the final hidden vector of the
first token, `[CLS]`, as the definition-level embedding.
"""

from __future__ import annotations

import numpy as np


class BertClsEncoder:
    def __init__(self, model_name: str, device: str | None = None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are not installed. Install them before "
                "running the BERT CLS encoder path."
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        output_batches: list[np.ndarray] = []

        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                tokenized = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                tokenized = {
                    key: value.to(self.device) for key, value in tokenized.items()
                }

                outputs = self.model(**tokenized)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                output_batches.append(cls_embeddings.cpu().numpy().astype(np.float32))

        return np.vstack(output_batches)

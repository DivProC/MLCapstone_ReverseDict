"""
BiLSTM encoder adapter for the shared reverse-dictionary pipeline.

Wraps the trained BiLSTMEncoder + PredictorMLP so the shared pipeline
can treat it like any other encoder via the standard
    encode(texts: list[str], batch_size: int) -> np.ndarray
interface.

The full model (encoder -> predictor) is used for both index and query
texts, producing L2-normalised 256-dim embeddings. This is a symmetric
approximation of the JEPA design (which uses encoder-only for the index
and predictor-after-encoder for the query); both directions are similar
enough after training that symmetric cosine similarity still works.

Default paths are resolved relative to this file so the adapter works
regardless of the working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ENCODERS_DIR = Path(__file__).resolve().parent           # scripts/encoders/
_SCRIPTS_DIR  = _ENCODERS_DIR.parent                      # scripts/
_PROJECT_ROOT = _SCRIPTS_DIR.parent                       # project root

DEFAULT_CKPT  = _PROJECT_ROOT / "results" / "bilstm" / "bilstm_best.pt"
DEFAULT_VOCAB = _PROJECT_ROOT / "data"    / "processed" / "vocab.pkl"

sys.path.insert(0, str(_SCRIPTS_DIR))


class BiLSTMReverseDictEncoder:
    """Loads a trained BiLSTM checkpoint and exposes an encode() method."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CKPT,
        vocab_path:       str | Path = DEFAULT_VOCAB,
        device:           str | None = None,
    ) -> None:
        import torch
        from bilstm_vocab   import Vocabulary
        from bilstm_encoder import BiLSTMEncoder, PredictorMLP

        self._torch = torch

        if device is None:
            if torch.backends.mps.is_available():  device = "mps"
            elif torch.cuda.is_available():        device = "cuda"
            else:                                  device = "cpu"
        self._device = torch.device(device)

        self._vocab = Vocabulary.load(Path(vocab_path))

        ckpt = torch.load(Path(checkpoint_path), map_location=self._device)
        cfg  = ckpt["config"]

        self._encoder = BiLSTMEncoder(
            len(self._vocab.token2idx),
            cfg["embed_dim"], cfg["hidden_dim"], cfg["output_dim"],
            cfg["num_layers"], dropout=0.0,
        ).to(self._device)
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._encoder.eval()

        self._predictor = PredictorMLP(
            cfg["output_dim"], cfg["output_dim"] * 2, cfg["output_dim"],
        ).to(self._device)
        self._predictor.load_state_dict(ckpt["predictor_state"])
        self._predictor.eval()

        self._max_len = 64

    @property
    def embedding_dim(self) -> int:
        return 256

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """
        Encode a list of definition strings to L2-normalised embeddings.

        Parameters
        ----------
        texts      : raw definition strings (definition_original column)
        batch_size : texts processed per forward pass

        Returns
        -------
        np.ndarray of shape (len(texts), 256), dtype float32
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence

        all_embs: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]

                seqs = []
                for text in chunk:
                    tids = self._vocab.encode_definition(str(text))[: self._max_len]
                    seqs.append(torch.tensor(tids if tids else [1], dtype=torch.long))

                lens   = torch.tensor([len(s) for s in seqs], dtype=torch.long)
                padded = pad_sequence(seqs, batch_first=True, padding_value=0)
                padded = padded.to(self._device)
                lens   = lens.to(self._device)

                embs = self._predictor(self._encoder(padded, lens))
                all_embs.append(embs.cpu().numpy().astype(np.float32))

        return np.vstack(all_embs)

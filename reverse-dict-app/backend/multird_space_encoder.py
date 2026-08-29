"""
Demo encoder for the DefGen2 and unifiedRevdicDefmod baselines trained on MultiRD's
English reverse-dictionary data (see ../../../model_benchmark.ipynb / kaggle_scripts/
in the Capstone repo root).

Unlike the OPTED-based encoders in app.py, these two models were trained against
MultiRD's own ~50k-word candidate space with fixed 300-d pretrained target vectors, so
they can't share the app's OPTED candidate index -- this module rebuilds that MultiRD
vocabulary/candidate space at load time and ranks queries against it directly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DefGen2Model(nn.Module):
    def __init__(self, vocab_size, dim_word, hidden, ctx_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_word, padding_idx=0)
        self.lstm = nn.LSTM(dim_word, hidden, batch_first=True)
        self.out = nn.Linear(hidden, ctx_dim)

    def forward(self, x, lens):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        h_packed, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(2).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.out(pooled), dim=1)


class MultiRDModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, class_vecs, pretrained_emb):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embedding.weight.data.copy_(pretrained_emb)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, emb_dim)
        self.register_buffer("class_vecs", class_vecs)

    def forward(self, x, lens):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        h_packed, (ht, _) = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True, total_length=x.size(1))
        ht_cat = torch.cat([ht[0], ht[1]], dim=1)
        scores = torch.bmm(h, ht_cat.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(x == 0, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(2)
        pooled = (h * attn).sum(1)
        vd = self.fc(pooled)
        return vd @ self.class_vecs.t()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class UnifiedRevdictModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x, lens):
        pad_mask = (x == 0)
        h = self.encoder(self.pos(self.embedding(x)), src_key_padding_mask=pad_mask)
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0)
        return self.proj(F.relu(h.sum(1)))


_MULTIRD_VOCAB_CACHE: dict | None = None


def _load_multird_vocab(data_dir: Path) -> dict:
    """Rebuilds the exact vocab/candidate-space used at training time. Cached across
    encoders since DefGen2 and unifiedRevdicDefmod share the same MultiRD vocabulary."""
    global _MULTIRD_VOCAB_CACHE
    if _MULTIRD_VOCAB_CACHE is not None:
        return _MULTIRD_VOCAB_CACHE

    with open(data_dir / "vec_inuse.json", encoding="utf-8") as f:
        vec_inuse = json.load(f)
    target_words = [
        line.strip() for line in open(data_dir / "target_words.txt", encoding="utf-8")
    ]
    emb_dim = len(next(iter(vec_inuse.values())))

    word2cls = {w: i for i, w in enumerate(target_words)}
    class_vecs = np.zeros((len(target_words), emb_dim), dtype=np.float32)
    for w, i in word2cls.items():
        if w in vec_inuse:
            class_vecs[i] = vec_inuse[w]

    input_vocab = ["<PAD>", "<OOV>"] + list(vec_inuse.keys())
    word2idx = {w: i for i, w in enumerate(input_vocab)}

    example_definition: dict[str, str] = {}
    for filename in ("data_train.json", "data_dev.json", "data_test_500_rand1_seen.json", "data_test_500_rand1_unseen.json"):
        path = data_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for record in json.load(f):
                w = record.get("word")
                if w and w not in example_definition:
                    example_definition[w] = record.get("definitions", "")
        if len(example_definition) >= len(target_words):
            break

    _MULTIRD_VOCAB_CACHE = {
        "target_words": target_words,
        "word2idx": word2idx,
        "class_vecs": class_vecs,
        "emb_dim": emb_dim,
        "example_definition": example_definition,
    }
    return _MULTIRD_VOCAB_CACHE


class MultiRDSpaceDemoEncoder:
    """Loads a DefGen2 or unifiedRevdicDefmod checkpoint and ranks queries against
    MultiRD's own candidate word list (not the app's shared OPTED index)."""

    MODEL_CLASS_BY_ENCODER = {
        "defgen2": "DefGen2Model",
        "unified": "UnifiedRevdictModel",
        "multird": "MultiRDModel",
    }

    def __init__(self, encoder_name: str, data_dir: Path, checkpoint_path: Path, config_path: Path, device: str | None = None):
        if encoder_name not in self.MODEL_CLASS_BY_ENCODER:
            raise ValueError(f"Unknown MultiRD-space encoder: {encoder_name}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.encoder_name = encoder_name

        vocab = _load_multird_vocab(data_dir)
        self.word2idx = vocab["word2idx"]
        self.target_words = vocab["target_words"]
        self.example_definition = vocab["example_definition"]
        class_vecs_t = torch.from_numpy(vocab["class_vecs"]).to(self.device)
        self.class_vecs_norm = F.normalize(class_vecs_t, dim=1)

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)["config"]

        if encoder_name == "defgen2":
            self.model = DefGen2Model(
                vocab_size=cfg["vocab_size"],
                dim_word=cfg["dim_word"],
                hidden=cfg["hidden"],
                ctx_dim=cfg["ctx_dim"],
            ).to(self.device)
        elif encoder_name == "unified":
            self.model = UnifiedRevdictModel(
                vocab_size=cfg["vocab_size"],
                d_model=cfg["d_model"],
                n_head=cfg["n_head"],
                n_layers=cfg["n_layers"],
                out_dim=cfg["out_dim"],
            ).to(self.device)
        else:
            # MultiRD's embedding is frozen and overwritten by load_state_dict below,
            # so a zero-initialized placeholder of the right shape is sufficient here.
            placeholder_emb = torch.zeros((cfg["vocab_size"], cfg["emb_dim"]))
            self.model = MultiRDModel(
                vocab_size=cfg["vocab_size"],
                emb_dim=cfg["emb_dim"],
                hidden=cfg["hidden"],
                class_vecs=class_vecs_t,
                pretrained_emb=placeholder_emb,
            ).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def query(self, text: str, top_k: int) -> list[dict]:
        tokens = str(text).strip().split()
        token_ids = [self.word2idx.get(t, self.word2idx["<OOV>"]) for t in tokens] or [self.word2idx["<OOV>"]]

        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        lens = torch.tensor([len(token_ids)], dtype=torch.long)

        with torch.no_grad():
            out = self.model(x, lens)
            if self.encoder_name == "multird":
                # MultiRD is a softmax classifier over the candidate space -- rank by
                # raw logits directly, same as rank_predictions_softmax at training time.
                sims = out.squeeze(0)
            else:
                out_norm = F.normalize(out, dim=1)
                sims = (out_norm @ self.class_vecs_norm.t()).squeeze(0)
            top = torch.argsort(sims, descending=True)[:top_k].tolist()

        return [
            {
                "rank": rank,
                "word": self.target_words[idx],
                "definition": self.example_definition.get(self.target_words[idx], ""),
                "score": float(sims[idx]),
            }
            for rank, idx in enumerate(top, start=1)
        ]

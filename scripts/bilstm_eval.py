"""
Evaluation script for the BiLSTM reverse dictionary encoder.

Uses a by-definition split to avoid the data-leakage problem that arises
when the word-candidate index is built from the same split used for queries.

Evaluation protocol
-------------------
For each word in the full vocabulary that has >= 2 clean definitions
(from opted_preprocessed.csv):
  - The LAST definition of that word  → held-out query
  - All other definitions of that word → contribute to the word's index entry

Words with exactly one definition are included in the candidate pool (their
single definition is indexed) but are never used as queries.

This guarantees that the definition used as a query is never present in the
word's index entry, so there is no vector-level leakage.

Candidate pool : all 108,839 words in the vocabulary.
Query set      : all words with >= 2 definitions (~24,792 words).

Usage:
    cd <project root>
    python3 scripts/bilstm_eval.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).parent))

from bilstm_vocab import Vocabulary
from bilstm_encoder import BiLSTMEncoder, PredictorMLP

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "bilstm"
CHECKPOINT  = RESULTS_DIR / "bilstm_best.pt"
VOCAB_PATH  = DATA_DIR / "vocab.pkl"
MAX_LEN     = 64


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def tie_averaged_rank(scores: torch.Tensor, target_idx: int) -> float:
    tscore = scores[target_idx].item()
    above  = int((scores > tscore).sum().item())
    tied   = int((scores == tscore).sum().item())
    return above + (tied + 1) / 2


def headword_in_def(word_norm: str, definition: str) -> bool:
    return bool(re.search(r'\b' + re.escape(word_norm.lower()) + r'\b',
                          definition.lower()))


def build_eval_split(vocab: Vocabulary):
    """
    Build a by-definition eval split from opted_preprocessed.csv.

    Returns
    -------
    queries : list of dicts with keys
        word_norm, word_id, query_tids, definition_original
    index_defs : dict[word_norm -> list[list[int]]]
        Definitions used to build each word's index entry.
        For multi-def words: all defs EXCEPT the held-out query def.
        For single-def words: their single definition.
    """
    all_df = pd.read_csv(DATA_DIR / "opted_preprocessed.csv",
                         dtype=str, keep_default_na=False)
    all_df = all_df[all_df["is_unresolved_cross_reference"] != "True"].reset_index(drop=True)

    # Collect encoded definitions per word, preserving row order
    word_rows: dict[str, list[dict]] = defaultdict(list)
    for _, row in all_df.iterrows():
        tids = vocab.encode_definition(row["definition_model_input"])[:MAX_LEN]
        if not tids:
            continue
        word_rows[row["word_norm"]].append({
            "tids": tids,
            "definition_original": row["definition_original"],
        })

    queries    = []
    index_defs = {}

    for word_norm, rows in word_rows.items():
        wid = vocab.encode_word(word_norm)
        if wid < 0:
            continue
        if len(rows) >= 2:
            # Hold out the last definition as the query
            query_row = rows[-1]
            queries.append({
                "word_norm":          word_norm,
                "word_id":            wid,
                "query_tids":         query_row["tids"],
                "definition_original": query_row["definition_original"],
            })
            index_defs[word_norm] = [r["tids"] for r in rows[:-1]]
        else:
            # Single-def word: goes into the candidate index but never queried
            index_defs[word_norm] = [rows[0]["tids"]]

    return queries, index_defs


@torch.no_grad()
def build_word_index(encoder: BiLSTMEncoder,
                     vocab: Vocabulary,
                     index_defs: dict,
                     device: torch.device) -> torch.Tensor:
    """
    Build the candidate word index from the provided definitions dict.
    index_defs[word_norm] must already exclude any definition being used as a query.
    Returns an (num_words, D) L2-normalised tensor.
    """
    encoder.eval()

    flat_seqs, flat_wids = [], []
    for wid, word in enumerate(vocab.idx2word):
        for tids in index_defs.get(word, []):
            flat_seqs.append(torch.tensor(tids, dtype=torch.long))
            flat_wids.append(wid)

    if not flat_seqs:
        return torch.zeros(len(vocab.idx2word), 256, device=device)

    flat_wids_t = torch.tensor(flat_wids, dtype=torch.long)
    padded      = pad_sequence(flat_seqs, batch_first=True, padding_value=0)
    lengths     = torch.tensor([len(s) for s in flat_seqs], dtype=torch.long)

    all_embs = []
    for i in range(0, len(flat_seqs), 512):
        all_embs.append(
            encoder(padded[i:i+512].to(device), lengths[i:i+512].to(device)).cpu()
        )
    all_embs = torch.cat(all_embs, dim=0)

    word_embs = torch.zeros(len(vocab.idx2word), 256)
    counts    = torch.zeros(len(vocab.idx2word))
    word_embs.index_add_(0, flat_wids_t, all_embs)
    counts.index_add_(0, flat_wids_t, torch.ones(len(flat_seqs)))
    word_embs = word_embs / counts.clamp(min=1).unsqueeze(1)
    return F.normalize(word_embs.to(device), dim=1)


@torch.no_grad()
def run_eval():
    device = get_device()
    print(f"Device: {device}")

    vocab = Vocabulary.load(VOCAB_PATH)
    print(vocab)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg  = ckpt["config"]

    encoder = BiLSTMEncoder(
        len(vocab.token2idx), cfg["embed_dim"], cfg["hidden_dim"],
        cfg["output_dim"], cfg["num_layers"], dropout=0.0,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    predictor = PredictorMLP(cfg["output_dim"], cfg["output_dim"] * 2, cfg["output_dim"]).to(device)
    predictor.load_state_dict(ckpt["predictor_state"])
    predictor.eval()

    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, "
          f"val R@10={ckpt['metrics'].get('recall_at_10', 0):.4f}")

    print("Building by-definition eval split...")
    queries, index_defs = build_eval_split(vocab)
    print(f"  Query set : {len(queries):,} definitions  "
          f"(words with >= 2 definitions)")
    print(f"  Candidate pool: {len(vocab.idx2word):,} words")

    print("Building word index (query definitions excluded)...")
    word_index = build_word_index(encoder, vocab, index_defs, device)

    # --- Batched evaluation ---
    BATCH = 128
    all_ranks, no_hw_ranks = [], []
    top_preds = []

    for start in range(0, len(queries), BATCH):
        batch = queries[start: start + BATCH]
        seqs  = [torch.tensor(q["query_tids"], dtype=torch.long) for q in batch]
        lens  = torch.tensor([len(q["query_tids"]) for q in batch], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True, padding_value=0).to(device)

        predicted = predictor(encoder(padded, lens.to(device)))
        sims      = predicted @ word_index.T   # (batch, num_words)

        for i, q in enumerate(batch):
            rank = tie_averaged_rank(sims[i], q["word_id"])
            all_ranks.append(rank)

            if not headword_in_def(q["word_norm"], q["definition_original"]):
                no_hw_ranks.append(rank)

            if len(top_preds) < 20:
                top_idx = int(sims[i].argmax().item())
                top_preds.append({
                    "query_word":       q["word_norm"],
                    "query_definition": q["definition_original"],
                    "predicted_word":   vocab.idx2word[top_idx],
                    "correct_rank":     rank,
                    "top_score":        float(sims[i][top_idx].item()),
                    "target_score":     float(sims[i][q["word_id"]].item()),
                    "headword_in_def":  headword_in_def(q["word_norm"],
                                                        q["definition_original"]),
                })

    V = len(vocab.idx2word)

    def summary(ranks, label):
        r = np.array(ranks)
        return {
            "model":         "BiLSTM",
            "subset":        label,
            "eval_protocol": "by_definition_split",
            "num_queries":   len(r),
            "num_candidates": V,
            "recall_at_1":   float((r <= 1).mean()),
            "recall_at_10":  float((r <= 10).mean()),
            "recall_at_100": float((r <= 100).mean()),
            "median_rank":   float(np.median(r)),
        }

    rows = [
        summary(all_ranks,    "all_queries"),
        summary(no_hw_ranks,  "headword_not_in_definition"),
    ]

    print("\n" + "="*60)
    print("TEST SET RESULTS  (by-definition split, no index leakage)")
    print("="*60)
    for row in rows:
        print(f"\n  Subset     : {row['subset']}")
        print(f"  Queries    : {row['num_queries']:,}   Candidates: {row['num_candidates']:,}")
        print(f"  Recall@1   : {row['recall_at_1']:.4f}")
        print(f"  Recall@10  : {row['recall_at_10']:.4f}")
        print(f"  Recall@100 : {row['recall_at_100']:.4f}")
        print(f"  Median Rank: {row['median_rank']:.1f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "bilstm_metrics.csv", index=False)
    pd.DataFrame(top_preds).to_csv(RESULTS_DIR / "bilstm_sample_predictions.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_eval()

"""
Evaluation script for the BiLSTM reverse dictionary encoder.

Runs on the held-out test split and reports:
  - Recall@1, Recall@10, Recall@100, Median Rank (tie-averaged)
  - Full candidate pool (all words in the dictionary)
  - All test queries
  - Subset where the answer word does not appear inside the query definition

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
from bilstm_dataset import make_dataloader
from bilstm_encoder import BiLSTMEncoder, PredictorMLP

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "bilstm"
CHECKPOINT  = RESULTS_DIR / "bilstm_best.pt"
VOCAB_PATH  = DATA_DIR / "vocab.pkl"
TEST_CSV    = DATA_DIR / "opted_test.csv"
MAX_LEN     = 64


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def build_word_index(encoder, vocab, device):
    all_defs: dict[str, list[list[int]]] = defaultdict(list)
    for csv in [DATA_DIR/"opted_train.csv", DATA_DIR/"opted_valid.csv", DATA_DIR/"opted_test.csv"]:
        df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        df = df[df["is_unresolved_cross_reference"] != "True"]
        for _, row in df.iterrows():
            tids = vocab.encode_definition(row["definition_model_input"])[:MAX_LEN]
            if tids:
                all_defs[row["word_norm"]].append(tids)

    flat_seqs, flat_wids = [], []
    for wid, word in enumerate(vocab.idx2word):
        for tids in all_defs.get(word, []):
            flat_seqs.append(torch.tensor(tids, dtype=torch.long))
            flat_wids.append(wid)

    flat_wids_t = torch.tensor(flat_wids, dtype=torch.long)
    padded      = pad_sequence(flat_seqs, batch_first=True, padding_value=0)
    lengths     = torch.tensor([len(s) for s in flat_seqs], dtype=torch.long)

    all_embs = []
    for i in range(0, len(flat_seqs), 512):
        all_embs.append(encoder(padded[i:i+512].to(device), lengths[i:i+512].to(device)).cpu())
    all_embs = torch.cat(all_embs, dim=0)

    word_embs = torch.zeros(len(vocab.idx2word), 256)
    counts    = torch.zeros(len(vocab.idx2word))
    word_embs.index_add_(0, flat_wids_t, all_embs)
    counts.index_add_(0, flat_wids_t, torch.ones(len(flat_seqs)))
    word_embs = word_embs / counts.clamp(min=1).unsqueeze(1)
    return F.normalize(word_embs.to(device), dim=1)


def tie_averaged_rank(scores, target_idx):
    tscore = scores[target_idx].item()
    above  = int((scores > tscore).sum().item())
    tied   = int((scores == tscore).sum().item())
    return above + (tied + 1) / 2


def headword_in_def(word_norm: str, definition: str) -> bool:
    return bool(re.search(r'\b' + re.escape(word_norm.lower()) + r'\b', definition.lower()))


@torch.no_grad()
def run_eval():
    device = get_device()
    print(f"Device: {device}")

    vocab = Vocabulary.load(VOCAB_PATH)
    print(vocab)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    cfg  = ckpt["config"]

    encoder = BiLSTMEncoder(len(vocab.token2idx), cfg["embed_dim"], cfg["hidden_dim"],
                             cfg["output_dim"], cfg["num_layers"], 0.0).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    predictor = PredictorMLP(cfg["output_dim"], cfg["output_dim"] * 2, cfg["output_dim"]).to(device)
    predictor.load_state_dict(ckpt["predictor_state"])
    predictor.eval()

    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, "
          f"val R@10={ckpt['metrics'].get('recall_at_10', 0):.4f}")

    print("Building word index...")
    word_index = build_word_index(encoder, vocab, device)

    print("Loading test set...")
    test_loader = make_dataloader(TEST_CSV, vocab, batch_size=128, shuffle=False)
    raw_df = (pd.read_csv(TEST_CSV, dtype=str, keep_default_na=False)
                .pipe(lambda d: d[d["is_unresolved_cross_reference"] != "True"])
                .reset_index(drop=True))

    all_ranks, no_hw_ranks = [], []
    top_preds  = []
    sample_idx = 0

    for q_ids, q_lens, _, _, word_ids in test_loader:
        predicted = predictor(encoder(q_ids.to(device), q_lens.to(device)))
        sims      = predicted @ word_index.T

        for i, wid in enumerate(word_ids):
            rank = tie_averaged_rank(sims[i], wid)
            all_ranks.append(rank)

            src = raw_df.iloc[sample_idx]
            if not headword_in_def(src["word_norm"], src["definition_original"]):
                no_hw_ranks.append(rank)

            if len(top_preds) < 20:
                top_idx = int(sims[i].argmax().item())
                top_preds.append({
                    "query_word":       src["word_original"],
                    "query_definition": src["definition_original"],
                    "predicted_word":   vocab.idx2word[top_idx],
                    "correct_rank":     rank,
                    "top_score":        float(sims[i][top_idx].item()),
                    "target_score":     float(sims[i][wid].item()),
                    "headword_in_def":  headword_in_def(src["word_norm"], src["definition_original"]),
                })
            sample_idx += 1

    V = len(vocab.idx2word)

    def summary(ranks, label):
        r = np.array(ranks)
        return {"model": "BiLSTM", "subset": label, "num_queries": len(r),
                "num_candidates": V,
                "recall_at_1":   float((r <= 1).mean()),
                "recall_at_10":  float((r <= 10).mean()),
                "recall_at_100": float((r <= 100).mean()),
                "median_rank":   float(np.median(r))}

    rows = [summary(all_ranks, "all_queries"),
            summary(no_hw_ranks, "headword_not_in_definition")]

    print("\n" + "="*55)
    print("TEST SET RESULTS")
    print("="*55)
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

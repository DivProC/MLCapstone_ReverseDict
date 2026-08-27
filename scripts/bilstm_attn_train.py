"""
Training script for the BiLSTM reverse dictionary encoder.

Query definition  -> BiLSTMEncoder -> PredictorMLP -> predicted embedding
Target definition -> BiLSTMEncoder  (stop-gradient, different def of same word)
Loss: symmetric in-batch contrastive (CLIP-style, temperature-scaled).

Usage:
    cd <project root>
    python3 -u scripts/bilstm_attn_train.py
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent))

from bilstm_vocab import Vocabulary
from bilstm_dataset import make_dataloader
from bilstm_attn_encoder import BiLSTMAttnEncoder as BiLSTMEncoder, PredictorMLP

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "bilstm_attn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH      = DATA_DIR / "vocab.pkl"
TRAIN_CSV       = DATA_DIR / "opted_train.csv"
VALID_CSV       = DATA_DIR / "opted_valid.csv"
CHECKPOINT_PATH = RESULTS_DIR / "bilstm_attn_best.pt"

EMBED_DIM   = 256
HIDDEN_DIM  = 256
OUTPUT_DIM  = 256
NUM_LAYERS  = 2
ATTN_DIM    = 128   # new
NUM_HEADS   = 4     # new
DROPOUT     = 0.3
BATCH_SIZE  = 256
MAX_LEN     = 64
EPOCHS      = 30
LR          = 1e-3
TEMPERATURE = 0.07
PATIENCE    = 4
INDEX_EVERY = 1   # rebuild word index every epoch for accurate checkpoint selection


def get_device() -> torch.device:
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")


def contrastive_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sim    = predicted @ target.T / TEMPERATURE
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def load_all_defs(vocab: Vocabulary) -> dict[str, list[list[int]]]:
    """
    Load encoded definitions for the word index used during training validation.
    Intentionally excludes opted_test.csv so test-split definitions never
    appear in the candidate index during training.
    """
    all_defs: dict[str, list[list[int]]] = defaultdict(list)
    for csv in [DATA_DIR/"opted_train.csv", DATA_DIR/"opted_valid.csv"]:
        df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        df = df[df["is_unresolved_cross_reference"] != "True"]
        for _, row in df.iterrows():
            tids = vocab.encode_definition(row["definition_model_input"])[:MAX_LEN]
            if tids:
                all_defs[row["word_norm"]].append(tids)
    return all_defs


@torch.no_grad()
def build_word_index(encoder: BiLSTMEncoder, vocab: Vocabulary,
                     all_defs: dict, device: torch.device) -> torch.Tensor:
    """
    Encode every candidate word as the mean of its definition embeddings.
    All definitions are batched together for efficiency, then grouped by word.
    Returns (num_words, OUTPUT_DIM) L2-normalised tensor.
    """
    encoder.eval()

    # Flatten all (def_tokens, word_id) pairs into one big list
    flat_seqs, flat_wids = [], []
    for wid, word in enumerate(vocab.idx2word):
        for tids in all_defs.get(word, []):
            flat_seqs.append(torch.tensor(tids, dtype=torch.long))
            flat_wids.append(wid)

    if not flat_seqs:
        return torch.zeros(len(vocab.idx2word), OUTPUT_DIM, device=device)

    flat_wids_t = torch.tensor(flat_wids, dtype=torch.long)
    padded = pad_sequence(flat_seqs, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(s) for s in flat_seqs], dtype=torch.long)

    # Single batched forward pass over all definitions
    all_embs = []
    CHUNK = 512
    for i in range(0, len(flat_seqs), CHUNK):
        e = encoder(padded[i:i+CHUNK].to(device), lengths[i:i+CHUNK].to(device))
        all_embs.append(e.cpu())
    all_embs = torch.cat(all_embs, dim=0)  # (total_defs, D)

    # Mean-pool per word
    word_embs = torch.zeros(len(vocab.idx2word), OUTPUT_DIM)
    counts    = torch.zeros(len(vocab.idx2word))
    word_embs.index_add_(0, flat_wids_t, all_embs)
    counts.index_add_(0, flat_wids_t, torch.ones(len(flat_seqs)))
    counts = counts.clamp(min=1).unsqueeze(1)
    word_embs = word_embs / counts

    return F.normalize(word_embs.to(device), dim=1)


@torch.no_grad()
def evaluate(encoder, predictor, loader, vocab, device, word_index):
    import numpy as np
    encoder.eval(); predictor.eval()
    ranks = []
    for q_ids, q_lens, _, _, word_ids in loader:
        predicted = predictor(encoder(q_ids.to(device), q_lens.to(device)))
        sims      = predicted @ word_index.T
        for i, wid in enumerate(word_ids):
            row    = sims[i]
            tscore = row[wid].item()
            above  = int((row > tscore).sum().item())
            tied   = int((row == tscore).sum().item())
            ranks.append(above + (tied + 1) / 2)
    ranks = np.array(ranks)
    return {
        "recall_at_1":   float((ranks <= 1).mean()),
        "recall_at_10":  float((ranks <= 10).mean()),
        "recall_at_100": float((ranks <= 100).mean()),
        "median_rank":   float(__import__("numpy").median(ranks)),
        "num_queries":   len(ranks),
    }


def train():
    device = get_device()
    print(f"Device: {device}")

    if VOCAB_PATH.exists():
        vocab = Vocabulary.load(VOCAB_PATH)
    else:
        vocab = Vocabulary(min_freq=2); vocab.build(TRAIN_CSV); vocab.save(VOCAB_PATH)
    print(vocab)

    print("\nLoading data...")
    train_loader = make_dataloader(TRAIN_CSV, vocab, BATCH_SIZE, shuffle=True,  max_len=MAX_LEN)
    valid_loader = make_dataloader(VALID_CSV, vocab, 128,        shuffle=False, max_len=MAX_LEN)

    print("Pre-loading all definitions for word index...")
    all_defs = load_all_defs(vocab)

    encoder   = BiLSTMEncoder(len(vocab.token2idx), EMBED_DIM, HIDDEN_DIM,
                               OUTPUT_DIM, NUM_LAYERS, DROPOUT,attn_dim=ATTN_DIM, num_heads=NUM_HEADS).to(device)
    predictor = PredictorMLP(OUTPUT_DIM, OUTPUT_DIM * 2, OUTPUT_DIM).to(device)

    print(f"Encoder params : {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    optimizer = AdamW(list(encoder.parameters()) + list(predictor.parameters()),
                      lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=PATIENCE)

    start_epoch, best_r10 = 1, -1.0
    log_path = RESULTS_DIR / "train_log.csv"

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        if "predictor_state" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state"])
            predictor.load_state_dict(ckpt["predictor_state"])
            start_epoch = ckpt["epoch"] + 1
            best_r10    = ckpt["metrics"].get("recall_at_10", -1.0)
            print(f"Resumed from epoch {start_epoch}, best R@10={best_r10:.4f}")
        else:
            print("Checkpoint architecture mismatch — starting from scratch.")

    if not log_path.exists() or start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_r1,val_r10,val_r100,val_median_rank,epoch_secs\n")

    word_index = None

    print(f"\n{'='*60}\nTraining epochs {start_epoch} to {EPOCHS}\n{'='*60}")

    for epoch in range(start_epoch, EPOCHS + 1):
        encoder.train(); predictor.train()
        epoch_loss, t0 = 0.0, time.time()

        for step, (q_ids, q_lens, t_ids, t_lens, _) in enumerate(train_loader, 1):
            q_ids = q_ids.to(device); q_lens = q_lens.to(device)
            t_ids = t_ids.to(device); t_lens = t_lens.to(device)

            predicted = predictor(encoder(q_ids, q_lens))
            with torch.no_grad():
                target = encoder(t_ids, t_lens)

            loss = contrastive_loss(predicted, target)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}", end="\r")

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"\nEpoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  ({elapsed:.0f}s)")

        if word_index is None or epoch % INDEX_EVERY == 0:
            print("  Building word index...")
            word_index = build_word_index(encoder, vocab, all_defs, device)

        metrics = evaluate(encoder, predictor, valid_loader, vocab, device, word_index)
        print(f"  R@1={metrics['recall_at_1']:.4f}  R@10={metrics['recall_at_10']:.4f}  "
              f"R@100={metrics['recall_at_100']:.4f}  MedRank={metrics['median_rank']:.0f}")

        scheduler.step(1 - metrics["recall_at_10"])

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{metrics['recall_at_1']:.6f},"
                    f"{metrics['recall_at_10']:.6f},{metrics['recall_at_100']:.6f},"
                    f"{metrics['median_rank']:.1f},{elapsed:.1f}\n")

        if metrics["recall_at_10"] > best_r10:
            best_r10 = metrics["recall_at_10"]
            torch.save({
                "epoch":           epoch,
                "encoder_state":   encoder.state_dict(),
                "predictor_state": predictor.state_dict(),
                "metrics":         metrics,
                "config": dict(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                               output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS,
                               dropout=DROPOUT, attn_dim=ATTN_DIM, num_heads=NUM_HEADS),
            }, CHECKPOINT_PATH)
            print(f"  Saved best checkpoint  R@10={best_r10:.4f}")

    print(f"\nTraining complete. Best val R@10={best_r10:.4f}")


if __name__ == "__main__":
    train()

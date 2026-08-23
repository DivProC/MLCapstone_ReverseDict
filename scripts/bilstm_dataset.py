"""
Dataset for the BiLSTM reverse dictionary encoder.

Each training sample is a (query_def, target_def, word_id) triple where
query_def and target_def are two definitions of the same word. When a word
has only one definition, the same text is used for both sides.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from bilstm_vocab import Vocabulary


class ReverseDictDataset(Dataset):

    def __init__(self, csv_path: Path, vocab: Vocabulary, max_len: int = 64) -> None:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        df = df[df["is_unresolved_cross_reference"] != "True"].reset_index(drop=True)

        # word -> all encoded definitions
        self.word_to_defs: dict[str, list[list[int]]] = defaultdict(list)
        for _, row in df.iterrows():
            tids = vocab.encode_definition(row["definition_model_input"])[:max_len]
            if tids:
                self.word_to_defs[row["word_norm"]].append(tids)

        self.samples: list[tuple[list[int], str, int]] = []
        skipped = 0
        for _, row in df.iterrows():
            wid  = vocab.encode_word(row["word_norm"])
            tids = vocab.encode_definition(row["definition_model_input"])[:max_len]
            if wid < 0 or not tids:
                skipped += 1
                continue
            self.samples.append((tids, row["word_norm"], wid))

        if skipped:
            print(f"  [{csv_path.name}] skipped {skipped:,} rows")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        query_ids, word_norm, word_id = self.samples[idx]
        all_defs = self.word_to_defs[word_norm]
        if len(all_defs) > 1:
            others = [d for d in all_defs if d != query_ids]
            target_ids = random.choice(others) if others else query_ids
        else:
            target_ids = query_ids

        return (
            torch.tensor(query_ids,  dtype=torch.long),
            torch.tensor(len(query_ids),  dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(len(target_ids), dtype=torch.long),
            torch.tensor(word_id,    dtype=torch.long),
        )


def collate_fn(batch):
    q_seqs, q_lens, t_seqs, t_lens, wids = zip(*batch)
    return (
        pad_sequence(q_seqs, batch_first=True, padding_value=0),
        torch.stack(q_lens),
        pad_sequence(t_seqs, batch_first=True, padding_value=0),
        torch.stack(t_lens),
        torch.stack(wids),
    )


def make_dataloader(
    csv_path: Path,
    vocab: Vocabulary,
    batch_size: int = 256,
    shuffle: bool = True,
    max_len: int = 64,
) -> DataLoader:
    dataset = ReverseDictDataset(csv_path, vocab, max_len=max_len)
    print(f"  Dataset size: {len(dataset):,} examples")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0, pin_memory=False)

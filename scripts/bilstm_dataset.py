"""
Dataset for the BiLSTM reverse dictionary encoder.

Each training sample is a (query_def, target_def, word_id) triple where
query_def and target_def are two DISTINCT definitions of the same word.
Words with only one definition are excluded — passing the same text as both
query and target would teach the predictor to be an identity map.

Target selection uses index-based lookup (not value equality) so that two
definitions which happen to tokenise identically are still treated as
separate objects and can serve as each other's target.
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

        # Build a flat list of all valid (tids, word_norm, wid) entries and a
        # per-word index so we can pick a distinct TARGET by position, not by value.
        self._defs: list[tuple[list[int], str, int]] = []
        self._word_def_indices: dict[str, list[int]] = defaultdict(list)

        for _, row in df.iterrows():
            wid  = vocab.encode_word(row["word_norm"])
            tids = vocab.encode_definition(row["definition_model_input"])[:max_len]
            if wid < 0 or not tids:
                continue
            pos = len(self._defs)
            self._defs.append((tids, row["word_norm"], wid))
            self._word_def_indices[row["word_norm"]].append(pos)

        # Samples: only positions whose word has at least 2 definitions so we
        # can always supply a strictly different target position.
        self.samples: list[int] = []
        skipped_single = 0
        for pos, (_, word_norm, _) in enumerate(self._defs):
            if len(self._word_def_indices[word_norm]) >= 2:
                self.samples.append(pos)
            else:
                skipped_single += 1

        total_skipped = len(self._defs) + skipped_single - len(self.samples) - len(self._defs)
        print(f"  [{csv_path.name}] {len(self.samples):,} training samples  "
              f"({skipped_single:,} rows skipped — single-definition words)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pos = self.samples[idx]
        query_ids, word_norm, word_id = self._defs[pos]

        # Pick any other position for the same word as the target.
        other_positions = [p for p in self._word_def_indices[word_norm] if p != pos]
        target_pos  = random.choice(other_positions)
        target_ids  = self._defs[target_pos][0]

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0, pin_memory=False)

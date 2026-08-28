"""
Vocabulary builder for the BiLSTM reverse dictionary encoder.
Reads opted_train.csv and builds token -> index mappings for
definition tokens and candidate words.
"""
from __future__ import annotations

import re
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
START_IDX = 2
END_IDX = 3

TOKEN_RE = re.compile(r"<[A-Z]+>|[a-z']+")

# character vocabulary for word encoding
CHAR_PAD = 0
CHAR_UNK = 1
CHAR_VOCAB = ["\x00", "\x01"] + list("abcdefghijklmnopqrstuvwxyz'-0123456789 ")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2idx: dict[str, int] = {}
        self.idx2token: list[str] = []
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    # ------------------------------------------------------------------
    # Build from training CSV
    # ------------------------------------------------------------------
    def build(self, train_csv: Path) -> None:
        df = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
        df = df[df["is_unresolved_cross_reference"] != "True"]

        # --- definition vocabulary (tokens from training definitions only) ---
        token_counts: Counter = Counter()
        for text in df["definition_model_input"]:
            token_counts.update(tokenize(text))

        self.idx2token = list(SPECIAL_TOKENS)
        for token, count in token_counts.most_common():
            if token in SPECIAL_TOKENS:
                continue
            if count >= self.min_freq:
                self.idx2token.append(token)
        self.token2idx = {t: i for i, t in enumerate(self.idx2token)}

        # --- candidate word vocabulary (ALL words across all splits) ---
        # We need embeddings for test/valid words at eval time too,
        # since those splits are held-out words not seen in training.
        preprocessed_csv = train_csv.parent / "opted_preprocessed.csv"
        if preprocessed_csv.exists():
            all_df = pd.read_csv(preprocessed_csv, dtype=str, keep_default_na=False)
            all_df = all_df[all_df["is_unresolved_cross_reference"] != "True"]
        else:
            # fallback: load all three splits
            dfs = [df]
            for split in ("opted_valid.csv", "opted_test.csv"):
                p = train_csv.parent / split
                if p.exists():
                    extra = pd.read_csv(p, dtype=str, keep_default_na=False)
                    extra = extra[extra["is_unresolved_cross_reference"] != "True"]
                    dfs.append(extra)
            all_df = pd.concat(dfs, ignore_index=True)

        self.idx2word = sorted(all_df["word_norm"].unique().tolist())
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    # ------------------------------------------------------------------
    # Encode helpers
    # ------------------------------------------------------------------
    def encode_definition(self, text: str) -> list[int]:
        return [
            self.token2idx.get(t, UNK_IDX)
            for t in tokenize(text)
        ]

    def encode_word(self, word: str) -> int:
        return self.word2idx.get(str(word).lower(), -1)

    def encode_word_chars(self, word: str, max_len: int = 32) -> list[int]:
        char2idx = {c: i for i, c in enumerate(CHAR_VOCAB)}
        chars = list(str(word).lower())[:max_len]
        return [char2idx.get(c, CHAR_UNK) for c in chars]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved vocabulary to {path}")

    @staticmethod
    def load(path: Path) -> "Vocabulary":
        # vocab.pkl may have been saved while bilstm_vocab was the __main__
        # module, so pickle stores the class as __main__.Vocabulary.
        # The custom unpickler remaps that to the live Vocabulary class so
        # loading works regardless of how the module was invoked.
        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "Vocabulary":
                    return Vocabulary
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return _Unpickler(f).load()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Vocabulary(tokens={len(self.token2idx):,}, "
            f"words={len(self.word2idx):,})"
        )


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    train_csv = base / "data" / "processed" / "opted_train.csv"
    vocab_path = base / "data" / "processed" / "vocab.pkl"

    print("Building vocabulary from training data...")
    vocab = Vocabulary(min_freq=2)
    vocab.build(train_csv)
    print(vocab)
    vocab.save(vocab_path)

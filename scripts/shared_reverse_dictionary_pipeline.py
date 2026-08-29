"""
Shared reverse-dictionary pipeline for OPTED experiments.

This file is meant to be the common base for the encoder variants. The encoder
can change, but data loading, candidate indexing, training, and evaluation should
stay the same.

Examples
--------
Plan the data split without loading a neural model:

    python scripts/shared_reverse_dictionary_pipeline.py --plan-only

Run a Sentence-BERT encoder once dependencies are installed:

    python scripts/shared_reverse_dictionary_pipeline.py \
        --encoder sentence-transformer \
        --model-name sentence-transformers/all-MiniLM-L6-v2 \
        --predictor none \
        --split-mode by-definition \
        --eval-split test

Run BERT CLS encoder path with the shared predictor:

    python scripts/shared_reverse_dictionary_pipeline.py \
        --encoder bert-cls \
        --model-name bert-base-uncased \
        --predictor torch-mlp-infonce \
        --split-mode by-definition \
        --eval-split test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

try:
    from encoders import DEFAULT_MODEL_BY_ENCODER, ENCODER_CHOICES, make_encoder
except ImportError:
    from scripts.encoders import (
        DEFAULT_MODEL_BY_ENCODER,
        ENCODER_CHOICES,
        make_encoder,
    )


TEXT_COLUMN = "definition_original"
CORRECT_KEY_COLUMN = "definition_basic_clean"
RANDOM_SEED = 42


@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    split_summary: pd.DataFrame


@dataclass
class TargetIndex:
    candidate_df: pd.DataFrame
    candidate_words: list[str]
    candidate_word_ids: np.ndarray
    word_to_id: dict[str, int]
    definition_to_words: dict[str, set[str]]


class TorchMlpInfoNcePredictor:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        temperature: float,
        seed: int,
        device: str | None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is not installed. Install it before using "
                "--predictor torch-mlp-infonce."
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.seed = seed

        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        word_ids: np.ndarray,
    ) -> "TorchMlpInfoNcePredictor":
        torch = self.torch
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        word_id_tensor = torch.tensor(word_ids, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, word_id_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_rows = 0
            for batch_x, batch_y, batch_word_ids in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_word_ids = batch_word_ids.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                predictions = torch.nn.functional.normalize(predictions, dim=1)
                targets = torch.nn.functional.normalize(batch_y, dim=1)

                logits = predictions @ targets.T
                logits = logits / self.temperature

                batch_size = len(batch_x)
                same_word = batch_word_ids[:, None] == batch_word_ids[None, :]
                diagonal = torch.eye(batch_size, dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(same_word & ~diagonal, -1e9)

                labels = torch.arange(batch_size, device=self.device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item()) * len(batch_x)
                total_rows += len(batch_x)

            print(
                f"infonce epoch {epoch}/{self.epochs} - "
                f"loss {total_loss / max(total_rows, 1):.6f}"
            )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                batch = torch.tensor(
                    x[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                outputs.append(self.model(batch).cpu().numpy().astype(np.float32))
        return np.vstack(outputs)

    def save(self, path: Path, extra_config: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "predictor_config": {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "hidden_dim": self.hidden_dim,
                    "dropout": self.dropout,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "temperature": self.temperature,
                    "seed": self.seed,
                },
                "run_config": extra_config or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "TorchMlpInfoNcePredictor":
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is not installed. Install it before loading an InfoNCE predictor."
            ) from exc

        load_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=load_device)
        config = checkpoint["predictor_config"]
        predictor = cls(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            epochs=int(config.get("epochs", 1)),
            batch_size=int(config.get("batch_size", 256)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            temperature=float(config.get("temperature", 0.05)),
            seed=int(config.get("seed", RANDOM_SEED)),
            device=load_device,
        )
        predictor.model.load_state_dict(checkpoint["model_state_dict"])
        predictor.model.eval()
        return predictor


def read_processed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def add_headword_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def headword_in_definition(row: pd.Series) -> bool:
        word = normalize_for_match(row["word_norm"])
        definition = normalize_for_match(row[CORRECT_KEY_COLUMN])
        if not word or not definition:
            return False
        pattern = rf"(^|\s){re.escape(word)}(\s|$)"
        return bool(re.search(pattern, definition))

    df["headword_in_own_definition"] = df.apply(headword_in_definition, axis=1)
    return df


def normalize_for_match(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_by_headword_split(processed_dir: Path) -> SplitData:
    train_df = read_processed_csv(processed_dir / "opted_train.csv")
    valid_df = read_processed_csv(processed_dir / "opted_valid.csv")
    test_df = read_processed_csv(processed_dir / "opted_test.csv")

    split_summary = pd.DataFrame(
        [
            summarize_split("train", train_df),
            summarize_split("valid", valid_df),
            summarize_split("test", test_df),
        ]
    )
    return SplitData(train_df, valid_df, test_df, split_summary)


def make_by_definition_split(processed_dir: Path, seed: int) -> SplitData:
    full_df = read_processed_csv(processed_dir / "opted_preprocessed.csv")
    rng = random.Random(seed)

    split_names = pd.Series("train", index=full_df.index, dtype="object")

    for _, word_df in full_df.groupby("word_norm", sort=True):
        row_indices = list(word_df.index)
        rng.shuffle(row_indices)
        row_count = len(row_indices)

        if row_count == 1:
            train_indices = row_indices
            valid_indices: list[int] = []
            test_indices: list[int] = []
        elif row_count == 2:
            train_indices = row_indices[:1]
            valid_indices = []
            test_indices = row_indices[1:]
        else:
            train_count = max(1, int(row_count * 0.80))
            valid_count = max(1, int(row_count * 0.10))
            if train_count + valid_count >= row_count:
                train_count = row_count - 2
                valid_count = 1

            valid_end = train_count + valid_count
            train_indices = row_indices[:train_count]
            valid_indices = row_indices[train_count:valid_end]
            test_indices = row_indices[valid_end:]

        split_names.loc[train_indices] = "train"
        split_names.loc[valid_indices] = "valid"
        split_names.loc[test_indices] = "test"

    full_df = full_df.copy()
    full_df["pipeline_split"] = split_names

    train_df = full_df[full_df["pipeline_split"] == "train"].copy()
    valid_df = full_df[full_df["pipeline_split"] == "valid"].copy()
    test_df = full_df[full_df["pipeline_split"] == "test"].copy()

    train_words = set(train_df["word_norm"])
    missing_valid_words = set(valid_df["word_norm"]) - train_words
    missing_test_words = set(test_df["word_norm"]) - train_words
    if missing_valid_words or missing_test_words:
        raise RuntimeError(
            "By-definition split failed: validation/test contains words missing "
            "from training."
        )

    split_summary = pd.DataFrame(
        [
            summarize_split("train", train_df),
            summarize_split("valid", valid_df),
            summarize_split("test", test_df),
        ]
    )
    return SplitData(train_df, valid_df, test_df, split_summary)


def summarize_split(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "split": name,
        "rows": len(df),
        "unique_words": df["word_norm"].nunique(),
        "avg_definition_words": round(df["definition_word_count"].astype(float).mean(), 2),
    }


def load_split(args: argparse.Namespace) -> SplitData:
    processed_dir = Path(args.processed_dir)
    if args.split_mode == "by-headword":
        split_data = make_by_headword_split(processed_dir)
    elif args.split_mode == "by-definition":
        split_data = make_by_definition_split(processed_dir, args.random_seed)
    else:
        raise ValueError(f"Unknown split mode: {args.split_mode}")

    split_data.train_df = add_headword_filter(split_data.train_df)
    split_data.valid_df = add_headword_filter(split_data.valid_df)
    split_data.test_df = add_headword_filter(split_data.test_df)
    return split_data


def maybe_limit_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True).copy()
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True).copy()


def add_augmented_training_rows(
    train_df: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    augmented_path = processed_dir / "opted_train_augmented_basic.csv"
    augmented_df = read_processed_csv(augmented_path)
    augmented_only_df = augmented_df[
        augmented_df["augmentation_type"].ne("original")
    ].copy()

    train_entry_ids = set(train_df["entry_id"])
    augmented_only_df = augmented_only_df[
        augmented_only_df["source_entry_id"].isin(train_entry_ids)
    ].copy()

    shared_columns = [column for column in train_df.columns if column in augmented_only_df.columns]
    augmented_only_df = augmented_only_df[shared_columns]

    model_train_df = pd.concat(
        [train_df, augmented_only_df],
        ignore_index=True,
    )
    print(f"Augmented predictor-training rows added: {len(augmented_only_df):,}")
    return model_train_df


def build_target_index(train_df: pd.DataFrame, all_df: pd.DataFrame) -> TargetIndex:
    candidate_df = train_df.reset_index(drop=True).copy()
    candidate_words = sorted(candidate_df["word_norm"].unique())
    word_to_id = {word: idx for idx, word in enumerate(candidate_words)}
    candidate_word_ids = candidate_df["word_norm"].map(word_to_id).to_numpy(dtype=np.int64)

    definition_to_words: dict[str, set[str]] = {}
    for definition_key, rows in all_df.groupby(CORRECT_KEY_COLUMN):
        definition_to_words[definition_key] = set(rows["word_norm"])

    return TargetIndex(
        candidate_df=candidate_df,
        candidate_words=candidate_words,
        candidate_word_ids=candidate_word_ids,
        word_to_id=word_to_id,
        definition_to_words=definition_to_words,
    )


def train_predictor(
    predictor_name: str,
    train_embeddings: np.ndarray,
    train_words: pd.Series,
    index_embeddings: np.ndarray,
    index_word_ids: np.ndarray,
    word_to_id: dict[str, int],
    args: argparse.Namespace,
):
    if predictor_name == "none":
        return None

    word_prototypes = build_word_prototypes(
        index_embeddings=index_embeddings,
        index_word_ids=index_word_ids,
        num_words=len(word_to_id),
    )
    target_ids = train_words.map(word_to_id).to_numpy(dtype=np.int64)
    target_embeddings = word_prototypes[target_ids]

    if predictor_name == "ridge":
        predictor = Ridge(alpha=args.ridge_alpha, random_state=args.random_seed)
        predictor.fit(train_embeddings, target_embeddings)
        return predictor

    if predictor_name == "torch-mlp-infonce":
        predictor = TorchMlpInfoNcePredictor(
            input_dim=train_embeddings.shape[1],
            output_dim=target_embeddings.shape[1],
            hidden_dim=args.mlp_hidden_dim,
            dropout=args.dropout,
            epochs=args.projection_epochs,
            batch_size=args.projection_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            seed=args.random_seed,
            device=args.device,
        )
        return predictor.fit(train_embeddings, target_embeddings, target_ids)

    raise ValueError(f"Unknown predictor: {predictor_name}")


def build_word_prototypes(
    index_embeddings: np.ndarray,
    index_word_ids: np.ndarray,
    num_words: int,
) -> np.ndarray:
    sums = np.zeros((num_words, index_embeddings.shape[1]), dtype=np.float64)
    counts = np.zeros(num_words, dtype=np.float64)
    np.add.at(sums, index_word_ids, index_embeddings)
    np.add.at(counts, index_word_ids, 1)
    counts[counts == 0] = 1
    return (sums / counts[:, None]).astype(np.float32)


def apply_predictor(predictor, embeddings: np.ndarray) -> np.ndarray:
    if predictor is None:
        return embeddings
    return predictor.predict(embeddings).astype(np.float32)


def correct_word_ids_for_query(
    row: pd.Series,
    target_index: TargetIndex,
) -> list[int]:
    definition_key = row[CORRECT_KEY_COLUMN]
    correct_words = set(target_index.definition_to_words.get(definition_key, set()))
    correct_words.add(row["word_norm"])
    return sorted(
        target_index.word_to_id[word]
        for word in correct_words
        if word in target_index.word_to_id
    )


def tie_averaged_rank(word_scores: np.ndarray, correct_word_ids: list[int]) -> float:
    if not correct_word_ids:
        return float(len(word_scores) + 1)

    best_rank = math.inf
    for word_id in correct_word_ids:
        score = word_scores[word_id]
        higher_count = int(np.sum(word_scores > score))
        tied_count = int(np.sum(word_scores == score))
        average_rank = higher_count + (tied_count + 1) / 2.0
        best_rank = min(best_rank, average_rank)
    return float(best_rank)


def tie_aware_recall_at_k(
    word_scores: np.ndarray,
    correct_word_ids: list[int],
    k: int,
) -> float:
    if not correct_word_ids:
        return 0.0

    recall_credit = 0.0
    for word_id in correct_word_ids:
        score = word_scores[word_id]
        higher_count = int(np.sum(word_scores > score))
        tied_count = int(np.sum(word_scores == score))
        slots_left = k - higher_count

        if slots_left <= 0:
            credit = 0.0
        elif slots_left >= tied_count:
            credit = 1.0
        else:
            credit = slots_left / tied_count

        recall_credit += credit

    return float(recall_credit / len(correct_word_ids))


def top_predictions(
    word_scores: np.ndarray,
    candidate_words: list[str],
    top_k: int,
) -> list[str]:
    top_indices = np.argsort(-word_scores)[:top_k]
    return [candidate_words[index] for index in top_indices]


def evaluate_queries(
    eval_df: pd.DataFrame,
    query_embeddings: np.ndarray,
    index_embeddings: np.ndarray,
    target_index: TargetIndex,
    batch_size: int,
    top_k_for_samples: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_embeddings = normalize(index_embeddings, norm="l2", axis=1)
    query_embeddings = normalize(query_embeddings, norm="l2", axis=1)

    rows = []
    sample_rows = []
    candidate_pool_size = len(target_index.candidate_words)

    for start in range(0, len(eval_df), batch_size):
        batch_df = eval_df.iloc[start : start + batch_size]
        batch_embeddings = query_embeddings[start : start + len(batch_df)]
        definition_scores = batch_embeddings @ index_embeddings.T

        for local_index, (_, row) in enumerate(batch_df.iterrows()):
            word_scores = np.full(candidate_pool_size, -np.inf, dtype=np.float32)
            np.maximum.at(
                word_scores,
                target_index.candidate_word_ids,
                definition_scores[local_index],
            )

            correct_ids = correct_word_ids_for_query(row, target_index)
            rank = tie_averaged_rank(word_scores, correct_ids)
            recall_at_1 = tie_aware_recall_at_k(word_scores, correct_ids, 1)
            recall_at_10 = tie_aware_recall_at_k(word_scores, correct_ids, 10)
            recall_at_100 = tie_aware_recall_at_k(word_scores, correct_ids, 100)
            predictions = top_predictions(
                word_scores,
                target_index.candidate_words,
                top_k_for_samples,
            )

            result_row = {
                "entry_id": row["entry_id"],
                "word_original": row["word_original"],
                "word_norm": row["word_norm"],
                "definition_original": row[TEXT_COLUMN],
                "headword_in_own_definition": bool(row["headword_in_own_definition"]),
                "candidate_pool_size": candidate_pool_size,
                "num_correct_words_in_pool": len(correct_ids),
                "tie_averaged_rank": rank,
                "recall_at_1": recall_at_1,
                "recall_at_10": recall_at_10,
                "recall_at_100": recall_at_100,
            }
            rows.append(result_row)

            if len(sample_rows) < 50:
                sample_rows.append(
                    {
                        **result_row,
                        "top_predictions": " | ".join(predictions),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def summarize_metrics(
    per_query_df: pd.DataFrame,
    run_name: str,
    split_mode: str,
    eval_split: str,
    encoder: str,
    model_name: str,
    predictor: str,
) -> pd.DataFrame:
    metric_rows = []
    groups = {
        "all_queries": per_query_df,
        "no_headword_in_definition": per_query_df[
            ~per_query_df["headword_in_own_definition"]
        ],
    }

    for group_name, group_df in groups.items():
        if group_df.empty:
            continue
        metric_rows.append(
            {
                "run_name": run_name,
                "split_mode": split_mode,
                "eval_split": eval_split,
                "query_group": group_name,
                "encoder": encoder,
                "model_name": model_name,
                "predictor": predictor,
                "metric_family": "tie_aware_recall_at_k",
                "num_queries": len(group_df),
                "queries_with_candidate_answer": int(
                    (group_df["num_correct_words_in_pool"] > 0).sum()
                ),
                "candidate_pool_size": int(group_df["candidate_pool_size"].iloc[0]),
                "recall_at_1": float(group_df["recall_at_1"].mean()),
                "recall_at_10": float(group_df["recall_at_10"].mean()),
                "recall_at_100": float(group_df["recall_at_100"].mean()),
                "median_rank": float(group_df["tie_averaged_rank"].median()),
            }
        )
    return pd.DataFrame(metric_rows)


def save_outputs(
    args: argparse.Namespace,
    metrics_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    split_summary: pd.DataFrame,
    predictor=None,
) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name

    metrics_df.to_csv(results_dir / f"{run_name}_metrics.csv", index=False)
    per_query_df.to_csv(results_dir / f"{run_name}_per_query.csv", index=False)
    sample_df.to_csv(results_dir / f"{run_name}_sample_predictions.csv", index=False)
    split_summary.to_csv(results_dir / f"{run_name}_split_summary.csv", index=False)

    config = vars(args).copy()
    predictor_path = None
    if predictor is not None and hasattr(predictor, "save"):
        predictor_path = results_dir / f"{run_name}_predictor.pt"
        predictor.save(predictor_path, extra_config=config)

    config["predictor_artifact"] = str(predictor_path) if predictor_path else ""
    (results_dir / f"{run_name}_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def print_plan(args: argparse.Namespace, split_data: SplitData) -> None:
    eval_df = split_data.test_df if args.eval_split == "test" else split_data.valid_df
    train_words = set(split_data.train_df["word_norm"])
    eval_words = set(eval_df["word_norm"])
    augmented_rows_used = 0

    if args.use_augmented_train:
        augmented_path = Path(args.processed_dir) / "opted_train_augmented_basic.csv"
        augmented_df = read_processed_csv(augmented_path)
        train_entry_ids = set(split_data.train_df["entry_id"])
        augmented_rows_used = len(
            augmented_df[
                augmented_df["augmentation_type"].ne("original")
                & augmented_df["source_entry_id"].isin(train_entry_ids)
            ]
        )

    print("Shared pipeline plan")
    print("--------------------")
    print(f"Split mode: {args.split_mode}")
    print(f"Evaluation split: {args.eval_split}")
    print(f"Encoder: {args.encoder}")
    print(f"Predictor: {args.predictor}")
    print()
    print(split_data.split_summary.to_string(index=False))
    print()
    print(f"Candidate words from train definitions: {len(train_words):,}")
    print(f"Extra augmented rows for predictor training: {augmented_rows_used:,}")
    print(f"Evaluation rows: {len(eval_df):,}")
    print(f"Evaluation words missing from train index: {len(eval_words - train_words):,}")
    print(
        "Queries with headword printed in own definition: "
        f"{int(eval_df['headword_in_own_definition'].sum()):,}"
    )


def run_pipeline(args: argparse.Namespace) -> None:
    split_data = load_split(args)

    if args.plan_only:
        print_plan(args, split_data)
        return

    train_df = maybe_limit_rows(split_data.train_df, args.max_train_rows, args.random_seed)
    eval_df = split_data.test_df if args.eval_split == "test" else split_data.valid_df
    eval_df = maybe_limit_rows(eval_df, args.max_eval_rows, args.random_seed)
    model_train_df = train_df

    if args.use_augmented_train:
        model_train_df = add_augmented_training_rows(
            train_df=train_df,
            processed_dir=Path(args.processed_dir),
        )
        model_train_df = maybe_limit_rows(
            model_train_df,
            args.max_train_rows,
            args.random_seed,
        )

    all_rows_for_answers = pd.concat(
        [split_data.train_df, split_data.valid_df, split_data.test_df],
        ignore_index=True,
    )
    target_index = build_target_index(train_df, all_rows_for_answers)

    encoder = make_encoder(
        encoder_name=args.encoder,
        model_name=args.model_name,
        device=args.device,
    )

    eval_texts = eval_df[TEXT_COLUMN].tolist()
    index_texts = target_index.candidate_df[TEXT_COLUMN].tolist()

    print(f"Encoding candidate index definitions: {len(index_texts):,}")
    index_embeddings = encoder.encode(index_texts, args.encode_batch_size)

    if args.predictor == "none":
        predictor = None
        print("Training predictor: none")
    else:
        train_texts = model_train_df[TEXT_COLUMN].tolist()
        print(f"Encoding predictor-training rows: {len(train_texts):,}")
        train_embeddings = encoder.encode(train_texts, args.encode_batch_size)

        print(f"Training predictor: {args.predictor}")
        predictor = train_predictor(
            predictor_name=args.predictor,
            train_embeddings=train_embeddings,
            train_words=model_train_df["word_norm"],
            index_embeddings=index_embeddings,
            index_word_ids=target_index.candidate_word_ids,
            word_to_id=target_index.word_to_id,
            args=args,
        )

    print(f"Encoding evaluation rows: {len(eval_texts):,}")
    eval_embeddings = encoder.encode(eval_texts, args.encode_batch_size)
    projected_eval_embeddings = apply_predictor(predictor, eval_embeddings)

    per_query_df, sample_df = evaluate_queries(
        eval_df=eval_df,
        query_embeddings=projected_eval_embeddings,
        index_embeddings=index_embeddings,
        target_index=target_index,
        batch_size=args.score_batch_size,
    )

    metrics_df = summarize_metrics(
        per_query_df=per_query_df,
        run_name=args.run_name,
        split_mode=args.split_mode,
        eval_split=args.eval_split,
        encoder=args.encoder,
        model_name=args.model_name,
        predictor=args.predictor,
    )

    save_outputs(
        args=args,
        metrics_df=metrics_df,
        per_query_df=per_query_df,
        sample_df=sample_df,
        split_summary=split_data.split_summary,
        predictor=predictor,
    )

    print(metrics_df.to_string(index=False))
    print(f"Saved outputs under: {args.results_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared reverse-dictionary pipeline for OPTED."
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results/shared_pipeline")
    parser.add_argument("--run-name", default="shared_pipeline_run")
    parser.add_argument(
        "--split-mode",
        choices=["by-definition", "by-headword"],
        default="by-definition",
        help=(
            "by-definition is the primary evaluation: unseen definition, seen word. "
            "by-headword reuses opted_train/valid/test."
        ),
    )
    parser.add_argument("--eval-split", choices=["valid", "test"], default="test")
    parser.add_argument(
        "--encoder",
        choices=ENCODER_CHOICES,
        default="sentence-transformer",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "SentenceTransformer or HuggingFace model name. If omitted, the "
            "default model for the selected encoder is used."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--predictor",
        choices=["ridge", "torch-mlp-infonce", "none"],
        default="none",
        help=(
            "none is retrieval-only baseline; torch-mlp-infonce is the shared "
            "trained predictor for fair encoder comparisons; ridge is a quick "
            "linear diagnostic."
        ),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--projection-epochs", type=int, default=5)
    parser.add_argument("--projection-batch-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--use-augmented-train",
        action="store_true",
        help=(
            "Use opted_train_augmented_basic.csv as extra predictor-training data. "
            "Leave this off for the controlled comparison."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional smoke-test cap. Use 0 for full train rows.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=0,
        help="Optional smoke-test cap. Use 0 for full evaluation rows.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Load and summarize data only. No encoder or model package is needed.",
    )
    args = parser.parse_args(argv)
    if args.model_name is None:
        args.model_name = DEFAULT_MODEL_BY_ENCODER[args.encoder]
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())

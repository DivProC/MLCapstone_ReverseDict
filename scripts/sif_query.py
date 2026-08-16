from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


TOKEN_PATTERN = re.compile(r"\b[a-z]+(?:'[a-z]+)?\b")
CROSS_REFERENCE_PATTERN = re.compile(
    r"^\s*(see|same as|alt\.?\s+of|alternative form of|another form of)\b",
    flags=re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def clean_query(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_sif_index(
    data_path: Path,
    subset_frac: float,
    random_seed: int,
    embedding_dim: int,
    max_features: int,
    sif_a: float,
) -> dict:
    df = pd.read_csv(data_path, dtype=str, keep_default_na=False)

    subset_df = (
        df.sample(frac=subset_frac, random_state=random_seed)
        .reset_index(drop=True)
        .copy()
    )

    subset_df["is_cross_reference_definition"] = subset_df[
        "definition_original"
    ].map(lambda text: bool(CROSS_REFERENCE_PATTERN.search(str(text))))

    cross_reference_rows_removed = int(
        subset_df["is_cross_reference_definition"].sum()
    )

    subset_df = (
        subset_df.loc[~subset_df["is_cross_reference_definition"]]
        .drop(columns=["is_cross_reference_definition"])
        .reset_index(drop=True)
        .copy()
    )

    subset_df["definition_tokens"] = subset_df["definition_basic_clean"].map(tokenize)
    subset_df["target_tokens"] = subset_df["word_norm"].map(tokenize)

    embedding_corpus = pd.concat(
        [subset_df["definition_basic_clean"], subset_df["word_norm"]],
        ignore_index=True,
    )

    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        min_df=1,
    )

    term_matrix = vectorizer.fit_transform(embedding_corpus)
    actual_dim = min(embedding_dim, term_matrix.shape[1] - 1)

    svd = TruncatedSVD(n_components=actual_dim, random_state=random_seed)
    svd.fit(term_matrix)

    vocab = vectorizer.vocabulary_
    term_embeddings = (svd.components_.T * svd.singular_values_).astype(np.float32)

    definition_token_counts = Counter(
        token for tokens in subset_df["definition_tokens"] for token in tokens
    )
    total_definition_tokens = sum(definition_token_counts.values())

    def sif_weight(token: str) -> float:
        frequency = definition_token_counts.get(token, 1)
        probability = frequency / max(total_definition_tokens, 1)
        return sif_a / (sif_a + probability)

    def encode_tokens(tokens: list[str]) -> np.ndarray:
        weighted_vectors = []

        for token in tokens:
            token_index = vocab.get(token)
            if token_index is None:
                continue

            weighted_vectors.append(term_embeddings[token_index] * sif_weight(token))

        if not weighted_vectors:
            return np.zeros(actual_dim, dtype=np.float32)

        return np.mean(weighted_vectors, axis=0).astype(np.float32)

    candidate_words = (
        subset_df[["word_original", "word_norm", "definition_original", "target_tokens"]]
        .drop_duplicates(subset=["word_norm"])
        .reset_index(drop=True)
    )

    definition_embeddings = np.vstack(
        subset_df["definition_tokens"].map(encode_tokens).to_numpy()
    )
    candidate_embeddings = np.vstack(
        candidate_words["target_tokens"].map(encode_tokens).to_numpy()
    )
    nonzero_definition_mask = np.linalg.norm(definition_embeddings, axis=1) > 0
    nonzero_candidate_mask = np.linalg.norm(candidate_embeddings, axis=1) > 0

    candidate_words = candidate_words.loc[nonzero_candidate_mask].reset_index(drop=True)
    candidate_embeddings = candidate_embeddings[nonzero_candidate_mask]

    if len(candidate_embeddings) == 0:
        raise ValueError("No usable candidate word embeddings were created.")

    pc_training_matrix = np.vstack(
        [
            definition_embeddings[nonzero_definition_mask],
            candidate_embeddings,
        ]
    )

    pc_svd = TruncatedSVD(n_components=1, random_state=random_seed)
    pc_svd.fit(pc_training_matrix)
    common_direction = pc_svd.components_[0]

    def remove_common_direction(matrix: np.ndarray) -> np.ndarray:
        return matrix - matrix.dot(common_direction.reshape(-1, 1)) * common_direction

    candidate_embeddings = remove_common_direction(candidate_embeddings).astype(np.float32)
    candidate_embeddings_norm = l2_normalize(candidate_embeddings)

    return {
        "subset_rows": len(subset_df),
        "cross_reference_rows_removed": cross_reference_rows_removed,
        "candidate_words": candidate_words,
        "candidate_embeddings_norm": candidate_embeddings_norm,
        "encode_tokens": encode_tokens,
        "remove_common_direction": remove_common_direction,
    }


def query_sif(index: dict, query: str, top_k: int) -> pd.DataFrame:
    query_tokens = tokenize(clean_query(query))
    query_embedding = index["encode_tokens"](query_tokens).reshape(1, -1)

    if np.linalg.norm(query_embedding) == 0:
        raise ValueError(
            "The query has no usable words in the SIF vocabulary. Try a longer definition."
        )

    query_embedding = index["remove_common_direction"](query_embedding).astype(np.float32)
    query_embedding_norm = l2_normalize(query_embedding)

    scores = query_embedding_norm @ index["candidate_embeddings_norm"].T
    scores = scores.ravel()

    top_indices = np.argsort(-scores)[:top_k]
    candidate_words = index["candidate_words"]

    results = candidate_words.iloc[top_indices][
        ["word_original", "word_norm", "definition_original"]
    ].copy()
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results["score"] = scores[top_indices]

    return results


def get_target_rank(index: dict, query: str, target: str) -> tuple[int | None, float | None]:
    query_tokens = tokenize(clean_query(query))
    query_embedding = index["encode_tokens"](query_tokens).reshape(1, -1)

    if np.linalg.norm(query_embedding) == 0:
        return None, None

    query_embedding = index["remove_common_direction"](query_embedding).astype(np.float32)
    query_embedding_norm = l2_normalize(query_embedding)
    scores = (query_embedding_norm @ index["candidate_embeddings_norm"].T).ravel()

    target_norm = clean_query(target)
    candidate_words = index["candidate_words"]
    matches = np.where(candidate_words["word_norm"].to_numpy() == target_norm)[0]

    if len(matches) == 0:
        return None, None

    target_index = int(matches[0])
    target_score = float(scores[target_index])
    target_rank = int(np.sum(scores > target_score) + 1)
    return target_rank, target_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual queries against the Stage 1 SIF baseline."
    )
    parser.add_argument("--query", required=True, help="Definition-style query text.")
    parser.add_argument("--target", help="Optional expected word, used to print rank.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to show.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/opted_train.csv"),
        help="Path to opted_train.csv.",
    )
    parser.add_argument("--subset-frac", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--sif-a", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    print("Building SIF index...")
    index = build_sif_index(
        data_path=args.data_path,
        subset_frac=args.subset_frac,
        random_seed=args.random_seed,
        embedding_dim=args.embedding_dim,
        max_features=args.max_features,
        sif_a=args.sif_a,
    )

    print(f"Rows used: {index['subset_rows']:,}")
    print(f"Cross-reference rows removed: {index['cross_reference_rows_removed']:,}")
    print(f"Candidate words: {len(index['candidate_words']):,}")
    print()

    results = query_sif(index, args.query, args.top_k)
    print("Top predictions:")
    print(
        results[["rank", "word_original", "score", "definition_original"]].to_string(
            index=False,
            max_colwidth=80,
        )
    )

    if args.target:
        target_rank, target_score = get_target_rank(index, args.query, args.target)
        print()
        if target_rank is None:
            print(f"Target '{args.target}' was not found in the candidate pool.")
        else:
            print(
                f"Target '{args.target}' rank: {target_rank:,} "
                f"(score={target_score:.4f})"
            )


if __name__ == "__main__":
    main()

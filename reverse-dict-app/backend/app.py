"""
Reverse Dictionary backend API.

Serves a small REST API on localhost. The same FastAPI service also serves the
static demo page, so Java/Gradle is not needed for the local demo.

  - "sentence-transformer" -> plain SBERT retrieval
  - "sbert-infonce"        -> SBERT plus saved InfoNCE predictor
  - "bert-cls"             -> bert_cls.py (needs `torch` + `transformers`)
  - "bilstm"               -> trained BiLSTM checkpoint
  - "bilstm_attn"          -> trained BiLSTM + attention checkpoint
  - "defgen2"              -> DefGen2 checkpoint, trained on MultiRD English data
  - "unified"              -> unifiedRevdicDefmod checkpoint, trained on MultiRD English data
  - "multird"              -> MultiRD checkpoint, trained on MultiRD English data

Run:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /api/encoders            -> which encoders are available right now
    POST /api/query                {"text": "...", "encoder": "sbert-infonce", "top_k": 10}
    GET  /api/health
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

REPO_DIR = Path(
    os.environ.get("REVDICT_PROJECT_DIR", Path(__file__).resolve().parents[2])
).resolve()
DATA_DIR = Path(
    os.environ.get("REVDICT_DATA_DIR", REPO_DIR / "data" / "processed")
).resolve()
FRONTEND_STATIC_DIR = (
    REPO_DIR / "reverse-dict-app" / "frontend-gradle" / "src" / "main" / "resources" / "static"
)

for import_path in [REPO_DIR / "scripts", REPO_DIR / "scripts" / "encoders"]:
    sys.path.insert(0, str(import_path))

import shared_reverse_dictionary_pipeline as shared_pipeline  # noqa: E402

DEFAULT_TOP_K = 10
MAX_TOP_K = 500
ENCODE_BATCH_SIZE = int(os.environ.get("REVDICT_ENCODE_BATCH_SIZE", "64"))
BILSTM_MAX_LEN = int(os.environ.get("REVDICT_BILSTM_MAX_LEN", "64"))
PREDICTOR_PATH_BY_ENCODER = {
    "sbert-infonce": Path(
        os.environ.get(
            "REVDICT_SBERT_PREDICTOR_PATH",
            REPO_DIR
            / "results"
            / "shared_pipeline"
            / "sbert_mlp_infonce_test_predictor.pt",
        )
    ).resolve(),
    "bert-cls": Path(
        os.environ.get(
            "REVDICT_BERT_CLS_PREDICTOR_PATH",
            REPO_DIR
            / "results"
            / "shared_pipeline"
            / "bert_cls_mlp_infonce_test_predictor.pt",
        )
    ).resolve(),
}
BILSTM_VOCAB_PATH = Path(
    os.environ.get("REVDICT_BILSTM_VOCAB_PATH", REPO_DIR / "data" / "processed" / "vocab.pkl")
).resolve()
BILSTM_CHECKPOINT_BY_ENCODER = {
    "bilstm": Path(
        os.environ.get(
            "REVDICT_BILSTM_CHECKPOINT",
            REPO_DIR / "results" / "bilstm" / "bilstm_best.pt",
        )
    ).resolve(),
    "bilstm_attn": Path(
        os.environ.get(
            "REVDICT_BILSTM_ATTN_CHECKPOINT",
            REPO_DIR / "results" / "bilstm_attn" / "bilstm_attn_best.pt",
        )
    ).resolve(),
}

# DefGen2 / unifiedRevdicDefmod -- trained on MultiRD's English data (Capstone/kaggle_scripts/),
# not OPTED, so they carry their own candidate index instead of sharing the OPTED target_index.
MULTIRD_SPACE_DATA_DIR = Path(
    os.environ.get(
        "REVDICT_MULTIRD_DATA_DIR",
        REPO_DIR.parent / "MultiRD_data" / "English" / "data",
    )
).resolve()
MULTIRD_SPACE_CHECKPOINT_BY_ENCODER = {
    "defgen2": Path(
        os.environ.get(
            "REVDICT_DEFGEN2_CHECKPOINT",
            REPO_DIR.parent / "saved_models" / "DefGen2" / "model_v1.pt",
        )
    ).resolve(),
    "unified": Path(
        os.environ.get(
            "REVDICT_UNIFIED_CHECKPOINT",
            REPO_DIR.parent / "saved_models" / "unifiedRevdicDefmod" / "model_v1.pt",
        )
    ).resolve(),
    "multird": Path(
        os.environ.get(
            "REVDICT_MULTIRD_CHECKPOINT",
            REPO_DIR.parent / "saved_models" / "MultiRD" / "model_v1.pt",
        )
    ).resolve(),
}
MULTIRD_SPACE_CONFIG_BY_ENCODER = {
    "defgen2": Path(
        os.environ.get(
            "REVDICT_DEFGEN2_CONFIG",
            REPO_DIR.parent / "saved_models" / "DefGen2" / "config_v1.json",
        )
    ).resolve(),
    "unified": Path(
        os.environ.get(
            "REVDICT_UNIFIED_CONFIG",
            REPO_DIR.parent / "saved_models" / "unifiedRevdicDefmod" / "config_v1.json",
        )
    ).resolve(),
    "multird": Path(
        os.environ.get(
            "REVDICT_MULTIRD_CONFIG",
            REPO_DIR.parent / "saved_models" / "MultiRD" / "config_v1.json",
        )
    ).resolve(),
}


def _pick_data_path() -> Path:
    candidates = [
        DATA_DIR / "opted_train.csv",
        DATA_DIR / "opted_valid.csv",
        DATA_DIR / "opted_test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No processed OPTED split (opted_train/valid/test.csv) found under "
        f"{DATA_DIR}. Set REVDICT_DATA_DIR to the folder containing them."
    )


DATA_PATH = _pick_data_path()
_split_data = None
_target_index = None


def _load_shared_index():
    global _split_data, _target_index
    if _split_data is not None and _target_index is not None:
        return _split_data, _target_index

    split_args = argparse.Namespace(
        processed_dir=str(DATA_DIR),
        split_mode="by-definition",
        random_seed=shared_pipeline.RANDOM_SEED,
    )
    split_data = shared_pipeline.load_split(split_args)
    all_rows_for_answers = pd.concat(
        [split_data.train_df, split_data.valid_df, split_data.test_df],
        ignore_index=True,
    )
    target_index = shared_pipeline.build_target_index(
        split_data.train_df,
        all_rows_for_answers,
    )
    _split_data = split_data
    _target_index = target_index
    return split_data, target_index

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Reverse Dictionary API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Definition-style query text")
    encoder: str = Field(
        "sbert-infonce",
        description=(
            "sentence-transformer | sbert-infonce | bert-cls | "
            "bilstm | bilstm_attn | defgen2 | unified | multird"
        ),
    )
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class ResultRow(BaseModel):
    rank: int
    word: str
    definition: str
    score: float


class QueryResponse(BaseModel):
    encoder: str
    predictor: str
    top_k: int
    took_ms: float
    results: list[ResultRow]


# Encoders are loaded lazily on first use. The candidate index is the shared
# pipeline train-definition index, so the demo follows the same retrieval setup.

_neural_state: dict[str, dict] = {}
_multird_space_state: dict[str, dict] = {}


class BilstmDemoEncoder:
    def __init__(self, encoder_name: str, device: str | None = None):
        try:
            import torch
            from torch.nn.utils.rnn import pad_sequence
            from bilstm_vocab import Vocabulary
            from bilstm_encoder import BiLSTMEncoder, PredictorMLP as PlainPredictor
            from bilstm_attn_encoder import (
                BiLSTMAttnEncoder,
                PredictorMLP as AttentionPredictor,
            )
        except ImportError as exc:
            raise ImportError(
                "torch and the BiLSTM project modules are needed for this encoder."
            ) from exc

        self.torch = torch
        self.pad_sequence = pad_sequence
        self.vocab = Vocabulary.load(BILSTM_VOCAB_PATH)
        self.max_len = BILSTM_MAX_LEN

        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        checkpoint_path = BILSTM_CHECKPOINT_BY_ENCODER[encoder_name]
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        cfg = checkpoint["config"]
        if encoder_name == "bilstm_attn":
            encoder_cls = BiLSTMAttnEncoder
            predictor_cls = AttentionPredictor
            encoder_kwargs = {
                "attn_dim": cfg.get("attn_dim", 128),
                "num_heads": cfg.get("num_heads", 4),
            }
        else:
            encoder_cls = BiLSTMEncoder
            predictor_cls = PlainPredictor
            encoder_kwargs = {}

        self.encoder = encoder_cls(
            len(self.vocab.token2idx),
            cfg["embed_dim"],
            cfg["hidden_dim"],
            cfg["output_dim"],
            cfg["num_layers"],
            dropout=0.0,
            **encoder_kwargs,
        ).to(self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.encoder.eval()

        self.predictor = predictor_cls(
            cfg["output_dim"],
            cfg["output_dim"] * 2,
            cfg["output_dim"],
        ).to(self.device)
        self.predictor.load_state_dict(checkpoint["predictor_state"])
        self.predictor.eval()

    def encode_index(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, batch_size, use_predictor=False)

    def encode_query(self, texts: list[str], batch_size: int) -> np.ndarray:
        wrapped_texts = [f"<START> {text} <END>" for text in texts]
        return self._encode(wrapped_texts, batch_size, use_predictor=True)

    def _encode(
        self,
        texts: list[str],
        batch_size: int,
        use_predictor: bool,
    ) -> np.ndarray:
        torch = self.torch
        outputs: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                seqs = []
                for text in batch_texts:
                    token_ids = self.vocab.encode_definition(str(text))[: self.max_len]
                    seqs.append(torch.tensor(token_ids if token_ids else [1], dtype=torch.long))

                lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
                padded = self.pad_sequence(seqs, batch_first=True, padding_value=0)
                padded = padded.to(self.device)
                lengths = lengths.to(self.device)

                embeddings = self.encoder(padded, lengths)
                if use_predictor:
                    embeddings = self.predictor(embeddings)

                outputs.append(embeddings.cpu().numpy().astype(np.float32))

        return np.vstack(outputs)


def _predictor_path_for_encoder(encoder_name: str) -> Path | None:
    return PREDICTOR_PATH_BY_ENCODER.get(encoder_name)


def _load_predictor_for_encoder(encoder_name: str, required: bool = False):
    predictor_path = _predictor_path_for_encoder(encoder_name)
    if predictor_path is None:
        if required:
            raise HTTPException(
                status_code=503,
                detail=f"No predictor path is configured for {encoder_name}.",
            )
        return None

    if not predictor_path.exists():
        if required:
            raise HTTPException(
                status_code=503,
                detail=f"Missing predictor checkpoint: {predictor_path}",
            )
        return None

    try:
        predictor = shared_pipeline.TorchMlpInfoNcePredictor.load(
            predictor_path,
            device=os.environ.get("REVDICT_DEVICE") or None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load predictor checkpoint: {predictor_path}",
        ) from exc

    print(f"Loaded predictor checkpoint for {encoder_name}: {predictor_path}")
    return predictor


def _load_shared_encoder_state(encoder_name: str):
    if encoder_name in _neural_state:
        state = _neural_state[encoder_name]
        if encoder_name == "sbert-infonce" and state["predictor"] is None:
            predictor = _load_predictor_for_encoder(encoder_name, required=True)
            if predictor is not None:
                state["predictor"] = predictor
                state["predictor_name"] = "mlp-infonce"
        return state

    base_encoder_name = (
        "sentence-transformer" if encoder_name == "sbert-infonce" else encoder_name
    )

    if base_encoder_name == "sentence-transformer":
        model_name = os.environ.get(
            "REVDICT_ST_MODEL",
            shared_pipeline.DEFAULT_MODEL_BY_ENCODER["sentence-transformer"],
        )
    elif base_encoder_name == "bert-cls":
        model_name = os.environ.get(
            "REVDICT_BERT_MODEL",
            shared_pipeline.DEFAULT_MODEL_BY_ENCODER["bert-cls"],
        )
    elif base_encoder_name in ("bilstm", "bilstm_attn"):
        model_name = encoder_name
    else:
        raise HTTPException(status_code=400, detail=f"Unknown encoder: {encoder_name}")

    device = os.environ.get("REVDICT_DEVICE") or None
    if base_encoder_name in ("bilstm", "bilstm_attn"):
        if not BILSTM_VOCAB_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Missing BiLSTM vocabulary: {BILSTM_VOCAB_PATH}",
            )
        checkpoint_path = BILSTM_CHECKPOINT_BY_ENCODER[encoder_name]
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Missing BiLSTM checkpoint: {checkpoint_path}",
            )
        try:
            encoder = BilstmDemoEncoder(encoder_name=encoder_name, device=device)
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    else:
        try:
            encoder = shared_pipeline.make_encoder(
                encoder_name=base_encoder_name,
                model_name=model_name,
                device=device,
            )
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    _, target_index = _load_shared_index()
    index_text_column = (
        "definition_model_input"
        if base_encoder_name in ("bilstm", "bilstm_attn")
        and "definition_model_input" in target_index.candidate_df.columns
        else shared_pipeline.TEXT_COLUMN
    )
    index_texts = target_index.candidate_df[index_text_column].tolist()
    print(
        f"Encoding {len(index_texts):,} shared candidate definitions "
        f"with {encoder_name} ..."
    )
    if base_encoder_name in ("bilstm", "bilstm_attn"):
        candidate_embeddings = encoder.encode_index(index_texts, ENCODE_BATCH_SIZE)
    else:
        candidate_embeddings = encoder.encode(index_texts, ENCODE_BATCH_SIZE)

    candidate_embeddings_norm = shared_pipeline.normalize(
        candidate_embeddings,
        norm="l2",
        axis=1,
    ).astype(np.float32)
    predictor = (
        None
        if base_encoder_name in ("bilstm", "bilstm_attn", "sentence-transformer")
        and encoder_name != "sbert-infonce"
        else _load_predictor_for_encoder(
            encoder_name,
            required=encoder_name == "sbert-infonce",
        )
    )

    state = {
        "encoder": encoder,
        "model_name": model_name,
        "predictor": predictor,
        "predictor_name": (
            "checkpoint-mlp"
            if base_encoder_name in ("bilstm", "bilstm_attn")
            else "mlp-infonce"
            if predictor is not None
            else "none"
        ),
        "candidate_df": target_index.candidate_df,
        "candidate_embeddings_norm": candidate_embeddings_norm,
        "candidate_word_ids": target_index.candidate_word_ids,
        "candidate_words": target_index.candidate_words,
    }
    _neural_state[encoder_name] = state
    return state


def _query_shared_pipeline(
    encoder_name: str,
    text: str,
    top_k: int,
) -> tuple[pd.DataFrame, str]:
    state = _load_shared_encoder_state(encoder_name)
    if encoder_name in ("bilstm", "bilstm_attn"):
        query_embedding = state["encoder"].encode_query([text], batch_size=1)
    else:
        query_embedding = state["encoder"].encode([text], batch_size=1)

    if state["predictor"] is not None:
        query_embedding = state["predictor"].predict(query_embedding)

    query_embedding_norm = shared_pipeline.normalize(
        query_embedding,
        norm="l2",
        axis=1,
    ).astype(np.float32)

    definition_scores = (query_embedding_norm @ state["candidate_embeddings_norm"].T).ravel()
    word_scores = np.full(len(state["candidate_words"]), -np.inf, dtype=np.float32)
    np.maximum.at(word_scores, state["candidate_word_ids"], definition_scores)

    top_word_ids = np.argsort(-word_scores)[:top_k]
    rows = []
    for rank, word_id in enumerate(top_word_ids, start=1):
        candidate_indices = np.where(state["candidate_word_ids"] == word_id)[0]
        best_candidate_index = candidate_indices[
            np.argmax(definition_scores[candidate_indices])
        ]
        candidate_row = state["candidate_df"].iloc[int(best_candidate_index)]
        rows.append(
            {
                "rank": rank,
                "word_original": candidate_row["word_original"],
                "definition_original": candidate_row["definition_original"],
                "score": float(word_scores[word_id]),
            }
        )

    return pd.DataFrame(rows), state["predictor_name"]


def _load_multird_space_encoder_state(encoder_name: str) -> dict:
    if encoder_name in _multird_space_state:
        return _multird_space_state[encoder_name]

    checkpoint_path = MULTIRD_SPACE_CHECKPOINT_BY_ENCODER[encoder_name]
    config_path = MULTIRD_SPACE_CONFIG_BY_ENCODER[encoder_name]
    if not checkpoint_path.exists():
        raise HTTPException(status_code=503, detail=f"Missing checkpoint: {checkpoint_path}")
    if not MULTIRD_SPACE_DATA_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Missing MultiRD data directory: {MULTIRD_SPACE_DATA_DIR}",
        )

    try:
        from multird_space_encoder import MultiRDSpaceDemoEncoder
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    print(f"Loading {encoder_name} checkpoint: {checkpoint_path}")
    encoder = MultiRDSpaceDemoEncoder(
        encoder_name=encoder_name,
        data_dir=MULTIRD_SPACE_DATA_DIR,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=os.environ.get("REVDICT_DEVICE") or None,
    )
    state = {"encoder": encoder}
    _multird_space_state[encoder_name] = state
    return state


def _query_multird_space_pipeline(
    encoder_name: str,
    text: str,
    top_k: int,
) -> tuple[pd.DataFrame, str]:
    state = _load_multird_space_encoder_state(encoder_name)
    rows = state["encoder"].query(text, top_k)
    df = pd.DataFrame(rows).rename(
        columns={"word": "word_original", "definition": "definition_original"}
    )
    return df, "checkpoint-mlp"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    train_rows = len(_split_data.train_df) if _split_data is not None else None
    candidate_words = (
        len(_target_index.candidate_words) if _target_index is not None else None
    )
    return {
        "status": "ok",
        "repo_dir": str(REPO_DIR),
        "data_path": str(DATA_PATH),
        "split_mode": "by-definition",
        "shared_index_loaded": _target_index is not None,
        "train_rows": train_rows,
        "candidate_words": candidate_words,
        "default_encoder": "sbert-infonce",
        "sbert_predictor_path": str(PREDICTOR_PATH_BY_ENCODER["sbert-infonce"]),
        "sbert_predictor_available": PREDICTOR_PATH_BY_ENCODER["sbert-infonce"].exists(),
    }


@app.get("/", include_in_schema=False)
def frontend_index():
    return FileResponse(FRONTEND_STATIC_DIR / "index.html")


@app.get("/{asset_name}", include_in_schema=False)
def frontend_asset(asset_name: str):
    if asset_name not in {"app.js", "style.css"}:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(FRONTEND_STATIC_DIR / asset_name)


@app.get("/api/encoders")
def list_encoders():
    """Report which encoders are currently usable without triggering a
    heavy load, so the frontend dropdown can grey out unavailable ones."""

    def _installed(module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    torch_available = _installed("torch")
    bilstm_vocab_available = BILSTM_VOCAB_PATH.exists()

    return {
        "encoders": [
            {
                "id": "sentence-transformer",
                "label": "SBERT",
                "available": _installed("sentence_transformers"),
                "predictor": "none",
            },
            {
                "id": "sbert-infonce",
                "label": "SBERT + InfoNCE",
                "available": (
                    _installed("sentence_transformers")
                    and _installed("torch")
                    and PREDICTOR_PATH_BY_ENCODER["sbert-infonce"].exists()
                ),
                "predictor": "mlp-infonce",
            },
            {
                "id": "bert-cls",
                "label": (
                    "BERT [CLS] + InfoNCE"
                    if PREDICTOR_PATH_BY_ENCODER["bert-cls"].exists()
                    else "BERT [CLS]"
                ),
                "available": _installed("transformers") and _installed("torch"),
                "predictor": (
                    "mlp-infonce"
                    if PREDICTOR_PATH_BY_ENCODER["bert-cls"].exists()
                    else "none"
                ),
            },
            {
                "id": "bilstm",
                "label": "BiLSTM",
                "available": (
                    torch_available
                    and bilstm_vocab_available
                    and BILSTM_CHECKPOINT_BY_ENCODER["bilstm"].exists()
                ),
                "predictor": "checkpoint-mlp",
            },
            {
                "id": "bilstm_attn",
                "label": "BiLSTM + Attention",
                "available": (
                    torch_available
                    and bilstm_vocab_available
                    and BILSTM_CHECKPOINT_BY_ENCODER["bilstm_attn"].exists()
                ),
                "predictor": "checkpoint-mlp",
            },
            {
                "id": "defgen2",
                "label": "DefGen2 (MultiRD-trained)",
                "available": (
                    torch_available
                    and MULTIRD_SPACE_DATA_DIR.exists()
                    and MULTIRD_SPACE_CHECKPOINT_BY_ENCODER["defgen2"].exists()
                ),
                "predictor": "checkpoint-mlp",
            },
            {
                "id": "unified",
                "label": "unifiedRevdicDefmod (MultiRD-trained)",
                "available": (
                    torch_available
                    and MULTIRD_SPACE_DATA_DIR.exists()
                    and MULTIRD_SPACE_CHECKPOINT_BY_ENCODER["unified"].exists()
                ),
                "predictor": "checkpoint-mlp",
            },
            {
                "id": "multird",
                "label": "MultiRD (MultiRD-trained)",
                "available": (
                    torch_available
                    and MULTIRD_SPACE_DATA_DIR.exists()
                    and MULTIRD_SPACE_CHECKPOINT_BY_ENCODER["multird"].exists()
                ),
                "predictor": "checkpoint-mlp",
            },
        ]
    }


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    started = time.perf_counter()
    encoder_name = request.encoder.strip().lower()

    if encoder_name in (
        "sentence-transformer",
        "sbert-infonce",
        "bert-cls",
        "bilstm",
        "bilstm_attn",
    ):
        results_df, predictor_name = _query_shared_pipeline(
            encoder_name,
            request.text,
            request.top_k,
        )
        word_col = "word_original"
    elif encoder_name in ("defgen2", "unified", "multird"):
        results_df, predictor_name = _query_multird_space_pipeline(
            encoder_name,
            request.text,
            request.top_k,
        )
        word_col = "word_original"
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "encoder must be one of: sentence-transformer, sbert-infonce, "
                "bert-cls, bilstm, bilstm_attn, defgen2, unified, multird"
            ),
        )

    took_ms = (time.perf_counter() - started) * 1000
    results = [
        ResultRow(
            rank=int(row["rank"]),
            word=str(row[word_col]),
            definition=str(row["definition_original"]),
            score=float(row["score"]),
        )
        for _, row in results_df.iterrows()
    ]

    return QueryResponse(
        encoder=encoder_name,
        predictor=predictor_name,
        top_k=request.top_k,
        took_ms=round(took_ms, 1),
        results=results,
    )

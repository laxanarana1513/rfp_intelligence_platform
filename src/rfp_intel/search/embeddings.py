"""Local embedding model. The Google API key is not used here."""

from __future__ import annotations

import threading

from rfp_intel.config import get_settings

_model = None
_lock = threading.Lock()


def get_embedding_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(get_settings().embedding_model)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    # bge v1.5 expects an instruction prefix on queries only.
    prefixed = text
    model_name = get_settings().embedding_model.lower()
    if "bge" in model_name and not text.lower().startswith("represent this sentence"):
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
    vector = model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.tolist()

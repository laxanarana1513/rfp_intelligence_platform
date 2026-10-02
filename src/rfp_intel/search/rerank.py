"""Cross-encoder reranker over the fused candidate set."""

from __future__ import annotations

import threading

from rfp_intel.config import get_settings

_model = None
_lock = threading.Lock()


def get_reranker():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                _model = CrossEncoder(get_settings().reranker_model)
    return _model


def rerank(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    scores = get_reranker().predict([(query, text) for text in texts], show_progress_bar=False)
    return [float(score) for score in scores]

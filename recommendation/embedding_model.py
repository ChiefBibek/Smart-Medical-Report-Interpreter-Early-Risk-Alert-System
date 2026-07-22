"""
Lazy singleton loader for the Sentence-BERT embedding model.

Loaded once per process, behind a lock, on first use — not per request and
not per disease. `warmup()` (called from predictor.py::train_all()) forces
the load eagerly so the first real HTTP request isn't slow.
"""

import threading

from recommendation.config import MODEL_NAME

_lock = threading.Lock()
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def warmup():
    """Force the embedder to load now rather than on first request."""
    get_embedder()

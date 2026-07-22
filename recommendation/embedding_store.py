"""
Disk-cached embeddings for the knowledge base.

The KB is tiny (~24 short sentences), so a single cache pair covers all
diseases; filtering by disease happens at query time via a boolean mask in
retrieval_engine.py. Cache invalidation is keyed off a content hash of
(id, disease, clinical_scenario) — the only fields that feed the embedding —
so editing KB text or adding/removing an entry rebuilds automatically, while
re-ordering entries in the JSON files does not.
"""

import json
import os

import numpy as np

from knowledge_base.loader import content_hash
from recommendation.config import MODEL_NAME
from recommendation.embedding_model import get_embedder

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base", "cache")
_EMB_FILE = os.path.join(_CACHE_DIR, "kb_embeddings.npz")
_META_FILE = os.path.join(_CACHE_DIR, "kb_embeddings.meta.json")


class EmbeddingStore:
    def __init__(self, emb_file=_EMB_FILE, meta_file=_META_FILE):
        self.emb_file = emb_file
        self.meta_file = meta_file
        self._embeddings = None
        self._entry_ids = None

    def get_or_build(self, entries):
        """
        entries: list of KB entry dicts (all diseases).
        Returns (embeddings: float32[N,384] L2-normalized, entry_ids: list[str]),
        reindexed to match `entries`' current order.
        """
        h = content_hash(entries)
        if self._cache_valid(h):
            embeddings, entry_ids = self._load()
        else:
            embeddings, entry_ids = self._rebuild(entries, h)

        # Reindex the cached vectors to the caller's current entry order.
        id_to_row = {eid: i for i, eid in enumerate(entry_ids)}
        order = [id_to_row[e["id"]] for e in entries]
        self._embeddings = embeddings[order]
        self._entry_ids = [entries[i]["id"] for i in range(len(entries))]
        return self._embeddings, self._entry_ids

    def _cache_valid(self, content_hash_value):
        if not (os.path.exists(self.emb_file) and os.path.exists(self.meta_file)):
            return False
        try:
            with open(self.meta_file, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return False
        return meta.get("content_hash") == content_hash_value and meta.get("model_name") == MODEL_NAME

    def _load(self):
        data = np.load(self.emb_file)
        return data["embeddings"], list(data["entry_ids"])

    def _rebuild(self, entries, content_hash_value):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        embedder = get_embedder()
        texts = [e["clinical_scenario"] for e in entries]
        embeddings = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        entry_ids = [e["id"] for e in entries]

        # np.savez silently appends ".npz" to any filename that doesn't already end
        # with it, so the temp name must itself end in ".npz" or os.replace() below
        # would look for a file numpy never actually wrote.
        tmp_emb = self.emb_file[:-4] + ".tmp.npz" if self.emb_file.endswith(".npz") else self.emb_file + ".tmp.npz"
        tmp_meta = self.meta_file + ".tmp"
        np.savez(tmp_emb, embeddings=embeddings, entry_ids=np.array(entry_ids))
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump({
                "content_hash": content_hash_value, "model_name": MODEL_NAME,
                "embedding_dim": int(embeddings.shape[1]), "num_entries": len(entries),
            }, fh)
        os.replace(tmp_emb, self.emb_file)
        os.replace(tmp_meta, self.meta_file)
        return embeddings, entry_ids

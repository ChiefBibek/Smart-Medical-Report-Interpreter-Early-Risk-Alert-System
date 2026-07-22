"""
Knowledge Base loader.

Loads the structured clinical-scenario entries used by the recommendation
engine's semantic retrieval (recommendation/retrieval_engine.py). Entries are
plain JSON, one file per disease, under knowledge_base/entries/.
"""

import hashlib
import json
import os

_ENTRIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entries")

REQUIRED_FIELDS = (
    "id", "disease", "clinical_scenario", "recommendation", "diet_advice",
    "lifestyle_advice", "suggested_follow_up_tests", "urgency_level",
    "medical_keywords", "evidence_tags",
)

_cache = None  # process-wide list[dict], loaded once


def load_all_entries():
    """Return all KB entries across all diseases (list of dicts), loaded once per process."""
    global _cache
    if _cache is None:
        entries = []
        for fname in sorted(os.listdir(_ENTRIES_DIR)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(_ENTRIES_DIR, fname), encoding="utf-8") as fh:
                disease_entries = json.load(fh)
            for entry in disease_entries:
                missing = [f for f in REQUIRED_FIELDS if f not in entry]
                if missing:
                    raise ValueError(f"KB entry in {fname} missing required fields: {missing}")
                entries.append(entry)
        ids = [e["id"] for e in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate KB entry id detected across knowledge_base/entries/*.json")
        _cache = entries
    return _cache


def load_disease_entries(disease):
    """Return only the KB entries for one disease."""
    disease = disease.lower().strip()
    return [e for e in load_all_entries() if e["disease"] == disease]


def content_hash(entries=None):
    """
    Stable hash over (id, disease, clinical_scenario) only — the fields that
    actually feed the embedding — sorted by id so file re-ordering doesn't
    spuriously invalidate the embedding cache.
    """
    entries = entries if entries is not None else load_all_entries()
    canonical = sorted(
        ({"id": e["id"], "disease": e["disease"], "clinical_scenario": e["clinical_scenario"]}
         for e in entries),
        key=lambda e: e["id"],
    )
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()

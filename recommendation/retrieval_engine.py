"""
RecommendationEngine — Module 2 semantic retrieval, shared by Module 1's
"Suggested Additional Tests" (suggest_tests) and the post-prediction
personalized recommendation (retrieve_recommendation).

Cosine similarity (embeddings are L2-normalized at build time, so dot
product == cosine similarity) plus a sentence-scoped keyword boost. STRICT
top-1 for the actual clinical recommendation/diet/lifestyle/urgency text —
natural-language advice from two different KB entries can be directly
contradictory (that's the whole point of the anemia iron-deficiency vs.
further-workup pair), so there is no safe way to blend two sentences.
`suggested_follow_up_tests` is the one field that gets UNIONED across entries
within CLOSE_SCORE_MARGIN of the top score, since suggesting an extra test is
low-risk, unlike blending conflicting treatment advice.
"""

import numpy as np

from knowledge_base.loader import load_all_entries
from recommendation import config
from recommendation.embedding_model import get_embedder
from recommendation.embedding_store import EmbeddingStore
from recommendation.fallback import build_fallback


def _keyword_boost(entry_keywords, context):
    """
    +KEYWORD_BOOST per matching keyword (accumulated, capped) if ALL of that
    keyword's tokens co-occur within the SAME sentence/clause of the context
    (not just anywhere in the paragraph) — avoids a false hit like "low
    ferritin" firing because "low" appears in an unrelated clause (e.g. the
    hemoglobin sentence) and "ferritin" in another. Accumulating rather than
    stopping at the first match matters here: a small embedding model's raw
    cosine similarity can rank a lexically-overlapping but semantically wrong
    entry (e.g. "all normal" sharing every lab-test name with a severely
    abnormal panel) within a few hundredths of the correct one, so multiple
    specific keyword hits need to be able to compound rather than cap at one.
    """
    sentences = [s.lower() for s in context.split(".")]
    boost = 0.0
    for keyword in entry_keywords:
        tokens = keyword.lower().split()
        if any(all(tok in sentence for tok in tokens) for sentence in sentences):
            boost += config.KEYWORD_BOOST
    return min(boost, config.KEYWORD_BOOST_CAP)


class RecommendationEngine:
    def __init__(self):
        self.embedding_store = EmbeddingStore()

    def warmup(self):
        """Force the embedder + KB embedding cache to load/build now, not on first request."""
        get_embedder()
        self.embedding_store.get_or_build(load_all_entries())

    def _disease_scores(self, context, disease):
        all_entries = load_all_entries()
        embeddings, _ = self.embedding_store.get_or_build(all_entries)
        query_vec = get_embedder().encode([context], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = embeddings @ query_vec

        entries, disease_scores = [], []
        for entry, score in zip(all_entries, scores):
            if entry["disease"] != disease:
                continue
            boosted = min(1.0, float(score) + _keyword_boost(entry["medical_keywords"], context))
            entries.append(entry)
            disease_scores.append(boosted)
        return entries, np.array(disease_scores)

    def retrieve_recommendation(self, context, disease, top_k=config.TOP_K_RECOMMENDATION):
        entries, scores = self._disease_scores(context, disease)
        if not entries:
            return build_fallback(disease, reason="no_kb_entries_for_disease")

        order = np.argsort(-scores)[:top_k]
        ranked = [(entries[i], float(scores[i])) for i in order]
        top_entry, top_score = ranked[0]
        matched_entries = [{"entry_id": e["id"], "similarity_score": round(s, 4)} for e, s in ranked]

        if top_score < config.SIMILARITY_THRESHOLD_FALLBACK:
            return build_fallback(disease, matched_entries=matched_entries)

        close = [(e, s) for e, s in ranked if (top_score - s) <= config.CLOSE_SCORE_MARGIN]
        merged_tests = sorted({t for e, _ in close for t in e["suggested_follow_up_tests"]})

        return {
            "recommendation": top_entry["recommendation"],
            "diet_advice": top_entry["diet_advice"],
            "lifestyle_advice": top_entry["lifestyle_advice"],
            "suggested_follow_up_tests": merged_tests,
            "urgency_level": top_entry["urgency_level"],
            "evidence_tags": top_entry["evidence_tags"],
            "matched_entries": matched_entries,
            "primary_entry_id": top_entry["id"],
            "similarity_score": round(top_score, 4),
            "match_confidence": "high" if top_score >= config.SIMILARITY_THRESHOLD_HIGH else "moderate",
            "is_fallback": False,
            "disease": disease,
        }

    def suggest_tests(self, context, disease, top_k=config.TOP_K_SUGGEST_TESTS,
                       similarity_threshold=config.SUGGEST_TESTS_THRESHOLD):
        """
        Shared with validation/engine.py's ValidationEngine ("Suggested Additional
        Tests"). Ranks/filters by each KB entry's `suggested_follow_up_tests` field
        rather than surfacing the whole recommendation — reuses the same
        embeddings/cache as retrieve_recommendation(), but with a wider top_k and
        lower threshold, since an unnecessary extra test suggestion is low-stakes
        while a wrong holistic recommendation is not.
        """
        entries, scores = self._disease_scores(context, disease)
        test_scores = {}
        for entry, score in zip(entries, scores):
            if score < similarity_threshold:
                continue
            for test_name in entry["suggested_follow_up_tests"]:
                best = test_scores.get(test_name)
                if best is None or score > best["similarity_score"]:
                    test_scores[test_name] = {
                        "test_name": test_name,
                        "reason": entry["recommendation"],
                        "matched_scenario": entry["id"],
                        "similarity_score": round(float(score), 4),
                    }
        return sorted(test_scores.values(), key=lambda r: -r["similarity_score"])[:top_k]

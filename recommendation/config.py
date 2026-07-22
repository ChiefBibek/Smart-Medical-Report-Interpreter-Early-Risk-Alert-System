"""Tunable constants for the semantic recommendation/retrieval engine."""

MODEL_NAME = "all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD_FALLBACK = 0.30  # below this -> generic safe fallback (retrieve_recommendation)
SIMILARITY_THRESHOLD_HIGH = 0.55      # above this -> match_confidence = "high"
KEYWORD_BOOST = 0.08                  # additive bonus per matching keyword
KEYWORD_BOOST_CAP = 0.25              # ceiling on the total accumulated keyword boost
CLOSE_SCORE_MARGIN = 0.05             # entries within this of the top score get their tests unioned in

TOP_K_RECOMMENDATION = 3
TOP_K_SUGGEST_TESTS = 5
SUGGEST_TESTS_THRESHOLD = 0.25        # lower than the fallback threshold: an extra test suggestion is low-risk

MEAN_DEVIATION_BAND = 0.15            # normalized-space band for the low/high/normal qualitative label

CREATE TABLE IF NOT EXISTS decision_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,                          -- ISO8601 UTC, business event time
    disease                 TEXT    NOT NULL,
    raw_input               TEXT    NOT NULL,                          -- JSON
    normalized_input        TEXT,                                      -- JSON, nullable
    validation_warnings     TEXT,                                      -- JSON array, nullable
    completeness_score      REAL,
    prediction              TEXT    NOT NULL,                          -- JSON: {probability, risk_level, alerts}
    shap_explanation        TEXT    NOT NULL,                          -- JSON: {top_factors, all_factors}
    matched_kb_entry_id     TEXT,                                      -- NULL if fallback was used
    similarity_score        REAL,
    recommendation_returned TEXT    NOT NULL,                          -- JSON, full object returned to the caller
    doctor_modifications    TEXT,                                      -- JSON, nullable, populated later via feedback
    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_decision_log_disease   ON decision_log(disease);
CREATE INDEX IF NOT EXISTS idx_decision_log_timestamp ON decision_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_decision_log_kb_entry  ON decision_log(matched_kb_entry_id);

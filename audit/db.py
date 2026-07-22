"""
Traceability / audit trail — Module 3.

Every AI decision (raw input, normalized input, validation warnings,
prediction, SHAP explanation, matched KB entry + similarity score, the
recommendation actually returned, and later any doctor-submitted correction)
is logged to a local SQLite file so it can be reconstructed and reviewed.

Connections are short-lived (opened and closed per call) rather than a
single shared connection, since sqlite3.Connection objects are not safe to
share across threads by default and Flask's dev server is threaded — this
avoids the standard Flask+sqlite3 footgun without needing an explicit lock.
WAL mode is enabled once in init_db() so concurrent reads don't block writes.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(_DB_DIR, "decision_log.sqlite3")
_SCHEMA_PATH = os.path.join(_DB_DIR, "schema.sql")


def _connect(db_path):
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DEFAULT_DB_PATH):
    """Idempotent: creates the table/indexes if absent, enables WAL mode."""
    conn = _connect(db_path)
    try:
        with open(_SCHEMA_PATH, encoding="utf-8") as fh:
            conn.executescript(fh.read())
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
    finally:
        conn.close()


def log_decision(disease, raw_input, normalized_input, validation_warnings, completeness_score,
                  prediction, shap_explanation, matched_kb_entry_id, similarity_score,
                  recommendation_returned, db_path=DEFAULT_DB_PATH, timestamp=None):
    """Inserts one row, returns the new integer decision_id."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO decision_log
               (timestamp, disease, raw_input, normalized_input, validation_warnings,
                completeness_score, prediction, shap_explanation, matched_kb_entry_id,
                similarity_score, recommendation_returned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts, disease,
                json.dumps(raw_input, ensure_ascii=False),
                json.dumps(normalized_input, ensure_ascii=False) if normalized_input is not None else None,
                json.dumps(validation_warnings, ensure_ascii=False) if validation_warnings is not None else None,
                completeness_score,
                json.dumps(prediction, ensure_ascii=False),
                json.dumps(shap_explanation, ensure_ascii=False),
                matched_kb_entry_id, similarity_score,
                json.dumps(recommendation_returned, ensure_ascii=False),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def record_doctor_feedback(decision_id, modifications, db_path=DEFAULT_DB_PATH):
    """Sets doctor_modifications and bumps updated_at. Returns False if decision_id not found."""
    ts = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "UPDATE decision_log SET doctor_modifications = ?, updated_at = ? WHERE id = ?",
            (json.dumps(modifications, ensure_ascii=False), ts, decision_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def _row_to_dict(row):
    d = dict(row)
    for field in ("raw_input", "normalized_input", "validation_warnings",
                   "prediction", "shap_explanation", "recommendation_returned", "doctor_modifications"):
        if d.get(field):
            d[field] = json.loads(d[field])
    return d


def get_decision(decision_id, db_path=DEFAULT_DB_PATH):
    """Returns the decision record with JSON columns parsed back into dicts/lists, or None."""
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM decision_log WHERE id = ?", (decision_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_decisions(disease=None, limit=100, db_path=DEFAULT_DB_PATH):
    """Most-recent-first listing, optionally filtered by disease."""
    conn = _connect(db_path)
    try:
        if disease:
            rows = conn.execute(
                "SELECT * FROM decision_log WHERE disease = ? ORDER BY id DESC LIMIT ?",
                (disease, limit),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM decision_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()

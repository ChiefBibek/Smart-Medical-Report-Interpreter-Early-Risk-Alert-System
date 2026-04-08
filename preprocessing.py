"""
Feature Normalization & Data Preprocessing
============================================
Min-max normalization from scratch. No sklearn.
"""

import numpy as np


# ─────────────────────────────────────────────
# Lab parameter reference ranges
# ─────────────────────────────────────────────
REFERENCE_RANGES = {
    "hemoglobin":  {"min": 12.0, "max": 17.5, "unit": "g/dL",  "global_min": 4.0,  "global_max": 20.0},
    "glucose":     {"min": 70.0, "max": 99.0,  "unit": "mg/dL", "global_min": 40.0, "global_max": 400.0},
    "rbc":         {"min": 4.5,  "max": 5.5,   "unit": "M/uL",  "global_min": 1.0,  "global_max": 7.0},
    "wbc":         {"min": 4.0,  "max": 11.0,  "unit": "K/uL",  "global_min": 1.0,  "global_max": 30.0},
    "platelets":   {"min": 150,  "max": 400,   "unit": "K/uL",  "global_min": 50,   "global_max": 800},
    "creatinine":  {"min": 0.6,  "max": 1.2,   "unit": "mg/dL", "global_min": 0.3,  "global_max": 5.0},
}

FEATURE_NAMES = list(REFERENCE_RANGES.keys())


class MinMaxNormalizer:
    """
    Min-Max normalization: x_norm = (x - min) / (max - min)
    Fits on training data or uses known physiological ranges.
    """

    def __init__(self):
        self.min_vals = {}
        self.max_vals = {}

    def fit(self, X, feature_names):
        """Fit on training data."""
        for i, name in enumerate(feature_names):
            self.min_vals[name] = float(X[:, i].min())
            self.max_vals[name] = float(X[:, i].max())
        return self

    def fit_from_ranges(self):
        """Use known physiological ranges (preferred for small datasets)."""
        for name, r in REFERENCE_RANGES.items():
            self.min_vals[name] = r["global_min"]
            self.max_vals[name] = r["global_max"]
        return self

    def transform(self, X, feature_names):
        """Normalize X using stored min/max."""
        X_norm = np.zeros_like(X, dtype=float)
        for i, name in enumerate(feature_names):
            lo = self.min_vals[name]
            hi = self.max_vals[name]
            X_norm[:, i] = np.clip((X[:, i] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        return X_norm

    def transform_single(self, values_dict):
        """Normalize a single patient's values."""
        row = np.array([[values_dict[f] for f in FEATURE_NAMES]])
        return self.transform(row, FEATURE_NAMES)[0]


def flag_abnormal(values_dict):
    """
    Returns a dict of {feature: status} where status is
    'normal', 'low', 'high', or 'critical_low' / 'critical_high'.
    """
    flags = {}
    for name, val in values_dict.items():
        if name not in REFERENCE_RANGES:
            continue
        r = REFERENCE_RANGES[name]
        if val < r["min"] * 0.75:
            flags[name] = "critical_low"
        elif val < r["min"]:
            flags[name] = "low"
        elif val > r["max"] * 1.5:
            flags[name] = "critical_high"
        elif val > r["max"]:
            flags[name] = "high"
        else:
            flags[name] = "normal"
    return flags

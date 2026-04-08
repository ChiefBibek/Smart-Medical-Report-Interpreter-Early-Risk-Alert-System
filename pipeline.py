"""
Main Pipeline — End-to-End Medical Risk Assessment
====================================================
Connects: OCR values → preprocessing → ML model → risk engine
          → explainability → alerts → recommendations

Usage:
    python pipeline.py
"""

import numpy as np
import os

from ml_model import LogisticRegressionScratch
from preprocessing import MinMaxNormalizer, FEATURE_NAMES, flag_abnormal
from risk_engine import RiskResult
from explainability import (
    compute_feature_contributions,
    format_explanation_report,
    generate_explanation_text,
)
from alerts_recommendations import (
    generate_alerts,
    generate_recommendations,
    format_alert_report,
)
from train_model import train_and_save


DISCLAIMER = """
╔══════════════════════════════════════════════════════════╗
║  DISCLAIMER: This system is for awareness only and is   ║
║  NOT a replacement for professional medical advice.     ║
║  Always consult a qualified healthcare professional.    ║
╚══════════════════════════════════════════════════════════╝
"""


class MedicalRiskPipeline:
    """
    Full pipeline: input → prediction → explanation → report.
    """

    def __init__(self):
        self.model      = LogisticRegressionScratch(learning_rate=0.1, n_iterations=2000)
        self.normalizer = MinMaxNormalizer()
        self._trained   = False

    def train(self):
        """Train the model (or load saved weights if available)."""
        if os.path.exists("model_weights.npy") and os.path.exists("model_bias.npy"):
            print("Loading saved model weights...")
            self.model.weights = np.load("model_weights.npy")
            self.model.bias    = float(np.load("model_bias.npy")[0])
            self.normalizer.fit_from_ranges()
        else:
            print("Training model from scratch...")
            trained_model, normalizer = train_and_save()
            self.model      = trained_model
            self.normalizer = normalizer
        self._trained = True

    def predict(self, raw_values: dict):
        """
        Full prediction pipeline for one patient.

        Parameters
        ----------
        raw_values : dict
            { 'hemoglobin': 8.2, 'glucose': 210, 'rbc': 3.1,
              'wbc': 11.8, 'platelets': 155, 'creatinine': 0.9 }

        Returns
        -------
        dict with all results
        """
        if not self._trained:
            self.train()

        # 1. Normalize
        norm_values = self.normalizer.transform_single(raw_values)

        # 2. ML prediction
        X = norm_values.reshape(1, -1)
        raw_prob = float(self.model.predict_proba(X)[0])

        # 3. Risk categorization + safety overrides
        result = RiskResult(raw_prob, raw_values)

        # 4. Flag abnormal parameters
        flags = flag_abnormal(raw_values)

        # 5. Explainability
        contributions, pct_importance = compute_feature_contributions(
            norm_values, self.model.weights, FEATURE_NAMES
        )

        # 6. Alerts
        alerts = generate_alerts(flags, result.overrides, result.category)

        # 7. Recommendations
        recommendations = generate_recommendations(flags, result.category)

        return {
            "raw_values":         raw_values,
            "normalized_values":  norm_values,
            "raw_probability":    raw_prob,
            "adjusted_probability": result.adjusted_probability,
            "risk_category":      result.category,
            "risk_icon":          result.icon,
            "confidence":         result.confidence,
            "override_applied":   result.override_applied,
            "overrides":          result.overrides,
            "flags":              flags,
            "contributions":      contributions,
            "pct_importance":     pct_importance,
            "explanation_text":   generate_explanation_text(contributions, flags),
            "alerts":             alerts,
            "recommendations":    recommendations,
        }

    def full_report(self, raw_values: dict):
        """Generate and print the complete formatted report."""
        r = self.predict(raw_values)

        print(DISCLAIMER)

        # Risk summary
        print(f"\n{'=' * 55}")
        print(f"  RISK ASSESSMENT SUMMARY")
        print(f"{'=' * 55}")
        print(f"  {r['risk_icon']}  Category   : {r['risk_category'].upper()}")
        print(f"  Probability : {r['adjusted_probability']:.3f}  ({r['confidence']})")
        if r["override_applied"]:
            print(f"  ⚠️  Override  : Rule-based override applied")

        # Abnormal flags
        print(f"\n  Lab parameter flags:")
        for feature, status in r["flags"].items():
            icon = "✅" if status == "normal" else "❌"
            print(f"    {icon}  {feature:<12}: {status}")

        # Explainability
        print("\n" + format_explanation_report(
            r["contributions"], r["pct_importance"],
            r["flags"], r["adjusted_probability"], r["risk_category"]
        ))

        # Alerts and recommendations
        print(format_alert_report(r["alerts"], r["recommendations"]))


# ─────────────────────────────────────────────
# Demo run with sample patient data
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = MedicalRiskPipeline()
    pipeline.train()

    # Sample patient (high-risk case — matches viva demo)
    sample_patient = {
        "hemoglobin":  8.2,   # critically low  (normal: 12–17.5 g/dL)
        "glucose":     210,   # elevated        (normal: 70–99 mg/dL)
        "rbc":         3.1,   # low             (normal: 4.5–5.5 M/uL)
        "wbc":         11.8,  # slightly high   (normal: 4–11 K/uL)
        "platelets":   155,   # normal
        "creatinine":  0.9,   # normal
    }

    print("\n" + "─" * 55)
    print("  SAMPLE PATIENT: High-Risk Case")
    print("─" * 55)
    pipeline.full_report(sample_patient)

    # Second example: low-risk case
    normal_patient = {
        "hemoglobin":  14.5,
        "glucose":     88,
        "rbc":         4.9,
        "wbc":         6.5,
        "platelets":   250,
        "creatinine":  0.8,
    }

    print("\n" + "─" * 55)
    print("  SAMPLE PATIENT: Low-Risk Case")
    print("─" * 55)
    pipeline.full_report(normal_patient)

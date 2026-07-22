"""
AI-Based Smart Medical Report Interpreter & Early Risk Detection System
Main Prediction Engine

Field Rules
-----------
Each disease has:
  - MANDATORY_FIELDS : prediction is blocked if any of these are missing
  - OPTIONAL_FIELDS  : prediction runs with imputation from training mean if absent
                       confidence score drops with each missing optional field

Usage
-----
    from predictor import MedicalRiskPredictor

    predictor = MedicalRiskPredictor()
    predictor.train_all()

    # Minimal input (mandatory only)
    result = predictor.predict({"disease": "anemia", "hemoglobin": 8.5})

    # Full input (mandatory + all optional)
    result = predictor.predict({
        "disease": "anemia",
        "hemoglobin": 8.5, "rbc": 3.2, "mcv": 70,
        "mch": 21, "hematocrit": 28, "ferritin": 6
    })
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.disease_models import AnemiaModel, DiabetesModel, InfectionModel, CholesterolModel
from validation.engine import ValidationEngine
from recommendation.context_builder import build_clinical_context
from recommendation.retrieval_engine import RecommendationEngine
from audit import db as audit_db

FIELD_SCHEMA = {
    "anemia": {
        "mandatory": {"hemoglobin": "Hemoglobin (g/dL)"},
        "optional":  {
            "rbc":        "Red Blood Cell count (M/uL)",
            "mcv":        "Mean Corpuscular Volume (fL)",
            "mch":        "Mean Corpuscular Hemoglobin (pg)",
            "hematocrit": "Hematocrit (%)",
            "ferritin":   "Serum Ferritin (ng/mL)",
        }
    },
    "diabetes": {
        "mandatory": {"glucose": "Fasting Blood Glucose (mg/dL)"},
        "optional":  {
            "hba1c":          "Glycated Hemoglobin HbA1c (%)",
            "bmi":            "Body Mass Index",
            "age":            "Patient Age (years)",
            "insulin":        "Serum Insulin (uU/mL)",
            "blood_pressure": "Diastolic Blood Pressure (mmHg)",
        }
    },
    "infection": {
        "mandatory": {"wbc": "White Blood Cell count (cells/uL)"},
        "optional":  {
            "neutrophils":  "Neutrophil percentage (%)",
            "lymphocytes":  "Lymphocyte percentage (%)",
            "crp":          "C-Reactive Protein (mg/L)",
            "esr":          "Erythrocyte Sedimentation Rate (mm/hr)",
            "temperature":  "Body Temperature (°C)",
        }
    },
    "cholesterol": {
        "mandatory": {"total_cholesterol": "Total Cholesterol (mg/dL)"},
        "optional":  {
            "ldl":               "LDL Cholesterol (mg/dL)",
            "hdl":               "HDL Cholesterol (mg/dL)",
            "triglycerides":     "Triglycerides (mg/dL)",
            "vldl":              "VLDL Cholesterol (mg/dL)",
            "cholesterol_ratio": "Total Cholesterol / HDL Ratio",
        }
    }
}


class MedicalRiskPredictor:
    SUPPORTED_DISEASES = ["anemia", "diabetes", "infection", "cholesterol"]

    def __init__(self):
        self.models = {
            "anemia":      AnemiaModel(),
            "diabetes":    DiabetesModel(),
            "infection":   InfectionModel(),
            "cholesterol": CholesterolModel(),
        }
        self.training_metrics = {}
        self._trained = False
        # Module 2/3: semantic recommendation engine (Sentence-BERT + knowledge base).
        self.recommender = RecommendationEngine()
        # Module 1: validation engine. Its "Suggested Additional Tests" reuses the same
        # embedding/KB machinery as the recommender via dependency injection, so the two
        # modules share one retrieval implementation instead of duplicating it.
        self.validator = ValidationEngine(FIELD_SCHEMA, test_suggestion_client=self.recommender)

    def train_all(self, verbose=True):
        if verbose:
            print("\n" + "=" * 60)
            print("  TRAINING ALL DISEASE MODELS")
            print("=" * 60)

        for disease, model in self.models.items():
            if verbose:
                print(f"\n🔬 Training {disease.upper()} model...")
            metrics = model.train_on_generated_data()
            self.training_metrics[disease] = metrics
            if verbose:
                print(f"   ✅ Accuracy : {metrics['accuracy']*100:.2f}%")
                print(f"   ✅ F1 Score : {metrics['f1_score']*100:.2f}%")
                print(f"   ✅ AUC      : {metrics['auc']:.4f}")

        if verbose:
            print("\n🔬 Loading recommendation knowledge base + embedding model "
                  "(all-MiniLM-L6-v2)...")
        self.recommender.warmup()
        audit_db.init_db()

        self._trained = True
        if verbose:
            print("\n" + "=" * 60)
            print("  ✅ ALL MODELS TRAINED SUCCESSFULLY")
            print("=" * 60 + "\n")
        return self.training_metrics

    def get_field_schema(self, disease=None):
        """Return mandatory/optional field schema for one or all diseases."""
        if disease:
            return FIELD_SCHEMA.get(disease.lower(), {"error": "Unknown disease"})
        return FIELD_SCHEMA

    def predict(self, input_data):
        if not self._trained:
            raise RuntimeError("Models not trained. Call train_all() first.")

        if "disease" not in input_data:
            return {
                "error": "Missing 'disease' field",
                "supported_diseases": self.SUPPORTED_DISEASES,
                "field_schema": FIELD_SCHEMA,
            }

        disease = input_data["disease"].lower().strip()
        if disease not in self.SUPPORTED_DISEASES:
            return {
                "error": f"Disease '{disease}' not supported",
                "supported_diseases": self.SUPPORTED_DISEASES,
            }

        # --- Module 1: validate + normalize (units, physiological bounds, completeness,
        # consistency) before the model ever sees the input. Impossible values (likely
        # OCR/data-entry errors) short-circuit here; the model never runs on them.
        validation_result = self.validator.validate(disease, input_data)
        if validation_result.blocked:
            return validation_result.to_blocking_error_dict()

        result = self.models[disease].predict(validation_result.normalized_input)
        if "error" in result:
            return result  # e.g. missing mandatory fields — unchanged existing shape

        # --- Module 2/3: semantic recommendation retrieval, replacing the old
        # hardcoded per-disease recommendation lists.
        field_labels = {**FIELD_SCHEMA[disease]["mandatory"], **FIELD_SCHEMA[disease]["optional"]}
        context = build_clinical_context(
            disease, result["risk_level"], result["risk_probability"],
            result["top_factors"], result["all_factors"],
            validation_result.normalized_input, result["fields_imputed"],
            result["fields_missing_optional"], field_labels,
            validation_result.adjusted_confidence,
            validation_warnings=[w["message"] for w in validation_result.consistency_warnings],
            completeness_label=validation_result.completeness_label,
            patient_history=input_data.get("patient_history"),
            alerts=result["alerts"],
        )
        rec = self.recommender.retrieve_recommendation(context, disease)

        result["recommendations"] = [
            rec["recommendation"],
            f"Diet: {rec['diet_advice']}",
            f"Lifestyle: {rec['lifestyle_advice']}",
        ]
        result["recommendation_detail"] = rec
        result["prediction_confidence"] = validation_result.adjusted_confidence
        result["validation"] = validation_result.to_public_dict()

        # --- Module 3: traceability — log every decision for later audit/doctor feedback.
        result["decision_id"] = audit_db.log_decision(
            disease=disease,
            raw_input=input_data,
            normalized_input=validation_result.normalized_input,
            validation_warnings=[w["message"] for w in validation_result.consistency_warnings],
            completeness_score=validation_result.completeness_score,
            prediction={"probability": result["risk_probability"], "risk_level": result["risk_level"],
                        "alerts": result["alerts"]},
            shap_explanation={"top_factors": result["top_factors"], "all_factors": result["all_factors"]},
            matched_kb_entry_id=rec.get("primary_entry_id"),
            similarity_score=rec.get("similarity_score"),
            recommendation_returned=rec,
        )

        result["model_metrics"] = self.training_metrics.get(disease, {})
        return result

    def predict_batch(self, input_list):
        return [self.predict(inp) for inp in input_list]

    def get_training_metrics(self):
        return self.training_metrics


# ─── Flask API ────────────────────────────────────────────────────────────────

def create_flask_app(predictor):
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("Run: pip install flask flask-cors")
        return None

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "models_trained": predictor._trained,
            "supported_diseases": MedicalRiskPredictor.SUPPORTED_DISEASES,
        })

    @app.route("/schema", methods=["GET"])
    def schema():
        """Return mandatory/optional field schema for all diseases."""
        disease = request.args.get("disease")
        return jsonify(predictor.get_field_schema(disease))

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            return jsonify(predictor.predict(request.get_json()))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        try:
            return jsonify(predictor.predict_batch(request.get_json()))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/metrics", methods=["GET"])
    def metrics():
        return jsonify(predictor.get_training_metrics())

    @app.route("/decisions/<int:decision_id>", methods=["GET"])
    def get_decision_route(decision_id):
        """Module 3 traceability: fetch the full logged record for one prediction."""
        record = audit_db.get_decision(decision_id)
        if record is None:
            return jsonify({"error": "decision not found", "decision_id": decision_id}), 404
        return jsonify(record)

    @app.route("/decisions/<int:decision_id>/feedback", methods=["POST"])
    def submit_feedback_route(decision_id):
        """Module 3 traceability: record a doctor's correction/override against a decision."""
        try:
            body = request.get_json() or {}
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        updated = audit_db.record_doctor_feedback(decision_id, body)
        if not updated:
            return jsonify({"error": "decision not found", "decision_id": decision_id}), 404
        return jsonify({"decision_id": decision_id, "status": "updated"}), 200

    return app


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_demo():
    predictor = MedicalRiskPredictor()
    predictor.train_all(verbose=True)

    test_cases = [
        # Full input
        {
            "label": "Patient A — Anemia (full input, all 6 fields)",
            "input": {"disease": "anemia", "hemoglobin": 8.5, "rbc": 3.2,
                      "mcv": 70, "mch": 21, "hematocrit": 28, "ferritin": 6}
        },
        # Mandatory only
        {
            "label": "Patient B — Anemia (mandatory only — hemoglobin alone)",
            "input": {"disease": "anemia", "hemoglobin": 8.5}
        },
        # Partial optional
        {
            "label": "Patient C — Diabetes (mandatory + 2 optional)",
            "input": {"disease": "diabetes", "glucose": 185, "hba1c": 7.8}
        },
        # Missing mandatory → error
        {
            "label": "Patient D — Diabetes (missing mandatory glucose → ERROR expected)",
            "input": {"disease": "diabetes", "hba1c": 7.8, "bmi": 32}
        },
        # Infection full
        {
            "label": "Patient E — Infection (full input)",
            "input": {"disease": "infection", "wbc": 16000, "neutrophils": 82,
                      "lymphocytes": 12, "crp": 75, "esr": 55, "temperature": 39.2}
        },
        # Infection minimal
        {
            "label": "Patient F — Infection (mandatory WBC only)",
            "input": {"disease": "infection", "wbc": 16000}
        },
        # Cholesterol with alias
        {
            "label": "Patient G — Cholesterol ('cholesterol' alias + LDL/HDL)",
            "input": {"disease": "cholesterol", "cholesterol": 265, "ldl": 178, "hdl": 38}
        },
        # Healthy anemia full
        {
            "label": "Patient H — Healthy anemia screen (full input)",
            "input": {"disease": "anemia", "hemoglobin": 14.5, "rbc": 4.9,
                      "mcv": 91, "mch": 30, "hematocrit": 44, "ferritin": 95}
        },
        # Module 1: blocking validation error (physiologically impossible value)
        {
            "label": "Patient I — Diabetes (BLOCKING validation error: impossible negative glucose)",
            "input": {"disease": "diabetes", "glucose": -50, "hba1c": 7.8, "bmi": 26}
        },
        # Module 2/3 discrimination pair — ferritin is the ONLY field that differs between J and K
        {
            "label": "Patient J — Anemia (low Hb + LOW ferritin → iron-deficiency pattern)",
            "input": {"disease": "anemia", "hemoglobin": 9.0, "rbc": 3.6, "mcv": 68,
                      "mch": 20, "hematocrit": 29, "ferritin": 5}
        },
        {
            "label": "Patient K — Anemia (low Hb + NORMAL ferritin → further-workup pattern, NOT iron)",
            "input": {"disease": "anemia", "hemoglobin": 9.0, "rbc": 3.6, "mcv": 68,
                      "mch": 20, "hematocrit": 29, "ferritin": 120}
        },
    ]

    results_by_label = {}

    print("\n" + "=" * 65)
    print("  PREDICTION DEMO")
    print("=" * 65)

    for case in test_cases:
        print(f"\n📋 {case['label']}")
        print("-" * 55)
        result = predictor.predict(case["input"])
        results_by_label[case["label"]] = result

        if "error" in result:
            if result.get("error_type") == "validation_blocking":
                print(f"  🚫 BLOCKED: {result['message']}")
                for be in result["blocking_errors"]:
                    print(f"     - {be['field']}={be['value']}: {be['message']}")
            else:
                print(f"  ❌ ERROR: {result['message']}")
                print(f"     Mandatory fields required: {result['mandatory_fields']}")
        else:
            conf_icon = "🟢" if result["prediction_confidence"] == 100 else "🟡"
            print(f"  Disease      : {result['disease']}")
            print(f"  Risk Level   : {result['risk_emoji']} {result['risk_level']}")
            print(f"  Probability  : {result['risk_probability']*100:.1f}%")
            print(f"  Confidence   : {conf_icon} {result['prediction_confidence']}%")
            print(f"  Provided     : {result['fields_provided']}")
            if result["fields_imputed"]:
                print(f"  Imputed      : {result['fields_imputed']}")
            print(f"  Explanation  : {result['explanation']}")
            if result["alerts"]:
                print(f"  Alerts       :")
                for a in result["alerts"]:
                    print(f"    {a}")
            print(f"  Top factors  :")
            for f in result["top_factors"]:
                imp_tag = " [estimated]" if f["imputed"] else ""
                print(f"    • {f['feature']}: {f['contribution']:.1f}%{imp_tag}")
            if "validation" in result:
                v = result["validation"]
                print(f"  Validation   : completeness={v['completeness_score']}% ({v['completeness_label']})")
                if v["unit_conversions_applied"]:
                    print(f"                 unit conversions: {v['unit_conversions_applied']}")
                if v["consistency_warnings"]:
                    print(f"                 consistency warnings: {[w['message'] for w in v['consistency_warnings']]}")
            if "recommendation_detail" in result:
                rd = result["recommendation_detail"]
                print(f"  Recommend    : [{rd['urgency_level']}] matched '{rd['primary_entry_id']}' "
                      f"(similarity={rd['similarity_score']})")
                for r in result["recommendations"]:
                    print(f"    • {r}")
            if "decision_id" in result:
                print(f"  Decision ID  : {result['decision_id']}")

    print("\n" + "=" * 65)
    print("  FERRITIN DISCRIMINATION CHECK (Patient J vs Patient K)")
    print("=" * 65)
    r_j = results_by_label.get("Patient J — Anemia (low Hb + LOW ferritin → iron-deficiency pattern)")
    r_k = results_by_label.get("Patient K — Anemia (low Hb + NORMAL ferritin → further-workup pattern, NOT iron)")
    if r_j and r_k and "recommendation_detail" in r_j and "recommendation_detail" in r_k:
        entry_j = r_j["recommendation_detail"]["primary_entry_id"]
        entry_k = r_k["recommendation_detail"]["primary_entry_id"]
        print(f"  J matched_scenario : {entry_j}")
        print(f"  K matched_scenario : {entry_k}")
        if entry_j != entry_k:
            print("  Result: ✅ PASS — recommendation differs by ferritin (iron-deficiency vs. further-workup)")
        else:
            print("  Result: ❌ FAIL — identical recommendation despite differing ferritin")
    else:
        print("  Result: ⚠️  SKIPPED — one or both patients did not produce a recommendation_detail")

    print("\n" + "=" * 65)
    print("  SAMPLE JSON OUTPUT (mandatory-only anemia prediction)")
    print("=" * 65)
    sample = predictor.predict({"disease": "anemia", "hemoglobin": 8.5})
    sample.pop("all_factors", None)
    sample.pop("model_metrics", None)
    print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if "--serve" in sys.argv:
        predictor = MedicalRiskPredictor()
        predictor.train_all(verbose=True)
        app = create_flask_app(predictor)
        if app:
            print("🚀 Starting Flask API on http://localhost:5000")
            app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        run_demo()

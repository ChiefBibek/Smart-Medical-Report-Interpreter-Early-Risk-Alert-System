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

        result = self.models[disease].predict(input_data)
        if "error" not in result:
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
    ]

    print("\n" + "=" * 65)
    print("  PREDICTION DEMO")
    print("=" * 65)

    for case in test_cases:
        print(f"\n📋 {case['label']}")
        print("-" * 55)
        result = predictor.predict(case["input"])

        if "error" in result:
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

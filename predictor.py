"""
AI-Based Smart Medical Report Interpreter & Early Risk Detection System
Main Prediction Engine

Usage:
    from predictor import MedicalRiskPredictor
    
    predictor = MedicalRiskPredictor()
    predictor.train_all()
    
    result = predictor.predict({
        "disease": "anemia",
        "hemoglobin": 10.2,
        "rbc": 3.8,
        "hematocrit": 31,
        "mcv": 74,
        "mch": 23,
        "ferritin": 7
    })
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.disease_models import AnemiaModel, DiabetesModel, InfectionModel, CholesterolModel


class MedicalRiskPredictor:
    """
    Central prediction engine.
    Manages all disease-specific models.
    """

    SUPPORTED_DISEASES = ["anemia", "diabetes", "infection", "cholesterol"]

    def __init__(self):
        self.models = {
            "anemia": AnemiaModel(),
            "diabetes": DiabetesModel(),
            "infection": InfectionModel(),
            "cholesterol": CholesterolModel()
        }
        self.training_metrics = {}
        self._trained = False

    def train_all(self, verbose=True):
        """Train all disease models on generated datasets."""
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
                print(f"   ✅ Accuracy:  {metrics['accuracy'] * 100:.2f}%")
                print(f"   ✅ F1 Score:  {metrics['f1_score'] * 100:.2f}%")
                print(f"   ✅ AUC:       {metrics['auc']:.4f}")
                print(f"   ✅ Precision: {metrics['precision'] * 100:.2f}%")
                print(f"   ✅ Recall:    {metrics['recall'] * 100:.2f}%")

        self._trained = True

        if verbose:
            print("\n" + "=" * 60)
            print("  ✅ ALL MODELS TRAINED SUCCESSFULLY")
            print("=" * 60 + "\n")

        return self.training_metrics

    def predict(self, input_data):
        """
        Main prediction endpoint.
        
        Args:
            input_data (dict): Must contain "disease" key + lab values
            
        Returns:
            dict: Structured JSON response for backend
        """
        if not self._trained:
            raise RuntimeError("Models not trained. Call train_all() first.")

        # Validate input
        if "disease" not in input_data:
            return {
                "error": "Missing 'disease' field in input",
                "supported_diseases": self.SUPPORTED_DISEASES
            }

        disease = input_data["disease"].lower().strip()

        if disease not in self.SUPPORTED_DISEASES:
            return {
                "error": f"Disease '{disease}' not supported",
                "supported_diseases": self.SUPPORTED_DISEASES
            }

        # Route to correct model
        model = self.models[disease]
        result = model.predict(input_data)

        # Add model performance info
        result["model_metrics"] = self.training_metrics.get(disease, {})

        return result

    def get_training_metrics(self):
        """Return training metrics for all models."""
        return self.training_metrics

    def predict_batch(self, input_list):
        """Predict for multiple patients at once."""
        return [self.predict(inp) for inp in input_list]


# ─────────────────────────────────────────────────────────────────────────────
# FLASK API WRAPPER (for .NET Backend Integration)
# ─────────────────────────────────────────────────────────────────────────────

def create_flask_app(predictor):
    """
    Create Flask REST API for .NET backend integration.
    Run with: python predictor.py --serve
    """
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("Flask not installed. Run: pip install flask flask-cors")
        return None

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "models_trained": predictor._trained,
            "supported_diseases": MedicalRiskPredictor.SUPPORTED_DISEASES
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json()
            result = predictor.predict(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        try:
            data = request.get_json()
            results = predictor.predict_batch(data)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/metrics", methods=["GET"])
    def metrics():
        return jsonify(predictor.get_training_metrics())

    return app


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    """Run a full demonstration of all disease models."""

    predictor = MedicalRiskPredictor()
    predictor.train_all(verbose=True)

    test_cases = [
        {
            "name": "Patient A — Suspected Anemia",
            "input": {
                "disease": "anemia",
                "hemoglobin": 8.5,
                "rbc": 3.2,
                "mcv": 70,
                "mch": 21,
                "hematocrit": 28,
                "ferritin": 6
            }
        },
        {
            "name": "Patient B — Diabetes Risk",
            "input": {
                "disease": "diabetes",
                "glucose": 185,
                "hba1c": 7.8,
                "bmi": 32,
                "age": 55,
                "insulin": 200,
                "blood_pressure": 92
            }
        },
        {
            "name": "Patient C — Possible Infection",
            "input": {
                "disease": "infection",
                "wbc": 16000,
                "neutrophils": 82,
                "lymphocytes": 12,
                "crp": 75,
                "esr": 55,
                "temperature": 39.2
            }
        },
        {
            "name": "Patient D — Cholesterol Risk",
            "input": {
                "disease": "cholesterol",
                "total_cholesterol": 265,
                "ldl": 178,
                "hdl": 38,
                "triglycerides": 230,
                "vldl": 48,
                "cholesterol_ratio": 7.0
            }
        },
        {
            "name": "Patient E — Healthy (Anemia Screen)",
            "input": {
                "disease": "anemia",
                "hemoglobin": 14.5,
                "rbc": 4.9,
                "mcv": 91,
                "mch": 30,
                "hematocrit": 44,
                "ferritin": 95
            }
        }
    ]

    print("\n" + "=" * 60)
    print("  PREDICTION DEMO")
    print("=" * 60)

    for case in test_cases:
        print(f"\n📋 {case['name']}")
        print("-" * 40)

        result = predictor.predict(case["input"])

        print(f"  Disease:      {result['disease']}")
        print(f"  Risk Level:   {result['risk_emoji']} {result['risk_level']}")
        print(f"  Probability:  {result['risk_probability'] * 100:.1f}%")
        print(f"  Explanation:  {result['explanation']}")

        print(f"  Top Factors:")
        for f in result["top_factors"]:
            print(f"    • {f['feature']}: {f['contribution']:.1f}%")

        if result["alerts"]:
            print(f"  🚨 Alerts:")
            for alert in result["alerts"]:
                print(f"    {alert}")

        print(f"  💡 Recommendations:")
        for rec in result["recommendations"][:2]:
            print(f"    • {rec}")

    print("\n✅ Demo completed successfully!")
    print("\n📤 Sample JSON Output (for .NET backend):")
    sample = predictor.predict(test_cases[0]["input"])
    # Clean for JSON output
    sample.pop("all_factors", None)
    sample.pop("model_metrics", None)
    print(json.dumps(sample, indent=2))


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

"""
Disease-Specific Model Trainers
Each disease has its own model, feature set, safety layer, and explainer.
"""

import numpy as np
from models.logistic_regression import LogisticRegressionScratch
from data.data_generator import (
    MedicalDataGenerator, MinMaxNormalizer,
    train_test_split, compute_metrics
)


class DiseaseModel:
    """Base class for all disease models."""

    def __init__(self, disease_name, learning_rate=0.05, n_iterations=2000):
        self.disease_name = disease_name
        self.model = LogisticRegressionScratch(
            learning_rate=learning_rate,
            n_iterations=n_iterations
        )
        self.normalizer = MinMaxNormalizer()
        self.feature_names = []
        self.is_trained = False
        self.metrics = {}

    def train(self, X, y, feature_names):
        """Train model with normalized data."""
        self.feature_names = feature_names
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

        # Normalize
        X_train_norm = self.normalizer.fit_transform(X_train)
        X_test_norm = self.normalizer.transform(X_test)

        # Train
        self.model.fit(X_train_norm, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_norm)
        y_prob = self.model.predict_proba(X_test_norm)
        self.metrics = compute_metrics(y_test, y_pred, y_prob)
        self.is_trained = True

        return self.metrics

    def predict(self, input_dict):
        """
        Predict risk from input dictionary.
        Returns: probability, risk_level, contributions
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def _map_input_to_features(self, input_dict):
        """Map OCR JSON keys to expected feature array."""
        values = []
        for fname in self.feature_names:
            val = input_dict.get(fname, np.nan)
            values.append(float(val) if val is not None else 0.0)
        return np.array(values)

    def _get_risk_level(self, probability):
        """Categorize risk probability."""
        if probability < 0.2:
            return "Low", "🟢"
        elif probability < 0.5:
            return "Borderline", "⚠️"
        elif probability < 0.75:
            return "Moderate", "🟡"
        else:
            return "High", "🔴"

    def _compute_contributions(self, raw_input_array):
        """
        Compute feature contributions for explainability.
        contribution = feature_value × weight
        importance = |contribution| / sum(|contributions|) × 100
        """
        weights, bias = self.model.get_weights()
        contributions = raw_input_array * weights

        total = np.sum(np.abs(contributions)) + 1e-10
        importances = (np.abs(contributions) / total) * 100

        result = []
        for i, fname in enumerate(self.feature_names):
            result.append({
                "feature": fname,
                "raw_value": round(float(raw_input_array[i]), 3),
                "weight": round(float(weights[i]), 4),
                "contribution": round(float(contributions[i]), 4),
                "importance_pct": round(float(importances[i]), 2)
            })

        # Sort by importance descending
        result.sort(key=lambda x: x["importance_pct"], reverse=True)
        return result

    def _generate_explanation(self, top_factors):
        """Generate human-readable explanation."""
        top2 = top_factors[:2]
        factors_str = " and ".join([f["feature"].replace("_", " ") for f in top2])
        return f"Risk is primarily influenced by {factors_str}."


# ─────────────────────────────────────────────────────────────────────────────
# ANEMIA MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AnemiaModel(DiseaseModel):
    """Anemia Risk Detection Model."""

    def __init__(self):
        super().__init__("Anemia", learning_rate=0.05, n_iterations=2000)

    def train_on_generated_data(self):
        X, y, feature_names = MedicalDataGenerator.generate_anemia_dataset(n_samples=2000)
        return self.train(X, y, feature_names)

    def _safety_override(self, input_dict, probability):
        """
        Safety Layer: Rule-based overrides for critical values.
        Always prefer false positives (conservative).
        """
        alerts = []
        forced_high = False

        hemoglobin = input_dict.get("hemoglobin", 99)
        rbc = input_dict.get("rbc", 99)
        hematocrit = input_dict.get("hematocrit", 99)
        ferritin = input_dict.get("ferritin", 99)

        if hemoglobin < 9:
            alerts.append("🔴 CRITICAL: Hemoglobin critically low (< 9 g/dL) — Severe anemia")
            forced_high = True
        elif hemoglobin < 12:
            alerts.append("⚠️ Hemoglobin below normal range (< 12 g/dL)")

        if rbc < 3.5:
            alerts.append("🔴 CRITICAL: RBC count critically low (< 3.5 M/uL)")
            forced_high = True
        elif rbc < 4.2:
            alerts.append("⚠️ RBC count below normal range")

        if hematocrit < 30:
            alerts.append("🔴 CRITICAL: Hematocrit critically low (< 30%)")
            forced_high = True
        elif hematocrit < 36:
            alerts.append("⚠️ Hematocrit below normal range")

        if ferritin < 10:
            alerts.append("🔴 LOW FERRITIN: Iron stores depleted (< 10 ng/mL)")

        if forced_high:
            probability = max(probability, 0.85)

        return probability, alerts

    def predict(self, input_dict):
        raw_array = self._map_input_to_features(input_dict)
        norm_array = self.normalizer.transform(raw_array.reshape(1, -1))[0]

        probability = float(self.model.predict_proba(norm_array.reshape(1, -1))[0])
        probability, alerts = self._safety_override(input_dict, probability)

        risk_level, risk_emoji = self._get_risk_level(probability)
        contributions = self._compute_contributions(norm_array)
        explanation = self._generate_explanation(contributions)

        recommendations = self._get_recommendations(risk_level, input_dict)

        return {
            "disease": "Anemia",
            "risk_probability": round(probability, 4),
            "risk_level": risk_level,
            "risk_emoji": risk_emoji,
            "top_factors": [
                {"feature": c["feature"], "contribution": c["importance_pct"]}
                for c in contributions[:3]
            ],
            "all_factors": contributions,
            "explanation": explanation,
            "alerts": alerts,
            "recommendations": recommendations,
            "disclaimer": "This system is for awareness only and not a replacement for professional medical advice."
        }

    def _get_recommendations(self, risk_level, input_dict):
        recs = []
        hemoglobin = input_dict.get("hemoglobin", 99)

        if risk_level in ["High", "Moderate"]:
            recs.append("Increase consumption of iron-rich foods (red meat, spinach, lentils)")
            recs.append("Consider Vitamin C intake to enhance iron absorption")
            recs.append("Consult a hematologist for complete blood work")
            if hemoglobin < 10:
                recs.append("🚨 Seek immediate medical attention — may require iron supplementation or transfusion")
        elif risk_level == "Borderline":
            recs.append("Monitor hemoglobin levels regularly")
            recs.append("Include iron-rich and folate-rich foods in diet")
            recs.append("Schedule a follow-up blood test in 4-6 weeks")
        else:
            recs.append("Maintain a balanced diet rich in iron and B12")
            recs.append("Continue regular health checkups")

        return recs


# ─────────────────────────────────────────────────────────────────────────────
# DIABETES MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DiabetesModel(DiseaseModel):
    """Diabetes Risk Detection Model."""

    def __init__(self):
        super().__init__("Diabetes", learning_rate=0.05, n_iterations=2000)

    def train_on_generated_data(self):
        X, y, feature_names = MedicalDataGenerator.generate_diabetes_dataset(n_samples=2000)
        return self.train(X, y, feature_names)

    def _safety_override(self, input_dict, probability):
        alerts = []
        forced_high = False

        glucose = input_dict.get("glucose", 0)
        hba1c = input_dict.get("hba1c", 0)
        bmi = input_dict.get("bmi", 0)

        if glucose > 200:
            alerts.append("🔴 CRITICAL: Blood glucose > 200 mg/dL — Possible hyperglycemia/diabetic crisis")
            forced_high = True
        elif glucose > 126:
            alerts.append("⚠️ Fasting glucose above diabetic threshold (> 126 mg/dL)")
        elif glucose > 100:
            alerts.append("⚠️ Pre-diabetic fasting glucose range (100-125 mg/dL)")

        if hba1c > 8.0:
            alerts.append("🔴 CRITICAL: HbA1c > 8.0% — Poor long-term glucose control")
            forced_high = True
        elif hba1c > 6.5:
            alerts.append("⚠️ HbA1c in diabetic range (> 6.5%)")
        elif hba1c > 5.7:
            alerts.append("⚠️ HbA1c in pre-diabetic range (5.7-6.4%)")

        if bmi > 35:
            alerts.append("⚠️ BMI > 35 — Obesity significantly increases diabetes risk")

        if forced_high:
            probability = max(probability, 0.85)

        return probability, alerts

    def predict(self, input_dict):
        raw_array = self._map_input_to_features(input_dict)
        norm_array = self.normalizer.transform(raw_array.reshape(1, -1))[0]

        probability = float(self.model.predict_proba(norm_array.reshape(1, -1))[0])
        probability, alerts = self._safety_override(input_dict, probability)

        risk_level, risk_emoji = self._get_risk_level(probability)
        contributions = self._compute_contributions(norm_array)
        explanation = self._generate_explanation(contributions)
        recommendations = self._get_recommendations(risk_level, input_dict)

        return {
            "disease": "Diabetes",
            "risk_probability": round(probability, 4),
            "risk_level": risk_level,
            "risk_emoji": risk_emoji,
            "top_factors": [
                {"feature": c["feature"], "contribution": c["importance_pct"]}
                for c in contributions[:3]
            ],
            "all_factors": contributions,
            "explanation": explanation,
            "alerts": alerts,
            "recommendations": recommendations,
            "disclaimer": "This system is for awareness only and not a replacement for professional medical advice."
        }

    def _get_recommendations(self, risk_level, input_dict):
        recs = []
        glucose = input_dict.get("glucose", 0)

        if risk_level in ["High", "Moderate"]:
            recs.append("Significantly reduce sugar and refined carbohydrate intake")
            recs.append("Increase physical activity — aim for 150 minutes/week")
            recs.append("Consult an endocrinologist immediately")
            if glucose > 200:
                recs.append("🚨 Seek emergency medical care — blood glucose critically elevated")
        elif risk_level == "Borderline":
            recs.append("Adopt a low-glycemic diet plan")
            recs.append("Monitor blood sugar levels daily")
            recs.append("Schedule HbA1c test in 3 months")
        else:
            recs.append("Maintain healthy weight and active lifestyle")
            recs.append("Annual fasting glucose screening recommended")

        return recs


# ─────────────────────────────────────────────────────────────────────────────
# INFECTION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class InfectionModel(DiseaseModel):
    """Infection Risk Detection Model."""

    def __init__(self):
        super().__init__("Infection", learning_rate=0.05, n_iterations=2000)

    def train_on_generated_data(self):
        X, y, feature_names = MedicalDataGenerator.generate_infection_dataset(n_samples=2000)
        return self.train(X, y, feature_names)

    def _safety_override(self, input_dict, probability):
        alerts = []
        forced_high = False

        wbc = input_dict.get("wbc", 0)
        crp = input_dict.get("crp", 0)
        temperature = input_dict.get("temperature", 37)
        neutrophils = input_dict.get("neutrophils", 0)

        if wbc > 20000:
            alerts.append("🔴 CRITICAL: WBC > 20,000 — Severe leukocytosis, possible severe infection/sepsis")
            forced_high = True
        elif wbc > 12000:
            alerts.append("⚠️ Elevated WBC count (> 12,000) — Possible infection")

        if crp > 100:
            alerts.append("🔴 CRITICAL: CRP > 100 mg/L — Severe systemic inflammation")
            forced_high = True
        elif crp > 10:
            alerts.append("⚠️ Elevated CRP (> 10 mg/L) — Inflammation detected")

        if temperature > 39.5:
            alerts.append("🔴 CRITICAL: High fever (> 39.5°C) — Possible serious infection")
            forced_high = True
        elif temperature > 38.0:
            alerts.append("⚠️ Fever detected (> 38°C)")

        if neutrophils > 80:
            alerts.append("⚠️ Elevated neutrophils (> 80%) — Bacterial infection suspected")

        if forced_high:
            probability = max(probability, 0.85)

        return probability, alerts

    def predict(self, input_dict):
        raw_array = self._map_input_to_features(input_dict)
        norm_array = self.normalizer.transform(raw_array.reshape(1, -1))[0]

        probability = float(self.model.predict_proba(norm_array.reshape(1, -1))[0])
        probability, alerts = self._safety_override(input_dict, probability)

        risk_level, risk_emoji = self._get_risk_level(probability)
        contributions = self._compute_contributions(norm_array)
        explanation = self._generate_explanation(contributions)
        recommendations = self._get_recommendations(risk_level, input_dict)

        return {
            "disease": "Infection",
            "risk_probability": round(probability, 4),
            "risk_level": risk_level,
            "risk_emoji": risk_emoji,
            "top_factors": [
                {"feature": c["feature"], "contribution": c["importance_pct"]}
                for c in contributions[:3]
            ],
            "all_factors": contributions,
            "explanation": explanation,
            "alerts": alerts,
            "recommendations": recommendations,
            "disclaimer": "This system is for awareness only and not a replacement for professional medical advice."
        }

    def _get_recommendations(self, risk_level, input_dict):
        recs = []
        wbc = input_dict.get("wbc", 0)

        if risk_level in ["High", "Moderate"]:
            recs.append("Seek immediate medical evaluation")
            recs.append("Complete blood culture and sensitivity testing recommended")
            recs.append("Do not self-medicate with antibiotics")
            if wbc > 20000:
                recs.append("🚨 Emergency: May indicate sepsis — go to hospital immediately")
        elif risk_level == "Borderline":
            recs.append("Monitor temperature and symptoms for 24-48 hours")
            recs.append("Increase fluid intake and rest")
            recs.append("Consult a doctor if symptoms worsen")
        else:
            recs.append("Maintain good hygiene and nutrition")
            recs.append("Ensure vaccinations are up to date")

        return recs


# ─────────────────────────────────────────────────────────────────────────────
# CHOLESTEROL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CholesterolModel(DiseaseModel):
    """Cholesterol Risk Detection Model."""

    def __init__(self):
        super().__init__("Cholesterol", learning_rate=0.05, n_iterations=2000)

    def train_on_generated_data(self):
        X, y, feature_names = MedicalDataGenerator.generate_cholesterol_dataset(n_samples=2000)
        return self.train(X, y, feature_names)

    def _safety_override(self, input_dict, probability):
        alerts = []
        forced_high = False

        total_cholesterol = input_dict.get("total_cholesterol", 0) or input_dict.get("cholesterol", 0)
        ldl = input_dict.get("ldl", 0)
        hdl = input_dict.get("hdl", 99)
        triglycerides = input_dict.get("triglycerides", 0)

        if total_cholesterol > 280:
            alerts.append("🔴 CRITICAL: Total cholesterol > 280 mg/dL — Very high cardiovascular risk")
            forced_high = True
        elif total_cholesterol > 240:
            alerts.append("⚠️ Total cholesterol > 240 mg/dL — High risk range")

        if ldl > 190:
            alerts.append("🔴 CRITICAL: LDL > 190 mg/dL — Possible familial hypercholesterolemia")
            forced_high = True
        elif ldl > 160:
            alerts.append("⚠️ LDL cholesterol in high risk range (> 160 mg/dL)")

        if hdl < 35:
            alerts.append("🔴 Very low HDL (< 35 mg/dL) — Significant cardiovascular risk factor")
        elif hdl < 40:
            alerts.append("⚠️ Low HDL cholesterol (< 40 mg/dL) — Increases heart disease risk")

        if triglycerides > 500:
            alerts.append("🔴 CRITICAL: Triglycerides > 500 mg/dL — Risk of pancreatitis")
            forced_high = True
        elif triglycerides > 200:
            alerts.append("⚠️ Elevated triglycerides (> 200 mg/dL)")

        if forced_high:
            probability = max(probability, 0.85)

        return probability, alerts

    def predict(self, input_dict):
        # Handle alias: "cholesterol" → "total_cholesterol"
        if "cholesterol" in input_dict and "total_cholesterol" not in input_dict:
            input_dict = dict(input_dict)
            input_dict["total_cholesterol"] = input_dict["cholesterol"]

        raw_array = self._map_input_to_features(input_dict)
        norm_array = self.normalizer.transform(raw_array.reshape(1, -1))[0]

        probability = float(self.model.predict_proba(norm_array.reshape(1, -1))[0])
        probability, alerts = self._safety_override(input_dict, probability)

        risk_level, risk_emoji = self._get_risk_level(probability)
        contributions = self._compute_contributions(norm_array)
        explanation = self._generate_explanation(contributions)
        recommendations = self._get_recommendations(risk_level, input_dict)

        return {
            "disease": "Cholesterol",
            "risk_probability": round(probability, 4),
            "risk_level": risk_level,
            "risk_emoji": risk_emoji,
            "top_factors": [
                {"feature": c["feature"], "contribution": c["importance_pct"]}
                for c in contributions[:3]
            ],
            "all_factors": contributions,
            "explanation": explanation,
            "alerts": alerts,
            "recommendations": recommendations,
            "disclaimer": "This system is for awareness only and not a replacement for professional medical advice."
        }

    def _get_recommendations(self, risk_level, input_dict):
        recs = []

        if risk_level in ["High", "Moderate"]:
            recs.append("Adopt a heart-healthy diet: reduce saturated fats, eliminate trans fats")
            recs.append("Increase soluble fiber intake (oats, beans, fruits)")
            recs.append("Exercise at least 30 minutes daily")
            recs.append("Consult a cardiologist for lipid-lowering therapy evaluation")
        elif risk_level == "Borderline":
            recs.append("Reduce dietary cholesterol and saturated fats")
            recs.append("Increase omega-3 fatty acids (fish, flaxseed)")
            recs.append("Recheck lipid panel in 3 months")
        else:
            recs.append("Maintain healthy dietary habits")
            recs.append("Annual lipid profile screening recommended")

        return recs

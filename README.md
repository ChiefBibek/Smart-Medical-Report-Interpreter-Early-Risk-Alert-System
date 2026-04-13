# AI-Based Smart Medical Report Interpreter & Early Risk Detection System
### Final Year Project — Python ML System

---

## System Overview

A disease-focused AI system that predicts risk for 4 diseases using Logistic Regression
implemented **FROM SCRATCH** using NumPy only. No sklearn. No pre-built ML libraries.

**4 Supported Diseases:**
- Anemia
- Diabetes
- Infection (Bacterial)
- Cholesterol (Cardiovascular)

---

## Project Structure

```
medical_ai/
├── predictor.py              ← MAIN ENTRY POINT (run this)
├── models/
│   ├── logistic_regression.py  ← LR from scratch (NumPy only)
│   └── disease_models.py       ← 4 disease models + safety + explainability
├── data/
│   └── data_generator.py       ← Synthetic data generation + metrics
└── README.md
```

---

## Quick Start

### 1. Install dependencies (NumPy only required for core ML)
```bash
pip install numpy
pip install flask flask-cors  # Optional: only for API server
```

### 2. Run demo
```bash
cd medical_ai
python predictor.py
```

### 3. Run as Flask API (for .NET backend)
```bash
python predictor.py --serve
# → API starts at http://localhost:5000
```

---

## API Endpoints (Flask)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + training status |
| POST | `/predict` | Single patient prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/metrics` | Training metrics for all models |

---

## Input Format (from .NET OCR backend)

```json
{
  "disease": "anemia",
  "hemoglobin": 10.2,
  "rbc": 3.8,
  "mcv": 74,
  "mch": 23,
  "hematocrit": 31,
  "ferritin": 7
}
```

**Supported field names per disease:**

| Disease | Fields |
|---------|--------|
| anemia | hemoglobin, rbc, mcv, mch, hematocrit, ferritin |
| diabetes | glucose, hba1c, bmi, age, insulin, blood_pressure |
| infection | wbc, neutrophils, lymphocytes, crp, esr, temperature |
| cholesterol | total_cholesterol (or cholesterol), ldl, hdl, triglycerides, vldl, cholesterol_ratio |

---

## Output Format (to .NET backend)

```json
{
  "disease": "Anemia",
  "risk_probability": 0.89,
  "risk_level": "High",
  "risk_emoji": "🔴",
  "top_factors": [
    {"feature": "hematocrit", "contribution": 24.54},
    {"feature": "mch", "contribution": 22.08},
    {"feature": "hemoglobin", "contribution": 18.33}
  ],
  "explanation": "Risk is primarily influenced by hematocrit and mch.",
  "alerts": [
    "🔴 CRITICAL: Hemoglobin critically low (< 9 g/dL) — Severe anemia",
    "🔴 CRITICAL: RBC count critically low (< 3.5 M/uL)"
  ],
  "recommendations": [
    "Increase consumption of iron-rich foods (red meat, spinach, lentils)",
    "Consider Vitamin C intake to enhance iron absorption"
  ],
  "disclaimer": "This system is for awareness only and not a replacement for professional medical advice.",
  "model_metrics": {
    "accuracy": 1.0,
    "f1_score": 1.0,
    "auc": 1.0
  }
}
```

---

## Risk Categorization

| Probability | Level | Emoji |
|-------------|-------|-------|
| p < 0.20 | Low Risk | 🟢 |
| 0.20 – 0.50 | Borderline | ⚠️ |
| 0.50 – 0.75 | Moderate | 🟡 |
| p > 0.75 | High | 🔴 |

---

## ML Implementation (Logistic Regression FROM SCRATCH)

### Core Math
```
z = w1*x1 + w2*x2 + ... + wn*xn + b     (linear combination)
p = 1 / (1 + e^(-z))                     (sigmoid activation)
L = -[y*log(p) + (1-y)*log(1-p)]         (binary cross-entropy loss)
w = w - lr * (dL/dw)                     (gradient descent)
```

### Features per disease
- Separate feature sets per disease
- Min-Max normalization (from scratch)
- 80/20 train/test split (from scratch)
- Metrics: Accuracy, Precision, Recall, F1, AUC (all from scratch)

---

## Safety Layer (Critical Overrides)

Each disease has rule-based overrides. Example:
- Hemoglobin < 9 → Force HIGH risk (anemia)
- Glucose > 200 → Force HIGH risk (diabetes)
- WBC > 20,000 → Force HIGH risk (infection)
- Total Cholesterol > 280 → Force HIGH risk (cholesterol)

**Philosophy: Prefer false positives. Safety first.**

---

## Explainability

Feature importance computed as:
```
contribution = feature_value × weight
importance = |contribution| / sum(|contributions|) × 100
```

Returns human-readable explanation of top contributing factors.

---

## Training Results (on 2000 synthetic samples per disease)

| Disease | Accuracy | F1 Score | AUC |
|---------|----------|----------|-----|
| Anemia | 100.00% | 100.00% | 1.000 |
| Diabetes | 100.00% | 100.00% | 0.998 |
| Infection | 100.00% | 100.00% | 0.999 |
| Cholesterol | 100.00% | 100.00% | 0.999 |

---

## Viva Preparation Summary

**Why logistic regression?**
- Interpretable and medically acceptable
- Works well for binary classification with small-medium datasets
- Weights directly show feature importance

**Why synthetic data?**
- Real medical data is limited, imbalanced, and privacy-restricted
- Synthetic data generated from clinical reference ranges
- Allows controlled class balance and edge case coverage

**Why separate models?**
- Different diseases use completely different biomarkers
- Separate models = better accuracy + cleaner explainability
- Modular design allows independent updates

**Why not deep learning?**
- Not explainable (black box)
- Requires massive datasets
- Not medically acceptable without clinical validation

**How is safety ensured?**
- Rule-based overrides for critical values
- Conservative thresholds (prefer false positives)
- Safety layer runs AFTER ML prediction and can override it

---

## Integration with .NET Backend

```python
# In your Flask app (predictor.py --serve)
# POST to http://localhost:5000/predict
# with JSON body containing disease + lab values
# Returns structured JSON response

# .NET can call this with HttpClient:
# var response = await httpClient.PostAsJsonAsync("http://localhost:5000/predict", labData);
```

---

*This system is for academic and awareness purposes only.*
*Not a replacement for professional medical diagnosis.*

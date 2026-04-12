# AI-Based Smart Medical Report Interpreter & Early Risk Detection System

> Final Year Project — Python ML System  
> Author: [Your Name] | Supervisor: [Supervisor Name]

---

## Overview

A modular, disease-focused AI system that analyzes structured medical lab report data (from OCR JSON) and predicts patient risk for four diseases using **Logistic Regression implemented from scratch** using NumPy only — no sklearn, no pre-built ML libraries.

The Python system integrates with a .NET backend via a REST API and returns fully explainable, structured JSON predictions.

---

## Supported Diseases

| Disease | Key Biomarkers |
|---------|----------------|
| Anemia | Hemoglobin, RBC, MCV, MCH, Hematocrit, Ferritin |
| Diabetes | Glucose, HbA1c, BMI, Age, Insulin, Blood Pressure |
| Infection | WBC, Neutrophils, CRP, ESR, Lymphocytes, Temperature |
| Cholesterol | Total Cholesterol, LDL, HDL, Triglycerides, VLDL, TC/HDL Ratio |

---

## Project Structure

```
medical_ai/
├── predictor.py              ← Main entry point
├── models/
│   ├── logistic_regression.py  ← LR from scratch (NumPy only)
│   └── disease_models.py       ← 4 disease models + safety + explainability
├── data/
│   └── data_generator.py       ← Synthetic data + preprocessing
└── requirements.txt
```

---

## Installation

```bash
git clone <your-repo-url>
cd medical_ai

pip install -r requirements.txt
```

---

## Usage

### Run Demo (CLI)

```bash
python predictor.py
```

Trains all 4 models and runs 5 test patient predictions, printing results to console.

### Run as Flask REST API

```bash
python predictor.py --serve
```

Starts the API at `http://localhost:5000`. Your .NET backend can POST to `/predict`.

---

## Input / Output Format

**Input (POST to `/predict`):**
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

**Output:**
```json
{
  "disease": "Anemia",
  "risk_probability": 0.89,
  "risk_level": "High",
  "top_factors": [
    { "feature": "hematocrit", "contribution": 24.54 }
  ],
  "explanation": "Risk is primarily influenced by hematocrit and mch.",
  "alerts": ["CRITICAL: Hemoglobin critically low (< 9 g/dL)"],
  "recommendations": ["Increase iron-rich foods", "Consult a hematologist"],
  "disclaimer": "For awareness only. Not a replacement for medical advice."
}
```

---

## Risk Levels

| Probability | Level | Indicator |
|-------------|-------|-----------|
| < 0.20 | Low | 🟢 |
| 0.20 – 0.50 | Borderline | ⚠️ |
| 0.50 – 0.75 | Moderate | 🟡 |
| > 0.75 | High | 🔴 |

---

## Model Performance (on 2000 synthetic samples per disease)

| Disease | Accuracy | F1 Score | AUC |
|---------|----------|----------|-----|
| Anemia | 100.00% | 100.00% | 1.000 |
| Diabetes | 100.00% | 100.00% | 0.998 |
| Infection | 100.00% | 100.00% | 0.999 |
| Cholesterol | 100.00% | 100.00% | 0.999 |

---

## Key Design Decisions

- **Logistic Regression from scratch** — NumPy only; no sklearn
- **Separate model per disease** — better accuracy, cleaner explainability
- **Synthetic data** — generated from clinical reference ranges; balanced classes
- **Safety overrides** — rule-based critical value detection overrides ML output
- **Explainability** — feature contributions computed as `value × weight` normalized to %
- **JSON pipeline** — drop-in compatible with .NET OCR backend

---

*This system is for academic and awareness purposes only.*  
*Not a replacement for professional medical diagnosis.*

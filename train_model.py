"""
Training Data Generator & Model Training
==========================================
Generates synthetic but medically plausible training data,
then trains the logistic regression model from scratch.

Run this file first: python train_model.py
"""

import numpy as np
from ml_model import LogisticRegressionScratch
from preprocessing import MinMaxNormalizer, FEATURE_NAMES

np.random.seed(42)

# ─────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────
def generate_training_data(n_samples=1000):
    """
    Generate medically plausible lab results with binary labels.

    Label = 1 (at-risk) when parameters deviate significantly from normal.
    Label = 0 (not at-risk) for normal ranges.
    """
    data = []
    labels = []

    for _ in range(n_samples):
        at_risk = np.random.random() < 0.45  # ~45% at-risk class

        if at_risk:
            hgb   = np.random.uniform(6.0, 11.5)    # low hemoglobin
            gluc  = np.random.uniform(120, 350)      # elevated glucose
            rbc   = np.random.uniform(2.0, 4.2)      # low RBC
            wbc   = np.random.uniform(11.0, 22.0)    # elevated WBC
            plt   = np.random.uniform(50, 149)       # low platelets
            creat = np.random.uniform(1.2, 4.5)      # elevated creatinine
        else:
            hgb   = np.random.uniform(12.0, 17.5)
            gluc  = np.random.uniform(70, 99)
            rbc   = np.random.uniform(4.5, 5.5)
            wbc   = np.random.uniform(4.0, 10.5)
            plt   = np.random.uniform(150, 400)
            creat = np.random.uniform(0.6, 1.1)

        # Add noise to simulate borderline cases
        hgb   += np.random.normal(0, 0.5)
        gluc  += np.random.normal(0, 8.0)
        rbc   += np.random.normal(0, 0.15)
        wbc   += np.random.normal(0, 0.8)
        plt   += np.random.normal(0, 15)
        creat += np.random.normal(0, 0.05)

        data.append([hgb, gluc, rbc, wbc, plt, creat])
        labels.append(1 if at_risk else 0)

    X = np.array(data)
    y = np.array(labels)
    return X, y


def train_and_save():
    print("Generating synthetic training data...")
    X_raw, y = generate_training_data(n_samples=1200)

    print(f"Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
    print(f"Class distribution: {int(y.sum())} at-risk, {int((1-y).sum())} normal")

    # Normalize
    normalizer = MinMaxNormalizer()
    normalizer.fit_from_ranges()
    X = normalizer.transform(X_raw, FEATURE_NAMES)

    # Train/test split (80/20) — manual, no sklearn
    split = int(0.8 * len(X))
    idx = np.random.permutation(len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("\nTraining logistic regression from scratch...")
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=2000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    # Manual precision, recall, F1
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n── Model Evaluation ──────────────────────────")
    print(f"  Accuracy  : {accuracy:.3f}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1-Score  : {f1:.3f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("──────────────────────────────────────────────")

    print("\nFinal weights:")
    for name, w in zip(FEATURE_NAMES, model.weights):
        print(f"  {name:<12}: {w:+.4f}")
    print(f"  {'bias':<12}: {model.bias:+.4f}")

    # Save weights to a simple file
    np.save("model_weights.npy", model.weights)
    np.save("model_bias.npy",    np.array([model.bias]))
    print("\nWeights saved to model_weights.npy and model_bias.npy")
    return model, normalizer


if __name__ == "__main__":
    train_and_save()

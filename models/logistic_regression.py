"""
Logistic Regression FROM SCRATCH using NumPy only.
No sklearn or any pre-built model library.

Enhancements over baseline:
  - L2 (Ridge) regularisation via l2_lambda parameter (default 0 = off)
    Reduces overfitting on small / noisy real-world datasets.
"""

import numpy as np


class LogisticRegressionScratch:
    """
    Binary Logistic Regression with optional L2 regularisation.

    Math:
        z    = w1*x1 + w2*x2 + ... + b
        p    = 1 / (1 + e^(-z))          [Sigmoid]
        Loss = -[y*log(p)+(1-y)*log(1-p)] + (λ/2n)||w||²   [BCE + L2]
        Gradient:
          dw = (XᵀΔ)/n + λ·w/n
          db = mean(Δ)                    Δ = p - y
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 l2_lambda=0.01, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.l2_lambda     = l2_lambda      # 0 = no regularisation
        self.random_state  = random_state
        self.weights       = None
        self.bias          = None
        self.loss_history  = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        eps    = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        bce    = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2     = (self.l2_lambda / (2 * len(y_true))) * np.sum(self.weights ** 2)
        return bce + l2

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        self.weights      = np.random.randn(n_features) * 0.01
        self.bias         = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            z           = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            loss = self._binary_cross_entropy(y, predictions)
            self.loss_history.append(loss)

            delta = predictions - y
            dw    = np.dot(X.T, delta) / n_samples + (self.l2_lambda / n_samples) * self.weights
            db    = np.mean(delta)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_weights(self):
        return self.weights, self.bias

"""
Logistic Regression from Scratch
=================================
Implements: sigmoid, binary cross-entropy, gradient descent
NO sklearn or pre-built ML models used.
"""

import numpy as np


class LogisticRegressionScratch:
    """
    Binary logistic regression implemented purely with NumPy.

    Forward pass:
        z = X @ w + b
        p = sigmoid(z) = 1 / (1 + e^-z)

    Loss (binary cross-entropy):
        L = -[y * log(p) + (1-y) * log(1-p)]

    Gradient:
        dL/dw = X.T @ (p - y) / n
        dL/db = mean(p - y)

    Update:
        w = w - lr * dL/dw
        b = b - lr * dL/db
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def _binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-9  # clip to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Train using gradient descent.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,)  — binary labels 0 or 1
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iter):
            z = X @ self.weights + self.bias
            p = self._sigmoid(z)

            # Gradients
            dw = (X.T @ (p - y)) / n_samples
            db = np.mean(p - y)

            # Weight update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Track loss every 100 iterations
            if i % 100 == 0:
                loss = self._binary_cross_entropy(y, p)
                self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """Return probability of class 1 for each sample."""
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_weights_summary(self, feature_names):
        """Return a dict of feature -> weight for explainability."""
        return {name: float(w) for name, w in zip(feature_names, self.weights)}

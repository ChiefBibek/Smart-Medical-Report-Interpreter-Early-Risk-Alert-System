"""
Logistic Regression FROM SCRATCH using NumPy only.
No sklearn or any pre-built model library.
"""

import numpy as np


class LogisticRegressionScratch:
    """
    Binary Logistic Regression implemented from scratch.
    
    Math:
        z = w1*x1 + w2*x2 + ... + b
        p = 1 / (1 + e^(-z))   [Sigmoid]
        Loss = -[y*log(p) + (1-y)*log(1-p)]  [Binary Cross-Entropy]
        Gradient Descent: w = w - lr * dL/dw
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Sigmoid activation: p = 1 / (1 + e^(-z))"""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        """Binary Cross-Entropy Loss"""
        eps = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) — 0 or 1
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Weight initialization (small random values)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.loss_history = []

        for iteration in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            # Compute loss
            loss = self._binary_cross_entropy(y, predictions)
            self.loss_history.append(loss)

            # Compute gradients
            error = predictions - y
            dw = np.dot(X.T, error) / n_samples
            db = np.mean(error)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        """Return probability of positive class (risk)."""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_weights(self):
        """Return model weights and bias."""
        return self.weights, self.bias

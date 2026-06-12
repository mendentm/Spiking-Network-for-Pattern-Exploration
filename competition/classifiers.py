# competition/classifiers.py
"""
Classifiers used in the competition.
====================================

RidgeReadout - the shared linear readout trained on top of each spiking
               reservoir. Closed-form (regularized least squares to one-hot
               targets), so the reservoir comparison is about the *representation*,
               not readout tuning. This is the standard echo-state/LSM readout.

NumpyMLP     - the "traditional NN" contestant: a from-scratch one-hidden-layer
               multilayer perceptron (ReLU + softmax, cross-entropy, Adam),
               trained end-to-end on the raw flattened input. Pure NumPy, CPU,
               no PyTorch - per project constraints.

Both standardize their inputs using statistics estimated on the training set
only (stored and reused at predict time) to avoid train/test leakage.
"""

import numpy as np


def _one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


class _Standardizer:
    """Z-score features using train-set statistics."""

    def fit(self, X):
        self.mu_ = X.mean(axis=0, keepdims=True)
        self.sd_ = X.std(axis=0, keepdims=True)
        self.sd_[self.sd_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu_) / self.sd_


class RidgeReadout:
    """Closed-form regularized linear classifier (ridge regression to one-hot)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, Phi, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        # Map labels to 0..C-1 (classes_ is sorted, labels are already 0..C-1 here).
        y_idx = np.searchsorted(self.classes_, y)

        self.scaler_ = _Standardizer().fit(Phi)
        Z = self.scaler_.transform(Phi)
        Z1 = np.hstack([Z, np.ones((Z.shape[0], 1))])     # bias column

        Y = _one_hot(y_idx, n_classes)
        A = Z1.T @ Z1 + self.alpha * np.eye(Z1.shape[1])
        self.W_ = np.linalg.solve(A, Z1.T @ Y)
        return self

    def predict(self, Phi):
        Z = self.scaler_.transform(Phi)
        Z1 = np.hstack([Z, np.ones((Z.shape[0], 1))])
        scores = Z1 @ self.W_
        return self.classes_[scores.argmax(axis=1)]

    def score(self, Phi, y):
        return float(np.mean(self.predict(Phi) == y))

    def n_params(self):
        return int(self.W_.size)


class NumpyMLP:
    """One-hidden-layer MLP (ReLU + softmax) trained with Adam. Pure NumPy."""

    def __init__(self, hidden=128, lr=1e-3, epochs=120, batch_size=64,
                 l2=1e-4, seed=0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed

    def _init_params(self, n_features, n_classes):
        rng = np.random.default_rng(self.seed)
        # He initialization for the ReLU layer.
        self.W1 = rng.standard_normal((n_features, self.hidden)) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.standard_normal((self.hidden, n_classes)) * np.sqrt(2.0 / self.hidden)
        self.b2 = np.zeros(n_classes)
        self._adam = {k: [np.zeros_like(getattr(self, k)), np.zeros_like(getattr(self, k))]
                      for k in ("W1", "b1", "W2", "b2")}
        self._t = 0

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _adam_step(self, grads):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            m, v = self._adam[k]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * (g * g)
            m_hat = m / (1 - b1 ** self._t)
            v_hat = v / (1 - b2 ** self._t)
            getattr(self, k)[...] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)

        self.scaler_ = _Standardizer().fit(X)
        Xs = self.scaler_.transform(X)
        self._init_params(Xs.shape[1], n_classes)
        Y = _one_hot(y_idx, n_classes)

        rng = np.random.default_rng(self.seed + 7)
        n = Xs.shape[0]
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                xb, yb = Xs[idx], Y[idx]
                m = len(idx)

                # Forward.
                z1 = xb @ self.W1 + self.b1
                h = np.maximum(0.0, z1)
                z2 = h @ self.W2 + self.b2
                p = self._softmax(z2)

                # Backward (cross-entropy + L2).
                dz2 = (p - yb) / m
                gW2 = h.T @ dz2 + self.l2 * self.W2
                gb2 = dz2.sum(axis=0)
                dh = dz2 @ self.W2.T
                dz1 = dh * (z1 > 0)
                gW1 = xb.T @ dz1 + self.l2 * self.W1
                gb1 = dz1.sum(axis=0)

                self._adam_step({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2})
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(X)
        h = np.maximum(0.0, Xs @ self.W1 + self.b1)
        p = self._softmax(h @ self.W2 + self.b2)
        return self.classes_[p.argmax(axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def n_params(self):
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

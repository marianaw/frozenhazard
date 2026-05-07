"""Tabular foundation model backends for in-context regression and classification.

Each callable implements:
    fn(X_ctx, y_ctx, X_query) -> np.ndarray
  - Regressor  : returns predicted values (same scale as y_ctx)
  - Classifier : returns P(class=1)  for binary y_ctx

Subsampling and batching are handled internally so callers are model-agnostic.
"""
import warnings
import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor

_TRAIN_MAX = 1_000   # conservative CPU limit for in-context samples
_BATCH     = 164     # max test queries per TabPFN forward pass on CPU


def _subsample(X, y, n_max=_TRAIN_MAX):
    if len(X) <= n_max:
        return X, y
    idx = np.random.choice(len(X), n_max, replace=False)
    return X[idx], y[idx]


# ── TabPFN ────────────────────────────────────────────────────────────────────

def tabpfn_regressor(X_ctx, y_ctx, X_query):
    X_ctx, y_ctx = _subsample(X_ctx, y_ctx)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
        m = TabPFNRegressor(device="cpu")
        m.fit(X_ctx, y_ctx)
        return np.concatenate([
            m.predict(X_query[i : i + _BATCH], output_type="median")
            for i in range(0, len(X_query), _BATCH)
        ])


def tabpfn_classifier(X_ctx, y_ctx, X_query):
    X_ctx, y_ctx = _subsample(X_ctx, y_ctx)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
        m = TabPFNClassifier(device="cpu")
        m.fit(X_ctx, y_ctx)
        return np.concatenate([
            m.predict_proba(X_query[i : i + _BATCH])[:, 1]
            for i in range(0, len(X_query), _BATCH)
        ])


# ── TabICL ────────────────────────────────────────────────────────────────────

def tabicl_regressor(X_ctx, y_ctx, X_query):
    from tabicl import TabICLRegressor
    X_ctx, y_ctx = _subsample(X_ctx, y_ctx)
    m = TabICLRegressor()
    m.fit(X_ctx, y_ctx)
    return m.predict(X_query)


def tabicl_classifier(X_ctx, y_ctx, X_query):
    from tabicl import TabICLClassifier
    X_ctx, y_ctx = _subsample(X_ctx, y_ctx)
    m = TabICLClassifier()
    m.fit(X_ctx, y_ctx)
    return m.predict_proba(X_query)[:, 1]

"""Frozen-sigma AFT: TabPFN backbone + scipy sigma fitting."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from tabpfn import TabPFNRegressor

EPS              = 1e-8
TABPFN_TRAIN_MAX = 1_000   # CPU limit for in-context training samples
TABPFN_BATCH     = 164     # max test queries per forward pass on CPU


def predict_log_time(X_train, T_train, X_test):
    """Predict log-times via TabPFN in-context regression.

    Trains on actual times T_train (uncensored), predicts median T for X_test,
    returns log of those predictions as mu_log.
    """
    X_train = np.asarray(X_train, dtype=float)
    T_train = np.asarray(T_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)
    if len(X_train) > TABPFN_TRAIN_MAX:
        idx = np.random.choice(len(X_train), TABPFN_TRAIN_MAX, replace=False)
        X_train, T_train = X_train[idx], T_train[idx]
    model = TabPFNRegressor(device="cpu")
    model.fit(X_train, T_train)
    preds = np.concatenate([
        model.predict(X_test[i : i + TABPFN_BATCH], output_type="median")
        for i in range(0, len(X_test), TABPFN_BATCH)
    ])
    return np.log(np.clip(preds, EPS, None))


def fit_sigma(T, Delta, mu_log):
    """Fit σ by censored log-likelihood MLE (log-normal AFT).

    T: observed times, Delta: event indicators, mu_log: predicted log-times.
    Returns scalar sigma.
    """
    T      = np.asarray(T,      dtype=float)
    Delta  = np.asarray(Delta,  dtype=float)
    mu_log = np.asarray(mu_log, dtype=float)
    log_t  = np.log(np.clip(T, EPS, None))

    def neg_ll(sigma_raw):
        sigma   = np.log1p(np.exp(sigma_raw)) + EPS
        z       = (log_t - mu_log) / sigma
        log_f   = -log_t - np.log(sigma) + norm.logpdf(z)
        log_S   = norm.logsf(z)
        return -np.sum(Delta * log_f + (1 - Delta) * log_S)

    res = minimize(neg_ll, x0=0.0, method="L-BFGS-B")
    return float(np.log1p(np.exp(res.x[0])) + EPS)


def survival_lognormal(t_grid, mu_log, sigma):
    """S(t | x) = 1 - Φ((log t - μ) / σ).

    t_grid: (T,)  mu_log: (n,)  →  returns (n, T).
    """
    z = (np.log(np.clip(t_grid[None, :], EPS, None)) - np.asarray(mu_log)[:, None]) / sigma
    return norm.sf(z)


def predicted_median(surv_matrix, t_grid):
    """First t where S(t) ≤ 0.5; inf if never crosses within the grid."""
    t_grid  = np.asarray(t_grid)
    crossed = surv_matrix <= 0.5
    has     = crossed.any(axis=1)
    idx     = crossed.argmax(axis=1)
    med     = np.full(len(surv_matrix), np.inf)
    med[has] = t_grid[idx[has]]
    return med

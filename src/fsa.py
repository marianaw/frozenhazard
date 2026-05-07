"""Frozen-sigma AFT: tabular foundation model backbone + scipy sigma fitting."""
import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from src.foundation_models import tabpfn_regressor

EPS              = 1e-8
TABPFN_TRAIN_MAX = 1_000   # re-exported for legacy imports / META logging
TABPFN_BATCH     = 164


def predict_log_time(X_train, T_train, X_test, model=tabpfn_regressor):
    """Predict log-times via in-context regression.

    Fits on actual times T_train, returns log of predicted times for X_test.
    """
    X_train = np.asarray(X_train, dtype=float)
    T_train = np.asarray(T_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)
    preds   = model(X_train, T_train, X_test)
    return np.log(np.clip(preds, EPS, None))


def fit_sigma(T, Delta, mu_log):
    """Fit σ by censored log-likelihood MLE (log-normal AFT).

    T: observed times, Delta: event indicators, mu_log: predicted log-times.
    """
    T      = np.asarray(T,      dtype=float)
    Delta  = np.asarray(Delta,  dtype=float)
    mu_log = np.asarray(mu_log, dtype=float)
    log_t  = np.log(np.clip(T, EPS, None))

    def neg_ll(sigma_raw):
        sigma = np.log1p(np.exp(sigma_raw)) + EPS
        z     = (log_t - mu_log) / sigma
        log_f = -log_t - np.log(sigma) + norm.logpdf(z)
        log_S = norm.logsf(z)
        return -np.sum(Delta * log_f + (1 - Delta) * log_S)

    res   = minimize(neg_ll, x0=0.0, method="L-BFGS-B")
    sigma = float(np.log1p(np.exp(res.x[0])) + EPS)
    if not res.success or not np.isfinite(sigma):
        warnings.warn(f"fit_sigma: optimization did not converge — σ={sigma:.4g}. "
                      f"Reason: {res.message}", RuntimeWarning, stacklevel=2)
    return sigma


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


def run_fsa(X_tr, T_tr, D_tr, X_te, t_grid, model):
    mask         = D_tr == 1
    X_all        = np.vstack([X_tr, X_te])
    mu_all       = predict_log_time(X_tr[mask], T_tr[mask], X_all, model)
    mu_tr, mu_te = mu_all[:len(X_tr)], mu_all[len(X_tr):]
    sigma        = fit_sigma(T_tr, D_tr, mu_tr)
    S            = survival_lognormal(t_grid, mu_te, sigma)
    return S, predicted_median(S, t_grid), sigma

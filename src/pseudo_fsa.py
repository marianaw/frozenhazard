"""Pseudo-observation AFT: log-normal AFT with pseudo-obs imputation for censored subjects.

Pipeline:
  1. Compute jackknife KM pseudo-observations θ_i at the median event time.
  2. Invert the log-normal survival function to get initial log-time targets for
     censored subjects: μ_i = log(t0) − σ_0 · Φ⁻¹(1 − θ_i).
  3. Train foundation model on (X_train, y_all): actual times for uncensored, imputed for censored.
  4. Fit σ via censored log-likelihood MLE.
  5. Return log-normal AFT survival curves — same interface as run_fsa.
"""
import numpy as np
from scipy.stats import norm

from src.foundation_models import tabpfn_regressor
from src.fsa import (EPS, fit_sigma, predict_log_time,
                     survival_lognormal, predicted_median)
from src.pseudo_obs import compute_pseudo_obs


def run_pseudo_fsa(X_tr, T_tr, D_tr, X_te, t_grid, sigma_init=0.5,
                   model=tabpfn_regressor):
    """Log-normal AFT with pseudo-observation imputation. Returns (S, med, sigma)."""
    unc = D_tr == 1

    # --- Pseudo-obs → initial log-time targets for censored subjects ---
    t0      = float(np.median(T_tr[unc]))
    theta   = compute_pseudo_obs(T_tr, D_tr, t0=t0)
    theta_c = np.clip(theta[~unc], EPS, 1 - EPS)
    log_y_c = np.maximum(
        np.log(np.clip(T_tr[~unc], EPS, None)),
        np.log(t0) - sigma_init * norm.ppf(1 - theta_c),
    )

    # --- Build training targets (time scale) ---
    y_all       = np.empty(len(T_tr))
    y_all[unc]  = T_tr[unc]
    y_all[~unc] = np.exp(log_y_c)

    # --- Predict log-times, fit sigma, survival function ---
    X_all          = np.vstack([X_tr, X_te])
    mu_all         = predict_log_time(X_tr, y_all, X_all, model=model)
    mu_tr, mu_te   = mu_all[:len(X_tr)], mu_all[len(X_tr):]
    sigma          = fit_sigma(T_tr, D_tr, mu_tr)
    S              = survival_lognormal(t_grid, mu_te, sigma)
    return S, predicted_median(S, t_grid), sigma

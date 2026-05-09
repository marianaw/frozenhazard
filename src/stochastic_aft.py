"""Iterative stochastic imputation for Log-Normal AFT via a tabular foundation model.

Each iteration:
  - Context : random 50% uncensored + 50% censored (current pseudo log-times)
  - E-step  : predict mu for all train + test subjects
  - Update  : log_y_c ← max(log C_c, mu_c + sigma * eps)
  - M-step  : fit sigma by censored log-likelihood MLE
  - Evaluate: CI and IBS on held-out test set
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.foundation_models import tabpfn_regressor
from src.fsa import (EPS, fit_sigma, survival_lognormal, predicted_median)
from src.pseudo_obs import compute_pseudo_obs
from src.pseudo_fsa import run_pseudo_fsa
from src.utils import compute_ci, compute_ibs

N_ITER = 20
FRAC   = 0.5


def _pseudo_obs_init(T, E, C_c, sigma=0.5):
    """Init log pseudo-times for censored subjects via jackknife KM pseudo-observations.

    Inverts the log-normal survival function: given θ_i ≈ S(t0 | x_i),
    μ_i = log(t0) − σ · Φ⁻¹(1 − θ_i).
    """
    t0      = float(np.median(T[E == 1]))
    theta   = compute_pseudo_obs(T, E, t0=t0)
    theta_c = np.clip(theta[~(E == 1)], EPS, 1 - EPS)
    mu_c    = np.log(t0) - sigma * norm.ppf(1 - theta_c)
    return np.maximum(np.log(np.clip(C_c, EPS, None)), mu_c)


def run_stochastic_aft(X_u, y_u, X_c, C_c, X_test, T_test, E_test, t_grid,
                       n_iter=N_ITER, seed=0, model=tabpfn_regressor):
    """Iterative stochastic AFT imputation.

    Parameters
    ----------
    X_u, y_u   : uncensored covariates and event times
    X_c, C_c   : censored covariates and censoring times
    X_test     : test covariates
    T_test, E_test : test times and event indicators
    t_grid     : evaluation time grid
    model      : regression backend callable (X_ctx, y_ctx, X_query) -> predictions
    Returns DataFrame indexed by iteration: CI | IBS | Sigma
    """
    rng      = np.random.default_rng(seed)
    log_y_u  = np.log(np.clip(y_u, EPS, None))
    log_C_c  = np.log(np.clip(C_c, EPS, None))
    n_u, n_c = len(X_u), len(X_c)

    T_train = np.concatenate([y_u, C_c])
    E_train = np.concatenate([np.ones(n_u), np.zeros(n_c)])
    X_train = np.vstack([X_u, X_c])

    # --- Warm start: one pass of pseudo_fsa → data-driven sigma ---
    _, _, sigma = run_pseudo_fsa(X_train, T_train, E_train, X_test, t_grid, model=model)
    log_y_c     = _pseudo_obs_init(T_train, E_train, C_c, sigma=sigma)

    print(f"  Warm-start σ = {sigma:.4f}")
    print(f"{'Iter':>4} | {'CI':>7} | {'IBS':>7} | {'Sigma':>7}")
    print("-" * 36)

    rows = []
    for it in range(1, n_iter + 1):
        # --- Context: all uncensored + 50% censored (pseudo-targets) ---
        idx_c     = rng.choice(n_c, max(1, int(FRAC * n_c)), replace=False)
        X_ctx     = np.vstack([X_u, X_c[idx_c]])
        log_y_ctx = np.concatenate([log_y_u, log_y_c[idx_c]])

        # --- Predict mu for all train + test in one forward pass ---
        mu_all  = model(X_ctx, log_y_ctx, np.vstack([X_train, X_test]))
        mu_tr   = mu_all[:len(X_train)]
        mu_c    = mu_all[n_u : len(X_train)]
        mu_test = mu_all[len(X_train):]

        # --- Update: imputed log-times floored at censoring time ---
        log_y_c = np.maximum(log_C_c, mu_c + sigma * rng.standard_normal(n_c))

        # --- M-step: fit sigma via censored log-likelihood MLE ---
        sigma = fit_sigma(T_train, E_train, mu_tr)

        # --- Evaluate ---
        S   = survival_lognormal(t_grid, mu_test, sigma)
        med = predicted_median(S, t_grid)
        ci  = compute_ci(T_test, E_test, med)
        ibs = compute_ibs(T_train, E_train, T_test, E_test, S, t_grid)

        print(f"{it:>4} | {ci:>7.4f} | {ibs:>7.4f} | {sigma:>7.4f}")
        rows.append({"Iter": it, "CI": ci, "IBS": ibs, "Sigma": sigma})

    return pd.DataFrame(rows).set_index("Iter")

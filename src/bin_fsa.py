"""Discrete survival analysis via binary classification (bin-FSA).

For K time quantiles t_1 < ... < t_K, trains one binary classifier per
quantile on the observable subset O_k, obtaining P(T > t_k | x) for all subjects.
Survival function is a step function over the K quantiles.
"""
import numpy as np

from src.foundation_models import tabpfn_classifier
from src.fsa import predicted_median

K_DEFAULT = 10


def predict_survival_binary(X_train, T_train, Delta_train, X_test, t_quantiles,
                             model=tabpfn_classifier):
    """Train one binary classifier per quantile; return (n_test, K) survival probabilities.

    O_k = uncensored subjects ∪ censored subjects with C_i > t_k.
    Y_ik = 1 if T_i > t_k (subject alive), 0 otherwise.
    """
    n_test  = len(X_test)
    K       = len(t_quantiles)
    S_quant = np.ones((n_test, K))

    for k, t_k in enumerate(t_quantiles):
        in_Ok = (Delta_train == 1) | (T_train > t_k)
        X_ok  = X_train[in_Ok]
        Y_ok  = (T_train[in_Ok] > t_k).astype(int)

        if Y_ok.sum() == 0 or Y_ok.sum() == len(Y_ok):
            S_quant[:, k] = float(Y_ok.mean())
            continue

        S_quant[:, k] = model(X_ok, Y_ok, X_test)

    return S_quant


def _stepwise_surv(S_quant, t_quantiles, t_grid):
    """Extend (n, K) step survival onto t_grid.

    S(t) = 1            for t < t_1
    S(t) = S_quant[:,k] for t in [t_k, t_{k+1})
    """
    idx = np.searchsorted(t_quantiles, t_grid, side="right") - 1   # (T,)
    pre = idx < 0
    S   = S_quant[:, np.clip(idx, 0, len(t_quantiles) - 1)]        # (n, T)
    S[:, pre] = 1.0
    return S


def run_bin_fsa(X_tr, T_tr, D_tr, X_te, t_grid, K=K_DEFAULT, model=tabpfn_classifier):
    """Discrete-time FSA via binary classification.

    Returns (S, med, np.nan) — same signature as run_fsa.
    """
    event_times = T_tr[D_tr == 1]
    t_quantiles = np.percentile(event_times, np.linspace(5, 95, K))

    S_quant = predict_survival_binary(X_tr, T_tr, D_tr, X_te, t_quantiles, model=model)
    S       = _stepwise_surv(S_quant, t_quantiles, t_grid)
    return S, predicted_median(S, t_grid), np.nan

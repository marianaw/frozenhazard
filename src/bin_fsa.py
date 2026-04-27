"""Discrete survival analysis via binary classification (bin-FSA).

For K time quantiles t_1 < ... < t_K, trains one TabPFN binary classifier per
quantile on the observable subset O_k, obtaining P(T > t_k | x) for all subjects.
Survival function is a step function over the K quantiles.
"""
import warnings
import numpy as np
from tabpfn import TabPFNClassifier

from src.fsa import TABPFN_BATCH, TABPFN_TRAIN_MAX, predicted_median

K_DEFAULT = 10


def predict_survival_binary(X_train, T_train, Delta_train, X_test, t_quantiles):
    """Train one binary classifier per quantile; return (n_test, K) survival probabilities.

    O_k = uncensored subjects ∪ censored subjects with C_i > t_k.
    Y_ik = 1 if T_i > t_k (subject alive), 0 otherwise.
    """
    n_test = len(X_test)
    K      = len(t_quantiles)
    S_quant = np.ones((n_test, K))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
        clf = TabPFNClassifier(device="cpu")

        for k, t_k in enumerate(t_quantiles):
            in_Ok = (Delta_train == 1) | (T_train > t_k)
            X_ok  = X_train[in_Ok]
            Y_ok  = (T_train[in_Ok] > t_k).astype(int)

            # skip degenerate quantiles (all same class)
            if Y_ok.sum() == 0 or Y_ok.sum() == len(Y_ok):
                S_quant[:, k] = float(Y_ok.mean())
                continue

            if len(X_ok) > TABPFN_TRAIN_MAX:
                idx  = np.random.choice(len(X_ok), TABPFN_TRAIN_MAX, replace=False)
                X_ok, Y_ok = X_ok[idx], Y_ok[idx]

            clf.fit(X_ok, Y_ok)
            S_quant[:, k] = np.concatenate([
                clf.predict_proba(X_test[i : i + TABPFN_BATCH])[:, 1]
                for i in range(0, n_test, TABPFN_BATCH)
            ])

    return S_quant


def _stepwise_surv(S_quant, t_quantiles, t_grid):
    """Extend (n, K) step survival onto t_grid.

    S(t) = 1           for t < t_1
    S(t) = S_quant[:,k] for t in [t_k, t_{k+1})
    """
    idx = np.searchsorted(t_quantiles, t_grid, side="right") - 1   # (T,)
    pre = idx < 0
    S   = S_quant[:, np.clip(idx, 0, len(t_quantiles) - 1)]        # (n, T)
    S[:, pre] = 1.0
    return S


def run_bin_fsa(X_tr, T_tr, D_tr, X_te, t_grid, K=K_DEFAULT):
    """Discrete-time FSA via binary classification.

    Returns (S, med, np.nan) — same signature as run_fsa.
    """
    event_times = T_tr[D_tr == 1]
    t_quantiles = np.percentile(event_times, np.linspace(5, 95, K))

    S_quant = predict_survival_binary(X_tr, T_tr, D_tr, X_te, t_quantiles)
    S       = _stepwise_surv(S_quant, t_quantiles, t_grid)
    return S, predicted_median(S, t_grid), np.nan

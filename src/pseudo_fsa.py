"""Pseudo-observation FSA: TabPFN regression on LOO-KM pseudo-observations.

For K landmark times, compute pseudo-observations θ_ik = n·Ŝ(t_k) − (n−1)·Ŝ_{−i}(t_k)
for all training subjects. Train one TabPFNRegressor per landmark on (X_train, θ_k),
predict for X_test, clip to [0,1], then step-interpolate onto t_grid.
"""
import warnings
import numpy as np
from tabpfn import TabPFNRegressor

from src.fsa import TABPFN_BATCH, TABPFN_TRAIN_MAX, predicted_median
from src.bin_fsa import _stepwise_surv
from src.pseudo_obs import compute_pseudo_obs

K_DEFAULT = 10


def run_pseudo_fsa(X_tr, T_tr, D_tr, X_te, t_grid, K=K_DEFAULT):
    """Pseudo-observation FSA. Returns (S, med, np.nan) — same signature as run_fsa."""
    event_times = T_tr[D_tr == 1]
    t_landmarks = np.percentile(event_times, np.linspace(5, 95, K))

    n_te    = len(X_te)
    S_quant = np.ones((n_te, K))

    for k, t0 in enumerate(t_landmarks):
        theta = compute_pseudo_obs(T_tr, D_tr, t0=t0)
        theta = np.clip(theta, 0.0, 1.0)

        X_fit, y_fit = X_tr, theta
        if len(X_fit) > TABPFN_TRAIN_MAX:
            idx    = np.random.choice(len(X_fit), TABPFN_TRAIN_MAX, replace=False)
            X_fit  = X_fit[idx]
            y_fit  = y_fit[idx]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
            model = TabPFNRegressor(device="cpu")
            model.fit(X_fit, y_fit)
            preds = np.concatenate([
                model.predict(X_te[i : i + TABPFN_BATCH])
                for i in range(0, n_te, TABPFN_BATCH)
            ])

        S_quant[:, k] = np.clip(preds, 0.0, 1.0)

    S   = _stepwise_surv(S_quant, t_landmarks, t_grid)
    med = predicted_median(S, t_grid)
    return S, med, np.nan

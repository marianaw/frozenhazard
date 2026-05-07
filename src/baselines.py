"""Survival analysis baselines via lifelines and scikit-survival."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
from sksurv.ensemble import RandomSurvivalForest

from src.fsa import predicted_median


def _df(X, T, Delta):
    df = pd.DataFrame(X.astype(float))
    df['T'] = T
    df['E'] = Delta
    return df


def _lifelines_predict(model, X_test, t_grid):
    """(surv_matrix, median_pred) from a fitted lifelines model."""
    X_df = pd.DataFrame(X_test.astype(float))
    surv = model.predict_survival_function(X_df, times=t_grid).values.T  # (n, T)
    med = model.predict_median(X_df).values
    return surv, med


def fit_predict_cox(X_tr, T_tr, D_tr, X_te, t_grid, penalizer=0.0):
    m = CoxPHFitter(penalizer=penalizer).fit(_df(X_tr, T_tr, D_tr), duration_col='T', event_col='E')
    return _lifelines_predict(m, X_te, t_grid)


def fit_predict_weibull(X_tr, T_tr, D_tr, X_te, t_grid):
    m = WeibullAFTFitter().fit(_df(X_tr, T_tr, D_tr), duration_col='T', event_col='E')
    return _lifelines_predict(m, X_te, t_grid)


def fit_predict_lognormal(X_tr, T_tr, D_tr, X_te, t_grid):
    m = LogNormalAFTFitter().fit(_df(X_tr, T_tr, D_tr), duration_col='T', event_col='E')
    return _lifelines_predict(m, X_te, t_grid)


def fit_predict_rsf(X_tr, T_tr, D_tr, X_te, t_grid):
    y_tr = np.array([(bool(d), float(t)) for d, t in zip(D_tr, T_tr)],
                    dtype=[('event', bool), ('time', float)])
    m = RandomSurvivalForest(n_estimators=100, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    surv_fns = m.predict_survival_function(X_te)
    S = np.row_stack([[fn(t) for t in t_grid] for fn in surv_fns])
    return S, predicted_median(S, t_grid)


BASELINES = {
    'cox': fit_predict_cox,
    'weibull': fit_predict_weibull,
    'lognormal': fit_predict_lognormal,
    'rsf': fit_predict_rsf,
}

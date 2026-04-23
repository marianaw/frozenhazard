"""Dataset loading, splits, and evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index as _ci
from sksurv.metrics import integrated_brier_score as _ibs


# --- Preprocessing ---

def _preprocess(X_df, T, Delta):
    """One-hot encode categoricals, standardize continuous columns only, drop NaNs."""
    df = pd.DataFrame(X_df).copy()
    df['__T__'] = np.asarray(T, dtype=float)
    df['__D__'] = np.asarray(Delta, dtype=int)
    df = df.dropna().reset_index(drop=True)
    T     = df.pop('__T__').values.clip(1e-8)
    Delta = df.pop('__D__').values

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    return df.values.astype(float), T, Delta


# --- Dataset loaders ---

def _sksurv_load(loader_fn, time_field, event_field):
    X, y = loader_fn()
    return _preprocess(X, y[time_field].astype(float), y[event_field].astype(int))


def load_whas500():
    from sksurv.datasets import load_whas500
    return _sksurv_load(load_whas500, 'lenfol', 'fstat')


def load_gbsg():
    from sksurv.datasets import load_gbsg2
    return _sksurv_load(load_gbsg2, 'time', 'cens')


def load_support():
    from pycox.datasets import support
    df = support.read_df()
    return _preprocess(df.drop(columns=['duration', 'event']), df['duration'], df['event'])


def load_metabric():
    from pycox.datasets import metabric
    df = metabric.read_df()
    return _preprocess(df.drop(columns=['duration', 'event']), df['duration'], df['event'])


def load_flchain():
    from sksurv.datasets import load_flchain
    X, y = load_flchain()
    # 'chapter' is cause-of-death (NaN for censored) — drop to avoid losing all censored rows
    X = X.drop(columns=['chapter'])
    X['creatinine'] = X['creatinine'].fillna(X['creatinine'].median())
    return _preprocess(X, y['futime'].astype(float), y['death'].astype(int))


DATASETS = {
    'whas500': load_whas500,
    'gbsg': load_gbsg,
    # 'support': load_support,
    'metabric': load_metabric,
    # 'flchain': load_flchain,
}


# --- Splits ---

def make_splits(n, n_splits=10, test_size=0.2, seed=0):
    """Return list of (train_idx, test_idx) tuples."""
    rng = np.random.default_rng(seed)
    cut = int(n * (1 - test_size))
    splits = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        splits.append((idx[:cut], idx[cut:]))
    return splits


# --- Metrics ---

def _sksurv_y(T, Delta):
    return np.array([(bool(d), float(t)) for d, t in zip(Delta, T)],
                    dtype=[('event', bool), ('time', float)])


def compute_ci(T_test, Delta_test, median_pred):
    """Harrell C-index. median_pred: higher = longer survival."""
    return _ci(T_test, median_pred, Delta_test)


def compute_ibs(T_train, Delta_train, T_test, Delta_test, surv_matrix, t_grid):
    """IPCW integrated Brier score from t5 to t95 of test times."""
    y_train = _sksurv_y(T_train, Delta_train)

    # IPCW KM is fit on train — drop test subjects with T >= max(T_train)
    max_train = float(y_train['time'].max())
    valid = T_test < max_train
    if valid.sum() < 2:
        return np.nan
    T_test, Delta_test, surv_matrix = T_test[valid], Delta_test[valid], surv_matrix[valid]

    y_test = _sksurv_y(T_test, Delta_test)
    t5, t95 = np.percentile(T_test, 5), np.percentile(T_test, 95)
    t_lo = max(float(y_train['time'].min()), float(y_test['time'].min()), t5)
    t_hi = min(max_train, float(y_test['time'].max()), t95) - 1e-6
    mask = (t_grid >= t_lo) & (t_grid <= t_hi)
    if mask.sum() < 2:
        return np.nan
    return float(_ibs(y_train, y_test, surv_matrix[:, mask], t_grid[mask]))

"""Dataset loading, splits, and evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index as _ci
from sksurv.metrics import integrated_brier_score as _ibs


# --- Preprocessing ---

def _preprocess(X_df, T, Delta):
    """Drop NaNs, one-hot encode categoricals, standardize numerics."""
    df = pd.DataFrame(X_df).copy()
    df['__T__'] = np.asarray(T, dtype=float)
    df['__D__'] = np.asarray(Delta, dtype=int)
    df = df.dropna().reset_index(drop=True)
    T = df.pop('__T__').values
    Delta = df.pop('__D__').values
    cats = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if cats:
        df = pd.get_dummies(df, columns=cats, drop_first=True)
    df = df.astype(float)
    df[df.columns] = StandardScaler().fit_transform(df)
    return df.values, T, Delta


# --- Dataset loaders ---

def load_whas500():
    from lifelines.datasets import load_whas500
    df = load_whas500()
    return _preprocess(df.drop(columns=['lenfol', 'fstat']), df['lenfol'], df['fstat'])


def load_gbsg():
    from lifelines.datasets import load_gbsg2
    df = load_gbsg2()
    return _preprocess(df.drop(columns=['time', 'cens']), df['time'], df['cens'])


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
    T = y['futime'].astype(float)
    Delta = y['death'].astype(int)
    return _preprocess(X, T, Delta)


DATASETS = {
    'whas500': load_whas500,
    'gbsg': load_gbsg,
    'support': load_support,
    'metabric': load_metabric,
    'flchain': load_flchain,
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
    y_test = _sksurv_y(T_test, Delta_test)
    t5, t95 = np.percentile(T_test, 5), np.percentile(T_test, 95)
    # sksurv requires times strictly within [min(y_test.time), max(y_test.time)]
    t_lo = max(float(y_test['time'].min()), t5)
    t_hi = min(float(y_test['time'].max()) - 1e-6, t95)
    mask = (t_grid >= t_lo) & (t_grid <= t_hi)
    if mask.sum() < 2:
        return np.nan
    return float(_ibs(y_train, y_test, surv_matrix[:, mask], t_grid[mask]))

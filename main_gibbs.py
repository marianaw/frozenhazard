"""Gibbs/stochastic-imputation AFT experiment: one entry per dataset × split × iteration."""
import json
import logging
import os
from datetime import datetime, timezone

logging.getLogger("analytics").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import numpy as np

from src.stochastic_aft import run_stochastic_aft, N_ITER
from src.utils import DATASETS, make_splits
from main import T_GRID_PTS, TEST_SIZE, SEED

N_SPLITS     = 10
RESULTS_FILE = "results/results_gibbs.json"

META = {
    "n_splits":      N_SPLITS,
    "test_size":     TEST_SIZE,
    "seed":          SEED,
    "t_grid_points": T_GRID_PTS,
    "n_iter":        N_ITER,
}


def run_gibbs_dataset(name, from_split=0):
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    X, T, Delta = DATASETS[name]()
    splits  = make_splits(len(T), n_splits=N_SPLITS, test_size=TEST_SIZE, seed=SEED)
    results = []

    for k, (tr_idx, te_idx) in enumerate(splits):
        if k < from_split:
            continue
        X_tr, T_tr, D_tr = X[tr_idx], T[tr_idx], Delta[tr_idx]
        X_te, T_te, D_te = X[te_idx], T[te_idx], Delta[te_idx]
        t_grid = np.linspace(np.percentile(T_tr, 5), np.percentile(T_tr, 95), T_GRID_PTS)

        unc = D_tr == 1
        print(f"\n  Split {k+1}/{N_SPLITS}")
        df = run_stochastic_aft(
            X_u=X_tr[unc],  y_u=T_tr[unc],
            X_c=X_tr[~unc], C_c=T_tr[~unc],
            X_test=X_te, T_test=T_te, E_test=D_te,
            t_grid=t_grid, seed=SEED + k,
        )
        results.append(df.reset_index().to_dict(orient="records"))

    return results


def load_results(path=RESULTS_FILE):
    if not os.path.exists(path):
        return {"meta": META, "datasets": {}}
    with open(path) as f:
        return json.load(f)


def save_results(store, path=RESULTS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


if __name__ == "__main__":
    store = load_results()

    for ds in DATASETS:
        cached_splits = len(store["datasets"].get(ds, []))
        if cached_splits >= N_SPLITS:
            print(f"  {ds}: all {N_SPLITS} splits cached, skipping.")
            continue
        print(f"  {ds}: {cached_splits}/{N_SPLITS} splits cached, resuming from split {cached_splits+1}.")
        new_splits = run_gibbs_dataset(ds, from_split=cached_splits)
        store["datasets"].setdefault(ds, []).extend(new_splits)
        save_results(store)

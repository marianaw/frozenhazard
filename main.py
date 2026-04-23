"""Main experiment pipeline: runs all datasets × all methods × k splits."""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.fsa import (TABPFN_BATCH, TABPFN_TRAIN_MAX, fit_sigma,
                     predict_log_time, predicted_median, survival_lognormal)
from src.baselines import BASELINES
from src.utils import DATASETS, compute_ci, compute_ibs, make_splits

N_SPLITS     = 10
TEST_SIZE    = 0.2
SEED         = 0
T_GRID_PTS   = 200
RESULTS_FILE = "results/results.json"

META = {
    "n_splits":         N_SPLITS,
    "test_size":        TEST_SIZE,
    "seed":             SEED,
    "t_grid_points":    T_GRID_PTS,
    "tabpfn_train_max": TABPFN_TRAIN_MAX,
    "tabpfn_batch":     TABPFN_BATCH,
}


# --- Core experiment logic ---

def run_fsa(X_tr, T_tr, D_tr, X_te, t_grid):
    uncensored   = D_tr == 1
    X_all        = np.vstack([X_tr, X_te])
    mu_all       = predict_log_time(X_tr[uncensored], T_tr[uncensored], X_all)
    mu_tr, mu_te = mu_all[:len(X_tr)], mu_all[len(X_tr):]
    sigma        = fit_sigma(T_tr, D_tr, mu_tr)
    S            = survival_lognormal(t_grid, mu_te, sigma)
    return S, predicted_median(S, t_grid), sigma


def evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid):
    return {
        "ci":  float(compute_ci(T_te, D_te, med)),
        "ibs": float(compute_ibs(T_tr, D_tr, T_te, D_te, S, t_grid)),
    }


def run_dataset(name, n_splits=N_SPLITS):
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    X, T, Delta = DATASETS[name]()
    splits = make_splits(len(T), n_splits=n_splits, test_size=TEST_SIZE, seed=SEED)
    results = {m: [] for m in ["fsa", *BASELINES]}

    for k, (tr_idx, te_idx) in enumerate(splits):
        X_tr, T_tr, D_tr = X[tr_idx], T[tr_idx], Delta[tr_idx]
        X_te, T_te, D_te = X[te_idx], T[te_idx], Delta[te_idx]
        t_grid = np.linspace(np.percentile(T_tr, 5), np.percentile(T_tr, 95), T_GRID_PTS)

        S, med, sigma = run_fsa(X_tr, T_tr, D_tr, X_te, t_grid)
        row = evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid)
        row["sigma"] = float(sigma)
        results["fsa"].append(row)

        for bname, fn in BASELINES.items():
            S_b, med_b = fn(X_tr, T_tr, D_tr, X_te, t_grid)
            results[bname].append(evaluate(T_tr, D_tr, T_te, D_te, S_b, med_b, t_grid))

        print(f"  [{k+1}/{n_splits}] σ={sigma:.3f}  "
              f"CI(fsa)={results['fsa'][-1]['ci']:.3f}  "
              f"IBS(fsa)={results['fsa'][-1]['ibs']:.3f}")

    return results


# --- Persistence ---

def load_results(path=RESULTS_FILE):
    if not os.path.exists(path):
        return {"meta": META, "datasets": {}}
    with open(path) as f:
        return json.load(f)


def save_results(store, path=RESULTS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store["last_updated"] = datetime.utcnow().isoformat()
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


# --- Summary ---

def summarize(dataset_results):
    rows = []
    for method, metrics in dataset_results.items():
        cis  = [m["ci"]  for m in metrics]
        ibss = [m["ibs"] for m in metrics if not np.isnan(m["ibs"])]
        rows.append({
            "method":  method,
            "C-index": f"{np.mean(cis):.3f} ± {np.std(cis):.3f}",
            "IBS":     f"{np.mean(ibss):.3f} ± {np.std(ibss):.3f}",
        })
    return pd.DataFrame(rows).set_index("method")


if __name__ == "__main__":
    store = load_results()

    for ds in DATASETS:
        if ds in store["datasets"]:
            print(f"  {ds}: already done, skipping.")
            continue
        store["datasets"][ds] = run_dataset(ds)
        save_results(store)                        # save after each dataset
        print(f"\n{ds} results:")
        print(summarize(store["datasets"][ds]).to_string())

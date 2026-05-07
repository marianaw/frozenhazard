"""Main experiment pipeline: runs all datasets × all methods × k splits."""
import json
import logging
import os
from datetime import datetime, timezone

logging.getLogger("analytics").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from functools import partial

from src.foundation_models import tabpfn_regressor, tabpfn_classifier, tabicl_regressor, tabicl_classifier
from src.fsa import (TABPFN_BATCH, TABPFN_TRAIN_MAX, fit_sigma,
                     predict_log_time, predicted_median, survival_lognormal)
from src.bin_fsa import run_bin_fsa
from src.pseudo_fsa import run_pseudo_fsa
from src.baselines import BASELINES
from src.utils import DATASETS, compute_ci, compute_ibs, make_splits

# ── Backend selection ─────────────────────────────────────────────────────────
# Change BACKEND (and the two callables) to switch foundation models.
BACKEND    = "tabicl"
REGRESSOR  = tabicl_regressor
CLASSIFIER = tabicl_classifier

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
    "backend":          BACKEND,
}

# --- Core experiment logic ---

def run_fsa(X_tr, T_tr, D_tr, X_te, t_grid):
    uncensored   = D_tr == 1
    X_all        = np.vstack([X_tr, X_te])
    mu_all       = predict_log_time(X_tr[uncensored], T_tr[uncensored], X_all, model=REGRESSOR)
    mu_tr, mu_te = mu_all[:len(X_tr)], mu_all[len(X_tr):]
    sigma        = fit_sigma(T_tr, D_tr, mu_tr)
    S            = survival_lognormal(t_grid, mu_te, sigma)
    return S, predicted_median(S, t_grid), sigma


def evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid):
    return {
        "ci":  float(compute_ci(T_te, D_te, med)),
        "ibs": float(compute_ibs(T_tr, D_tr, T_te, D_te, S, t_grid)),
    }


PROPOSED    = {
    f"fsa_{BACKEND}":        run_fsa,
    f"bin_fsa_{BACKEND}":    partial(run_bin_fsa,    model=CLASSIFIER),
    f"pseudo_fsa_{BACKEND}": partial(run_pseudo_fsa, model=REGRESSOR),
}
ALL_METHODS = [*PROPOSED, *BASELINES]


def run_dataset(name, methods_to_run, n_splits=N_SPLITS):
    print(f"\n{'='*40}\n{name}  (running: {methods_to_run})\n{'='*40}")
    X, T, Delta = DATASETS[name]()
    splits  = make_splits(len(T), n_splits=n_splits, test_size=TEST_SIZE, seed=SEED)
    results = {m: [] for m in methods_to_run}

    proposed_run  = {m: PROPOSED[m]  for m in methods_to_run if m in PROPOSED}
    baselines_run = {m: BASELINES[m] for m in methods_to_run if m in BASELINES}

    for k, (tr_idx, te_idx) in enumerate(splits):
        X_tr, T_tr, D_tr = X[tr_idx], T[tr_idx], Delta[tr_idx]
        X_te, T_te, D_te = X[te_idx], T[te_idx], Delta[te_idx]
        t_grid = np.linspace(np.percentile(T_tr, 5), np.percentile(T_tr, 95), T_GRID_PTS)

        for pname, pfn in proposed_run.items():
            S, med, sigma = pfn(X_tr, T_tr, D_tr, X_te, t_grid)
            row = evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid)
            if not np.isnan(sigma):
                row["sigma"] = float(sigma)
            results[pname].append(row)

        for bname, fn in baselines_run.items():
            S_b, med_b = fn(X_tr, T_tr, D_tr, X_te, t_grid)
            results[bname].append(evaluate(T_tr, D_tr, T_te, D_te, S_b, med_b, t_grid))

        line = f"  [{k+1}/{n_splits}]"
        if "fsa" in results:
            line += f"  σ={results['fsa'][-1].get('sigma', float('nan')):.3f}"
        for m in methods_to_run:
            line += f"  CI({m})={results[m][-1]['ci']:.3f}"
        print(line)

    return results


# --- Persistence ---

# Proposed method names that predate backend suffixes (legacy keys).
_LEGACY_PROPOSED = {"fsa", "bin_fsa", "pseudo_fsa"}


def load_results(path=RESULTS_FILE):
    if not os.path.exists(path):
        return {"meta": META, "datasets": {}}
    with open(path) as f:
        store = json.load(f)
    # Migrate legacy keys (no backend suffix) → tabpfn suffix.
    # Safe: only renames if new key doesn't already exist.
    migrated = False
    for ds_data in store.get("datasets", {}).values():
        for old in list(ds_data):
            new = f"{old}_tabpfn"
            if old in _LEGACY_PROPOSED and new not in ds_data:
                ds_data[new] = ds_data.pop(old)
                migrated = True
    if migrated:
        save_results(store, path)
        print("  Migrated legacy method keys → *_tabpfn suffix.")
    return store


def save_results(store, path=RESULTS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store["last_updated"] = datetime.now(timezone.utc).isoformat()
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
        cached      = store["datasets"].get(ds, {})
        missing     = [m for m in ALL_METHODS if m not in cached]
        if not missing:
            print(f"  {ds}: all methods cached, skipping.")
            continue
        print(f"  {ds}: running missing methods: {missing}")
        new_results = run_dataset(ds, missing)
        store["datasets"].setdefault(ds, {}).update(new_results)
        save_results(store)
        print(f"\n{ds} results:")
        print(summarize(store["datasets"][ds]).to_string())

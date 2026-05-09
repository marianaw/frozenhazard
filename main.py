"""Main experiment pipeline: all datasets, all methods, both backends, all seeds.
Note that one seed produces 10 splits of the data."""
import json
import logging
import os
from datetime import datetime, timezone
from functools import partial

logging.getLogger("analytics").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

from src.foundation_models import tabpfn_regressor, tabpfn_classifier, tabicl_regressor, tabicl_classifier
from src.fsa import run_fsa
from src.bin_fsa import run_bin_fsa
from src.pseudo_fsa import run_pseudo_fsa
from src.baselines import BASELINES
from src.utils import DATASETS, compute_ci, compute_ibs, make_splits

BACKENDS = {
    "tabpfn": (tabpfn_regressor, tabpfn_classifier),
    "tabicl": (tabicl_regressor, tabicl_classifier),
}
SEED       = 0
N_SPLITS   = 10
TEST_SIZE  = 0.2
T_GRID_PTS = 200
RESULTS_FILE = "results/results.json"


def make_proposed(backend, regressor, classifier):
    return {
        f"fsa_{backend}":        partial(run_fsa, model=regressor),
        f"bin_fsa_{backend}":    partial(run_bin_fsa,    model=classifier),
        f"pseudo_fsa_{backend}": partial(run_pseudo_fsa, model=regressor),
    }


def evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid):
    return {
        "ci":  float(compute_ci(T_te, D_te, med)),
        "ibs": float(compute_ibs(T_tr, D_tr, T_te, D_te, S, t_grid)),
    }


def run_dataset(name, proposed, seed):
    print(f"\n{'='*40}\n{name}  (seed={seed})\n{'='*40}")
    X, T, Delta = DATASETS[name]()
    splits  = make_splits(len(T), n_splits=N_SPLITS, test_size=TEST_SIZE, seed=seed)
    results = {m: [] for m in [*proposed, *BASELINES]}

    for k, (tr_idx, te_idx) in enumerate(splits):
        X_tr, T_tr, D_tr = X[tr_idx], T[tr_idx], Delta[tr_idx]
        X_te, T_te, D_te = X[te_idx], T[te_idx], Delta[te_idx]
        t_grid = np.linspace(np.percentile(T_tr, 5), np.percentile(T_tr, 95), T_GRID_PTS)

        for pname, pfn in proposed.items():
            S, med, sigma = pfn(X_tr, T_tr, D_tr, X_te, t_grid)
            row = evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid)
            if not np.isnan(sigma):
                row["sigma"] = float(sigma)
            results[pname].append(row)

        for bname, fn in BASELINES.items():
            kwargs = {"penalizer": 0.1} if (bname == "cox" and name == "flchain") else {}
            S_b, med_b = fn(X_tr, T_tr, D_tr, X_te, t_grid, **kwargs)
            results[bname].append(evaluate(T_tr, D_tr, T_te, D_te, S_b, med_b, t_grid))

        cis = "  ".join(f"CI({m})={results[m][-1]['ci']:.3f}" for m in results)
        print(f"  [{k+1}/{N_SPLITS}]  {cis}")

    return results


def load_results(path=RESULTS_FILE):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_results(store, path=RESULTS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


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

    proposed = {}
    for backend, (reg, clf) in BACKENDS.items():
        proposed.update(make_proposed(backend, reg, clf))
    all_methods = [*proposed, *BASELINES]

    for ds in DATASETS:
        cached = store.get(ds, {})
        if all(m in cached for m in all_methods):
            print(f"  {ds}: all cached, skipping.")
            continue
        new_results = run_dataset(ds, proposed, SEED)
        store.setdefault(ds, {}).update(new_results)
        save_results(store)
        print(f"\n{ds}:")
        print(summarize(store[ds]).to_string())

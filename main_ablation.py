"""Ablation: bin_fsa C-index and IBS vs number of bins K across all datasets."""
import json
import logging
import os
from datetime import datetime, timezone

logging.getLogger("analytics").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import numpy as np

from src.bin_fsa import run_bin_fsa
from src.utils import DATASETS, make_splits
from main import evaluate, T_GRID_PTS, TEST_SIZE, SEED

K_VALUES      = [3, 5, 10, 20]
N_SPLITS      = 10
ABLATION_FILE = "results/results_ablation.json"


def run_ablation_dataset(name, ks_to_run):
    print(f"\n{'='*40}\n{name}  (K={ks_to_run})\n{'='*40}")
    X, T, Delta = DATASETS[name]()
    splits  = make_splits(len(T), n_splits=N_SPLITS, test_size=TEST_SIZE, seed=SEED)
    results = {k: [] for k in ks_to_run}

    for split_idx, (tr_idx, te_idx) in enumerate(splits):
        X_tr, T_tr, D_tr = X[tr_idx], T[tr_idx], Delta[tr_idx]
        X_te, T_te, D_te = X[te_idx], T[te_idx], Delta[te_idx]
        t_grid = np.linspace(np.percentile(T_tr, 5), np.percentile(T_tr, 95), T_GRID_PTS)

        for k in ks_to_run:
            S, med, _ = run_bin_fsa(X_tr, T_tr, D_tr, X_te, t_grid, K=k)
            row = evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid)
            results[k].append(row)

        line = f"  [{split_idx+1}/{N_SPLITS}]"
        for k in ks_to_run:
            line += f"  CI(K={k})={results[k][-1]['ci']:.3f}"
        print(line)

    return {str(k): results[k] for k in ks_to_run}


def load_ablation(path=ABLATION_FILE):
    if not os.path.exists(path):
        return {"k_values": K_VALUES, "datasets": {}}
    with open(path) as f:
        return json.load(f)


def save_ablation(store, path=ABLATION_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


if __name__ == "__main__":
    store = load_ablation()

    for ds in DATASETS:
        cached_ks  = set(store["datasets"].get(ds, {}).keys())
        missing_ks = [k for k in K_VALUES if str(k) not in cached_ks]
        if not missing_ks:
            print(f"  {ds}: all K cached, skipping.")
            continue
        print(f"  {ds}: running missing K={missing_ks}")
        new_results = run_ablation_dataset(ds, missing_ks)
        store["datasets"].setdefault(ds, {}).update(new_results)
        save_ablation(store)

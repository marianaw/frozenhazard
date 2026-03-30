"""Main experiment pipeline: runs all datasets × all methods × k splits."""
import numpy as np
import pandas as pd

from src.utils import DATASETS, make_splits, compute_ci, compute_ibs
from src.fsa import predict_log_time, fit_sigma, survival_lognormal, predicted_median
from src.baselines import BASELINES

EPS = 1e-8
N_SPLITS = 10
T_GRID_POINTS = 200


def run_fsa(X_tr, T_tr, D_tr, X_te, t_grid):
    """Frozen-sigma AFT with TabPFN backbone."""
    uncensored = D_tr == 1
    log_T_unc = np.log(np.clip(T_tr[uncensored], EPS, None))
    # One TabPFN call for all subjects (train + test)
    X_all = np.vstack([X_tr, X_te])
    mu_all = predict_log_time(X_tr[uncensored], log_T_unc, X_all)
    mu_tr, mu_te = mu_all[:len(X_tr)], mu_all[len(X_tr):]
    sigma = fit_sigma(T_tr, D_tr, mu_tr)
    S = survival_lognormal(t_grid, mu_te, sigma)
    return S, predicted_median(S, t_grid), sigma


def evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid):
    return {
        'ci': compute_ci(T_te, D_te, med),
        'ibs': compute_ibs(T_tr, D_tr, T_te, D_te, S, t_grid),
    }


def run_dataset(name, n_splits=N_SPLITS):
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    X, T, Delta = DATASETS[name]()
    splits = make_splits(len(T), n_splits=n_splits)
    results = {m: [] for m in ['fsa', *BASELINES]}

    for k, (tr_idx, te_idx) in enumerate(splits):
        X_tr, T_tr, D_tr = X[tr_idx], T[tr_idx], Delta[tr_idx]
        X_te, T_te, D_te = X[te_idx], T[te_idx], Delta[te_idx]
        t_grid = np.linspace(np.percentile(T_tr, 5), np.percentile(T_tr, 95), T_GRID_POINTS)

        S, med, sigma = run_fsa(X_tr, T_tr, D_tr, X_te, t_grid)
        results['fsa'].append(evaluate(T_tr, D_tr, T_te, D_te, S, med, t_grid))

        for bname, fn in BASELINES.items():
            S_b, med_b = fn(X_tr, T_tr, D_tr, X_te, t_grid)
            results[bname].append(evaluate(T_tr, D_tr, T_te, D_te, S_b, med_b, t_grid))

        print(f"  [{k+1}/{n_splits}] σ={sigma:.3f}  "
              f"CI(fsa)={results['fsa'][-1]['ci']:.3f}  "
              f"IBS(fsa)={results['fsa'][-1]['ibs']:.3f}")

    return results


def summarize(results):
    rows = []
    for method, metrics in results.items():
        cis  = [m['ci']  for m in metrics]
        ibss = [m['ibs'] for m in metrics if not np.isnan(m['ibs'])]
        rows.append({
            'method': method,
            'C-index': f"{np.mean(cis):.3f} ± {np.std(cis):.3f}",
            'IBS':     f"{np.mean(ibss):.3f} ± {np.std(ibss):.3f}",
        })
    return pd.DataFrame(rows).set_index('method')


if __name__ == '__main__':
    all_results = {}
    for ds in DATASETS:
        all_results[ds] = run_dataset(ds)
        print(f"\n{ds} results:")
        print(summarize(all_results[ds]).to_string())

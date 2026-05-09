"""Significance analysis for Table 1 bold/underline decisions."""
import json, numpy as np, sys
from scipy import stats

sys.path.insert(0, '..')

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, 'results/results.json')) as f: res = json.load(f)
with open(os.path.join(ROOT, 'results/results_bj.json')) as f: bj = json.load(f)

DATASETS = ['whas500', 'gbsg', 'metabric', 'support', 'flchain']
ALPHA = 0.05

# ── Assemble per-split data ───────────────────────────────────────────────────
DATA = {}
classical_map = {'Cox PH':'cox','Weibull AFT':'weibull','Log-Normal AFT':'lognormal','RSF':'rsf'}
tabpfn_map    = {'TabSA-CCA (PFN)':'fsa_tabpfn','TabSA-Bin (PFN)':'bin_fsa_tabpfn','TabSA-PO (PFN)':'pseudo_fsa_tabpfn'}
tabicl_map    = {'TabSA-CCA (ICL)':'fsa_tabicl','TabSA-Bin (ICL)':'bin_fsa_tabicl','TabSA-PO (ICL)':'pseudo_fsa_tabicl'}

for ds in DATASETS:
    for label, key in {**classical_map, **tabpfn_map, **tabicl_map}.items():
        if ds in res and key in res[ds]:
            DATA[(label,ds)] = {'ci': [x['ci'] for x in res[ds][key]],
                                 'ibs':[x['ibs'] for x in res[ds][key]]}
    for bk, label in [('tabpfn_seed0','TabSA-BJ (PFN)'),('tabicl_seed0','TabSA-BJ (ICL)')]:
        if bk in bj and ds in bj[bk]:
            DATA[(label,ds)] = {'ci': [s[-1]['CI'] for s in bj[bk][ds]],
                                 'ibs':[s[-1]['IBS'] for s in bj[bk][ds]]}

ALL_METHODS = list(classical_map) + [
    'TabSA-CCA (PFN)','TabSA-Bin (PFN)','TabSA-PO (PFN)','TabSA-BJ (PFN)',
    'TabSA-CCA (ICL)','TabSA-Bin (ICL)','TabSA-PO (ICL)','TabSA-BJ (ICL)',
]
ZERO_SHOT = [m for m in ALL_METHODS if m not in classical_map]
PFN_METHODS = [m for m in ZERO_SHOT if '(PFN)' in m]
ICL_METHODS = [m for m in ZERO_SHOT if '(ICL)' in m]

def paired_p(a, b): return stats.ttest_rel(a, b).pvalue
def sig_stars(p): return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))

def champions(methods, ds, metric, hib):
    avail = [m for m in methods if (m,ds) in DATA]
    if not avail: return set()
    scores = {m: np.mean(DATA[(m,ds)][metric]) for m in avail}
    best_m = max(scores, key=scores.__getitem__) if hib else min(scores, key=scores.__getitem__)
    return {m for m in avail if paired_p(DATA[(m,ds)][metric], DATA[(best_m,ds)][metric]) >= ALPHA} | {best_m}

bold_set = {}; ul_set = {}
for ds in DATASETS:
    for metric, hib in [('ci',True),('ibs',False)]:
        for m in champions(ALL_METHODS, ds, metric, hib): bold_set[(m,ds,metric)] = True
        for m in champions(ZERO_SHOT,   ds, metric, hib): ul_set[(m,ds,metric)]   = True

# ── 1. Table markup summary ───────────────────────────────────────────────────
print("=" * 70)
print("1. TABLE MARKUP SUMMARY  (B=bold best-overall, U=underline best-zero-shot)")
print("=" * 70)
for ds in DATASETS:
    print(f"\n── {ds.upper()} ──")
    for m in ALL_METHODS:
        if (m,ds) not in DATA: continue
        ci  = np.mean(DATA[(m,ds)]['ci']);  ibs = np.mean(DATA[(m,ds)]['ibs'])
        tc  = ('B' if bold_set.get((m,ds,'ci'))  else ' ') + ('U' if ul_set.get((m,ds,'ci'))  else ' ')
        ti  = ('B' if bold_set.get((m,ds,'ibs')) else ' ') + ('U' if ul_set.get((m,ds,'ibs')) else ' ')
        print(f"  {m:25s}  CI={ci:.3f}[{tc}]  IBS={ibs:.3f}[{ti}]")

# ── 2. Intra-TFM: TabPFN vs TabICL per method variant ────────────────────────
print("\n\n" + "=" * 70)
print("2. INTRA-TFM: paired t-test per variant  (PFN vs ICL, per dataset)")
print("=" * 70)
variants = [('TabSA-CCA (PFN)','TabSA-CCA (ICL)','CCA'),
            ('TabSA-Bin (PFN)','TabSA-Bin (ICL)','Bin'),
            ('TabSA-PO (PFN)', 'TabSA-PO (ICL)', 'PO'),
            ('TabSA-BJ (PFN)', 'TabSA-BJ (ICL)', 'BJ')]

for pfn, icl, v in variants:
    print(f"\n  {v}:")
    for ds in DATASETS:
        if (pfn,ds) not in DATA or (icl,ds) not in DATA: continue
        for metric, hib in [('ci',True),('ibs',False)]:
            a = np.array(DATA[(pfn,ds)][metric]); b = np.array(DATA[(icl,ds)][metric])
            p = paired_p(a, b); diff = np.mean(a) - np.mean(b)
            winner = ('PFN' if (diff>0)==hib else 'ICL') if p < ALPHA else '='
            print(f"    {ds:10s} {metric.upper()}: PFN={np.mean(a):.3f}  ICL={np.mean(b):.3f}"
                  f"  p={p:.4f}{sig_stars(p):3s}  better={winner}")

# ── 3. Grouped by backbone: TabPFN vs TabICL across all datasets & variants ───
print("\n\n" + "=" * 70)
print("3. BACKBONE GROUP TEST: TabPFN vs TabICL pooled across datasets & variants")
print("   (paired on matched variant×dataset, excluding TabSA-PO ICL degenerate cases)")
print("=" * 70)

DEGENERATE = {('TabSA-PO (ICL)','gbsg'), ('TabSA-PO (ICL)','metabric'), ('TabSA-PO (ICL)','flchain')}

for metric, hib in [('ci', True), ('ibs', False)]:
    pfn_pool, icl_pool = [], []
    for pfn, icl, v in variants:
        for ds in DATASETS:
            if (pfn,ds) not in DATA or (icl,ds) not in DATA: continue
            if (icl, ds) in DEGENERATE: continue
            pfn_pool.extend(DATA[(pfn,ds)][metric])
            icl_pool.extend(DATA[(icl,ds)][metric])
    p = paired_p(pfn_pool, icl_pool)
    diff = np.mean(pfn_pool) - np.mean(icl_pool)
    winner = ('PFN' if (diff>0)==hib else 'ICL') if p < ALPHA else 'no significant difference'
    print(f"\n  {metric.upper()} ({'↑' if hib else '↓'}):  PFN={np.mean(pfn_pool):.3f}  "
          f"ICL={np.mean(icl_pool):.3f}  p={p:.4f}{sig_stars(p):3s}  => {winner}")

# Also test per-dataset pooled across variants
print("\n  Per-dataset pooled across variants:")
for ds in DATASETS:
    for metric, hib in [('ci', True), ('ibs', False)]:
        pfn_pool, icl_pool = [], []
        for pfn, icl, v in variants:
            if (pfn,ds) not in DATA or (icl,ds) not in DATA: continue
            if (icl, ds) in DEGENERATE: continue
            pfn_pool.extend(DATA[(pfn,ds)][metric])
            icl_pool.extend(DATA[(icl,ds)][metric])
        if not pfn_pool: continue
        p = paired_p(pfn_pool, icl_pool)
        diff = np.mean(pfn_pool) - np.mean(icl_pool)
        winner = ('PFN' if (diff>0)==hib else 'ICL') if p < ALPHA else '='
        print(f"    {ds:10s} {metric.upper()}: PFN={np.mean(pfn_pool):.3f}  ICL={np.mean(icl_pool):.3f}"
              f"  p={p:.4f}{sig_stars(p):3s}  better={winner}")

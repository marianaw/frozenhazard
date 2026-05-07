"""Publication-quality plots for ablation and results."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns

os.makedirs("figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

DATASET_LABELS = {
    "whas500":  "WHAS500",
    "gbsg":     "GBSG",
    "metabric": "METABRIC",
    "support":  "SUPPORT",
    "flchain":  "FLCHAIN",
}

METRICS = [("ci", "C-index", "↑"), ("ibs", "IBS", "↓")]

# ── Method catalogue ──────────────────────────────────────────────────────────
# Colors: purple = ours, green = BinFSA, blue = baselines.
# Lighter shade = TabICL variant.  Hatches distinguish method type.
_C_FM       = "#9b8ec4"   # TabPFN — our methods
_C_FM_ALT   = "#c4bcd9"   # TabICL — our methods
_C_BIN      = "#6aab7e"   # TabPFN — BinFSA
_C_BIN_ALT  = "#a8cdb4"   # TabICL — BinFSA
_C_BASELINE = "#6baed6"   # baselines (no backend)

METHOD_ORDER = [
    "fsa_tabpfn",        "fsa_tabicl",
    "pseudo_fsa_tabpfn", "pseudo_fsa_tabicl",
    "fsa_bs_tabpfn",     "fsa_bs_tabicl",
    "bin_fsa_tabpfn",    "bin_fsa_tabicl",
    "cox", "weibull", "lognormal", "rsf",
]

METHOD_LABELS = {
    "fsa_tabpfn":        "FSA (PFN)",
    "fsa_tabicl":        "FSA (ICL)",
    "pseudo_fsa_tabpfn": "FSA-PO (PFN)",
    "pseudo_fsa_tabicl": "FSA-PO (ICL)",
    "fsa_bs_tabpfn":     "FSA-BS (PFN)",
    "fsa_bs_tabicl":     "FSA-BS (ICL)",
    "bin_fsa_tabpfn":    "BinFSA (PFN)",
    "bin_fsa_tabicl":    "BinFSA (ICL)",
    "cox":       "Cox PH",
    "weibull":   "Weibull AFT",
    "lognormal": "LogNorm AFT",
    "rsf":       "RSF",
}

METHOD_COLORS = {
    "fsa_tabpfn":        _C_FM,     "fsa_tabicl":        _C_FM_ALT,
    "pseudo_fsa_tabpfn": _C_FM,     "pseudo_fsa_tabicl": _C_FM_ALT,
    "fsa_bs_tabpfn":     _C_FM,     "fsa_bs_tabicl":     _C_FM_ALT,
    "bin_fsa_tabpfn":    _C_BIN,    "bin_fsa_tabicl":    _C_BIN_ALT,
    "cox":       _C_BASELINE, "weibull":   _C_BASELINE,
    "lognormal": _C_BASELINE, "rsf":       _C_BASELINE,
}

METHOD_HATCHES = {
    "fsa_tabpfn":        "",    "fsa_tabicl":        "",
    "pseudo_fsa_tabpfn": "//",  "pseudo_fsa_tabicl": "//",
    "fsa_bs_tabpfn":     "||",  "fsa_bs_tabicl":     "||",
    "bin_fsa_tabpfn":    "",    "bin_fsa_tabicl":    "",
    "cox":       "",    "weibull":   "//",
    "lognormal": "xx",  "rsf":       "..",
}

# Vertical separator after the last present method from each group.
_SEP_GROUPS = [
    {"fsa_bs_tabpfn",  "fsa_bs_tabicl"},
    {"bin_fsa_tabpfn", "bin_fsa_tabicl"},
]


def _se(vals):
    v = np.asarray(vals)
    return v.std() / np.sqrt(len(v))


def _draw_separators(ax, present):
    for group in _SEP_GROUPS:
        indices = [present.index(m) for m in group if m in present]
        if indices:
            ax.axvline(max(indices) + 1.5, color="#cccccc",
                       linewidth=0.8, linestyle="--", zorder=1)


def _load_gibbs_as_records(gibbs_path):
    """Last-iteration CI/IBS per split → flat records for fsa_bs_{backend}."""
    if not os.path.exists(gibbs_path):
        return []
    with open(gibbs_path) as f:
        store = json.load(f)
    backend = store.get("meta", {}).get("backend", "tabpfn")
    method  = f"fsa_bs_{backend}"
    records = []
    for ds, splits in store["datasets"].items():
        for split_iters in splits:
            last = split_iters[-1]
            records.append({"dataset": ds, "method": method,
                             "C-index": last["CI"], "IBS": last["IBS"]})
    return records


def _discover_gibbs_paths(results_dir="results"):
    """All gibbs result files: legacy results_gibbs.json + results_gibbs_*.json."""
    paths = []
    for fname in sorted(os.listdir(results_dir)):
        if fname == "results_gibbs.json" or \
                (fname.startswith("results_gibbs_") and fname.endswith(".json")):
            paths.append(os.path.join(results_dir, fname))
    return paths


def plot_results_boxplot(results_path="results/results.json"):
    """Box plots of C-index and IBS for all methods × datasets."""
    with open(results_path) as f:
        store = json.load(f)

    records = []
    for ds, methods in store["datasets"].items():
        for method, splits in methods.items():
            for s in splits:
                records.append({"dataset": ds, "method": method,
                                 "C-index": s["ci"], "IBS": s["ibs"]})

    results_dir = os.path.dirname(os.path.abspath(results_path))
    for gp in _discover_gibbs_paths(results_dir):
        records += _load_gibbs_as_records(gp)

    df       = pd.DataFrame(records)
    datasets = list(store["datasets"].keys())
    present  = [m for m in METHOD_ORDER if m in df["method"].unique()]

    n_ds = len(datasets)
    fig, axes = plt.subplots(2, n_ds, figsize=(2.8 * n_ds, 6), sharey="row")
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    for col, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        for row, (metric, better) in enumerate([("C-index", "↑"), ("IBS", "↓")]):
            ax = axes[row, col]

            data = [sub[sub["method"] == m][metric].dropna().values for m in present]
            bp   = ax.boxplot(
                data, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8, color="#444444"),
                capprops=dict(linewidth=0.8, color="#444444"),
                flierprops=dict(marker="o", markersize=2.5, alpha=0.4,
                                linestyle="none", markeredgecolor="#444444"),
            )
            for patch, m in zip(bp["boxes"], present):
                patch.set_facecolor(METHOD_COLORS[m])
                patch.set_hatch(METHOD_HATCHES[m])
                patch.set_linewidth(0.8)
                patch.set_alpha(0.85)

            if row == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11,
                             fontweight="bold", pad=5)
            ax.set_xticks([])
            ax.set_xlabel("")
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.tick_params(axis="y", labelsize=8)
            ax.set_ylabel(f"{metric}  {better}" if col == 0 else "", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            _draw_separators(ax, present)

    legend_handles = [
        mpatches.Patch(facecolor=METHOD_COLORS[m], hatch=METHOD_HATCHES[m],
                       edgecolor="#444444", linewidth=0.5, label=METHOD_LABELS[m],
                       alpha=0.85)
        for m in present
    ]
    axes[0, -1].legend(handles=legend_handles, fontsize=8, frameon=False,
                       loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.suptitle("C-index and IBS across datasets", fontsize=12,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("figures/results_boxplots.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("figures/results_boxplots.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved figures/results_boxplots.{pdf,png}")


def plot_k_ablation(ablation_path="results/results_ablation.json",
                    results_path="results/results.json"):
    """C-index and IBS vs K for bin_fsa, one curve per dataset.

    FSA-PO reference shown as filled stars with ±SE error bars.
    """
    if not os.path.exists(ablation_path):
        return
    with open(ablation_path) as f:
        abl = json.load(f)

    pseudo_ref = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            res = json.load(f)
        for ds, methods in res["datasets"].items():
            po_keys = [k for k in methods if k.startswith("pseudo_fsa_")]
            if po_keys:
                splits = methods[po_keys[0]]
                pseudo_ref[ds] = {
                    key: (np.mean([s[key] for s in splits if not np.isnan(s[key])]),
                          _se([s[key]  for s in splits if not np.isnan(s[key])]))
                    for key, _, _ in METRICS
                }

    datasets  = list(abl["datasets"].keys())
    palette   = sns.color_palette("tab10", n_colors=len(datasets))
    ks_all    = sorted(int(k) for k in next(iter(abl["datasets"].values())))
    xs_k      = list(range(len(ks_all)))
    x_fsa_po  = len(ks_all)
    xtick_pos = xs_k + [x_fsa_po]
    xtick_lbl = [str(k) for k in ks_all] + ["FSA-PO"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, (key, label, better) in zip(axes, METRICS):
        for ds, color in zip(datasets, palette):
            ks    = sorted(int(k) for k in abl["datasets"][ds])
            means = np.array([np.mean([s[key] for s in abl["datasets"][ds][str(k)]
                                       if not np.isnan(s[key])]) for k in ks])
            ses   = np.array([_se([s[key] for s in abl["datasets"][ds][str(k)]
                                   if not np.isnan(s[key])]) for k in ks])
            ds_label = DATASET_LABELS.get(ds, ds)

            ax.plot(xs_k, means, marker="o", markersize=4.5, linewidth=1.8,
                    color=color, label=ds_label, zorder=3)
            ax.fill_between(xs_k, means - ses, means + ses,
                            alpha=0.18, color=color, zorder=2)

            if ds in pseudo_ref:
                ref_mean, ref_se = pseudo_ref[ds][key]
                ax.errorbar(x_fsa_po, ref_mean, yerr=ref_se,
                            fmt="*", markersize=12, capsize=4, capthick=1.3,
                            linewidth=0, elinewidth=1.5,
                            color=color, markerfacecolor=color,
                            markeredgewidth=0.5, zorder=5)

        ax.axvline(x_fsa_po - 0.5, color="#cccccc", linewidth=0.8,
                   linestyle="--", zorder=1)
        ax.set_xlabel("$K$ (number of bins)", fontsize=11)
        ax.set_ylabel(f"{label}  {better}", fontsize=11)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lbl, fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(*axes[0].get_legend_handles_labels(), fontsize=8.5, frameon=False)
    fig.suptitle("bin-FSA ablation: effect of $K$", fontsize=12,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/ablation_k.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("figures/ablation_k.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved figures/ablation_k.{pdf,png}")


if __name__ == "__main__":
    plot_results_boxplot()
    plot_k_ablation()

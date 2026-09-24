"""Summarize and plot compute_capacity_use.py results.

Per model (step 1M), with compartment 0 = English and compartment 1 = the other data:
  penalty        English loss minus the c=1 model's English loss, on identical tokens (nats)
  fisher_share   F1 / (F0 + F1), F = total trunk empirical Fisher on that compartment's text
  fisher_cos     cosine(F0, F1) over trunk parameters; ceiling = English split-half cosine
  neuron_share   N1 / (N0 + N1), N = summed positive loss increase from zero-ablating each
                 MLP neuron on that compartment's text
  neuron_corr    corr(d0, d1) across MLP neurons, divided by the English split-half
                 corr(d0, d0b) (disattenuated; 1 = as similar as two English samples)
  en_mass        N0 in nats, the English necessity mass (c=1 included for reference)

Usage: python plot_capacity_use.py RESULTS_DIR OUT_PREFIX
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["EN-uniform", "EN-unigram", "EN-1.5gram", "EN-2gram", "EN-2.5gram", "EN-3gram",
         "EN-3.5gram", "EN-4gram", "EN-RU", "EN-EN"]
LABEL = {"EN-uniform": "uniform", "EN-unigram": "unigram", "EN-1.5gram": "1.5-gram",
         "EN-2gram": "2-gram", "EN-2.5gram": "2.5-gram", "EN-3gram": "3-gram",
         "EN-3.5gram": "3.5-gram", "EN-4gram": "4-gram", "EN-RU": "Russian", "EN-EN": "English"}


def metrics(r):
    A = {k: np.array(v["mlp"]).ravel() for k, v in r["ablate"].items()}
    pos = lambda x: x[x > 0].sum()
    m = {"en_loss": r["ablate"]["0"]["loss"], "en_mass": pos(A["0"]),
         "fisher_split": r["fisher_cos_split_half_en"],
         "neuron_split": float(np.corrcoef(A["0"], A["0b"])[0, 1])}
    if r["n_compartments"] > 1:
        F0, F1 = r["fisher_total"]["0"], r["fisher_total"]["1"]
        m.update(fisher_share=F1 / (F0 + F1), fisher_cos=r["fisher_cos_0_1"],
                 neuron_share=pos(A["1"]) / (pos(A["0"]) + pos(A["1"])),
                 neuron_corr=float(np.corrcoef(A["0"], A["1"])[0, 1]) / m["neuron_split"],
                 other_loss=r["ablate"]["1"]["loss"])
    return m


def main():
    rows = defaultdict(list)
    for f in sorted(Path(sys.argv[1]).glob("*.json")):
        r = json.loads(f.read_text())
        rows[r["condition"]].append(metrics(r))
    c1 = np.mean([m["en_loss"] for m in rows["c=1"]])
    for ms in rows.values():
        for m in ms:
            m["penalty"] = m["en_loss"] - c1

    keys = ["penalty", "fisher_share", "neuron_share", "fisher_cos", "neuron_corr", "en_mass", "other_loss"]
    table = {c: {k: [m[k] for m in rows[c] if k in m] for k in keys} for c in rows}
    summary = {c: {k: (float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)) for k, v in d.items() if v}
               for c, d in table.items()}
    out = Path(sys.argv[2])
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=1))
    print(f"{'condition':12s} {'n':>2s} {'penalty':>8s} {'F share':>8s} {'N share':>8s} {'F cos':>6s} {'N corr':>7s} {'EN mass':>8s} {'other L':>8s}")
    for c in ["c=1"] + ORDER:
        if c not in summary:
            continue
        s = summary[c]
        g = lambda k: f"{s[k][0]:8.3f}" if k in s else f"{'-':>8s}"
        print(f"{c:12s} {s['en_mass'][3]:2d} {g('penalty')} {g('fisher_share')} {g('neuron_share')} {g('fisher_cos')[2:]} {g('neuron_corr')[1:]} {g('en_mass')} {g('other_loss')}")
    print(f"ceilings: Fisher split-half cos {np.mean([m['fisher_split'] for ms in rows.values() for m in ms]):.3f}, "
          f"neuron split-half corr {np.mean([m['neuron_split'] for ms in rows.values() for m in ms]):.3f}")

    conds = [c for c in ORDER if c in summary]
    x = np.arange(len(conds))
    rcParams = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4}
    plt.rcParams.update(rcParams)
    panels = [("penalty", "English loss penalty vs c=1 (nats)", None),
              ("fisher_share", "other compartment's share\nof trunk Fisher", 0.5),
              ("neuron_share", "other compartment's share\nof neuron-ablation loss", 0.5),
              ("fisher_cos", "trunk Fisher overlap\n(cosine, English vs other)", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 3.0))
    for ax, (k, title, ref) in zip(axes, panels):
        mu = np.array([summary[c][k][0] for c in conds])
        lo = np.array([summary[c][k][1] for c in conds])
        hi = np.array([summary[c][k][2] for c in conds])
        ax.fill_between(x, lo, hi, color="#2a78d6", alpha=0.18, linewidth=0)
        ax.plot(x, mu, "-o", color="#2a78d6", ms=4, lw=1.4)
        if ref is not None:
            ax.axhline(ref, color="black", lw=0.6, ls=":", alpha=0.5)
        if k == "fisher_cos":
            ceil = np.mean([m["fisher_split"] for ms in rows.values() for m in ms])
            ax.axhline(ceil, color="black", lw=0.6, ls=":", alpha=0.5)
        ax.set_xticks(x, [LABEL[c] for c in conds], rotation=45, ha="right")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png", dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()

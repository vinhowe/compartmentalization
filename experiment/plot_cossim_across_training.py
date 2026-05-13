"""Cross-compartment cosine similarity across training, paired by c.

For each c ∈ {2, 4, 5, 6, 8}: solid line = InfoNCE run, dotted = no-InfoNCE
baseline. Same c-color used in plot_infonce_8_256_c1_final_gap.py so the
two charts pair visually.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style, C_COLOR


STEP_MATCH_DROP = {5}


def render(figsize, fontsize_tweak=False, step_match=False, drop_c5=False):
    cache = json.loads(Path("cossim_across_training.json").read_text())
    by_run = defaultdict(list)
    for v in cache.values():
        by_run[v["label"]].append((v["step"], v["cossim"]))
    for k in by_run:
        by_run[k].sort()

    XMIN = 100
    XMAX = None
    if step_match and "infonce_n8" in by_run:
        XMAX = max(s for s, _ in by_run["infonce_n8"])
    fig, ax = plt.subplots(figsize=figsize)

    pairs = [(2, "infonce_n2", "baseline_n2"),
             (4, "infonce_n4", "baseline_n4"),
             (5, "infonce_n5", "baseline_n5"),
             (6, "infonce_n6", "baseline_n6"),
             (8, "infonce_n8", "baseline_n8")]

    for c, lab_inf, lab_base in pairs:
        if (step_match or drop_c5) and c in STEP_MATCH_DROP:
            continue
        color = C_COLOR.get(c, "k")
        if lab_inf in by_run:
            xs, ys = zip(*by_run[lab_inf])
            xs = np.array(xs); ys = np.array(ys)
            mask = xs >= XMIN
            if XMAX is not None:
                mask &= xs <= XMAX
            if mask.any():
                ax.plot(xs[mask], ys[mask], color=color, linewidth=1.2,
                        label=f"c={c}")
        if lab_base in by_run:
            xs, ys = zip(*by_run[lab_base])
            xs = np.array(xs); ys = np.array(ys)
            mask = xs >= XMIN
            if XMAX is not None:
                mask &= xs <= XMAX
            if mask.any():
                ax.plot(xs[mask], ys[mask], color=color, linewidth=1.0,
                        linestyle=":", alpha=0.7)

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(left=XMIN, right=XMAX)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("off-diag cosine sim. (layer 4)")
    ax.yaxis.label.set_size(8)
    legend_kwargs = dict(loc="upper left", frameon=False,
                         handlelength=1.3, handletextpad=0.5)
    if fontsize_tweak:
        legend_kwargs.update(fontsize=7, columnspacing=0.6,
                             borderpad=0.2, ncol=2)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    ax.legend(**legend_kwargs)
    fig.tight_layout(pad=0.3 if fontsize_tweak else 1.08)
    return fig


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)

    fig = render(figsize=(4.5, 3.0))
    out = Path("../figures/infonce_8_256_cossim_across_training.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    fig = render(figsize=(3.5, 2.3), fontsize_tweak=False, drop_c5=True)
    out = Path("../figures/infonce_8_256_cossim_across_training_half.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    fig = render(figsize=(4.5, 3.0), step_match=True)
    out = Path("../figures/infonce_8_256_cossim_across_training_stepmatched.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    fig = render(figsize=(3.5, 2.3), fontsize_tweak=False, step_match=True)
    out = Path("../figures/infonce_8_256_cossim_across_training_stepmatched_half.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

"""Loss-plateau chart: final val loss vs n_compartments, one line per scale.

Reads val_metrics.json. For each (scale, c) entry in PANELS, takes the
last checkpoint's averaged compartment loss as the "final" val. Plots all
scales on one axis; expectation is that higher c plateaus at higher val,
and the gap is larger for smaller scales (capacity-bound).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import (
    setup_paper_style, PANELS, avg_compartment_loss, d_label,
)


D_COLORS = {
    32:   "#440154",
    64:   "#3b528b",
    128:  "#21918c",
    256:  "#5ec962",
    512:  "#fde725",
    1024: "#d62728",
}
D_MARKERS = {32: "o", 64: "s", 128: "^", 256: "D", 512: "v", 1024: "P"}


SCALE_PARAMS_M = {
    32: 1.15, 64: 2.49, 128: 5.77, 256: 14.69, 512: 41.94, 1024: 983.6,
}


def scale_label(d: int) -> str:
    p = SCALE_PARAMS_M[d]
    if p >= 1000:
        return f"{p / 1000:.2g}B"
    if p >= 100:
        return f"{p:.0f}M"
    return f"{p:.1f}M"


def plot_into(ax, metrics):
    for d, runs in PANELS:
        cs, finals = [], []
        for c, key in sorted(runs):
            if key not in metrics:
                continue
            v = metrics[key]
            steps = v["checkpoints"]
            losses = avg_compartment_loss(v["metrics"], c, len(steps))
            if not losses:
                continue
            cs.append(c)
            finals.append(losses[-1])
        if not cs:
            continue
        ax.plot(cs, finals, color=D_COLORS[d], marker=D_MARKERS[d],
                markersize=5, label=scale_label(d))
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
    ax.set_xlabel("c")
    ax.set_ylabel("Final val loss (nats)")


def main():
    setup_paper_style()
    metrics = json.loads(Path("val_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    plot_into(ax, metrics)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, handlelength=1.3, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = Path("../figures/loss_plateau.pdf")
    fig.savefig(out)
    print(f"  {out}")
    plt.close(fig)

    # Compact (1/3-textwidth ready) variant for triptych use. No room in-axis
    # for a 6-entry legend (the high-c values smush against the lines), so
    # legend lives below the chart.
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    plot_into(ax, metrics)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, handlelength=1.0,
               handletextpad=0.3, columnspacing=0.6,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    out = Path("../figures/loss_plateau_compact.pdf")
    fig.savefig(out)
    print(f"  {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

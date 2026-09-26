"""Validation loss curves for the baseline compartmented models — 6 panels.

One panel per model size (d_model in {32, 64, 128, 256, 512, 1024}).
Each panel shows curves for each available c (n_compartments).

All runs are rope-era. c>1 runs use tr=0.1 compartment-mode; c=1 and 1B (c=2,8)
are tr=0. All effective tr near 0 — treated as the compartment baseline series.
Sources: synthetic-compartment-baselines (c=1, d≤256), bpe16384-rope-8-256
(c≥2, d=256), bpe16384-rope-small-scale-tr01-epoch (d∈{32,64,128} c≥2),
bpe16384-rope-8-512-sweep (d=512), 1b-scale (1B). 1B only has c∈{1,2,8}.

Reads val_metrics.json. Averages val loss across compartments.
Paper-ready vector PDF, log-log axes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.pyplot as _plt

from _run_paths import (
    C1_BASELINE_BY_SCALE, RUNS_SMALL_SCALE_TR01, RUNS_N3_ROPE,
    NO_INFONCE_8_256_BY_C, RUN_8_256_C3, RUNS_8_512_LEGACY_BY_C,
    RUN_1B_C1_BASELINE, RUN_1B_C2_NOTRANS, RUN_1B_C8_NOTRANS,
)


# Four-across layout (Fig. 4 with a fourth panel). The three-across panels are
# 2.4in wide in a 0.32\textwidth subfigure, i.e. drawn at 0.733 scale on the
# page. A 0.245\textwidth subfigure at the same scale is 1.84in wide, so these
# panels keep the same height and the same on-page font size, only narrower.
# Fixed margins (not tight_layout) so all four axes boxes are identical.
FOURUP_FIGSIZE = (1.84, 2.0)
FOURUP_ADJUST = dict(left=0.30, right=0.96, bottom=0.22, top=0.97)

# 2x2 layout: each panel in a 0.49\textwidth subfigure, 3.3in wide like the
# paper's other half-width panels (capacity sharing, multilingual), so text
# prints at the same size as theirs. Flattened to 1.75in tall (they are 2.3in)
# so the grid costs less page; the bottom margin is just the tick labels plus
# the x-label. Fixed margins so the four axes align.
GRID2_FIGSIZE = (3.3, 1.75)
GRID2_ADJUST = dict(left=0.16, right=0.97, bottom=0.25, top=0.97)


def setup_paper_style():
    rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "lines.linewidth": 1.3, "lines.markersize": 0,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4,
        "axes.grid": True, "grid.alpha": 0.3,
    })


# (d_model, [(c, key), ...])
def _panel_small(d, n3_key):
    return [
        (1, C1_BASELINE_BY_SCALE[d]),
        (2, RUNS_SMALL_SCALE_TR01[(d, 2)]),
        (3, n3_key),
        (4, RUNS_SMALL_SCALE_TR01[(d, 4)]),
        (5, RUNS_SMALL_SCALE_TR01[(d, 5)]),
        (6, RUNS_SMALL_SCALE_TR01[(d, 6)]),
        (8, RUNS_SMALL_SCALE_TR01[(d, 8)]),
    ]


PANELS = [
    (32,  _panel_small(32,  RUNS_N3_ROPE[2])),
    (64,  _panel_small(64,  RUNS_N3_ROPE[1])),
    (128, _panel_small(128, RUNS_N3_ROPE[0])),
    (256, [
        (1, C1_BASELINE_BY_SCALE[256]),
        (2, NO_INFONCE_8_256_BY_C[2]),
        (3, RUN_8_256_C3),
        (4, NO_INFONCE_8_256_BY_C[4]),
        (5, NO_INFONCE_8_256_BY_C[5]),
        (6, NO_INFONCE_8_256_BY_C[6]),
        (8, NO_INFONCE_8_256_BY_C[8]),
    ]),
    (512, [(c, RUNS_8_512_LEGACY_BY_C[c]) for c in (1, 2, 3, 4, 5, 6, 8)]),
    (1024, [
        (1, RUN_1B_C1_BASELINE),
        (2, RUN_1B_C2_NOTRANS),
        (8, RUN_1B_C8_NOTRANS),
    ]),
]

# Color palette indexed by c, shared across panels.
ALL_C = [1, 2, 3, 4, 5, 6, 8]
CMAP = _plt.get_cmap("viridis")
C_COLOR = {c: CMAP(i / (len(ALL_C) - 1)) for i, c in enumerate(ALL_C)}


def avg_compartment_loss(metric_dict: dict, n_compartments: int, n_steps: int) -> list[float]:
    arrs = []
    for c in range(n_compartments):
        k = f"loss_compartment_{c}"
        if k in metric_dict and len(metric_dict[k]) == n_steps:
            arrs.append(metric_dict[k])
    if not arrs:
        return []
    return [sum(a[i] for a in arrs) / len(arrs) for i in range(n_steps)]


def d_label(d: int) -> str:
    return "1B" if d == 1024 else f"d={d}"


def main():
    setup_paper_style()
    metrics = json.loads(Path("val_metrics.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True)
    flat = axes.flatten()
    for i, (d, runs) in enumerate(PANELS):
        ax = flat[i]
        for c, key in runs:
            if key not in metrics:
                print(f"  {d_label(d)} c={c}: missing metrics for {key}")
                continue
            v = metrics[key]
            steps = v["checkpoints"]
            losses = avg_compartment_loss(v["metrics"], c, len(steps))
            if not losses:
                print(f"  {d_label(d)} c={c}: no compartment losses")
                continue
            ax.plot(steps, losses, color=C_COLOR[c], label=f"c={c}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(d_label(d))
        if i // 3 == 1:
            ax.set_xlabel("Step")
        if i % 3 == 0:
            ax.set_ylabel("Val loss (nats)")
        ax.legend(loc="upper right", frameon=False, ncol=2,
                  handlelength=1.3, handletextpad=0.5,
                  columnspacing=0.8, labelspacing=0.25)
    fig.suptitle("Compartment-baseline val loss vs step (BPE-16384, fineweb)", y=1.0)
    fig.tight_layout()
    out = Path("../figures/baseline_val_curves.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  {out}")


if __name__ == "__main__":
    main()

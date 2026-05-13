"""Trajectory view for the finetune-from-c=1 experiments.

X-axis: finetune step (0 to ~30k, 1k cadence). Y-axis: val loss averaged over
compartments. One solid line per c (2, 4, 8). The 1M-step from-scratch
baselines for c=1, c=2, c=4, c=8 appear as horizontal dotted reference lines,
labeled at the right edge of the chart.

The point: each finetune trajectory dives below its from-scratch counterpart
within the first few thousand steps and lands at (or below) the c=1 floor —
the entire 1M-step compartmentation tax recouped in <1% of the training budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style


N1_BASELINE_KEY = (
    "synthetic-compartment-baselines/"
    "2026-03-06T18-11-45Z__english-baseline-rope-bpe16384-8-256"
    "__2df56182__s64__4b68526__51c738c2"
)

# (c, scratch_baseline_key, ft_metrics_key, color, marker)
ROWS = [
    (2, "bpe16384-rope-8-256/217ca694_s64", "ce-full-2comp-rope", "tab:blue",  "o"),
    (4, "bpe16384-rope-8-256/53e73c3d_s64", "ce-full-4comp-rope", "tab:green", "s"),
    (8, "bpe16384-rope-8-256/868ef4a8_s64", "ce-full-8comp-rope", "tab:red",   "^"),
]


def avg_compartment_arr(metrics: dict, c: int, n_steps: int) -> np.ndarray:
    arrs = [np.array(metrics[f"loss_compartment_{i}"][:n_steps]) for i in range(c)]
    return np.mean(arrs, axis=0)


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)
    main_metrics = json.loads(Path("fineweb_val_metrics.json").read_text())
    ft = json.loads(Path("finetune_val_metrics.json").read_text())

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    # Finetune trajectories — drive the x-range.
    rightmost = 1_000
    for c, _, ft_key, color, marker in ROWS:
        v = ft.get(ft_key)
        if not v or not v["checkpoints"]:
            continue
        steps = np.array(v["checkpoints"], dtype=float)
        losses = avg_compartment_arr(v["metrics"], c, len(steps))
        ax.plot(steps, losses, color=color, marker=marker, markersize=3,
                linewidth=1.4, label=f"c={c}")
        rightmost = max(rightmost, float(steps.max()))

    # From-scratch reference levels: horizontal dotted lines + right-edge labels.
    n1_final = main_metrics[N1_BASELINE_KEY]["metrics"]["loss_compartment_0"][-1]
    refs = [(1, n1_final, "tab:gray")]
    for c, scratch_key, _, color, _ in ROWS:
        if scratch_key in main_metrics:
            v = main_metrics[scratch_key]["metrics"]
            refs.append((c, float(np.mean([v[f"loss_compartment_{i}"][-1]
                                           for i in range(c)])), color))
    for c, val, color in refs:
        ax.axhline(val, color=color, linewidth=0.7, alpha=0.55, linestyle=":")
        ax.text(rightmost * 1.005, val, f"c={c}", color=color, alpha=0.9,
                fontsize=7, va="center", ha="left")

    ax.set_xlim(0, rightmost * 1.08)  # leave room for the right-edge labels
    ax.set_xlabel("Finetune step (after baseline-end)")
    ax.set_ylabel("Val loss (avg over compartments, nats)")
    ax.legend(loc="upper right", frameon=False, title="ft from c=1",
              title_fontsize=7.5, handlelength=1.4, handletextpad=0.5,
              borderpad=0.3)
    fig.tight_layout()
    out = Path("../figures/finetune_offset_trajectory.pdf")
    fig.savefig(out)
    print(f"  {out}")


if __name__ == "__main__":
    main()

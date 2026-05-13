"""Visual explanation of the slowdown metric used in the compartmentation
section. Shows two val-loss-vs-step trajectories at the 8-512 scale (c=1
baseline and c=8 compartmented), with horizontal/vertical guides indicating
how slowdown is measured at a representative val-loss target.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import (
    setup_paper_style, C_COLOR, avg_compartment_loss,
)
from plot_compartmented_slowdown import interp_iter_at_loss
from _run_paths import RUNS_8_512_LEGACY_BY_C


KEY_C1 = RUNS_8_512_LEGACY_BY_C[1]
KEY_C8 = RUNS_8_512_LEGACY_BY_C[8]
TARGET_VAL = 4.5  # representative val in overlap of c=1 [3.79, 8.91] & c=8 [4.18, 11.1]


def get_curve(metrics, key, c):
    v = metrics[key]
    s = np.array(v["checkpoints"], dtype=float)
    losses = avg_compartment_loss(v["metrics"], c, len(s))
    l = np.array(losses, dtype=float)
    order = np.argsort(s)
    return s[order], l[order]


def main():
    setup_paper_style()
    metrics = json.loads(Path("val_metrics.json").read_text())
    Path("../figures").mkdir(exist_ok=True)

    s_base, l_base = get_curve(metrics, KEY_C1, 1)
    s_comp, l_comp = get_curve(metrics, KEY_C8, 8)

    s_base_at = interp_iter_at_loss(TARGET_VAL, s_base, l_base)
    s_comp_at = interp_iter_at_loss(TARGET_VAL, s_comp, l_comp)
    slowdown = s_comp_at / s_base_at

    # Compact (1/3-textwidth ready): minimal annotations. Two curves, one
    # horizontal guide at the matched-val target, one slowdown arrow with the
    # ratio inline. Inline labels for the curves replace the legend; step
    # numbers and vertical drops are intentionally omitted to declutter.
    # Override C_COLOR for this panel only — its viridis-extreme c=8 yellow
    # is hard to read; use a high-contrast pair instead.
    BASE_COLOR = "tab:gray"
    COMP_COLOR = "tab:red"
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    ax.plot(s_base, l_base, color=BASE_COLOR, linewidth=1.4)
    ax.plot(s_comp, l_comp, color=COMP_COLOR, linewidth=1.4)

    # Common bbox: white halo so text reads cleanly over any line/arrow.
    bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.9)

    # Curve labels: each placed in the open quadrant relative to its curve
    # endpoint, with enough offset to clear the trajectory line.
    ax.text(s_base[-1], l_base[-1] + 0.09, "c=1", color=BASE_COLOR,
            ha="right", va="bottom", fontsize=8, fontweight="bold", bbox=bbox)
    ax.text(s_comp[-1], l_comp[-1] + 0.13, "c=8", color=COMP_COLOR,
            ha="right", va="bottom", fontsize=8, fontweight="bold", bbox=bbox)

    # Light dashed horizontal guide at target val.
    ax.axhline(TARGET_VAL, color="black", linewidth=0.5, alpha=0.35,
               linestyle="--")

    # Slowdown arrow at the target val.
    ax.annotate(
        "", xy=(s_comp_at, TARGET_VAL), xytext=(s_base_at, TARGET_VAL),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )
    # Slowdown ratio: above the arrow (open space — both curves are
    # descending past this region from above).
    mid_x = float(np.exp(0.5 * (np.log(s_base_at) + np.log(s_comp_at))))
    # tiny up-and-left nudge from the prior "nearly perfect" (mid_x*0.78, +0.02).
    label_x = mid_x * 0.72
    ax.text(label_x, TARGET_VAL + 0.05, f"{slowdown:.1f}×",
            ha="center", va="bottom", fontsize=9, fontweight="bold", bbox=bbox)

    ax.set_xscale("log")
    # Crop y to the convergence band; the early random-init descent is
    # cosmetic and dragged the label/arrow placement down.
    YMIN, YMAX = 3.7, 5.0
    ax.set_ylim(YMIN, YMAX)
    # Then crop x to where the data is actually inside the y-band, so the
    # first half of the panel isn't empty.
    base_in = s_base[(l_base >= YMIN) & (l_base <= YMAX)]
    comp_in = s_comp[(l_comp >= YMIN) & (l_comp <= YMAX)]
    if base_in.size and comp_in.size:
        x_left = min(base_in.min(), comp_in.min()) * 0.85
        x_right = max(base_in.max(), comp_in.max()) * 1.15
        ax.set_xlim(x_left, x_right)
    ax.set_xlabel("step")
    ax.set_ylabel("val loss (nats)")
    fig.tight_layout(pad=0.3)
    out = Path("../figures/slowdown_explainer.pdf")
    fig.savefig(out)
    print(f"  {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

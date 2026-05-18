"""Sample-efficiency / capacity as inverse: slowdown of compartmented models
relative to the c=1 (rope, no-comp) baseline at each scale.

For each (scale, c=N>1) pair:
  - Pull both trajectories of (iter, val_loss) from val_metrics.json.
  - For every original checkpoint on either model whose val_loss falls in the
    overlap of realized val_loss ranges, linearly interpolate the matching
    iter on the other model and compute slowdown = iter_compartmented /
    iter_baseline.
  - Plot each slowdown point at its baseline-iter x-coordinate.

Outputs:
  - ../figures/slowdown_compartmented_all.pdf (6 panels: 8-32, 8-64,
    8-128, 8-256, 8-512, 1B)
  - ../figures/slowdown_compartmented_8_256.pdf
  - ../figures/slowdown_compartmented_8_512.pdf
  - ../figures/slowdown_compartmented_1b.pdf

We do not label translation_ratio anywhere — these are simply the
"worst-case compartmented" curves.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import (
    setup_paper_style, PANELS, C_COLOR, avg_compartment_loss, d_label,
)
from _run_paths import filter_to_loggy


def get_curve(metrics, key, c) -> tuple[np.ndarray, np.ndarray]:
    if key not in metrics:
        return np.array([]), np.array([])
    v = metrics[key]
    steps = v["checkpoints"]
    losses = avg_compartment_loss(v["metrics"], c, len(steps))
    if not losses:
        return np.array([]), np.array([])
    s = np.array(steps, dtype=float)
    l = np.array(losses, dtype=float)
    # Sort by step (should already be) and ensure monotone-decreasing val
    # (clip non-monotone tail by enforcing running min so interp is sensible).
    order = np.argsort(s)
    return filter_to_loggy(s[order], l[order])


def interp_iter_at_loss(target_loss: float, steps: np.ndarray, losses: np.ndarray):
    """Return interpolated iter where val_loss == target_loss, or None if
    target_loss is outside the realized range. Assumes losses is monotone
    decreasing in step (we sort/check)."""
    if len(steps) < 2:
        return None
    # Make sure ordered by step, losses approximately monotonic.
    # Find adjacent pair where losses bracket target.
    for i in range(len(losses) - 1):
        a, b = losses[i], losses[i + 1]
        sa, sb = steps[i], steps[i + 1]
        # target between a and b (whichever order)
        lo, hi = min(a, b), max(a, b)
        if lo <= target_loss <= hi:
            if a == b:
                return float(0.5 * (sa + sb))
            alpha = (a - target_loss) / (a - b)
            return float(sa + alpha * (sb - sa))
    return None


def slowdown_points(s_base: np.ndarray, l_base: np.ndarray,
                    s_comp: np.ndarray, l_comp: np.ndarray, *, x_axis="step"):
    """One slowdown point per *target* (c=N) checkpoint whose val falls in the
    overlap of realized val ranges. Interpolates the baseline curve linearly
    to find iter at matching val. Slowdown = comp_iter / interpolated_base_iter.

    x_axis="step": plot at the interpolated baseline-iter x-coordinate.
    x_axis="val":  plot at the val-loss x-coordinate (the value at which the
                   slowdown is measured)."""
    base_l_min, base_l_max = min(l_base), max(l_base)
    comp_l_min, comp_l_max = min(l_comp), max(l_comp)
    overlap_lo = max(base_l_min, comp_l_min)
    overlap_hi = min(base_l_max, comp_l_max)
    pts = []
    for sc, lc in zip(s_comp, l_comp):
        if not (overlap_lo <= lc <= overlap_hi):
            continue
        ib = interp_iter_at_loss(float(lc), s_base, l_base)
        if ib is None or ib <= 0:
            continue
        x = float(lc) if x_axis == "val" else float(ib)
        pts.append((x, float(sc) / float(ib)))
    pts.sort()
    if not pts:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def plot_panel(ax, metrics, panel, *, show_legend=True, show_title=True, x_axis="step"):
    d, runs = panel
    rd = dict(runs)
    if 1 not in rd:
        ax.text(0.5, 0.5, "no c=1 baseline", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(d_label(d))
        return
    s_base, l_base = get_curve(metrics, rd[1], 1)
    if s_base.size == 0:
        ax.set_title(d_label(d))
        return
    for c, key in runs:
        if c == 1:
            continue
        s_comp, l_comp = get_curve(metrics, key, c)
        if s_comp.size == 0:
            continue
        xs, ys = slowdown_points(s_base, l_base, s_comp, l_comp, x_axis=x_axis)
        if xs.size == 0:
            continue
        ax.plot(xs, ys, color=C_COLOR[c], marker="o", markersize=2.5,
                linewidth=1.0, label=f"c={c}")
    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    if x_axis == "step":
        ax.set_xscale("log")
    if x_axis == "val":
        # Val loss decreases over training; reverse so improvement reads left→right.
        ax.invert_xaxis()
    if show_title:
        ax.set_title(d_label(d))
    if show_legend:
        ax.legend(loc="upper left", frameon=False, ncol=2,
                  handlelength=1.3, handletextpad=0.5,
                  columnspacing=0.8, labelspacing=0.25)


def fig_all(metrics, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.4), sharex=False, sharey=False)
    flat = axes.flatten()
    for i, panel in enumerate(PANELS):
        ax = flat[i]
        plot_panel(ax, metrics, panel)
        if i // 3 == 1:
            ax.set_xlabel("c=1 baseline step")
        if i % 3 == 0:
            ax.set_ylabel("Slowdown (×)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  {out_path}")


def fig_single(metrics, panel, out_path: Path, *, x_axis="step"):
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    plot_panel(ax, metrics, panel, show_legend=False, show_title=False, x_axis=x_axis)
    ax.set_xlabel("Val loss (nats)" if x_axis == "val" else "c=1 baseline step")
    ax.set_ylabel("Slowdown (×)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, handlelength=1.3, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
               markerscale=2.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out_path)
    print(f"  {out_path}")


def fig_compact(metrics, panel, out_path: Path, *, x_axis="val"):
    """1/3-textwidth ready compact variant: smaller fonts, in-axis legend."""
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    plot_panel(ax, metrics, panel, show_legend=False, show_title=False, x_axis=x_axis)
    ax.set_xlabel("val loss (nats)" if x_axis == "val" else "step")
    ax.set_ylabel("slowdown (×)")
    ax.legend(loc="upper left", frameon=False,
              handlelength=1.0, handletextpad=0.3, ncol=2,
              columnspacing=0.5, borderpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    print(f"  {out_path}")
    plt.close(fig)


def main():
    setup_paper_style()
    metrics = json.loads(Path("val_metrics.json").read_text())
    Path("../figures").mkdir(exist_ok=True)
    fig_all(metrics, Path("../figures/slowdown_compartmented_all.pdf"))
    # Last 3 scales as separate single-panel PDFs.
    by_d = {p[0]: p for p in PANELS}
    fig_single(metrics, by_d[256], Path("../figures/slowdown_compartmented_8_256.pdf"))
    fig_single(metrics, by_d[512], Path("../figures/slowdown_compartmented_8_512.pdf"),
               x_axis="val")
    fig_single(metrics, by_d[1024], Path("../figures/slowdown_compartmented_1b.pdf"))
    # Compact (1/3-textwidth) variant of 8-512 for the triptych.
    fig_compact(metrics, by_d[512], Path("../figures/slowdown_compartmented_8_512_compact.pdf"),
                x_axis="val")


if __name__ == "__main__":
    main()

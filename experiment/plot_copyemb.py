"""copyemb existence-proof figures.

Body: copyemb_8_256.pdf — single panel, val loss over training, 5 lines:
  - c=1 baseline (English, single compartment)
  - c=2 random init
  - c=2 copyemb (init copies compartment-0 embeddings to compartment-1)
  - c=8 random init
  - c=8 copyemb

Appendix: copyemb_scaling.pdf — final val vs scale, 3 lines:
  - c=1 baseline
  - c=2 random init
  - c=2 copyemb

All 8-256-class rope, V=16384, 1M iters, wd=0.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style, avg_compartment_loss
from plot_compartmented_slowdown import slowdown_points
from _run_paths import (
    C1_BASELINE_8_256, C1_BASELINE_BY_SCALE,
    NO_INFONCE_8_256_BY_C, COPYEMB_8_256_BY_C, COPYEMB_C2_BY_SCALE,
    RUNS_SMALL_SCALE_TR01,
)


KEY_C1_8_256 = C1_BASELINE_8_256
KEY_C2_RAND_8_256 = NO_INFONCE_8_256_BY_C[2]
KEY_C2_COPYEMB_8_256 = COPYEMB_8_256_BY_C[2]
KEY_C8_RAND_8_256 = NO_INFONCE_8_256_BY_C[8]
KEY_C8_COPYEMB_8_256 = COPYEMB_8_256_BY_C[8]

def _c2_rand_for_scale(d):
    return NO_INFONCE_8_256_BY_C[2] if d == 256 else RUNS_SMALL_SCALE_TR01[(d, 2)]


# Scaling: c=2 copyemb at d ∈ {32, 64, 128, 256} + c=1 baseline + c=2 random
SCALING = [
    (d, C1_BASELINE_BY_SCALE[d], _c2_rand_for_scale(d), COPYEMB_C2_BY_SCALE[d])
    for d in (32, 64, 128, 256)
]
SCALE_PARAMS_M = {32: 1.15, 64: 2.49, 128: 5.77, 256: 14.69}


def get_curve(metrics, key, c):
    if key not in metrics:
        return np.array([]), np.array([])
    v = metrics[key]
    s = np.array(v["checkpoints"], dtype=float)
    losses = avg_compartment_loss(v["metrics"], c, len(s))
    if not losses:
        return np.array([]), np.array([])
    l = np.array(losses, dtype=float)
    order = np.argsort(s)
    return s[order], l[order]


def fig_body(metrics):
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    LINES = [
        ("c=1 baseline",  KEY_C1_8_256,         1, "tab:gray",   "x", "--"),
        ("c=2 random",    KEY_C2_RAND_8_256,    2, "tab:blue",   "o", "-"),
        ("c=2 copyemb",   KEY_C2_COPYEMB_8_256, 2, "tab:cyan",   "s", "-"),
        ("c=8 random",    KEY_C8_RAND_8_256,    8, "tab:red",    "^", "-"),
        ("c=8 copyemb",   KEY_C8_COPYEMB_8_256, 8, "tab:orange", "D", "-"),
    ]
    for label, key, c, color, marker, ls in LINES:
        s, l = get_curve(metrics, key, c)
        if s.size == 0:
            continue
        ax.plot(s, l, color=color, marker=marker, markersize=3,
                linewidth=1.0, linestyle=ls, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Val loss (nats)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, handlelength=1.5, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
               markerscale=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = Path("../figures/copyemb_8_256.pdf")
    fig.savefig(out)
    print(f"  {out}")


def _plot_slowdown_lines(ax, metrics, lines):
    s_base, l_base = get_curve(metrics, KEY_C1_8_256, 1)
    for label, key, c, color, marker in lines:
        s_comp, l_comp = get_curve(metrics, key, c)
        if s_comp.size == 0:
            continue
        xs, ys = slowdown_points(s_base, l_base, s_comp, l_comp, x_axis="val")
        if xs.size == 0:
            continue
        ax.plot(xs, ys, color=color, marker=marker, markersize=3,
                linewidth=1.0, label=label)
    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ax.invert_xaxis()
    ax.set_xlabel("Val loss (nats)")
    ax.set_ylabel("Slowdown vs c=1 (×)")


def fig_slowdown(metrics):
    """Option C: single panel, color = init, lightness = c, all solid."""
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    LINES = [
        ("c=2 default init", KEY_C2_RAND_8_256,    2, "#9ecae1", "o"),  # light blue
        ("c=2 init copying", KEY_C2_COPYEMB_8_256, 2, "#1f77b4", "s"),  # dark blue
        ("c=8 default init", KEY_C8_RAND_8_256,    8, "#fcae91", "^"),  # light red
        ("c=8 init copying", KEY_C8_COPYEMB_8_256, 8, "#d62728", "D"),  # dark red
    ]
    _plot_slowdown_lines(ax, metrics, LINES)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=False, handlelength=1.5, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
               markerscale=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = Path("../figures/copyemb_slowdown_8_256.pdf")
    fig.savefig(out)
    print(f"  {out}")


def fig_slowdown_c8(metrics):
    """Option A: just c=8 (2 lines), maximally legible."""
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    LINES = [
        ("default init", KEY_C8_RAND_8_256,    8, "tab:red",  "^"),
        ("init copying", KEY_C8_COPYEMB_8_256, 8, "tab:blue", "D"),
    ]
    _plot_slowdown_lines(ax, metrics, LINES)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=False, handlelength=1.5, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
               markerscale=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = Path("../figures/copyemb_slowdown_c8_only.pdf")
    fig.savefig(out)
    print(f"  {out}")


def fig_scaling(metrics):
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    xs = []
    pts = {"c=1 baseline": [], "c=2 default init": [], "c=2 init copying": []}
    for d, c1_key, c2_rand_key, c2_copyemb_key in SCALING:
        params_m = SCALE_PARAMS_M[d]
        xs.append(params_m)
        for label, key, c in [
            ("c=1 baseline",     c1_key, 1),
            ("c=2 default init", c2_rand_key, 2),
            ("c=2 init copying", c2_copyemb_key, 2),
        ]:
            s, l = get_curve(metrics, key, c)
            pts[label].append(l[-1] if l.size else None)
    color_marker = {
        "c=1 baseline":      ("tab:gray", "x"),
        "c=2 default init":  ("#9ecae1", "o"),  # light blue
        "c=2 init copying":  ("#1f77b4", "s"),  # dark blue
    }
    for label, ys in pts.items():
        c, m = color_marker[label]
        ax.plot(xs, ys, color=c, marker=m, markersize=5, linewidth=1.0,
                label=label)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.1f}M" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("Model size")
    ax.set_ylabel("Final val loss (nats)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, handlelength=1.5, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
               markerscale=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = Path("../figures/copyemb_scaling.pdf")
    fig.savefig(out)
    print(f"  {out}")


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)
    metrics = json.loads(Path("val_metrics.json").read_text())
    fig_body(metrics)
    fig_slowdown(metrics)
    fig_slowdown_c8(metrics)
    fig_scaling(metrics)


if __name__ == "__main__":
    main()

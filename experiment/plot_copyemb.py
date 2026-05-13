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


# 8-256 keys for the main body chart
KEY_C1_8_256 = (
    "synthetic-compartment-baselines/"
    "2026-03-06T18-11-45Z__english-baseline-rope-bpe16384-8-256"
    "__2df56182__s64__4b68526__51c738c2"
)
KEY_C2_RAND_8_256 = "bpe16384-rope-8-256/217ca694_s64"
KEY_C2_COPYEMB_8_256 = (
    "synthetic-compartment-baselines/"
    "2026-03-11T06-39-01Z__english-copyemb-2comp-rope-bpe16384-8-256"
    "__55197561__s64__4b68526__09338b7c"
)
KEY_C8_RAND_8_256 = "bpe16384-rope-8-256/868ef4a8_s64"
KEY_C8_COPYEMB_8_256 = (
    "synthetic-compartment-baselines/"
    "2026-03-11T23-33-54Z__english-copyemb-8comp-rope-bpe16384-8-256"
    "__28f1b316__s64__4b68526__8fa7f919"
)

# Scaling: c=2 copyemb at d ∈ {32, 64, 128, 256} + c=1 baseline + c=2 random
SCALING = [
    (32,  87.2 * 0.013,  # placeholder, we'll use proper params per scale
        ("synthetic-compartment-baselines/"
         "2026-03-06T18-19-13Z__english-baseline-rope-bpe16384-8-32"
         "__41ba658f__s64__4b68526__66b66981"),
        "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-14Z__8-32-n2-tr01__276e6011__s64__fd9c538__fc9992cd",
        ("synthetic-compartment-baselines/"
         "2026-03-11T06-39-22Z__english-copyemb-2comp-rope-bpe16384-8-32"
         "__e362045d__s64__4b68526__55aa6f95"),
    ),
    (64, 0,
        ("synthetic-compartment-baselines/"
         "2026-03-06T18-18-46Z__english-baseline-rope-bpe16384-8-64"
         "__50fa3055__s64__4b68526__007674d4"),
        "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-64-n2-tr01__a654d23e__s64__fd9c538__4bc16fd9",
        ("synthetic-compartment-baselines/"
         "2026-03-11T06-39-12Z__english-copyemb-2comp-rope-bpe16384-8-64"
         "__c148c701__s64__4b68526__675100fa"),
    ),
    (128, 0,
        ("synthetic-compartment-baselines/"
         "2026-03-06T18-17-16Z__english-baseline-rope-bpe16384-8-128"
         "__bafaffdf__s64__4b68526__e939a660"),
        "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n2-tr01__29620da8__s64__fd9c538__e33f2900",
        ("synthetic-compartment-baselines/"
         "2026-03-11T06-39-05Z__english-copyemb-2comp-rope-bpe16384-8-128"
         "__2510e3fb__s64__4b68526__a830bae0"),
    ),
    (256, 0, KEY_C1_8_256, KEY_C2_RAND_8_256, KEY_C2_COPYEMB_8_256),
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
    for d, _, c1_key, c2_rand_key, c2_copyemb_key in SCALING:
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
    metrics = json.loads(Path("fineweb_val_metrics.json").read_text())
    fig_body(metrics)
    fig_slowdown(metrics)
    fig_slowdown_c8(metrics)
    fig_scaling(metrics)


if __name__ == "__main__":
    main()

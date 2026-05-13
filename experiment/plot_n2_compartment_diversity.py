"""n=2 rope, 8-256: how do different "other-compartment" contents affect English
learning? Compares to the c=1 English baseline.

Conditions (all n_compartments=2, tr=0, mode=compartment, rope, 8-256, 1M iters):
  - english-english (copyemb): both compartments see English (homogeneous)
  - english-russian:           c0 = English, c1 = Russian
  - english-uniform-noise:     c0 = English, c1 = synthetic uniform-random tokens
  - english-frequency-noise:   c0 = English, c1 = synthetic unigram-frequency tokens
  - c=1 baseline:              vanilla single-compartment English (reference)

For all conditions we plot ONLY compartment 0's (English) val loss, so the
comparison is apples-to-apples on the English task.

Outputs:
  ../figures/capacity_sharing_val.pdf       — val loss vs step
  ../figures/capacity_sharing_slowdown.pdf  — slowdown vs c=1 baseline (val x)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style
from plot_compartmented_slowdown import (
    interp_iter_at_loss, slowdown_points,
)
from _run_paths import C1_BASELINE_8_256, NO_INFONCE_8_256_BY_C, N2_DIVERSITY_RUNS


N1_BASELINE_KEY = C1_BASELINE_8_256

# (label, key_or_keys, color, marker)
CONDITIONS = [
    ("EN-EN",      [NO_INFONCE_8_256_BY_C[2]],          "tab:blue",   "o"),
    ("EN-RU",      [N2_DIVERSITY_RUNS["EN-RU"]],        "tab:orange", "s"),
    ("EN-unigram", [N2_DIVERSITY_RUNS["EN-unigram"]],   "tab:green",  "^"),
    ("EN-uniform", [N2_DIVERSITY_RUNS["EN-uniform"]],   "tab:red",    "v"),
]


def get_english_curve(metrics, key_or_keys):
    """Return (steps, loss_compartment_0) — English-side loss only.
    If key_or_keys is a list, average loss across runs at common steps."""
    keys = key_or_keys if isinstance(key_or_keys, list) else [key_or_keys]
    per_run = []
    for key in keys:
        if key not in metrics:
            continue
        v = metrics[key]
        s = np.array(v["checkpoints"], dtype=float)
        l = np.array(v["metrics"]["loss_compartment_0"], dtype=float)
        n = min(len(s), len(l))
        order = np.argsort(s[:n])
        per_run.append(dict(zip(s[:n][order].tolist(), l[:n][order].tolist())))
    if not per_run:
        return np.array([]), np.array([])
    common = sorted(set.intersection(*[set(d.keys()) for d in per_run]))
    if not common:
        return np.array([]), np.array([])
    means = [float(np.mean([d[s] for d in per_run])) for s in common]
    return np.array(common), np.array(means)


def plot_val_into(ax, metrics):
    s_n1, l_n1 = get_english_curve(metrics, N1_BASELINE_KEY)
    if s_n1.size:
        ax.plot(s_n1, l_n1, color="tab:gray", marker="x", markersize=3,
                linestyle="--", linewidth=1.0, label="c=1")
    for label, key, color, marker in CONDITIONS:
        s, l = get_english_curve(metrics, key)
        if s.size:
            ax.plot(s, l, color=color, marker=marker, markersize=3,
                    linewidth=1.2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=10**3.5)
    ax.set_ylim(top=6)
    ax.set_xlabel("Step")
    ax.set_ylabel("English val loss (nats)")


def plot_slowdown_into(ax, metrics):
    s_base, l_base = get_english_curve(metrics, N1_BASELINE_KEY)
    if s_base.size == 0:
        return
    for label, key, color, marker in CONDITIONS:
        s_comp, l_comp = get_english_curve(metrics, key)
        if s_comp.size == 0:
            continue
        xs, ys = slowdown_points(s_base, l_base, s_comp, l_comp, x_axis="val")
        if xs.size == 0:
            continue
        ax.plot(xs, ys, color=color, marker=marker, markersize=3,
                linewidth=1.0, label=label)
    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ax.invert_xaxis()
    ax.set_xlabel("English val loss (nats)")
    ax.set_ylabel("Slowdown vs c=1 (×)")


def _bottom_legend(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               frameon=False, handlelength=1.3, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
               markerscale=2.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1))


def fig_val(metrics):
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    plot_val_into(ax, metrics)
    ax.legend(loc="upper right", frameon=False,
              handlelength=1.3, handletextpad=0.5)
    fig.tight_layout()
    out = Path("../figures/capacity_sharing_val.pdf")
    fig.savefig(out)
    print(f"  {out}")


def fig_slowdown(metrics):
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    plot_slowdown_into(ax, metrics)
    ax.legend(loc="upper left", frameon=False,
              handlelength=1.3, handletextpad=0.5)
    fig.tight_layout()
    out = Path("../figures/capacity_sharing_slowdown.pdf")
    fig.savefig(out)
    print(f"  {out}")


def fig_combined(metrics):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.75, 2.8))
    plot_val_into(ax_l, metrics)
    plot_slowdown_into(ax_r, metrics)
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               frameon=False, handlelength=1.3, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
               markerscale=2.0)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = Path("../figures/capacity_sharing_combined.pdf")
    fig.savefig(out)
    print(f"  {out}")


def main():
    setup_paper_style()
    metrics = json.loads(Path("fineweb_val_metrics.json").read_text())
    Path("../figures").mkdir(exist_ok=True)
    fig_val(metrics)
    fig_slowdown(metrics)
    fig_combined(metrics)


if __name__ == "__main__":
    main()

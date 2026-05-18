"""c=2 InfoNCE batch-size comparison: 32 vs 128 vs 512.

Plots val-loss residual to fully-trained c=1 baseline over training step,
to mirror body Fig 7's framing. Includes the c=2 no-InfoNCE plateau as a
horizontal reference.

Reads named-checkpoint eval results from `val_metrics.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style
from _run_paths import (
    C1_BASELINE_8_256, NO_INFONCE_8_256_BY_C, INFONCE_8_256_C2_BY_BATCH,
)


def avg_compartment_curve(metrics, key, c):
    v = metrics[key]
    s = np.array(v["checkpoints"], dtype=float)
    losses = np.mean(
        [np.array(v["metrics"][f"loss_compartment_{ci}"]) for ci in range(c)], axis=0
    )
    o = np.argsort(s)
    return s[o], losses[o]


C1_BASELINE_KEY = C1_BASELINE_8_256
C2_BASELINE_KEY = NO_INFONCE_8_256_BY_C[2]


# (label, run_key, color)
RUNS = [
    ("InfoNCE n=32 (canonical)", INFONCE_8_256_C2_BY_BATCH[32],  "tab:blue"),
    ("InfoNCE n=128",            INFONCE_8_256_C2_BY_BATCH[128], "tab:orange"),
    ("InfoNCE n=512",            INFONCE_8_256_C2_BY_BATCH[512], "tab:green"),
]

STEP_MATCH_CAP = 1_580_000


def _render(figsize, fontsize_tweak=False, xmin=10_000, xmax=STEP_MATCH_CAP):
    metrics = json.loads(Path("val_metrics.json").read_text())
    c1_final = float(avg_compartment_curve(metrics, C1_BASELINE_KEY, 1)[1][-1])
    c2_final = float(avg_compartment_curve(metrics, C2_BASELINE_KEY, 2)[1][-1])

    fig, ax = plt.subplots(figsize=figsize)
    XMAX = 0
    for label, run_key, color in RUNS:
        if run_key not in metrics:
            print(f"  {label}: NO DATA (key={run_key})")
            continue
        s, l = avg_compartment_curve(metrics, run_key, 2)
        gap = l - c1_final
        mask = s >= xmin
        if xmax is not None:
            mask &= s <= xmax
        if mask.any():
            ax.plot(s[mask], gap[mask], color=color,
                    linewidth=1.4 if not fontsize_tweak else 1.2,
                    label=label)
            XMAX = max(XMAX, float(s[mask].max()))

    c2_resid = c2_final - c1_final
    plateau_label = (
        f"c=2 no-InfoNCE ({c2_final:.3f})" if fontsize_tweak
        else f"c=2 no-InfoNCE plateau ({c2_final:.3f})"
    )
    ax.axhline(c2_resid, color="black", linewidth=1.2, alpha=0.6,
               linestyle=":", label=plateau_label)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_xscale("log")
    ax.set_xlim(left=xmin, right=(xmax if xmax is not None else (XMAX if XMAX else None)))
    ax.set_ylim(-0.05, 0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("c=2 val − c=1 final val (nats)")
    legend_kwargs = dict(loc="upper right", frameon=True, facecolor="white",
                         edgecolor="none", framealpha=0.9,
                         handlelength=1.3, handletextpad=0.5)
    if fontsize_tweak:
        legend_kwargs.update(fontsize=7, handletextpad=0.4, framealpha=0.85)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    ax.legend(**legend_kwargs)
    fig.tight_layout(pad=0.3 if fontsize_tweak else 1.08)
    return fig


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)

    fig = _render(figsize=(4.5, 3.0), xmin=10_000)
    out = Path("../figures/infonce_n2_batch_compare.pdf")
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"  {out}"); plt.close(fig)

    fig = _render(figsize=(3.6, 2.8), fontsize_tweak=True, xmin=100_000)
    out = Path("../figures/infonce_n2_batch_compare_half.pdf")
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

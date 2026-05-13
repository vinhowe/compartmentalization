"""c=2 InfoNCE batch-size comparison: 32 vs 128 vs (later) 512.

Plots val-loss residual to fully-trained c=1 baseline over training step,
to mirror body Fig 7's framing. Includes the c=2 no-InfoNCE plateau as a
horizontal reference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style
from _run_paths import (
    C1_BASELINE_8_256, NO_INFONCE_8_256_BY_C, INFONCE_8_256_C2_LOGS_BY_BATCH,
)


VAL_PAT = re.compile(r"^step (\d+): train loss [\d.]+, val loss ([\d.]+)")


def parse_val_log(path):
    if not Path(path).exists():
        return {}
    seen = {}
    with open(path) as f:
        for line in f:
            m = VAL_PAT.match(line)
            if m:
                seen[int(m.group(1))] = float(m.group(2))
    return seen


C1_BASELINE_KEY = C1_BASELINE_8_256
C2_BASELINE_KEY = NO_INFONCE_8_256_BY_C[2]


# (label, log_paths, color)
RUNS = [
    ("InfoNCE n=32 (canonical)", INFONCE_8_256_C2_LOGS_BY_BATCH[32], "tab:blue"),
    ("InfoNCE n=128",            INFONCE_8_256_C2_LOGS_BY_BATCH[128], "tab:orange"),
    ("InfoNCE n=512",            INFONCE_8_256_C2_LOGS_BY_BATCH[512], "tab:green"),
]


def get_baseline_finals():
    m = json.loads(Path("val_metrics.json").read_text())
    c1 = m[C1_BASELINE_KEY]["metrics"]["loss_compartment_0"][-1]
    v2 = m[C2_BASELINE_KEY]
    losses = np.mean(
        [np.array(v2["metrics"][f"loss_compartment_{i}"]) for i in range(2)],
        axis=0,
    )
    c2 = float(losses[-1])
    return float(c1), c2


STEP_MATCH_CAP = 1_580_000


def _render(figsize, fontsize_tweak=False, xmin=10_000, xmax=STEP_MATCH_CAP):
    c1_final, c2_final = get_baseline_finals()

    fig, ax = plt.subplots(figsize=figsize)
    XMAX = 0
    for label, log_paths, color in RUNS:
        merged = {}
        for p in log_paths:
            merged.update(parse_val_log(p))
        if not merged:
            continue
        steps = np.array(sorted(merged), dtype=float)
        loss = np.array([merged[int(s)] for s in steps])
        gap = loss - c1_final
        mask = steps >= xmin
        if xmax is not None:
            mask &= steps <= xmax
        if mask.any():
            ax.plot(steps[mask], gap[mask], color=color,
                    linewidth=1.4 if not fontsize_tweak else 1.2,
                    label=label)
            XMAX = max(XMAX, float(steps[mask].max()))

    # c=2 no-InfoNCE plateau as reference.
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

    # Subfigure-ready (matches lambda sweep half size + XMIN convention).
    fig = _render(figsize=(3.6, 2.8), fontsize_tweak=True, xmin=100_000)
    out = Path("../figures/infonce_n2_batch_compare_half.pdf")
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

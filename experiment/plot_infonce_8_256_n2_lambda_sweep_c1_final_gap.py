"""InfoNCE λ sweep at c=2, residual gap to fully-trained c=1 baseline.

Five lines, one per λ ∈ {0.1, 0.7, 1.0, 1.3, 10}, all at c=2 / 8-256 rope / tr=0.
Trajectory: InfoNCE_λ(step) − baseline_c1_final.
Single horizontal reference: c=2 no-InfoNCE final − c=1 final (the tax to close).
y=0 is the c=1 floor — a curve dipping below it has beaten c=1.

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
    C1_BASELINE_8_256, NO_INFONCE_8_256_BY_C, INFONCE_8_256_C2_BY_LAMBDA,
    filter_to_loggy,
)


def avg_compartment_curve(metrics, key, c):
    v = metrics[key]
    s = np.array(v["checkpoints"], dtype=float)
    losses = np.mean(
        [np.array(v["metrics"][f"loss_compartment_{ci}"]) for ci in range(c)], axis=0
    )
    o = np.argsort(s)
    return filter_to_loggy(s[o], losses[o])


C1_BASELINE_KEY = C1_BASELINE_8_256
C2_BASELINE_KEY = NO_INFONCE_8_256_BY_C[2]

RUNS = sorted(INFONCE_8_256_C2_BY_LAMBDA.items())

STEP_MATCH_CAP = 1_580_000


def _render(figsize, fontsize_tweak=False, xmax=STEP_MATCH_CAP):
    metrics = json.loads(Path("val_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=figsize)

    s_c1, l_c1 = avg_compartment_curve(metrics, C1_BASELINE_KEY, 1)
    c1_final = float(l_c1[-1])

    XMIN = 100_000

    cmap = plt.get_cmap("viridis")
    lambdas = [r[0] for r in RUNS]
    log_lams = np.log10(lambdas)
    norm_lams = (log_lams - log_lams.min()) / (log_lams.max() - log_lams.min())

    for (lam, run_key), nl in zip(RUNS, norm_lams):
        if run_key not in metrics:
            print(f"  λ={lam}: NO DATA (key={run_key})")
            continue
        s, l = avg_compartment_curve(metrics, run_key, 2)
        gap = l - c1_final
        mask = s >= XMIN
        if xmax is not None:
            mask &= s <= xmax
        if mask.any():
            ax.plot(s[mask], gap[mask], color=cmap(nl),
                    linewidth=1.2, label=f"λ={lam:g}")

    # Horizontal reference: c=2 no-InfoNCE final − c=1 final.
    if C2_BASELINE_KEY in metrics:
        _, l_base = avg_compartment_curve(metrics, C2_BASELINE_KEY, 2)
        no_infonce_residual = float(l_base[-1] - c1_final)
        ax.axhline(no_infonce_residual, color="black",
                   linewidth=1.4, alpha=0.6, linestyle=":",
                   label="c=2 no-InfoNCE")

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(left=XMIN, right=xmax)
    ax.set_ylim(-0.05, 0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("c=2 val − c=1 final val (nats)")
    legend_kwargs = dict(loc="upper right", frameon=False,
                         handlelength=1.3, handletextpad=0.5)
    if fontsize_tweak:
        legend_kwargs.update(fontsize=7, handletextpad=0.4)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    ax.legend(**legend_kwargs)
    fig.tight_layout(pad=0.3 if fontsize_tweak else 1.08)
    return fig


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)

    fig = _render(figsize=(4.5, 3.0))
    out = Path("../figures/infonce_8_256_n2_lambda_sweep_c1_final_gap.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    fig = _render(figsize=(3.6, 2.8), fontsize_tweak=True)
    out = Path("../figures/infonce_8_256_n2_lambda_sweep_c1_final_gap_half.pdf")
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

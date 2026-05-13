"""InfoNCE residual gap to the *fully-trained* c=1 baseline (final-step
value, not step-matched).

Companion to plot_infonce_8_256_c1_gap.py. Difference: the c=1 baseline is
evaluated only at its 1M-iter final value (a constant), so:
  trajectory: InfoNCE_cN(step) − baseline_c1_final
  horizontal: baseline_cN_final − baseline_c1_final
Both reference the same fully-trained c=1 floor. A trajectory dropping below
its same-color horizontal means InfoNCE c=N has beaten the no-InfoNCE c=N
asymptote, both compared against the strongest available c=1 number.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style, C_COLOR
from _run_paths import (
    C1_BASELINE_8_256, NO_INFONCE_8_256_BY_C, INFONCE_8_256_LOGS_BY_C,
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


def parse_baseline(metrics, key, c):
    v = metrics[key]
    s = np.array(v["checkpoints"], dtype=float)
    losses = np.mean(
        [np.array(v["metrics"][f"loss_compartment_{ci}"]) for ci in range(c)], axis=0
    )
    order = np.argsort(s)
    return s[order], losses[order]


C1_BASELINE_KEY = C1_BASELINE_8_256

# (c, infonce_log_paths, c=N from-scratch baseline key)
RUNS = [(c, INFONCE_8_256_LOGS_BY_C[c], NO_INFONCE_8_256_BY_C[c]) for c in (2, 4, 5, 6, 8)]

# Step-matched view: clip everyone to the latest step c=8 has reached, drop c=5.
# c=6 will catch up to this bound by submission deadline.
STEP_MATCH_DROP = {5}


def render(figsize, fontsize_tweak=False, step_match=False, drop_c5=False):
    metrics = json.loads(Path("fineweb_val_metrics.json").read_text())
    Path("../figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    s_c1, l_c1 = parse_baseline(metrics, C1_BASELINE_KEY, 1)
    c1_final = float(l_c1[-1])

    XMIN = 100_000
    XMAX = None
    if step_match:
        # Clip to min(latest val step) across all included runs (drop c=5).
        per_c_max = []
        for c, log_paths, _ in RUNS:
            if c in STEP_MATCH_DROP:
                continue
            merged = {}
            for p in log_paths:
                merged.update(parse_val_log(p))
            if merged:
                per_c_max.append(max(merged))
        XMAX = min(per_c_max) if per_c_max else None

    for c, log_paths, base_key in RUNS:
        if (step_match or drop_c5) and c in STEP_MATCH_DROP:
            continue
        merged = {}
        for p in log_paths:
            merged.update(parse_val_log(p))
        steps = np.array(sorted(merged), dtype=float) if merged else np.array([])
        loss = np.array([merged[int(s)] for s in steps]) if merged else np.array([])
        gap = loss - c1_final

        if steps.size:
            mask = steps >= XMIN
            if XMAX is not None:
                mask &= steps <= XMAX
            if mask.any():
                ax.plot(steps[mask], gap[mask], color=C_COLOR[c],
                        linewidth=1.2, label=f"c={c}")

        if base_key in metrics:
            s_base, l_base = parse_baseline(metrics, base_key, c)
            no_infonce_residual = float(l_base[-1] - c1_final)
            ax.axhline(no_infonce_residual, color=C_COLOR[c],
                       linewidth=1.6, alpha=0.6, linestyle=":")

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(left=XMIN, right=XMAX)
    ax.set_ylim(-0.05, 0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("c=N val − c=1 final val (nats)")
    ax.yaxis.label.set_size(8)
    legend_kwargs = dict(loc="upper right", frameon=False,
                         handlelength=1.3, handletextpad=0.5, ncol=2,
                         columnspacing=0.8)
    if fontsize_tweak:
        legend_kwargs.update(fontsize=7, columnspacing=0.6,
                             borderpad=0.2, ncol=2)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    ax.legend(**legend_kwargs)
    fig.tight_layout(pad=0.3 if fontsize_tweak else 1.08)
    return fig


def main():
    setup_paper_style()

    fig = render(figsize=(4.5, 3.0))
    out = Path("../figures/infonce_8_256_c1_final_gap.pdf")
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"  {out}"); plt.close(fig)

    fig = render(figsize=(3.5, 2.3), fontsize_tweak=False, drop_c5=True)
    out = Path("../figures/infonce_8_256_c1_final_gap_half.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    # Step-matched: clip to c=8's latest, drop c=5.
    fig = render(figsize=(4.5, 3.0), step_match=True)
    out = Path("../figures/infonce_8_256_c1_final_gap_stepmatched.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    fig = render(figsize=(3.5, 2.3), fontsize_tweak=False, step_match=True)
    out = Path("../figures/infonce_8_256_c1_final_gap_stepmatched_half.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

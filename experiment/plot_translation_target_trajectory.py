"""Translation-task convergence: target-half val loss vs training step,
one line per c, at fixed translation ratio.

For each (c, tr) cell at 8-256 rope, wd=0, absolute-mode tr=0.5, we average
`loss_target_compartment_i > compartment_j` across all i→j pairs at each
checkpoint. Plot vs step.

The companion claim to the phase transition: even at compartment counts
(c≤6) where compartmented val stays plateau-locked under translation
intervention, the local translation task — predict compartment-j tokens
given a compartment-i source — is *easily learned*. Target-half loss falls
to within 0.01 nats of the c=1 floor by step ~50k for every c.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style, C_COLOR
from _run_paths import C1_BASELINE_8_256


GROUP = "bpe16384-rope-8-256"
C1_BASELINE_KEY = C1_BASELINE_8_256
TARGET_TR = 0.5  # fixed translation ratio for the trajectory chart


def main():
    setup_paper_style()
    metrics = json.loads(Path("fineweb_val_metrics.json").read_text())
    Path("../figures").mkdir(exist_ok=True)

    # Find one cell per c at the target tr (absolute mode, wd=0)
    by_c = {}
    for d in sorted((Path("..") / "out" / "translation-compression" / GROUP).iterdir()):
        cf = d / "meta" / "config.json"
        if not cf.exists():
            continue
        cobj = json.loads(cf.read_text())
        e, o = cobj["experiment"], cobj["optimizer"]
        if e["translation_ratio_mode"] != "absolute":
            continue
        if abs(e["translation_ratio"] - TARGET_TR) > 1e-6:
            continue
        if o["weight_decay"] != 0:
            continue
        c = e["n_compartments"]
        key = f"{GROUP}/{d.name}"
        if key not in metrics:
            continue
        by_c[c] = metrics[key]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    for c in sorted(by_c):
        v = by_c[c]
        steps = np.array(v["checkpoints"], dtype=float)
        # Average loss_target over all i->j pairs present
        target_arrs = []
        for k, arr in v["metrics"].items():
            if k.startswith("loss_target_"):
                target_arrs.append(np.array(arr, dtype=float))
        if not target_arrs:
            continue
        avg_target = np.mean(target_arrs, axis=0)
        ax.plot(steps, avg_target, color=C_COLOR.get(c, "k"), marker="o",
                markersize=3, linewidth=1.2, label=f"c={c}")

    # c=1 reference floor (plain val, since c=1 has no translation pairs)
    n1 = metrics.get(C1_BASELINE_KEY, {})
    if n1:
        c1_final = n1["metrics"]["loss_compartment_0"][-1]
        ax.axhline(c1_final, color="black", linewidth=0.6, alpha=0.5,
                   linestyle=":", label="c=1 val")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("target-half val loss (nats)")
    ax.legend(loc="lower left", frameon=False, fontsize=7,
              handlelength=1.3, handletextpad=0.4, ncol=2)
    fig.tight_layout()
    out = Path("../figures/tr_target_trajectory.pdf")
    fig.savefig(out)
    print(f"  {out}")
    plt.close(fig)

    # Compact (1/3-column-ready)
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    for c in sorted(by_c):
        v = by_c[c]
        steps = np.array(v["checkpoints"], dtype=float)
        target_arrs = []
        for k, arr in v["metrics"].items():
            if k.startswith("loss_target_"):
                target_arrs.append(np.array(arr, dtype=float))
        if not target_arrs:
            continue
        avg_target = np.mean(target_arrs, axis=0)
        ax.plot(steps, avg_target, color=C_COLOR.get(c, "k"), marker="o",
                markersize=2.5, linewidth=1.0, label=f"c={c}")
    if n1:
        ax.axhline(c1_final, color="black", linewidth=0.6, alpha=0.5,
                   linestyle=":", label="c=1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("target val loss (nats)")
    ax.legend(loc="lower left",
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.9,
              handlelength=1.0, handletextpad=0.3, ncol=2,
              columnspacing=0.6, borderpad=0.2)
    fig.tight_layout(pad=0.3)
    out = Path("../figures/tr_target_trajectory_compact.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

"""Plot signed interference across scale (compute_interference.py output).

Left: share = cos(other-compartment gradient, compartment-0 task direction) /
cos(fresh compartment-0 gradient, task direction), on compartment 0's top-10% MLP
neurons, vs model size. c=1 (other = more compartment-0 text) is the shared ceiling.
Middle: the raw other-compartment cosine (sign: + cooperates, - conflicts).
Right: the compartmentalization gap (mean per-compartment val loss minus c=1, formal
eval, step 1M) at each scale, for comparison.

Usage: python plot_interference.py RESULTS_DIR OUT_PREFIX
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression")
COL = {1: "#8a8985", 2: "#2a78d6", 8: "#e34948"}


def main():
    rows = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("*.json"))]
    M = json.loads((MAIN / "experiment" / "val_metrics.json").read_text())

    def val(key, c):
        r = M.get(key)
        if not r or 1_000_000 not in r["checkpoints"]:
            return None
        i = r["checkpoints"].index(1_000_000)
        return float(np.mean([r["metrics"][f"loss_compartment_{j}"][i] for j in range(c)]))

    by = {(r["scale"], r["c"]): r for r in rows}
    scales = sorted({r["scale"] for r in rows})
    params = {s: by[(s, 1)]["n_params"] for s in scales if (s, 1) in by}
    gap = {}
    for (s, c), r in by.items():
        v, v1 = val(r["key"], c), val(by[(s, 1)]["key"], 1) if (s, 1) in by else None
        if v is not None and v1 is not None:
            gap[(s, c)] = v - v1

    print(f"{'scale':>6s} {'c':>2s} {'params':>9s} {'share S':>8s} {'share S1':>8s} {'share trunk':>11s} {'cos_other S':>11s} {'gap':>7s}")
    for (s, c), r in sorted(by.items()):
        g = gap.get((s, c))
        print(f"{s:6d} {c:2d} {r['n_params']/1e6:8.1f}M {r['S']['share']:8.2f} {r['S1']['share']:8.2f} "
              f"{r['trunk']['share']:11.2f} {r['S']['cos_other']:+11.3f} {'' if g is None else f'{g:7.3f}'}")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for c in (1, 2, 8):
        pts = [(params[s], by[(s, c)]) for s in scales if (s, c) in by and s in params]
        if not pts:
            continue
        x = [p for p, _ in pts]
        axes[0].plot(x, [r["S"]["share"] for _, r in pts], "-o", color=COL[c], ms=4, lw=1.4, label=f"c={c}")
        axes[1].plot(x, [r["S"]["cos_other"] for _, r in pts], "-o", color=COL[c], ms=4, lw=1.4, label=f"c={c}")
        if c > 1:
            gp = [(params[s], gap[(s, c)]) for s in scales if (s, c) in gap and s in params]
            axes[2].plot([p for p, _ in gp], [g for _, g in gp], "-o", color=COL[c], ms=4, lw=1.4, label=f"c={c}")
    axes[0].set_ylabel("share of task-gradient alignment\n(other compartment / same, task neurons)")
    axes[1].set_ylabel("cos(other-compartment grad, task dir.)")
    axes[2].set_ylabel("val loss gap to c=1 (nats)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("parameters (c=1 model)")
        ax.axhline(0, color="black", lw=0.6, ls=":")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(sys.argv[2] + ".png", dpi=160)
    fig.savefig(sys.argv[2] + ".pdf")
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()

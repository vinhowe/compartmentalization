"""Early look at the avgemb arms from training logs (loss every 10 steps).

Left: smoothed training loss vs steps since the intervention (log x).
Right: each averaged arm minus the control. All arms see the identical data stream,
so the per-step difference cancels shared batch noise. Training loss includes the
~1% translation examples; the formal per-compartment eval replaces this at the end.

Usage: python plot_avgemb_trainlog.py OUT_PREFIX
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path("/mnt/pccfs2/backed_up/vin/dev/tc-avgemb-f718f07/logs")
START = 888_000
ALL_ARMS = [("avgboth", "input + output rows averaged", "#e34948"),
            ("avgwte", "input rows averaged", "#eda100"),
            ("copyboth", "input + output rows copied from comp. 1", "#8f3bb8"),
            ("copywte", "input rows copied from comp. 1", "#1baf7a"),
            ("control", "unmodified continuation", "#2a78d6")]
ARMS = [a for a in ALL_ARMS if (LOGS / f"8-256-n8-tr01comp-s66-{a[0]}.log").exists()]


def read(arm):
    txt = (LOGS / f"8-256-n8-tr01comp-s66-{arm}.log").read_text(errors="ignore")
    rows = re.findall(r"^iter (\d+): loss ([\d.]+)", txt, flags=re.M)
    d = {int(i): float(l) for i, l in rows}
    return d


def smooth_log(steps, y):
    """Mean over windows that widen with step (~5% of the step), so early detail survives."""
    out = np.empty_like(y)
    for k, s in enumerate(steps):
        w = max(10, 0.05 * s)
        m = (steps >= s - w / 2) & (steps <= s + w / 2)
        out[k] = y[m].mean()
    return out


def main():
    out = sys.argv[1]
    data = {a: read(a) for a, _, _ in ARMS}
    # Each arm on its own steps (arms started at different times); gaps use the steps
    # an arm shares with the control.
    series = {}
    for a in data:
        s = np.array(sorted(k for k in data[a] if k in data["control"] and k - START >= 10))
        series[a] = (s - START, np.array([data[a][k] for k in s]), np.array([data["control"][k] for k in s]))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, label, col in ARMS:
        steps, y, yc = series[a]
        axes[0].plot(steps, smooth_log(steps, y), color=col, lw=1.4, label=label)
        if a != "control":
            axes[1].plot(steps, smooth_log(steps, y - yc), color=col, lw=1.4, label=label)
    # c=1 reference. Training loss is not comparable to c=1's val loss directly, so use
    # the formal-eval val-loss difference at 1M steps (c=1 minus the source c=8 run) as
    # an offset from the control.
    import json, sys as _sys
    main_dir = Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression/experiment")
    _sys.path.insert(0, str(main_dir))
    from _run_paths import C1_BASELINE_8_256
    M = json.loads((main_dir / "val_metrics.json").read_text())
    def at1m(k, c):
        r = M[k]; i = r["checkpoints"].index(1_000_000)
        return float(np.mean([r["metrics"][f"loss_compartment_{j}"][i] for j in range(c)]))
    dc1 = at1m(C1_BASELINE_8_256, 1) - at1m("8-256-reseed/8-256-n8-tr01comp-s66", 8)
    cs, _, cy = series["control"]
    ctrl_level = np.mean(smooth_log(cs, cy)[cs >= cs[-1] - 5000])
    for ax, y in ((axes[0], ctrl_level + dc1), (axes[1], dc1)):
        ax.axhline(y, color="black", lw=0.8, ls="--")
        ax.text(0.02, y, " c=1", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7)
    axes[0].set_xscale("log")
    axes[0].set_ylim(3.9, 7.5)
    axes[0].set_ylabel("training loss (nats, smoothed)")
    axes[1].set_xscale("log")
    axes[1].set_ylim(dc1 - 0.1, 3.5)
    axes[1].axhline(0, color="black", lw=0.6, ls=":")
    axes[1].set_ylabel("loss minus control (nats)")
    for ax in axes:
        ax.set_xlabel("steps since intervention")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=160)
    for a, _, _ in ARMS:
        if a == "control":
            continue
        steps, y, yc = series[a]
        row = [f"through +{steps[-1]}"]
        for s in (100, 1000, 3000, 10000, 20000, 30000):
            m = np.abs(steps - s) <= max(50, 0.05 * s)
            if m.any():
                row.append(f"+{s}: {np.mean(y[m] - yc[m]):+.3f}")
        print(f"{a:9s} " + "  ".join(row))


if __name__ == "__main__":
    main()

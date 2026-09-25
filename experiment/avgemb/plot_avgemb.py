"""Plot the avgemb (unify a trained c=8 model's vocabularies) trajectories.

Mean per-compartment val loss (formal eval, val_metrics.json) vs steps since the
intervention for the three arms, with the c=1 and original c=8 1M-step losses as
references and the paper's post-hoc duplication run (c=1 -> c=8, the forward
direction) overlaid for comparison.

Usage: python plot_avgemb.py OUT_PREFIX
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression")
sys.path.insert(0, str(MAIN / "experiment"))
from _run_paths import C1_BASELINE_8_256  # noqa: E402

START, C = 888_000, 8
M = json.loads((MAIN / "experiment" / "val_metrics.json").read_text())
FT = json.loads((MAIN / "experiment" / "finetune_val_metrics.json").read_text())
ARMS = [("avgboth", "input + output rows averaged", "#e34948"),
        ("avgwte", "input rows averaged", "#eda100"),
        ("control", "unmodified continuation", "#2a78d6")]


def mean_loss(rec, c, i):
    return float(np.mean([rec["metrics"][f"loss_compartment_{j}"][i] for j in range(c)]))


def at(key, c, step):
    r = M[key]
    return mean_loss(r, c, r["checkpoints"].index(step))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "avgemb"
    c1 = at(C1_BASELINE_8_256, 1, 1_000_000)
    c8 = at("8-256-reseed/8-256-n8-tr01comp-s66", C, 1_000_000)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    summary = {"c1_1M": c1, "c8_s66_1M": c8, "arms": {}}
    for arm, label, col in ARMS:
        r = M[f"8-256-avgemb/8-256-n8-tr01comp-s66-{arm}"]
        steps = np.array(r["checkpoints"]) - START
        y = np.array([mean_loss(r, C, i) for i in range(len(steps))])
        summary["arms"][arm] = dict(zip(steps.tolist(), y.tolist()))
        for ax in axes:
            ax.plot(np.maximum(steps, 10), y, "-o", color=col, ms=3, lw=1.4, label=label)
    ft = FT["ce-full-8comp-rope"]
    fs = np.array(ft["checkpoints"])
    fy = np.array([mean_loss(ft, C, i) for i in range(len(fs))])
    for ax in axes:
        ax.plot(fs, fy, "--", color="#52514e", lw=1.2, label="post-hoc: c=1 duplicated to c=8")
        ax.axhline(c1, color="black", lw=0.6, ls=":", alpha=0.6)
        ax.axhline(c8, color="black", lw=0.6, ls="-.", alpha=0.6)
        ax.set_xlabel("steps since intervention")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("mean per-compartment val loss (nats)")
    axes[1].set_ylim(c1 - 0.05, c8 + 0.25)
    axes[1].text(0.99, c1, " c=1", transform=axes[1].get_yaxis_transform(), va="bottom", ha="right", fontsize=7)
    axes[1].text(0.99, c8, " c=8 original", transform=axes[1].get_yaxis_transform(), va="bottom", ha="right", fontsize=7)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=160)
    Path(out + ".json").write_text(json.dumps(summary, indent=1))
    print("c=1 at 1M %.4f | original c=8 s66 at 1M %.4f" % (c1, c8))
    for arm, d in summary["arms"].items():
        pts = sorted(d.items())
        print(arm, " ".join(f"{s}:{v:.3f}" for s, v in pts if s in (0, 100, 1000, 5000, 10000, 20000, 30000)))


if __name__ == "__main__":
    main()

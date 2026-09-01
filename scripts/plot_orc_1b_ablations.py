#!/usr/bin/env python3
"""Plot the two 1B-scale ablations that already exist on ORC.

Both are clean single-variable comparisons at 16x2048, tr=0, seed 1024,
lr 4e-4 FIXED:

  weight decay      c=1, compemb on:  wd 0   vs wd 0.1
  compartment emb   c=8, hash order:  compemb on vs off

Data comes from each checkpoint's trainer_state.json (`best_val_loss`), not
from wandb -- no val_metrics.json exists for any ORC 1B run, so the formal
eval pipeline has not been run on these. Training-time val is defensible here
because every arm is tr=0 and each pair holds n_compartments fixed, so neither
the translation-contamination nor the compartment-mixture caveat applies.

The LR is fixed at 4e-4 in every arm, so these are MATCHED-LR comparisons, not
optimum-vs-optimum. Each delta is therefore an upper bound on the true cost of
the alternative if that alternative would prefer a different LR.

Usage: .venv/bin/python scripts/plot_orc_1b_ablations.py <traj.json>
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ("/mnt/pccfs2/backed_up/vin/dev/translation-compression/"
       "figures/orc-1b-ablations.png")


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "traj.json"
    d = json.load(open(src))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    for col, (title, arms) in enumerate(d.items()):
        ax, axd = axes[0][col], axes[1][col]
        series = {}
        for lab, pts in arms.items():
            xy = sorted((int(k) / 1e9, v) for k, v in pts.items())
            series[lab] = dict(xy)
            ax.plot([p[0] for p in xy], [p[1] for p in xy], "-o", ms=2.5,
                    label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("best val loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        labs = list(series)
        common = sorted(set(series[labs[0]]) & set(series[labs[1]]))
        # drop the flat pre-warmup points, where every arm still reads ln(V)
        common = [t for t in common if t >= 4.0]
        axd.plot(common,
                 [series[labs[1]][t] - series[labs[0]][t] for t in common],
                 "-o", ms=2.5, color="#c05621")
        axd.axhline(0, color="k", lw=0.8)
        axd.set_xscale("log")
        axd.set_xlabel("tokens (B)")
        axd.set_ylabel(f"{labs[1]}\n- {labs[0]}", fontsize=7)
        axd.grid(alpha=0.25)

    fig.suptitle("1B-scale ablations already on ORC "
                 "(old recipe, lr 4e-4 fixed in every arm)", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)

    for title, arms in d.items():
        labs = list(arms)
        a, b = arms[labs[0]], arms[labs[1]]
        common = sorted(set(a) & set(b), key=int)
        if not common:
            continue
        t = common[-1]
        print(f"{title}\n   at {int(t)/1e9:.1f}B: {labs[0]}={a[t]:.4f}  "
              f"{labs[1]}={b[t]:.4f}  delta={b[t]-a[t]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

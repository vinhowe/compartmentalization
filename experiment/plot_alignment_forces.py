"""What the direct force measurement does and does not establish.

Reads alignment_forces.json. Two claims are at stake, and they come apart:

  1. "Translation rows are the aligning force, monolingual rows are not."
     SUPPORTED. The projection of -dL_translation/dE onto the q-increasing
     direction is positive at every checkpoint after the start, in both runs, and
     larger than the spread across independent batch draws. The monolingual
     projection changes sign from checkpoint to checkpoint and straddles zero.

  2. "The pull grows as the codes overlap, which is what makes it autocatalytic."
     NOT ESTABLISHED by this measurement. The translation projection does not
     grow with q here -- if anything it shrinks. Two reasons that stop this being
     a refutation, both worth stating rather than burying: the quantity plotted
     is a projection of the RAW gradient, while training uses Adam, whose
     per-coordinate rescaling changes the step direction substantially for
     embedding rows with very different update frequencies; and the projection is
     onto a unit vector, so it is a rate per unit distance moved, not the dq per
     optimiser step that the phase portrait measures.

So the phase portrait in plot_transition_mechanism.py remains a description of
the dynamics that this has not yet reduced to a force law. Closing that gap
needs the Adam-preconditioned update rather than the gradient.

Run from experiment/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style

FIGDIR = Path("../figures")
RUNLABEL = {"tr025": ("tr = 0.25 (stalls)", "#ea580c"),
            "tr05": ("tr = 0.50 (breaks)", "#2563eb")}


def main():
    setup_paper_style()
    FIGDIR.mkdir(exist_ok=True)
    d = json.loads(Path("alignment_forces.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)

    for ax, key in zip(axes, ["push_trans", "push_mono"]):
        for run, (lab, col) in RUNLABEL.items():
            if run not in d:
                continue
            pts = d[run]["points"]
            steps = sorted(int(s) for s in pts)
            q = [pts[str(s)]["q"] for s in steps]
            v = np.array([pts[str(s)][key] for s in steps])
            e = np.array([pts[str(s)][key + "_spread"] for s in steps])
            ax.errorbar(q, v, yerr=e, color=col, lw=1.5, marker="o", ms=4,
                        capsize=2.5, elinewidth=1.0, label=lab)
        ax.axhline(0, color="k", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlabel("agreement so far  $q$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("push on $q$ per unit step\n"
                       "$\\langle -g,\\ \\hat g_q\\rangle$")
    axes[0].set_title("translation rows\nconsistently POSITIVE: they align",
                      fontsize=9)
    axes[1].set_title("monolingual rows\nsign flips: no consistent pull",
                      fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Measuring the force directly: project each loss's gradient onto the "
        "direction that increases agreement\n"
        "error bars = spread over 3 independent batch draws of 2048 sequences "
        "(raw gradients, not Adam-preconditioned)",
        fontsize=9, y=1.04)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"alignment_forces.{ext}", bbox_inches="tight",
                    dpi=220)
    plt.close(fig)
    print("  wrote figures/alignment_forces.{pdf,png}")

    for run in d:
        pts = d[run]["points"]
        steps = sorted(int(s) for s in pts)[1:]      # skip step 100
        t = np.array([pts[str(s)]["push_trans"] for s in steps])
        m = np.array([pts[str(s)]["push_mono"] for s in steps])
        print(f"  {run}: translation positive at {(t > 0).sum()}/{len(t)} "
              f"checkpoints; monolingual positive at {(m > 0).sum()}/{len(m)}")


if __name__ == "__main__":
    main()

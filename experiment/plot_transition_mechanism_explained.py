"""The same mechanism result as plot_transition_mechanism.py, drawn to be read
without knowing the vocabulary.

The original figure asked the reader to interpret a log-log plot of dq/dlog(step)
against q, which is a phase portrait -- a standard object if you already think in
dynamical systems, and opaque otherwise. Three changes here:

  * every panel title is the question it answers, in words;
  * the flow panel is NORMALISED, so every curve starts at 1.0 and the only
    thing to read is whether it goes up (agreement speeds up as it grows:
    snowball) or down (it slows as it grows: fizzle). No units to interpret;
  * arrows mark the direction training travels, since the x-axis is a state and
    not time, which is the part that trips people up.

Same numbers, same conclusions, fewer prerequisites.

Run from experiment/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style
from plot_transition_mechanism import (ORDER, LABEL, COLOR, LAYER_LABEL,
                                       LAYER_COLOR, traj, flow, Q_REF)

FIGDIR = Path("../figures")


def annotate_arrow(ax, x, y, frac=0.55, color="k", size=9):
    """A small arrowhead partway along a curve, marking time's direction."""
    n = len(x)
    if n < 4:
        return
    i = max(1, min(n - 2, int(frac * n)))
    ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i - 1], y[i - 1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0,
                                mutation_scale=size, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=0.6)
    args = ap.parse_args()

    setup_paper_style()
    plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 9.5})
    FIGDIR.mkdir(exist_ok=True)
    d = json.loads(Path("embedding_order.json").read_text())
    lay = json.loads(Path("layerwise_alignment.json").read_text())

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))

    # ── A ───────────────────────────────────────────────────────────────────
    ax = axes[0]
    for k in ORDER:
        if k not in d:
            continue
        st, q, tr = traj(d, k)
        ax.plot(st, q, color=COLOR[k], lw=1.8, label=f"tr = {LABEL[k]}")
        ax.annotate(f"{LABEL[k]}", xy=(st[-1], q[-1]), fontsize=6.6,
                    color=COLOR[k], xytext=(3, -1), textcoords="offset points",
                    va="center")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("agreement between compartments  $q$")
    ax.set_title("(A) Do the 8 compartments agree\non how to encode a token?",
                 fontsize=9.5)
    ax.text(0.03, 0.95,
            "$q=0$: every compartment invents\nits own code for each token\n"
            "$q=1$: they all agree",
            transform=ax.transAxes, fontsize=6.8, va="top", color="0.25")
    ax.set_xlim(80, 6e6)
    ax.grid(alpha=0.25)

    # ── B: the normalised flow ──────────────────────────────────────────────
    ax = axes[1]
    for k in ORDER:
        if k not in d:
            continue
        st, q, tr = traj(d, k)
        f = flow(st, q, args.win)
        ok = ~np.isnan(f) & (q > 0.015)
        if ok.sum() < 4 or q[ok].max() < Q_REF:
            continue
        i0 = int(np.argmin(np.abs(q[ok] - Q_REF)))
        f0 = f[ok][i0]
        if f0 <= 0:
            continue
        x, y = q[ok], f[ok] / f0
        ax.plot(x, y, color=COLOR[k], lw=1.9, label=f"tr = {LABEL[k]}")
        annotate_arrow(ax, x, y, 0.6, COLOR[k], 10)
        ax.annotate(f"{LABEL[k]}", xy=(x[-1], y[-1]), fontsize=6.8,
                    color=COLOR[k], xytext=(4, 0), textcoords="offset points",
                    va="center")
    ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("agreement so far  $q$   (arrows: direction of training)")
    ax.set_ylabel("how fast agreement is growing\n(relative to its own early rate)")
    ax.set_title("(B) As they agree more, does agreement\n"
                 "speed up or slow down?", fontsize=9.5)
    ax.text(0.04, 0.94, "ABOVE the line: speeding up\n$\\rightarrow$ snowballs "
            "to full agreement", transform=ax.transAxes, fontsize=7.2,
            va="top", color="#1d4ed8")
    # Right of the orange curves and under the dashed line is the only clear
    # region in this panel.
    ax.text(0.97, 0.13, "BELOW: slowing down\n$\\rightarrow$ fizzles out",
            transform=ax.transAxes, fontsize=7.2, va="top", ha="right",
            color="#c2410c")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1.0, 1.5, 2, 3, 5, 8]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(alpha=0.25, which="both")

    # ── C ───────────────────────────────────────────────────────────────────
    ax = axes[2]
    trs, gains, cols = [], [], []
    for k in ORDER:
        if k not in d:
            continue
        st, q, tr = traj(d, k)
        f = flow(st, q, args.win)
        ok = np.where(~np.isnan(f))[0]
        if len(ok) == 0 or q.max() < Q_REF:
            continue
        i0 = ok[int(np.argmin(np.abs(q[ok] - Q_REF)))]
        if f[i0] <= 0:
            continue
        trs.append(tr); gains.append(f[ok[-1]] / f[i0]); cols.append(COLOR[k])
    ax.axhspan(0.5, 1.0, color="#ea580c", alpha=0.09, lw=0)
    ax.axhspan(1.0, 12, color="#2563eb", alpha=0.09, lw=0)
    ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.75)
    ax.plot(trs, gains, color="0.4", lw=1.2, zorder=1)
    ax.scatter(trs, gains, c=cols, s=62, zorder=3, edgecolors="white",
               linewidths=0.9)
    for tr, g in zip(trs, gains):
        ax.annotate(f"{g:.2f}", xy=(tr, g), xytext=(0, 8),
                    textcoords="offset points", fontsize=6.8, ha="center")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 12)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("translation ratio  tr")
    ax.set_ylabel("end rate $\\div$ early rate")
    ax.set_title("(C) The tipping point sits between\ntr = 0.25 and tr = 0.5",
                 fontsize=9.5)
    ax.text(0.60, 6.0, "snowball", fontsize=8, color="#1d4ed8", ha="center")
    ax.text(0.17, 0.62, "fizzle", fontsize=8, color="#c2410c", ha="center")
    ax.grid(alpha=0.25, which="both")

    # ── D ───────────────────────────────────────────────────────────────────
    ax = axes[3]
    for k in ["tr011", "tr025", "tr05", "infonce", "copyemb"]:
        if k not in lay:
            continue
        prof = lay[k]["profile"]
        xs = sorted(int(x) for x in prof)
        ax.plot(xs, [prof[str(x)] for x in xs], marker="o", ms=3.5, lw=1.6,
                color=LAYER_COLOR[k], label=LAYER_LABEL[k])
    ax.axvspan(6.5, 7.4, color="0.5", alpha=0.12, lw=0)
    ax.annotate("last block separates them again,\n"
                "to read out through the\nper-compartment head",
                xy=(7, 0.44), xytext=(1.15, 0.64), fontsize=6.8, color="0.25",
                ha="left",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    ax.set_ylim(-0.06, 1.42)
    ax.set_xlabel("depth  ($-1$ = raw embedding)")
    ax.set_ylabel("agreement, measured in activations")
    ax.set_title("(D) Agreement is built through the trunk,\n"
                 "then undone at the very end", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=2,
              columnspacing=0.8, handlelength=1.4)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Why translation ratio has a threshold: translation rows make agreement "
        "self-reinforcing, and the loop only closes above tr $\\approx$ 0.25–0.5"
        "      (c=8, 8-256, wd=0, constant learning rate)",
        fontsize=10.5, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"transition_mechanism_explained.{ext}",
                    bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("  wrote figures/transition_mechanism_explained.{pdf,png}")


if __name__ == "__main__":
    main()

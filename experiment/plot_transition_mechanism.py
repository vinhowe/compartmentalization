"""A mechanism for the compartmentalisation transition: autocatalysis in an
order parameter, and a bifurcation in tr.

The setup that makes this tractable is architectural. The trunk is shared across
all compartments; the only per-compartment parameters are the vocabulary blocks
of wte and lm_head plus an 8x256 compartment embedding. So "compartmentalised"
is a statement about those blocks, and

    q = mean_t mean_{i<j} cos( E_i[t], E_j[t] )

is an order parameter: 0 when every compartment invents its own code for a
token, 1 when they agree. It is read off the weights, with no forward pass and
no data, so it is independent of the activation-space cossim used elsewhere --
and it reproduces the same transition, which is the point of measuring it.

Treating training as a dynamical system in q and plotting the flow

    f(q) = dq / d log10(step)

against q -- a phase portrait, the standard move for this kind of question --
separates the two regimes cleanly, and does it without extrapolating:

  * tr >= 0.5:  f INCREASES with q. More agreement makes agreement grow faster.
    That is autocatalysis, and it runs away to the unified phase.
  * tr <= 0.25: f rises, turns over, and decays. The aligning force weakens as
    it succeeds, so the trajectory creeps and stalls.

The gain ratio f(q_final) / f(q=0.04) puts one number on it and crosses 1
between tr=0.25 and tr=0.5: 0.80, 0.82, then 3.3, 6.1. That crossing is the
transition, and it is robust to the smoothing window (0.4 to 1.0 decades changes
the ratios by a few percent).

Why should the force be autocatalytic? A translation row shows the model the
same content twice, once in compartment i's vocabulary and once in j's, and asks
it to predict the second half from the first. The easier it already is to map
i's code onto j's, the more that objective pays for pushing them closer -- a
partially shared code is worth more than an unshared one. Monolingual rows
supply the opposing force and do not scale with q. Whether the feedback loop
closes depends on how much of the batch is translation, which is tr.

Panel D asks where in the network the agreement lives. It rises with depth
(embeddings agree least, mid-trunk most) and then collapses in the last block --
the trunk pulls the compartments together, and the final layer has to push them
apart again to read out through the per-compartment lm_head.

Reads embedding_order.json and layerwise_alignment.json. Run from experiment/.
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

FIGDIR = Path("../figures")

ORDER = ["tr011", "tr01", "tr025", "tr05", "tr075", "tr10"]
LABEL = {"tr011": "0.011", "tr01": "0.10", "tr025": "0.25",
         "tr05": "0.50", "tr075": "0.75", "tr10": "1.00"}
# Below / above the transition, warm / cool.
COLOR = {"tr011": "#9a3412", "tr01": "#c2410c", "tr025": "#ea580c",
         "tr05": "#2563eb", "tr075": "#1d4ed8", "tr10": "#1e3a8a"}

LAYER_LABEL = {"tr025": "tr=0.25", "tr05": "tr=0.50", "tr011": "tr$\\approx$0.011",
               "copyemb": "init-copy", "infonce": "InfoNCE"}
LAYER_COLOR = {"tr025": "#ea580c", "tr05": "#2563eb", "tr011": "#9a3412",
               "copyemb": "#059669", "infonce": "#7c3aed"}

Q_REF = 0.04       # where the gain ratio is anchored


def traj(d, k):
    v = d[k]
    st = np.array(sorted(int(x) for x in v["data"]), dtype=float)
    q = np.array([v["data"][str(int(s))]["q_emb"] for s in st])
    return st, q, v["tr"]


def flow(st, q, win=0.6):
    """dq/dlog10(step), local linear fit in a window of `win` decades."""
    ls = np.log10(st)
    out = np.full_like(q, np.nan)
    for i in range(len(q)):
        m = np.abs(ls - ls[i]) <= win / 2
        if m.sum() >= 4:
            out[i] = np.polyfit(ls[m], q[m], 1)[0]
    return out


def gain_ratio(st, q, win=0.6, q_ref=Q_REF):
    f = flow(st, q, win)
    ok = np.where(~np.isnan(f))[0]
    if len(ok) == 0 or q.max() < q_ref:
        return None
    i0 = ok[int(np.argmin(np.abs(q[ok] - q_ref)))]
    i1 = ok[-1]
    if f[i0] <= 0:
        return None
    return f[i1] / f[i0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=0.6)
    args = ap.parse_args()

    setup_paper_style()
    FIGDIR.mkdir(exist_ok=True)
    d = json.loads(Path("embedding_order.json").read_text())
    lay = json.loads(Path("layerwise_alignment.json").read_text())

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6))

    # ── A: trajectories ─────────────────────────────────────────────────────
    ax = axes[0]
    for k in ORDER:
        if k not in d:
            continue
        st, q, tr = traj(d, k)
        ax.plot(st, q, color=COLOR[k], lw=1.5, label=f"tr = {LABEL[k]}")
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("order parameter  $q$\n(embedding-block agreement)")
    ax.set_title("(A) the order parameter, from weights alone", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", ncol=2,
              columnspacing=0.8, handlelength=1.4)
    ax.grid(alpha=0.25)

    # ── B: the phase portrait ───────────────────────────────────────────────
    ax = axes[1]
    for k in ORDER:
        if k not in d:
            continue
        st, q, tr = traj(d, k)
        f = flow(st, q, args.win)
        m = ~np.isnan(f) & (q > 0.015)
        ax.plot(q[m], f[m], color=COLOR[k], lw=1.6, label=f"tr = {LABEL[k]}")
        if m.sum():
            ax.scatter(q[m][-1], f[m][-1], s=16, color=COLOR[k], zorder=3,
                       edgecolors="white", linewidths=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("order parameter  $q$")
    ax.set_ylabel("flow  $f(q) = dq\\,/\\,d\\log_{10}$ step")
    ax.set_title("(B) flow vs state: rising = autocatalytic,\n"
                 "falling = stalling  (dot = end of run)", fontsize=8.5)
    ax.grid(alpha=0.25, which="both")

    # ── C: the bifurcation ──────────────────────────────────────────────────
    ax = axes[2]
    trs, gains, cols = [], [], []
    for k in ORDER:
        if k not in d:
            continue
        st, q, tr = traj(d, k)
        g = gain_ratio(st, q, args.win)
        if g is None:
            continue
        trs.append(tr); gains.append(g); cols.append(COLOR[k])
    ax.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.7)
    ax.axhspan(min(gains) * 0.8, 1.0, color="#ea580c", alpha=0.07, lw=0)
    ax.axhspan(1.0, max(gains) * 1.25, color="#2563eb", alpha=0.07, lw=0)
    ax.plot(trs, gains, color="0.35", lw=1.0, zorder=1)
    ax.scatter(trs, gains, c=cols, s=46, zorder=3, edgecolors="white",
               linewidths=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("translation ratio  tr")
    ax.set_ylabel("gain  $f(q_{\\rm end})\\,/\\,f(q{=}0.04)$")
    ax.set_title("(C) gain crosses 1 between tr=0.25\n"
                 "and tr=0.5 — the transition", fontsize=8.5)
    ax.annotate("runaway", xy=(0.62, max(gains) * 0.62), fontsize=7,
                color="#1d4ed8", ha="center")
    ax.annotate("stalls", xy=(0.14, min(gains) * 0.90), fontsize=7,
                color="#c2410c", ha="center")
    ax.grid(alpha=0.25, which="both")

    # ── D: where the agreement lives ────────────────────────────────────────
    ax = axes[3]
    for k in ["tr011", "tr025", "tr05", "infonce", "copyemb"]:
        if k not in lay:
            continue
        prof = lay[k]["profile"]
        xs = sorted(int(x) for x in prof)
        ys = [prof[str(x)] for x in xs]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color=LAYER_COLOR[k],
                label=LAYER_LABEL[k])
    ax.axvline(-0.5, color="k", lw=0.6, alpha=0.4, ls=":")
    ax.set_xlabel("depth  (-1 = token embedding, 0-7 = after block)")
    ax.set_ylabel("cross-compartment cosine")
    ax.set_title("(D) the trunk builds agreement,\n"
                 "the last block tears it down", fontsize=8.5)
    # Headroom above the unified curves (which sit at ~0.98) so the legend has
    # somewhere to go that is not on top of them.
    ax.set_ylim(-0.06, 1.42)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", ncol=2,
              columnspacing=0.8, handlelength=1.4)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "A mechanism for the transition: translation rows make agreement "
        "self-reinforcing, and the feedback loop closes only above a critical tr"
        "   (c=8, 8-256, wd=0, constant LR)",
        fontsize=9.5, y=1.03)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"transition_mechanism.{ext}", bbox_inches="tight",
                    dpi=220)
    plt.close(fig)

    print("  gain ratios (f at end / f at q=0.04):")
    for tr, g in zip(trs, gains):
        print(f"    tr={tr:<7} {g:6.2f}   {'runaway' if g > 1 else 'stalls'}")
    print("  wrote figures/transition_mechanism.{pdf,png}")


if __name__ == "__main__":
    main()

"""Cosine-similarity version of the representational alignment chart.

Same panel structure as plot_cka_phase_transition.py but with mean per-token
cosine similarity (across c·(c-1)/2 compartment pairs and 4096 token positions
per pair) on the y-axis. Cosine sim is rigid — not invariant to rotations —
so it asks "do the actual vectors point the same way" rather than "is the same
subspace covered" (CKA).
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import (setup_paper_style, C_COLOR, FOURUP_FIGSIZE, FOURUP_ADJUST,
                                      GRID2_FIGSIZE, GRID2_ADJUST)


def load():
    """Main sweep plus the reseed cells, which supply seeds 65/66 of the same
    (c, tr) grid. Kept in a separate file so the default sweep's output is
    untouched; merged here so each point becomes a 3-seed estimate."""
    cells = list(json.loads(Path("cossim_sweep.json").read_text()).values())
    extra = Path("cossim_sweep_reseed.json")
    if extra.exists():
        cells += list(json.loads(extra.read_text()).values())
    return cells


def panel_A(ax, cells):
    rows = [r for r in cells if r["mode"] == "absolute" and r["wd"] == 0
            and r["tr_raw"] < 1.0]
    by_c = defaultdict(dict)
    for r in rows:
        by_c[r["c"]].setdefault(r["tr_eff"], []).append(r["mean_off_diag_cossim"])
    for c in sorted(by_c):
        trs = sorted(by_c[c])
        # Mean with a min-max band. NOT max: with one seed max was that seed,
        # but across three it is best-of-three, which would inflate the
        # apparent alignment for a reason that is not physical -- the mirror of
        # the min() problem in the val-loss panel.
        vals = [float(np.mean(by_c[c][t])) for t in trs]
        lo = [float(np.min(by_c[c][t])) for t in trs]
        hi = [float(np.max(by_c[c][t])) for t in trs]
        col = C_COLOR.get(c, "k")
        ax.fill_between(trs, lo, hi, color=col, alpha=0.18, linewidth=0)
        ax.plot(trs, vals, color=col, marker="o",
                markersize=4, linewidth=1.4, label=f"c={c}")
    ax.set_xlabel("translation ratio")
    ax.set_ylabel("cosine sim. (layer 4)")
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4, linestyle=":")


def panel_wd_for_c(ax, cells, target_c, title=None):
    rows = [r for r in cells if r["mode"] == "absolute" and r["c"] == target_c
            and r["tr_raw"] < 1.0]
    by_tr = defaultdict(dict)
    for r in rows:
        by_tr[r["tr_eff"]].setdefault(r["wd"], []).append(r["mean_off_diag_cossim"])
    cmap = plt.get_cmap("viridis")
    trs = sorted(by_tr)
    for i, tr in enumerate(trs):
        wds = sorted(by_tr[tr])
        vals = [max(by_tr[tr][w]) for w in wds]
        color = cmap(i / max(1, len(trs) - 1))
        ax.plot(wds, vals, color=color, marker="o", markersize=4,
                linewidth=1.4, label=f"tr={tr:g}")
    ax.set_xlabel("wd")
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4, linestyle=":")


def main():
    setup_paper_style()
    cells = load()
    Path("../figures").mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0), sharey=True)
    panel_A(axes[0], cells)
    panel_wd_for_c(axes[1], cells, 5, "(B) c=5: cossim × wd × tr")
    panel_wd_for_c(axes[2], cells, 6, "(C) c=6: cossim × wd × tr")
    panel_wd_for_c(axes[3], cells, 8, "(D) c=8: cossim × wd × tr")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7,
                   handlelength=1.3, handletextpad=0.4, ncol=2)
    for ax in axes[1:]:
        ax.legend(loc="upper left", frameon=False, fontsize=7,
                  handlelength=1.3, handletextpad=0.4, title="translation ratio",
                  title_fontsize=7)
    fig.tight_layout()
    out = Path("../figures/cossim_phase_transition.pdf")
    fig.savefig(out)
    print(f"  {out}")
    plt.close(fig)

    # Section-ready PDFs ----------------------------------------------------
    # §4.1 no-WD: standalone cossim panel A. No title — caption supplies wd=0.
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    panel_A(ax, cells)
    ax.legend(loc="upper left", frameon=False, fontsize=7,
              handlelength=1.3, handletextpad=0.4, ncol=2)
    fig.tight_layout()
    out = Path("../figures/tr_phase_no_wd_cossim.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    # Compact (1/3-column-ready)
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    panel_A(ax, cells)
    ax.legend(loc="upper left",
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.9,
              handlelength=1.0, handletextpad=0.3, ncol=2,
              columnspacing=0.6, borderpad=0.2)
    fig.tight_layout(pad=0.3)
    out = Path("../figures/tr_phase_no_wd_cossim_compact.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    # Four-across variant (see FOURUP_FIGSIZE): same fonts, narrower; one
    # legend column, since a second one covers the rising c=6/c=8 lines.
    fig, ax = plt.subplots(figsize=FOURUP_FIGSIZE)
    panel_A(ax, cells)
    ax.legend(loc="upper left",
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.9,
              handlelength=1.0, handletextpad=0.3, ncol=1,
              borderpad=0.2, labelspacing=0.2)
    fig.subplots_adjust(**FOURUP_ADJUST)
    out = Path("../figures/tr_phase_no_wd_cossim_4up.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    # 2x2 variant (see GRID2_FIGSIZE): legend in two rows of three, since the
    # flattened panel has little height to spare.
    fig, ax = plt.subplots(figsize=GRID2_FIGSIZE)
    panel_A(ax, cells)
    ax.legend(loc="upper left",
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.9,
              handlelength=1.0, handletextpad=0.3, ncol=3,
              columnspacing=0.6, borderpad=0.2)
    fig.subplots_adjust(**GRID2_ADJUST)
    out = Path("../figures/tr_phase_no_wd_cossim_2x2.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    # §4.2 wd: 1x3 strip for c=5, 6, 8 (cossim), shared y; in-axis c-label.
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), sharey=True)
    for ax, c in zip(axes, [5, 6, 8]):
        panel_wd_for_c(ax, cells, c)
        ax.text(0.97, 0.03, f"c={c}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
    axes[0].set_ylabel("cosine sim. (layer 4)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7,
                   handlelength=1.3, handletextpad=0.4,
                   title="translation ratio", title_fontsize=7)
    fig.tight_layout()
    out = Path("../figures/tr_phase_wd_cossim.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

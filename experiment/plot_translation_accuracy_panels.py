"""Exact-match translation accuracy panels replacing the target-half loss curves.

Commitment A2: the target-half validation-loss panels were objected to as an
indirect way of showing the translation task is learned. This plots the direct
quantity -- greedy-decoded exact-match accuracy on held-out translation pairs at
each checkpoint -- and the loss panels move to the appendix.

CONDITIONS AND COLORS ARE INHERITED FROM THE PANELS BEING REPLACED. A reader
comparing the new panel against the old one (or against the other c-indexed
figures in the paper) must see the same line for the same condition, so:

  Fig 3b <- tr_target_trajectory_compact.pdf
      c in {2,3,4,5,6,8} at tr=0.5, wd=0, absolute mode.
      Color: C_COLOR, the shared viridis-over-ALL_C=[1,2,3,4,5,6,8] map used by
      every c-indexed figure in the paper. NOT a local palette.

  Fig 4b <- 1b_tr_trajectories.pdf
      c=8 at tr in {0.056,0.25,0.5,0.75} plus the c=2 tr=0.167 contrast.
      Color: viridis at 0.15+0.7*i/(n-1) for the c=8 family (a *different*
      ramp from C_COLOR -- here the gradient encodes tr, not c) and tab:red
      dashed with square markers for the c=2 contrast, exactly as the original.

The `notrans` (tr=0) controls sit at exactly 0.0 and are drawn in grey. They are
an addition, not part of the original panels: they are the sanity check that the
metric measures translation and not format luck, which matters more for an
accuracy metric than it did for a loss curve.

GEOMETRY IS MATCHED TO THE PANELS BEING REPLACED, not to a general default:
  Fig 3b sits in a 0.32\\textwidth subfigure -> 172.8 x 144 pt -> figsize (2.4, 2.0)
  Fig 4b sits in a 0.49\\textwidth subfigure -> 252 x 165.6 pt -> figsize (3.5, 2.3)
Plain savefig, no bbox_inches="tight": tight sizes the canvas to content, so the
output width would drift with label length and the type would no longer match
its neighbours after LaTeX rescales it.

Legends sit upper-left: accuracy rises from 0, so unlike the loss panels these
replace, the empty corner is top-left rather than bottom-left.

Data: experiment/transacc_*.json, one record per checkpoint with an `aggregate`
dict carrying `exact_match`.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figures"
sys.path.insert(0, str(HERE))
from plot_baseline_val_curves import (setup_paper_style, C_COLOR, FOURUP_FIGSIZE,  # noqa: E402
                                      FOURUP_ADJUST, GRID2_FIGSIZE, GRID2_ADJUST)

CONTROL_GREY = "#999999"


def series(fname):
    try:
        recs = json.loads((HERE / fname).read_text())
    except Exception:
        return None
    recs = [r for r in recs if "aggregate" in r and "exact_match" in r["aggregate"]]
    recs.sort(key=lambda r: r["step"])
    return [r["step"] for r in recs], [100.0 * r["aggregate"]["exact_match"] for r in recs]


def panel(specs, figsize, out, *, legend_kw, ymax, pad, adjust=None):
    setup_paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    drew = 0
    for spec in specs:
        fname, label, col, ls, marker = spec[:5]
        seed_files = spec[5] if len(spec) > 5 else []
        s = series(fname)
        if not s or not s[0]:
            print(f"    MISSING {fname}  ({label})")
            continue
        # Min-max band across seeds, on the steps every seed actually has.
        # The reseed grid skips step 850, so intersect rather than assume a
        # common grid -- zipping mismatched step lists would silently pair a
        # seed's step-3500 accuracy against another's step-850.
        avail = [series(f) for f in seed_files]
        avail = [a for a in avail if a and a[0]]
        if avail:
            maps = [dict(zip(*a)) for a in ([s] + avail)]
            common = sorted(set.intersection(*(set(d) for d in maps)))
            if common:
                lo = [min(d[x] for d in maps) for x in common]
                hi = [max(d[x] for d in maps) for x in common]
                ax.fill_between(common, lo, hi, color=col, alpha=0.20, linewidth=0)
        ax.plot(s[0], s[1], ls, color=col, lw=1.3, marker=marker, ms=2.5, label=label)
        drew += 1
    if not drew:
        print(f"  nothing to draw for {out}"); plt.close(fig); return
    ax.set_xscale("log")
    # Accuracy rises from 0 and every curve saturates at 100 only on the right,
    # so the upper-LEFT corner is genuinely empty and the legend goes there with
    # the axes at their natural range. ymax stays a parameter because the legend
    # column count differs per panel and a crowded one may need headroom; ticks
    # stop at 100 so any such band reads as margin rather than as data range.
    ax.set_ylim(-3, ymax)
    ax.set_yticks(range(0, 101, 25))
    ax.set_xlabel("step")
    # Short label deliberately: at the 2.4in width of the 3b panel the rotated
    # "exact-match accuracy (%)" is taller than the axes and gets clipped.
    ax.set_ylabel("exact match (\\%)" if matplotlib.rcParams.get("text.usetex")
                  else "exact match (%)")
    ax.grid(alpha=0.3)
    ax.legend(**legend_kw)
    # Padding must match the panel's NEIGHBOUR, not be a house default: equal
    # figsize is not enough, since tight_layout pad decides where the axes box
    # sits inside that canvas. Fig 3's other panels use pad=0.3 and the 1B
    # figure's other panel uses matplotlib's default, so a single value here
    # would visibly misalign one figure or the other.
    if adjust is not None:
        fig.subplots_adjust(**adjust)
    else:
        fig.tight_layout(**({} if pad is None else {"pad": pad}))
    fig.savefig(FIGS / out)
    # PNG alongside the PDF purely for review/preview; the PDF is the artifact.
    fig.savefig((FIGS / out).with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {out}  ({drew}/{len(specs)} series)")


def placeholder_4up(out, figsize=FOURUP_FIGSIZE, adjust=FOURUP_ADJUST):
    """Empty panel with the four-across (or 2x2) geometry, so Fig. 4 can be
    laid out before its fourth panel exists."""
    setup_paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel(" "); ax.set_ylabel(" ")
    ax.text(0.5, 0.5, "panel (d)", transform=ax.transAxes,
            ha="center", va="center", color="0.6")
    fig.subplots_adjust(**adjust)
    fig.savefig(FIGS / out)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    # ---- Fig 3b replacement: 14.7M, tr=0.5, one line per c ------------------
    # Same six compartment counts as tr_target_trajectory_compact.pdf.
    # Seed band: s64 is the line, s65/s66 come from the reseed grid. All three
    # saturate above 99.6%, so the band is narrow by construction -- it is here
    # to show that, not because it is expected to be wide.
    def _seed_files(c):
        return [f"transacc_reseed_8-256-n{c}-tr05abs-s{s}.json" for s in (65, 66)]
    fig3b = [(f"transacc_8256_n{c}-tr05.json", f"c={c}", C_COLOR[c], "-", "o",
              _seed_files(c)) for c in (2, 3, 4, 5, 6, 8)]
    # Legend at 7.5pt like its neighbours (panel c uses the 7.5 default, panel
    # a 7.3); it was 6.5 and read visibly smaller on the page.
    fig3b_legend = dict(loc="upper left", frameon=False, fontsize=7.5,
                        handlelength=1.0, handletextpad=0.3, ncol=2,
                        columnspacing=0.8, borderpad=0.2)
    panel(fig3b, (2.4, 2.0), "transacc_8_256_em_compact.pdf", ymax=103, pad=0.3,
          legend_kw=fig3b_legend)
    # Four-across variant (see FOURUP_FIGSIZE): same fonts, narrower.
    panel(fig3b, FOURUP_FIGSIZE, "transacc_8_256_em_4up.pdf", ymax=103, pad=None,
          adjust=FOURUP_ADJUST,
          # one column: at this width a second column sits on the rising curves
          legend_kw=dict(fig3b_legend, ncol=1, labelspacing=0.2))
    placeholder_4up("fig4d_placeholder_4up.pdf")
    # 2x2 variant (see GRID2_FIGSIZE): room for the compact two-column legend.
    panel(fig3b, GRID2_FIGSIZE, "transacc_8_256_em_2x2.pdf", ymax=103, pad=None,
          adjust=GRID2_ADJUST, legend_kw=fig3b_legend)
    placeholder_4up("fig4d_placeholder_2x2.pdf", GRID2_FIGSIZE, GRID2_ADJUST)

    # ---- Fig 4b replacement: 1B, c=8 across tr, + c=2 contrast --------------
    cmap = plt.get_cmap("viridis")
    c8 = [(0.056, "transacc_1b_8comp.json"),
          (0.25,  "transacc_1b_8comp-tr025abs.json"),
          (0.5,   "transacc_1b_8comp-tr05abs.json"),
          (0.75,  "transacc_1b_8comp-tr075abs.json")]
    fig4b = [(f, f"c=8, tr={tr:g}", cmap(0.15 + 0.7 * i / max(1, len(c8) - 1)), "-", "o")
             for i, (tr, f) in enumerate(c8)]
    # c=2 contrast: converges faster despite less translation data.
    fig4b.append(("transacc_1b_2comp.json", "c=2, tr=0.167", "tab:red", "--", "s"))
    # tr=0 control: pins the metric at 0, proving it is not format luck.
    fig4b.append(("transacc_1b_8comp-notrans.json", "c=8, tr=0", CONTROL_GREY, ":", ""))
    panel(fig4b, (3.5, 2.3), "transacc_1b_em.pdf", ymax=103, pad=None,
          legend_kw=dict(loc="upper left", frameon=False, fontsize=6.5,
                         handlelength=1.3, handletextpad=0.4, ncol=1,
                         columnspacing=0.8, borderpad=0.2))

    for f in ("transacc_8_256_em_compact.pdf", "transacc_1b_em.pdf"):
        p = FIGS / f
        print(f"  {f}: {'exists' if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()

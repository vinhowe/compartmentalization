"""Paper figure: downstream accuracy as a function of validation loss.

Presentation follows Du et al. (arXiv:2403.15796): a plain scatter -- no trend
lines, no connecting lines between a run's checkpoints -- with loss running
HIGH to LOW left to right, so better models sit to the right and each panel
reads up-and-to-the-right. The x-axis is inverted rather than negated so the
tick labels stay real loss values.

NO CORRELATION STATISTIC IS ANNOTATED. An earlier version put a Spearman rho in
each panel title; that was our own addition and appears in neither Du et al. nor
Gadre et al., so it is gone. The scatter carries the claim on its own.

COLOR ENCODES MODEL SIZE, NOTHING ELSE. Compartment count and translation ratio
are deliberately absorbed into the size color: the claim is that accuracy is a
function of loss *regardless* of how a model got to that loss, so distinguishing
the ways it got there works against the point. Three sizes carry the argument
(8-256, 8-512, ~1B); the other configurations are subsets of these.

THE THICK LINES ARE THE c=1 TRAJECTORIES, one per size, drawn from that model's
own checkpoints across training. They are the reference curve: they show what
loss-to-accuracy looks like when nothing is compartmentalized. A compartmentalized
point landing ON its size's line means its downstream deficit is entirely
explained by its loss deficit. They are stroked with a white outline so they stay
legible where they pass through the scatter.

TWO FILTERS, BOTH DELIBERATE:
  * tr=1.0 runs are excluded. With every token in a translation pair,
    compartment 0 never sees plain text and its val loss is meaningless -- these
    sit at 14-20 nats at step 1e6, worse than uniform over the 16k vocab, and
    would stretch the x-axis by an order of magnitude. The paper's phase-
    transition figure already filters tr<1.0 for the same reason.
  * The scatter is step 1e6 only, so every point is a finished model. Eleven
    8-512 tr-sweep runs stop between 7k and 500k steps and are reported as
    dropped rather than silently mixed in at a different amount of training.
The c=1 reference lines are NOT restricted to 1e6 -- their whole purpose is to
trace the path, and the axis limits (set from the scatter) clip them naturally.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_ROOT = REPO / "out" / "translation-compression"
FIGS = REPO / "figures"
sys.path.insert(0, str(HERE))
from plot_baseline_val_curves import setup_paper_style  # noqa: E402

TASKS = ["hellaswag", "arc_easy", "piqa", "sciq", "lambada"]
CHANCE = {"hellaswag": 25.0, "arc_easy": 25.0, "piqa": 50.0, "sciq": 25.0,
          "lambada": 0.0}
NICE = {"hellaswag": "HellaSwag", "arc_easy": "ARC-e", "piqa": "PIQA",
        "sciq": "SciQ", "lambada": "LAMBADA"}
FINAL_STEP = 1000000
CHANCE_OF = dict(CHANCE, _margin=0.0)
X_SCALE = "linear"  # Du et al. use a linear loss axis

# Sequential by size so the legend reads small -> large.
SIZE_ORDER = ["8-256", "8-512", "~1B"]
SIZE_COLOR = {"8-256": "#4C9BE8", "8-512": "#E8913A", "~1B": "#111111"}
# Config names as Table~\ref{tab:model-sizes} lists them, annotated with the
# c=1 parameter count. Those counts are only meaningful because the legend also
# says the LINE is the c=1 model: the dots at the same color are c>=2 and carry
# up to 5x the parameters, so an unqualified "14.7M" next to the color would be
# wrong for most of what it labels.
SIZE_LABEL = {"8-256": "8-256 (14.7M)", "8-512": "8-512 (42.0M)",
              "~1B": "24-1792 (983.7M)"}

# ---- type sizing -------------------------------------------------------------
# This figure is 5.5in wide and included at \textwidth (measured: 397.485pt =
# 5.5000in exactly), so LaTeX rescales it by 1.0 and every fontsize below is a
# real on-page point size. Every OTHER figure in the paper is authored larger
# than its slot and shrunk -- a 0.49\textwidth panel drawn at 3.5in is scaled by
# 0.770 -- so setup_paper_style's nominal 9/8/7.5pt land at ~6.9/6.2/5.8pt on
# the page. Inheriting those nominal sizes here would render this figure ~30%
# larger than its neighbours; using a hand-picked "looks small enough" number
# made it ~10% smaller. So derive them: same on-page size as a 0.49 panel.
PAPER_SCALE = 0.49 * 5.5 / 3.5          # 0.770, what a 0.49\textwidth panel gets
TITLE_FS = 9.0 * PAPER_SCALE            # axes.titlesize
LABEL_FS = 9.0 * PAPER_SCALE            # axes.labelsize
TICK_FS = 8.0 * PAPER_SCALE             # x/ytick.labelsize
LEG_FS = 7.5 * PAPER_SCALE              # legend.fontsize


def classify(nl, ne):
    if (nl, ne) == (8, 256):
        return "8-256"
    if (nl, ne) == (8, 512):
        return "8-512"
    return "~1B"


def load():
    vm = json.loads((HERE / "val_metrics.json").read_text())
    vloss = {}
    for k, v in vm.items():
        m = v["metrics"].get("loss_compartment_0")
        if m and len(m) == len(v["checkpoints"]):
            vloss[k] = dict(zip(v["checkpoints"], m))

    # Only models the paper itself uses. Globbing benchtraj_*.json picks up
    # whatever is on disk -- including bpe16384-rope-8-512-tr-abs-sweep, an
    # abandoned 13-run sweep no paper figure references, and tr=1.0 cells. See
    # build_paper_run_inventory.py, which derives this list by walking
    # paper/*.tex -> generator scripts -> run keys.
    allow = set(json.loads((HERE / "paper_run_inventory.json").read_text()))
    rows, dropped_tr1, truncated, off_paper = [], set(), [], set()
    for f in sorted(glob.glob(str(HERE / "benchtraj_*.json"))):
        recs = [r for r in json.load(open(f)) if "tasks" in r]
        if not recs:
            continue
        rk = recs[0]["run"]
        cf = OUT_ROOT / rk / "meta" / "config.json"
        if not cf.exists() or rk not in vloss:
            continue
        if rk not in allow:
            off_paper.add(Path(f).name)
            continue
        cfg = json.load(open(cf))
        e = cfg["experiment"]
        nc = e["n_compartments"]
        tr = e["translation_ratio"]
        eff = tr / (nc + 1) if e.get("translation_ratio_mode") == "compartment" else tr
        if abs(eff - 1.0) < 1e-9:
            dropped_tr1.add(Path(f).name)
            continue
        size = classify(cfg["model"]["n_layer"], cfg["model"]["n_embd"])
        steps = [r["step"] for r in recs]
        if max(steps) < FINAL_STEP and nc > 1:
            truncated.append((Path(f).name, max(steps)))
        for r in recs:
            if r["step"] not in vloss[rk]:
                continue
            per = {t: 100 * (r["tasks"][t]["acc_norm"] if t != "lambada"
                             else r["tasks"][t]["acc"])
                   for t in TASKS if t in r["tasks"]}
            if len(per) < len(TASKS):
                continue
            per["_margin"] = float(np.mean([per[t] - CHANCE[t] for t in TASKS]))
            rows.append(dict(size=size, c=nc, step=r["step"],
                             loss=vloss[rk][r["step"]], per=per))
    return rows, sorted(dropped_tr1), sorted(truncated), sorted(off_paper)


def main():
    rows, dropped_tr1, truncated, off_paper = load()
    if not rows:
        print("  no data"); return

    # EVERY checkpoint of every compartmentalized run, not only the finished
    # models. Each (loss, accuracy) pair is a valid observation of the relation
    # regardless of how much training produced it -- that is the point of
    # plotting against loss rather than against compute -- and the intermediate
    # points are what show the chance plateau and the rise out of it.
    scatter = [r for r in rows if r["c"] > 1]
    print(f"  {len(scatter)} checkpoints (c>=2) in the scatter, "
          f"{len([r for r in scatter if r['step'] == FINAL_STEP])} of them finished")
    print(f"  dropped {len(off_paper)} runs not used by any paper figure:")
    for n in off_paper:
        print(f"      {n}")
    if truncated:
        # No longer excluded: the step-1e6 restriction was the only reason these
        # were dropped, and it is gone. Kept as a printout so their short length
        # stays visible rather than being quietly folded in.
        print(f"  {len(truncated)} runs stop short of {FINAL_STEP}; their "
              f"checkpoints are included as ordinary points")

    # c=1 reference trajectory per size, ordered along its own training.
    ref = {}
    for s in SIZE_ORDER:
        pts = sorted([r for r in rows if r["size"] == s and r["c"] == 1],
                     key=lambda r: r["step"])
        if pts:
            ref[s] = pts
            print(f"  c=1 reference {s}: {len(pts)} checkpoints, "
                  f"loss {pts[-1]['loss']:.3f} at step {pts[-1]['step']}")

    setup_paper_style()
    panels = [("_margin", "margin over chance (pp)", "Aggregate")] + \
             [(t, "accuracy (\\%)" if matplotlib.rcParams.get("text.usetex")
               else "accuracy (%)", NICE[t]) for t in TASKS]

    # 5.5in = NeurIPS \textwidth: included at scale 1.0, so type is unscaled.
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.7), sharex=True)

    lo = min(r["loss"] for r in scatter)
    hi = max(r["loss"] for r in scatter)
    pad = 0.04 * (hi - lo)

    for ax, (key, ylab, title) in zip(axes.ravel(), panels):
        for s in SIZE_ORDER:
            pts = [r for r in scatter if r["size"] == s]
            if pts:
                # Above the reference lines (zorder), with a thin white ring so a
                # point sitting exactly on its own size's curve -- which is the
                # expected outcome, and the whole finding -- still reads as a
                # point rather than dissolving into the line.
                ax.scatter([r["loss"] for r in pts], [r["per"][key] for r in pts],
                           s=4.5, color=SIZE_COLOR[s], zorder=3, alpha=0.45,
                           edgecolor="none")
        for s in SIZE_ORDER:
            if s not in ref:
                continue
            # Reference curves sit BEHIND the data. They are context for reading
            # the scatter, not the subject: drawn on top at 1.9pt plus a 3.2pt
            # white halo they occluded most of the points they were meant to
            # explain. Thinner, and underneath.
            ax.plot([r["loss"] for r in ref[s]], [r["per"][key] for r in ref[s]],
                    color=SIZE_COLOR[s], lw=1.3, zorder=6, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=2.5, foreground="white"),
                                  pe.Normal()])
        # Chance floor. With intermediate checkpoints in view most of the mass
        # sits on it, so without the line the flat left-hand band reads as a
        # weak trend rather than as "not yet above chance".
        ax.axhline(CHANCE_OF[key], color="0.55", lw=0.6, ls=(0, (2.5, 2)),
                   zorder=2)
        ax.set_title(title, fontsize=TITLE_FS, pad=2.5)
        ax.set_ylabel(ylab, fontsize=LABEL_FS)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=TICK_FS)
        # Integer ticks everywhere. Left to itself matplotlib picks 2.5-unit
        # steps on ARC-e and PIQA and whole units elsewhere, so those two panels
        # get an extra ".5" digit of tick-label width and shove their y-label
        # into the panel on their left. Same tick width in every panel also
        # keeps the six axes boxes the same size.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        # Limits come from the SCATTER, not the reference lines: the c=1
        # trajectories start above 10 nats and would otherwise compress every
        # finished model into the right-hand tenth of the panel.
        # Log loss axis: with every checkpoint plotted the range is 3.1-11.5
        # nats, and on a linear axis the converged models -- the comparison the
        # figure exists to make -- are crushed into the right fifth while the
        # left half is an empty chance plateau. Log gives the low-loss end room
        # without discarding the plateau. Ticks stay real loss values.
        if X_SCALE == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks([3, 4, 6, 8, 11])
        ax.set_xlim(hi + pad, lo - pad)

    for ax in axes[1]:
        ax.set_xlabel("compartment-0 val loss (nats)", fontsize=LABEL_FS)

    # Legend placement, in axes fractions. Inset from the corner rather than
    # flush to it: at (0.02, 1.0) the "model size" title sat on the top spine.
    # The second legend is placed from the FIRST one's measured height rather
    # than a hand-tuned offset: matplotlib cannot stack one legend beneath
    # another, and a literal offset silently breaks whenever the font size or
    # the number of size entries changes (it did, twice -- once overlapping the
    # data, once colliding with the block above).
    LEG_X, LEG_TOP, LEG_GAP = 0.035, 0.968, 0.035

    # Two legends on one Axes, each with a native title. This is matplotlib's
    # documented way to show two independent encodings ("Multiple legends on the
    # same Axes"): build each with proxy artists, and re-add the first with
    # add_artist because a second legend() call would otherwise replace it. The
    # title= parameter is the supported group heading -- an earlier revision
    # faked headings and a divider with invisible-handle text rows and a
    # box-drawing rule, which is not a real divider and read as one.
    common = dict(frameon=False, fontsize=LEG_FS, title_fontsize=LEG_FS,
                  handletextpad=0.4, borderpad=0.0, labelspacing=0.25,
                  handlelength=1.5, borderaxespad=0.0, alignment="left")
    size_h = [plt.Line2D([], [], color=SIZE_COLOR[s], lw=1.6,
                         label=SIZE_LABEL[s]) for s in SIZE_ORDER if s in ref]
    shape_h = [
        plt.Line2D([], [], color="0.35", lw=1.6, label="$c=1$, across training"),
        plt.Line2D([], [], color="0.35", lw=0, marker="o", markersize=2.6,
                   markeredgecolor="none", alpha=0.6,
                   label="$c\\geq2$, all checkpoints"),
        plt.Line2D([], [], color="0.55", lw=0.6, ls=(0, (2.5, 2)),
                   label="chance"),
    ]
    ax0 = axes[0][0]
    leg1 = ax0.legend(handles=size_h, title="model size", loc="upper left",
                      bbox_to_anchor=(LEG_X, LEG_TOP), **common)
    ax0.add_artist(leg1)
    # Needs a renderer before the extent is known.
    fig.canvas.draw()
    bb = leg1.get_window_extent().transformed(ax0.transAxes.inverted())
    ax0.legend(handles=shape_h, title="series", loc="upper left",
               bbox_to_anchor=(LEG_X, bb.y0 - LEG_GAP), **common)

    fig.tight_layout(pad=0.3)
    out = FIGS / "loss_vs_downstream_paper.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()

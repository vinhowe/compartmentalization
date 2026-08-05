"""Capacity-competition ladder: how much English capacity does the neighbour take?

The existing "noise" compartments are a weak control, and the numbers show why:
at 200k steps uniform noise costs English +0.048 nats and unigram
noise +0.047, while a real second language costs +0.229. That is a two-point jump
from "trivial" to "real" with nothing in between.

This interpolates it. Compartment 1 is filled with order-n n-gram text generated
from the SAME FineWeb corpus, for n = 2, 3, 4, so the neighbour becomes
progressively more English-like. If capacity competition is driven by how much
structure the neighbour has, the English-side loss should climb monotonically from
the noise floor toward the real-language ceiling as n rises.

Left  -- English-side (compartment 0) val loss vs step, one line per rung.
Right -- the same thing as a gap against the c=1 English-only baseline at matched
         steps, which is the quantity the claim is about.

Everything is measured through the formal fineweb_val_metrics pipeline and read
from val_metrics.json, NOT from training-time logs: the log's "val loss" is a
single number averaged over both compartments, so it mixes English with synthetic
and cannot answer this question.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
OUT = HERE.parent / "out" / "translation-compression"

sys.path.insert(0, str(HERE))
from plot_baseline_val_curves import setup_paper_style  # noqa: E402

# label, glob for the run key, colour, style.  Existing rungs are matched by
# directory glob; the new ones live in capacity-ngram-ladder.
RUNGS = [
    ("c=1 English only",      "synthetic-compartment-baselines/*english-baseline-rope-bpe16384-8-256*", "black",      "-"),
    ("+ uniform noise",       "synthetic-compartment-baselines/*english-uniform-2comp*8-256*",          "#bbbbbb",    "-"),
    ("+ unigram (n=1)",       "synthetic-compartment-baselines/*english-frequency-2comp*8-256*",        "#7fb3d5",    "-"),
    ("+ n-gram n=1.5",        "capacity-ngram-ladder/8-256-c2-english-vs-ngram1p5",                     "#2e86c1",    ":"),
    # n=2/3/4 are the clean re-runs. The originals hit a checkpoint-loading
    # fault on restart that reset Adam at step 50k; these were retrained from
    # scratch at the same seed (configs are otherwise identical), which cost
    # 0.005-0.012 nats and left the ordering unchanged.
    ("+ n-gram n=2",          "capacity-ngram-ladder/8-256-c2-english-vs-ngram2",             "#4a9f45",    "-."),
    ("+ n-gram n=2.5",        "capacity-ngram-ladder/8-256-c2-english-vs-ngram2p5",                     "#7dcea0",    ":"),
    ("+ n-gram n=3",          "capacity-ngram-ladder/8-256-c2-english-vs-ngram3",             "#d98c00",    "-."),
    ("+ n-gram n=3.5",        "capacity-ngram-ladder/8-256-c2-english-vs-ngram3p5",                     "#e59866",    ":"),
    ("+ n-gram n=4",          "capacity-ngram-ladder/8-256-c2-english-vs-ngram4",             "#c0392b",    "-."),
    ("+ Russian (real lang)", "*/*russian-english-baseline-rope*",                                      "purple",     "--"),
]


def load_metrics() -> dict:
    m = {}
    p = HERE / "val_metrics.json"
    if p.exists():
        m.update(json.loads(p.read_text()))
    for f in sorted(glob.glob(str(HERE / "val_metrics_gpu*.json"))):
        try:
            m.update(json.loads(Path(f).read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    for f in sorted(glob.glob(str(HERE / ".eval_ladder" / "val_metrics_gpu*.json"))):
        try:
            m.update(json.loads(Path(f).read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    return m


def english_curve(m: dict, pattern: str):
    """compartment-0 (English) loss for the first run key matching pattern."""
    import fnmatch
    for k in m:
        if fnmatch.fnmatch(k, pattern) or k == pattern:
            v = m[k]
            a = v["metrics"].get("loss_compartment_0")
            if not a:
                continue
            x = np.array(v["checkpoints"][:len(a)], float)
            y = np.array(a, float)
            o = np.argsort(x)
            return x[o], y[o]
    return None



# Rungs that appear in the PAPER figure. The fractional rungs (n=1.5/2.5/3.5)
# are deliberately excluded: they are Jelinek-Mercer interpolations we ran to
# check that the ordering is smooth rather than an artifact of four coarse
# points, which is a diagnostic, not a result the paper claims. They remain in
# the monitoring view above.
PAPER_RUNGS = {
    "c=1 English only", "+ uniform noise", "+ unigram (n=1)",
    "+ n-gram n=2", "+ n-gram n=3", "+ n-gram n=4", "+ Russian (real lang)",
}


def _paper_panels(curves, base):
    """Emit the same three panels as separate paper-sized PDFs.

    The combined 14.6in canvas above is a monitoring view: dropped into a
    0.49\\textwidth subfigure it would scale to ~0.22 and its type would render
    at a third the size of every neighbouring figure. Each panel is therefore
    re-emitted at figsize (3.3, 2.6) -- 237.6pt, identical to figures/copyemb_*.pdf
    -- so at 0.49\\textwidth the scale is 0.817 and 9pt type stays 9pt.

    NO bbox_inches="tight": it sizes output to content, so a longer legend label
    silently widens the PDF and shrinks everything once LaTeX rescales it.

    The legend is a separate full-width strip rather than living inside a panel:
    ten rungs at a legible size would occupy over half of a 2.6in panel.
    """
    import matplotlib.pyplot as _plt

    curves = {k: v for k, v in curves.items() if k in PAPER_RUNGS}

    def _fresh():
        fig, ax = _plt.subplots(figsize=(3.3, 2.6))
        return fig, ax

    # (a) English-side loss
    fig, ax = _fresh()
    for lab, ((x, y), col, ls) in curves.items():
        ax.plot(x, y, ls, color=col, lw=1.3, marker="o", ms=2.5, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("English val loss (nats)")
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "ngram_ladder_loss.pdf")
    _plt.close(fig)

    if base is None:
        return
    bx, by = base[0]
    bmap = dict(zip(bx, by))

    # (b) gap vs the c=1 baseline
    fig, ax = _fresh()
    for lab, ((x, y), col, ls) in curves.items():
        if lab == "c=1 English only":
            continue
        shared = [s for s in x if s in bmap]
        if not shared:
            continue
        ymap = dict(zip(x, y))
        ax.plot(shared, [ymap[s] - bmap[s] for s in shared], ls,
                color=col, lw=1.3, marker="o", ms=2.5, label=lab)
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"English loss $-$ c=1 (nats)")
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "ngram_ladder_gap.pdf")
    _plt.close(fig)

    # (c) slowdown at matched English loss -- the axis Fig 2b uses
    def step_at(x, y, L):
        if not (y.min() <= L <= y.max()):
            return None
        return float(np.interp(L, y[::-1], x[::-1]))

    fig, ax = _fresh()
    handles, labels = [], []
    for lab, ((x, y), col, ls) in curves.items():
        if lab == "c=1 English only":
            continue
        lo, hi = max(by.min(), y.min()), min(min(by.max(), y.max()), 4.6)
        if hi <= lo:
            continue
        pts = []
        for L in np.linspace(hi, lo, 24):
            sb, sc = step_at(bx, by, L), step_at(x, y, L)
            if sb and sc and sb > 0:
                pts.append((L, sc / sb))
        if pts:
            a = np.array(pts)
            line, = ax.plot(a[:, 0], a[:, 1], ls, color=col, lw=1.3, label=lab)
            handles.append(line); labels.append(lab)
    ax.axhline(1.0, color="black", lw=0.8, alpha=0.6)
    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xlabel("English val loss target (nats)")
    ax.set_ylabel(r"slowdown vs c=1 ($\times$)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "ngram_ladder_slowdown.pdf")
    _plt.close(fig)

    # legend strip, authored at ~full textwidth so it scales like the panels
    lfig = _plt.figure(figsize=(6.75, 0.5))
    lh, ll = [], []
    for lab, ((x, y), col, ls) in curves.items():
        lh.append(Line2D([], [], color=col, linestyle=ls, lw=1.3))
        ll.append(lab)
    lfig.legend(lh, ll, loc="center", ncol=5, frameon=False,
                handlelength=1.6, columnspacing=1.1)
    lfig.savefig(FIGS / "ngram_ladder_legend.pdf")
    _plt.close(lfig)
    print("  wrote paper panels: ngram_ladder_{loss,gap,slowdown,legend}.pdf")


def main():
    setup_paper_style()
    m = load_metrics()
    fig, (axL, axR, axS) = plt.subplots(1, 3, figsize=(14.6, 3.9))

    curves = {}
    for lab, pat, col, ls in RUNGS:
        c = english_curve(m, pat)
        if c is None:
            print(f"  (no data yet) {lab}")
            continue
        curves[lab] = (c, col, ls)
        axL.plot(c[0], c[1], ls, color=col, lw=1.7, marker="o", ms=3, label=lab)
    axL.set_xscale("log")
    axL.set_xlabel("training step")
    axL.set_ylabel("English val loss (nats)")
    axL.set_title("English-side loss, compartment 0", fontsize=10)
    axL.grid(alpha=0.3)
    axL.legend(fontsize=7, frameon=False)

    base = curves.get("c=1 English only")
    if base is not None:
        bx, by = base[0]
        bmap = dict(zip(bx, by))
        print(f"\n  {'rung':<24}" + "".join(f"{s//1000:>8}k" for s in (29000, 60000, 120000, 200000)))
        for lab, ((x, y), col, ls) in curves.items():
            if lab == "c=1 English only":
                continue
            shared = [s for s in x if s in bmap]
            if not shared:
                continue
            d = np.array([dict(zip(x, y))[s] - bmap[s] for s in shared])
            axR.plot(shared, d, ls, color=col, lw=1.7, marker="o", ms=3, label=lab)
            row = "".join(
                f"{dict(zip(shared, d))[s]:>+9.3f}" if s in dict(zip(shared, d)) else f"{'-':>9}"
                for s in (29000, 60000, 120000, 200000))
            print(f"  {lab:<24}{row}")
        axR.axhline(0, color="black", lw=0.8, alpha=0.6)
    axR.set_xscale("log")
    axR.set_xlabel("training step")
    axR.set_ylabel("English loss $-$ c=1 baseline (nats)")
    axR.set_title("Capacity taken from English", fontsize=10)
    axR.grid(alpha=0.3)
    axR.legend(fontsize=7, frameon=False, loc="upper right")

    # Third panel: SLOWDOWN at matched English loss -- the axis Fig 2b of the paper
    # uses. The loss-gap panels above are not directly comparable to it, and the
    # published claim ("with noise, English matches baseline sample efficiency")
    # is a slowdown claim, so the new rungs have to be shown the same way.
    if base is not None:
        bx, by = base[0]

        def step_at(x, y, L):
            if not (y.min() <= L <= y.max()):
                return None
            return float(np.interp(L, y[::-1], x[::-1]))

        for lab, ((x, y), col, ls) in curves.items():
            if lab == "c=1 English only":
                continue
            lo = max(by.min(), y.min())
            hi = min(by.max(), y.max())
            if hi <= lo:
                continue
            hi = min(hi, 4.6)      # early high-loss region is all ~1x; Fig 2b
            if hi <= lo:           # shows the converged range only
                continue
            targets = np.linspace(hi, lo, 24)
            pts = []
            for L in targets:
                sb, sc = step_at(bx, by, L), step_at(x, y, L)
                if sb and sc and sb > 0:
                    pts.append((L, sc / sb))
            if pts:
                a = np.array(pts)
                axS.plot(a[:, 0], a[:, 1], ls, color=col, lw=1.7, label=lab)
        axS.axhline(1.0, color="black", lw=0.8, alpha=0.6)
        axS.invert_xaxis()          # training runs left -> right, as in Fig 2b
        axS.set_yscale("log")
    axS.set_xlabel("English val loss target (nats)")
    axS.set_ylabel(r"slowdown vs c=1 ($\times$)")
    axS.set_title("Slowdown at matched English loss", fontsize=10)
    axS.grid(alpha=0.3, which="both")
    axS.legend(fontsize=6.5, frameon=False, loc="upper left")

    _paper_panels(curves, base)

    fig.suptitle("Capacity competition rises with the neighbour's n-gram order", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGS / "ngram_ladder.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

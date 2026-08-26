"""Is tr=0.25 the same transition, just later?

The tr=0.25 cossim curve is still rising at 1M steps, which invites the reading
that the transition happens for tr=0.25 too and the run simply stops before it.
This script tests that directly.

If tr=0.25 were the tr=0.5 transition delayed by a constant factor, then

  * the delay measured at each cossim threshold would be the SAME factor, and
  * sliding the tr=0.5 curve right in log-step by that factor would land it on
    top of the tr=0.25 curve.

Neither holds. The delay grows with the threshold (10.7x at cos=0.15, 31.4x at
cos=0.25) and the best single shift leaves a residual worth a quarter of the
fitted range. So tr=0.25 is not tr=0.5 translated in time; it is both later and
slower.

What DOES support the reading: over the last half-decade the tr=0.25 curve
accelerates (+0.058 -> +0.175 per decade) while tr=0.5 decelerates into its
ceiling. Acceleration is what the foot of a sigmoid looks like, and it is not an
artifact of the schedule -- these runs hold the learning rate constant at 2e-5
after a 1000-step warmup (decay_lr=False), so nothing about the end of training
is special.

What argues against it, and this is the part that keeps the question open: the
near-zero-tr run (tr_eff~0.011) accelerates late in exactly the same way, its
slope roughly tripling from ~0.035 to ~0.12 per decade between 3e4 and 3e6 --
and that run is plainly not transitioning, sitting at cos 0.134 after 2.95M
steps. So late acceleration on its own is NOT diagnostic of an oncoming
transition; some of it is just the slow drift every compartmentalised run shows,
which curves upward mildly in log-time.

1M steps cannot separate the two. The decisive experiment is to extend tr=0.25.

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

CACHE = Path("../cache_twinclouds")
FIGDIR = Path("../figures")
MAIN_EXP = Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression/experiment")

C25, C05 = "#c2410c", "#1d4ed8"
CLOW = "#6b7280"


def cossim_curve(name: str):
    d = np.load(CACHE / f"traj_{name}_L4.npz")
    S = d["sub_feats"].astype(np.float64)
    c = int(d["c"])
    out = []
    for t in range(S.shape[0]):
        F = S[t]
        Fn = F / (np.linalg.norm(F, axis=2, keepdims=True) + 1e-12)
        out.append(np.mean([(Fn[i] * Fn[j]).sum(1).mean()
                            for i in range(c) for j in range(i + 1, c)]))
    return d["steps"].astype(float), np.array(out)


def low_tr_curve():
    """The tr_eff~0.011 run, trained to 2.95M -- the long-horizon reference."""
    a = json.loads((MAIN_EXP / "cossim_across_training.json").read_text())
    rows = sorted((v["step"], v["cossim"]) for v in a.values()
                  if v["label"] == "baseline_n8")
    s, c = zip(*rows)
    return np.array(s, dtype=float), np.array(c)


def first_cross(s, c, thr):
    idx = np.where(c >= thr)[0]
    return None if len(idx) == 0 else s[idx[0]]


def best_shift(s25, c25, s05, c05, lo=0.10):
    hi = c25.max()
    m = (c25 >= lo) & (c25 <= hi)
    grid = np.linspace(0.0, 2.5, 501)
    errs = [np.sqrt(np.mean((np.interp(np.log10(s25[m]) - sh,
                                       np.log10(s05), c05) - c25[m]) ** 2))
            for sh in grid]
    i = int(np.argmin(errs))
    return grid[i], errs[i], (lo, hi)


def local_slope(s, c, win=0.5):
    """d(cos)/d(log10 step), fit in a sliding log-step window."""
    ls = np.log10(s)
    out = np.full_like(c, np.nan)
    for i in range(len(c)):
        m = np.abs(ls - ls[i]) <= win / 2
        if m.sum() >= 3:
            out[i] = np.polyfit(ls[m], c[m], 1)[0]
    return out


def main():
    setup_paper_style()
    FIGDIR.mkdir(exist_ok=True)

    s25, c25 = cossim_curve("tr025")
    s05, c05 = cossim_curve("tr05")
    sl, cl = low_tr_curve()
    sh, err, (lo, hi) = best_shift(s25, c25, s05, c05)
    factor = 10 ** sh

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    # ── A: the curves, with the delay measured at each threshold ────────────
    ax = axes[0]
    ax.plot(s05, c05, color=C05, lw=1.6, label="tr = 0.50")
    ax.plot(s25, c25, color=C25, lw=1.6, label="tr = 0.25")
    ax.plot(sl, cl, color=CLOW, lw=1.3, ls="--",
            label="tr $\\approx$ 0.011 (to 2.95M)")
    for thr in (0.15, 0.20, 0.25):
        a, b = first_cross(s25, c25, thr), first_cross(s05, c05, thr)
        if a and b:
            ax.plot([b, a], [thr, thr], color="0.35", lw=0.8, ls=":")
            ax.annotate(f"{a/b:.0f}$\\times$", xy=(np.sqrt(a * b), thr),
                        fontsize=7, ha="center", va="bottom", color="0.25")
    ax.set_xscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("cosine sim. (layer 4)")
    ax.set_title("(A) the delay grows with the threshold\n"
                 "a constant delay would give one number", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.grid(alpha=0.25)

    # ── B: the time-shift collapse that does not collapse ───────────────────
    ax = axes[1]
    ax.plot(s25, c25, color=C25, lw=1.8, label="tr = 0.25")
    ax.plot(s05 * factor, c05, color=C05, lw=1.4, ls="--",
            label=f"tr = 0.50, shifted {factor:.0f}$\\times$ later")
    ax.axvspan(s25[0], s25[-1], color="0.85", alpha=0.35, lw=0,
               zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(s25[0], s25[-1] * 1.05)
    ax.set_ylim(-0.05, max(c25.max(), 0.35) + 0.05)
    ax.set_xlabel("step"); ax.set_ylabel("cosine sim. (layer 4)")
    ax.set_title(f"(B) best single shift still misfits\n"
                 f"rmse {err:.3f} = {100*err/(hi-lo):.0f}% of the fitted range",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.grid(alpha=0.25)

    # ── C: who is speeding up ───────────────────────────────────────────────
    ax = axes[2]
    ax.plot(s05, local_slope(s05, c05), color=C05, lw=1.6, label="tr = 0.50")
    ax.plot(s25, local_slope(s25, c25), color=C25, lw=1.6, label="tr = 0.25")
    ax.plot(sl, local_slope(sl, cl), color=CLOW, lw=1.3, ls="--",
            label="tr $\\approx$ 0.011")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("d(cos) / d(log$_{10}$ step)")
    ax.set_title("(C) tr=0.25 is accelerating where tr=0.5\n"
                 "is saturating — the foot of a sigmoid?", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Does tr=0.25 transition later, or not at all?  "
        "c=8, 8-256, constant LR (no decay) — 1M steps cannot settle it",
        fontsize=9.5, y=1.04)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"delayed_transition.{ext}", bbox_inches="tight",
                    dpi=220)
    plt.close(fig)

    print(f"  best shift {factor:.1f}x, rmse {err:.4f} "
          f"({100*err/(hi-lo):.0f}% of range {lo:.2f}-{hi:.2f})")
    for thr in (0.15, 0.20, 0.25):
        a, b = first_cross(s25, c25, thr), first_cross(s05, c05, thr)
        if a and b:
            print(f"  cos>={thr}: tr0.25 {a:,.0f} vs tr0.5 {b:,.0f} = {a/b:.1f}x")
    print("  wrote figures/delayed_transition.{pdf,png}")


if __name__ == "__main__":
    main()

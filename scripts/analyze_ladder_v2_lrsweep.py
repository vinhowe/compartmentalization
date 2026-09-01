#!/usr/bin/env python3
"""Read the ladder-v2 LR sweep and fit the rule the ladder will use.

Produces three things:

  1. A table of post-decay val loss for every (rung, arm, lr) cell.
  2. The fitted rule LR*(d, c), with the two questions the sweep exists to
     answer stated explicitly: does the optimum shift with c, and does that
     shift depend on width (the interaction)?
  3. A figure: loss vs LR, one panel per rung, one line per arm.

The measured quantity is the loss at the LAST eval, which the configs arrange
to fall on the final training step and therefore inside the decay window. A
pre-decay reading would be the loss of a schedule nobody runs -- see
eval_interval_for in gen_ladder_v2_configs.py.

Usage:  .venv/bin/python scripts/analyze_ladder_v2_lrsweep.py
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression")
SUITE = "ladder-v2"
LOG_ROOT = ROOT / "logs" / f"{SUITE}-lrsweep"
FIG = ROOT / "figures" / f"{SUITE}-lrsweep.png"
MAX_ITERS = 1431

EVAL_RE = re.compile(
    r"^step (\d+): train loss ([\d.]+|nan), val loss ([\d.]+|nan)", re.M)
def _name_re():
    return re.compile(rf"^{re.escape(SUITE)}-lrsweep-(r\d+)-(.+)-lr([\d.e+-]+)$")

# rung -> d_model, for the LR*(d) fit
WIDTH = {"r1": 512, "r3": 768, "r4": 1024}


def parse() -> dict:
    """(rung, arm) -> {lr: (final_val, final_iter, diverged)}"""
    out: dict = defaultdict(dict)
    for p in sorted(LOG_ROOT.glob("*.log")):
        if p.name.startswith("_"):
            continue
        m = _name_re().match(p.stem)
        if not m:
            continue
        rung, arm, lr = m.group(1), m.group(2), float(m.group(3))
        evals = EVAL_RE.findall(p.read_text(errors="replace"))
        if not evals:
            continue
        it, _, val = evals[-1]
        it = int(it)
        diverged = val == "nan" or any(v == "nan" for _, _, v in evals)
        out[(rung, arm)][lr] = (
            float("nan") if diverged else float(val), it, diverged)
    return out


def argmin_lr(cell: dict) -> tuple[float, float] | None:
    """Best (lr, loss) among completed, non-diverged runs."""
    ok = {lr: v for lr, (v, it, d) in cell.items()
          if not d and it >= MAX_ITERS - 2 and not math.isnan(v)}
    if not ok:
        return None
    lr = min(ok, key=ok.get)
    return lr, ok[lr]


def main() -> int:
    global SUITE, LOG_ROOT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default=SUITE)
    a = ap.parse_args()
    SUITE = a.suite
    LOG_ROOT = ROOT / 'logs' / f'{SUITE}-lrsweep'
    FIG = ROOT / 'figures' / f'{SUITE}-lrsweep.png'
    data = parse()
    if not data:
        print("no sweep logs parsed yet")
        return 1

    # ---------------- table
    print("post-decay val loss (last eval, inside the decay window)\n")
    lrs = sorted({lr for cell in data.values() for lr in cell})
    hdr = ["rung/arm"] + [f"{lr:g}" for lr in lrs] + ["argmin"]
    rows = []
    for (rung, arm) in sorted(data):
        cell = data[(rung, arm)]
        r = [f"{rung}/{arm}"]
        for lr in lrs:
            if lr not in cell:
                r.append("-")
                continue
            v, it, d = cell[lr]
            if d:
                r.append("diverged")
            elif it < MAX_ITERS - 2:
                r.append(f"({100*it/MAX_ITERS:.0f}%)")
            else:
                r.append(f"{v:.4f}")
        best = argmin_lr(cell)
        r.append(f"{best[0]:g}" if best else "-")
        rows.append(r)
    w = [max(len(str(x[i])) for x in ([hdr] + rows)) for i in range(len(hdr))]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(len(hdr))))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))

    # ---------------- the two questions
    print("\n" + "=" * 66)
    print("Q1: does the optimum shift with c?")
    shifts = {}
    for rung in sorted(WIDTH):
        a, b = argmin_lr(data.get((rung, "c1"), {})), argmin_lr(
            data.get((rung, "c8"), {}))
        if a and b:
            shifts[rung] = math.log10(b[0] / a[0])
            print(f"  {rung} (d={WIDTH[rung]}): c1 {a[0]:g} -> c8 {b[0]:g}  "
                  f"({shifts[rung]:+.2f} decades)")
        else:
            print(f"  {rung}: incomplete")
    if len(shifts) >= 2:
        spread = max(shifts.values()) - min(shifts.values())
        print(f"\nQ2: does that shift depend on width (the interaction)?")
        print(f"  shifts span {spread:.2f} decades across rungs")
        print("  -> " + ("CONSTANT offset: apply one c-correction at every rung"
                         if spread < 0.5 else
                         "INTERACTION present: the c-correction is width-dependent, "
                         "so it cannot be applied as a single constant"))

    # ---------------- fit LR*(d) for each arm
    print("\n" + "=" * 66)
    print("LR*(d) fit  (log10 LR = a + b*log10 d)")
    for arm in ("c1", "c8"):
        pts = [(WIDTH[r], argmin_lr(data.get((r, arm), {})))
               for r in sorted(WIDTH)]
        pts = [(d, b[0]) for d, b in pts if b]
        if len(pts) < 2:
            print(f"  {arm}: need >=2 completed rungs, have {len(pts)}")
            continue
        xs = [math.log10(d) for d, _ in pts]
        ys = [math.log10(lr) for _, lr in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        rms = math.sqrt(sum(r * r for r in resid) / n)
        print(f"  {arm}: exponent {b:+.2f}  (muP expects ~-1.0), "
              f"rms resid {rms:.3f} dec, n={n}")
        if n >= 3:
            print(f"       -> 3 points: the power-law form is CHECKABLE. "
                  f"{'consistent' if rms < 0.15 else 'POOR FIT -- do not extrapolate'}")
        for d in (1536, 2048):
            print(f"       predicted LR* at d={d}: {10**(a + b*math.log10(d)):.2e}"
                  + ("   (R6 anchors: Pythia-1B ~3e-4, OLMo-1B ~4e-4)"
                     if d == 2048 else ""))

    # ---------------- figure
    rungs = [r for r in sorted(WIDTH) if any(k[0] == r for k in data)]
    if rungs:
        fig, axes = plt.subplots(1, len(rungs), figsize=(4.2 * len(rungs), 3.8),
                                 sharey=False, squeeze=False)
        for ax, rung in zip(axes[0], rungs):
            for arm, colour in (("c1", "#2b6cb0"), ("c8", "#c05621"),
                                ("c1-padded", "#718096")):
                cell = data.get((rung, arm))
                if not cell:
                    continue
                pts = sorted((lr, v) for lr, (v, it, d) in cell.items()
                             if not d and it >= MAX_ITERS - 2
                             and not math.isnan(v))
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                        color=colour, label=arm, ms=4)
                best = argmin_lr(cell)
                if best:
                    ax.axvline(best[0], color=colour, ls=":", alpha=0.5)
            ax.set_xscale("log")
            ax.set_title(f"{rung}  (d={WIDTH[rung]})")
            ax.set_xlabel("learning rate")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[0][0].set_ylabel("post-decay val loss")
        fig.suptitle("ladder-v2 LR sweep — 3B tokens, WSD with decay", y=1.02)
        fig.tight_layout()
        FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG, dpi=150, bbox_inches="tight")
        print(f"\nfigure -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

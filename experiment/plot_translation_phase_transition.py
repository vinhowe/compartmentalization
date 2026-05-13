"""Translation-ratio phase transition charts (8-256 rope, BPE16384, 1M iters).

Data: bpe16384-rope-{8-256, wd-n2, wd-n3-n8} groups. tr is the *effective*
ratio (absolute-mode raw value; compartment-mode runs are converted via
raw/(c+1)) but we restrict here to absolute-mode runs to match the paper's tr
sweep at {0.1, 0.25, 0.5, 0.75}. tr=1.0 is dropped — pure translation has no
compartmented signal.

Three panels:
  A — final val vs tr at wd=0, line per c. Plateau-break at c=8 past tr~0.25.
  B — c=6: final val vs wd, line per tr. Sharp inflection at (tr=0.75, wd=0.1).
  C — c=8: final val vs wd, line per tr. Plateau-break shifts deeper with wd.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style, C_COLOR
from _run_paths import C1_BASELINE_8_256


C1_BASELINE_KEY = C1_BASELINE_8_256


def collect_cells():
    """Return list[dict] with one entry per (run, last-step val). Effective tr."""
    m = json.loads(Path("val_metrics.json").read_text())
    out = []
    for g in ["bpe16384-rope-wd-n2", "bpe16384-rope-wd-n3-n8", "bpe16384-rope-8-256"]:
        for d in sorted((Path("..") / "out" / "translation-compression" / g).iterdir()):
            cf = d / "meta" / "config.json"
            if not cf.exists():
                continue
            cobj = json.loads(cf.read_text())
            e, o = cobj["experiment"], cobj["optimizer"]
            c, tr_raw, mode = e["n_compartments"], e["translation_ratio"], e["translation_ratio_mode"]
            tr_eff = tr_raw / (c + 1) if mode == "compartment" else tr_raw
            v = m.get(f"{g}/{d.name}")
            if not v or not v.get("checkpoints"):
                continue
            losses = [v["metrics"][f"loss_compartment_{i}"][-1]
                      for i in range(c)
                      if f"loss_compartment_{i}" in v["metrics"]]
            if not losses:
                continue
            out.append({
                "c": c, "tr_eff": round(tr_eff, 4), "tr_raw": tr_raw,
                "mode": mode, "wd": o["weight_decay"],
                "val": float(np.mean(losses)),
                "last_step": v["checkpoints"][-1],
            })
    return out


def panel_A(ax, cells, c1_floor):
    """Final val vs tr at wd=0, lines per c."""
    rows = [r for r in cells
            if r["mode"] == "absolute" and r["wd"] == 0
            and r["last_step"] >= 1_000_000 and r["tr_raw"] < 1.0]
    by_c = defaultdict(dict)
    for r in rows:
        by_c[r["c"]].setdefault(r["tr_eff"], []).append(r["val"])
    for c in sorted(by_c):
        trs = sorted(by_c[c])
        vals = [min(by_c[c][t]) for t in trs]  # min across seeds (best)
        ax.plot(trs, vals, color=C_COLOR.get(c, "k"), marker="o",
                markersize=4, linewidth=1.4, label=f"c={c}")
    ax.axhline(c1_floor, color="black", linewidth=0.6, alpha=0.5,
               linestyle=":", label="c=1")
    ax.set_xlabel("translation ratio")
    ax.set_ylabel("val loss (nats)")


def panel_wd_for_c(ax, cells, target_c, c1_floor, title=None):
    """Final val vs wd, lines per tr."""
    rows = [r for r in cells
            if r["mode"] == "absolute" and r["c"] == target_c
            and r["last_step"] >= 1_000_000 and r["tr_raw"] < 1.0]
    by_tr = defaultdict(dict)
    for r in rows:
        by_tr[r["tr_eff"]].setdefault(r["wd"], []).append(r["val"])
    cmap = plt.get_cmap("viridis")
    trs = sorted(by_tr)
    for i, tr in enumerate(trs):
        wds = sorted(by_tr[tr])
        vals = [min(by_tr[tr][w]) for w in wds]
        color = cmap(i / max(1, len(trs) - 1))
        ax.plot(wds, vals, color=color, marker="o",
                markersize=4, linewidth=1.4, label=f"tr={tr:g}")
    ax.axhline(c1_floor, color="black", linewidth=0.6, alpha=0.5, linestyle=":")
    ax.set_xlabel("wd")
    if title is not None:
        ax.set_title(title)


def main():
    setup_paper_style()
    cells = collect_cells()
    main_metrics = json.loads(Path("val_metrics.json").read_text())
    c1_floor = main_metrics[C1_BASELINE_KEY]["metrics"]["loss_compartment_0"][-1]

    Path("../figures").mkdir(exist_ok=True)

    # 4-panel combined figure (A, B-c5, C-c6, D-c8)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0), sharey=True)
    panel_A(axes[0], cells, c1_floor)
    panel_wd_for_c(axes[1], cells, 5, c1_floor, "(B) c=5: wd does not unlock transition")
    panel_wd_for_c(axes[2], cells, 6, c1_floor, "(C) c=6: wd unlocks transition at tr=0.75")
    panel_wd_for_c(axes[3], cells, 8, c1_floor, "(D) c=8: wd deepens the post-transition val")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7,
                   handlelength=1.3, handletextpad=0.4, ncol=2)
    for ax in axes[1:]:
        ax.legend(loc="upper right", frameon=False, fontsize=7,
                  handlelength=1.3, handletextpad=0.4, title="translation ratio",
                  title_fontsize=7)
    fig.tight_layout()
    out = Path("../figures/translation_phase_transition.pdf")
    fig.savefig(out)
    print(f"  {out}")
    plt.close(fig)

    # Standalone single panels for paper layout flexibility
    for name, fn in [
        ("A_phase_transition",  lambda ax: panel_A(ax, cells, c1_floor)),
        ("B_c5_wd_no_unlock",   lambda ax: panel_wd_for_c(ax, cells, 5, c1_floor, "c=5: wd × tr")),
        ("C_c6_wd_inflection",  lambda ax: panel_wd_for_c(ax, cells, 6, c1_floor, "c=6: wd × tr")),
        ("D_c8_wd_deepening",   lambda ax: panel_wd_for_c(ax, cells, 8, c1_floor, "c=8: wd × tr")),
    ]:
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        fn(ax)
        ax.legend(loc="best", frameon=False, fontsize=7,
                  handlelength=1.3, handletextpad=0.4, ncol=2)
        fig.tight_layout()
        out = Path(f"../figures/translation_phase_transition_{name}.pdf")
        fig.savefig(out)
        print(f"  {out}")
        plt.close(fig)

    # Section-ready PDFs ----------------------------------------------------
    # §4.1 no-WD: standalone val panel A. No title — caption supplies wd=0 etc.
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    panel_A(ax, cells, c1_floor)
    ax.legend(loc="lower right", frameon=False, fontsize=7,
              handlelength=1.3, handletextpad=0.4, ncol=2)
    fig.tight_layout()
    out = Path("../figures/tr_phase_no_wd_val.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    # Compact (1/3-column-ready): 2.4 x 2.0 with smaller fonts so when LaTeX
    # scales to ~1.8" wide via \linewidth the text stays legible.
    # Also overlays the 1B c=8 line (matched at step 744k) as a single red
    # series on the same axes, since 1B and 8-256 share the val-loss y-axis
    # nearly perfectly in the relevant range.
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    panel_A(ax, cells, c1_floor)

    ax.legend(loc="lower left",
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.9, fontsize=7.3,
              handlelength=1.0, handletextpad=0.3, ncol=4,
              columnspacing=0.6, borderpad=0.2)
    fig.tight_layout(pad=0.3)
    out = Path("../figures/tr_phase_no_wd_val_compact.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    # §4.2 wd: 1x3 strip for c=5, 6, 8 (val), shared y. Per-panel c-label as
    # an in-axis text annotation so subcaptions are unobstructed.
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), sharey=True)
    for ax, c in zip(axes, [5, 6, 8]):
        panel_wd_for_c(ax, cells, c, c1_floor)
        ax.text(0.97, 0.03, f"c={c}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
    axes[0].set_ylabel("val loss (nats)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=7,
                   handlelength=1.3, handletextpad=0.4,
                   title="translation ratio", title_fontsize=7)
    fig.tight_layout()
    out = Path("../figures/tr_phase_wd_val.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

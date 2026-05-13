"""Compact tr=0.75 slice through the (c, wd) plane for the WD subsection.

Two panels at 0.49 textwidth each:
  (a) val vs wd, lines per c ∈ {5, 6, 8}
  (b) cossim vs wd, lines per c ∈ {5, 6, 8}
Both at fixed tr=0.75 (the corner column where the c=6 inflection lives).
Captures the load-bearing claim:
  c=5 wd-immune; c=6 inflects sharply; c=6 wd=0.2 reaches c=8 best val (~4.09)
  but at cossim ~0.49 vs c=8's ~0.99 → two routes, same plateau.
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


C_LIST = [5, 6, 8]
TR_TARGET = 0.75


def collect_val_at_tr075():
    """Return dict {c: [(wd, val), ...]} for absolute-mode tr=0.75 cells."""
    m = json.loads(Path("val_metrics.json").read_text())
    out = defaultdict(dict)  # c -> wd -> list of vals (best across seeds)
    for g in ["bpe16384-rope-wd-n2", "bpe16384-rope-wd-n3-n8", "bpe16384-rope-8-256"]:
        for d in sorted((Path("..") / "out" / "translation-compression" / g).iterdir()):
            cf = d / "meta" / "config.json"
            if not cf.exists():
                continue
            cobj = json.loads(cf.read_text())
            e, o = cobj["experiment"], cobj["optimizer"]
            if e["translation_ratio_mode"] != "absolute":
                continue
            if abs(e["translation_ratio"] - TR_TARGET) > 1e-6:
                continue
            c = e["n_compartments"]
            if c not in C_LIST:
                continue
            v = m.get(f"{g}/{d.name}")
            if not v or not v.get("checkpoints"):
                continue
            if v["checkpoints"][-1] < 1_000_000:
                continue
            losses = [v["metrics"][f"loss_compartment_{i}"][-1]
                      for i in range(c) if f"loss_compartment_{i}" in v["metrics"]]
            if not losses:
                continue
            wd = o["weight_decay"]
            out[c].setdefault(wd, []).append(float(np.mean(losses)))
    return out


def collect_cossim_at_tr075():
    """Return dict {c: [(wd, cossim), ...]} for absolute-mode tr=0.75 cells."""
    out = defaultdict(dict)
    cka_data = json.loads(Path("cossim_sweep.json").read_text())
    for k, v in cka_data.items():
        if v["mode"] != "absolute":
            continue
        if abs(v["tr_raw"] - TR_TARGET) > 1e-6:
            continue
        c = v["c"]
        if c not in C_LIST:
            continue
        wd = v["wd"]
        out[c].setdefault(wd, []).append(float(v["mean_off_diag_cossim"]))
    return out


def panel(ax, by_c, ylabel, ylim=None, hline=None):
    for c in C_LIST:
        rows = by_c.get(c, {})
        if not rows:
            continue
        wds = sorted(rows.keys())
        ys = [min(rows[wd]) if "val" in ylabel else max(rows[wd]) for wd in wds]
        ax.plot(wds, ys, color=C_COLOR.get(c, "k"), marker="o",
                markersize=4, linewidth=1.4, label=f"c={c}")
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.6, alpha=0.5,
                   linestyle=":", label="c=1")
    ax.set_xlabel("weight decay")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)
    val_by_c = collect_val_at_tr075()
    cossim_by_c = collect_cossim_at_tr075()

    # c=1 floor
    m = json.loads(Path("val_metrics.json").read_text())
    c1_floor = m[C1_BASELINE_8_256]["metrics"]["loss_compartment_0"][-1]

    # Render each panel as a separate compact PDF for subfigure inclusion.
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    panel(ax, val_by_c, "val loss (nats)", hline=c1_floor)
    ax.legend(loc="lower left", frameon=False,
              handlelength=1.3, handletextpad=0.4, ncol=2,
              columnspacing=0.8)
    fig.tight_layout()
    out = Path("../figures/tr_wd_tr075_val.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    panel(ax, cossim_by_c, "cosine sim. (layer 4)", ylim=(-0.05, 1.05))
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4, linestyle=":")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.6),
              frameon=False,
              handlelength=1.3, handletextpad=0.4)
    fig.tight_layout()
    out = Path("../figures/tr_wd_tr075_cossim.pdf")
    fig.savefig(out); print(f"  {out}"); plt.close(fig)


if __name__ == "__main__":
    main()

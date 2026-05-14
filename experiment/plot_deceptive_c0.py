"""Compartment-0 val loss: c=1 vs c=2 vs deceptive-c8.

Tests whether adding 6 decoy compartments (sharing a 3rd vocab chunk via
distinct random permutations, seen only as translation targets) breaks the
c=2 plateau that we see in the standard sweep. All runs are 8-256 rope,
bpe16384. Compartment 0 is "real English" in all three.

Sources:
- C1_BASELINE_8_256              (c=1 baseline; loss_compartment_0)
- NO_INFONCE_8_256_BY_C[2]       (c=2 no-InfoNCE; loss_compartment_0)
- bpe16384-rope-8-256-deceptive/* (the deceptive run; loss_compartment_0)

All read from val_metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style
from _run_paths import C1_BASELINE_8_256, NO_INFONCE_8_256_BY_C, OUT_ROOT


def comp0_curve(metrics, key):
    v = metrics.get(key)
    if not v or not v.get("checkpoints"):
        return np.array([]), np.array([])
    s = np.array(v["checkpoints"], dtype=float)
    l = np.array(v["metrics"]["loss_compartment_0"], dtype=float)
    n = min(len(s), len(l))
    order = np.argsort(s[:n])
    return s[:n][order], l[:n][order]


def latest_deceptive_key():
    g = OUT_ROOT / "bpe16384-rope-8-256-deceptive"
    if not g.exists():
        return None
    dirs = sorted(d.name for d in g.iterdir() if d.is_dir())
    if not dirs:
        return None
    return f"bpe16384-rope-8-256-deceptive/{dirs[-1]}"


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)
    m = json.loads(Path("val_metrics.json").read_text())

    deceptive_key = latest_deceptive_key()

    lines = [
        ("c=1 baseline",          C1_BASELINE_8_256,        "tab:gray",   "x", "--"),
        ("c=2 (no decoys, tr=0)", NO_INFONCE_8_256_BY_C[2], "tab:blue",   "o", "-"),
        ("c=fake8 (2 real + 6 decoys, tr=0.5)",
                                  deceptive_key,            "tab:red",    "s", "-"),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for label, key, color, marker, ls in lines:
        if key is None:
            continue
        s, l = comp0_curve(m, key)
        if s.size == 0:
            print(f"  {label}: NO DATA  (key={key})")
            continue
        ax.plot(s, l, color=color, marker=marker, markersize=3,
                linewidth=1.3, linestyle=ls, label=label)
        print(f"  {label}: {len(s)} pts, last step {int(s[-1])}, val {l[-1]:.4f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=10**1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Comp-0 (English) val loss (nats)")
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              handlelength=1.5, handletextpad=0.5)
    fig.tight_layout()
    out = Path("../figures/deceptive_c0.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"  {out}")


if __name__ == "__main__":
    main()

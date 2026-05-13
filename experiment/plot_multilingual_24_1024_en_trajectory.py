"""EN val loss over training at 24-1024 (620M), single-panel for appendix.

Shows the full EN-side trajectory for shared / compartmented / en-only at the
largest multilingual scale we trained, complementing body Fig.~\\ref{fig:multilingual-scaling}
(which shows finals across scales) and Fig.~\\ref{fig:appendix-multi}(a)
(ZH finals across scales).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_baseline_val_curves import setup_paper_style


CONDITIONS = [
    ("shared",        "tab:blue",   "o"),
    ("compartmented", "tab:orange", "s"),
    ("en-only",       "tab:green",  "^"),
]


def main():
    setup_paper_style()
    data = json.loads(Path("multilingual_24_1024_per_lang.json").read_text())
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for label, color, marker in CONDITIONS:
        rows = data.get(label, [])
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        en = [r["en"] for r in rows]
        ax.plot(steps, en, color=color, marker=marker, markersize=3,
                linewidth=1.2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("EN val loss (nats)")
    ax.legend(loc="upper right", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9)
    fig.tight_layout(pad=0.3)
    Path("../figures").mkdir(exist_ok=True)
    out = Path("../figures/multilingual_24_1024_en_trajectory.pdf")
    fig.savefig(out)
    print(f"  {out}")


if __name__ == "__main__":
    main()

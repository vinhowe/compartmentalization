"""Bio capacity Phase-2 N=15k, seed-averaged trajectories.

For each condition × cell, plot mean across all available seeds. Error bars
show min/max range (since n is small, std isn't meaningful).
"""
from __future__ import annotations

import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def setup_paper_style():
    rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "lines.linewidth": 1.4, "lines.markersize": 3.0,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4,
        "axes.grid": True, "grid.alpha": 0.3,
    })


# (label, glob pattern (across seeds), color, linestyle)
CONDITIONS = [
    ("biography-only",            "../out/translation-compression/bio-capacity/*bio-cap-decl_only-N15000{seed_suffix}__*",     "#2ca02c", "-"),
    ("QA-only",              "../out/translation-compression/bio-capacity/*bio-cap-qa_only-N15000{seed_suffix}__*",       "#ff7f0e", "-"),
    ("BOTH shared",          "../out/translation-compression/bio-capacity/*bio-cap-both-N15000{seed_suffix}__*",          "#1f77b4", "-"),
    ("BOTH compartmented",   "../out/translation-compression/bio-capacity/*bio-cap-both-comp-N15000{seed_suffix}__*",     "#17becf", "--"),
    ("SPLIT shared",         "../out/translation-compression/bio-capacity/*bio-cap-split-N15000{seed_suffix}__*",         "#9467bd", "-"),
    ("SPLIT compartmented",  "../out/translation-compression/bio-capacity/*bio-cap-split-comp-N15000{seed_suffix}__*",    "#e377c2", "--"),
    ("SPLIT shared+InfoNCE", "../out/translation-compression/bio-capacity/*bio-cap-split-infonce-N15000{seed_suffix}__*", "#8c564b", "-"),
    ("SPLIT comp+InfoNCE",   "../out/translation-compression/bio-capacity/*bio-cap-split-comp-infonce-N15000{seed_suffix}__*", "#d62728", "--"),
]


def collect_seed_runs(condition_pattern: str) -> list[dict]:
    """Find all seed runs matching the pattern. Each pattern uses {seed_suffix}.
    seed_suffix is "" for s=42 (default seed) or "-s{N}" for explicit seeds."""
    seeds = []
    # original seed=42 (no suffix on dir name)
    p1 = condition_pattern.replace("{seed_suffix}", "")
    for f in sorted(glob.glob(p1 + "/bio_extraction_eval.json")):
        seeds.append(json.loads(Path(f).read_text()))
    # explicit seed dirs (-sN)
    for s in (2, 3, 4, 5):
        p2 = condition_pattern.replace("{seed_suffix}", f"-s{s}")
        for f in sorted(glob.glob(p2 + "/bio_extraction_eval.json")):
            seeds.append(json.loads(Path(f).read_text()))
    return seeds


def get_trajectory(seed_data: dict, train_fmt: str, probe_fmt: str):
    pop_options = ["decl_only", "bridge"] if train_fmt == "decl" else ["qa_only", "bridge"]
    xs, ys = [], []
    for step in sorted(seed_data.keys(), key=int):
        for pop in pop_options:
            if pop in seed_data[step] and probe_fmt in seed_data[step][pop]:
                xs.append(int(step))
                ys.append(seed_data[step][pop][probe_fmt]["overall"]["acc"] * 100)
                break
    return xs, ys


def aggregate_at_steps(seeds: list[dict], train_fmt: str, probe_fmt: str):
    """Return (steps, mean, lo, hi) — only at steps available in all seeds."""
    if not seeds:
        return [], [], [], []
    per_seed_traj = []
    for sd in seeds:
        xs, ys = get_trajectory(sd, train_fmt, probe_fmt)
        per_seed_traj.append(dict(zip(xs, ys)))
    # Common steps
    common = sorted(set.intersection(*[set(d.keys()) for d in per_seed_traj]))
    if not common:
        return [], [], [], []
    means, los, his = [], [], []
    for s in common:
        vals = [d[s] for d in per_seed_traj]
        means.append(np.mean(vals))
        los.append(np.min(vals))
        his.append(np.max(vals))
    return common, means, los, his


def fig_seed_avg_trajectories():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    panels = [
        (0, 0, "decl", "decl_continuation", "biography format × biography probe"),
        (0, 1, "decl", "qa_prompt",         "biography format × QA probe"),
        (1, 0, "qa",   "decl_continuation", "QA format × biography probe"),
        (1, 1, "qa",   "qa_prompt",         "QA format × QA probe"),
    ]
    handles_global, labels_global = [], []
    for r, c, train_fmt, probe_fmt, title in panels:
        ax = axes[r, c]
        for label, pat, color, ls in CONDITIONS:
            seeds = collect_seed_runs(pat)
            if not seeds:
                continue
            steps, means, los, his = aggregate_at_steps(seeds, train_fmt, probe_fmt)
            if not steps:
                continue
            line, = ax.plot(steps, means, color=color, linestyle=ls,
                            marker="o", markersize=2.0, label=f"{label} (n={len(seeds)})")
            ax.fill_between(steps, los, his, color=color, alpha=0.15)
            if label not in labels_global:
                handles_global.append(line)
                labels_global.append(label)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_ylim(0, 105)
        if r == 1: ax.set_xlabel("Step")
        if c == 0: ax.set_ylabel("Extraction (%)")
    fig.legend(handles_global, labels_global, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path("../figures/bio_capacity_n15k_seedavg.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  {out}")
    plt.close(fig)


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)
    print("Generating seed-averaged trajectories:")
    fig_seed_avg_trajectories()


if __name__ == "__main__":
    main()

"""Validation loss curves for the baseline compartmented models — 6 panels.

One panel per model size (d_model in {32, 64, 128, 256, 512, 1024}).
Each panel shows curves for each available c (n_compartments).

All runs are rope-era. c>1 runs use tr=0.1 compartment-mode; c=1 and 1B (c=2,8)
are tr=0. All effective tr near 0 — treated as the compartment baseline series.
Sources: synthetic-compartment-baselines (c=1, d≤256), bpe16384-rope-8-256
(c≥2, d=256), bpe16384-rope-small-scale-tr01-epoch (d∈{32,64,128} c≥2),
bpe16384-rope-8-512-sweep (d=512), 1b-scale (1B). 1B only has c∈{1,2,8}.

Reads fineweb_val_metrics.json. Averages val loss across compartments.
Paper-ready vector PDF, log-log axes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.pyplot as _plt


def setup_paper_style():
    rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "lines.linewidth": 1.3, "lines.markersize": 0,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4,
        "axes.grid": True, "grid.alpha": 0.3,
    })


# (d_model, [(c, key), ...])
PANELS = [
    (32, [
        (1, "synthetic-compartment-baselines/2026-03-06T18-19-13Z__english-baseline-rope-bpe16384-8-32__41ba658f__s64__4b68526__66b66981"),
        (2, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-14Z__8-32-n2-tr01__276e6011__s64__fd9c538__fc9992cd"),
        (3, "bpe16384-n3-rope/b7d883ff_s64"),
        (4, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-15Z__8-32-n4-tr01__182e8587__s64__fd9c538__87cfdd84"),
        (5, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-15Z__8-32-n5-tr01__5fe56681__s64__fd9c538__64ba6d8f"),
        (6, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T21-46-13Z__8-32-n6-tr01__d3d70929__s64__fd9c538__6ff7bf7e"),
        (8, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-32-n8-tr01__3de8c1bc__s64__fd9c538__28614139"),
    ]),
    (64, [
        (1, "synthetic-compartment-baselines/2026-03-06T18-18-46Z__english-baseline-rope-bpe16384-8-64__50fa3055__s64__4b68526__007674d4"),
        (2, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-64-n2-tr01__a654d23e__s64__fd9c538__4bc16fd9"),
        (3, "bpe16384-n3-rope/9f60d15d_s64"),
        (4, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-15Z__8-64-n4-tr01__1f914b4a__s64__fd9c538__2921ef4c"),
        (5, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-14Z__8-64-n5-tr01__81cf31a3__s64__fd9c538__faac50d4"),
        (6, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-14Z__8-64-n6-tr01__701205b7__s64__fd9c538__a00c89fe"),
        (8, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-64-n8-tr01__bb0b7e57__s64__fd9c538__ebbb76e1"),
    ]),
    (128, [
        (1, "synthetic-compartment-baselines/2026-03-06T18-17-16Z__english-baseline-rope-bpe16384-8-128__bafaffdf__s64__4b68526__e939a660"),
        (2, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n2-tr01__29620da8__s64__fd9c538__e33f2900"),
        (3, "bpe16384-n3-rope/4bb14425_s64"),
        (4, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n4-tr01__fdd9ff02__s64__fd9c538__2eba5b2e"),
        (5, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n5-tr01__1ec60abf__s64__fd9c538__7a44f455"),
        (6, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n6-tr01__a6e55f01__s64__fd9c538__309edd86"),
        (8, "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n8-tr01__0caca982__s64__fd9c538__be47d71b"),
    ]),
    (256, [
        (1, "synthetic-compartment-baselines/2026-03-06T18-11-45Z__english-baseline-rope-bpe16384-8-256__2df56182__s64__4b68526__51c738c2"),
        (2, "bpe16384-rope-8-256/217ca694_s64"),
        (3, "bpe16384-rope-8-256/c5ac7e54_s64"),
        (4, "bpe16384-rope-8-256/53e73c3d_s64"),
        (5, "bpe16384-rope-8-256/918122e2_s64"),
        (6, "bpe16384-rope-8-256/b4d95a94_s64"),
        (8, "bpe16384-rope-8-256/868ef4a8_s64"),
    ]),
    (512, [
        (1, "bpe16384-rope-8-512-sweep/2026-04-27T22-12-00Z__8-512-n1-tr0__6a459969__s64__fd9c538__cd128e95"),
        (2, "bpe16384-rope-8-512-sweep/2026-04-27T22-12-03Z__8-512-n2-tr01__1ac70722__s64__fd9c538__959443a0"),
        (3, "bpe16384-rope-8-512-sweep/2026-04-27T22-12-00Z__8-512-n3-tr01__de18a18b__s64__fd9c538__bdbec307"),
        (4, "bpe16384-rope-8-512-sweep/2026-04-27T22-12-00Z__8-512-n4-tr01__79908dbc__s64__fd9c538__c785cc59"),
        (5, "bpe16384-rope-8-512-sweep/2026-04-27T22-12-06Z__8-512-n5-tr01__cff4ea6b__s64__fd9c538__716a61dc"),
        (6, "bpe16384-rope-8-512-sweep/2026-04-27T22-12-01Z__8-512-n6-tr01__490675c3__s64__fd9c538__464afe7c"),
        (8, "bpe16384-rope-8-512-sweep/2026-04-27T22-12-00Z__8-512-n8-tr01__a7ddefbd__s64__fd9c538__198f3d6b"),
    ]),
    (1024, [
        (1, "1b-scale/2026-04-11T20-29-41Z__1b-1comp-baseline-bpe16384__9e005d30__s64__75a29e5__3acd587e"),
        (2, "1b-scale/2026-04-14T20-06-42Z__1b-2comp-notrans-bpe16384-correct__836ab6ce__s64__75a29e5__55830146"),
        (8, "1b-scale/2026-04-11T20-30-51Z__1b-8comp-notrans-bpe16384-correct__9c199809__s64__75a29e5__596929d1"),
    ]),
]

# Color palette indexed by c, shared across panels.
ALL_C = [1, 2, 3, 4, 5, 6, 8]
CMAP = _plt.get_cmap("viridis")
C_COLOR = {c: CMAP(i / (len(ALL_C) - 1)) for i, c in enumerate(ALL_C)}


def avg_compartment_loss(metric_dict: dict, n_compartments: int, n_steps: int) -> list[float]:
    arrs = []
    for c in range(n_compartments):
        k = f"loss_compartment_{c}"
        if k in metric_dict and len(metric_dict[k]) == n_steps:
            arrs.append(metric_dict[k])
    if not arrs:
        return []
    return [sum(a[i] for a in arrs) / len(arrs) for i in range(n_steps)]


def d_label(d: int) -> str:
    return "1B" if d == 1024 else f"d={d}"


def main():
    setup_paper_style()
    metrics = json.loads(Path("fineweb_val_metrics.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True)
    flat = axes.flatten()
    for i, (d, runs) in enumerate(PANELS):
        ax = flat[i]
        for c, key in runs:
            if key not in metrics:
                print(f"  {d_label(d)} c={c}: missing metrics for {key}")
                continue
            v = metrics[key]
            steps = v["checkpoints"]
            losses = avg_compartment_loss(v["metrics"], c, len(steps))
            if not losses:
                print(f"  {d_label(d)} c={c}: no compartment losses")
                continue
            ax.plot(steps, losses, color=C_COLOR[c], label=f"c={c}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(d_label(d))
        if i // 3 == 1:
            ax.set_xlabel("Step")
        if i % 3 == 0:
            ax.set_ylabel("Val loss (nats)")
        ax.legend(loc="upper right", frameon=False, ncol=2,
                  handlelength=1.3, handletextpad=0.5,
                  columnspacing=0.8, labelspacing=0.25)
    fig.suptitle("Compartment-baseline val loss vs step (BPE-16384, fineweb)", y=1.0)
    fig.tight_layout()
    out = Path("../figures/baseline_val_curves.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  {out}")


if __name__ == "__main__":
    main()

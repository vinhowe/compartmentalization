"""1B-scale figure pair (paper §1B-scale companion to the no-WD section).

Two panels at 0.49 textwidth each:
  (a) val loss vs translation ratio at 1B, lines for c=2 and c=8 (matched at
      step 744k; c=1 floor as dotted reference).
  (b) overall val-loss trajectory (slurm log) per c=8 tr cell — illustrates
      how rapidly the local translation task is mastered as tr increases.

Solid c=2/c=8 colors (not the C_COLOR gradient — only two lines so the
gradient endpoints would hit yellow which reads poorly).
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style


C2_COLOR = "tab:blue"
C8_COLOR = "tab:red"
TARGET_STEP = 1_000_000  # all c=2/c=8 cells now have data here (preprint refresh)


def collect_matched_val():
    """Return {c: [(tr_eff, val), ...]} at TARGET_STEP, plus c=1 floor.
    Interpolated in log-step from each cell's trajectory. Now that all
    trajectories use 10-batch-averaged eval, bf16 named-checkpoint noise is
    suppressed enough to interpolate cleanly across them."""
    m = json.loads(Path("fineweb_val_metrics.json").read_text())
    out = defaultdict(list)
    c1 = None
    for d in sorted((Path("..") / "out" / "translation-compression" / "1b-scale").iterdir()):
        cf = d / "meta" / "config.json"
        if not cf.exists():
            continue
        cobj = json.loads(cf.read_text())
        e, o = cobj["experiment"], cobj["optimizer"]
        if o.get("weight_decay", 0) != 0:
            continue
        c, tr_raw, mode = e["n_compartments"], e["translation_ratio"], e["translation_ratio_mode"]
        tr_eff = tr_raw / (c + 1) if mode == "compartment" else tr_raw
        # Drop the under-trained tr=0.1 abs cell.
        if mode == "absolute" and abs(tr_raw - 0.1) < 1e-6:
            continue
        v = m.get(f"1b-scale/{d.name}")
        if not v or not v.get("checkpoints"):
            continue
        steps = np.array(v["checkpoints"], dtype=float)
        arrs = [v["metrics"][f"loss_compartment_{i}"]
                for i in range(c)
                if f"loss_compartment_{i}" in v["metrics"]
                and len(v["metrics"][f"loss_compartment_{i}"]) == len(steps)]
        if not arrs:
            continue
        avg = np.array(arrs, dtype=float).mean(axis=0)
        order = np.argsort(steps)
        steps = steps[order]; avg = avg[order]
        if len(steps) == 1:
            val = float(avg[0])
        else:
            val = float(np.interp(np.log(TARGET_STEP), np.log(steps), avg))
        if c == 1:
            c1 = val
        else:
            out[c].append((round(tr_eff, 4), val))
    for c in out:
        out[c].sort()
    return out, c1


def parse_trajectories(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Parse the 1b_tr_trajectories.txt from the slurm logs into
    {tr_label: [(step, val), ...]}."""
    out: dict[str, list[tuple[int, float]]] = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("==="):
                cur = line.strip("=").strip()
                out[cur] = []
            elif cur and line:
                parts = line.split()
                if len(parts) == 2:
                    step, val = parts
                    try:
                        out[cur].append((int(step), float(val)))
                    except ValueError:
                        pass
    return out


def panel_val(ax, by_c, c1):
    for c, color in [(2, C2_COLOR), (8, C8_COLOR)]:
        if c not in by_c:
            continue
        xs, ys = zip(*by_c[c])
        ax.plot(xs, ys, color=color, marker="o", markersize=5,
                linewidth=1.4, label=f"c={c}")
    if c1 is not None:
        ax.axhline(c1, color="black", linewidth=0.6, alpha=0.5,
                   linestyle=":", label=f"c=1 ({c1:.3f})")
    ax.set_xlabel("translation ratio")
    ax.set_ylabel("val loss (nats)")
    ax.legend(loc="upper right", frameon=False, fontsize=7,
              handlelength=1.3, handletextpad=0.4)


def panel_target_trajectories(ax):
    """Target-half val loss vs step at 1B. c=8 line family (tr=0.056 from
    compartment-mode + tr=0.25/0.5/0.75 from absolute-mode), plus c=2 at
    tr=0.167 as a contrast that converges even faster despite less translation
    data, because the c=2 model has more capacity per compartment."""
    import json, numpy as np
    m = json.loads(Path("fineweb_val_metrics.json").read_text())

    # No artificial start clip — show whatever steps each line has data for.
    # Abs runs only have rsync'd ckpts from step 7k onward; long-trajectory
    # compartment-mode runs have step 100 onward.
    MIN_STEP = 0
    # c=8 family: viridis from light (tr=0.056) to dark (tr=0.75).
    cmap = plt.get_cmap("viridis")
    C8_RUNS = [
        (0.056, "1b-scale/2026-04-12T04-30-30Z__1b-8comp-bpe16384-correct__47875262__s64__75a29e5__c1a66f59"),
        (0.25,  "1b-scale/2026-04-28T07-25-54Z__1b-8comp-tr025abs-bpe16384__7f828f8a__s64__75a29e5__10e98f4e"),
        (0.5,   "1b-scale/2026-04-27T23-16-17Z__1b-8comp-tr05abs-bpe16384__8c45bf62__s64__75a29e5__984e0ac2"),
        (0.75,  "1b-scale/2026-04-27T23-16-44Z__1b-8comp-tr075abs-bpe16384__cf2767fd__s64__75a29e5__36396f38"),
    ]
    for i, (tr, k) in enumerate(C8_RUNS):
        v = m.get(k, {})
        if not v: continue
        steps = np.array(v["checkpoints"], dtype=float)
        target_arrs = [v["metrics"][kk] for kk in v["metrics"]
                       if kk.startswith("loss_target_")]
        if not target_arrs: continue
        avg = np.array(target_arrs, dtype=float).mean(axis=0)
        order = np.argsort(steps)
        steps = steps[order]; avg = avg[order]
        # Drop step-7k point — single-batch eval noise on the bf16 named
        # checkpoint produced misleadingly low values there for the abs runs.
        mask = (steps >= MIN_STEP) & (avg > 0)
        color = cmap(0.15 + 0.7 * i / max(1, len(C8_RUNS) - 1))
        ax.plot(steps[mask], avg[mask], color=color, marker="o",
                markersize=2.5, linewidth=1.3, label=f"c=8, tr={tr:g}")

    # c=2 contrast.
    k = "1b-scale/2026-04-14T20-35-26Z__1b-2comp-bpe16384-correct__586efbbc__s64__75a29e5__6c0d1003"
    v = m.get(k, {})
    if v:
        steps = np.array(v["checkpoints"], dtype=float)
        target_arrs = [v["metrics"][kk] for kk in v["metrics"]
                       if kk.startswith("loss_target_")]
        if target_arrs:
            avg = np.array(target_arrs, dtype=float).mean(axis=0)
            mask = (steps >= MIN_STEP) & (avg > 0)
            ax.plot(steps[mask], avg[mask], color="tab:red", marker="s",
                    markersize=2.5, linewidth=1.3, linestyle="--",
                    label="c=2, tr=0.167")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("target-half val loss (nats)")
    ax.legend(loc="lower left", frameon=False, fontsize=7,
              handlelength=1.3, handletextpad=0.4)


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)
    by_c, c1 = collect_matched_val()

    # 0.49-textwidth twin: 3.5 x 2.6 each (roughly matches multilingual fig
    # size pattern). Render each as a separate PDF for subfigure inclusion.
    for name, fn in [
        ("1b_phase_val", lambda ax: panel_val(ax, by_c, c1)),
        ("1b_tr_trajectories", lambda ax: panel_target_trajectories(ax)),
    ]:
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        fn(ax)
        fig.tight_layout()
        out = Path(f"../figures/{name}.pdf")
        fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=180)
        print(f"  {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()

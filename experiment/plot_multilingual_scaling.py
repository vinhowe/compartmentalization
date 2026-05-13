"""Multilingual scaling figure: final per-language val loss vs scale.

4 scales: 12-256 (~87M params), 24-512 (~230M), 24-768 (~370M), 24-1024 (~620M).
3 conditions: shared, compartmented, en-only.

Two panels (EN, ZH), x = scale (with M-params labels), y = final val loss.
Legend at bottom for caption space.

Output:
  ../figures/multilingual_scaling_final.pdf  (combined, full-width)
  ../figures/multilingual_scaling_en.pdf      (EN only, half-width)
  ../figures/multilingual_scaling_zh.pdf      (ZH only, half-width)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_baseline_val_curves import setup_paper_style


SCALES = [
    ("12-256", 87.2,    "multilingual_val_curves_no_infonce.json"),
    ("24-512", 230.0,   "multilingual_24_512_per_lang.json"),
    ("24-768", 370.0,   "multilingual_24_768_per_lang.json"),
    ("24-1024", 620.0,  "multilingual_24_1024_per_lang.json"),
]

CONDITIONS = [
    ("shared",        "tab:blue",   "o"),
    ("compartmented", "tab:orange", "s"),
    ("en-only",       "tab:green",  "^"),
]


def scale_label(label, params_m):
    return f"{params_m:.0f}M" if params_m < 1000 else f"{params_m/1000:.2g}B"


def _load_finals():
    """Returns dict {condition: list of (scale_label, params_m, en, zh)}.
    Values are at the last step common to all 3 conditions at that scale —
    typically en-only's last logged ckpt (~5000). This makes the comparison
    iter-matched (en-only stops earlier than shared/compartmented)."""
    out = {c: [] for c, _, _ in CONDITIONS}
    for sc_label, params_m, json_file in SCALES:
        data = json.loads(Path(json_file).read_text())
        # Build per-condition step→(en,zh) maps
        per_cond = {}
        for cond, _, _ in CONDITIONS:
            rows = data.get(cond, [])
            per_cond[cond] = {r["step"]: (r["en"], r["zh"]) for r in rows}
        common_steps = sorted(set.intersection(*[set(d.keys()) for d in per_cond.values()]))
        if not common_steps:
            continue
        match_step = common_steps[-1]  # last step where all 3 have data
        for cond in per_cond:
            en, zh = per_cond[cond][match_step]
            out[cond].append((sc_label, params_m, en, zh))
        print(f"  scale {sc_label}: matched step = {match_step}")
    return out


def plot_lang_into(ax, finals, lang_idx, lang_label):
    for cond, color, marker in CONDITIONS:
        rows = finals[cond]
        if not rows:
            continue
        xs = [r[1] for r in rows]
        ys = [r[lang_idx] for r in rows]
        ax.plot(xs, ys, color=color, marker=marker, markersize=5,
                linewidth=1.2, label=cond)
    ax.set_xscale("log")
    xs = [r[1] for r in finals["shared"]]
    ax.set_xticks(xs)
    ax.set_xticklabels([scale_label(r[0], r[1]) for r in finals["shared"]])
    ax.minorticks_off()
    ax.set_xlabel("Model size")
    ax.set_ylabel(f"{lang_label} val at matched step (nats)")


def _bottom_legend(fig, handles_labels, ncol):
    handles, labels = handles_labels
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               frameon=False, handlelength=1.3, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
               markerscale=1.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1))


def fig_lang(finals, lang_idx, lang_label, out_path, legend_loc=None):
    figsize = (3.3, 2.3) if legend_loc is not None else (3.3, 2.6)
    fig, ax = plt.subplots(figsize=figsize)
    plot_lang_into(ax, finals, lang_idx, lang_label)
    if legend_loc is None:
        _bottom_legend(fig, ax.get_legend_handles_labels(), ncol=3)
    else:
        ax.legend(loc=legend_loc, frameon=False, handlelength=1.3,
                  handletextpad=0.5)
        fig.tight_layout()
    fig.savefig(out_path)
    print(f"  {out_path}")


def fig_gap(out_path, legend_loc=None):
    figsize = (3.3, 2.3) if legend_loc is not None else (3.3, 2.6)
    fig, ax = plt.subplots(figsize=figsize)
    plot_gap_into(ax)
    handles, labels = ax.get_legend_handles_labels()
    if legend_loc is None:
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   frameon=False, handlelength=1.3, handletextpad=0.5,
                   columnspacing=1.2, bbox_to_anchor=(0.5, -0.02),
                   markerscale=2.0)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
    else:
        ax.legend(handles, labels, loc=legend_loc, frameon=False,
                  handlelength=1.3, handletextpad=0.5)
        fig.tight_layout()
    fig.savefig(out_path)
    print(f"  {out_path}")


SCALE_COLORS = {
    "12-256":  "#440154",
    "24-512":  "#3b528b",
    "24-768":  "#21918c",
    "24-1024": "#fde725",
}
SCALE_MARKERS = {"12-256": "o", "24-512": "s", "24-768": "^", "24-1024": "D"}


def plot_gap_into(ax):
    """comp - shared EN val over training steps, one line per scale."""
    for sc_label, params_m, json_file in SCALES:
        data = json.loads(Path(json_file).read_text())
        shared_rows = {r["step"]: r["en"] for r in data.get("shared", [])}
        comp_rows = {r["step"]: r["en"] for r in data.get("compartmented", [])}
        common = sorted(set(shared_rows) & set(comp_rows))
        if not common:
            continue
        steps = common
        gaps = [comp_rows[s] - shared_rows[s] for s in steps]
        ax.plot(steps, gaps, color=SCALE_COLORS[sc_label],
                marker=SCALE_MARKERS[sc_label], markersize=3,
                linewidth=1.0, label=scale_label(sc_label, params_m))
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("comp − shared EN val (nats)")


def fig_combined(finals, out_path):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.75, 2.8))
    plot_lang_into(ax_l, finals, 2, "EN")
    plot_gap_into(ax_r)
    # Two separate legends, each in its own panel (different label families).
    ax_l.legend(loc="upper right", frameon=False, fontsize=7,
                handlelength=1.3, handletextpad=0.5)
    ax_r.legend(loc="upper right", frameon=False, fontsize=7,
                handlelength=1.3, handletextpad=0.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  {out_path}")


def main():
    setup_paper_style()
    Path("../figures").mkdir(exist_ok=True)
    finals = _load_finals()
    fig_lang(finals, 2, "EN", Path("../figures/multilingual_scaling_en.pdf"),
             legend_loc="lower left")
    fig_lang(finals, 3, "ZH", Path("../figures/multilingual_scaling_zh.pdf"))
    fig_gap(Path("../figures/multilingual_scaling_gap.pdf"),
            legend_loc="upper right")
    fig_combined(finals, Path("../figures/multilingual_scaling_final.pdf"))


if __name__ == "__main__":
    main()

"""The phase transition, as motion in a fixed basis.

Two c=8 runs at 8-256 that differ only in translation ratio: tr=0.25 stays
compartmentalised (final layer-4 cossim 0.257) and tr=0.5 breaks (0.919). Every
saved checkpoint is forwarded on the same canonical batch, the eight compartment
centroids are projected into ONE basis, and their paths are drawn.

    stuck  -> the centroids wander, roughly in parallel, and stay apart
    broken -> the centroids fall together

Underneath, the formal per-compartment val loss and the cossim metric on a
shared step axis, with a scrubber that ties a position in the paths to a
position on those curves.

Three things this script is careful about
----------------------------------------

*Fixed basis.* Refitting PCA per checkpoint manufactures motion: the basis
rotates and the points appear to move even when nothing changed. The basis here
is fit once and every frame is projected into it. `--basis` picks what it is fit
on; the default fits on the centroids of ALL frames, which is still one fixed
basis but guarantees it spans the directions the centroids actually travel in.
`final` reproduces the fit-on-the-last-checkpoint convention -- worth knowing
that for the tr=0.5 run the last checkpoint is the COLLAPSED configuration, so
that basis is fit on the small residual spread that survives the collapse and is
correspondingly noisy. `extremes` fits on the first and last frames together,
which is the usual compromise.

*Scale.* The RMS activation norm grows about 4.5x over training (0.6 -> 2.7).
Left alone, that radial blow-up dominates the animation and every centroid marches
outward regardless of what the compartments are doing. Each checkpoint is divided
by its own global RMS first, so the picture is about geometry, not gain. The raw
norm is plotted as its own curve rather than being silently discarded.

*Val loss.* Cross-tr comparisons use `loss_compartment_i` from val_metrics.json,
the formal per-compartment eval on a fixed monolingual set -- not the
training-time val loss, which mixes in translation pairs and is biased downward
by an amount that scales with tr, i.e. exactly the axis being compared here.

Reads the npz cache from compute_twin_clouds.py --mode trajectory.
Run from experiment/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np

from plot_baseline_val_curves import setup_paper_style
from _run_paths import C1_BASELINE_8_256


CACHE = Path("../cache_twinclouds")
FIGDIR = Path("../figures")

RUNS = [
    ("tr025", "bpe16384-rope-8-256/93c21853_s64", "tr = 0.25  (stays compartmentalised)"),
    ("tr05",  "bpe16384-rope-8-256/8143ba31_s64", "tr = 0.50  (breaks)"),
]

COMP_CMAP = "tab10"
RUN_COLOR = {"tr025": "#c2410c", "tr05": "#1d4ed8"}


def comp_colors(c: int):
    cmap = plt.get_cmap(COMP_CMAP)
    return [cmap(i % 10) for i in range(c)]


# ────────────────────────────── loading ──────────────────────────────────────

def load_traj(name: str, layer: int):
    d = np.load(CACHE / f"traj_{name}_L{layer}.npz")
    return {
        "steps": d["steps"].astype(int),
        "sub": d["sub_feats"].astype(np.float32),     # (S, c, Nsub, D)
        "cen": d["centroids"].astype(np.float64),     # (S, c, D)
        "c": int(d["c"]),
        "tr": float(d["tr"]),
    }


def normalise(tr: dict):
    """Divide every frame by its own global RMS norm.

    RMS is computed on the token subsample, which is the same rows at every
    step, so the scalar is comparable frame to frame.
    """
    S = tr["sub"]                                              # float32
    rms = np.sqrt((S.astype(np.float64) ** 2).sum(axis=3).mean(axis=(1, 2)))
    tr["rms"] = rms
    tr["sub_n"] = (S / rms[:, None, None, None].astype(np.float32))
    tr["cen_n"] = tr["cen"] / rms[:, None, None]
    return tr


def per_frame_cossim(sub_n: np.ndarray) -> np.ndarray:
    """Mean off-diagonal per-token cosine sim at each frame. Scale-invariant, so
    it does not matter that this runs on the normalised features."""
    S, c, N, D = sub_n.shape
    out = np.empty(S)
    for t in range(S):
        F = sub_n[t]
        Fn = F / (np.linalg.norm(F, axis=2, keepdims=True) + 1e-12)
        out[t] = np.mean([(Fn[i] * Fn[j]).sum(1).mean()
                          for i in range(c) for j in range(i + 1, c)])
    return out


def centroid_spread(cen_n: np.ndarray) -> np.ndarray:
    """Mean distance of a compartment centroid from the centroid of centroids."""
    g = cen_n.mean(axis=1, keepdims=True)
    return np.sqrt(((cen_n - g) ** 2).sum(axis=2)).mean(axis=1)


# ────────────────────────────── the basis ────────────────────────────────────

def fit_basis(tr: dict, kind: str):
    """One fixed 2-D basis. Returns (mean, components(2,D), label)."""
    cen = tr["cen_n"]                                    # (S, c, D)
    if kind == "centroids-all":
        X = cen.reshape(-1, cen.shape[2])
        lab = "PCA on compartment centroids, all checkpoints"
    elif kind == "final":
        X = cen[-1]
        lab = "PCA on compartment centroids, final checkpoint"
    elif kind == "extremes":
        X = np.concatenate([cen[0], cen[-1]], axis=0)
        lab = "PCA on compartment centroids, first + final checkpoint"
    elif kind == "tokens-final":
        X = tr["sub_n"][-1].reshape(-1, cen.shape[2])
        lab = "PCA on token residuals, final checkpoint"
    else:
        raise ValueError(kind)
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    evr = (S**2) / (S**2).sum()
    return mu, Vt[:2], evr[:2], lab


def project(V, mu, arr):
    return np.einsum("...d,kd->...k", arr - mu, V)


# ────────────────────────────── val curves ───────────────────────────────────

def val_curve(key: str, c: int):
    m = json.loads(Path("val_metrics.json").read_text())[key]
    steps = np.asarray(m["checkpoints"], dtype=float)
    per = np.array([m["metrics"][f"loss_compartment_{i}"] for i in range(c)],
                   dtype=float)
    return steps, per.mean(0)


def c1_floor():
    m = json.loads(Path("val_metrics.json").read_text())[C1_BASELINE_8_256]
    return (np.asarray(m["checkpoints"], dtype=float),
            np.asarray(m["metrics"]["loss_compartment_0"], dtype=float))


# ────────────────────────────── drawing ──────────────────────────────────────

def prepare(args):
    """Everything both the static figure and the animation need."""
    data = {}
    for name, key, label in RUNS:
        tr = normalise(load_traj(name, args.layer))
        mu, V, evr, blab = fit_basis(tr, args.basis)
        tr["cen_p"] = project(V, mu, tr["cen_n"])            # (S, c, 2)
        tr["sub_p"] = project(V, mu, tr["sub_n"])            # (S, c, N, 2)
        tr["cos"] = per_frame_cossim(tr["sub_n"])
        tr["spread"] = centroid_spread(tr["cen_n"])
        tr["evr"], tr["basis_label"], tr["label"], tr["key"] = evr, blab, label, key
        vs, vl = val_curve(key, tr["c"])
        tr["val_steps"], tr["val"] = vs, vl
        data[name] = tr
    return data


def path_limits(tr, use_tokens: bool, pad=0.10):
    pts = [tr["cen_p"].reshape(-1, 2)]
    if use_tokens:
        # 1st/99th percentile so a few far tokens do not set the frame.
        P = tr["sub_p"].reshape(-1, 2)
        lo = np.percentile(P, 1, axis=0)
        hi = np.percentile(P, 99, axis=0)
        pts.append(np.stack([lo, hi]))
    A = np.concatenate(pts, axis=0)
    lo, hi = A.min(0), A.max(0)
    span = (hi - lo).max()
    ctr = (hi + lo) / 2
    half = span * (0.5 + pad)
    return (ctr[0] - half, ctr[0] + half), (ctr[1] - half, ctr[1] + half)


def draw_bottom(ax_val, ax_cos, data):
    """Left: the compartmentalisation penalty. Right: the cossim metric.

    Plotted as excess over the c=1 baseline rather than as raw val loss. The two
    raw curves are within ~0.2 nats of each other on an axis that spans 11 down
    to 4, so at full scale they lie on top of one another and the panel shows
    nothing; the gap to the single-compartment floor is both the quantity the
    paper reports and the one that actually separates.
    """
    cs, cl = c1_floor()
    for name, tr in data.items():
        # Both runs and the baseline are saved on the same loggy grid, so the
        # subtraction is pointwise; guard rather than assume.
        if not np.array_equal(tr["val_steps"], cs):
            raise ValueError("val grids differ from the c=1 baseline grid")
        ax_val.plot(tr["val_steps"], tr["val"] - cl, color=RUN_COLOR[name],
                    lw=1.4, marker="o", ms=2.5, label=f"tr={tr['tr']:g}")
        ax_cos.plot(tr["steps"], tr["cos"], color=RUN_COLOR[name], lw=1.4,
                    label=f"tr={tr['tr']:g}")
    ax_val.set_xscale("log"); ax_cos.set_xscale("log")
    ax_val.set_yscale("log")
    ax_val.set_xlabel("step"); ax_cos.set_xlabel("step")
    ax_val.set_ylabel("excess val loss over\nc=1 baseline (nats)")
    ax_cos.set_ylabel("cosine sim.\n(layer 4)")
    ax_val.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax_cos.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax_cos.axhline(0, color="k", lw=0.5, alpha=0.4, ls=":")
    # A log axis spanning ~2.2 down to ~0.2 gets exactly one decade tick, so
    # label the minor ticks too or the panel carries no readable scale.
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax_val.yaxis.set_major_formatter(ScalarFormatter())
    ax_val.yaxis.set_minor_formatter(ScalarFormatter())
    ax_val.tick_params(axis="y", which="minor", labelsize=6)
    for a in (ax_val, ax_cos):
        a.grid(alpha=0.25)


def build_figure(data, args):
    setup_paper_style()
    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.52], hspace=0.34,
                          wspace=0.22, top=0.88, bottom=0.09,
                          left=0.09, right=0.97)
    ax_paths = [fig.add_subplot(gs[0, i]) for i in range(2)]
    ax_val = fig.add_subplot(gs[1, 0])
    ax_cos = fig.add_subplot(gs[1, 1])
    draw_bottom(ax_val, ax_cos, data)
    return fig, ax_paths, ax_val, ax_cos


def render_paths(ax, tr, upto: int, args, static: bool):
    """Draw the centroid paths of `tr` up to frame `upto` (inclusive)."""
    c = tr["c"]
    cols = comp_colors(c)
    ax.clear()
    xl, yl = tr["_lims"]

    if args.show_tokens:
        sub = tr["sub_p"][upto]
        keep = tr["_tok_idx"]
        for j in range(c):
            ax.scatter(sub[j, keep, 0], sub[j, keep, 1], s=1.4, color=cols[j],
                       alpha=0.14, linewidths=0, zorder=1)

    P = tr["cen_p"]
    for j in range(c):
        ax.plot(P[: upto + 1, j, 0], P[: upto + 1, j, 1], color=cols[j],
                lw=0.9, alpha=0.55, zorder=2)
        ax.scatter(P[0, j, 0], P[0, j, 1], s=9, facecolors="none",
                   edgecolors=cols[j], linewidths=0.7, zorder=3)
        ax.scatter(P[upto, j, 0], P[upto, j, 1], s=34, color=cols[j],
                   zorder=4, edgecolors="white", linewidths=0.5)

    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.set_xlabel(f"fixed PC1 ({100*tr['evr'][0]:.0f}%)")
    ax.set_ylabel(f"fixed PC2 ({100*tr['evr'][1]:.0f}%)")
    ax.set_title(
        f"{tr['label']}\nstep {tr['steps'][upto]:,}   "
        f"cos {tr['cos'][upto]:.3f}   spread {tr['spread'][upto]:.3f}",
        fontsize=8.5, pad=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--basis", default="centroids-all",
                    choices=["centroids-all", "final", "extremes", "tokens-final"])
    ap.add_argument("--show-tokens", action="store_true", default=True)
    ap.add_argument("--no-show-tokens", dest="show_tokens", action="store_false")
    ap.add_argument("--n-tokens", type=int, default=300,
                    help="tokens scattered per compartment per animation frame")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--hold", type=int, default=10,
                    help="extra frames held on the last checkpoint")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    FIGDIR.mkdir(exist_ok=True)
    data = prepare(args)

    rng = np.random.Generator(np.random.PCG64(11))
    for name, tr in data.items():
        tr["_lims"] = path_limits(tr, args.show_tokens)
        n = tr["sub_p"].shape[2]
        tr["_tok_idx"] = rng.choice(n, size=min(args.n_tokens, n), replace=False)
        print(f"  {name}: {len(tr['steps'])} frames, c={tr['c']}, "
              f"cos {tr['cos'][0]:+.3f} -> {tr['cos'][-1]:+.3f}, "
              f"spread {tr['spread'][0]:.3f} -> {tr['spread'][-1]:.3f}, "
              f"rms {tr['rms'][0]:.2f} -> {tr['rms'][-1]:.2f}")
        print(f"     basis: {tr['basis_label']}  "
              f"evr {tr['evr'][0]:.2f}/{tr['evr'][1]:.2f}")

    tag = f"_{args.tag}" if args.tag else ""

    # ── static: the whole path, both runs ────────────────────────────────────
    fig, ax_paths, ax_val, ax_cos = build_figure(data, args)
    for ax, (name, _, _) in zip(ax_paths, RUNS):
        tr = data[name]
        render_paths(ax, tr, len(tr["steps"]) - 1, args, static=True)
    fig.suptitle(
        "Compartment centroids across training, projected into one fixed basis "
        "(hollow marker = step 100)\n"
        "same architecture, same seed, only the translation ratio differs",
        fontsize=9.5, y=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"centroid_paths{tag}.{ext}", bbox_inches="tight",
                    dpi=220)
    plt.close(fig)
    print(f"  wrote figures/centroid_paths{tag}.{{pdf,png}}")

    if not args.animate:
        return

    # ── animation ───────────────────────────────────────────────────────────
    fig, ax_paths, ax_val, ax_cos = build_figure(data, args)
    n_frames = min(len(data[n]["steps"]) for n, _, _ in RUNS)
    # Pin the step axes before adding the scrubber: an axvline placed at x=1
    # counts as data on a log axis and drags the lower limit down to 10^0,
    # squeezing the whole curve into the right-hand third of the panel.
    first_step = float(data[RUNS[0][0]]["steps"][0])
    for a in (ax_val, ax_cos):
        a.set_xlim(a.get_xlim())
    scrub = [ax_val.axvline(first_step, color="k", lw=1.0, alpha=0.55),
             ax_cos.axvline(first_step, color="k", lw=1.0, alpha=0.55)]
    fig.suptitle(
        "Compartment centroids across training, projected into one fixed basis\n"
        "same architecture, same seed, only the translation ratio differs",
        fontsize=9.5, y=1.0)

    def update(f):
        i = min(f, n_frames - 1)
        for ax, (name, _, _) in zip(ax_paths, RUNS):
            render_paths(ax, data[name], i, args, static=False)
        step = data[RUNS[0][0]]["steps"][i]
        for s in scrub:
            s.set_xdata([step, step])
        return []

    total = n_frames + args.hold
    anim = manim.FuncAnimation(fig, update, frames=total, interval=1000 / args.fps,
                               blit=False)
    out = FIGDIR / f"centroid_trajectory{tag}.gif"
    anim.save(out, writer=manim.PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"  wrote {out}  ({total} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()

"""Twin clouds: the same tokens, forwarded under every compartment, in a fixed
basis.

Two views of the same layer-4 features, one per figure:

  twin_clouds_2d      Naive union PCA of the raw residuals. One PCA is fit on
                      the union of all c blocks, everything is projected into
                      it, and a thin line joins each token in compartment 0 to
                      its twin in compartment j. Compartmentalised -> displaced
                      clouds and long crossing lines. Unified -> the clouds sit
                      on top of each other and the lines shrink to nothing.

  twin_clouds_book    The honest version. A model that uses compartment
                      embeddings carries a per-compartment CONSTANT offset, so
                      the naive view above shows c separated blobs even when the
                      content is shared, and "they look separated" is not
                      evidence of anything. Split it:

                          h = g + (mu_j - g) + (h - mu_j)
                              ^^^   ^^^^^^^^^   ^^^^^^^^^
                            global   identity     content

                      Identity is by construction constant within a compartment,
                      so it is exactly one coordinate per sheet -- the sheets get
                      zero thickness along it. Content is the top-2 PCs of the
                      pooled per-compartment mean-centred residual. The result is
                      c sheets stacked along an identity axis: a unified model
                      gives congruent pages joined by vertical threads, a
                      compartmentalised one gives pages whose content disagrees
                      and threads that slant.

The identity axis is the leading difference-of-compartment-means direction. With
c compartments the identity subspace is (c-1)-dimensional and, empirically, close
to isotropic -- PC1 holds only ~1/(c-1) of it -- so the axis is labelled with the
fraction it actually captures rather than being passed off as the whole story.

Reads the npz cache written by compute_twin_clouds.py --mode models.
Run from experiment/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from plot_baseline_val_curves import setup_paper_style


CACHE = Path("../cache_twinclouds")
FIGDIR = Path("../figures")

# Panel order and the label each one carries. The three models differ only in
# what pushes the compartments together: nothing, a shared init, or an explicit
# alignment loss.
PANELS = [
    ("default", "Default init\n(no alignment pressure)"),
    ("copyemb", "Init-copy\n(embeddings + head shared at init)"),
    ("infonce", "InfoNCE\n(explicit layer-4 alignment)"),
]

COMP_CMAP = "tab10"


def comp_colors(c: int):
    cmap = plt.get_cmap(COMP_CMAP)
    return [cmap(i % 10) for i in range(c)]


def load(name: str, layer: int) -> np.ndarray:
    """(c, N, D) float64 layer-`layer` residuals."""
    d = np.load(CACHE / f"clouds_{name}_L{layer}.npz")
    return d["feats"].astype(np.float64)


def pca_basis(X: np.ndarray, k: int):
    """Top-k PCs of X (rows = samples). Returns (mean, components, evr)."""
    mu = X.mean(0)
    Xc = X - mu
    # D is small (256) here, so an SVD on the full matrix is fine.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / max(len(Xc) - 1, 1)
    evr = var / var.sum()
    return mu, Vt[:k], evr[:k]


def robust_lim(v: np.ndarray, q: float = 1.0, pad: float = 0.06):
    """Axis limits that ignore the extreme `q` percent at each end.

    A handful of tokens sit far outside the bulk -- in the init-copy model a
    dozen points reach content PC1 = -8 while the cloud lives inside [-3, 2] --
    and letting them set the limits squashes everything that matters into a
    smear.
    """
    lo, hi = np.percentile(v, [q, 100 - q])
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    m = (hi - lo) * pad
    return lo - m, hi + m


def mean_pairwise_cossim(F: np.ndarray) -> float:
    """Mean off-diagonal per-token cosine similarity -- the paper's metric."""
    c = F.shape[0]
    Fn = F / (np.linalg.norm(F, axis=2, keepdims=True) + 1e-12)
    vals = [(Fn[i] * Fn[j]).sum(1).mean() for i in range(c) for j in range(i + 1, c)]
    return float(np.mean(vals))


def twin_distance_ratio(F: np.ndarray) -> float:
    """Median distance from a token to its twin, over the cloud's own radius.

    In the collapsed panels the twin lines are too short to see, which is the
    point but also indistinguishable from "the lines were never drawn". This
    puts a number on it: 1.0 means a token sits as far from its twin as a
    typical token sits from the centre of the cloud, ~0 means twins coincide.
    Computed in the full D-dimensional space, not the 2-D projection.
    """
    c, N, D = F.shape
    g = F.reshape(c * N, D).mean(0)
    radius = np.sqrt(((F.reshape(c * N, D) - g) ** 2).sum(1).mean())
    d = [np.linalg.norm(F[j] - F[0], axis=1) for j in range(1, c)]
    return float(np.median(np.concatenate(d)) / radius)


def decompose(F: np.ndarray):
    """h = global + identity + content.

    Returns (identity_1d, content_2d, stats) where identity_1d is (c,) -- one
    constant per compartment -- and content_2d is (c, N, 2).
    """
    c, N, D = F.shape
    g = F.reshape(c * N, D).mean(0)
    mu = F.mean(1)                      # (c, D) compartment means
    ident = mu - g                      # (c, D) identity component

    # Identity axis: leading difference-of-means direction.
    _, S_id, Vt_id = np.linalg.svd(ident, full_matrices=False)
    id_evr = (S_id**2) / (S_id**2).sum()
    z = ident @ Vt_id[0]                # (c,) constant per compartment

    # Content axes: top-2 PCs of the pooled per-compartment mean-centred data.
    resid = (F - mu[:, None, :]).reshape(c * N, D)
    _, comp, evr = pca_basis(resid, 2)
    xy = np.einsum("cnd,kd->cnk", F - mu[:, None, :], comp)

    tot = ((F.reshape(c * N, D) - g) ** 2).sum()
    stats = {
        "id_frac_of_var": float(N * (ident**2).sum() / tot),
        "id_pc1_frac_of_id": float(id_evr[0]),
        "content_evr": evr.tolist(),
        "cossim_raw": mean_pairwise_cossim(F),
        "cossim_content": mean_pairwise_cossim(F - mu[:, None, :]),
    }
    return z, xy, stats


# ─────────────────────────── figure 1: naive union PCA ───────────────────────

def draw_2d(ax, F: np.ndarray, n_show: int, n_lines: int, rng, title: str):
    c, N, D = F.shape
    mu, comp, evr = pca_basis(F.reshape(c * N, D), 2)
    P = np.einsum("cnd,kd->cnk", F - mu, comp)          # (c, N, 2)

    show = rng.choice(N, size=min(n_show, N), replace=False)
    line_rows = show[: min(n_lines, len(show))]
    cols = comp_colors(c)

    # Twin lines first so the points sit on top of them.
    segs, seg_cols = [], []
    for j in range(1, c):
        for r in line_rows:
            segs.append([P[0, r], P[j, r]])
            seg_cols.append(cols[j])
    ax.add_collection(LineCollection(segs, colors=seg_cols, linewidths=0.28,
                                     alpha=0.30, zorder=1))
    for j in range(c):
        ax.scatter(P[j, show, 0], P[j, show, 1], s=3.0, color=cols[j],
                   linewidths=0, alpha=0.85, zorder=2,
                   label=f"comp {j}" if j < 8 else None)

    cs = mean_pairwise_cossim(F)
    tw = twin_distance_ratio(F)
    ax.set_title(f"{title}\ncos = {cs:.3f}    twin dist = {tw:.2f} $\\times$ cloud radius",
                 pad=6)
    ax.set_xlabel(f"PC1 ({100*evr[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100*evr[1]:.0f}%)")
    ax.set_xlim(*robust_lim(P[:, show, 0].ravel(), q=2.0))
    ax.set_ylim(*robust_lim(P[:, show, 1].ravel(), q=2.0))
    # datalim, not box: equal aspect is what makes the displacement between
    # clouds readable as a distance, but "box" resizes the axes themselves and
    # leaves the three panels misaligned at different heights.
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    return P


def figure_2d(args, feats: dict):
    setup_paper_style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.0, 3.9))
    rng = np.random.Generator(np.random.PCG64(args.plot_seed))
    for ax, (name, label) in zip(axes, PANELS):
        draw_2d(ax, feats[name], args.n_show, args.n_lines, rng, label)
    axes[0].legend(loc="upper left", frameon=False, fontsize=6.0,
                   markerscale=2.2, ncol=2, handletextpad=0.2,
                   columnspacing=0.6, borderpad=0.2)
    fig.suptitle(
        "Same 4096 tokens, forwarded under all 8 compartments, layer-4 residual "
        "in one shared PCA basis\n"
        "each line joins a token in compartment 0 to its twin in compartment $j$",
        fontsize=9, y=1.06)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"twin_clouds_2d.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("  wrote figures/twin_clouds_2d.{pdf,png}")


# ─────────────────────── figure 2: content (+) identity ──────────────────────

def draw_book(ax, F, n_show, n_lines, rng, title, elev, azim):
    """Pages of a book: one sheet per compartment, threads through the stack.

    The sheets sit at evenly spaced slots ordered by their identity coordinate,
    not at the coordinate itself. The identity subspace is (c-1)-dimensional and
    close to isotropic, so its leading direction spaces the sheets unevenly and,
    in the unified models, barely at all -- true, but it collapses the pages into
    a slab and hides the claim this panel is making, which is about whether the
    pages are CONGRUENT. The companion strip below plots the identity
    coordinates on their real shared scale, so the collapse is not hidden, just
    moved to the axis that can show it.

    Threads join consecutive sheets rather than fanning out from one base sheet:
    a token then becomes a single strand running through the whole book, and
    "vertical strand" reads directly as "this token lands in the same place in
    every compartment".
    """
    c, N, D = F.shape
    z_true, xy, st = decompose(F)

    order = np.argsort(z_true)
    slot = {j: i for i, j in enumerate(order)}
    show = rng.choice(N, size=min(n_show, N), replace=False)
    line_rows = show[: min(n_lines, len(show))]
    cols = comp_colors(c)

    # A harder clip than the 2-D panel uses: in 3-D a single far point drags the
    # whole box and flattens the pages into a smear.
    xl = robust_lim(xy[:, show, 0].ravel(), q=2.0)
    yl = robust_lim(xy[:, show, 1].ravel(), q=2.0)

    segs, seg_cols = [], []
    for a, b in zip(order[:-1], order[1:]):
        for r in line_rows:
            segs.append([(xy[a, r, 0], xy[a, r, 1], slot[a]),
                         (xy[b, r, 0], xy[b, r, 1], slot[b])])
            seg_cols.append(cols[b])
    ax.add_collection3d(Line3DCollection(segs, colors=seg_cols,
                                         linewidths=0.3, alpha=0.28))
    for j in order:
        ax.scatter(xy[j, show, 0], xy[j, show, 1],
                   np.full(len(show), slot[j]),
                   s=2.4, color=cols[j], linewidths=0, alpha=0.85,
                   depthshade=False)

    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_zlim(-0.6, c - 0.4)
    ax.set_xlabel("content PC1", labelpad=-8, fontsize=7)
    ax.set_ylabel("content PC2", labelpad=-8, fontsize=7)
    ax.set_zlabel("compartment", labelpad=-8, fontsize=7)
    ax.set_zticks(range(c))
    ax.set_zticklabels([str(j) for j in order], fontsize=5)
    ax.set_title(
        f"{title}\n"
        f"cos {st['cossim_raw']:.3f} $\\to$ {st['cossim_content']:.3f} "
        f"with identity removed",
        pad=-2)
    ax.tick_params(labelsize=5, pad=-3)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 0.95))
    return st


def draw_identity_strip(ax, F, title, xlim, show_ylabel):
    """The identity coordinates on their true, shared scale."""
    c = F.shape[0]
    z_true, _, st = decompose(F)
    cols = comp_colors(c)
    for j in range(c):
        ax.plot([z_true[j], z_true[j]], [0.12, 0.88], color=cols[j], lw=1.6,
                solid_capstyle="butt")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("identity coordinate (shared scale)", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("identity", fontsize=7)
    ax.set_title(f"identity = {100*st['id_frac_of_var']:.1f}% of variance",
                 fontsize=7.5, pad=3)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, axis="x")


def figure_book(args, feats: dict):
    setup_paper_style()
    rng = np.random.Generator(np.random.PCG64(args.plot_seed))

    fig = plt.figure(figsize=(12.0, 5.0))
    # 3D axes leave a lot of slack inside their cell, so the strip row is pulled
    # up tight against them rather than centred in its own band.
    gs = fig.add_gridspec(2, len(PANELS), height_ratios=[1.0, 0.13],
                          hspace=0.02, wspace=0.10,
                          top=0.80, bottom=0.09, left=0.03, right=0.97)

    # One shared scale for the identity strips, so the panels are comparable.
    all_z = np.concatenate([decompose(feats[n])[0] for n, _ in PANELS])
    zl = robust_lim(all_z, q=0.0, pad=0.12)

    stats = {}
    for i, (name, label) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        ax.grid(False)
        stats[name] = draw_book(ax, feats[name], args.n_show, args.n_lines,
                                rng, label, args.elev, args.azim)
        axs = fig.add_subplot(gs[1, i])
        draw_identity_strip(axs, feats[name], label, zl, show_ylabel=(i == 0))

    fig.suptitle(
        "Content $\\oplus$ identity: each token's layer-4 residual split into a "
        "per-compartment constant (identity) and the rest (content)\n"
        "top: one sheet per compartment, a thread per token running through the "
        "stack -- vertical threads mean the pages are congruent\n"
        "bottom: where those sheets actually sit along the identity axis, on one "
        "shared scale",
        fontsize=8.5, y=0.995)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"twin_clouds_book.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("  wrote figures/twin_clouds_book.{pdf,png}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-show", type=int, default=200,
                    help="tokens scattered per compartment")
    ap.add_argument("--n-lines", type=int, default=120,
                    help="tokens that get a twin line (<= n-show)")
    ap.add_argument("--plot-seed", type=int, default=3)
    ap.add_argument("--elev", type=float, default=16)
    ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--only", default="", help="2d | book")
    args = ap.parse_args()

    FIGDIR.mkdir(exist_ok=True)
    feats = {name: load(name, args.layer) for name, _ in PANELS}
    for name, F in feats.items():
        print(f"  loaded {name}: {F.shape}")

    if args.only in ("", "2d"):
        figure_2d(args, feats)
    if args.only in ("", "book"):
        stats = figure_book(args, feats)
        out = FIGDIR / "twin_clouds_stats.json"
        out.write_text(json.dumps(stats, indent=2))
        print(f"  wrote {out}")
        for k, v in stats.items():
            print(f"    {k}: identity={100*v['id_frac_of_var']:.1f}% of variance, "
                  f"ID-PC1 holds {100*v['id_pc1_frac_of_id']:.0f}% of identity, "
                  f"cos {v['cossim_raw']:.3f} -> {v['cossim_content']:.3f}")


if __name__ == "__main__":
    main()

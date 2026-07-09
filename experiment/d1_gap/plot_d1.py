"""Plot the D1 K×K cross-eval heatmaps for semantic vs random partition."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--out", default="d1_gap.pdf")
    args = ap.parse_args()

    data = json.loads(Path(args.matrices).read_text())
    ks = sorted(int(k) for k in data)

    fig, axes = plt.subplots(2, len(ks), figsize=(3.8 * len(ks), 7.2))
    if len(ks) == 1:
        axes = axes.reshape(2, 1)
    vmin = 6.9
    vmax = 7.25
    for col, K in enumerate(ks):
        for row, partition in enumerate(["semantic", "random"]):
            ax = axes[row, col]
            mat = np.array(data[str(K)][partition])
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="magma_r")
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            ax.set_xticklabels([f"c{c}" for c in range(K)], fontsize=8)
            ax.set_yticklabels([f"c{c}" for c in range(K)], fontsize=8)
            if row == 0:
                ax.set_title(f"K={K} {partition}", fontsize=11)
            else:
                ax.set_title(f"K={K} {partition}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{partition}-LM trained on cluster", fontsize=9)
            if row == 1:
                ax.set_xlabel("evaluated on cluster val", fontsize=9)
            # Text-annotate each cell
            for i in range(K):
                for j in range(K):
                    txt = f"{mat[i, j]:.2f}"
                    color = "white" if mat[i, j] > (vmax + vmin) / 2 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=6 if K == 8 else 8, color=color)
            # Compute + display gap
            diag = np.diag(mat).mean()
            off = (mat.sum() - np.trace(mat)) / (K * K - K) if K > 1 else diag
            gap = off - diag
            ax.text(0.5, -0.20, f"diag {diag:.3f}, off {off:.3f}, gap {gap:+.3f}",
                    ha="center", va="top", transform=ax.transAxes, fontsize=9,
                    fontweight="bold" if partition == "semantic" else "normal")

    fig.suptitle(
        "Phase-A D1 gap: semantic tf-idf clusters vs random partition\n"
        "8-32 mini-LMs, 500 iters, 100k FineWeb docs",
        fontsize=12, y=1.01,
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="val loss (nats/token)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

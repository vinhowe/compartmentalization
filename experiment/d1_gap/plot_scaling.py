"""Plot D1 gap as a function of model size, per K."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_gaps(mat: list[list[float]]) -> tuple[float, float, float]:
    m = np.array(mat)
    K = m.shape[0]
    diag = float(np.diag(m).mean())
    off = float((m.sum() - np.trace(m)) / (K * K - K)) if K > 1 else diag
    return diag, off, off - diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Dir containing d1_matrices_<size>.json for each size")
    ap.add_argument("--sizes", default="8-32,8-64,8-128,8-256",
                    help="Comma-separated list of size tier names")
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--out", default="d1_scaling.pdf")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    sizes = args.sizes.split(",")
    ks = [int(x) for x in args.ks.split(",")]

    # Rough param counts for 8-<D> at vocab=16384
    D_from_size = {s: int(s.split("-")[1]) for s in sizes}
    params = {}
    for s, D in D_from_size.items():
        # Simple approx: 2 * V * D (embed + head) + 8L * (12 * D^2)  — attention 4*D^2 + FFN 8*D^2
        p = 2 * 16384 * D + 8 * 12 * D * D
        params[s] = p / 1e6  # M params

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    colors = {2: "tab:blue", 4: "tab:orange", 8: "tab:green"}
    for K in ks:
        gaps = []
        for s in sizes:
            path = data_root / f"d1_matrices_{s}.json"
            data = json.loads(path.read_text())
            _, _, gap = compute_gaps(data[str(K)]["semantic"])
            gaps.append(gap)
        ax.plot([params[s] for s in sizes], gaps, "-o", color=colors[K],
                label=f"K={K}", lw=2, markersize=7)

    # Random baseline for reference (should be flat near zero)
    for K in ks:
        gaps = []
        for s in sizes:
            path = data_root / f"d1_matrices_{s}.json"
            data = json.loads(path.read_text())
            _, _, gap = compute_gaps(data[str(K)]["random"])
            gaps.append(gap)
        ax.plot([params[s] for s in sizes], gaps, "--", color=colors[K],
                alpha=0.3, lw=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("model params (M)")
    ax.set_ylabel("D1 gap = off-diag − diag (nats)")
    ax.set_title("D1 gap grows with both K and model size\n"
                 "(dashed = random control, ≈ 0 everywhere)")
    ax.legend(title="clusters", loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

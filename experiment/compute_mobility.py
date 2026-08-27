"""How much of the flow difference is just embeddings slowing down?

With weight decay 0 and a constant learning rate there is no rotational
equilibrium: Adam's roughly fixed-size updates make weight norms grow, and since
everything downstream of an embedding sees it through a LayerNorm, only the
DIRECTION of a row matters. A growing norm with a fixed step size therefore means
a shrinking angular step -- the run self-anneals with no schedule having been
set.

That gives a deflationary alternative to the autocatalysis story: maybe the flow
f(q) = dq/dlog10(step) turns over at low tr not because the aligning force
weakens, but because the embedding rows simply stop being able to rotate. And
maybe the 3x flow gap at matched q (tr=0.25 at 1M vs tr=0.5 at 80k) is mostly
that the older model has larger norms.

So measure the mobility directly:

    mu = median over rows of  angle( E_row(t1), E_row(t2) )  /  (log10 t2 - log10 t1)

and compare f/mu across runs. If the matched-q gap collapses to ~1 under f/mu,
mobility explains it and the feedback story is unnecessary. If the gap survives,
mobility is a contributing factor and something else carries the rest.

Weights only -- no forward passes.

Run from experiment/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _run_paths import OUT_ROOT
from compute_embedding_order import RUNS, named_steps, BASE_VOCAB, load_blocks


def rows_at(ckpt: Path, ids: np.ndarray, c: int) -> np.ndarray:
    E, _, _, _ = load_blocks(ckpt, ids, c)
    return E.reshape(-1, E.shape[-1])          # (c*N, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="tr01,tr025,tr05,tr075")
    ap.add_argument("--n-tokens", type=int, default=2048)
    ap.add_argument("--out", default="mobility.json")
    args = ap.parse_args()

    rng = np.random.Generator(np.random.PCG64(0))
    ids = np.sort(rng.choice(BASE_VOCAB, args.n_tokens, replace=False))

    order = json.loads(Path("embedding_order.json").read_text())
    res = {}
    for name in args.runs.split(","):
        rel, tr = RUNS[name]
        rd = OUT_ROOT / rel
        cfg = json.loads((rd / "meta" / "config.json").read_text())
        c = int(cfg["experiment"]["n_compartments"])
        steps = [s for s in named_steps(rd) if s >= 1000]
        # Thin to ~24 points so consecutive pairs span a meaningful log gap;
        # adjacent 10k-step checkpoints late in training differ by too little
        # angle to measure against float noise.
        keep = [steps[int(i)] for i in np.linspace(0, len(steps) - 1, 24).round()]
        keep = sorted(set(keep))

        prev, prev_step = None, None
        pts = []
        for st in keep:
            R = rows_at(rd / "checkpoints" / f"step-{st:06d}", ids, c)
            Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
            if prev is not None:
                cosd = np.clip((prev * Rn).sum(1), -1.0, 1.0)
                ang = float(np.median(np.arccos(cosd)))
                dlog = np.log10(st) - np.log10(prev_step)
                qk = order[name]["data"]
                q_mid = None
                for s in (st, prev_step):
                    if str(s) in qk:
                        q_mid = qk[str(s)]["q_emb"]
                        break
                pts.append({"step_from": prev_step, "step_to": st,
                            "angle": ang, "dlog": float(dlog),
                            "mu": ang / float(dlog), "q": q_mid,
                            "rms": float(np.linalg.norm(R, axis=1).mean())})
            prev, prev_step = Rn, st
        res[name] = {"tr": tr, "points": pts}
        print(f"{name} (tr={tr}):")
        for p in pts[::4]:
            print(f"   {p['step_from']:>8,}->{p['step_to']:<8,} "
                  f"q={p['q']:.4f}  mu={p['mu']:.4f} rad/decade  "
                  f"|E|={p['rms']:.3f}")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

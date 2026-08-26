"""The order parameter of the compartmentalisation transition, read off weights.

The trunk of this model is shared across all compartments. The ONLY
per-compartment parameters are the vocabulary blocks:

    wte.weight       (c * base_vocab + 1, d)   block j = rows [j*V, (j+1)*V)
    lm_head.weight   (c * base_vocab + 1, d)   same blocking
    comp_emb.weight  (c, d)                    one constant offset per compartment

So "compartmentalised" versus "unified" is not a diffuse property of the network,
it is a statement about those blocks: does compartment i embed token t the same
way compartment j does? If it does, the shared trunk receives the same vector for
the same underlying token and there is only one circuit to learn. If it does not,
the trunk has to serve c unrelated codes.

That makes

    q = mean_t mean_{i<j} cos( E_i[t], E_j[t] )

an order parameter in the physical sense: near 0 in the broken (compartmentalised)
phase, near 1 in the symmetric (unified) phase. It is computed from the weights
alone -- no forward pass, no data -- which makes it cheap enough to evaluate at
every checkpoint and independent of the activation-space measurement the rest of
the analysis uses. Getting the same transition out of two independent
measurements is the point.

Also recorded, for the optimiser-side story:

  * ||E|| and the per-step displacement of the blocks, which set the scale of any
    "effective learning rate" argument (cf. Omnigrok, Liu et al. 2022, where the
    timing of a delayed transition is governed by weight norm rather than by the
    loss).
  * the same statistic with each block's mean removed, which separates "the codes
    agree" from "the codes share a constant offset".

Run from experiment/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _run_paths import OUT_ROOT

BASE_VOCAB = 16384

RUNS = {
    "tr025": ("bpe16384-rope-8-256/93c21853_s64", 0.25),
    "tr05": ("bpe16384-rope-8-256/8143ba31_s64", 0.5),
    # The near-zero-tr reference, trained ~3x longer. Its cossim creeps upward
    # without transitioning, so it is the control for "slow drift".
    "tr011": ("bpe16384-rope-8-256/868ef4a8_s64", 0.0111),
    # The rest of the c=8, wd=0 absolute-mode tr ladder. Six values is enough to
    # draw a flow field in q rather than argue about two trajectories.
    "tr01":  ("bpe16384-rope-8-256/ed486d1e_s64", 0.10),
    "tr075": ("bpe16384-rope-8-256/39909470_s64", 0.75),
    "tr10":  ("bpe16384-rope-8-256/33d22f84_s64", 1.00),
}


def named_steps(run_dir: Path) -> list[int]:
    out = []
    for d in (run_dir / "checkpoints").iterdir():
        if d.name.startswith("step-") and (d / "model.pt").exists():
            try:
                out.append(int(d.name.split("-")[1]))
            except ValueError:
                pass
    return sorted(out)


def load_blocks(ckpt: Path, ids: np.ndarray, c: int):
    """Return (E, H, comp_emb, norms) with E,H shape (c, len(ids), d).

    mmap keeps this to the handful of rows actually indexed rather than paging
    in the whole 134 MB embedding table for every checkpoint.
    """
    sd = torch.load(ckpt / "model.pt", map_location="cpu", mmap=True,
                    weights_only=True)
    sd = sd.get("model", sd) if ("model" in sd and not any(
        k.endswith("wte.weight") for k in sd)) else sd
    key = {k.split("_orig_mod.")[-1]: k for k in sd}
    wte, head = sd[key["transformer.wte.weight"]], sd[key["lm_head.weight"]]
    ce = sd[key["comp_emb.weight"]].float().numpy()

    idx = np.concatenate([ids + j * BASE_VOCAB for j in range(c)])
    t = torch.from_numpy(idx)
    E = wte[t].float().numpy().reshape(c, len(ids), -1)
    H = head[t].float().numpy().reshape(c, len(ids), -1)
    norms = {
        "wte_rms": float(wte.float().pow(2).mean().sqrt()),
        "head_rms": float(head.float().pow(2).mean().sqrt()),
        "comp_emb_rms": float(np.sqrt((ce ** 2).mean())),
    }
    return E, H, ce, norms


def order_parameter(B: np.ndarray):
    """(q_raw, q_centred) for a (c, N, d) stack of per-compartment codes."""
    c = B.shape[0]

    def q(X):
        Xn = X / (np.linalg.norm(X, axis=2, keepdims=True) + 1e-12)
        return float(np.mean([(Xn[i] * Xn[j]).sum(1).mean()
                              for i in range(c) for j in range(i + 1, c)]))

    return q(B), q(B - B.mean(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="tr025,tr05,tr011")
    ap.add_argument("--n-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="embedding_order.json")
    args = ap.parse_args()

    rng = np.random.Generator(np.random.PCG64(args.seed))
    ids = np.sort(rng.choice(BASE_VOCAB, size=args.n_tokens, replace=False))

    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name in args.runs.split(","):
        rel, tr = RUNS[name]
        run_dir = OUT_ROOT / rel
        cfg = json.loads((run_dir / "meta" / "config.json").read_text())
        c = int(cfg["experiment"]["n_compartments"])
        steps = named_steps(run_dir)
        print(f"{name}: c={c} tr={tr} {len(steps)} checkpoints")
        rows = res.setdefault(name, {"tr": tr, "c": c, "steps": [], "data": {}})
        for i, st in enumerate(steps):
            if str(st) in rows["data"]:
                continue
            try:
                E, H, ce, norms = load_blocks(
                    run_dir / "checkpoints" / f"step-{st:06d}", ids, c)
            except Exception as exc:
                print(f"  step {st}: FAILED {exc!r}")
                continue
            qe, qe_c = order_parameter(E)
            qh, qh_c = order_parameter(H)
            rows["data"][str(st)] = {
                "q_emb": qe, "q_emb_centred": qe_c,
                "q_head": qh, "q_head_centred": qh_c, **norms,
            }
            if (i + 1) % 20 == 0 or i == len(steps) - 1:
                print(f"  [{i+1}/{len(steps)}] step {st:>8d} "
                      f"q_emb={qe:+.4f} q_head={qh:+.4f}")
                out_path.write_text(json.dumps(res, indent=2))
        rows["steps"] = sorted(int(k) for k in rows["data"])
        out_path.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

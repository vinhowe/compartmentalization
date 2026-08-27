"""Is there a conversion step inside the trunk?

The model translates between compartments almost losslessly (0.002-0.034 nats on
the translation half of a translation row) while their vocabularies stay largely
unaligned. So it is converting somehow. The hypothesis is that it keeps c
separate codes and learns a general conversion between them inside the trunk,
rather than merging the codes -- which is exactly the structure the cross-lingual
embedding literature would predict, since independently trained embedding spaces
for different languages are related by an approximately linear map (Mikolov et
al. 2013; Conneau et al. MUSE).

The test: run the same text through compartment i and through compartment j,
take the activations at some depth, and ask how well ONE matrix W turns the first
into the second. Three numbers per (layer, pair):

    err_identity = |h_i - h_j| / |h_j|      how wrong you are doing nothing
    err_linear   = |W h_i - h_j| / |h_j|    how wrong you are after converting
    dist_from_I  = |W - I| / |I|            how far the conversion is from nothing

Readings:
  * err_identity high, err_linear low, dist_from_I large
        -> separate codes plus a real linear translator. The conversion exists.
  * both errors low, dist_from_I small
        -> the codes already agree; there is nothing to convert.
  * both errors high
        -> no LINEAR translator; whatever it does is nonlinear (or absent).

W is fit on half the tokens and scored on the other half. With d=256 the matrix
has 65k parameters and a 4096-token batch gives ~1M equations, so it is
comfortably overdetermined, but the split makes the claim honest anyway.

The comparison that matters is at MATCHED agreement: tr=0.25 at 1M and tr=0.5 at
~70k sit at the same q~0.08. If the stalled model needs a strong conversion there
and the breaking one does not, that difference is the mechanism, not a
consequence of one being further along.

Run from experiment/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "..")
from eval_utils import load_eval_model_from_checkpoint
from compute_cossim_sweep import load_canonical_batch, BASE_VOCAB
from _run_paths import OUT_ROOT, NO_INFONCE_8_256_BY_C, INFONCE_8_256_BY_C

# (label, run, step or None for final)
TARGETS = {
    "tr025@1M":    ("bpe16384-rope-8-256/93c21853_s64", 1000000),
    "tr05@70k":    ("bpe16384-rope-8-256/8143ba31_s64", 70000),
    "tr05@1M":     ("bpe16384-rope-8-256/8143ba31_s64", 1000000),
    "tr011@final": (NO_INFONCE_8_256_BY_C[8], None),
    "infonce":     (INFONCE_8_256_BY_C[8], None),
}


def final_step(run_dir: Path) -> int:
    return max(int(d.name.split("-")[1])
               for d in (run_dir / "checkpoints").iterdir()
               if d.name.startswith("step-") and (d / "model.pt").exists())


@torch.no_grad()
def activations(model, batch, c, layer, device):
    """(c, N, D) activations at `layer`; layer -1 = raw token embedding."""
    out = []
    if layer == -1:
        wte = model.transformer.wte.weight
        flat = batch.flatten()
        for j in range(c):
            out.append(wte[flat + j * BASE_VOCAB].float().cpu().numpy())
        return np.stack(out)
    for j in range(c):
        x = batch + j * BASE_VOCAB
        cid = torch.full_like(batch, j)
        _, _, h = model(x, compartment_ids=cid, capture_layer=layer)
        out.append(h.flatten(0, 1).float().cpu().numpy())
    return np.stack(out)


def probe_pair(hi, hj, tr_idx, te_idx):
    """Fit W on tr_idx, score on te_idx."""
    A, B = hi[tr_idx], hj[tr_idx]
    # least squares: minimise |A W^T - B|
    W, *_ = np.linalg.lstsq(A, B, rcond=None)        # (D, D), maps A -> B
    At, Bt = hi[te_idx], hj[te_idx]
    denom = np.linalg.norm(Bt) + 1e-12
    err_id = np.linalg.norm(At - Bt) / denom
    err_lin = np.linalg.norm(At @ W - Bt) / denom
    D = hi.shape[1]
    dist_I = np.linalg.norm(W - np.eye(D)) / np.sqrt(D)
    return err_id, err_lin, dist_I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--layers", default="-1,0,1,2,4,6,7")
    ap.add_argument("--pairs", type=int, default=6, help="compartment pairs sampled")
    ap.add_argument("--out", default="translator_probe.json")
    args = ap.parse_args()

    batch = load_canonical_batch(seed=0)
    layers = [int(x) for x in args.layers.split(",")]
    rng = np.random.Generator(np.random.PCG64(0))
    res = {}

    for name in args.targets.split(","):
        rel, step = TARGETS[name]
        run_dir = OUT_ROOT / rel
        if not run_dir.exists():
            print(f"  {name}: MISSING")
            continue
        cfg = json.loads((run_dir / "meta" / "config.json").read_text())
        c = int(cfg["experiment"]["n_compartments"])
        st = step if step is not None else final_step(run_dir)
        model, _, _ = load_eval_model_from_checkpoint(
            run_dir / "checkpoints" / f"step-{st:06d}", run_dir, args.device)
        model.eval()

        pairs = [(int(i), int(j)) for i, j in
                 rng.choice(c, size=(args.pairs, 2)) if i != j]
        while len(pairs) < args.pairs:
            i, j = rng.integers(0, c, 2)
            if i != j:
                pairs.append((int(i), int(j)))

        res[name] = {"run": rel, "step": st, "c": c, "layers": {}}
        print(f"\n{name}  (step {st:,})")
        print(f"  {'layer':>6s} {'do-nothing err':>15s} {'after-convert err':>18s} "
              f"{'conversion size':>16s}")
        for L in layers:
            H = activations(model, batch.to(args.device), c, L, args.device)
            N = H.shape[1]
            perm = rng.permutation(N)
            tr_idx, te_idx = perm[: N // 2], perm[N // 2:]
            eid, eli, dI = [], [], []
            for (i, j) in pairs:
                a, b, d_ = probe_pair(H[i].astype(np.float64),
                                      H[j].astype(np.float64), tr_idx, te_idx)
                eid.append(a); eli.append(b); dI.append(d_)
            row = {"err_identity": float(np.mean(eid)),
                   "err_linear": float(np.mean(eli)),
                   "dist_from_identity": float(np.mean(dI))}
            res[name]["layers"][str(L)] = row
            print(f"  {L:>6d} {row['err_identity']:>15.3f} "
                  f"{row['err_linear']:>18.3f} {row['dist_from_identity']:>16.3f}")
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

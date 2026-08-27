"""The alignment force as the optimiser actually applies it. Corrected.

Two bugs invalidated the first attempt at this, and both are worth recording.

1. `load_eval_model_from_checkpoint` returns a **bfloat16** model unless a dtype
   is passed. Every gradient here was therefore computed in bf16, which has ~8
   mantissa bits -- hopeless for a quantity that is a ~0.1% component of the
   gradient, and the giveaway was output like "+560.0000" and "+900.0000",
   spaced exactly like bf16 at that magnitude. Everything now runs in float32.

2. Adam's second moment was being *reconstructed* by averaging squared gradients
   over a few dozen frozen-weight batches. Any embedding row whose token was not
   sampled got v = 0, hence a preconditioner of 1/(sqrt(0)+1e-8) = 1e8, and those
   rows -- pure sampling noise -- swamped the result. The real v cannot be
   reconstructed this way because it carries a million steps of history: measured
   on the saved state, it has a median of 1.4e-10 and essentially no zeros
   (0.005%), which no short frozen-weight average reproduces.

So this uses the REAL optimizer state. That constrains the measurement to the
final checkpoint, since `_rolling` is the only checkpoint that saves
optimizer.pt -- one point per run rather than a trajectory. Enough to answer
"does Adam's rescaling change the sign or the ranking of the two forces?", not
enough to answer "does the pull grow with q".

Optimizer state is indexed decay-group-first (all dim>=2 parameters in model
order, then all dim<2), verified to match all 52 parameters by shape; `wte` is
index 0.

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
from _run_paths import OUT_ROOT
from compute_embedding_order import RUNS, BASE_VOCAB
from compute_alignment_forces import (load_tokens, trans_batch, mono_batch,
                                      lm_loss, q_of, grad_wrt_wte, cos)

EPS = 1e-8


def adam_v_for_wte(run_dir: Path, model) -> torch.Tensor:
    """Real exp_avg_sq for transformer.wte.weight, from the saved optimizer."""
    opt = torch.load(run_dir / "checkpoints" / "_rolling" / "optimizer.pt",
                     map_location="cpu", weights_only=False)
    state = opt["state"]
    nps = list(model.named_parameters())
    order = [n for n, p in nps if p.dim() >= 2] + [n for n, p in nps if p.dim() < 2]
    idx = order.index("transformer.wte.weight")
    v = state[idx]["exp_avg_sq"].float()
    assert tuple(v.shape) == tuple(model.transformer.wte.weight.shape), v.shape
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="tr025,tr05")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="alignment_forces_adam.json")
    args = ap.parse_args()

    tokens = load_tokens()
    res = {}
    for name in args.runs.split(","):
        rel, tr = RUNS[name]
        run_dir = OUT_ROOT / rel
        cfg = json.loads((run_dir / "meta" / "config.json").read_text())
        c = int(cfg["experiment"]["n_compartments"])
        T = int(cfg["model"]["block_size"])
        trans_token = BASE_VOCAB * c

        model, _, _ = load_eval_model_from_checkpoint(
            run_dir / "checkpoints" / "_rolling", run_dir, args.device,
            dtype=torch.float32)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)
        assert next(model.parameters()).dtype == torch.float32

        v = adam_v_for_wte(run_dir, model).to(args.device)
        precond = 1.0 / (v.sqrt() + EPS)
        del v

        reps = []
        for rep in range(args.repeats):
            rng = np.random.Generator(np.random.PCG64(4242 + 31 * rep))
            batches, base_ids = [], []
            half = T // 2
            for _ in range(args.accum):
                xt, yt, ct = trans_batch(tokens, args.batch, T, c, trans_token,
                                         rng, args.device)
                xm, ym, cm_ = mono_batch(tokens, args.batch, T, c, rng,
                                         args.device)
                batches.append((xt, yt, ct, xm, ym, cm_))
                base_ids.append((xt[:, 1:half] % BASE_VOCAB).reshape(-1))
                base_ids.append((xm % BASE_VOCAB).reshape(-1))
            ids = torch.unique(torch.cat(base_ids))
            ids = ids[ids < BASE_VOCAB]

            wte = model.transformer.wte.weight
            gq = grad_wrt_wte(model, q_of(wte, ids, c))
            q_val = float(q_of(wte, ids, c).detach())

            gt = torch.zeros_like(gq); gm = torch.zeros_like(gq)
            for (xt, yt, ct, xm, ym, cm_) in batches:
                gt += grad_wrt_wte(model, lm_loss(model, xt, yt, ct)) / args.accum
                gm += grad_wrt_wte(model, lm_loss(model, xm, ym, cm_)) / args.accum

            sel = torch.cat([ids + j * BASE_VOCAB for j in range(c)])
            gqs = gq[sel]
            qhat = gqs / (gqs.norm() + 1e-24)
            pt, pm = (gt * precond)[sel], (gm * precond)[sel]
            gts, gms = gt[sel], gm[sel]
            reps.append({
                "q": q_val,
                "push_raw_trans": float((-gts * qhat).sum()),
                "push_raw_mono": float((-gms * qhat).sum()),
                "push_adam_trans": float((-pt * qhat).sum()),
                "push_adam_mono": float((-pm * qhat).sum()),
                "cos_raw_trans": cos(-gts, gqs), "cos_raw_mono": cos(-gms, gqs),
                "cos_adam_trans": cos(-pt, gqs), "cos_adam_mono": cos(-pm, gqs),
            })
            del gq, gt, gm, gqs, pt, pm, gts, gms
            torch.cuda.empty_cache()

        agg = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
        agg.update({k + "_sd": float(np.std([r[k] for r in reps])) for k in reps[0]})
        res[name] = {"tr": tr, "step": "final(_rolling)", **agg}
        print(f"\n{name} (tr={tr}), float32, real Adam v, {args.repeats} draws")
        print(f"  q = {agg['q']:.4f}")
        print(f"  RAW   translation {agg['push_raw_trans']:+.3e} +/- {agg['push_raw_trans_sd']:.1e}"
              f"   monolingual {agg['push_raw_mono']:+.3e} +/- {agg['push_raw_mono_sd']:.1e}")
        print(f"  ADAM  translation {agg['push_adam_trans']:+.3e} +/- {agg['push_adam_trans_sd']:.1e}"
              f"   monolingual {agg['push_adam_mono']:+.3e} +/- {agg['push_adam_mono_sd']:.1e}")
        del model, precond
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

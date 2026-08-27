"""Measure the force, instead of inferring it from the trajectory.

The phase portrait shows that the flow in q is autocatalytic above a critical tr
and decays below it. That is a description of the dynamics, not a mechanism: it
says the rate depends on the state in a particular way, but it does not say what
produces the rate. The proposed mechanism -- translation rows pull the
compartment codes together, monolingual rows do not, and the pull grows as the
codes overlap -- is a claim about gradients, so it should be tested on gradients.

At a checkpoint, with E the embedding table:

    g_q     = d q / d E                     the direction that increases agreement
    g_trans = d L_translation / d E         from translation-format rows only
    g_mono  = d L_monolingual / d E         from ordinary rows only

and the quantities of interest are

    cos(-g_trans, g_q)   does a step on translation loss increase agreement?
    cos(-g_mono,  g_q)   does a step on monolingual loss decrease it?
    <-g, g_q_hat>        the actual push on q per unit step, so the two
                         components can be combined as tr and (1-tr) weight them

Measured across checkpoints spanning the run, this tests the autocatalysis claim
directly: the translation pull should GROW as q grows.

Construction notes
------------------
Translation rows follow the training format exactly (src/data.py,
_fill_translation_rows_standard): [TRANS, content in i's vocab, TRANS, content in
j's vocab], with targets shifted and the final position ignored. Content is real
fineweb tokens.

For monolingual rows each compartment is given INDEPENDENT text, which is the
honest analogue of training, where compartments hold different slices of the
corpus. Feeding all compartments the same text would manufacture an alignment
pressure that training never applies.

Both losses are measured on the same fixed token pool, and q is defined over that
same pool, so every gradient touches the same embedding rows and the inner
products are not dominated by rows only one of them reaches.

Caveat kept explicit: these are raw gradients. Training uses Adam, whose
per-coordinate normalisation rescales the update, so the cosines here describe
the force from the loss geometry rather than the exact step taken.

Run from experiment/.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "..")
from eval_utils import load_eval_model_from_checkpoint
from _run_paths import OUT_ROOT
from compute_embedding_order import RUNS, named_steps, BASE_VOCAB

VAL_BIN = "../data/fineweb350B-dedup-bpe16384/fineweb350b-dedup_val_*.bin"


def load_tokens() -> np.ndarray:
    f = sorted(glob.glob(VAL_BIN))[0]
    with open(f, "rb") as fh:
        header = np.frombuffer(fh.read(256 * 4), dtype=np.int32)
        ntok = int(header[2])
        return np.frombuffer(fh.read(ntok * 4), dtype=np.uint32).astype(np.int64)


def spans(tokens: np.ndarray, n: int, length: int, rng) -> torch.Tensor:
    out = np.empty((n, length), dtype=np.int64)
    for i in range(n):
        s = int(rng.integers(0, len(tokens) - length - 1))
        out[i] = tokens[s: s + length]
    return torch.from_numpy(out)


def mono_batch(tokens, B, T, c, rng, device):
    """Ordinary rows: each row is one compartment, each compartment its own text."""
    base = spans(tokens, B, T + 1, rng)
    comp = torch.from_numpy(rng.integers(0, c, size=B))
    x = base[:, :T] + (comp[:, None] * BASE_VOCAB)
    y = base[:, 1:] + (comp[:, None] * BASE_VOCAB)
    cids = comp[:, None].expand(B, T).contiguous()
    return x.to(device), y.to(device), cids.to(device)


def trans_batch(tokens, B, T, c, trans_token, rng, device):
    """[TRANS, src content, TRANS, dst content] -- the training format."""
    half = T // 2
    content = spans(tokens, B, half - 1, rng)                    # (B, half-1)
    src = torch.from_numpy(rng.integers(0, c, size=B))
    dst = torch.from_numpy(rng.integers(0, c - 1, size=B))
    dst = dst + (dst >= src).long()                              # dst != src
    c_src = content + src[:, None] * BASE_VOCAB
    c_dst = content + dst[:, None] * BASE_VOCAB

    x = torch.empty((B, T), dtype=torch.long)
    x[:, 0] = trans_token
    x[:, 1:half] = c_src
    x[:, half] = trans_token
    x[:, half + 1:] = c_dst

    y = torch.empty((B, T), dtype=torch.long)
    y[:, 0] = c_src[:, 0]
    y[:, 1:half - 1] = c_src[:, 1:]
    y[:, half - 1] = trans_token
    y[:, half] = c_dst[:, 0]
    y[:, half + 1:-1] = c_dst[:, 1:]
    y[:, -1] = -1

    cids = torch.empty((B, T), dtype=torch.long)
    cids[:, :half] = src[:, None]
    cids[:, half:] = dst[:, None]
    return x.to(device), y.to(device), cids.to(device)


def lm_loss(model, x, y, cids):
    logits, _ = model(x, targets=None, compartment_ids=cids,
                      full_sequence_logits=True)[:2]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           y.reshape(-1), ignore_index=-1)


def q_of(wte: torch.Tensor, ids: torch.Tensor, c: int) -> torch.Tensor:
    """Differentiable order parameter over the token pool `ids`."""
    idx = torch.cat([ids + j * BASE_VOCAB for j in range(c)])
    E = wte[idx].view(c, len(ids), -1)
    En = E / (E.norm(dim=2, keepdim=True) + 1e-12)
    tot, n = 0.0, 0
    for i in range(c):
        for j in range(i + 1, c):
            tot = tot + (En[i] * En[j]).sum(1).mean()
            n += 1
    return tot / n


def grad_wrt_wte(model, scalar) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    scalar.backward()
    return model.transformer.wte.weight.grad.detach().clone()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-24))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="tr025,tr05")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-ckpt", type=int, default=8)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--pool", type=int, default=2048)
    ap.add_argument("--out", default="alignment_forces.json")
    ap.add_argument("--repeats", type=int, default=2,
                    help="independent batch draws per checkpoint; the spread "
                         "between them is the noise floor for these cosines")
    ap.add_argument("--seed0", type=int, default=1234)
    args = ap.parse_args()

    tokens = load_tokens()
    res = {}
    out_path = Path(args.out)
    if out_path.exists():
        res = json.loads(out_path.read_text())

    for name in args.runs.split(","):
        rel, tr = RUNS[name]
        run_dir = OUT_ROOT / rel
        cfg = json.loads((run_dir / "meta" / "config.json").read_text())
        c = int(cfg["experiment"]["n_compartments"])
        T = int(cfg["model"]["block_size"])
        trans_token = BASE_VOCAB * c

        steps = named_steps(run_dir)
        pick = [steps[int(i)] for i in
                np.linspace(0, len(steps) - 1, args.n_ckpt).round()]
        rows = res.setdefault(name, {"tr": tr, "c": c, "points": {}})
        print(f"{name}: tr={tr} c={c} -> {pick}")

        for st in pick:
            if str(st) in rows["points"]:
                continue
            model, _, _ = load_eval_model_from_checkpoint(
                run_dir / "checkpoints" / f"step-{st:06d}", run_dir, args.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(True)

            # Build the batches FIRST and define q over the base tokens they
            # actually contain. Sampling the q pool independently makes g_q and
            # the loss gradients live on almost disjoint sets of embedding rows,
            # and the cosine between them is then diluted to ~0 by rows where one
            # side is identically zero -- a measurement artifact, not a null.
            #
            # Each repeat is an independent batch draw. The alignment-relevant
            # part of an LM gradient is a ~0.1% sliver of its norm, so the spread
            # across repeats is the only honest way to know whether a difference
            # between the two forces means anything.
            reps = []
            for rep in range(args.repeats):
                rng = np.random.Generator(np.random.PCG64(args.seed0 + 977 * rep))
                batches, base_ids = [], []
                half = T // 2
                for _ in range(args.accum):
                    xt, yt, ct = trans_batch(tokens, args.batch, T, c,
                                             trans_token, rng, args.device)
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

                gt = torch.zeros_like(gq)
                gm = torch.zeros_like(gq)
                lt = lm = 0.0
                for (xt, yt, ct, xm, ym, cm_) in batches:
                    loss = lm_loss(model, xt, yt, ct)
                    lt += float(loss.detach()) / args.accum
                    gt += grad_wrt_wte(model, loss) / args.accum

                    loss = lm_loss(model, xm, ym, cm_)
                    lm += float(loss.detach()) / args.accum
                    gm += grad_wrt_wte(model, loss) / args.accum

                # Restrict every vector to the rows g_q can be nonzero on, so
                # the norms in the cosine are over the same subspace.
                sel = torch.cat([ids + j * BASE_VOCAB for j in range(c)])
                gqs, gts, gms = gq[sel], gt[sel], gm[sel]
                gq_hat = gqs / (gqs.norm() + 1e-24)
                reps.append({
                    "q": q_val,
                    "cos_trans": cos(-gts, gqs),
                    "cos_mono": cos(-gms, gqs),
                    "push_trans": float((-gts * gq_hat).sum()),
                    "push_mono": float((-gms * gq_hat).sum()),
                    "norm_trans": float(gts.norm()),
                    "norm_mono": float(gms.norm()),
                    "n_tokens": int(len(ids)),
                    "loss_trans": lt, "loss_mono": lm,
                })
                del gq, gt, gm, gqs, gts, gms
                torch.cuda.empty_cache()

            agg = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
            agg.update({k + "_spread": float(np.std([r[k] for r in reps]))
                        for k in ("cos_trans", "cos_mono",
                                  "push_trans", "push_mono")})
            agg["repeats"] = args.repeats
            rows["points"][str(st)] = agg
            r = rows["points"][str(st)]
            print(f"  step {st:>8d}  q={r['q']:.4f}  "
                  f"push_trans={r['push_trans']:+.3e} +/- {r['push_trans_spread']:.1e}   "
                  f"push_mono={r['push_mono']:+.3e} +/- {r['push_mono_spread']:.1e}")
            del model
            torch.cuda.empty_cache()
            out_path.write_text(json.dumps(res, indent=2))

    out_path.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

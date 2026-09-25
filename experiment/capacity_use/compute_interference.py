"""Signed gradient interference across scale (after Huang et al. 2026, arXiv 2605.29548).

Treat compartment 0 as the "task". Per model at step 1M:
  g_r   reference direction: gradient of compartment-0 loss summed over N_REF_BATCHES
        held-out batches.
  S     "task neurons": the top TOPK_FRAC of MLP hidden neurons by ||g_r|| restricted to the
        neuron's parameters (its c_fc row and c_proj column). S1 = the same, first layer only
        (Huang et al. use first-layer neurons). "trunk" = all transformer-block parameters.
  per measurement (N_MEAS aggregates of GROUP x B sequences each, all different text):
        cos_same  = cos(g of fresh compartment-0 text, g_r)
        cos_other = cos(g of text from another compartment, g_r)
        For c=1 there is no other compartment; its "other" batch is more compartment-0 text,
        so c=1 gives the fully-shared reference.
  share = mean cos_other / mean cos_same: 1 = the other forms train compartment 0's features
        as well as its own data does, 0 = separate machinery, < 0 = conflict.

Every compartment carries the same English text (compartment-mode tr ~0.01-0.03 at small
scale, tr = 0 at 1B), each run from its own loader offset so no text repeats.

Usage: python compute_interference.py OUT_DIR GPU[,GPU...]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

MAIN = Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression")
RUNS = MAIN / "out" / "translation-compression"
STEP = 1_000_000
B, T = 32, 64
# At step 1M a 32-sequence gradient is mostly noise (pilot: cosines ~0.01-0.03 with std
# as large, and c=1 "share" 0.47-0.85 where it must be ~1), so every gradient here is an
# aggregate, as in Huang et al.: reference 4,096 sequences, each measurement 1,024.
N_REF_BATCHES = 128
GROUP = 32          # loader batches per measurement (32 x 32 = 1,024 sequences)
N_MEAS = 8          # measurements each of same and other
TOPK_FRAC = 0.10


def run_keys():
    sys.path.insert(0, str(MAIN / "experiment"))
    import _run_paths as P
    k = {}
    for s in (32, 64, 128):
        k[(s, 1)] = P.C1_BASELINE_BY_SCALE[s]
        for c in (2, 8):
            k[(s, c)] = P.RUNS_SMALL_SCALE_TR01[(s, c)]
    k[(256, 1)] = P.C1_BASELINE_8_256
    for c in (2, 8):
        k[(256, c)] = P.NO_INFONCE_8_256_BY_C[c]
    for c in (1, 2, 8):
        k[(512, c)] = P.RUNS_8_512_LEGACY_BY_C[c]
    k[(1792, 1)] = P.RUN_1B_C1_BASELINE
    k[(1792, 2)] = P.RUN_1B_C2_NOTRANS
    k[(1792, 8)] = P.RUN_1B_C8_NOTRANS
    return k


def loader(eu, config, comp, n_model_c, device, skip_batches):
    V = eu.get_base_vocab_size(config)
    src = (config.data.compartment_val_bins or [config.data.val_bin])[0]
    shard = sorted(MAIN.glob(src))[0]
    it = iter(eu.SingleShardAssignedValLoader(
        str(shard), B=B, T=T, base_vocab_size=V, max_compartments=n_model_c,
        assignment=eu.Assignment(kind=0, src=comp), device=device, permute_tokens=False,
        permute_inputs=config.experiment.permute_input_tokens_per_compartment))
    for _ in range(skip_batches):
        next(it)
    return it


def grad_of(model, params, batches):
    model.zero_grad(set_to_none=True)
    for x, y, c in batches:
        _, loss = model(x, targets=y, compartment_ids=c)
        (loss / len(batches)).backward()
    return {n: p.grad.detach().clone() for n, p in params.items()}


def run_one(job):
    try:
        return _run_one(job)
    except Exception as e:  # noqa: BLE001
        import traceback
        return f"FAILED {job[0]}", f"{e}\n{traceback.format_exc()}"


def _run_one(job):
    (scale, c), key, gpu, out_dir = job
    out = Path(out_dir) / f"d{scale}_c{c}.json"
    if out.exists():
        return str(out), "cached"
    sys.path[:0] = [str(MAIN), str(MAIN / "experiment")]
    import eval_utils as eu
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    t0 = time.time()
    run_dir = RUNS / key
    model, config, n_model_c = eu.load_eval_model_from_checkpoint(
        run_dir / "checkpoints" / f"step-{STEP:06d}", run_dir, device, dtype=torch.float32)
    model.eval()
    n_c = int(config.experiment.n_compartments)
    blocks = model.transformer.h
    params = {n: p for n, p in model.named_parameters() if n.startswith("transformer.h.")}

    # Disjoint text: reference, same-compartment batches, then one span per other compartment.
    ref_it = loader(eu, config, 0, n_model_c, device, 0)
    g_r = grad_of(model, params, [next(ref_it) for _ in range(N_REF_BATCHES)])
    same_it = loader(eu, config, 0, n_model_c, device, N_REF_BATCHES)
    others = list(range(1, n_c)) or [0]
    span = GROUP * N_MEAS
    other_its = {j: loader(eu, config, j, n_model_c, device, N_REF_BATCHES + span * (1 + k))
                 for k, j in enumerate(others)}

    # Task neurons from g_r: norm over each hidden unit's c_fc row and c_proj column.
    norms = []
    for L, blk in enumerate(blocks):
        fc = g_r[f"transformer.h.{L}.mlp.c_fc.weight"]          # (4d, d)
        pr = g_r[f"transformer.h.{L}.mlp.c_proj.weight"]        # (d, 4d)
        norms.append((fc.pow(2).sum(1) + pr.pow(2).sum(0)).sqrt())
    norms = torch.stack(norms)                                   # (L, 4d)
    k_all = max(1, int(TOPK_FRAC * norms.numel()))
    thr = norms.flatten().topk(k_all).values[-1]
    S = norms >= thr
    S1 = torch.zeros_like(S)
    S1[0, norms[0].topk(max(1, int(TOPK_FRAC * norms.shape[1]))).indices] = True

    def vec(g, mask):
        parts = []
        for L in range(len(blocks)):
            m = mask[L]
            if m.any():
                parts += [g[f"transformer.h.{L}.mlp.c_fc.weight"][m].flatten(),
                          g[f"transformer.h.{L}.mlp.c_proj.weight"][:, m].flatten()]
        return torch.cat(parts)

    def trunk(g):
        return torch.cat([v.flatten() for v in g.values()])

    ref = {"S": vec(g_r, S), "S1": vec(g_r, S1), "trunk": trunk(g_r)}
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    rec = {k: {"same": [], "other": []} for k in ref}
    rng = np.random.default_rng(0)
    for i in range(N_MEAS):
        g_same = grad_of(model, params, [next(same_it) for _ in range(GROUP)])
        j = int(rng.choice(others))
        g_oth = grad_of(model, params, [next(other_its[j]) for _ in range(GROUP)])
        for k, f in (("S", lambda g: vec(g, S)), ("S1", lambda g: vec(g, S1)), ("trunk", trunk)):
            rec[k]["same"].append(cos(f(g_same), ref[k]))
            rec[k]["other"].append(cos(f(g_oth), ref[k]))
    res = {"scale": scale, "c": c, "key": key, "n_layer": len(blocks),
           "d_model": int(config.model.n_embd), "n_params": sum(p.numel() for p in model.parameters()),
           "translation_ratio": config.experiment.translation_ratio,
           "k_neurons": int(S.sum()), "seconds": None}
    for k, v in rec.items():
        same, oth = np.array(v["same"]), np.array(v["other"])
        res[k] = {"cos_same": float(same.mean()), "cos_same_std": float(same.std()),
                  "cos_other": float(oth.mean()), "cos_other_std": float(oth.std()),
                  "share": float(oth.mean() / same.mean())}
    res["seconds"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    return str(out), f"{res['seconds']:.0f}s"


if __name__ == "__main__":
    import multiprocessing as mp
    out_dir, gpus = sys.argv[1], [int(g) for g in sys.argv[2].split(",")]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    keys = run_keys()
    jobs = [(sc, k, gpus[i % len(gpus)], out_dir) for i, (sc, k) in enumerate(sorted(keys.items()))]
    with ProcessPoolExecutor(len(gpus), mp_context=mp.get_context("spawn")) as ex:
        for path, msg in ex.map(run_one, jobs):
            print(path, msg, flush=True)

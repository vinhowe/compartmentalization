"""Where in the network does unification happen?

The order parameter on the embedding blocks says the tr=0.5 model's embeddings
agree only weakly (q ~ 0.31) even though its layer-4 residuals agree strongly
(cos ~ 0.92). Those two numbers cannot both be the whole story, so this measures
the same quantity at every depth: the token embedding itself, then the residual
stream after each block.

If unification were achieved at the input -- compartment i learning to embed
token t exactly as compartment j does -- alignment would already be high at
layer 0 and flat thereafter. If instead the shared trunk is doing the work,
alignment climbs with depth, meaning the compartments keep distinct codes and
the trunk maps them onto a common representation.

Same canonical batch as everything else, so the layer-4 column reproduces the
number the paper reports.

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
from compute_cossim_sweep import load_canonical_batch, mean_cossim, BASE_VOCAB
from _run_paths import (OUT_ROOT, NO_INFONCE_8_256_BY_C, INFONCE_8_256_BY_C,
                        COPYEMB_8_256_BY_C)

MODELS = {
    "tr025":   ("bpe16384-rope-8-256/93c21853_s64", "tr=0.25 (stuck)"),
    "tr05":    ("bpe16384-rope-8-256/8143ba31_s64", "tr=0.50 (breaks)"),
    "tr011":   (NO_INFONCE_8_256_BY_C[8], "tr~0.011 (control)"),
    "copyemb": (COPYEMB_8_256_BY_C[8], "init-copy"),
    "infonce": (INFONCE_8_256_BY_C[8], "InfoNCE"),
}


def final_ckpt(run_dir: Path) -> Path:
    for n in ("_rolling", "latest"):
        d = run_dir / "checkpoints" / n
        if (d / "model.pt").exists():
            return d
    steps = sorted(int(d.name.split("-")[1])
                   for d in (run_dir / "checkpoints").iterdir()
                   if d.name.startswith("step-") and (d / "model.pt").exists())
    return run_dir / "checkpoints" / f"step-{steps[-1]:06d}"


@torch.no_grad()
def profile(run_dir: Path, batch_cpu, device: str):
    cfg = json.loads((run_dir / "meta" / "config.json").read_text())
    c = int(cfg["experiment"]["n_compartments"])
    n_layer = int(cfg["model"]["n_layer"])
    model, _, _ = load_eval_model_from_checkpoint(final_ckpt(run_dir), run_dir,
                                                  device)
    model.eval()
    batch = batch_cpu.to(device)

    out = {}

    # Depth -1: the raw token embedding, before the compartment offset and
    # before any block. Read straight off the weight matrix for the exact tokens
    # in the batch, so it is comparable row-for-row with the layers below.
    wte = model.transformer.wte.weight
    flat = batch.flatten()
    embs = [wte[flat + j * BASE_VOCAB].float().cpu() for j in range(c)]
    out["-1"] = float(np.mean([mean_cossim(embs[i], embs[j])
                               for i in range(c) for j in range(i + 1, c)]))

    for layer in range(n_layer):
        feats = []
        for ci in range(c):
            x = batch + ci * BASE_VOCAB
            cid = torch.full_like(batch, ci)
            _, _, h = model(x, compartment_ids=cid, capture_layer=layer)
            feats.append(h.flatten(0, 1).float().cpu())
        out[str(layer)] = float(np.mean([mean_cossim(feats[i], feats[j])
                                         for i in range(c)
                                         for j in range(i + 1, c)]))
    del model
    torch.cuda.empty_cache()
    return c, n_layer, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default="tr025,tr05,tr011,copyemb,infonce")
    ap.add_argument("--out", default="layerwise_alignment.json")
    args = ap.parse_args()

    batch = load_canonical_batch(seed=0)
    res = {}
    for name in args.models.split(","):
        rel, label = MODELS[name]
        run_dir = OUT_ROOT / rel
        if not run_dir.exists():
            print(f"  {name}: MISSING")
            continue
        c, n_layer, prof = profile(run_dir, batch, args.device)
        res[name] = {"label": label, "c": c, "n_layer": n_layer, "profile": prof}
        row = "  ".join(f"L{k}:{v:+.3f}" for k, v in prof.items())
        print(f"  {name:8s} {row}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

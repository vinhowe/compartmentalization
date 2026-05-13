"""Compute representational alignment (linear CKA) between compartments for
each (c, tr, wd) cell in the 8-256 rope sweep.

For each cell:
  1. Load the final checkpoint (model.pt at _rolling).
  2. Take a fixed batch of canonical fineweb val tokens (B, T).
  3. For each compartment i in 0..c-1:
     - x_i = canonical + base_vocab * i
     - cid = full of i
     - capture mid-layer (default: layer 4, same as InfoNCE) post-block hidden
       state via `capture_layer`. Shape: (B, T, D). Flatten to (B*T, D).
  4. Compute pairwise linear CKA across compartments. Report mean off-diagonal.

CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y)) with linear kernel,
which simplifies (after mean-centering) to:
  ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

Output: cossim_sweep.json — dict keyed by "<group>/<dirname>" with fields
  {c, tr_eff, tr_raw, mode, wd, mean_off_diag_cka, full_matrix_cka}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "..")
from eval_utils import load_eval_model_from_checkpoint


PROJECT_ROOT = Path("..")
GROUPS = ["bpe16384-rope-wd-n2", "bpe16384-rope-wd-n3-n8", "bpe16384-rope-8-256"]
VAL_BIN_PATTERN = "../data/fineweb350B-dedup-bpe16384/fineweb350b-dedup_val_*.bin"
BASE_VOCAB = 16384
DEFAULT_LAYER = 4
B = 64
T = 64


def load_canonical_batch(seed: int = 0) -> torch.Tensor:
    files = sorted(glob.glob(VAL_BIN_PATTERN))
    if not files:
        raise FileNotFoundError(VAL_BIN_PATTERN)
    with open(files[0], "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(ntok * 4), dtype=np.uint32)
    rng = np.random.Generator(np.random.PCG64(seed))
    out = np.empty((B, T), dtype=np.int64)
    for i in range(B):
        start = int(rng.integers(0, len(tokens) - T - 1))
        out[i] = tokens[start : start + T]
    return torch.from_numpy(out).long()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """x, y shape (N, D). Returns scalar in [0, 1]. Mean-centers both first.
    Linear CKA is invariant under orthogonal transforms and isotropic scaling,
    so it asks "is the same subspace covered" rather than "are these the same
    vectors"."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = (x.T @ y).pow(2).sum()
    auto_x = (x.T @ x).pow(2).sum()
    auto_y = (y.T @ y).pow(2).sum()
    return float(cross / (auto_x.sqrt() * auto_y.sqrt() + 1e-12))


def mean_cossim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Per-row cosine sim averaged. x, y shape (N, D). Each row in x is paired
    with the same-index row in y (same canonical token, different compartment).
    Range [-1, 1]; 1 = identical direction per token, -1 = opposite. Sensitive
    to absolute alignment, NOT invariant under rotations of the feature space.
    """
    xn = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    yn = y / (y.norm(dim=1, keepdim=True) + 1e-12)
    return float((xn * yn).sum(dim=1).mean())


@torch.no_grad()
def cka_for_run(run_dir: Path, batch_cpu: torch.Tensor, device: str,
                capture_layer: int = DEFAULT_LAYER):
    """Returns dict with c, mean_off_diag, full_matrix (list of lists)."""
    cf_path = run_dir / "meta" / "config.json"
    cobj = json.loads(cf_path.read_text())
    c = cobj["experiment"]["n_compartments"]

    ckpt_dir = run_dir / "checkpoints" / "_rolling"
    if not (ckpt_dir / "model.pt").exists():
        # Try latest as a symlink
        ckpt_dir = run_dir / "checkpoints" / "latest"
        if not (ckpt_dir / "model.pt").exists():
            return None

    model, _, _ = load_eval_model_from_checkpoint(ckpt_dir, run_dir, device)
    model.eval()

    batch = batch_cpu.to(device)  # (B, T)
    # Per-compartment activations at the chosen layer.
    feats = []
    for ci in range(c):
        x = batch + ci * BASE_VOCAB
        cid = torch.full_like(batch, ci)
        # capture_layer returns (None, None, hidden) before ln_f / lm_head
        out = model(x, compartment_ids=cid, capture_layer=capture_layer)
        if isinstance(out, tuple) and len(out) == 3:
            _, _, h = out
        else:
            raise RuntimeError(f"capture_layer didn't yield hidden: got {type(out)}")
        feats.append(h.flatten(0, 1).float().cpu())  # (B*T, D)
    # Pairwise CKA + per-row mean cosine sim, both c×c
    M_cka = np.zeros((c, c), dtype=np.float32)
    M_cos = np.zeros((c, c), dtype=np.float32)
    for i in range(c):
        for j in range(i, c):
            cka = linear_cka(feats[i], feats[j])
            cos = mean_cossim(feats[i], feats[j]) if i != j else 1.0
            M_cka[i, j] = cka; M_cka[j, i] = cka
            M_cos[i, j] = cos; M_cos[j, i] = cos
    if c < 2:
        mean_off_cka = float("nan")
        mean_off_cos = float("nan")
    else:
        off_cka = M_cka.copy(); np.fill_diagonal(off_cka, np.nan)
        off_cos = M_cos.copy(); np.fill_diagonal(off_cos, np.nan)
        mean_off_cka = float(np.nanmean(off_cka))
        mean_off_cos = float(np.nanmean(off_cos))

    del model
    torch.cuda.empty_cache()
    return {"c": c,
            "mean_off_diag": mean_off_cka, "full_matrix": M_cka.tolist(),
            "mean_off_diag_cossim": mean_off_cos, "full_matrix_cossim": M_cos.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--out_json", default="cossim_sweep.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only this many cells (for testing)")
    ap.add_argument("--mode", choices=["absolute", "compartment", "both"],
                    default="absolute")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    batch = load_canonical_batch(seed=0)
    print(f"Canonical batch: {batch.shape}, max_token={batch.max()}, vocab_capable=needed up to c*BASE_VOCAB")

    # Resume: load existing if any
    out_path = Path(args.out_json)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    print(f"Resume: {len(results)} cells already done")

    todo = []
    for g in GROUPS:
        for d in sorted((PROJECT_ROOT / "out" / "translation-compression" / g).iterdir()):
            if not (d / "meta" / "config.json").exists():
                continue
            cf = json.loads((d / "meta" / "config.json").read_text())
            mode = cf["experiment"]["translation_ratio_mode"]
            if args.mode != "both" and mode != args.mode:
                continue
            key = f"{g}/{d.name}"
            if key in results:
                continue
            todo.append((g, d, cf))

    print(f"To process: {len(todo)} cells")
    if args.limit is not None:
        todo = todo[: args.limit]
        print(f"Limited to: {len(todo)}")

    for i, (g, d, cf) in enumerate(todo):
        e, o = cf["experiment"], cf["optimizer"]
        c = e["n_compartments"]
        tr_raw = e["translation_ratio"]
        mode = e["translation_ratio_mode"]
        tr_eff = tr_raw / (c + 1) if mode == "compartment" else tr_raw
        wd = o["weight_decay"]
        key = f"{g}/{d.name}"
        print(f"[{i+1}/{len(todo)}] {key}: c={c} tr={tr_eff:.4f} ({mode} raw={tr_raw}) wd={wd}")
        try:
            res = cka_for_run(d, batch, args.device, capture_layer=args.layer)
            if res is None:
                print("  no checkpoint; skip")
                continue
        except Exception as exc:
            print(f"  ERROR: {exc!r}")
            continue
        results[key] = {
            "c": c, "tr_eff": round(tr_eff, 4), "tr_raw": tr_raw,
            "mode": mode, "wd": wd,
            "mean_off_diag_cka": res["mean_off_diag"],
            "full_matrix_cka": res["full_matrix"],
            "mean_off_diag_cossim": res["mean_off_diag_cossim"],
            "full_matrix_cossim": res["full_matrix_cossim"],
            "layer": args.layer,
        }
        out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path} ({len(results)} cells)")


if __name__ == "__main__":
    main()

"""Mean off-diagonal cosine similarity at layer 4 across training, for the
8-256 InfoNCE runs and matched no-InfoNCE c=N baselines.

For each (run, named-checkpoint) pair: forward a fixed canonical batch,
extract layer-4 hidden states for each compartment, compute c×c per-row
cosine sim, average off-diagonal. Cache to JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "..")
from eval_utils import load_eval_model_from_checkpoint
from compute_cossim_sweep import load_canonical_batch, mean_cossim, BASE_VOCAB, DEFAULT_LAYER
from _run_paths import INFONCE_8_256_BY_C, NO_INFONCE_8_256_BY_C, OUT_ROOT


OUT_BASE = OUT_ROOT
LAYER = 4

# (label, c, group/run_dir)
RUNS = [(f"infonce_n{c}", c, INFONCE_8_256_BY_C[c]) for c in (2, 4, 5, 6, 8)] + \
       [(f"baseline_n{c}", c, NO_INFONCE_8_256_BY_C[c]) for c in (2, 4, 5, 6, 8)]


def list_named_steps(run_dir: Path) -> list[int]:
    out = []
    for d in (run_dir / "checkpoints").iterdir():
        if d.name.startswith("step-") and (d / "model.pt").exists():
            try:
                out.append(int(d.name.split("-")[1]))
            except ValueError:
                pass
    return sorted(out)


# Canonical step grid; each run snaps to nearest available ckpt within tol.
CANONICAL_GRID = [
    100, 500, 2000, 10000, 50000,
    100000, 120000, 150000, 200000, 240000, 300000, 350000, 400000, 500000,
    700000, 1000000, 1250000, 1500000, 2000000, 2500000, 2950000,
]


def grid_subset(steps: list[int], tol: float = 0.05) -> list[int]:
    available = sorted(set(steps))
    if not available:
        return []
    arr = np.array(available)
    out = set()
    for g in CANONICAL_GRID:
        idx = int(np.argmin(np.abs(arr - g)))
        nearest = available[idx]
        if abs(nearest - g) / max(g, 1) <= tol:
            out.add(nearest)
    return sorted(out)


@torch.no_grad()
def cossim_for_ckpt(run_dir: Path, ckpt_dir: Path, batch_cpu: torch.Tensor,
                    c: int, device: str, layer: int) -> float:
    model, _, _ = load_eval_model_from_checkpoint(ckpt_dir, run_dir, device)
    model.eval()
    batch = batch_cpu.to(device)
    feats = []
    for ci in range(c):
        x = batch + ci * BASE_VOCAB
        cid = torch.full_like(batch, ci)
        out = model(x, compartment_ids=cid, capture_layer=layer)
        _, _, h = out
        feats.append(h.flatten(0, 1).float().cpu())
    if c < 2:
        result = float("nan")
    else:
        cos_off = []
        for i in range(c):
            for j in range(i + 1, c):
                cos_off.append(mean_cossim(feats[i], feats[j]))
        result = float(np.mean(cos_off))
    del model
    torch.cuda.empty_cache()
    return result


def main():
    device = "cuda"
    out_path = Path("cossim_across_training.json")
    cache = json.loads(out_path.read_text()) if out_path.exists() else {}

    batch_cpu = load_canonical_batch(seed=0)

    work = []
    for label, c, rel in RUNS:
        run_dir = OUT_BASE / rel
        if not run_dir.exists():
            print(f"  MISSING: {run_dir}")
            continue
        all_steps = list_named_steps(run_dir)
        sub = grid_subset(all_steps)
        for step in sub:
            key = f"{label}@{step}"
            if key in cache:
                continue
            work.append((label, c, run_dir, step, key))

    print(f"  {len(work)} ckpts to evaluate")
    for i, (label, c, run_dir, step, key) in enumerate(work):
        ckpt_dir = run_dir / "checkpoints" / f"step-{step:06d}"
        try:
            v = cossim_for_ckpt(run_dir, ckpt_dir, batch_cpu, c, device, LAYER)
        except Exception as e:
            print(f"  [{i+1}/{len(work)}] {key} FAIL: {e}")
            continue
        cache[key] = {"label": label, "c": c, "step": step, "cossim": v}
        if (i + 1) % 5 == 0 or i == len(work) - 1:
            out_path.write_text(json.dumps(cache, indent=2))
        print(f"  [{i+1}/{len(work)}] {key} cossim={v:.4f}")

    out_path.write_text(json.dumps(cache, indent=2))
    print(f"  wrote {out_path} ({len(cache)} entries)")


if __name__ == "__main__":
    main()

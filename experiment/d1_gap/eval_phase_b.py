"""Phase-B cross-eval: each Phase-B run's model on each cluster's val bin.

For 4 runs × 2 clusters = 8 val evaluations. For `comp` we evaluate each cluster
val through its NATIVE compartment (cluster i → compartment i).

Output:
  code_phase_b_matrix.json  { run_name: { "cluster_0": loss, "cluster_1": loss } }
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

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))
from eval_utils import load_eval_model_from_checkpoint  # noqa: E402

BASE_VOCAB = 16384


def load_val_tokens(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    arrs = []
    for f in files:
        with open(f, "rb") as fp:
            header = np.frombuffer(fp.read(256 * 4), dtype=np.int32)
            assert header[0] == 20251013, f"bad magic in {f}"
            ntok = int(header[2])
            arrs.append(np.frombuffer(fp.read(ntok * 4), dtype=np.uint32))
    return np.concatenate(arrs)


def latest_checkpoint(run_dir: Path) -> Path:
    ck = run_dir / "checkpoints"
    steps = []
    for d in ck.iterdir():
        if d.is_dir() and d.name.startswith("step-"):
            try:
                steps.append((int(d.name.split("-")[1]), d))
            except ValueError:
                pass
    steps.sort()
    return steps[-1][1]


@torch.no_grad()
def eval_loss(model, tokens: np.ndarray, block_size: int, batch_size: int,
              n_batches: int, device: str, token_offset: int = 0,
              compartment_id: int = 0) -> float:
    chunk = batch_size * (block_size + 1)
    total_chunks = len(tokens) // chunk
    n_batches = min(n_batches, total_chunks)
    total = 0.0
    n = 0
    for b in range(n_batches):
        sl = tokens[b * chunk : b * chunk + chunk]
        if len(sl) < chunk:
            break
        sl = sl.reshape(batch_size, block_size + 1).astype(np.int64) + token_offset
        x = torch.from_numpy(sl[:, :-1]).to(device)
        y = torch.from_numpy(sl[:, 1:]).to(device)
        cids = torch.full_like(x, compartment_id)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, compartment_ids=cids)
        total += float(loss.item()) * x.numel()
        n += x.numel()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--out-json", default="code_phase_b_matrix.json")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--tag", default="fineweb-phase-b")
    ap.add_argument("--runs-root", required=True,
                    help="Dir containing <tag>-cluster-*-only, -joint, -comp run dirs")
    ap.add_argument("--bins-root", required=True,
                    help="Dir containing cluster_*/val_*.bin files")
    args = ap.parse_args()

    RUNS_ROOT = Path(args.runs_root)
    BINS_ROOT = Path(args.bins_root)

    print(f"[val] loading K={args.K} cluster val bins from {BINS_ROOT}")
    val_tokens = {}
    for c in range(args.K):
        val_tokens[c] = load_val_tokens(str(BINS_ROOT / f"cluster_{c}" / "val_*.bin"))
        print(f"  cluster_{c}: {len(val_tokens[c]):,} tokens")

    RUN_NAMES = [f"{args.tag}-cluster-{i}-only" for i in range(args.K)] + \
                [f"{args.tag}-joint", f"{args.tag}-comp"]

    matrix = {}
    for name in RUN_NAMES:
        run_dir = RUNS_ROOT / name
        if not run_dir.exists():
            print(f"[skip] {name}: run dir missing")
            continue
        ck = latest_checkpoint(run_dir)
        step = int(ck.name.split("-")[1])
        print(f"\n=== {name} (step-{step}) ===")
        model, cfg, _ = load_eval_model_from_checkpoint(
            ck, run_dir, args.device, dtype=torch.bfloat16
        )
        block_size = cfg.model.block_size
        n_comp = cfg.experiment.n_compartments
        is_comp = (n_comp > 1)

        row = {"step": step, "n_compartments": n_comp}
        for target_cluster in range(args.K):
            # comp: use native cid for that cluster; else cid=0 offset=0
            cid = target_cluster if is_comp else 0
            offset = cid * BASE_VOCAB
            loss = eval_loss(
                model, val_tokens[target_cluster], block_size,
                args.batch_size, args.n_batches, args.device,
                token_offset=offset, compartment_id=cid,
            )
            row[f"cluster_{target_cluster}"] = loss
            print(f"  cluster_{target_cluster}: {loss:.4f}")
        matrix[name] = row

        del model
        torch.cuda.empty_cache()

    Path(args.out_json).write_text(json.dumps(matrix, indent=2))
    print(f"\nWrote {args.out_json}")

    # Summary: cluster-i-only ceiling vs joint vs comp on cluster i
    print("\n=== Phase B SUMMARY ===")
    ceilings = {c: matrix[f"{args.tag}-cluster-{c}-only"][f"cluster_{c}"] for c in range(args.K)}
    joint_by_c = {c: matrix[f"{args.tag}-joint"][f"cluster_{c}"] for c in range(args.K)}
    comp_by_c = {c: matrix[f"{args.tag}-comp"][f"cluster_{c}"] for c in range(args.K)}

    hdr = "  " + " ".join(f"{c:>8s}" for c in [""] + [f"c{i}" for i in range(args.K)])
    print(hdr)
    def _row(label, vals):
        print(f"  {label:<10s} " + " ".join(f"{vals[c]:>8.3f}" for c in range(args.K)))
    _row("ceiling", ceilings)
    _row("joint", joint_by_c)
    _row("comp", comp_by_c)
    print()
    _row("joint-gap", {c: joint_by_c[c] - ceilings[c] for c in range(args.K)})
    _row("comp-gap",  {c: comp_by_c[c]  - ceilings[c] for c in range(args.K)})


if __name__ == "__main__":
    main()

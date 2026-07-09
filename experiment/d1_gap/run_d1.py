"""Orchestrate the D1 experiment: for each K in --ks, train K semantic-cluster
LMs + K random-partition LMs, then cross-eval each LM on every cluster's val.

Parallelizes across GPUs by launching train_minilm.py subprocesses with pinned
CUDA_VISIBLE_DEVICES. Cross-eval runs in the same process at the end (loads all
saved model.pt files, does K^2 eval per K).

Output:
  <out-root>/lms/<partition>/k<K>/cluster_<c>/{model.pt, curve.json, config.json}
  <out-root>/d1_matrices.json    { K -> {"semantic": {K×K}, "random": {K×K}} }
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

# Reuse the trainer's MiniGPT for eval
sys.path.insert(0, str(Path(__file__).parent))
from train_minilm import MiniGPT, load_bin, eval_loss  # noqa: E402


def train_one(gpu: int, train_bin: str, val_bin: str, out_dir: str,
              n_iter: int, batch_size: int, block_size: int, n_layer: int,
              n_head: int, n_embd: int, vocab_size: int, lr: float,
              seed: int, python_bin: str, script: str) -> tuple[str, int, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        python_bin, script,
        "--train-bin", train_bin, "--val-bin", val_bin, "--out-dir", out_dir,
        "--n-iter", str(n_iter), "--batch-size", str(batch_size),
        "--block-size", str(block_size), "--n-layer", str(n_layer),
        "--n-head", str(n_head), "--n-embd", str(n_embd),
        "--vocab-size", str(vocab_size), "--lr", str(lr), "--seed", str(seed),
    ]
    rc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return (out_dir, rc.returncode, rc.stderr[-2000:])


def cross_eval_k(K: int, partition: str, lms_root: Path, bins_root: Path,
                 device: str, batch_size: int, block_size: int,
                 n_batches: int, n_layer: int, n_head: int, n_embd: int,
                 vocab_size: int) -> list[list[float]]:
    """Return K×K matrix of loss[i, j] = LM trained on cluster i eval on cluster j's val."""
    mat = [[float("nan")] * K for _ in range(K)]
    for i in range(K):
        model_path = lms_root / partition / f"k{K}" / f"cluster_{i}" / "model.pt"
        if not model_path.exists():
            print(f"    skip: {model_path} missing")
            continue
        model = MiniGPT(vocab_size, n_layer, n_head, n_embd, block_size).to(device)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        model.eval()
        for j in range(K):
            val_pat = str(bins_root / partition / f"k{K}" / f"cluster_{j}" / "val_*.bin")
            val_toks = load_bin(val_pat)
            loss = eval_loss(model, val_toks, block_size, batch_size, n_batches, device)
            mat[i][j] = loss
        del model
        torch.cuda.empty_cache()
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True,
                    help="Same root as prepare_bins.py --out-root")
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--n-iter", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=1)
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--python-bin",
                    default="/mnt/pccfs2/backed_up/vin/dev/translation-compression/.venv/bin/python")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training, go straight to cross-eval (assumes models exist)")
    ap.add_argument("--run-name", default="8-32",
                    help="Subdir under lms/ and suffix for matrices json — lets multiple model sizes share bins")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    out_root = Path(args.out_root)
    bins_root = out_root / "bins"
    lms_root = out_root / "lms" / args.run_name

    script = str(Path(__file__).parent / "train_minilm.py")

    if not args.skip_train:
        # Build (partition, K, cluster_i) task list
        tasks = []
        for K in ks:
            for partition in ["semantic", "random"]:
                for c in range(K):
                    train_bin = str(bins_root / partition / f"k{K}" / f"cluster_{c}" / "train_*.bin")
                    val_bin = str(bins_root / partition / f"k{K}" / f"cluster_{c}" / "val_*.bin")
                    out_dir = str(lms_root / partition / f"k{K}" / f"cluster_{c}")
                    tasks.append((partition, K, c, train_bin, val_bin, out_dir))

        print(f"[run] {len(tasks)} LMs to train, {len(gpus)} GPUs")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=len(gpus)) as ex:
            futures = {}
            gpu_pool = list(gpus)  # rotate GPUs
            for i, (partition, K, c, tb, vb, od) in enumerate(tasks):
                gpu = gpu_pool[i % len(gpu_pool)]
                fut = ex.submit(
                    train_one, gpu, tb, vb, od, args.n_iter, args.batch_size,
                    args.block_size, args.n_layer, args.n_head, args.n_embd,
                    args.vocab_size, args.lr, args.seed, args.python_bin, script,
                )
                futures[fut] = (partition, K, c, gpu)
            for fut in as_completed(futures):
                partition, K, c, gpu = futures[fut]
                out_dir, rc, stderr = fut.result()
                marker = "OK" if rc == 0 else "FAIL"
                print(f"  [{marker}] {partition} k={K} c={c} gpu={gpu} out={out_dir}")
                if rc != 0:
                    print(f"    stderr tail:\n{stderr}")
        print(f"[run] training done in {time.time()-t0:.1f}s")

    # Cross-eval on GPU 0 sequentially
    device = f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu"
    # torch will see whichever CUDA_VISIBLE_DEVICES is set — but subprocesses used
    # local device indices. For eval here, we don't override so we're on our full
    # local topology.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"\n[cross_eval] on {device}")
    matrices = {}
    for K in ks:
        matrices[K] = {}
        for partition in ["semantic", "random"]:
            print(f"  K={K} {partition}...")
            mat = cross_eval_k(K, partition, lms_root, bins_root, device,
                               args.batch_size, args.block_size, args.eval_batches,
                               args.n_layer, args.n_head, args.n_embd, args.vocab_size)
            matrices[K][partition] = mat

    out_path = out_root / f"d1_matrices_{args.run_name}.json"
    out_path.write_text(json.dumps(matrices, indent=2))
    print(f"[run] wrote {out_path}")

    # Quick summary
    print("\n=== SUMMARY ===")
    for K in ks:
        for partition in ["semantic", "random"]:
            mat = np.array(matrices[K][partition])
            diag = float(np.diag(mat).mean())
            off = float((mat.sum() - np.trace(mat)) / (K * K - K))
            print(f"  K={K:>2d} {partition:8s}: diag_mean={diag:.3f}  off_diag_mean={off:.3f}  "
                  f"gap={off - diag:+.3f}")


if __name__ == "__main__":
    main()

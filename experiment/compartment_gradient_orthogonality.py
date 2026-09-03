"""Compute gradient orthogonality between compartments for a trained checkpoint.

For a fixed batch of validation tokens, run the same forward pass under
compartment 0 and compartment 1 (only the compartment_ids differ — the input
token ids are identical). Compute the gradient of cross-entropy loss with respect
to the model's *shared trunk* parameters in each case and report the cosine
similarity (and other diagnostics) between the two flattened gradients.

A cosine similarity near 1 means the two compartments push the trunk in the same
direction (no specialization). Near 0 means orthogonal (specialized). Negative
means actively conflicting.

We exclude `comp_emb` from the comparison because compartment 0 vs 1 trivially
touch disjoint rows of comp_emb (so its grads are orthogonal by construction).

Usage:
    python3 compartment_gradient_orthogonality.py \
        --checkpoint <CKPT_DIR> \
        --experiment <EXPERIMENT_DIR> \
        --val-bin <PATH_TO_VAL_BIN> \
        --batches 8 --batch-size 4 --pairs 0,1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make project src importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from eval_utils import (  # type: ignore
    Assignment,
    SingleShardAssignedValLoader,
    get_base_vocab_size,
    load_eval_model_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoints/step-XXXXXX directory")
    p.add_argument("--experiment", required=True, help="Path to the run dir (containing meta/config.json)")
    p.add_argument("--val-bin", default=None,
                   help="Path to a val .bin shard. If omitted, uses config.data.val_bin (first match).")
    p.add_argument("--batches", type=int, default=8, help="Number of batches to accumulate gradients over")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=None,
                   help="Sequence length T (defaults to config.model.block_size)")
    p.add_argument("--pairs", default="0,1",
                   help="Comma-separated list of pairs 'a:b' (or single ints to pair with 0). "
                        "E.g. '0,1' means compare comp 0 vs comp 1.")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32",
                   help="Compute dtype (float32 recommended for stable cosines)")
    p.add_argument("--out", default=None, help="Optional JSON output file")
    return p.parse_args()


def parse_pairs(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    items = [x.strip() for x in s.split(",") if x.strip()]
    if len(items) == 1 and ":" not in items[0]:
        return [(0, int(items[0]))]
    if all(":" not in it for it in items):
        # Treat as a single list of compartment IDs to pair pairwise with first element
        ids = [int(x) for x in items]
        return [(ids[0], i) for i in ids[1:]]
    for it in items:
        a, b = it.split(":")
        out.append((int(a), int(b)))
    return out


def shared_param_iter(model: torch.nn.Module):
    """Yield (name, param) pairs we'll diff over: trainable params except `comp_emb`.

    `comp_emb` is excluded because grads on disjoint rows are orthogonal trivially.
    """
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "comp_emb" in name:
            continue
        yield name, p


def compute_grads(model, loader: SingleShardAssignedValLoader, num_batches: int, device, dtype) -> dict[str, torch.Tensor]:
    """Forward + backward over `num_batches` and return summed grads keyed by param name."""
    # Zero existing grads
    for _, p in shared_param_iter(model):
        if p.grad is not None:
            p.grad = None

    # Accumulate
    loader.reset()
    total_loss = 0.0
    seen = 0
    for i in range(num_batches):
        try:
            x, y, cids = loader.next_batch()
        except StopIteration:
            break
        x = x.to(device)
        y = y.to(device)
        cids = cids.to(device)
        # Forward in fp32 for stable gradient comparison (avoids bf16 noise)
        with torch.amp.autocast("cuda", enabled=False):
            logits, _ = model(x.long(), y.long(), compartment_ids=cids.long())
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.long().reshape(-1),
                ignore_index=-1,
            )
        loss.backward()
        total_loss += float(loss.detach().item())
        seen += 1

    # Snapshot grads
    snap: dict[str, torch.Tensor] = {}
    for name, p in shared_param_iter(model):
        if p.grad is None:
            continue
        # Average over batches and move to fp32 cpu for stability
        snap[name] = (p.grad.detach().to(dtype=torch.float32) / max(seen, 1)).clone()
        p.grad = None  # free for next pass

    snap["__loss__"] = torch.tensor(total_loss / max(seen, 1))  # type: ignore
    snap["__seen__"] = torch.tensor(seen)  # type: ignore
    return snap


def cosine_and_norms(g_a: dict[str, torch.Tensor], g_b: dict[str, torch.Tensor]) -> dict:
    """Compute global cosine similarity and per-tensor breakdown."""
    keys = [k for k in g_a if not k.startswith("__") and k in g_b]
    # Global flattened cosine
    flat_a = torch.cat([g_a[k].flatten() for k in keys])
    flat_b = torch.cat([g_b[k].flatten() for k in keys])
    na = flat_a.norm().item()
    nb = flat_b.norm().item()
    cos_global = float((flat_a @ flat_b).item() / max(na * nb, 1e-30))

    # Per-layer breakdown (top-K and bottom-K by cosine)
    per_tensor = []
    for k in keys:
        a = g_a[k].flatten()
        b = g_b[k].flatten()
        an = a.norm().item()
        bn = b.norm().item()
        if an < 1e-30 or bn < 1e-30:
            cs = float("nan")
        else:
            cs = float((a @ b).item() / (an * bn))
        per_tensor.append({"name": k, "cosine": cs, "norm_a": an, "norm_b": bn,
                            "numel": int(a.numel())})
    per_tensor.sort(key=lambda d: (d["cosine"] if d["cosine"] == d["cosine"] else -2.0))
    return {
        "cosine_global": cos_global,
        "norm_a_global": na,
        "norm_b_global": nb,
        "loss_a": float(g_a["__loss__"].item()),
        "loss_b": float(g_b["__loss__"].item()),
        "per_tensor": per_tensor,
    }


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    ckpt_dir = Path(args.checkpoint)
    exp_dir = Path(args.experiment)

    print(f"Loading model from {ckpt_dir}")
    model, config, n_comp = load_eval_model_from_checkpoint(ckpt_dir, exp_dir, device, dtype=dtype)
    model.train()  # ensure dropout off but grads enabled
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"Model loaded: n_compartments={n_comp}")

    base_vocab_size = get_base_vocab_size(config)
    block_size = args.block_size or config.model.block_size

    # Resolve val data path
    if args.val_bin:
        val_path = args.val_bin
    else:
        val_pattern = config.data.val_bin
        if not val_pattern:
            raise SystemExit("config has no val_bin and --val-bin not set")
        # Resolve relative to TC_STORAGE_ROOT (or cwd's ../)
        import os
        storage_root = Path(os.environ.get("TC_STORAGE_ROOT", "../"))
        matches = sorted(storage_root.glob(val_pattern))
        if not matches:
            raise SystemExit(f"No val files match {val_pattern} under {storage_root}")
        val_path = str(matches[0])
    print(f"Using val shard: {val_path}")

    # Compute grads for each compartment we're going to compare
    pairs = parse_pairs(args.pairs)
    comps_needed = sorted({c for p in pairs for c in p})
    print(f"Will compute grads for compartments: {comps_needed}")

    grads_by_comp: dict[int, dict[str, torch.Tensor]] = {}
    for c in comps_needed:
        loader = SingleShardAssignedValLoader(
            shard_path=val_path,
            B=args.batch_size,
            T=block_size,
            base_vocab_size=base_vocab_size,
            max_compartments=int(config.experiment.max_compartments),
            assignment=Assignment(kind=0, src=c),
            device=device,
        )
        print(f"  compartment {c}: computing grads over {args.batches} batches of {args.batch_size}x{block_size}...")
        grads_by_comp[c] = compute_grads(model, loader, args.batches, device, dtype)
        print(f"    avg loss: {grads_by_comp[c]['__loss__'].item():.4f}")

    # Compute cosine for each pair
    results = {
        "checkpoint": str(ckpt_dir),
        "experiment": str(exp_dir),
        "val_bin": val_path,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "block_size": block_size,
        "pairs": [],
    }
    print("\n=== Compartment gradient orthogonality ===")
    for a, b in pairs:
        r = cosine_and_norms(grads_by_comp[a], grads_by_comp[b])
        print(f"  comp {a} vs comp {b}:")
        print(f"    cosine_global = {r['cosine_global']:+.4f}")
        print(f"    grad norms     a={r['norm_a_global']:.3e}  b={r['norm_b_global']:.3e}")
        print(f"    loss          a={r['loss_a']:.4f}  b={r['loss_b']:.4f}")
        # Show 5 most similar and 5 most orthogonal layers
        pt = r["per_tensor"]
        print(f"    most orthogonal layers (lowest cosine):")
        for d in pt[:5]:
            print(f"      {d['cosine']:+.4f}  {d['name']}")
        print(f"    most aligned layers (highest cosine):")
        for d in pt[-5:][::-1]:
            print(f"      {d['cosine']:+.4f}  {d['name']}")
        results["pairs"].append({"a": a, "b": b, **{k: v for k, v in r.items() if k != "per_tensor"},
                                  "per_tensor": r["per_tensor"]})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

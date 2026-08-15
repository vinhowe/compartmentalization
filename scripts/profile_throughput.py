#!/usr/bin/env python3
"""Measure training throughput for the redesign, one configuration at a time.

Standalone on purpose: it imports only `src.model` and `src.config.job_config`,
so it can be staged in a scratch directory on a cluster without touching a
checkout that has uncommitted work in it.

Synthetic token ids, not real shards. Throughput is data-independent at this
level — the dataloader is a memmap read that overlaps with compute — and using
random ids means no corpus has to be present to get a kernel-level picture.

    # one config
    python3 profile_throughput.py --compartments 8 --batch-size 8

    # the number that decides whether fused cross-entropy is load-bearing
    python3 profile_throughput.py --compartments 8 --find-max-batch

    # 8-GPU scaling
    torchrun --nproc_per_node=8 profile_throughput.py --compartments 8 --batch-size 8

Reported MFU comes from GPT.estimate_mfu, which resolves peak FLOPS from the
live device. Before that fix it divided by a hardcoded A100 312 TFLOPS on every
device, so B200 numbers read ~7x high.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config.job_config import Model
from src.model import GPT

BASE_VOCAB = 16384


def build(args, device):
    # Composite vocabulary is c*V + 1: one shared translation token on top of
    # the per-compartment blocks. The head is what scales with c — at c=8 it is
    # ~20% of FLOPs versus ~3% at c=1 — which is the whole reason to profile
    # both ends rather than one.
    vocab = args.compartments * BASE_VOCAB + 1
    cfg = Model(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_embd // args.d_head,
        block_size=args.block_size,
        vocab_size=vocab,
        # Must be set explicitly: the field defaults to None on the dataclass, and
        # GPT.__init__ does getattr(config, "embedding_vocab_size", vocab_size),
        # which finds the attribute and gets None rather than the fallback.
        # train.py always sets it, so this only bites standalone callers.
        embedding_vocab_size=vocab,
        dropout=0.0,
        bias=False,
        weight_tying=args.weight_tying,
        use_rope=True,
    )
    model = GPT(cfg).to(device)
    if args.compile_mode != "none":
        kw = {} if args.compile_mode == "default" else {"mode": args.compile_mode}
        model = torch.compile(model, **kw)
    return model, cfg, vocab


def _fused_ce_loss(model, raw, x, y, args):
    """Loss via Liger's fused linear+cross-entropy, never materializing logits.

    Passing targets=None makes forward take its inference shortcut and project
    only the final position (B x 1 x V, negligible), while return_last_hidden
    hands back the full (B, T, d) hidden states. The vocab-width projection then
    happens *inside* the fused kernel, in chunks, so the B x T x V logits tensor
    that caps the batch at c=8 is never allocated.
    """
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction as FLCE,
    )
    out = model(x, targets=None, return_last_hidden=True)
    h = out[2]
    d = h.size(-1)
    out = FLCE.apply(
        h.reshape(-1, d), raw.lm_head.weight, y.reshape(-1),
        None,      # bias
        None,      # ce_weight
        -1,        # ignore_index — matches F.cross_entropy(ignore_index=-1)
    )
    # the op returns (loss, z_loss) when z-loss is requested and a bare loss
    # otherwise, depending on version; normalise.
    return out[0] if isinstance(out, tuple) else out


def run_steps(model, optimizer, args, vocab, device, n_steps, timed=False, raw=None):
    """Run n_steps full training steps; return seconds/step if timed."""
    B, T = args.batch_size, args.block_size
    x = torch.randint(0, vocab, (B, T), device=device, dtype=torch.long)
    y = torch.randint(0, vocab, (B, T), device=device, dtype=torch.long)
    dt_ = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=dt_)

    if timed:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    for _ in range(n_steps):
        for micro in range(args.grad_accum):
            with ctx:
                if args.fused_ce:
                    loss = _fused_ce_loss(model, raw, x, y, args)
                else:
                    _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if timed:
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_steps
    return None


def profile_one(args, device, world_size):
    model, cfg, vocab = build(args, device)
    raw = getattr(model, "_orig_mod", model)
    optimizer = raw.configure_optimizers(0.0, 1e-4, (0.9, 0.95), "cuda")

    torch.cuda.reset_peak_memory_stats()
    run_steps(model, optimizer, args, vocab, device, args.warmup, raw=raw)          # compile + autotune
    dt = run_steps(model, optimizer, args, vocab, device, args.steps, timed=True, raw=raw)

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum * world_size
    # estimate_mfu takes fwd/bwd passes per iteration and the per-iteration time
    mfu = raw.estimate_mfu(args.batch_size * args.grad_accum, dt)
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {
        "compartments": args.compartments,
        "vocab": vocab,
        "params_M": round(raw.get_num_params() / 1e6, 1),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "block_size": args.block_size,
        "world_size": world_size,
        "compile_mode": args.compile_mode,
        "dtype": args.dtype,
        "fused_ce": args.fused_ce,
        "sec_per_step": round(dt, 4),
        "tokens_per_step": tokens_per_step,
        "tokens_per_sec": round(tokens_per_step / dt),
        "mfu": round(mfu, 4),
        "peak_mem_GB": round(peak, 2),
    }


def find_max_batch(args, device, world_size):
    """Largest power-of-two batch that survives a full step, then refine.

    This is the number that decides whether fused linear+CE is load-bearing: at
    c=8, T=1024, V=131073 the materialized logits are ~1 GB/sample, so if the
    ceiling here is low it is the head, not the trunk, that caps the batch.
    """
    def fits(b):
        """One full training step at batch b. True if it survives."""
        args.batch_size = b
        model = raw = opt = None
        try:
            model, _, vocab = build(args, device)
            raw = getattr(model, "_orig_mod", model)
            opt = raw.configure_optimizers(0.0, 1e-4, (0.9, 0.95), "cuda")
            torch.cuda.reset_peak_memory_stats()
            run_steps(model, opt, args, vocab, device, 1, raw=raw)
            print(f"  batch {b:5d}  OK   peak "
                  f"{torch.cuda.max_memory_allocated()/1e9:6.2f} GB", flush=True)
            return True
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {b:5d}  OOM", flush=True)
            return False
        finally:
            # Drop every reference before empty_cache, or the allocator keeps
            # the arena and the next probe OOMs for the wrong reason.
            del model, raw, opt
            torch.cuda.empty_cache()

    lo, hi, best = 0, None, None
    b = 1
    while b <= 4096:
        if fits(b):
            lo = best = b
            b *= 2
        else:
            hi = b
            break
    if hi is not None:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits(mid):
                lo = best = mid
            else:
                hi = mid
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-layer", type=int, default=24)
    p.add_argument("--n-embd", type=int, default=1024)
    p.add_argument("--d-head", type=int, default=64, help="n_head = n_embd // d_head")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--compartments", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--weight-tying", action="store_true", default=False)
    p.add_argument("--compile-mode", choices=["none", "default", "max-autotune"], default="default")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps (pays compile/autotune)")
    p.add_argument("--steps", type=int, default=20, help="timed steps")
    p.add_argument("--find-max-batch", action="store_true")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                   help="fp16 for pre-Ampere (V100 has no bf16 tensor cores)")
    p.add_argument("--fused-ce", action="store_true",
                   help="Liger fused linear+CE; never materializes the B x T x V logits")
    p.add_argument("--ddp-flags", action="store_true",
                   help="gradient_as_bucket_view + static_graph")
    p.add_argument("--json", type=str, default=None, help="append the result record here")
    args = p.parse_args()

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda")
        world_size = 1

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    is_master = not ddp or int(os.environ["RANK"]) == 0

    if is_master:
        print(f"device: {torch.cuda.get_device_name()}  world_size={world_size}")

    if args.find_max_batch:
        if ddp:
            raise SystemExit("--find-max-batch is single-process only")
        print(f"searching max batch at c={args.compartments}, T={args.block_size}, "
              f"compile={args.compile_mode}")
        best = find_max_batch(args, device, world_size)
        print(f"\nmax batch that fits: {best}")
        return 0

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model, cfg, vocab = build(args, device)
        kw = dict(gradient_as_bucket_view=True, static_graph=True) if args.ddp_flags else {}
        model = DDP(model, device_ids=[device.index], **kw)
        raw = model.module
        raw = getattr(raw, "_orig_mod", raw)
        opt = raw.configure_optimizers(0.0, 1e-4, (0.9, 0.95), "cuda")
        torch.cuda.reset_peak_memory_stats()
        run_steps(model, opt, args, vocab, device, args.warmup, raw=raw)
        dt = run_steps(model, opt, args, vocab, device, args.steps, timed=True, raw=raw)
        tps = args.batch_size * args.block_size * args.grad_accum * world_size
        rec = {
            "compartments": args.compartments, "vocab": vocab,
            "params_M": round(raw.get_num_params() / 1e6, 1),
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "block_size": args.block_size, "world_size": world_size,
            "compile_mode": args.compile_mode, "ddp_flags": args.ddp_flags,
            "sec_per_step": round(dt, 4), "tokens_per_step": tps,
            "tokens_per_sec": round(tps / dt),
            "mfu": round(raw.estimate_mfu(args.batch_size * args.grad_accum, dt), 4),
            "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
    else:
        rec = profile_one(args, device, world_size)

    if is_master:
        print(json.dumps(rec, indent=2))
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

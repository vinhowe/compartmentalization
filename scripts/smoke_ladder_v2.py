#!/usr/bin/env python3
"""Measure peak VRAM for each ladder-v2 rung/arm before committing GPU time.

The micro-batch table in gen_ladder_v2_configs.py is the one thing in the
protocol that is estimated rather than derived, and it is estimated against a
hostile number: at the wide vocab (V=131073, T=1024) a SINGLE sequence costs
~268MB of bf16 logits, ~537MB for the fp32 cross-entropy upcast and ~268MB of
gradient before any trunk activation exists. Getting it wrong means an OOM
hours into a 30B-token run.

This does a real forward+backward at the configured micro-batch and reports
peak allocated memory, so the table is measured rather than hoped for.

Usage:
    .venv/bin/python scripts/smoke_ladder_v2.py                 # every rung/arm
    .venv/bin/python scripts/smoke_ladder_v2.py --only r1-c8    # one
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config.manager import ConfigManager  # noqa: E402
from src.config.job_config import Model as GPTConfig  # noqa: E402
from src.model import GPT  # noqa: E402

BUDGET_GB = 80.0     # A100-80GB
HEADROOM = 0.85      # leave room for fragmentation, DDP buckets, optimizer


def build(cfg):
    m, e = cfg.model, cfg.experiment
    composite = m.vocab_size * e.n_compartments + 1
    gptconf = GPTConfig(
        n_layer=m.n_layer,
        n_head=m.n_head,
        n_embd=m.n_embd,
        block_size=m.block_size,
        vocab_size=composite,
        embedding_vocab_size=composite,
        bias=False,
        dropout=0.0,
        weight_tying=False,
        use_rope=True,
        rope_base=10000.0,
        use_compartment_embeddings=bool(e.use_compartment_embeddings),
        max_compartments=e.n_compartments,
        base_vocab_size=m.vocab_size,
    )
    return GPT(gptconf), composite


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--config-dir", default="config/ladder-v2")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA available")
        return 1

    # one config per (rung, arm) -- the sweep arms share geometry with the ladder
    files = sorted(
        p for p in pathlib.Path(args.config_dir).glob("ladder-v2-r*.toml")
        if "lrsweep" not in p.name
    )
    if args.only:
        files = [p for p in files if args.only in p.name]

    rows = []
    worst_ok = True
    for p in files:
        cfg = ConfigManager().load_from_toml_file(str(p))
        micro = cfg.training.batch_size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model, composite = build(cfg)
            model = model.cuda().to(torch.bfloat16)
            x = torch.randint(0, composite - 1, (micro, cfg.model.block_size),
                              device="cuda")
            y = torch.randint(0, composite - 1, (micro, cfg.model.block_size),
                              device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            # optimizer state is 2 fp32 moments over all params, added at step 1
            n = sum(q.numel() for q in model.parameters())
            adam = 2 * n * 4 / 1024**3
            total = peak + adam
            ok = total < BUDGET_GB * HEADROOM
            worst_ok &= ok
            rows.append((p.stem, micro, composite, f"{peak:.1f}", f"{adam:.1f}",
                         f"{total:.1f}", "OK" if ok else "TOO BIG"))
            del model, x, y, loss
        except torch.cuda.OutOfMemoryError:
            rows.append((p.stem, micro, "-", "OOM", "-", "-", "OOM"))
            worst_ok = False
        torch.cuda.empty_cache()

    hdr = ("config", "micro", "vocab", "fwdbwd GB", "adam GB", "total GB", "")
    w = [max(len(str(r[i])) for r in ([hdr] + rows)) for i in range(len(hdr))]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(len(hdr))))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print()
    print(f"budget {BUDGET_GB:.0f}GB x {HEADROOM:.0%} headroom "
          f"= {BUDGET_GB*HEADROOM:.1f}GB per GPU")
    return 0 if worst_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

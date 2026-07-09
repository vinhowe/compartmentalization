"""Train a single tiny GPT-2-style LM on one (train_bin, val_bin) pair.

Handrolled minimal transformer so we don't pay the paper-model import overhead
(compartment framework, LR schedulers, wandb, etc). ~100 lines.

Records train + val loss curve, saves final model state + a stats JSON. Meant
to be called many times in parallel by run_d1.py.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_minilm.py \
      --train-bin path/train_*.bin --val-bin path/val_*.bin \
      --out-dir /some/where --n-iter 500
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MAGIC = 20251013


def load_bin(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    arrs = []
    for f in files:
        with open(f, "rb") as fp:
            header = np.frombuffer(fp.read(256 * 4), dtype=np.int32)
            assert header[0] == MAGIC, f"bad magic in {f}"
            ntok = int(header[2])
            arrs.append(np.frombuffer(fp.read(ntok * 4), dtype=np.uint32))
    return np.concatenate(arrs)


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_head: int, n_embd: int,
                 block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _Block(n_head, n_embd, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, target=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    target.reshape(-1), ignore_index=-1)
        return logits, loss


class _Block(nn.Module):
    def __init__(self, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = _CausalSelfAttention(n_head, n_embd, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


def get_batch(tokens: np.ndarray, batch_size: int, block_size: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng()  # per-call, fine for stochastic sampling
    idx = rng.integers(0, len(tokens) - block_size - 1, size=batch_size)
    x = np.stack([tokens[i:i + block_size] for i in idx]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + block_size + 1] for i in idx]).astype(np.int64)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


@torch.no_grad()
def eval_loss(model, tokens: np.ndarray, block_size: int, batch_size: int,
              n_batches: int, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    chunk = batch_size * (block_size + 1)
    max_start = len(tokens) - chunk
    step = chunk
    for i in range(n_batches):
        start = (i * step) % max(max_start, 1)
        sl = tokens[start:start + chunk]
        if len(sl) < chunk:
            break
        sl = sl.reshape(batch_size, block_size + 1).astype(np.int64)
        x = torch.from_numpy(sl[:, :-1]).to(device)
        y = torch.from_numpy(sl[:, 1:]).to(device)
        _, loss = model(x, y)
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    # 8-32 tier defaults
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=1)
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--vocab-size", type=int, default=16384)
    # training
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-iter", type=int, default=500)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-iters", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[train] loading {args.train_bin}")
    train_toks = load_bin(args.train_bin)
    print(f"[train] loading {args.val_bin}")
    val_toks = load_bin(args.val_bin)
    print(f"[train] train tokens={len(train_toks):,} val tokens={len(val_toks):,}")

    if len(train_toks) < args.block_size + 1 or len(val_toks) < args.block_size + 1:
        raise SystemExit(f"insufficient tokens for block_size={args.block_size}")

    model = MiniGPT(args.vocab_size, args.n_layer, args.n_head, args.n_embd,
                    args.block_size).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] MiniGPT n_layer={args.n_layer} n_head={args.n_head} "
          f"n_embd={args.n_embd} n_params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.99))

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    def get_lr(it):
        if it < args.warmup_iters:
            return args.lr * (it + 1) / (args.warmup_iters + 1)
        return args.lr  # constant after warmup (Phase-A doesn't need cosine)

    curve = []  # list of {iter, train_loss, val_loss (or None)}
    t0 = time.time()
    for it in range(args.n_iter):
        # LR update
        cur_lr = get_lr(it)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        x, y = get_batch(train_toks, args.batch_size, args.block_size, args.device)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        row = {"iter": it, "train_loss": float(loss.item()), "lr": cur_lr}
        if (it + 1) % args.eval_interval == 0 or it == args.n_iter - 1:
            row["val_loss"] = eval_loss(model, val_toks, args.block_size,
                                        args.batch_size, args.eval_batches, args.device)
        curve.append(row)

    dt = time.time() - t0
    print(f"[train] done: {args.n_iter} iters in {dt:.1f}s")

    # Save final model + curve + config
    torch.save(model.state_dict(), out / "model.pt")
    (out / "curve.json").write_text(json.dumps(curve))
    (out / "config.json").write_text(json.dumps({
        "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
        "block_size": args.block_size, "vocab_size": args.vocab_size,
        "n_params": n_params, "n_iter": args.n_iter, "batch_size": args.batch_size,
        "lr": args.lr, "train_bin": args.train_bin, "val_bin": args.val_bin,
        "final_train_loss": curve[-1]["train_loss"],
        "final_val_loss": curve[-1].get("val_loss"),
        "wall_time_sec": dt,
    }, indent=2))
    print(f"[train] wrote {out}/model.pt, curve.json, config.json")


if __name__ == "__main__":
    main()

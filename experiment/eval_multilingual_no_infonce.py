"""Per-language val loss curves over training steps for the 4 multilingual models.

For each run, walks all saved checkpoints (named step-NNNNNN + rolling) and
evaluates on EN-only and ZH-only val. Outputs two PNGs:
  experiment/multilingual_val_curves_en_no_infonce.png  — EN val loss vs step, all 4 runs
  experiment/multilingual_val_curves_zh_no_infonce.png  — ZH val loss vs step, all 4 runs
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from eval_utils import load_eval_model_from_checkpoint


PROJECT_ROOT = Path("..")
RUNS_ROOT = PROJECT_ROOT / "out" / "translation-compression" / "multilingual-wiki-qwen3"
EN_VAL_PATTERN = "../data/wiki-en-qwen3/wiki_val_*.bin"
ZH_VAL_PATTERN = "../data/wiki-zh-qwen3/wiki_val_*.bin"
QWEN3_VOCAB = 151_936


def load_val_tokens(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    arrs = []
    for fname in files:
        with open(fname, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            ntok = int(header[2])
            tokens = np.frombuffer(f.read(ntok * 4), dtype=np.uint32)
            arrs.append(tokens)
    return np.concatenate(arrs)


def find_latest_run(substr: str) -> Path:
    candidates = sorted(d for d in RUNS_ROOT.iterdir() if d.is_dir() and substr in d.name)
    if not candidates:
        raise FileNotFoundError(substr)
    return candidates[-1]


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return list of (step, ckpt_dir) for named + rolling checkpoints."""
    out = []
    ck_root = run_dir / "checkpoints"
    if not ck_root.exists():
        return out
    for d in ck_root.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("step-"):
            try:
                step = int(d.name.split("-")[1])
                out.append((step, d))
            except ValueError:
                pass
        elif d.name == "_rolling":
            ts_path = d / "trainer_state.json"
            if ts_path.exists():
                try:
                    state = json.loads(ts_path.read_text())
                    step = int(state.get("iter_num", 0))
                    out.append((step, d))
                except (json.JSONDecodeError, KeyError):
                    pass
    out.sort()
    # If rolling has the same step as a named ckpt, prefer named (more reliable)
    seen = set()
    deduped = []
    for step, d in out:
        if step in seen:
            continue
        seen.add(step)
        deduped.append((step, d))
    return deduped


@torch.no_grad()
def eval_loss(model, tokens, block_size, batch_size, n_batches, device, token_offset=0):
    chunk = batch_size * (block_size + 1)
    total_chunks = len(tokens) // chunk
    n_batches = min(n_batches, total_chunks)
    total_loss = 0.0
    total_tokens = 0
    for b in range(n_batches):
        sl = tokens[b * chunk : b * chunk + chunk]
        if len(sl) < chunk:
            break
        sl = sl.reshape(batch_size, block_size + 1).astype(np.int64) + token_offset
        x = torch.from_numpy(sl[:, :-1]).to(device)
        y = torch.from_numpy(sl[:, 1:]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        n = x.numel()
        total_loss += float(loss.item()) * n
        total_tokens += n
    return total_loss / max(1, total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=200)
    args = ap.parse_args()

    print("Loading val tokens...")
    en_val = load_val_tokens(EN_VAL_PATTERN)
    zh_val = load_val_tokens(ZH_VAL_PATTERN)
    print(f"  EN: {len(en_val):,} tokens   ZH: {len(zh_val):,} tokens")

    runs = [
        ("shared", "multilingual-shared-12-256-qwen3__", "tab:blue", "o"),
        ("compartmented", "multilingual-compartmented-12-256-qwen3__", "tab:orange", "s"),
        ("en-only", "multilingual-en-only-12-256-qwen3", "tab:green", "^"),
        ("zh-only", "multilingual-zh-only-12-256-qwen3", "tab:red", "v"),
    ]

    results: dict[str, list[tuple[int, float, float]]] = {}
    for label, substr, color, marker in runs:
        print(f"\n=== {label} ===")
        run_dir = find_latest_run(substr)
        checkpoints = list_checkpoints(run_dir)
        print(f"  {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")
        rows = []
        for step, ckpt_dir in checkpoints:
            try:
                model, cfg, _ = load_eval_model_from_checkpoint(
                    ckpt_dir, run_dir, args.device, dtype=torch.bfloat16
                )
            except Exception as e:
                print(f"    step {step}: load failed: {e}")
                continue
            block_size = cfg.model.block_size
            en_loss = eval_loss(model, en_val, block_size, args.batch_size, args.n_batches, args.device, 0)
            zh_offset = QWEN3_VOCAB if label == "compartmented" else 0
            zh_loss = eval_loss(model, zh_val, block_size, args.batch_size, args.n_batches, args.device, zh_offset)
            print(f"    step {step:>6d}: EN {en_loss:.4f}  ZH {zh_loss:.4f}")
            rows.append((step, en_loss, zh_loss))
            del model
            torch.cuda.empty_cache()
        results[label] = rows

    # Save raw
    with open("multilingual_val_curves_no_infonce.json", "w") as f:
        json.dump({k: [{"step": s, "en": e, "zh": z} for s, e, z in v] for k, v in results.items()}, f, indent=2)

    # Two plots
    for which, idx, fname in [("EN val loss", 1, "multilingual_val_curves_en_no_infonce.png"),
                               ("ZH val loss", 2, "multilingual_val_curves_zh_no_infonce.png")]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for label, substr, color, marker in runs:
            if label not in results or not results[label]:
                continue
            rows = results[label]
            xs = [r[0] for r in rows]
            ys = [r[idx] for r in rows]
            ax.plot(xs, ys, marker=marker, color=color, linewidth=2, markersize=6,
                    label=f"{label} (last: {ys[-1]:.3f} @ {xs[-1]})")
        ax.set_xlabel("Step")
        ax.set_ylabel(which + " (lower = better)")
        ax.set_title(f"{which} — all 4 runs over training\n12-256, Qwen3 152k vocab — shared model: 87.2M params")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)
        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()

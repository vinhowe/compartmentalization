"""Per-language val loss curves for the multilingual scaling runs.

Walks every saved checkpoint of one run (selected by --label) and evaluates
on EN and ZH val splits. Writes a JSON file keyed by label so multiple
parallel processes can each handle one run.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_multilingual_per_lang.py --scale 24-512 --label shared
  CUDA_VISIBLE_DEVICES=1 python eval_multilingual_per_lang.py --scale 24-512 --label compartmented
  CUDA_VISIBLE_DEVICES=2 python eval_multilingual_per_lang.py --scale 24-768 --label en-only
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "..")
from eval_utils import load_eval_model_from_checkpoint


PROJECT_ROOT = Path("..")
RUNS_ROOT = PROJECT_ROOT / "out" / "translation-compression" / "multilingual-scaling"
EN_VAL_PATTERN = "../data/wiki-en-qwen3/wiki_val_*.bin"
ZH_VAL_PATTERN = "../data/wiki-zh-qwen3/wiki_val_*.bin"
QWEN3_VOCAB = 151_936

SCALES = ("24-512", "24-768", "24-1024")
LABELS = ("shared", "compartmented", "en-only")


def run_substr(scale: str, label: str) -> str:
    return f"multilingual-{label}-{scale}-qwen3__"


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
    ap.add_argument("--scale", required=True, choices=SCALES,
                    help="Model size, e.g. 24-512, 24-768, 24-1024")
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_batches", type=int, default=200)
    ap.add_argument("--out_json", default=None,
                    help="Output JSON. Default: multilingual_<scale>_per_lang.json with underscore normalization.")
    args = ap.parse_args()

    if args.out_json is None:
        # 24-512 -> multilingual_24_512_per_lang.json (paper convention)
        scale_underscore = args.scale.replace("-", "_")
        args.out_json = f"multilingual_{scale_underscore}_per_lang.json"

    print(f"=== {args.scale} / {args.label} ===")
    en_val = load_val_tokens(EN_VAL_PATTERN)
    zh_val = load_val_tokens(ZH_VAL_PATTERN)
    print(f"  EN: {len(en_val):,} tokens   ZH: {len(zh_val):,} tokens")

    substr = run_substr(args.scale, args.label)
    run_dir = find_latest_run(substr)
    print(f"  run: {run_dir.name}")
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
        zh_offset = QWEN3_VOCAB if args.label == "compartmented" else 0
        zh_loss = eval_loss(model, zh_val, block_size, args.batch_size, args.n_batches, args.device, zh_offset)
        print(f"    step {step:>6d}: EN {en_loss:.4f}  ZH {zh_loss:.4f}", flush=True)
        rows.append({"step": step, "en": en_loss, "zh": zh_loss})
        del model
        torch.cuda.empty_cache()

    out_path = Path(args.out_json)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {}
    existing[args.label] = rows
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate finetune offset checkpoints (ce-full-{2,4,8}comp) on FineWeb BPE16384 validation data.

These are bare state_dict checkpoints without config files, so we construct model configs manually.
Results are saved to finetune_val_metrics.json.

Usage:
    python evaluate_finetune_checkpoints.py [--device cuda:0]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.append("..")

from src.config.job_config import Model
from src.model import GPT

from eval_utils import (
    Assignment,
    SingleShardAssignedValLoader,
)

# --- Constants ---
BASE_VOCAB = 16384
MAX_COMPARTMENTS = 16  # comp_emb has 16 rows in all checkpoints
N_LAYER = 8
N_EMBD = 256
N_HEAD = 4
BLOCK_SIZE = 64
B = 32  # batch size for eval
T = 64  # sequence length
N_EVAL_BATCHES = 10  # number of batches to average over per assignment

STORAGE_ROOT = Path("../")
VAL_BIN_PATTERN = "data/fineweb350B-dedup-bpe16384/fineweb350b-dedup_val_*.bin"

CHECKPOINT_BASE = Path("finetune_offset_checkpoints")

# Runs to evaluate: (dir_name, n_compartments_model, label, use_rope)
RUNS = [
    ("ce-full-2comp", 2, "ce-full-2comp", False),
    ("ce-full-4comp", 4, "ce-full-4comp", False),
    ("ce-full-8comp", 8, "ce-full-8comp", False),
    ("ce-full-2comp-rope", 2, "ce-full-2comp-rope", True),
    ("ce-full-4comp-rope", 4, "ce-full-4comp-rope", True),
    ("ce-full-8comp-rope", 8, "ce-full-8comp-rope", True),
]

# The 256 baseline ran with batch_size=2048, grad_accum=1, so 2048 examples/step
# These finetune runs also use 2048 examples/step
EXAMPLES_PER_STEP = 2048
# The baseline was trained for some number of steps before finetuning started.
# We need to know the offset to compute absolute token counts.
# The baseline checkpoint used was at step 1000000 (1M steps * 2048 examples * 64 tokens)
BASELINE_TOKENS = 1_000_000 * EXAMPLES_PER_STEP * T


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate finetune checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def build_model(n_compartments_model: int, device: str, use_rope: bool = False):
    """Build a GPT model for the given compartment count (non-permutation mode)."""
    vocab = BASE_VOCAB * n_compartments_model + 1
    translation_token_id = BASE_VOCAB * n_compartments_model

    gptconf = Model(
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        dropout=0.0,
        bias=False,
        weight_tying=False,
        vocab_size=vocab,
        embedding_vocab_size=vocab,
        shared_token_embeddings=False,
        use_compartment_embeddings=True,
        copy_compartment_embeddings=False,
        copy_compartment_lm_head=False,
        base_vocab_size=BASE_VOCAB,
        max_compartments=MAX_COMPARTMENTS,  # comp_emb size (16), not n_compartments
        translation_token_id=translation_token_id,
        use_rope=use_rope,
        rope_base=10000.0,
    )
    model = GPT(gptconf)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return model


def load_checkpoint(model, checkpoint_path: Path, device: str):
    """Load a bare state_dict checkpoint into the model."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model


def build_assignments(n_compartments):
    """Build evaluation assignments matching the main eval script's logic."""
    import itertools
    import random

    MAX_EVAL_COMPARTMENTS = 8
    MAX_BIDIRECTIONAL_PAIRS = 8

    rng = random.Random(1024)
    all_compartments = list(range(n_compartments))
    if n_compartments <= MAX_EVAL_COMPARTMENTS:
        sampled_compartments = all_compartments
    else:
        sampled_compartments = sorted(rng.sample(all_compartments, MAX_EVAL_COMPARTMENTS))

    assignments = []
    for i in sampled_compartments:
        assignments.append((Assignment(kind=0, src=i), f"compartment_{i}"))

    possible_links = list(itertools.combinations(sampled_compartments, 2))
    if len(possible_links) <= MAX_BIDIRECTIONAL_PAIRS:
        sampled_links = possible_links
    else:
        sampled_links = rng.sample(possible_links, MAX_BIDIRECTIONAL_PAIRS)

    for src, dst in sampled_links:
        assignments.append(
            (Assignment(kind=1, src=src, dst=dst), f"compartment_{src} > compartment_{dst}")
        )
        assignments.append(
            (Assignment(kind=1, src=dst, dst=src), f"compartment_{dst} > compartment_{src}")
        )

    return assignments


def masked_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if denom.item() == 0:
        return torch.tensor(float("nan"), device=losses.device)
    return (losses * mask.to(dtype=losses.dtype)).sum() / denom


def main():
    args = parse_args()
    device = args.device

    # Find validation data
    val_bin_matches = sorted(STORAGE_ROOT.glob(VAL_BIN_PATTERN))
    if not val_bin_matches:
        print(f"No validation files found for pattern: {VAL_BIN_PATTERN}")
        return
    val_bin_path = val_bin_matches[0]
    print(f"Using validation file: {val_bin_path}")

    # Load existing results for resume
    output_file = Path("finetune_val_metrics.json")
    if output_file.exists():
        with open(output_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    for dir_name, n_comp_model, label, use_rope in RUNS:
        print(f"\n=== Evaluating {label} (n_compartments={n_comp_model}, rope={use_rope}) ===")

        ckpt_dir = CHECKPOINT_BASE / dir_name
        if not ckpt_dir.exists():
            print(f"  Checkpoint dir not found: {ckpt_dir}")
            continue

        # Find all checkpoint steps
        step_numbers = sorted([
            int(m.group(1))
            for f in ckpt_dir.glob("model_step_*.pt")
            if (m := re.search(r"model_step_(\d+)\.pt", f.name))
        ])
        print(f"  Found {len(step_numbers)} checkpoints: {step_numbers}")

        # Resume: check what's already done
        existing = all_metrics.get(label, {})
        done_checkpoints = set(existing.get("checkpoints", []))
        new_steps = [s for s in step_numbers if s not in done_checkpoints]

        if not new_steps:
            print(f"  All checkpoints already evaluated, skipping")
            continue

        print(f"  {len(new_steps)} new checkpoints to evaluate")

        # Build model once, reload weights per checkpoint
        model = build_model(n_comp_model, device, use_rope=use_rope)
        assignments = build_assignments(n_comp_model)

        # Initialize from existing or empty
        metrics = defaultdict(list, existing.get("metrics", {}))
        checkpoints = sorted(existing.get("checkpoints", []))

        for step in tqdm(new_steps, desc=label):
            ckpt_path = ckpt_dir / f"model_step_{step}.pt"
            load_checkpoint(model, ckpt_path, device)

            with torch.inference_mode():
                for assignment, name in tqdm(assignments, unit="metric", leave=False):
                    torch.random.manual_seed(1024)

                    val_loader = SingleShardAssignedValLoader(
                        str(val_bin_path),
                        B=B,
                        T=T,
                        base_vocab_size=BASE_VOCAB,
                        max_compartments=n_comp_model,
                        assignment=assignment,
                        device=device,
                        permute_tokens=False,
                        permute_inputs=False,
                    )

                    batch_losses = []
                    batch_ref_losses = []
                    batch_tgt_losses = []
                    batch_trans_losses = []

                    for _ in range(N_EVAL_BATCHES):
                        x, y, cids = val_loader.next_batch()
                        logits, _ = model(x, y, compartment_ids=cids)

                        # Compute loss in float32 for precision
                        vocab = logits.size(-1)
                        per_token_loss = F.cross_entropy(
                            logits.float().reshape(-1, vocab),
                            y.reshape(-1),
                            reduction="none",
                            ignore_index=-1,
                        ).reshape(B, T)
                        valid = y != -1
                        batch_losses.append(masked_mean(per_token_loss, valid).item())

                        # For translation assignments, compute ref/target breakdown
                        if assignment.kind == 1:
                            half = T // 2

                            ref_mask = torch.zeros_like(valid)
                            ref_mask[:, :half - 1] = valid[:, :half - 1]

                            tgt_mask = torch.zeros_like(valid)
                            tgt_mask[:, half:T - 1] = valid[:, half:T - 1]

                            batch_ref_losses.append(
                                masked_mean(per_token_loss, ref_mask).item()
                            )
                            batch_tgt_losses.append(
                                masked_mean(per_token_loss, tgt_mask).item()
                            )
                            batch_trans_losses.append(
                                masked_mean(per_token_loss, ref_mask | tgt_mask).item()
                            )

                    # Average over batches
                    metrics[f"loss_{name}"].append(np.mean(batch_losses))

                    if assignment.kind == 1:
                        metrics[f"loss_reference_{name}"].append(np.mean(batch_ref_losses))
                        metrics[f"loss_target_{name}"].append(np.mean(batch_tgt_losses))
                        metrics[f"loss_translation_tokens_{name}"].append(np.mean(batch_trans_losses))

            checkpoints.append(step)
            checkpoints.sort()

            # Compute token counts: finetune step * examples_per_step * T + baseline offset
            token_counts = [
                [s * EXAMPLES_PER_STEP * T + BASELINE_TOKENS]
                for s in checkpoints
            ]

            all_metrics[label] = {
                "metrics": dict(metrics),
                "checkpoints": checkpoints,
                "token_counts": token_counts,
                "n_compartments": n_comp_model,
                "finetune_offset_tokens": BASELINE_TOKENS,
            }

            # Save after each checkpoint
            with open(output_file, "w") as f:
                json.dump(all_metrics, f, indent=4)

        print(f"  Done. Saved {len(checkpoints)} checkpoints for {label}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

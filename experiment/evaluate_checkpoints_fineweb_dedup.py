# %%
import argparse
import wandb
import re
import sys
import tempfile
from pathlib import Path
import shutil
import random
import itertools

from typing import cast
import numpy as np
import torch
import torch.nn.functional as F
from eval_utils import (
    Assignment,
    SingleShardAssignedValLoader,
    UniformAssignedValLoader,
    load_eval_model_from_checkpoint,
    get_base_vocab_size,
    is_uniform_data_source,
)
import json
from collections import defaultdict
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on FineWeb validation set")
    parser.add_argument('--rank', type=int, default=0, help='GPU rank (0-indexed)')
    parser.add_argument('--world-size', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--use-prefilled-checkpoints', action='store_true',
                        help='Only evaluate checkpoints already listed in val_metrics.json')
    parser.add_argument('--loggy-checkpoints', action='store_true',
                        help='Only evaluate log-spaced checkpoints: 100, 850, 3500, 7000, 14000, 29000, 60000, 120000, 240000, 500000, 1000000')
    parser.add_argument('--sweeps', type=str, default=None,
                        help='Comma-separated wandb sweep IDs to filter')
    parser.add_argument('--groups', type=str, default=None,
                        help='Comma-separated group names to filter (e.g., "bpe16384-baselines,bpe16384-sweep")')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate all checkpoints even if already present in metrics files')
    parser.add_argument('--scan-dir', action='store_true',
                        help='Discover runs by scanning the filesystem instead of querying wandb. '
                             'Use this for offline runs (WANDB_MODE=offline). Note: --sweeps cannot be '
                             'used with --scan-dir; use --groups for filtering.')
    return parser.parse_args()


# %%
sys.path.append("..")

# %%
# STORAGE_ROOT can be overridden via TC_STORAGE_ROOT env var (matches train.py).
# Used to locate out/translation-compression/<group>/<run_dir>/checkpoints/
import os as _os
STORAGE_ROOT = Path(_os.environ.get("TC_STORAGE_ROOT", "../"))

LOGGY_CHECKPOINTS = [100, 850, 3500, 7000, 14000, 29000, 60000, 120000, 200000, 240000, 300000, 350000, 400000, 500000, 1000000]


def backup_file(path: Path) -> None:
    """
    Create a simple on-disk backup alongside the given file path, if it exists.

    This is intentionally minimal: we only keep the most recent backup
    (e.g. `file.json.bak`) so that if a write or process crashes, we can
    recover the last known-good state.
    """
    if path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)


# %%
EXPERIMENT_GROUP_DIR = STORAGE_ROOT / "out/translation-compression/"

# %%
# every_10k = list(range(0, 1000000 + 1, 10000))

# %%
sweep_1_id = Path("seed-64-sweep-prototype-1")
dry_runs_id = Path("dw-2-1-dry-runs")


# %%
def find_all_checkpoints(experiment_dir):
    checkpoint_dir = EXPERIMENT_GROUP_DIR / experiment_dir / "checkpoints"
    return sorted(
        [
            int(match.group(1))
            for p in checkpoint_dir.glob("*")
            if (match := re.search(r"step-(\d+)", str(p.name))) is not None
        ]
    )


# %%
# experiments = [
#     # d_model=32 n=2 t=1 equal
#     (
#         sweep_1_id / "2025-11-16T17-36-53Z__sweep__cab70dd1__s64__b385cca__1a5c482c",
#         every_10k,
#     ),
#     # d_model=32 n=2 t=0 equal
#     (
#         sweep_1_id / "2025-11-15T16-39-29Z__sweep__601050da__s64__b385cca__0f1d6fd8",
#         every_10k,
#     ),
#     # d_model=256 n=2 t=1 equal
#     (
#         sweep_1_id / "2025-11-16T17-36-54Z__sweep__da5d9cdb__s64__b385cca__9c79d14f",
#         every_10k,
#     ),
#     # d_model=256 n=2 t=0 equal
#     (
#         sweep_1_id / "2025-11-15T16-39-29Z__sweep__454053cb__s64__b385cca__18607215",
#         every_10k,
#     ),
#     # d_model=32 n=4 t=0 equal
#     (
#         sweep_1_id / "2025-11-12T16-59-16Z__sweep__6e91ed43__s64__b385cca__9a9585fc",
#         every_10k,
#     ),
#     # d_model=32 n=4 t=1 equal
#     (
#         sweep_1_id / "2025-11-16T23-24-31Z__sweep__61f1bc35__s64__b385cca__8aeece84",
#         every_10k,
#     ),
#     # d_model=256 n=4 t=0 equal
#     (
#         sweep_1_id / "2025-11-12T17-04-53Z__sweep__2e672abb__s64__b385cca__4f079d12",
#         every_10k,
#     ),
#     # d_model=256 n=4 t=1 equal
#     (
#         sweep_1_id / "2025-11-17T04-24-50Z__sweep__ea79a276__s64__b385cca__0a6aa6a2",
#         every_10k,
#     ),
#     # d_model=32 n=1 baseline
#     dry_runs_id
#     / "2025-10-27T18-46-21Z__1-domain-baseline-fineweb-8-32-prototype__877363f5__s64__8000b93__ded62ee8",
# ]

# for i in range(len(experiments)):
#     experiment = experiments[i]
#     if isinstance(experiment, Path):
#         experiment = (str(experiment), find_all_checkpoints(experiment))
#     else:
#         experiment = (str(experiment[0]), experiment[1])
#     experiments[i] = experiment

# %%
# Device will be set based on --rank argument in main block


# %%
MAX_EVAL_COMPARTMENTS = 8
MAX_BIDIRECTIONAL_PAIRS = 8


def build_assignments(config):
    assignments = []
    rng = random.Random(1024)

    n_compartments = config.experiment.n_compartments
    all_compartments = list(range(n_compartments))
    if n_compartments <= MAX_EVAL_COMPARTMENTS:
        sampled_compartments = all_compartments
    else:
        sampled_compartments = sorted(
            rng.sample(all_compartments, MAX_EVAL_COMPARTMENTS)
        )

    for i in sampled_compartments:
        assignments.append((Assignment(kind=0, src=i), f"compartment_{i}"))

    possible_links = list(itertools.combinations(sampled_compartments, 2))
    if len(possible_links) <= MAX_BIDIRECTIONAL_PAIRS:
        sampled_links = possible_links
    else:
        sampled_links = rng.sample(possible_links, MAX_BIDIRECTIONAL_PAIRS)

    for src, dst in sampled_links:
        assignments.append(
            (
                Assignment(kind=1, src=src, dst=dst),
                f"compartment_{src} > compartment_{dst}",
            )
        )
        assignments.append(
            (
                Assignment(kind=1, src=dst, dst=src),
                f"compartment_{dst} > compartment_{src}",
            )
        )

    return assignments


# %%
def get_permutations_path(config):
    exp = config.experiment
    if not exp.permute_tokens_per_compartment:
        return None

    base_vocab = config.model.vocab_size
    if base_vocab is None:
        raise ValueError(
            "model.vocab_size is required for permutation cache resolution"
        )
    if exp.max_compartments is None:
        raise ValueError("experiment.max_compartments is required")

    max_compartments_int = cast(int, exp.max_compartments)
    cache_root = STORAGE_ROOT / "cache"
    training_seed = config.training.seed

    perms_desc = (
        f"basev{int(base_vocab)}_maxc{max_compartments_int}_seed{int(training_seed)}"
    )
    return cache_root / f"permutations_{perms_desc}.npy"


def masked_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `losses` over True entries in `mask`."""
    denom = mask.sum()
    if denom.item() == 0:
        return torch.tensor(float("nan"), device=losses.device)
    return (losses * mask.to(dtype=losses.dtype)).sum() / denom


if __name__ == "__main__":
    args = parse_args()
    # When CUDA_VISIBLE_DEVICES is set, PyTorch only sees those GPUs as cuda:0, cuda:1, etc.
    # So we use cuda:0 by default, letting CUDA_VISIBLE_DEVICES handle the actual GPU mapping.
    # Only use cuda:{rank} when running without CUDA_VISIBLE_DEVICES restriction.
    import os
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device = "cuda:0"
    else:
        device = f"cuda:{args.rank}"
    print(f"[Rank {args.rank}/{args.world_size}] Using device: {device}")

    blacklisted_run_names = {
        "128-domain-translations-fineweb-8-256-prototype",
        "16-domain-translations-fineweb-8-256-prototype",
        "4-domain-translations-fineweb-8-256-prototype",
    }

    path_to_run_name: dict[str, str] = {}
    path_to_run: dict[str, object] = {}  # only populated in wandb mode

    if args.scan_dir:
        if args.sweeps:
            raise SystemExit("--sweeps cannot be used with --scan-dir (sweep IDs only exist in wandb)")
        # Discover runs by walking the output directory tree.
        # Layout: EXPERIMENT_GROUP_DIR/<group>/<run_dir>/{checkpoints,meta}/...
        # Run name comes from meta/config.json -> logging.wandb_run_name (fallback to dir basename).
        print(f"Scanning {EXPERIMENT_GROUP_DIR} for runs with checkpoints...")
        if not EXPERIMENT_GROUP_DIR.exists():
            raise SystemExit(f"Experiment dir not found: {EXPERIMENT_GROUP_DIR}")
        for group_dir in sorted(EXPERIMENT_GROUP_DIR.iterdir()):
            if not group_dir.is_dir():
                continue
            for run_dir in sorted(group_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                ck_dir = run_dir / "checkpoints"
                if not ck_dir.is_dir():
                    continue
                rel_path = f"{group_dir.name}/{run_dir.name}"
                # Try to get pretty run name from meta/config.json
                run_name = run_dir.name
                meta_cfg = run_dir / "meta" / "config.json"
                if meta_cfg.exists():
                    try:
                        with open(meta_cfg) as f:
                            cfg = json.load(f)
                        run_name = cfg.get("logging", {}).get("wandb_run_name") or run_name
                    except (json.JSONDecodeError, OSError):
                        pass
                if run_name in blacklisted_run_names:
                    continue
                path_to_run_name[rel_path] = run_name
        print(f"Found {len(path_to_run_name)} runs on disk (after blacklist):")
        for p, name in list(path_to_run_name.items())[:20]:
            print(f"  - {name} ({p})")
        if len(path_to_run_name) > 20:
            print(f"  ... ({len(path_to_run_name) - 20} more)")
    else:
        # Initialize the API
        api = wandb.Api()
        runs = api.runs("pccl/translation-compression")
        runs = [run for run in runs if run.name not in blacklisted_run_names]

        print(f"Found {len(runs)} runs to process (after blacklist):")
        for run in runs:
            print(f"  - {run.name}")

        # Map experiment path (relative out_dir subpath) to the corresponding run name
        path_to_run_name = {
            str(Path(*Path(run.config["out_dir"]).parts[-2:])): run.name
            for run in runs
            if "out_dir" in run.config
        }

        # List number of experiments without out_dir in the config
        experiments_without_out_dir = [
            run.name for run in runs if "out_dir" not in run.config
        ]
        print(
            f"Found {len(experiments_without_out_dir)} experiments without out_dir in the config:"
        )
        for experiment in experiments_without_out_dir:
            print(f"  - {experiment}")

        # Build reverse mapping: path -> run object for sweep filtering
        path_to_run = {
            str(Path(*Path(run.config["out_dir"]).parts[-2:])): run
            for run in runs
            if "out_dir" in run.config
        }

    # Load original file for resume state (read-only reference)
    # Also used to get pre-filled checkpoints when --use-prefilled-checkpoints is set
    original_metrics = {}
    if Path("val_metrics.json").exists():
        with open("val_metrics.json", "r") as f:
            original_metrics = json.load(f)

    experiment_paths = list(path_to_run_name.keys())

    # Filter by wandb sweep IDs if specified
    if args.sweeps:
        sweep_ids = set(s.strip() for s in args.sweeps.split(','))
        experiment_paths = [
            p for p in experiment_paths
            if path_to_run.get(p) and path_to_run[p].sweep and path_to_run[p].sweep.id in sweep_ids
        ]
        print(f"Filtered to {len(experiment_paths)} experiments matching sweep IDs: {sweep_ids}")

    # Filter by group names if specified (first part of path, e.g., "bpe16384-baselines")
    if args.groups:
        group_names = set(s.strip() for s in args.groups.split(','))
        experiment_paths = [
            p for p in experiment_paths
            if p.split('/')[0] in group_names
        ]
        print(f"Filtered to {len(experiment_paths)} experiments matching groups: {group_names}")

    def get_checkpoints_of_interest(path: str) -> list[int]:
        """Get checkpoints to evaluate for an experiment."""
        if args.use_prefilled_checkpoints and path in original_metrics:
            prefilled = original_metrics[path].get("checkpoints", [])
            if prefilled:
                return prefilled
        all_ckpts = find_all_checkpoints(path)
        if args.loggy_checkpoints:
            loggy_set = set(LOGGY_CHECKPOINTS)
            return [c for c in all_ckpts if c in loggy_set]
        return all_ckpts

    all_experiments: list[tuple[str, list[int]]] = [
        (path, get_checkpoints_of_interest(path)) for path in experiment_paths
    ]

    # Partition experiments by rank (round-robin)
    experiments: list[tuple[str, list[int]]] = [
        exp for i, exp in enumerate(all_experiments) if i % args.world_size == args.rank
    ]
    print(f"[Rank {args.rank}/{args.world_size}] Processing {len(experiments)} of {len(all_experiments)} experiments")

    # Load per-GPU file for this run's state
    output_file = Path(f"val_metrics_gpu{args.rank}.json")
    if output_file.exists():
        with open(output_file, "r") as f:
            val_metrics = json.load(f)
    else:
        val_metrics = {}

    def is_checkpoint_done(experiment_key, checkpoint):
        """Check if a checkpoint is already done in either original or per-GPU metrics.

        A checkpoint is considered done only if it's in the checkpoints list AND
        there are corresponding metrics (i.e., metrics list length >= checkpoint index + 1).
        """
        for source in [val_metrics, original_metrics]:
            if experiment_key in source:
                ckpts = source[experiment_key].get("checkpoints", [])
                if checkpoint in ckpts:
                    # Check if metrics actually exist for this checkpoint
                    metrics = source[experiment_key].get("metrics", {})
                    if metrics:
                        ckpt_index = ckpts.index(checkpoint)
                        # Check if any metric has a value at this index
                        first_metric = next(iter(metrics.values()), [])
                        if len(first_metric) > ckpt_index:
                            return True
        return False

    def get_existing_metrics(experiment_key):
        """Get existing metrics for an experiment, preferring per-GPU file over original."""
        if experiment_key in val_metrics:
            return val_metrics[experiment_key]
        if experiment_key in original_metrics:
            return original_metrics[experiment_key]
        return {}

    for experiment, checkpoints_of_interest in experiments:
        run_name = path_to_run_name.get(experiment, "<unknown-run-name>")
        print(f"\n=== Processing run: {run_name} (experiment path: {experiment}) ===")

        experiment_dir = EXPERIMENT_GROUP_DIR / experiment
        checkpoint_dir = experiment_dir / "checkpoints"

        if args.force:
            existing_data = {}
        else:
            existing_data = get_existing_metrics(experiment)
        # Store metrics keyed by step so re-sort at save time stays aligned.
        # Earlier versions kept `metrics` as parallel arrays in append order
        # while saving `checkpoints: sorted(...)`; mixing old + new steps
        # silently misaligned losses with steps. By-step storage avoids this.
        existing_checkpoints: list[int] = list(existing_data.get("checkpoints", []))
        existing_metrics: dict = existing_data.get("metrics", {})
        metrics_by_step: dict[str, dict[int, float]] = defaultdict(dict)
        for _name, _vals in existing_metrics.items():
            for _step, _val in zip(existing_checkpoints, _vals):
                metrics_by_step[_name][int(_step)] = _val
        checkpoints = set(int(s) for s in existing_checkpoints)

        step_numbers = sorted(
            [
                int(match.group(1))
                for p in checkpoint_dir.glob("*")
                if (match := re.search(r"step-(\d+)", str(p.name))) is not None
                and int(match.group(1)) in checkpoints_of_interest
            ]
        )

        print(f"Found {len(step_numbers)} checkpoints (steps {step_numbers[0]}..{step_numbers[-1]})" if step_numbers else "No checkpoints found")

        if not step_numbers:
            print(
                f"Skipping run {run_name}: no matching checkpoints found in {checkpoint_dir}"
            )
            continue

        # Also check original metrics for done checkpoints (unless --force)
        if args.force:
            new_step_numbers = step_numbers
        else:
            new_step_numbers = sorted([
                step for step in step_numbers
                if not is_checkpoint_done(experiment, step)
            ])

        if not new_step_numbers:
            print(
                f"Skipping run {run_name}: all {len(step_numbers)} matching checkpoints are already processed"
            )
            continue

        # Only load a model/config once we know there's at least one new checkpoint to evaluate.
        step_dir = checkpoint_dir / f"step-{new_step_numbers[0]:06d}"
        try:
            eval_model, config, actual_model_compartments = load_eval_model_from_checkpoint(
                step_dir, experiment_dir, device
            )
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(
                    f"Skipping run {run_name}: checkpoint/config mismatch - {e}"
                )
                continue
            raise

        print(f"Model loaded with {actual_model_compartments} compartments")

        # Detect data source type
        use_uniform_data = is_uniform_data_source(config)
        base_vocab_size = get_base_vocab_size(config)

        # Resolve per-compartment validation sources (if compartment_val_bins is set)
        # Maps compartment index -> file path (str) or "synthetic:uniform"/"synthetic:frequency"
        compartment_val_sources: dict[int, str] | None = None
        synthetic_token_probs: np.ndarray | None = None

        if config.data.compartment_val_bins:
            compartment_val_sources = {}
            has_synthetic_frequency = False
            freq_source_pattern = None
            for comp_idx, pattern in enumerate(config.data.compartment_val_bins):
                if pattern.startswith("synthetic:"):
                    compartment_val_sources[comp_idx] = pattern
                    if pattern == "synthetic:frequency":
                        has_synthetic_frequency = True
                else:
                    matches = sorted(STORAGE_ROOT.glob(pattern))
                    if matches:
                        compartment_val_sources[comp_idx] = str(matches[0])
                        if freq_source_pattern is None:
                            freq_source_pattern = pattern
                    else:
                        print(f"Warning: no files match compartment {comp_idx} val pattern: {pattern}")

            # Compute frequency distribution for synthetic:frequency compartments
            if has_synthetic_frequency:
                # Find a pretokenized train pattern for frequency computation
                train_patterns = config.data.compartment_train_bins
                freq_train_source = None
                if train_patterns:
                    freq_train_source = next(
                        (p for p in train_patterns if not p.startswith("synthetic:")),
                        None,
                    )
                if freq_train_source:
                    sys.path.insert(0, str(STORAGE_ROOT))
                    from src.token_tying import compute_token_frequencies
                    freqs = compute_token_frequencies(
                        str(STORAGE_ROOT / freq_train_source), base_vocab_size
                    )
                    synthetic_token_probs = (freqs / freqs.sum()).astype(np.float64)
                    print(f"Computed token frequencies from {freq_train_source}")
                else:
                    print(f"Warning: synthetic:frequency needs pretokenized data for frequencies, falling back to uniform")

            print(f"Per-compartment val sources: {compartment_val_sources}")

        # Resolve validation data path from config (single-source fallback)
        val_bin_path = None
        if not use_uniform_data and compartment_val_sources is None:
            val_bin_pattern = config.data.val_bin
            if val_bin_pattern:
                # Resolve glob pattern relative to STORAGE_ROOT
                val_bin_matches = sorted(STORAGE_ROOT.glob(val_bin_pattern))
                if val_bin_matches:
                    val_bin_path = val_bin_matches[0]  # Use first shard
                    print(f"Using validation file: {val_bin_path} (vocab_size={base_vocab_size})")
                else:
                    print(
                        f"Skipping run {run_name}: validation file pattern '{val_bin_pattern}' "
                        f"matched no files"
                    )
                    continue
            else:
                print(f"Skipping run {run_name}: no val_bin configured")
                continue

        # Handle permutations - generate in-memory if not cached
        permutations_path = get_permutations_path(config)
        has_permutations = config.experiment.permute_tokens_per_compartment
        in_memory_permutations = None  # In-memory permutations when cache doesn't exist

        if has_permutations:
            if permutations_path is not None and permutations_path.exists():
                # Use cached permutations file
                pass
            else:
                # Generate permutations in-memory (works for both uniform and pretokenized data)
                # Use max_compartments from config for permutation generation (training used this)
                max_c = cast(int, config.experiment.max_compartments)
                training_seed = config.training.seed
                ss = np.random.SeedSequence(int(training_seed) & 0xFFFFFFFFFFFFFFFF)
                child_seeds = ss.spawn(max_c)
                in_memory_permutations = np.empty((max_c, base_vocab_size), dtype=np.int64)
                for c, child_ss in enumerate(child_seeds):
                    gen = np.random.Generator(np.random.PCG64(child_ss))
                    in_memory_permutations[c] = gen.permutation(base_vocab_size).astype(np.int64)
                print(f"Generated in-memory permutations with shape {in_memory_permutations.shape}")

        assignments = build_assignments(config)

        for step_number in tqdm(new_step_numbers):
            step_dir = checkpoint_dir / f"step-{step_number:06d}"
            try:
                eval_model, config, _ = load_eval_model_from_checkpoint(
                    step_dir, experiment_dir, device
                )
            except (RuntimeError, EOFError, OSError) as e:
                msg = str(e)
                if ("size mismatch" in msg
                        or "PytorchStreamReader" in msg
                        or "central directory" in msg
                        or "Ran out of input" in msg):
                    print(f"Skipping checkpoint {step_number}: {e}")
                    continue
                raise
            checkpoints.add(step_number)
            with torch.inference_mode():
                for assignment, name in tqdm(assignments, unit="metric"):
                    torch.random.manual_seed(1024)

                    # Determine loader type based on per-compartment val sources
                    val_loader = None
                    if compartment_val_sources is not None:
                        # Per-compartment mode: check if the assignment's compartment(s) are synthetic
                        src_source = compartment_val_sources.get(assignment.src)
                        if assignment.kind == 1:
                            dst_source = compartment_val_sources.get(assignment.dst)
                            # Skip translation assignments involving synthetic compartments
                            if (src_source and src_source.startswith("synthetic:")) or \
                               (dst_source and dst_source.startswith("synthetic:")):
                                # Can't do file-based translation eval with synthetic compartments
                                continue

                        if src_source and src_source.startswith("synthetic:"):
                            mode = src_source.split(":", 1)[1]
                            val_loader = UniformAssignedValLoader(
                                B=32,
                                T=64,
                                base_vocab_size=base_vocab_size,
                                max_compartments=actual_model_compartments,
                                assignment=assignment,
                                seed=0,
                                num_batches=64,
                                device=device,
                                permute_tokens=has_permutations,
                                permutations_path=(
                                    str(permutations_path)
                                    if permutations_path is not None
                                    else None
                                ),
                                permutations=in_memory_permutations,
                                permute_inputs=config.experiment.permute_input_tokens_per_compartment,
                                token_probs=synthetic_token_probs if mode == "frequency" else None,
                            )
                        elif src_source:
                            val_loader = SingleShardAssignedValLoader(
                                src_source,
                                B=32,
                                T=64,
                                base_vocab_size=base_vocab_size,
                                max_compartments=actual_model_compartments,
                                assignment=assignment,
                                device=device,
                                permute_tokens=has_permutations,
                                permutations_path=(
                                    str(permutations_path)
                                    if permutations_path is not None and permutations_path.exists()
                                    else None
                                ),
                                permutations=in_memory_permutations,
                                permute_inputs=config.experiment.permute_input_tokens_per_compartment,
                            )

                    if val_loader is None:
                        if use_uniform_data:
                            # Use uniform random data loader with seed 0 for validation
                            val_loader = UniformAssignedValLoader(
                                B=32,
                                T=64,
                                base_vocab_size=base_vocab_size,
                                max_compartments=actual_model_compartments,
                                assignment=assignment,
                                seed=0,  # Fixed seed for validation
                                num_batches=64,
                                device=device,
                                permute_tokens=has_permutations,
                                permutations_path=(
                                    str(permutations_path)
                                    if permutations_path is not None
                                    else None
                                ),
                                permutations=in_memory_permutations,
                                permute_inputs=config.experiment.permute_input_tokens_per_compartment,
                            )
                        else:
                            # Use pretokenized data from config
                            val_loader = SingleShardAssignedValLoader(
                                str(val_bin_path),
                                B=32,
                                T=64,
                                base_vocab_size=base_vocab_size,
                                max_compartments=actual_model_compartments,
                                assignment=assignment,
                                device=device,
                                permute_tokens=has_permutations,
                                permutations_path=(
                                    str(permutations_path)
                                    if permutations_path is not None and permutations_path.exists()
                                    else None
                                ),
                                permutations=in_memory_permutations,
                                permute_inputs=config.experiment.permute_input_tokens_per_compartment,
                            )
                    # Average across N_EVAL_BATCHES per assignment to suppress
                    # single-batch noise (especially visible on bf16-saved named
                    # checkpoints where one batch can land 0.05+ nats off mean).
                    N_EVAL_BATCHES = 100
                    batch_full = []
                    batch_ref = []
                    batch_tgt = []
                    batch_trans = []
                    for _ in range(N_EVAL_BATCHES):
                        x, y, cids = val_loader.next_batch()
                        logits, _ = eval_model(x, y, compartment_ids=cids)

                        # Recompute loss in float32 to avoid bfloat16 quantization
                        loss_f32 = F.cross_entropy(
                            logits.float().reshape(-1, logits.size(-1)),
                            y.reshape(-1),
                            ignore_index=-1,
                        )
                        batch_full.append(loss_f32.item())

                        if assignment.kind == 1:
                            B, T = y.shape
                            half = T // 2
                            vocab = logits.size(-1)
                            per_token_loss = F.cross_entropy(
                                logits.float().reshape(-1, vocab),
                                y.reshape(-1),
                                reduction="none",
                                ignore_index=-1,
                            ).reshape(B, T)
                            valid = y != -1
                            ref_mask = torch.zeros_like(valid)
                            ref_mask[:, : half - 1] = valid[:, : half - 1]
                            tgt_mask = torch.zeros_like(valid)
                            tgt_mask[:, half : T - 1] = valid[:, half : T - 1]
                            batch_ref.append(masked_mean(per_token_loss, ref_mask).item())
                            batch_tgt.append(masked_mean(per_token_loss, tgt_mask).item())
                            batch_trans.append(masked_mean(per_token_loss, ref_mask | tgt_mask).item())

                    import numpy as _np
                    metrics_by_step[f"loss_{name}"][step_number] = float(_np.mean(batch_full))
                    if assignment.kind == 1:
                        metrics_by_step[f"loss_reference_{name}"][step_number] = float(_np.mean(batch_ref))
                        metrics_by_step[f"loss_target_{name}"][step_number] = float(_np.mean(batch_tgt))
                        metrics_by_step[f"loss_translation_tokens_{name}"][step_number] = float(_np.mean(batch_trans))

        sorted_steps = sorted(checkpoints)
        aligned_metrics = {
            name: [step_map[s] for s in sorted_steps if s in step_map]
            for name, step_map in metrics_by_step.items()
        }
        # Sanity check: any metric whose length doesn't match sorted_steps
        # had per-step holes. Drop such metrics rather than emit misaligned
        # arrays — they would silently break the parallel-array invariant.
        for _name in list(aligned_metrics.keys()):
            if len(aligned_metrics[_name]) != len(sorted_steps):
                print(f"  WARNING: metric {_name!r} has {len(aligned_metrics[_name])} values for {len(sorted_steps)} steps; dropping")
                del aligned_metrics[_name]
        experiment_result = {
            "metrics": aligned_metrics,
            "checkpoints": sorted_steps,
            "data_source": "uniform" if use_uniform_data else "pretokenized",
        }
        val_metrics[experiment] = experiment_result

        # Do this every experiment so if things crash, we're okay.
        # Write to per-GPU file (atomic: write temp file, then rename)
        backup_file(output_file)

        fd, tmp_path = tempfile.mkstemp(dir=output_file.parent, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                json.dump(val_metrics, f, indent=4)
            Path(tmp_path).replace(output_file)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        print(f"[Rank {args.rank}] Saved results to {output_file}")

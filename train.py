"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Configuration is read from `config/job_config.py` dataclasses via `ConfigManager`.

To run on a single GPU, example:
$ python train.py --training.batch_size=32 --system.compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import json
import math
import os
import signal
import shutil
import time
import uuid
import glob
from contextlib import nullcontext
from dataclasses import asdict, replace
from typing import Any, Optional, cast

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config.job_config import JobConfig, Model
from src.config.manager import ConfigManager
from src.experiment import append_to_experiment_log, cfg_hash, run_dirs, slug, write_meta
from src.model import GPT
from src.dann import DANNModule, parse_dann_layers
from src.data import UniformBatchDataLoader, UniformCompartmentDataLoader
from src.assignments import write_assignments
from src.token_tying import compute_token_frequencies, compute_tied_mask, apply_tying_to_permutations, build_tying_remap
from src.config.presets import apply_size_tier, apply_bpe16384_batch_config, scale_batch_for_vram
from src.weights import compute_weights_map
from src.run_lock import ActiveRunLock
from src.datafile import load_data_shard, peek_data_shard
from src import lr_schedule
from src import checkpoints as ckpt_utils
import struct
import threading
from queue import Queue

# ---------------------------------------------------------------------------
# Graceful preemption via SIGUSR1
# ---------------------------------------------------------------------------
_preempt_requested = threading.Event()


def _handle_preempt_signal(signum, frame):
    """SIGUSR1 handler: set flag so the training loop can exit cleanly."""
    _preempt_requested.set()


def check_duplicate_run(
    config: JobConfig, wandb_project: str, wandb_group: Optional[str] = None
) -> Optional[str]:
    """Check if a completed run with the same config already exists in wandb.

    Returns the run ID if a duplicate is found, None otherwise.
    Only checks on master process (rank 0).
    """
    if int(os.environ.get("RANK", 0)) != 0:
        return None

    try:
        import wandb

        api = wandb.Api()

        # Extract meaningful config fields for comparison (exclude logging/system settings)
        config_dict = asdict(config)

        # For data section, only compare fields that affect training
        # uniform_seed only matters for uniform data source, not pretokenized
        data_compare = config_dict["data"].copy()
        if config_dict["data"].get("source") == "pretokenized":
            data_compare.pop("uniform_seed", None)

        compare_fields = {
            "model": config_dict["model"],
            "training": config_dict["training"],
            "experiment": config_dict["experiment"],
            "data": data_compare,
            "optimizer": config_dict["optimizer"],
            "lr": config_dict["lr"],
        }

        # Build filters - only query finished runs, optionally within a group
        filters = {"state": "finished"}
        if wandb_group:
            filters["group"] = wandb_group

        # Query completed runs in the project
        runs = api.runs(
            wandb_project,
            filters=filters,
            per_page=1000,
        )

        for run in runs:
            run_config = run.config
            # Compare the meaningful fields
            match = True
            for section, expected in compare_fields.items():
                run_section = run_config.get(section, {})

                # For data section with pretokenized source, ignore uniform_seed in existing run too
                if section == "data" and expected.get("source") == "pretokenized":
                    run_section = run_section.copy() if run_section else {}
                    run_section.pop("uniform_seed", None)

                if run_section != expected:
                    match = False
                    break

            if match:
                return run.id

    except Exception as e:
        # Don't fail training if wandb API is unavailable
        print(f"[dedupe] Warning: Could not check for duplicates: {e}")

    return None


# Limit CPU threads to avoid oversubscription when running multiple processes
# Default to 4 threads; override with OMP_NUM_THREADS env var
_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
torch.set_num_threads(_num_threads)


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def _peek_data_shard(filename):
    return peek_data_shard(filename)


def _load_data_shard(filename):
    return load_data_shard(filename)


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)  # pyright: ignore[reportOptionalOperand]
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def state_dict(self) -> dict:
        return {
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def load_state_dict(self, state: dict) -> None:
        shard = state["current_shard"]
        if shard != self.current_shard:
            self.current_shard = shard
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = state["current_position"]

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


# Granularity of the within-shard shuffle, in tokens. Larger than the 1024-token
# context so a shuffled block still holds a full window of contiguous text.
SHUFFLE_BLOCK = 8192


def _mix_seed(seed: int, salt: int) -> int:
    """Stable 64-bit mix. Same inputs -> same stream in every process."""
    z = (int(seed) * 0x9E3779B97F4A7C15 + int(salt)) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


class _TokenStream:
    """Token reader over a set of .bin shards, optionally shuffled."""

    def __init__(self, filename_pattern: str, process_rank: int, T: int,
                 shuffle_seed: Optional[int] = None):
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )
        # DATA ORDER. Without a shuffle_seed this reads shards in sorted order and
        # each shard front-to-back -- inherited from llm.c, which consumes its
        # whole (small) corpus so order is irrelevant there. Against FineWeb
        # sample-350BT it is not: files 0-29 are ALL CC-MAIN-2013-20, and a 30B
        # run reads only the first ~39 of 510 files, so ~77% of it is one 2013
        # crawl. A 100B run is ~23% that crawl. Different budgets therefore see
        # different data distributions, and the seed changes nothing about the
        # data at all.
        #
        # With a shuffle_seed, order is permuted at two levels: which shard comes
        # next, and which block within a shard. Both are pure functions of the
        # seed, so every rank derives the SAME order and a resume reproduces it
        # without storing a permutation. Rank disjointness still comes from the
        # process_rank stride below, exactly as before -- deliberately, so the
        # union of what the ranks read does not depend on world_size.
        self.shuffle_seed = shuffle_seed
        if shuffle_seed is None:
            self.file_order = np.arange(len(self.files))
        else:
            self.file_order = np.random.default_rng(
                _mix_seed(shuffle_seed, 0x5EED_F11E)
            ).permutation(len(self.files))
        self.current_shard = 0
        self.tokens = self._load_ordered(0)
        if len(self.tokens) > (T + 2):
            self.token_pos = (process_rank * T) % (len(self.tokens) - (T + 2))
        else:
            self.token_pos = 0

    def _load_ordered(self, ordinal: int) -> np.ndarray:
        """Shard at position `ordinal` of the permuted file order, blocks shuffled.

        Permuting whole blocks rather than individual tokens keeps each block a
        contiguous run of real text; only the order blocks appear in changes.
        The tail that does not fill a block is left in place rather than dropped,
        so no tokens are lost.
        """
        tokens = _load_data_shard(self.files[self.file_order[ordinal]])
        if self.shuffle_seed is None:
            return tokens
        n_full = len(tokens) // SHUFFLE_BLOCK
        if n_full < 2:
            return tokens
        rng = np.random.default_rng(
            _mix_seed(self.shuffle_seed, int(self.file_order[ordinal]))
        )
        head = tokens[: n_full * SHUFFLE_BLOCK].reshape(n_full, SHUFFLE_BLOCK)
        out = head[rng.permutation(n_full)].reshape(-1)
        tail = tokens[n_full * SHUFFLE_BLOCK :]
        return np.concatenate([out, tail]) if len(tail) else out

    def read_tokens(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int32)
        filled = 0
        while filled < n:
            remaining = len(self.tokens) - self.token_pos
            if remaining <= 1:
                self.advance()
                continue
            can_take = min(n - filled, remaining)
            out[filled : filled + can_take] = self.tokens[
                self.token_pos : self.token_pos + can_take
            ].astype(np.int32)
            self.token_pos += can_take
            filled += can_take
            if self.token_pos >= len(self.tokens):
                self.advance()
        return out

    def advance(self) -> None:
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.token_pos = 0
        self.tokens = self._load_ordered(self.current_shard)

    def load_shard(self, idx: int) -> None:
        new_idx = idx % len(self.files)
        if new_idx != self.current_shard:
            self.current_shard = new_idx
            self.tokens = self._load_ordered(self.current_shard)
        self.token_pos = 0

    def state_dict(self) -> dict:
        return {
            "current_shard": self.current_shard,
            "token_pos": self.token_pos,
        }

    def load_state_dict(self, state: dict) -> None:
        shard = state["current_shard"]
        if shard != self.current_shard:
            self.load_shard(shard)
        self.token_pos = state["token_pos"]

    def reset(self, process_rank: int, T: int) -> None:
        self.load_shard(0)
        if len(self.tokens) > (T + 2):
            self.token_pos = (process_rank * T) % (len(self.tokens) - (T + 2))
        else:
            self.token_pos = 0


class SyntheticTokenStream:
    """Token stream that generates synthetic tokens on-the-fly."""

    def __init__(self, mode: str, vocab_size: int, seed: int,
                 process_rank: int, frequencies: np.ndarray | None = None,
                 ngram_table_dir: str = "data/ngram-tables-bpe16384"):
        # "uniform" | "frequency" | "ngram{N}" -- ngram is the capacity-competition
        # ladder: order N n-grams estimated on this same FineWeb corpus, so the
        # synthetic compartment becomes progressively more English-like with N.
        self._mode = mode
        self._ngram = None
        if mode.startswith("ngram"):
            from src.ngram_fast import FastNGramSampler
            # "ngram3" = pure trigram; "ngram3x0.5" = Jelinek-Mercer 0.5*P_3 +
            # 0.5*P_2, i.e. a "2.5-gram" rung on the capacity ladder.
            spec = mode[5:]
            if "x" in spec:
                o_str, lam_str = spec.split("x", 1)
                order, lam = int(o_str), float(lam_str)
            else:
                order, lam = int(spec), 1.0
            self._ngram = FastNGramSampler(
                order=order, table_dir=ngram_table_dir, seed=seed,
                process_rank=process_rank, lam=lam,
            )
        self._vocab_size = vocab_size
        self._seed = seed
        self._process_rank = process_rank
        self._rng = np.random.Generator(
            np.random.PCG64(seed + process_rank)
        )
        if mode == "frequency":
            assert frequencies is not None, (
                "synthetic:frequency requires a frequencies array"
            )
            self._probs = (frequencies / frequencies.sum()).astype(np.float64)
        else:
            self._probs = None

    def read_tokens(self, n: int) -> np.ndarray:
        if self._ngram is not None:
            return self._ngram.read_tokens(n)
        if self._mode == "uniform":
            return self._rng.integers(0, self._vocab_size, size=n, dtype=np.int32)
        else:
            return self._rng.choice(
                self._vocab_size, size=n, replace=True, p=self._probs
            ).astype(np.int32)

    def state_dict(self) -> dict:
        if self._ngram is not None:
            return {"ngram": self._ngram.state_dict()}
        return {"rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        if self._ngram is not None and "ngram" in state:
            self._ngram.load_state_dict(state["ngram"])
            return
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]
        # Silently ignore incompatible state (e.g. _TokenStream checkpoint)

    def reset(self, process_rank: int, T: int) -> None:
        self._rng = np.random.Generator(
            np.random.PCG64(self._seed + process_rank)
        )


class AssignmentsDataLoader:
    def __init__(
        self,
        assignments_file: Optional[str],
        filename_pattern: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        base_vocab_size: int,
        max_compartments: int,
        n_compartments: int,
        # Record count to assume when assignments_file is None (c=1). Only sets
        # the wrap point for assignment_idx; the records themselves are constant.
        constant_records: Optional[int] = None,
        # Per-compartment BPE variants. <=1.0 disables, and the whole expansion
        # path is then dead code -- the offset encoding below is untouched.
        bpe_variant_expansion: float = 0.0,
        bpe_variant_tokenizer: str = "",
        bpe_variant_seed: int = 0,
        # Where merge frequencies are counted. Must be the TRAIN shards for every
        # loader, so train and val share one drop set per compartment.
        bpe_variant_freq_pattern: str = "",
        # First id of the per-compartment prefix markers, or None to disable.
        # Compartment i's marker is marker_base + i.
        compartment_marker_base: Optional[int] = None,
        # c-way in, 1-out: targets are base-vocab ids, inputs keep the offset.
        shared_output_vocab: bool = False,
        # Seed for corpus read order; None reads shards and blocks in file order.
        # Identical on every rank on purpose -- rank disjointness comes from the
        # process_rank stride, so the union of what the ranks read does not
        # depend on world_size.
        shuffle_seed: Optional[int] = None,
        permute_tokens: bool = False,
        permutations_path: Optional[str] = None,
        permute_inputs: bool = True,
        pin_memory: bool = True,
        tied_token_mask: Optional[np.ndarray] = None,
        tying_remap: Optional[np.ndarray] = None,
        translation_token_id: Optional[int] = None,
        compartment_filename_patterns: Optional[list[str]] = None,
        synthetic_seed: int = 0,
        synthetic_frequencies: Optional[np.ndarray] = None,
    ) -> None:
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.base_vocab_size = base_vocab_size
        self.max_compartments = max_compartments
        self.n_compartments = n_compartments
        self.permute_tokens = permute_tokens
        self.permute_inputs = permute_inputs
        self._tied_token_mask = tied_token_mask  # bool[base_vocab_size] or None
        self._tying_remap = tying_remap  # int64[n_compartments, base_vocab] or None
        # translation token id differs by mode; uses n_compartments to match model vocab size
        self.translation_token_id = translation_token_id if translation_token_id is not None else (
            base_vocab_size if permute_tokens else base_vocab_size * n_compartments
        )
        # Optionally load permutations array of shape [max_compartments, base_vocab]
        self._permutations: Optional[np.ndarray]
        if self.permute_tokens:
            if permutations_path is None or not os.path.exists(permutations_path):
                raise FileNotFoundError(
                    f"Permutations file not found: {permutations_path}"
                )
            perms = np.load(permutations_path)
            if perms.dtype != np.int64 and perms.dtype != np.int32:
                perms = perms.astype(np.int64)
            rows, cols = perms.shape
            if cols != base_vocab_size:
                raise ValueError(
                    f"permutations.npy base vocab mismatch: {cols} != {base_vocab_size}"
                )
            if rows < max_compartments:
                raise ValueError(
                    f"permutations.npy compartments {rows} < required {max_compartments}"
                )
            if rows > max_compartments:
                perms = perms[:max_compartments]
            self._permutations = perms
        else:
            self._permutations = None

        # Whether to return pinned-memory CPU tensors for faster H2D copies
        self._pin_memory = bool(pin_memory and torch.cuda.is_available())

        # Load assignments header and records.
        #
        # assignments_file is None exactly when n_compartments == 1, where the
        # array is a constant: one category, encoding compartment 0, so every
        # record is the integer 0. Materialising it cost 8 B/example of RAM PER
        # RANK for a value the per-batch gather below always decodes to
        # (kind=0, src=0, dst=0) -- 0.78 GB/rank on a 100B run, 2.34 GB/rank at
        # the pinned 300B horizon, times 8 ranks. Synthesize it instead.
        if assignments_file is None:
            assert n_compartments == 1, (
                "assignments_file may only be omitted at n_compartments == 1"
            )
            assert constant_records is not None, (
                "constant_records is required when assignments_file is None"
            )
            self._records = None
            # Behaviourally irrelevant here (the gather is constant), but it keeps
            # assignment_idx's wrap point -- and therefore the value checkpointed
            # in state_dict -- identical to the materialised path.
            self.num_records = int(constant_records)
            self._zero_words = np.zeros(B, dtype=np.uint64)
        else:
            with open(assignments_file, "rb") as f:
                header = f.read(32)
                magic, version, rec_size, flags, num_compartments, num_records, seed = (
                    struct.unpack("<8sBBHIQQ", header)
                )
                assert magic == b"TCASSIGN", "assignments magic mismatch"
                assert version == 1 and rec_size == 8, (
                    "unsupported assignments version/record size"
                )
                assert num_compartments == max_compartments, (
                    f"assignments num_compartments {num_compartments} != expected {max_compartments}"
                )
                self._records = np.frombuffer(f.read(), dtype=np.uint64)
            self.num_records = int(self._records.shape[0])
            self._zero_words = None

        # Multi-source mode: one _TokenStream or SyntheticTokenStream per compartment
        self._multi_source = compartment_filename_patterns is not None
        if self._multi_source:
            self._streams = []
            for i, pat in enumerate(compartment_filename_patterns):  # type: ignore[union-attr]
                if pat.startswith("synthetic:"):
                    mode = pat.split(":", 1)[1]
                    self._streams.append(SyntheticTokenStream(
                        mode=mode,
                        vocab_size=base_vocab_size,
                        seed=synthetic_seed + i * 1_000_000,
                        process_rank=process_rank,
                        frequencies=synthetic_frequencies,
                    ))
                else:
                    self._streams.append(
                        _TokenStream(pat, process_rank, T, shuffle_seed=shuffle_seed)
                    )
            self._stream: Optional[_TokenStream] = None
        else:
            self._stream = _TokenStream(
                filename_pattern, process_rank, T, shuffle_seed=shuffle_seed
            )
            self._streams: Optional[list[_TokenStream]] = None

        # Per-compartment merge-drop sets and expansion indices, built once.
        # Each compartment gets its own drop set seeded by (bpe_variant_seed,
        # compartment_id) and NOTHING else -- so a c=2 run's schemes are exactly a
        # c=8 run's first two, and cross-c comparisons are not confounded by the
        # compartments having different tokenizers.
        self._marker_base = compartment_marker_base
        self._shared_out = bool(shared_output_vocab)
        self._expand: Optional[list] = None
        # Unconsumed BASE tokens carried between examples. Compartment-agnostic,
        # so this splices nothing: it is the same contiguous stream either way.
        self._base_buf = np.empty(0, dtype=np.int64)
        self._base_chunk = max(4096, T * 4)
        if bpe_variant_expansion and bpe_variant_expansion > 1.0:
            from src import bpe_variants as bv
            table = bv.load_merge_table(bpe_variant_tokenizer)
            # Frequencies come from the TRAIN pattern, always -- never from this
            # loader's own data. Computing them per-loader gave the val loader a
            # DIFFERENT drop set than train (9,412 vs 9,436 merges for
            # compartment 0), i.e. the model would have been evaluated under a
            # tokenization it never trained on. Silent: both look like 2.0x.
            freq = bv.token_frequencies(
                bpe_variant_freq_pattern or filename_pattern, base_vocab_size)
            self._expand = []
            for ci in range(n_compartments):
                dropped, got = bv.select_dropped_merges(
                    freq, table, float(bpe_variant_expansion), bpe_variant_seed, ci)
                flat, off = bv.build_expansion_index(table, dropped, base_vocab_size)
                self._expand.append((flat, off))
                print0(f"[bpe-variant] compartment {ci}: {len(dropped):,} merges "
                       f"dropped, expansion {got:.4f}x")

        # assignment index start (strided by world size)
        self.assignment_idx = self.process_rank % max(1, self.num_records)

        # Pre-allocate pinned memory tensors to avoid expensive allocation per batch
        # Pinned memory allocation is ~200ms, so reusing saves significant time
        if self._pin_memory:
            self._x_buf = torch.empty((B, T), dtype=torch.long, pin_memory=True)
            self._y_buf = torch.empty((B, T), dtype=torch.long, pin_memory=True)
            self._cids_buf = torch.empty((B, T), dtype=torch.long, pin_memory=True)
        else:
            self._x_buf = None
            self._y_buf = None
            self._cids_buf = None

    @staticmethod
    def _decode_record(word: np.uint64) -> tuple[int, int, int]:
        w = int(word)
        kind = w & 0xFF
        src = (w >> 16) & 0xFFFF
        dst = (w >> 32) & 0xFFFF
        return kind, src, dst

    def state_dict(self) -> dict:
        if self._multi_source:
            return {
                "assignment_idx": self.assignment_idx,
                "streams": [s.state_dict() for s in self._streams],  # type: ignore[union-attr]
            }
        else:
            return {
                "assignment_idx": self.assignment_idx,
                **self._stream.state_dict(),  # type: ignore[union-attr]
            }

    def load_state_dict(self, state: dict) -> None:
        self.assignment_idx = state["assignment_idx"]
        if "streams" in state:
            # Multi-source checkpoint
            for s, sd in zip(self._streams, state["streams"]):  # type: ignore[arg-type]
                s.load_state_dict(sd)
        else:
            # Single-source checkpoint (backward compat)
            if self._multi_source:
                # Resuming a multi-source loader from an old single-source checkpoint:
                # reset all streams to initial state
                for s in self._streams:  # type: ignore[union-attr]
                    s.reset(self.process_rank, self.T)
            else:
                self._stream.load_state_dict(state)  # type: ignore[union-attr]

    def reset(self) -> None:
        if self._multi_source:
            for s in self._streams:  # type: ignore[union-attr]
                s.reset(self.process_rank, self.T)
        else:
            self._stream.reset(self.process_rank, self.T)  # type: ignore[union-attr]
        self.assignment_idx = self.process_rank % max(1, self.num_records)

    def _read_multi_source_tokens(
        self, B, is_trans, srcs, dsts, kinds, half, T
    ):
        """Read tokens from per-compartment streams, returning per-item arrays.

        Returns:
            comp_samples: dict mapping batch index -> np.ndarray[T] for compartment items
            trans_src_samples: dict mapping batch index -> np.ndarray[half] for translation src
            trans_dst_samples: dict mapping batch index -> np.ndarray[half] for translation dst
        """
        streams = self._streams  # type: ignore[union-attr]
        comp_samples: dict[int, np.ndarray] = {}
        trans_src_samples: dict[int, np.ndarray] = {}
        trans_dst_samples: dict[int, np.ndarray] = {}
        for b in range(B):
            if is_trans[b]:
                src_tokens = streams[int(srcs[b])].read_tokens(half).astype(np.int64)
                dst_tokens = streams[int(dsts[b])].read_tokens(half).astype(np.int64)
                trans_src_samples[b] = src_tokens
                trans_dst_samples[b] = dst_tokens
            else:
                comp_samples[b] = streams[int(srcs[b])].read_tokens(T).astype(np.int64)
        return comp_samples, trans_src_samples, trans_dst_samples

    def next_batch(self):
        B = self.B
        T = self.T
        half = T // 2
        assert T % 2 == 0 and half >= 2, "block_size must be even and >= 4"

        # Reuse pre-allocated pinned-memory tensors if available (avoids ~200ms alloc overhead)
        if self._x_buf is not None:
            x_t = self._x_buf
            y_t = self._y_buf
            cids_t = self._cids_buf
        else:
            x_t = torch.empty((B, T), dtype=torch.long)
            y_t = torch.empty((B, T), dtype=torch.long)
            cids_t = torch.empty((B, T), dtype=torch.long)

        x_np = x_t.numpy()
        y_np = y_t.numpy()
        cid_np = cids_t.numpy()

        # Vectorized decode of B assignment records (strided by num_processes)
        rec_indices = (
            self.assignment_idx + self.num_processes * np.arange(B, dtype=np.int64)
        ) % max(1, self.num_records)
        # _records is None at c=1, where the gather returns 0 for every index.
        # _zero_words is preallocated at length self.B, which is exactly len(rec_indices).
        words = (
            self._zero_words if self._records is None else self._records[rec_indices]
        )
        w = words.astype(np.uint64, copy=False)
        kinds = (w & np.uint64(0xFF)).astype(np.int64, copy=False)
        srcs = ((w >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.int64, copy=False)
        dsts = ((w >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.int64, copy=False)

        is_trans = kinds == 1
        idx_trans = np.nonzero(is_trans)[0]
        idx_comp = np.nonzero(~is_trans)[0]

        if B == 0:
            return x_t, y_t, cids_t

        if self._multi_source:
            # Multi-source: read from per-compartment streams
            comp_samples, trans_src_samples, trans_dst_samples = (
                self._read_multi_source_tokens(B, is_trans, srcs, dsts, kinds, half, T)
            )
            self._fill_batch_multi_source(
                x_np, y_np, cid_np, idx_comp, idx_trans, srcs, dsts,
                comp_samples, trans_src_samples, trans_dst_samples,
                half, T,
            )
        else:
            if self._expand is not None:
                # Each example consumes a VARIABLE number of base tokens, because
                # its compartment's segmentation decides how far T tokens reach.
                # Read until the budget is filled, per example, advancing one
                # shared cursor -- so examples stay on disjoint, contiguous text.
                samples = np.zeros((B, T), dtype=np.int64)
                from src import bpe_variants as bv
                for b in range(B):
                    flat, off = self._expand[int(srcs[b])]
                    need = int(half if is_trans[b] else T)
                    got = np.empty(0, dtype=np.int64)
                    while len(got) < need:
                        if self._base_buf.size == 0:
                            self._base_buf = self._stream.read_tokens(
                                self._base_chunk
                            ).astype(np.int64, copy=False)
                        piece, used = bv.expand(
                            self._base_buf, flat, off, limit=need - len(got))
                        got = np.concatenate([got, piece])
                        # `used` is how many BASE tokens that consumed. Keep the
                        # rest: base tokens carry no compartment identity, so an
                        # unconsumed remainder is usable by whichever example
                        # comes next, whatever compartment it belongs to. Without
                        # this the stream advances by the full chunk each time and
                        # the run silently reads the corpus at ~2x the rate,
                        # discarding half the text.
                        self._base_buf = self._base_buf[used:]
                    samples[b, :need] = got[:need]
                # Same fill as the normal path: expansion happens BEFORE the
                # compartment offset, so _fill_batch_single_source is untouched.
                tokens_batch = samples.reshape(-1)
                starts = np.arange(B, dtype=np.int64) * T
            else:
                # Single-source: read contiguous tokens from one stream
                size_per_b = np.where(is_trans, half, T).astype(np.int64, copy=False)
                starts = np.zeros(B, dtype=np.int64)
                if B > 1:
                    starts[1:] = np.cumsum(size_per_b[:-1], dtype=np.int64)
                total_needed = int(starts[-1] + size_per_b[-1])
                tokens_batch = self._stream.read_tokens(total_needed).astype(np.int64, copy=False)  # type: ignore[union-attr]
            self._fill_batch_single_source(
                x_np, y_np, cid_np, idx_comp, idx_trans, srcs, dsts,
                tokens_batch, starts, half, T,
            )

        # advance assignment pointer for next batch
        self.assignment_idx = (
            self.assignment_idx + self.num_processes * B
        ) % self.num_records

        # Optional runtime invariants for debugging. Enable with TC_DEBUG_LOADER=1
        if os.environ.get("TC_DEBUG_LOADER", "") == "1":
            assert (
                x_np.shape == (B, T) and y_np.shape == (B, T) and cid_np.shape == (B, T)
            )
            assert (y_np[:, -1] == -1).all()
            if (
                B > 0
                and T > 1
                and not (self.permute_tokens and not self.permute_inputs)
                and not self._multi_source
            ):
                assert (y_np[:, :-1] == x_np[:, 1:]).all()
            if idx_trans.size > 0:
                assert (x_np[idx_trans, 0] == self.translation_token_id).all()
                assert (x_np[idx_trans, half] == self.translation_token_id).all()
            # Basic bounds checking of vocab id space
            if self.permute_tokens:
                # Permuted mode uses base vocab ids and a single translation token at base_vocab_size
                assert (x_np >= 0).all()
                assert (x_np < (self.base_vocab_size + 1)).all()
            else:
                max_vocab = self.base_vocab_size * self.max_compartments + 1
                assert (x_np >= 0).all() and (x_np < max_vocab).all()

        return x_t, y_t, cids_t

    def _fill_batch_single_source(
        self, x_np, y_np, cid_np, idx_comp, idx_trans, srcs, dsts,
        tokens_batch, starts, half, T,
    ):
        """Fill batch arrays from a single contiguous token stream (original behavior)."""
        # Translation group: gather half-length segments in b-order
        if idx_trans.size > 0:
            m = int(idx_trans.size)
            base_idx_half = np.arange(half, dtype=np.int64)[None, :]
            trans_starts = starts[idx_trans][:, None]
            samples = tokens_batch[trans_starts + base_idx_half]

            self._fill_translation_rows(
                x_np, y_np, cid_np, idx_trans, srcs, dsts,
                samples, samples,  # same content for src and dst halves
                half, T,
            )

        # Compartment group: gather T-length segments in b-order
        if idx_comp.size > 0:
            base_idx_T = np.arange(T, dtype=np.int64)[None, :]
            comp_starts = starts[idx_comp][:, None]
            samples = tokens_batch[comp_starts + base_idx_T]
            self._fill_compartment_rows(x_np, y_np, cid_np, idx_comp, srcs, samples, T)

    def _fill_batch_multi_source(
        self, x_np, y_np, cid_np, idx_comp, idx_trans, srcs, dsts,
        comp_samples, trans_src_samples, trans_dst_samples, half, T,
    ):
        """Fill batch arrays from per-compartment token streams."""
        # Translation group
        if idx_trans.size > 0:
            m = int(idx_trans.size)
            # Stack per-item arrays into [m, half] arrays
            src_stacked = np.stack([trans_src_samples[int(b)] for b in idx_trans])
            dst_stacked = np.stack([trans_dst_samples[int(b)] for b in idx_trans])
            self._fill_translation_rows(
                x_np, y_np, cid_np, idx_trans, srcs, dsts,
                src_stacked, dst_stacked,
                half, T,
            )

        # Compartment group
        if idx_comp.size > 0:
            samples = np.stack([comp_samples[int(b)] for b in idx_comp])
            self._fill_compartment_rows(x_np, y_np, cid_np, idx_comp, srcs, samples, T)

    def _fill_translation_rows(
        self, x_np, y_np, cid_np, idx_trans, srcs, dsts,
        src_samples, dst_samples, half, T,
    ):
        """Fill translation rows into batch arrays.

        src_samples: [m, half] tokens for the source half
        dst_samples: [m, half] tokens for the destination half
        In single-source mode, src_samples and dst_samples are the same array.
        In multi-source mode, they contain different content from different streams.
        """
        m = int(idx_trans.size)
        seq_in = np.empty((m, T), dtype=np.int64)
        seq_out = np.empty((m, T), dtype=np.int64)
        cid_tr = np.empty((m, T), dtype=np.int64)

        # Set translation token positions and cids
        seq_in[:, 0] = self.translation_token_id
        seq_in[:, half] = self.translation_token_id
        seq_out[:, 0] = self.translation_token_id
        seq_out[:, half] = self.translation_token_id
        cid_tr[:, 0] = srcs[idx_trans]
        cid_tr[:, half] = dsts[idx_trans]

        src_slice = src_samples[:, : half - 1]  # [m, half-1]
        dst_slice = dst_samples[:, : half - 1]  # [m, half-1]

        if self.permute_tokens:
            perms = cast(np.ndarray, self._permutations)
            src_comp = srcs[idx_trans][:, None]  # [m, 1]
            dst_comp = dsts[idx_trans][:, None]  # [m, 1]
            perm_src = perms[
                np.broadcast_to(src_comp, src_slice.shape), src_slice
            ]
            perm_dst = perms[
                np.broadcast_to(dst_comp, dst_slice.shape), dst_slice
            ]
            if self.permute_inputs:
                seq_in[:, 1:half] = perm_src
                seq_in[:, half + 1 :] = perm_dst
            else:
                seq_in[:, 1:half] = src_slice
                seq_in[:, half + 1 :] = dst_slice
            seq_out[:, 1:half] = perm_src
            seq_out[:, half + 1 :] = perm_dst
        else:
            if self._tying_remap is not None:
                remap = self._tying_remap
                src_comp = srcs[idx_trans]
                dst_comp = dsts[idx_trans]
                seq_in[:, 1:half] = remap[src_comp[:, None], src_slice]
                seq_in[:, half + 1 :] = remap[dst_comp[:, None], dst_slice]
            else:
                base = self.base_vocab_size
                src_off = srcs[idx_trans][:, None] * base
                dst_off = dsts[idx_trans][:, None] * base
                seq_in[:, 1:half] = src_slice + src_off
                seq_in[:, half + 1 :] = dst_slice + dst_off
            seq_out[:, 1:half] = seq_in[:, 1:half]
            seq_out[:, half + 1 :] = seq_in[:, half + 1 :]

        cid_tr[:, 1:half] = srcs[idx_trans][:, None]
        cid_tr[:, half + 1 :] = dsts[idx_trans][:, None]

        y_tr = np.empty_like(seq_out)
        y_tr[:, :-1] = seq_out[:, 1:]
        y_tr[:, -1] = -1

        x_np[idx_trans] = seq_in
        y_np[idx_trans] = y_tr
        cid_np[idx_trans] = cid_tr

    def _fill_compartment_rows(
        self, x_np, y_np, cid_np, idx_comp, srcs, samples, T,
    ):
        """Fill compartment rows into batch arrays."""
        if self.permute_tokens:
            perms = cast(np.ndarray, self._permutations)
            src_comp = srcs[idx_comp][:, None]
            perm_samples = perms[np.broadcast_to(src_comp, samples.shape), samples]
            if self.permute_inputs:
                x_comp = perm_samples
            else:
                x_comp = samples
            y_comp = np.empty_like(perm_samples)
            y_comp[:, :-1] = perm_samples[:, 1:]
            y_comp[:, -1] = -1
        else:
            if self._tying_remap is not None:
                remap = self._tying_remap
                src_comp = srcs[idx_comp]
                x_comp = remap[src_comp[:, None], samples]
            else:
                base = self.base_vocab_size
                src_off = srcs[idx_comp][:, None] * base
                x_comp = samples + src_off
            if self._marker_base is not None:
                # Prefix each sequence with its compartment's marker id. The last
                # content token is dropped so the length stays exactly T, which
                # costs 0.1% of a 1024-token window. y is derived from x below,
                # so the marker's target is the first content token -- the model
                # is asked to predict from the marker, which is the point.
                mk = (self._marker_base + srcs[idx_comp][:, None]).astype(x_comp.dtype)
                x_comp = np.concatenate([mk, x_comp[:, :-1]], axis=1)
            # c-in/1-out: predict the SHARED base id, so every compartment maps
            # its private input vocabulary onto one output space. `samples` is
            # already the un-offset base ids; the marker branch above shifted the
            # inputs by one, so shift the targets to match or they misalign by a
            # position -- which would look like a merely harder task rather than
            # a broken one.
            src = samples if self._shared_out else x_comp
            if self._shared_out and self._marker_base is not None:
                src = np.concatenate([samples[:, :1], samples[:, :-1]], axis=1)
            y_comp = np.empty_like(x_comp)
            y_comp[:, :-1] = src[:, 1:]
            y_comp[:, -1] = -1

        x_np[idx_comp] = x_comp
        y_np[idx_comp] = y_comp
        cid_np[idx_comp, :] = srcs[idx_comp][:, None]


# Where data/, out/ and cache/ live. Resolution order:
#   1. TC_STORAGE_ROOT, if set -- explicit wins.
#   2. The historical pccfs2 path, if it exists on this host.
#   3. The directory containing this file.
#
# (3) exists because this repo runs on machines that do NOT mount pccfs2 (ORC
# uses /grphome + /nobackup). Hardcoding the pccfs2 path made every job there
# die instantly with "STORAGE_ROOT ... is not a directory" -- 13 runs failed
# that way, each in under 10s, and because the pool only records "FAILED rc=1"
# the cause was invisible until a config was run by hand. Falling back to the
# repo directory is correct on any host: data/, out/ and cache/ are repo-relative
# everywhere they are used.
_DEFAULT_STORAGE = "/mnt/pccfs2/backed_up/vin/dev/translation-compression"
STORAGE_ROOT = os.environ.get("TC_STORAGE_ROOT") or (
    _DEFAULT_STORAGE
    if os.path.isdir(_DEFAULT_STORAGE)
    else os.path.dirname(os.path.abspath(__file__))
)
if not os.path.isdir(STORAGE_ROOT):
    raise SystemExit(
        f"STORAGE_ROOT={STORAGE_ROOT!r} is not a directory. "
        f"Set TC_STORAGE_ROOT to the correct path for this host."
    )

ROLLING_CHECKPOINT_INTERVAL = 1000


DATALOADER_STATE = "dataloader.pt"          # pre-2026-08-15 layout: rank 0 only


def _dl_state_name(rank: int) -> str:
    return f"dataloader_rank{int(rank)}.pt"


def save_dataloader_state(ckpt_dir, train_loader, val_loader, rank: int) -> None:
    """Write THIS rank's dataloader state. Called by every rank, not just master.

    Every rank must persist its own position. Before 2026-08-15 only rank 0's
    state was written and every rank restored it, so after any resume all ranks
    replayed rank 0's stream: distinct sequences per optimizer step fell from
    2048 to 256 while the LR stayed tuned for 2048. It is silent -- training
    continues and the loss curve stays plausible.

    Rank 0's state cannot be used to reconstruct rank r's, either. Ranks advance
    in lockstep ONLY at translation_ratio=0; once tr>0 a translation example
    consumes T/2 tokens instead of T, and which examples are translations is a
    per-rank property of the assignments, so the streams drift apart by a
    data-dependent amount. Hence one file per rank rather than an offset.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {"train": train_loader.state_dict(), "rank": int(rank)}
    if val_loader is not None:
        state["val"] = val_loader.state_dict()
    torch.save(state, os.path.join(ckpt_dir, _dl_state_name(rank)))


def load_dataloader_state(ckpt_dir, rank: int):
    """This rank's saved state, or None.

    Falls back to the pre-fix single-file layout, which only ever holds rank 0's
    position. Resuming every rank from it is what the fix exists to prevent, so
    this refuses rather than silently reproducing the bug; set
    TC_ALLOW_RANK0_DATALOADER_RESUME=1 to accept it for a run whose checkpoints
    predate the fix and whose remaining budget makes a restart worse.
    """
    per_rank = os.path.join(ckpt_dir, _dl_state_name(rank))
    if os.path.exists(per_rank):
        return torch.load(per_rank, map_location="cpu", weights_only=False)

    legacy = os.path.join(ckpt_dir, DATALOADER_STATE)
    if not os.path.exists(legacy):
        return None
    if os.environ.get("TC_ALLOW_RANK0_DATALOADER_RESUME") == "1":
        print0(
            f"WARNING: {ckpt_dir} predates per-rank dataloader state; every rank "
            f"will resume from rank 0's position and train on IDENTICAL data "
            f"(TC_ALLOW_RANK0_DATALOADER_RESUME=1)."
        )
        return torch.load(legacy, map_location="cpu", weights_only=False)
    raise RuntimeError(
        f"{ckpt_dir} has only {DATALOADER_STATE} (rank 0's position). Resuming "
        f"every rank from it makes all ranks train on identical data -- the bug "
        f"fixed on 2026-08-15. Options:\n"
        f"  - restart this run from a checkpoint written after the fix\n"
        f"  - TC_ALLOW_RANK0_DATALOADER_RESUME=1 to accept the duplication\n"
        f"  - delete {legacy} to resume with fresh per-rank stream positions "
        f"(model and optimizer state are kept; only data position resets)"
    )


def _save_rolling_checkpoint(out_dir, raw_model, optimizer, train_loader,
                             val_loader, iter_num, best_val_loss,
                             rank: int = 0, master: bool = True):
    """Save a rolling 'latest' checkpoint for preemption resilience.

    Overwrites a single _rolling/ directory each time. trainer_state.json is
    written last so its presence signals a complete checkpoint.
    """
    ck_root = os.path.join(out_dir, "checkpoints")
    rolling_dir = os.path.join(ck_root, "_rolling")
    os.makedirs(rolling_dir, exist_ok=True)
    # EVERY rank records its own stream position; only master writes the weights.
    save_dataloader_state(rolling_dir, train_loader, val_loader, rank)
    if not master:
        return
    torch.save(raw_model.state_dict(), os.path.join(rolling_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(rolling_dir, "optimizer.pt"))
    # Write trainer_state.json LAST — its existence signals a complete checkpoint
    with open(os.path.join(rolling_dir, "trainer_state.json"), "w") as f:
        json.dump({"iter_num": iter_num, "best_val_loss": float(best_val_loss)}, f)
    # Update latest symlink to point to _rolling
    latest = os.path.join(ck_root, "latest")
    tmp_link = latest + f".tmp.{os.getpid()}"
    os.symlink("_rolling", tmp_link)
    os.replace(tmp_link, latest)


def main(config: JobConfig) -> None:
    # Register SIGUSR1 handler for graceful preemption (GPU rescheduling)
    signal.signal(signal.SIGUSR1, _handle_preempt_signal)
    _preempt_requested.clear()

    # Apply size tier overrides (if provided) and mirror assignment_seed to training.seed
    config = apply_size_tier(config)
    # Auto-configure batch/grad_accum for bpe16384 vocab if not explicitly set
    config = apply_bpe16384_batch_config(config)
    # Scale batch for available VRAM (preserves effective batch size)
    if torch.cuda.is_available():
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        _vram = torch.cuda.get_device_properties(_local_rank).total_memory
        config = scale_batch_for_vram(config, _vram)
        print0(f"VRAM: {_vram / 1024**3:.0f}GB -> batch_size={config.training.batch_size}, grad_accum={config.training.gradient_accumulation_steps}")
    config = replace(
        config,
        experiment=replace(config.experiment, assignment_seed=config.training.seed),
    )
    # If uniform_seed is 0 (default), inherit from training.seed for simpler sweep configs
    if config.data.uniform_seed == 0:
        config = replace(
            config,
            data=replace(config.data, uniform_seed=config.training.seed),
        )
    # -----------------------------------------------------------------------------
    # Unpack config into local variables (matches original script expectations)
    # wandb logging
    wandb_log = config.logging.wandb_log
    wandb_project = config.logging.wandb_project
    wandb_run_name = config.logging.wandb_run_name
    wandb_group = config.logging.wandb_group
    wandb_notes = config.logging.wandb_notes

    # Check for duplicate completed runs before expensive setup
    if wandb_log:
        dup_run_id = check_duplicate_run(config, wandb_project, wandb_group)
        if dup_run_id:
            print0(
                f"[dedupe] Skipping: found completed run with same config: {dup_run_id}"
            )
            print0(
                f"[dedupe] View at: https://wandb.ai/pccl/{wandb_project}/runs/{dup_run_id}"
            )
            return

    # I/O: structured run directory under hardcoded storage root
    run_id = os.environ.get("RUN_ID", uuid.uuid4().hex[:8])
    out_root = os.path.join(STORAGE_ROOT, "out")
    if os.environ.get("OUT_DIR"):
        out_dir = os.environ["OUT_DIR"]
        project_dir = os.path.join(out_root, slug(wandb_project))
        group_dir = os.path.join(project_dir, slug(wandb_group or "default"))
    else:
        project_dir, group_dir, out_dir = run_dirs(
            out_root,
            wandb_project,
            wandb_group,
            wandb_run_name,
            config,
            run_id,
        )
    eval_interval = config.training.eval_interval
    log_interval = config.training.log_interval
    eval_iters = config.training.eval_iters
    eval_only = config.training.eval_only
    always_save_checkpoint = config.training.always_save_checkpoint
    init_from = config.init.init_from
    # data
    train_bin = config.data.train_bin
    val_bin = config.data.val_bin
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    batch_size = config.training.batch_size
    block_size = config.model.block_size
    # model
    dropout = config.model.dropout
    # adamw optimizer
    learning_rate = config.optimizer.learning_rate
    max_iters = config.training.max_iters
    weight_decay = config.optimizer.weight_decay
    beta1 = config.optimizer.beta1
    beta2 = config.optimizer.beta2
    grad_clip = config.optimizer.grad_clip
    # learning rate decay settings
    warmup_iters = config.lr.warmup_iters
    decay_lr = config.lr.decay_lr
    schedule = getattr(config.lr, "schedule", "legacy")
    decay_start_iter = getattr(config.lr, "decay_start_iter", 0)
    decay_end_iter = getattr(config.lr, "decay_end_iter", 0)
    lr_decay_iters = config.lr.lr_decay_iters
    min_lr = config.lr.min_lr
    lr_schedule.validate(
        schedule=schedule,
        warmup_iters=warmup_iters,
        decay_start_iter=decay_start_iter,
        decay_end_iter=decay_end_iter,
        peak=learning_rate,
        min_lr=min_lr,
    )
    # DDP settings
    backend = config.distributed.backend
    # system
    device = config.system.device
    if config.system.dtype == "auto":
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    else:
        dtype = config.system.dtype
    compile = config.system.compile
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_rank = None
    if ddp:
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        # Bind this rank to its GPU *before* init_process_group. Otherwise NCCL
        # guesses the device from the global rank ("Guessing device ID based on
        # global rank. This can cause a hang...") and the first barrier can
        # deadlock — that cost us a full 24h B200 node-day on job 12901827,
        # which sat in the barrier and never ran a single iteration.
        torch.cuda.set_device(device)
        init_process_group(backend=backend, device_id=torch.device(device))
        ddp_rank = int(os.environ["RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    active_run_lock = None
    if master_process:
        active_run_lock = ActiveRunLock(
            out_dir,
            run_id=run_id,
            config_path=os.environ.get("CONFIG_PATH"),
            config_identity=cfg_hash(config),
        ).acquire()
    tokens_per_iter = (
        gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    )
    print(
        f"tokens per iteration will be: gradient_accumulation_steps={gradient_accumulation_steps} * ddp_world_size={ddp_world_size} * batch_size={batch_size} * block_size={block_size} = {tokens_per_iter:,}"
    )
    # Effective global batch size across all processes and gradient accumulation
    effective_batch_size = gradient_accumulation_steps * ddp_world_size * batch_size

    # Restrict checkpointing to these specific global steps only (reverse engineering
    # Morris figure)
    checkpoint_steps = {
        # Dense early-training cadence for resolution where curves diverge most.
        50, 100, 150, 200, 250, 300, 400, 500, 600, 750,
        1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000,
        5000, 6000, 7000, 8000, 9000, 10000,
        12000, 14000,
        # Sparser later
        20000, 29000,
        # 35k-45k fill the gap between 29000 and 50000: at 2.1M tok/step a 100B
        # run ends at 47,684 steps, so without these the last 39B tokens produce
        # no checkpoint at all -- unevaluable, and no branch point for an anneal.
        35000, 40000, 45000,
        50000, 60000, 80000, 100000, 120000, 150000, 200000, 240000,
        300000, 350000, 400000, 500000, 700000, 1000000,
        # Extension for the n=8 tr=0.1 epoch run (~1 epoch ≈ 2.96M steps)
        1250000, 1500000, 1750000, 2000000, 2250000, 2500000, 2750000, 2950000,
        # Dense extension for 1B tr=0.056 (1M → 2M), target-half loss curve.
        # Every 100k + 2 early resume-check points.
        1050000, 1100000, 1200000, 1300000, 1400000, 1600000, 1700000, 1800000, 1900000,
    }

    # Which of those also carry optimizer + dataloader state, making them
    # resumable and forkable. Everything else stays weights-only; see
    # Training.full_state_at_tokens for why the list is short.
    checkpoint_naming = getattr(config.training, "checkpoint_naming", "step")
    full_state_at = tuple(getattr(config.training, "full_state_at_tokens", ()) or ())
    # A decay child keeps its checkpoints under checkpoints/annealed/ so that a
    # bare glob for stable points cannot pick them up — annealed losses sit
    # below the stable curve and must never join that series by accident.
    is_anneal_child = schedule == "wsd" and decay_end_iter > 0
    if is_anneal_child:
        # Annealed checkpoints are terminal: you would not extend past an anneal
        # (that is the re-warm discontinuity WSD exists to avoid), so nothing
        # here needs optimizer state. _rolling still covers preemption during
        # the decay itself.
        full_state_iters = frozenset()
    else:
        full_state_iters = ckpt_utils.full_state_steps(
            full_state_at, tokens_per_iter, config.training.max_iters
        )
        # A branch point that is not a scheduled checkpoint never gets written.
        checkpoint_steps = checkpoint_steps | full_state_iters

    def _checkpoint_dir(ck_root, it):
        """Path for the named checkpoint at iteration `it`."""
        if checkpoint_naming == "tokens":
            name = ckpt_utils.tok_dirname(it * tokens_per_iter)
        else:
            name = f"step-{it:06d}"
        if is_anneal_child:
            return os.path.join(ck_root, ckpt_utils.ANNEALED, name)
        return os.path.join(ck_root, name)

    assignments_path = os.path.join(out_dir, "assignments.bin")
    permutations_path = os.path.join(out_dir, "permutations.npy")
    training_seed = config.training.seed + seed_offset
    # Base seed without DDP rank offset — used for assignment/permutation cache paths
    # so all ranks agree on the same file (only master generates it).
    base_seed = config.training.seed
    # Determine data source
    use_pretokenized = config.data.source == "pretokenized"

    # If using pretokenized data, compute deterministic cache paths for assignments/permutations
    if use_pretokenized:
        # Assignment cache location. Defaults to the shared storage root, so
        # nothing changes unless TC_ASSIGNMENT_CACHE is set explicitly.
        #
        # WHY THE OVERRIDE EXISTS. A multi-compartment 1M-step run needs a
        # 16.4GB assignment file (2.048e9 records x 8B). The shared mount is
        # NFS with rsize/wsize=32768 -- a 32KB transfer unit, 32x below the
        # modern default -- so that write costs 524,288 RPCs, and was measured
        # at ~76 MiB/min (about 3.6 hours) while the server was degraded.
        # Pointing this at node-local disk turns it into a local write. The
        # file is derived data, keyed by (weights, total_examples, seed), so a
        # per-node copy is safe: any node regenerates an identical file.
        cache_root = os.environ.get("TC_ASSIGNMENT_CACHE") or os.path.join(STORAGE_ROOT, "cache")
        exp = config.experiment
        if exp.max_compartments is None:
            raise ValueError("experiment.max_compartments is required")
        max_compartments_int = cast(int, exp.max_compartments)
        # The assignment array length is the CACHE KEY, and
        # _largest_remainder_allocations re-partitions the whole array when it
        # changes -- so deriving it from max_iters means extending a run silently
        # re-randomises every example's compartment. Measured on the 30B->100B
        # c=8 extension: 1/8 of assignments survived, val loss +0.19 nats.
        _needed = config.training.max_iters * effective_batch_size
        _horizon = int(getattr(config.experiment, "assignment_horizon_examples", 0) or 0)
        if _horizon and _horizon < _needed:
            raise ValueError(
                f"experiment.assignment_horizon_examples ({_horizon:,}) is smaller "
                f"than this run needs ({_needed:,} = max_iters "
                f"{config.training.max_iters:,} x batch {effective_batch_size:,}). "
                f"Raise it; lowering the horizon re-randomises every assignment."
            )
        total_examples = _horizon or _needed

        # c=1 needs no assignment array at all. One compartment means exactly one
        # category (compute_weights_map rejects translation_ratio > 0 at n < 2), so
        # every record encodes compartment 0 and the whole array is the constant 0.
        # Nothing consumes it: the loader's per-batch gather is constant, and the
        # only reader outside training -- experiment/eval_utils.token_counts_at_examples
        # -- has no callers. Skipping it saves 8 B/example of RAM on every rank.
        skip_assignments = exp.n_compartments == 1

        # Format float safely for filenames
        def _fmt_float(x: float) -> str:
            s = f"{x:.6g}".rstrip("0").rstrip(".")
            return s.replace(".", "p") if "." in s else s

        # Build description from inputs to assignments creation
        assignments_desc = (
            f"n{exp.n_compartments}_t{_fmt_float(max(0.0, float(exp.translation_ratio)))}_"
            f"m{exp.translation_ratio_mode}_"
            f"sc{exp.compartment_scaling}_total{int(total_examples)}_"
            f"maxc{max_compartments_int}_seed{int(base_seed)}"
            + ("" if getattr(exp, "assignment_order", "lcg") == "lcg" else "_hash")
        )
        # Point assignments_path to cached file. None at c=1 -- the loader
        # synthesizes the constant records rather than reading them.
        assignments_path = (
            None
            if skip_assignments
            else os.path.join(cache_root, f"assignments_{assignments_desc}.bin")
        )

        # If permuting tokens per compartment, also compute cached permutations path
        if exp.permute_tokens_per_compartment:
            if config.model.vocab_size is not None:
                base_vocab_int = cast(int, config.model.vocab_size)
                perms_desc = f"basev{base_vocab_int}_maxc{max_compartments_int}_seed{int(base_seed)}"
                permutations_path = os.path.join(
                    cache_root, f"permutations_{perms_desc}.npy"
                )

    if master_process:
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
        write_meta(out_dir, config)
        append_to_experiment_log(project_dir, group_dir, out_dir, config)
        # Generate assignments/permutations only for pretokenized source
        if use_pretokenized:
            exp = config.experiment
            try:
                if exp.max_compartments is None:
                    raise ValueError("experiment.max_compartments is required")
                max_compartments_int = cast(int, exp.max_compartments)
                # Horizon decoupled from max_iters; see
                # Experiment.assignment_horizon_examples.
                total_examples = int(
                    getattr(config.experiment, "assignment_horizon_examples", 0) or 0
                ) or (config.training.max_iters * effective_batch_size)
                # c=1 (skip_assignments) has no array to build: assignments_path
                # is None and the loader synthesizes the constant records. The
                # permutation block below still runs -- 10 configs pair c=1 with
                # permute_tokens_per_compartment and would break without it.
                if not skip_assignments:
                    # Ensure cache directory exists
                    cache_root = os.path.dirname(assignments_path)
                    os.makedirs(cache_root, exist_ok=True)
                    # Refuse to fill the volume. These files are 8 bytes/record, so a
                    # 1M-step multi-compartment run is 16.4GB -- enough to exhaust a
                    # node-local disk that is shared with other people's data. Fail
                    # loudly BEFORE writing rather than ENOSPC-ing hours in, which
                    # would leave a truncated .tmp and a held lock behind.
                    if not os.path.exists(assignments_path):
                        need = 32 + int(total_examples) * 8
                        margin = 20 * 1024**3
                        free = shutil.disk_usage(cache_root).free
                        if free < need + margin:
                            raise SystemExit(
                                f"refusing to write assignments to {cache_root}: need "
                                f"{need/1024**3:.1f}GB + {margin/1024**3:.0f}GB margin but only "
                                f"{free/1024**3:.1f}GB free. Point TC_ASSIGNMENT_CACHE at a "
                                f"volume with room, or free space here."
                            )
                    # Assignments: write only if not already cached, with simple cross-process locking
                    if not os.path.exists(assignments_path):
                        lock_path = assignments_path + ".lock"
                        tmp_path = assignments_path + f".tmp.{os.getpid()}"
                        acquired = False
                        while True:
                            try:
                                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                                os.close(fd)
                                acquired = True
                                break
                            except FileExistsError:
                                # Another process is writing; wait until file appears
                                if os.path.exists(assignments_path):
                                    break
                                time.sleep(0.1)
                        if acquired:
                            try:
                                write_assignments(
                                    tmp_path,
                                    weights_map=compute_weights_map(
                                        n=exp.n_compartments,
                                        t=max(0.0, float(exp.translation_ratio)),
                                        scaling=exp.compartment_scaling,
                                        mode=exp.translation_ratio_mode,
                                    ),
                                    total_examples=total_examples,
                                    max_compartments=max_compartments_int,
                                    seed=base_seed,
                                    order=getattr(config.experiment, "assignment_order", "lcg"),
                                    no_shuffle=False,
                                )
                                os.replace(tmp_path, assignments_path)
                                print0(
                                    f"Wrote assignments to {assignments_path} with total_examples={total_examples:,}"
                                )
                            finally:
                                try:
                                    if os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                                except Exception:
                                    pass
                                try:
                                    os.remove(lock_path)
                                except Exception:
                                    pass
                        else:
                            print0(f"Using cached assignments at {assignments_path}")
                    else:
                        print0(f"Using cached assignments at {assignments_path}")
                # If enabled, also create deterministic per-compartment permutations of base tokens
                if exp.permute_tokens_per_compartment:
                    if config.model.vocab_size is None:
                        raise ValueError(
                            "model.vocab_size (base) must be set to create permutations"
                        )
                    base_vocab_int = cast(int, config.model.vocab_size)
                    # Permutations: write only if not already cached, with simple cross-process locking
                    if not os.path.exists(permutations_path):
                        lock_path = permutations_path + ".lock"
                        tmp_path = permutations_path + f".tmp.{os.getpid()}"
                        acquired = False
                        while True:
                            try:
                                fd = os.open(
                                    lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR
                                )
                                os.close(fd)
                                acquired = True
                                break
                            except FileExistsError:
                                # Another process is writing; wait until file appears
                                if os.path.exists(permutations_path):
                                    break
                                time.sleep(0.1)
                        if acquired:
                            try:
                                # Use SeedSequence to spawn deterministic child RNGs per compartment
                                ss = np.random.SeedSequence(
                                    int(training_seed) & 0xFFFFFFFFFFFFFFFF
                                )
                                child_seeds = ss.spawn(max_compartments_int)
                                # Allocate permutations for all compartments up to max_compartments
                                perms = np.empty(
                                    (int(max_compartments_int), base_vocab_int),
                                    dtype=np.int64,
                                )
                                for c, child_ss in enumerate(child_seeds):
                                    gen = np.random.Generator(np.random.PCG64(child_ss))
                                    perms[c] = gen.permutation(base_vocab_int).astype(
                                        np.int64
                                    )
                                # Write using a file handle to avoid numpy appending an extra .npy
                                with open(tmp_path, "wb") as f:
                                    np.save(f, perms)
                                    f.flush()
                                    os.fsync(f.fileno())
                                os.replace(tmp_path, permutations_path)
                                print0(
                                    f"Wrote per-compartment permutations to {permutations_path} with shape {perms.shape}"
                                )
                            finally:
                                try:
                                    if os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                                except Exception:
                                    pass
                                try:
                                    os.remove(lock_path)
                                except Exception:
                                    pass
                        else:
                            print0(f"Using cached permutations at {permutations_path}")
                    else:
                        print0(f"Using cached permutations at {permutations_path}")
            except Exception as e:
                print0(f"Failed generating assignments: {e}")
                raise
    # Ensure all processes wait for assignments/permutations if using pretokenized data
    if ddp and use_pretokenized:
        torch.distributed.barrier()

    # --- Token tying: compute frequency cache and tied mask ---
    tied_token_mask: Optional[np.ndarray] = None
    if config.experiment.token_tying_mode != "none" and use_pretokenized:
        import hashlib

        _tying_vocab = config.model.vocab_size
        if _tying_vocab is None:
            raise ValueError("model.vocab_size must be set for token tying")

        freq_shards = config.experiment.token_tying_freq_shards
        data_hash = hashlib.sha256(train_bin.encode()).hexdigest()[:12]
        freq_cache_name = f"token_freqs_v{_tying_vocab}_d{data_hash}_s{freq_shards}.npy"
        freq_cache_dir = os.path.join(STORAGE_ROOT, "cache")
        freq_cache_path = os.path.join(freq_cache_dir, freq_cache_name)

        # Compute/load frequency cache (lock-based write-once, master only writes)
        if master_process:
            os.makedirs(freq_cache_dir, exist_ok=True)
            if not os.path.exists(freq_cache_path):
                lock_path = freq_cache_path + ".lock"
                tmp_path = freq_cache_path + f".tmp.{os.getpid()}"
                acquired = False
                while True:
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                        os.close(fd)
                        acquired = True
                        break
                    except FileExistsError:
                        if os.path.exists(freq_cache_path):
                            break
                        time.sleep(0.1)
                if acquired:
                    try:
                        freqs = compute_token_frequencies(
                            train_bin, _tying_vocab, max_shards=freq_shards
                        )
                        with open(tmp_path, "wb") as f:
                            np.save(f, freqs)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_path, freq_cache_path)
                        print0(f"Wrote token frequency cache to {freq_cache_path}")
                    finally:
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                        try:
                            os.remove(lock_path)
                        except Exception:
                            pass
                else:
                    print0(f"Using cached token frequencies at {freq_cache_path}")
            else:
                print0(f"Using cached token frequencies at {freq_cache_path}")

        if ddp:
            torch.distributed.barrier()

        # All processes load the frequency cache and compute the tied mask
        freqs = np.load(freq_cache_path)
        tied_token_mask = compute_tied_mask(
            freqs, config.experiment.token_tying_mode, config.experiment.token_tying_ratio
        )
        n_tied = int(tied_token_mask.sum())
        tied_mass_frac = float(freqs[tied_token_mask].sum()) / max(1, float(freqs.sum()))
        print0(
            f"Token tying: mode={config.experiment.token_tying_mode}, "
            f"ratio={config.experiment.token_tying_ratio}, "
            f"tied={n_tied}/{len(tied_token_mask)} tokens, "
            f"tied_mass={tied_mass_frac:.4f}"
        )

        # If using permutations, modify them so tied tokens are identity
        if config.experiment.permute_tokens_per_compartment and use_pretokenized:
            if master_process and os.path.exists(permutations_path):
                perms = np.load(permutations_path)
                perms = apply_tying_to_permutations(perms, tied_token_mask)
                # Overwrite the permutations file with tying-adjusted version
                tmp_path = permutations_path + f".tying.tmp.{os.getpid()}"
                with open(tmp_path, "wb") as f:
                    np.save(f, perms)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, permutations_path)
                print0(f"Applied token tying to permutations at {permutations_path}")
            if ddp:
                torch.distributed.barrier()

    # Build compact remap table for offset mode tying
    tying_remap: Optional[np.ndarray] = None
    tying_compact_vocab: Optional[int] = None
    if tied_token_mask is not None and not config.experiment.permute_tokens_per_compartment:
        tying_remap, tying_compact_vocab = build_tying_remap(
            tied_token_mask, config.experiment.n_compartments
        )
        print0(
            f"Compact vocab: {tying_compact_vocab + 1} "
            f"(vs {config.model.vocab_size * config.experiment.n_compartments + 1} without tying)"
        )

    torch.manual_seed(training_seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model: torch.nn.Module
    checkpoint: Optional[dict[str, Any]] = None
    # Compute composite vocabulary
    exp_cfg = config.experiment
    if exp_cfg.max_compartments is None:
        raise ValueError("experiment.max_compartments is required")
    base_vocab = config.model.vocab_size
    if base_vocab is None:
        raise ValueError(
            "model.vocab_size (base) must be set for composite vocab computation"
        )
    # If permuting per compartment, vocabulary is base_vocab + 1 (only translation token)
    # Otherwise, it's base_vocab * n_compartments + 1 (offset scheme)
    # With token tying compact layout, vocab shrinks to base_vocab + (n_comp-1)*n_untied + 1
    # Note: we use n_compartments (not max_compartments) to size the model efficiently
    if tying_compact_vocab is not None:
        composite_vocab = tying_compact_vocab + 1  # +1 for translation token
    elif exp_cfg.permute_tokens_per_compartment:
        composite_vocab = base_vocab + 1
    else:
        composite_vocab = base_vocab * exp_cfg.n_compartments + 1
        if getattr(exp_cfg, "compartment_marker_token", False):
            # c extra ids, one prefix marker per compartment, placed after the
            # translation token. Marker for compartment i is composite_vocab + i
            # BEFORE this widening, i.e. base_vocab*c + 1 + i.
            composite_vocab += exp_cfg.n_compartments
    # Translation token id is always the last id in the vocab
    if tying_compact_vocab is not None:
        translation_token_id_cfg = tying_compact_vocab
    elif exp_cfg.permute_tokens_per_compartment:
        translation_token_id_cfg = base_vocab
    else:
        translation_token_id_cfg = base_vocab * exp_cfg.n_compartments
    # Auto-resume from the newest resumable checkpoint; else init from scratch.
    # Raises if the run has progress on disk that cannot be resumed, rather than
    # silently starting over and overwriting it.
    _resume = ckpt_utils.find_resume_checkpoint(out_dir)
    # A path that cannot exist, so "nothing resumable" falls through to scratch
    # init below. Deliberately NOT checkpoints/latest: that symlink points at
    # the newest *named* checkpoint, which is exactly what must not be resumed.
    ckpt_dir = (
        _resume.path
        if _resume
        else os.path.join(out_dir, "checkpoints", "_no_resume_candidate")
    )
    # Fallback target for the rolling-rewind guard below: the newest resumable
    # NAMED checkpoint. Stable only -- an annealed point is off the trunk's loss
    # curve and must never be silently resumed as if it were on it.
    _named = [
        c
        for c in ckpt_utils.iter_checkpoints(out_dir, include_rolling=False)
        if c.resumable and c.phase == "stable"
    ]
    _best_named = max(_named, key=lambda c: c.iter_num, default=None)
    best_named_dir = _best_named.path if _best_named else None
    best_named_iter = _best_named.iter_num if _best_named else -1
    rolling_iter = (
        _resume.iter_num if _resume and _resume.name == ckpt_utils.ROLLING else -1
    )

    dataloader_state: Optional[dict] = None
    if os.path.islink(ckpt_dir) or os.path.isdir(ckpt_dir):
        model_ckpt = os.path.join(ckpt_dir, "model.pt")
        trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
        if os.path.exists(model_ckpt) and os.path.exists(trainer_state_path):
            print(f"Resuming training from checkpoint: {ckpt_dir}")
            try:
                with open(trainer_state_path, "r") as f:
                    trainer_state = json.load(f)
                iter_num = trainer_state["iter_num"]
                best_val_loss = trainer_state.get("best_val_loss", 1e9)
                # Load model state.
                #
                # map_location="cpu", NOT device. Loading straight onto CUDA raises
                # "Attempting to deserialize object on a CUDA device but
                # torch.cuda.is_available() is False" whenever the GPU is not ready
                # on the claiming node — and the except-branch below then treats a
                # perfectly good rolling checkpoint as "corrupt" and silently
                # rewinds to the last NAMED checkpoint, which additionally drops
                # Adam state because named checkpoints carry no optimizer. The ORC
                # pool logs show this firing 19 times (plus 11 more from ECC errors
                # on bad nodes), with single runs thrown back to step-500000 four
                # times over. load_state_dict moves tensors to the right device
                # itself, so CPU-loading costs nothing.
                model_state_dict = torch.load(model_ckpt, map_location="cpu")
                # convert torch.compile state dict back to regular state dict
                unwanted_prefix = "_orig_mod."
                for k, v in list(model_state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        model_state_dict[k[len(unwanted_prefix) :]] = model_state_dict.pop(k)
                checkpoint = {"model_state_dict": model_state_dict}
                # Load optimizer state if present
                opt_ckpt = os.path.join(ckpt_dir, "optimizer.pt")
                if os.path.exists(opt_ckpt):
                    checkpoint["optimizer"] = torch.load(opt_ckpt, map_location="cpu")
                # Load dataloader state if present.
                #
                # weights_only=False is REQUIRED here, not a convenience. Torch 2.6
                # flipped the default to True, which refuses to unpickle anything but
                # tensors — and SyntheticTokenStream's n-gram sampler state is numpy
                # (per-chain xorshift states + context window), so every ladder run's
                # rolling checkpoint raises `Unsupported global: numpy._core.
                # multiarray._reconstruct`. The except-branch below then reports that
                # as "corrupt rolling checkpoint" and silently resumes from the last
                # NAMED checkpoint instead, which for these runs is 262k steps stale.
                # The file is one we wrote ourselves, so unpickling it is safe.
                # This RANK's state, not rank 0's. See load_dataloader_state.
                dataloader_state = load_dataloader_state(ckpt_dir, ddp_rank or 0)
            except Exception as e:
                if "_rolling" in str(ckpt_dir):
                    # Refuse to SILENTLY rewind. This fallback previously turned an
                    # unreadable rolling checkpoint into a quiet resume from the last
                    # named checkpoint — which, with named steps as sparse as
                    # 700k -> 1M, can discard a quarter-million steps while the run
                    # looks healthy. A warning in a log nobody is tailing is not
                    # enough. If the loss is more than one rolling interval, abort and
                    # make a human decide; set TC_ALLOW_ROLLING_REWIND=1 to override.
                    rewind = rolling_iter - best_named_iter
                    if rewind > ROLLING_CHECKPOINT_INTERVAL and os.environ.get(
                        "TC_ALLOW_ROLLING_REWIND"
                    ) != "1":
                        raise RuntimeError(
                            f"rolling checkpoint at iter {rolling_iter} failed to load "
                            f"({type(e).__name__}: {e}); falling back to the newest named "
                            f"checkpoint at iter {best_named_iter} would discard {rewind} "
                            f"steps. Refusing. Fix the load, or set "
                            f"TC_ALLOW_ROLLING_REWIND=1 to accept the rewind."
                        ) from e
                    print(f"WARNING: corrupt rolling checkpoint, falling back to named: {e}")
                    checkpoint = None
                    dataloader_state = None
                    # Fall back to best named checkpoint
                    if best_named_dir is not None:
                        ckpt_dir = best_named_dir
                        model_ckpt = os.path.join(ckpt_dir, "model.pt")
                        trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
                        if os.path.exists(model_ckpt) and os.path.exists(trainer_state_path):
                            print(f"Resuming from named checkpoint: {ckpt_dir}")
                            with open(trainer_state_path, "r") as f:
                                trainer_state = json.load(f)
                            iter_num = trainer_state["iter_num"]
                            best_val_loss = trainer_state.get("best_val_loss", 1e9)
                            model_state_dict = torch.load(model_ckpt, map_location="cpu")
                            unwanted_prefix = "_orig_mod."
                            for k, v in list(model_state_dict.items()):
                                if k.startswith(unwanted_prefix):
                                    model_state_dict[k[len(unwanted_prefix) :]] = model_state_dict.pop(k)
                            checkpoint = {"model_state_dict": model_state_dict}
                            opt_ckpt = os.path.join(ckpt_dir, "optimizer.pt")
                            if os.path.exists(opt_ckpt):
                                checkpoint["optimizer"] = torch.load(opt_ckpt, map_location="cpu")
                            dataloader_state = load_dataloader_state(
                                ckpt_dir, ddp_rank or 0
                            )
                else:
                    raise
    if checkpoint is not None:
        # Build model config matching the current config (not from checkpoint)
        vocab = composite_vocab
        gptconf = Model(
            **{
                **asdict(config.model),
                "vocab_size": vocab,
                "embedding_vocab_size": (
                    (base_vocab + 1)
                    if config.experiment.shared_token_embeddings
                    else vocab
                ),
                "shared_token_embeddings": bool(
                    config.experiment.shared_token_embeddings
                ),
                "use_compartment_embeddings": bool(
                    config.experiment.use_compartment_embeddings
                ),
                "copy_compartment_embeddings": (
                    False
                    if exp_cfg.permute_tokens_per_compartment
                    else bool(config.experiment.copy_compartment_embeddings)
                ),
                "copy_compartment_lm_head": (
                    False
                    if exp_cfg.permute_tokens_per_compartment
                    else bool(config.experiment.copy_compartment_lm_head)
                ),
                "copy_compartment_id_embeddings": bool(
                    config.experiment.copy_compartment_id_embeddings
                ),
                "base_vocab_size": base_vocab,
                "max_compartments": exp_cfg.n_compartments,
                "translation_token_id": translation_token_id_cfg,
                "weight_tying": (
                    False
                    if config.experiment.shared_token_embeddings
                    else config.model.weight_tying
                ),
                "icl_enabled": config.experiment.icl_mode != "none",
                "icl_vocab_size": (
                    config.experiment.icl_vocab_size
                    if config.experiment.icl_mode != "none"
                    else 0
                ),
            }
        )
        model = GPT(gptconf)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        vocab = composite_vocab
        gptconf = Model(
            **{
                **asdict(config.model),
                "vocab_size": vocab,
                # pass advanced options to model
                "embedding_vocab_size": (
                    (base_vocab + 1)
                    if config.experiment.shared_token_embeddings
                    else vocab
                ),
                "shared_token_embeddings": bool(
                    config.experiment.shared_token_embeddings
                ),
                "use_compartment_embeddings": bool(
                    config.experiment.use_compartment_embeddings
                ),
                # When permuting per-compartment, copying compartment weights is a no-op and shape-incompatible
                "copy_compartment_embeddings": (
                    False
                    if exp_cfg.permute_tokens_per_compartment
                    else bool(config.experiment.copy_compartment_embeddings)
                ),
                "copy_compartment_lm_head": (
                    False
                    if exp_cfg.permute_tokens_per_compartment
                    else bool(config.experiment.copy_compartment_lm_head)
                ),
                "copy_compartment_id_embeddings": bool(
                    config.experiment.copy_compartment_id_embeddings
                ),
                "base_vocab_size": base_vocab,
                "max_compartments": exp_cfg.n_compartments,  # Use n_compartments for model sizing
                "translation_token_id": translation_token_id_cfg,
                # disable weight tying if shared embeddings are used (validated elsewhere)
                "weight_tying": (
                    False
                    if config.experiment.shared_token_embeddings
                    else config.model.weight_tying
                ),
                "icl_enabled": config.experiment.icl_mode != "none",
                "icl_vocab_size": (
                    config.experiment.icl_vocab_size
                    if config.experiment.icl_mode != "none"
                    else 0
                ),
            }
        )
        model = GPT(gptconf)
    # crop down the model block size if desired, using model surgery
    if isinstance(model, GPT) and block_size < model.config.block_size:
        model.crop_block_size(block_size)
    model.to(device)

    # --- DANN setup ---
    dann_layer_indices = parse_dann_layers(config.experiment.dann_layers, gptconf.n_layer)
    dann_lambda = config.experiment.dann_lambda
    dann_enabled = len(dann_layer_indices) > 0 and dann_lambda > 0
    dann_module: DANNModule | None = None
    if dann_enabled:
        # Set layer collection attribute on model before compile
        cast(GPT, model)._dann_collect_layers = frozenset(dann_layer_indices)
        dann_disc_hidden = config.experiment.dann_disc_hidden
        dann_module = DANNModule(
            layer_indices=dann_layer_indices,
            n_embd=gptconf.n_embd,
            n_domains=exp_cfg.n_compartments,
            hidden=dann_disc_hidden,
        )
        # Load DANN state if resuming
        dann_ckpt_path = os.path.join(ckpt_dir, "dann_discriminators.pt") if (
            os.path.islink(ckpt_dir) or os.path.isdir(ckpt_dir)
        ) else None
        if dann_ckpt_path and os.path.exists(dann_ckpt_path):
            dann_state = torch.load(dann_ckpt_path, map_location=device)
            dann_module.load_state_dict(dann_state)
            print0(f"Loaded DANN discriminator state from {dann_ckpt_path}")
        dann_module.to(device)
        print0(f"DANN enabled: lambda={dann_lambda}, layers={dann_layer_indices}, "
               f"n_domains={exp_cfg.n_compartments}, hidden={dann_disc_hidden or gptconf.n_embd}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    # Add DANN discriminator params to optimizer (no weight decay)
    if dann_enabled and dann_module is not None:
        optimizer.add_param_group({"params": dann_module.parameters(), "weight_decay": 0.0})
    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])  # pyright: ignore[reportArgumentType]
    elif checkpoint is not None and iter_num > 0:
        # Resuming mid-run with NO optimizer state: Adam restarts from zero
        # moments. This used to happen silently, because named checkpoints carry
        # no optimizer.pt and the key check above just skips the restore — so a
        # transient load failure of _rolling would fall back to a named
        # checkpoint and quietly wipe the moments. Say it loudly instead; an
        # unexplained loss bump hundreds of thousands of steps later is not
        # something anyone should have to reverse-engineer.
        print(
            f"WARNING: resuming at iter {iter_num} with NO optimizer state — Adam "
            f"moments reset to zero. Expect a transient loss bump. This is expected "
            f"only when deliberately seeding from a weights-only checkpoint; if this "
            f"run resumed from _rolling, something is wrong."
        )
    checkpoint = None  # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        # model = cast(torch.nn.Module, torch.compile(model, mode="max-autotune"))  # requires PyTorch 2.0
        model = cast(torch.nn.Module, torch.compile(model))  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])  # pyright: ignore[reportPossiblyUnboundVariable]
        if dann_enabled and dann_module is not None:
            dann_module = DDP(dann_module, device_ids=[ddp_local_rank])

    # Data loaders: pretokenized assignments-based vs uniform synthetic
    if use_pretokenized:
        # Custom assignments-driven dataloader that transforms tokens per assignments.bin
        compartment_train_patterns = config.data.compartment_train_bins
        compartment_val_patterns = config.data.compartment_val_bins

        # Compute frequency distribution if any compartment uses synthetic:frequency
        synthetic_freqs = None
        if compartment_train_patterns and any(
            p.startswith("synthetic:frequency") for p in compartment_train_patterns
        ):
            freq_source = next(
                (p for p in compartment_train_patterns if not p.startswith("synthetic:")),
                None,
            )
            assert freq_source, "synthetic:frequency needs at least one pretokenized compartment"
            from src.token_tying import compute_token_frequencies
            synthetic_freqs = compute_token_frequencies(freq_source, base_vocab)
            print0(f"Computed token frequencies from {freq_source}")

        synthetic_seed = int(config.data.uniform_seed + training_seed)

        train_loader = AssignmentsDataLoader(
            assignments_path,
            train_bin,
            batch_size,
            block_size,
            ddp_rank or 0,
            ddp_world_size,
            base_vocab,
            cast(int, exp_cfg.max_compartments),
            exp_cfg.n_compartments,
            constant_records=total_examples if assignments_path is None else None,
            shuffle_seed=(base_seed if config.data.shuffle else None),
            bpe_variant_expansion=float(getattr(exp_cfg, "bpe_variant_expansion", 0.0) or 0.0),
            bpe_variant_tokenizer=getattr(exp_cfg, "bpe_variant_tokenizer", ""),
            bpe_variant_seed=int(base_seed),
            bpe_variant_freq_pattern=train_bin,
            shared_output_vocab=bool(getattr(exp_cfg, 'shared_output_vocab', False)),
            compartment_marker_base=(base_vocab * exp_cfg.n_compartments + 1
                if getattr(exp_cfg, 'compartment_marker_token', False) else None),
            permute_tokens=exp_cfg.permute_tokens_per_compartment,
            permutations_path=(
                permutations_path if exp_cfg.permute_tokens_per_compartment else None
            ),
            permute_inputs=exp_cfg.permute_input_tokens_per_compartment,
            tied_token_mask=tied_token_mask,
            tying_remap=tying_remap,
            translation_token_id=translation_token_id_cfg,
            compartment_filename_patterns=compartment_train_patterns,
            synthetic_seed=synthetic_seed,
            synthetic_frequencies=synthetic_freqs,
        )
        val_loader = None
        if val_bin or compartment_val_patterns:
            val_loader = AssignmentsDataLoader(
                assignments_path,
                val_bin or "",
                batch_size,
                block_size,
                ddp_rank or 0,
                ddp_world_size,
                base_vocab,
                cast(int, exp_cfg.max_compartments),
                exp_cfg.n_compartments,
                constant_records=total_examples if assignments_path is None else None,
                shuffle_seed=(base_seed if config.data.shuffle else None),
                bpe_variant_expansion=float(getattr(exp_cfg, "bpe_variant_expansion", 0.0) or 0.0),
                bpe_variant_tokenizer=getattr(exp_cfg, "bpe_variant_tokenizer", ""),
                bpe_variant_seed=int(base_seed),
                bpe_variant_freq_pattern=train_bin,
                shared_output_vocab=bool(getattr(exp_cfg, 'shared_output_vocab', False)),
                compartment_marker_base=(base_vocab * exp_cfg.n_compartments + 1
                    if getattr(exp_cfg, 'compartment_marker_token', False) else None),
                permute_tokens=exp_cfg.permute_tokens_per_compartment,
                permutations_path=(
                    permutations_path
                    if exp_cfg.permute_tokens_per_compartment
                    else None
                ),
                permute_inputs=exp_cfg.permute_input_tokens_per_compartment,
                tied_token_mask=tied_token_mask,
                tying_remap=tying_remap,
                translation_token_id=translation_token_id_cfg,
                compartment_filename_patterns=compartment_val_patterns,
                synthetic_seed=synthetic_seed + 2_000_000_000,
                synthetic_frequencies=synthetic_freqs,
            )
    else:
        # Uniform synthetic stream
        # Check if we need compartment/translation support
        needs_compartments = exp_cfg.n_compartments > 1 or exp_cfg.translation_ratio > 0
        if needs_compartments:
            # Generate permutations in-memory if needed
            uniform_perms = None
            if exp_cfg.permute_tokens_per_compartment:
                max_c = cast(int, exp_cfg.max_compartments)
                ss = np.random.SeedSequence(int(training_seed) & 0xFFFFFFFFFFFFFFFF)
                child_seeds = ss.spawn(max_c)
                uniform_perms = np.empty((max_c, base_vocab), dtype=np.int64)
                for c, child_ss in enumerate(child_seeds):
                    gen = np.random.Generator(np.random.PCG64(child_ss))
                    uniform_perms[c] = gen.permutation(base_vocab).astype(np.int64)
                # Apply tying to in-memory permutations if needed
                if tied_token_mask is not None:
                    uniform_perms = apply_tying_to_permutations(uniform_perms, tied_token_mask)
                print0(
                    f"Generated in-memory permutations with shape {uniform_perms.shape}"
                )

            train_loader = UniformCompartmentDataLoader(
                B=batch_size,
                T=block_size,
                base_vocab_size=base_vocab,
                seed=config.data.uniform_seed + training_seed,
                n_compartments=exp_cfg.n_compartments,
                max_compartments=cast(int, exp_cfg.max_compartments),
                translation_ratio=exp_cfg.translation_ratio,
                translation_ratio_mode=exp_cfg.translation_ratio_mode,
                compartment_scaling=exp_cfg.compartment_scaling,
                process_rank=ddp_rank or 0,
                num_processes=ddp_world_size,
                permute_tokens=exp_cfg.permute_tokens_per_compartment,
                permutations=uniform_perms,
                permute_inputs=exp_cfg.permute_input_tokens_per_compartment,
                tied_token_mask=tied_token_mask,
                tying_remap=tying_remap,
                translation_token_id_override=translation_token_id_cfg if tying_remap is not None else None,
                pin_memory=True,
                translation_mode=exp_cfg.translation_mode,
                translation_chunk_size=exp_cfg.translation_chunk_size,
            )
        else:
            # Simple single-compartment uniform data (original behavior)
            train_loader = UniformBatchDataLoader(
                B=batch_size,
                T=block_size,
                vocab_size=composite_vocab,
                seed=config.data.uniform_seed + training_seed,
                process_rank=ddp_rank or 0,
                num_processes=ddp_world_size,
                return_compartment_ids=bool(exp_cfg.use_compartment_embeddings),
                pin_memory=True,
            )
        val_loader = None

    # Restore dataloader state if resuming from checkpoint
    if dataloader_state is not None:
        print0(f"Restoring dataloader state from checkpoint (resuming at iter {iter_num})")
        train_loader.load_state_dict(dataloader_state["train"])
        if val_loader is not None and "val" in dataloader_state:
            val_loader.load_state_dict(dataloader_state["val"])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        if val_loader is None:
            return None
        out = {}
        model.eval()
        _icl_eval = (
            getattr(config.experiment, "icl_mode", "none") == "dual_stream"
        )
        _icl_V_eval = getattr(config.experiment, "icl_vocab_size", 0)
        _icl_mask_p_eval = float(getattr(config.experiment, "icl_mask_p", 0.0))
        for split in ["train", "val"]:
            # Reset the validation data loader to ensure deterministic eval on the same
            # data
            if split == "val":
                val_loader.reset()
            losses = torch.zeros(eval_iters)
            icl_losses = torch.zeros(eval_iters) if _icl_eval else None
            for k in range(eval_iters):
                batch = val_loader.next_batch()
                if isinstance(batch, tuple) and len(batch) == 3:
                    X, Y, Cval = batch
                else:
                    X, Y = batch  # type: ignore[misc]
                    Cval = None
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                Cval = Cval.to(device, non_blocking=True) if Cval is not None else None
                _icl_idx_eval = None
                _icl_tgt_eval = None
                _icl_mask_eval = None
                if _icl_eval:
                    # Deterministic per-example salt derived from sum of first
                    # 4 canonical tokens — same example always hashes the same
                    # way across eval runs. Mixed with a Knuth constant for
                    # spread.
                    _salt = (
                        X[:, :4].clamp(min=0).sum(dim=1).to(torch.int64)
                        * 2654435761
                    )
                    _mixed_X = X.clamp(min=0).to(torch.int64) * 2654435761 + _salt.unsqueeze(1)
                    _mixed_Y = Y.clamp(min=0).to(torch.int64) * 2654435761 + _salt.unsqueeze(1)
                    _icl_idx_eval = (_mixed_X & (_icl_V_eval - 1)).to(X.dtype)
                    _icl_tgt_eval = (_mixed_Y & (_icl_V_eval - 1)).to(Y.dtype)
                    _icl_idx_eval = torch.where(X >= 0, _icl_idx_eval, X)
                    _icl_tgt_eval = torch.where(Y >= 0, _icl_tgt_eval, Y)
                    if _icl_mask_p_eval > 0.0:
                        # Deterministic mask per (split, k) for reproducibility
                        _gen = torch.Generator(device=device).manual_seed(
                            1234567 + (1 if split == "val" else 0) * 1_000_000 + k
                        )
                        _icl_mask_eval = (
                            torch.rand(X.shape, generator=_gen, device=device)
                            < _icl_mask_p_eval
                        )
                with ctx:
                    # mark cudagraph step begin if available to avoid output overwrite
                    torch.compiler.cudagraph_mark_step_begin()
                    if _icl_eval:
                        _, loss, _, icl_loss_v = model(
                            X, Y, compartment_ids=Cval,
                            icl_idx=_icl_idx_eval, icl_targets=_icl_tgt_eval,
                            icl_mask=_icl_mask_eval,
                        )
                    else:
                        result = model(X, Y, compartment_ids=Cval)
                        loss = result[1]  # works for both 2-tuple and 3-tuple
                losses[k] = loss.item()
                if _icl_eval and icl_loss_v is not None:
                    icl_losses[k] = icl_loss_v.item()
            out[split] = losses.mean()
            if _icl_eval:
                out[f"{split}_icl"] = icl_losses.mean()
        model.train()
        return out

    def get_lr(it):
        # `it` is the GLOBAL iteration number, restored from trainer_state.json,
        # so this is a pure function of position in the lineage: a run preempted
        # mid-decay resumes on the LR it would have had rather than restarting
        # the decay. See src/lr_schedule.py for why that property is
        # load-bearing for the branch/anneal layout.
        return lr_schedule.lr_at(
            it,
            peak=learning_rate,
            warmup_iters=warmup_iters,
            min_lr=min_lr,
            schedule=schedule,
            decay_lr=decay_lr,
            lr_decay_iters=lr_decay_iters,
            decay_start_iter=decay_start_iter,
            decay_end_iter=decay_end_iter,
        )

    # logging
    wandb_buffer_enabled = config.logging.wandb_buffer
    wandb_log_buffer: list[tuple[dict, int]] = []  # list of (metrics_dict, step)

    def wandb_log_or_buffer(metrics: dict, step: int) -> None:
        """Log to wandb directly, or buffer if wandb_buffer is enabled."""
        if wandb_buffer_enabled:
            wandb_log_buffer.append((metrics, step))
        else:
            import wandb
            wandb.log(metrics, step=step)

    def wandb_flush_buffer() -> None:
        """Flush all buffered wandb log entries."""
        if not wandb_log_buffer:
            return
        import wandb
        for metrics, step in wandb_log_buffer:
            wandb.log(metrics, step=step)
        wandb_log_buffer.clear()

    wandb_run: Optional[Any] = None
    if wandb_log and master_process:
        import wandb

        name_value: Optional[str] = None
        if isinstance(wandb_run_name, str):
            _nm = wandb_run_name.strip()
            if _nm and _nm.lower() != "sweep":
                name_value = wandb_run_name

        wandb_run = wandb.init(
            project=wandb_project,
            group=wandb_group,
            notes=wandb_notes,
            config={
                "cfg_hash": cfg_hash(config),
                "run_id": run_id,
                "out_dir": out_dir,
                **asdict(config),
            },
            dir=out_dir,
            id=run_id,
            resume="allow",
            name=name_value,
            settings=wandb.Settings(init_timeout=300),
        )

    # InfoNCE alignment pool (optional)
    infonce_pool = None
    infonce_mode = "wikimatrix"
    if config.experiment.infonce_enabled:
        infonce_mode = getattr(config.experiment, "infonce_pool_mode", "wikimatrix")
        if infonce_mode == "wikimatrix":
            if not config.experiment.infonce_pool_path:
                raise ValueError("experiment.infonce_pool_path must be set when infonce_enabled")
            from src.infonce import InfoNCEPool, compute_infonce_loss
            infonce_pool = InfoNCEPool(
                pool_path=config.experiment.infonce_pool_path,
                pool_frac=config.experiment.infonce_pool_frac,
                pool_seed=config.experiment.infonce_pool_seed,
                process_rank=ddp_rank or 0,
                zh_token_offset=config.experiment.infonce_zh_token_offset,
            )
            print0(
                f"[infonce] mode=wikimatrix pool_frac={config.experiment.infonce_pool_frac} -> "
                f"{infonce_pool.n_keep:,} of {infonce_pool.n_total:,} pairs"
            )
        elif infonce_mode == "compartment":
            from src.infonce import CompartmentOriginalPool, compute_infonce_compartment_loss
            # Load the same permutations the data loader uses (if any).
            perms_for_infonce = None
            if config.experiment.permute_input_tokens_per_compartment:
                _perms_path = os.path.join(out_dir, "permutations.npy")
                if os.path.exists(_perms_path):
                    perms_for_infonce = np.load(_perms_path)
            infonce_pool = CompartmentOriginalPool(
                train_bin_glob=config.data.train_bin,
                seq_len=config.model.block_size,
                base_vocab=config.model.vocab_size,
                permutations=perms_for_infonce,
                process_rank=ddp_rank or 0,
                seed=config.experiment.infonce_pool_seed,
            )
            print0(
                f"[infonce] mode=compartment pool_tokens={infonce_pool.total_tokens:,} "
                f"perms={'yes' if perms_for_infonce is not None else 'no'}"
            )
        elif infonce_mode == "bio_decl_qa":
            from src.infonce import BioDeclQAPool, compute_infonce_loss
            if not config.experiment.infonce_pool_decl_path:
                raise ValueError("experiment.infonce_pool_decl_path must be set")
            if not config.experiment.infonce_pool_qa_path:
                raise ValueError("experiment.infonce_pool_qa_path must be set")
            infonce_pool = BioDeclQAPool(
                decl_path=config.experiment.infonce_pool_decl_path,
                qa_path=config.experiment.infonce_pool_qa_path,
                qa_offset=config.experiment.infonce_pool_qa_offset,
                process_rank=ddp_rank or 0,
                seed=config.experiment.infonce_pool_seed,
            )
            print0(
                f"[infonce] mode=bio_decl_qa n_persons={infonce_pool.n_persons:,} "
                f"qa_offset={config.experiment.infonce_pool_qa_offset}"
            )
        else:
            raise ValueError(f"unknown infonce_pool_mode: {infonce_mode}")
        # Resolve infonce_layer (default to mid-trunk if -1)
        infonce_layer = config.experiment.infonce_layer
        if infonce_layer < 0:
            infonce_layer = config.model.n_layer // 2
        print0(
            f"[infonce] layer={infonce_layer}, n={config.experiment.infonce_n}, "
            f"tau={config.experiment.infonce_tau}, every={config.experiment.infonce_every}, "
            f"lambda={config.experiment.infonce_lambda}"
        )
    else:
        infonce_layer = None
    # Per-rank RNG for compartment-pair sampling (one-time init).
    if infonce_mode == "compartment":
        _infonce_pair_rng = np.random.default_rng(
            config.experiment.infonce_pool_seed + 100 + (ddp_rank or 0)
        )

    # Seed-anneal permutation pool. Drawn once from training.seed; identical on
    # every rank so the schedule is globally coherent. Row 0 = identity so the
    # post-anneal tail trains on natural data. Rows 1..n-1 are fresh random
    # permutations under the matching mode. See ExperimentConfig docstring.
    # Pool stored as int32 for vocab mode (n*V can hit ~1 GB); compartment mode
    # is trivially small so keep long. Row-gather + gather-along-dim-1 preserve
    # int32; result cast to X.dtype at the end.
    _seed_anneal_pool = None
    _seed_anneal_n = 0
    _seed_anneal_L_examples = 0
    _seed_anneal_pool_override_active = False
    if (
        config.experiment.permutation_schedule == "seed_anneal"
        and config.experiment.permutation_mode != "none"
    ):
        import math as _math
        _seed_anneal_L_iters = int(config.experiment.permutation_anneal_iters)
        if _seed_anneal_L_iters <= 0:
            raise ValueError(
                "permutation_schedule='seed_anneal' requires "
                "permutation_anneal_iters > 0"
            )
        _seed_anneal_L_examples = _seed_anneal_L_iters * effective_batch_size
        _seed_anneal_pool_override = int(config.experiment.permutation_pool_size)
        if _seed_anneal_pool_override > 0:
            _seed_anneal_n = _seed_anneal_pool_override
            _seed_anneal_pool_override_active = True
        else:
            _seed_anneal_n = int(round(_math.sqrt(_seed_anneal_L_examples)))
        _pool_seed = int(config.training.seed) + 20260618
        _pool_gen = torch.Generator(device="cpu").manual_seed(_pool_seed)
        _base_vocab_pool = config.model.vocab_size
        _n_comp_pool = config.experiment.n_compartments
        if config.experiment.permutation_mode == "vocab":
            _pool = torch.empty(
                (_seed_anneal_n, _base_vocab_pool), dtype=torch.int32
            )
            _pool[0] = torch.arange(_base_vocab_pool, dtype=torch.int32)
            for _s in range(1, _seed_anneal_n):
                _pool[_s] = torch.randperm(
                    _base_vocab_pool, generator=_pool_gen
                ).to(torch.int32)
            _seed_anneal_pool = _pool.to(device)
        elif config.experiment.permutation_mode == "compartment":
            _pool = torch.empty(
                (_seed_anneal_n, _n_comp_pool), dtype=torch.long
            )
            _pool[0] = torch.arange(_n_comp_pool, dtype=torch.long)
            for _s in range(1, _seed_anneal_n):
                _pool[_s] = torch.randperm(_n_comp_pool, generator=_pool_gen)
            _seed_anneal_pool = _pool.to(device)
        _pool_bytes = _seed_anneal_pool.element_size() * _seed_anneal_pool.numel()
        print0(
            f"[seed_anneal] mode={config.experiment.permutation_mode} "
            f"L_iters={_seed_anneal_L_iters} "
            f"L_examples={_seed_anneal_L_examples} n={_seed_anneal_n} "
            f"pool_shape={tuple(_seed_anneal_pool.shape)} "
            f"pool_MB={_pool_bytes/1024**2:.1f}"
        )

    # training loop
    batch0 = train_loader.next_batch()  # fetch the very first batch
    if isinstance(batch0, tuple) and len(batch0) == 3:
        X, Y, C = batch0
    else:
        X, Y = batch0  # type: ignore[misc]
        C = None
    X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
    C = C.to(device, non_blocking=True) if C is not None else None
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model: GPT = (
        cast(GPT, model.module) if ddp else cast(GPT, model)
    )  # unwrap DDP container if needed
    running_mfu = -1.0
    # ``max_iters`` is the number of optimizer updates, not the final
    # zero-based iteration label. Assignment budgets use the same convention.
    while iter_num < max_iters:
        # Check for preemption signal (GPU rescheduling)
        if _preempt_requested.is_set():
            # Every rank writes its own dataloader position; only master writes
            # the weights. A preempt that saved rank 0's position alone would
            # make the requeued job train all ranks on identical data.
            if not master_process:
                _save_rolling_checkpoint(
                    out_dir, raw_model, optimizer, train_loader,
                    val_loader, iter_num, best_val_loss,
                    rank=ddp_rank or 0, master=False,
                )
            if master_process:
                print(f"[preempt] SIGUSR1 received at iter {iter_num} — saving checkpoint and exiting")
                _save_rolling_checkpoint(
                    out_dir, raw_model, optimizer, train_loader,
                    val_loader, iter_num, best_val_loss,
                    rank=ddp_rank or 0, master=True,
                )
                if wandb_log and wandb_buffer_enabled:
                    wandb_flush_buffer()
                if wandb_log:
                    try:
                        import wandb
                        wandb.mark_preempting()
                    except Exception:
                        pass
            break

        # determine and set the learning rate for this iteration
        # lr = get_lr(iter_num) if decay_lr else learning_rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = None
            if eval_iters > 0 and val_loader is not None:
                losses = cast(dict[str, torch.Tensor], estimate_loss())
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if wandb_log and master_process:
                    _eval_metrics: dict[str, Any] = {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }
                    if "val_icl" in losses:
                        _eval_metrics["train/icl_loss"] = losses["train_icl"]
                        _eval_metrics["val/icl_loss"] = losses["val_icl"]
                    wandb_log_or_buffer(_eval_metrics, step=iter_num)
            if losses is not None and losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        # Save checkpoints only at log-spaced steps
        # Every rank persists its OWN stream position at exactly the iterations
        # that produce a resumable checkpoint; the master-only block below writes
        # weights and optimizer state. Conditions here are rank-uniform, so this
        # needs no collective -- deliberately, since the preempt path above is
        # not provably rank-uniform and a collective there would deadlock.
        if (
            not master_process
            and iter_num in checkpoint_steps
            and iter_num > 0
            and iter_num in full_state_iters
        ):
            save_dataloader_state(
                _checkpoint_dir(os.path.join(out_dir, "checkpoints"), iter_num),
                train_loader, val_loader, ddp_rank or 0,
            )

        if (
            master_process
            and iter_num in checkpoint_steps
            and iter_num > 0
        ):
            ck_root = os.path.join(out_dir, "checkpoints")
            step_dir = _checkpoint_dir(ck_root, iter_num)
            os.makedirs(step_dir, exist_ok=True)
            save_full_state = iter_num in full_state_iters
            print(
                f"saving checkpoint to {step_dir}"
                + (" (+optimizer: resumable)" if save_full_state else "")
            )
            # Named (log-spaced) checkpoints: model weights (in bf16) + metadata.
            # Weights are bf16 (halves size, negligible precision loss — eval casts
            # to bf16 anyway). Optimizer state is 5.9x the weights, so it is written
            # only at the points named in full_state_at_tokens; everywhere else
            # _rolling covers preemption. A checkpoint without optimizer.pt is
            # therefore not resumable, and the resume path will not select it.
            named_state = {
                k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                for k, v in raw_model.state_dict().items()
            }
            torch.save(named_state, os.path.join(step_dir, "model.pt"))
            # Save DANN discriminator state
            if dann_enabled and dann_module is not None:
                raw_dann = dann_module.module if isinstance(dann_module, DDP) else dann_module
                torch.save(raw_dann.state_dict(), os.path.join(step_dir, "dann_discriminators.pt"))
            if save_full_state:
                # fp32 optimizer state, deliberately not downcast: this exists to
                # be resumed from, and bf16 moments would perturb the trajectory.
                torch.save(optimizer.state_dict(), os.path.join(step_dir, "optimizer.pt"))
                save_dataloader_state(step_dir, train_loader, val_loader, ddp_rank or 0)
            # trainer_state.json LAST — its presence is what marks the checkpoint
            # complete, so it must not appear before optimizer.pt.
            with open(os.path.join(step_dir, "trainer_state.json"), "w") as f:
                json.dump(
                    {
                        "iter_num": iter_num,
                        "best_val_loss": float(best_val_loss),
                        "tokens": iter_num * tokens_per_iter,
                        "phase": "annealed" if is_anneal_child else "stable",
                    },
                    f,
                )
            # Atomically update latest symlink (after all files are written)
            # Only update if this step is >= the current latest to prevent
            # a lagging worker from clobbering a newer checkpoint.
            latest = os.path.join(ck_root, "latest")
            should_update_latest = True
            if os.path.islink(latest):
                try:
                    cur_target = os.readlink(latest)
                    # Compare by iter_num read from the target itself. Parsing the
                    # directory name worked only for step-* and would silently
                    # ValueError (and overwrite) on tok-* names.
                    cur_iter = ckpt_utils.checkpoint_iter(
                        os.path.join(ck_root, cur_target)
                    )
                    if cur_iter is not None and iter_num < cur_iter:
                        should_update_latest = False
                        print(f"WARNING: not updating latest symlink: current {cur_target} "
                              f"is newer than iter {iter_num}")
                except (ValueError, IndexError, OSError):
                    pass  # malformed symlink target — overwrite it
            if should_update_latest:
                tmp_link = latest + f".tmp.{os.getpid()}"
                # relpath, not basename: an annealed checkpoint lives one level
                # down in annealed/, and basename would produce a dangling link.
                os.symlink(os.path.relpath(step_dir, ck_root), tmp_link)
                os.replace(tmp_link, latest)
            # Flush buffered wandb logs now that checkpoint is durable
            if wandb_log and master_process and wandb_buffer_enabled:
                wandb_flush_buffer()

        # Rolling "latest" checkpoint for preemption resilience.
        # Skip if a named checkpoint was already saved this step.
        # NOT gated on master_process: every rank must record its own stream
        # position. _save_rolling_checkpoint writes weights only for master.
        if (
            ROLLING_CHECKPOINT_INTERVAL > 0
            and iter_num > 0
            and iter_num % ROLLING_CHECKPOINT_INTERVAL == 0
            # Skip only when this step ALREADY writes a resumable checkpoint.
            # The old condition skipped at every NAMED step, but named checkpoints
            # are weights-only unless the step is in full_state_iters -- so on a
            # run slow enough that the wall lands inside the dense early
            # checkpoint region, every rolling save was suppressed and the job
            # ended with nothing resumable. Observed on the H100 twin: dense
            # named steps cover every multiple of 1000 up to 10,000, the first
            # rolling save would have been iter 11,000, and 24h reached ~9,800.
            # The run was unresumable by construction.
            and iter_num not in full_state_iters
        ):
            _save_rolling_checkpoint(
                out_dir, raw_model, optimizer, train_loader,
                val_loader, iter_num, best_val_loss,
                rank=ddp_rank or 0, master=master_process,
            )

        if iter_num == 0 and eval_only:
            break

        # Per-example permutation pretraining. Two modes:
        #   - "vocab" (for c=1): a fraction `_frac` of the V token IDs are
        #     deterministically selected per example and permuted among
        #     themselves. Outside that subset, identity mapping. So _frac=1.0
        #     is a full vocab permutation; _frac=0 is identity.
        #   - "compartment" (for c>1): a fraction `_frac` of token positions
        #     in the (B, T+1) underlying sequence get a uniformly-random
        #     compartment in [0, n_comp); the rest keep the example's natural
        #     compartment. Both X composite IDs and compartment_ids C are
        #     remapped consistently.
        # Schedule selects how _frac evolves with iter_num. The actual
        # permutation application happens INSIDE the micro_step loop below
        # (each microstep gets a fresh permutation).
        _perm_mode = config.experiment.permutation_mode
        _perm_cliff = config.experiment.permutation_cliff_step
        _perm_sched = config.experiment.permutation_schedule
        _perm_floor = float(getattr(config.experiment, "permutation_floor", 0.0))
        if _perm_mode != "none":
            if _perm_sched == "seed_anneal":
                # _frac unused by this schedule (seed pool drives it). Keep 1.0
                # for logging so "permutation active" is trivially visible in
                # wandb; the actual seed count k is logged separately below.
                _frac = 1.0
            elif _perm_cliff > 0 and iter_num < _perm_cliff:
                if _perm_sched == "sharp":
                    _frac = 1.0
                elif _perm_sched == "linear":
                    _frac = max(_perm_floor, 1.0 - iter_num / _perm_cliff)
                else:
                    _frac = _perm_floor
            else:
                # Post-cliff (or cliff_step=0): hold at the floor. This makes
                # the floor a "constant residual permutation rate" — set
                # cliff_step=0 + a positive floor to get pure constant-rate
                # training (variant B).
                _frac = _perm_floor
        else:
            _frac = 0.0
        _permute_active_frac = _frac  # capture for wandb logging
        # Representative k for seed_anneal (from rank-0, micro-0, batch-0 j).
        _permute_active_k = None
        if _perm_sched == "seed_anneal" and _seed_anneal_n > 0:
            _n_sa = _seed_anneal_n
            _j_rep = iter_num * gradient_accumulation_steps * batch_size * ddp_world_size
            if _j_rep >= _n_sa * _n_sa:
                _permute_active_k = 1
            else:
                _permute_active_k = max(1, _n_sa - (_j_rep // _n_sa))

        # ICL dual-stream pretraining hash function.
        # Per-example salt drawn fresh each microstep. icl_idx[b, t] is a
        # deterministic multiplicative-hash mapping from canonical token IDs
        # into [0, icl_vocab_size). Stable within an example, different
        # across examples.
        _icl_mode = config.experiment.icl_mode
        _icl_V = config.experiment.icl_vocab_size
        _icl_lambda = config.experiment.icl_lambda
        _icl_mask_p = float(getattr(config.experiment, "icl_mask_p", 0.0))
        _icl_MULT = 2654435761  # Knuth's multiplicative constant; coprime with 2^k.

        def _icl_hash(toks, salt):
            """Hash canonical token IDs to ICL token IDs.
            toks: (B, T) int64 in [0, V).
            salt: (B,) int64 per-example salt.
            Returns: (B, T) int64 in [0, icl_V).
            """
            # Broadcast salt over T positions. Use bitwise-and modulus since
            # icl_V is a power of two.
            mixed = toks * _icl_MULT + salt.unsqueeze(1)
            return (mixed & (_icl_V - 1)).clamp(min=0)

        def _seed_anneal_seed_indices(Bsz):
            """Compute per-example seed indices for seed_anneal schedule.
            Returns a (B,) long tensor on `device`. Rank-interleaved global
            index: j_global = W * (t*G*B + m*B + b) + rank.
            """
            n = _seed_anneal_n
            _W = ddp_world_size
            _r = ddp_rank or 0
            _start = int(config.experiment.permutation_anneal_start_iter)
            _iter_eff = max(0, iter_num - _start)
            _base_j = _W * (
                _iter_eff * gradient_accumulation_steps * Bsz
                + micro_step * Bsz
            ) + _r
            _b_arange = torch.arange(Bsz, device=device, dtype=torch.long)
            j_global = _base_j + _b_arange * _W  # (B,)
            if iter_num < _start:
                # Pre-anneal: force identity, no pair loss (handled elsewhere).
                return torch.zeros(Bsz, device=device, dtype=torch.long)
            if _seed_anneal_pool_override_active:
                # Fixed-pool ablation: uniformly sample seed_idx ∈ [0, n) for
                # the entire anneal window (L_examples), drop to 0 after.
                _post_anneal = j_global >= _seed_anneal_L_examples
                _seed_idx = j_global % n
                _seed_idx = torch.where(
                    _post_anneal, torch.zeros_like(_seed_idx), _seed_idx
                )
                return _seed_idx.long()
            _post_anneal = j_global >= (n * n)
            _phase_idx = torch.div(j_global, n, rounding_mode="floor")
            _k = torch.clamp(n - _phase_idx, min=1, max=n)  # (B,)
            _j_in_phase = j_global - _phase_idx * n  # (B,) in [0, n)
            # Even-remainder mapping: seed = min(floor(j_in_phase*k/n), k-1)
            _seed_idx = torch.minimum(
                torch.div(_j_in_phase * _k, n, rounding_mode="floor"),
                _k - 1,
            )
            _seed_idx = torch.where(
                _post_anneal, torch.zeros_like(_seed_idx), _seed_idx
            )
            return _seed_idx.long()

        def _apply_seed_anneal_perm(X, Y, C, seed_idx):
            """Apply the seed-anneal permutation for the given per-example
            seed indices. Same math as the seed_anneal path in _maybe_permute
            but with externally-supplied seed_idx so pair paths can drive it.
            """
            _base_vocab = config.model.vocab_size
            _n_comp = config.experiment.n_compartments
            if _perm_mode == "vocab":
                _p = _seed_anneal_pool.index_select(0, seed_idx)
                _Xp = torch.gather(_p, 1, X.clamp(min=0).long())
                _Yp = torch.gather(_p, 1, Y.clamp(min=0).long())
                Xn = torch.where(X >= 0, _Xp.to(X.dtype), X)
                Yn = torch.where(Y >= 0, _Yp.to(Y.dtype), Y)
                return Xn, Yn, C
            if _perm_mode == "compartment":
                Bsz, Tsz = X.shape
                _sigma = _seed_anneal_pool.index_select(0, seed_idx)
                _natural_comp = torch.zeros(
                    (Bsz, Tsz + 1), dtype=torch.long, device=device
                )
                _natural_comp[:, :Tsz] = (X // _base_vocab).long()
                _natural_comp[:, Tsz] = (Y[:, -1] // _base_vocab).long()
                _natural_comp_idx = _natural_comp.clamp(0, _n_comp - 1)
                _new_comp = torch.gather(_sigma, 1, _natural_comp_idx).to(X.dtype)
                _trans_id = _base_vocab * _n_comp
                _base_X = X % _base_vocab
                _base_Y = Y % _base_vocab
                _new_X = _base_X + _new_comp[:, :Tsz] * _base_vocab
                _new_Y = _base_Y + _new_comp[:, 1:] * _base_vocab
                _keep_X = (X < 0) | (X == _trans_id)
                _keep_Y = (Y < 0) | (Y == _trans_id)
                Xn = torch.where(_keep_X, X, _new_X)
                Yn = torch.where(_keep_Y, Y, _new_Y)
                Cn = C
                if C is not None:
                    Cn = torch.where(_keep_X, C, _new_comp[:, :Tsz].to(C.dtype))
                return Xn, Yn, Cn
            return X, Y, C

        def _seed_anneal_partner(seed_idx, phase_k):
            """Deterministic partner seed derived from seed_idx and phase_k.
            partner = (seed_idx + 1 + hash % (k-1)) mod k. Guaranteed
            different from seed_idx when k >= 2. At k=1 caller must skip."""
            n = _seed_anneal_n
            Bsz = seed_idx.shape[0]
            _r = ddp_rank or 0
            _W = ddp_world_size
            _base_j = _W * (
                iter_num * gradient_accumulation_steps * Bsz
                + micro_step * Bsz
            ) + _r
            _b_arange = torch.arange(Bsz, device=device, dtype=torch.long)
            j_global = _base_j + _b_arange * _W
            # Cheap deterministic per-example scramble
            h = (j_global * 2654435761) & 0x7FFFFFFF
            shift = h % torch.clamp(phase_k - 1, min=1)
            partner = (seed_idx + 1 + shift) % phase_k
            return partner

        def _maybe_permute(X, Y, C):
            """Apply per-example permutation to the current batch.
            Called at the top of each micro_step so every accumulated batch
            sees fresh permutations. No-op when _frac == 0.0.
            """
            if _frac <= 0.0:
                return X, Y, C
            _base_vocab = config.model.vocab_size
            _n_comp = config.experiment.n_compartments
            # ---- seed_anneal schedule path ----
            if (
                _perm_sched == "seed_anneal"
                and _seed_anneal_pool is not None
            ):
                Bsz = X.shape[0]
                _seed_idx = _seed_anneal_seed_indices(Bsz)  # (B,)
                if _perm_mode == "vocab":
                    # Gather rows: (B, V) permutations per example.
                    _p = _seed_anneal_pool.index_select(0, _seed_idx)
                    _Xp = torch.gather(_p, 1, X.clamp(min=0).long())
                    _Yp = torch.gather(_p, 1, Y.clamp(min=0).long())
                    X = torch.where(X >= 0, _Xp.to(X.dtype), X)
                    Y = torch.where(Y >= 0, _Yp.to(Y.dtype), Y)
                elif _perm_mode == "compartment":
                    Bsz, Tsz = X.shape
                    _sigma = _seed_anneal_pool.index_select(0, _seed_idx)  # (B, n_comp)
                    _natural_comp = torch.zeros(
                        (Bsz, Tsz + 1), dtype=torch.long, device=device
                    )
                    _natural_comp[:, :Tsz] = (X // _base_vocab).long()
                    _natural_comp[:, Tsz] = (Y[:, -1] // _base_vocab).long()
                    # Ignore tokens (X<0) and translation tokens (X==trans_id
                    # -> X//V == n_comp) produce out-of-range gather indices.
                    # Clamp so the gather stays in bounds; the _keep_X/Y masks
                    # below discard the resulting junk at those positions.
                    _natural_comp_idx = _natural_comp.clamp(0, _n_comp - 1)
                    _new_comp = torch.gather(
                        _sigma, 1, _natural_comp_idx
                    ).to(X.dtype)
                    _trans_id = _base_vocab * _n_comp
                    _base_X = X % _base_vocab
                    _base_Y = Y % _base_vocab
                    _new_X = _base_X + _new_comp[:, :Tsz] * _base_vocab
                    _new_Y = _base_Y + _new_comp[:, 1:] * _base_vocab
                    _keep_X = (X < 0) | (X == _trans_id)
                    _keep_Y = (Y < 0) | (Y == _trans_id)
                    X = torch.where(_keep_X, X, _new_X)
                    Y = torch.where(_keep_Y, Y, _new_Y)
                    if C is not None:
                        C = torch.where(
                            _keep_X, C, _new_comp[:, :Tsz].to(C.dtype)
                        )
                return X, Y, C
            if _perm_mode == "vocab":
                # Vectorized B-independent partial vocab permutations.
                # Per-example construction:
                #   1. Sample subset S_b of size n_perm from [0, V) (random).
                #   2. Sample a within-S_b shuffle pi_b.
                #   3. Build p_b: identity, with p_b[S_b[k]] = S_b[pi_b[k]].
                #   4. Apply p_b to X_b and Y_b via gather.
                Bsz, _ = X.shape
                _n_perm = max(1, int(_frac * _base_vocab))
                # (B, n_perm): random subset S of [0, V) per example
                _S = torch.argsort(
                    torch.rand((Bsz, _base_vocab), device=device), dim=1
                )[:, :_n_perm]
                # (B, n_perm): within-S shuffle indices
                _pi = torch.argsort(
                    torch.rand((Bsz, _n_perm), device=device), dim=1
                )
                # _targets[b, k] = S[b, pi[b, k]]
                _targets = torch.gather(_S, 1, _pi)
                # Build p of shape (B, V): identity, scattered at S positions
                _p = (
                    torch.arange(_base_vocab, device=device)
                    .unsqueeze(0)
                    .expand(Bsz, _base_vocab)
                    .contiguous()
                )
                _p.scatter_(1, _S, _targets)
                # Apply p to X, Y via gather; preserve negative (ignore) tokens.
                _Xp = torch.gather(_p, 1, X.clamp(min=0).long())
                _Yp = torch.gather(_p, 1, Y.clamp(min=0).long())
                X = torch.where(X >= 0, _Xp.to(X.dtype), X)
                Y = torch.where(Y >= 0, _Yp.to(Y.dtype), Y)
            elif _perm_mode == "compartment":
                Bsz, Tsz = X.shape
                _natural_comp = torch.zeros(
                    (Bsz, Tsz + 1), dtype=X.dtype, device=device
                )
                _natural_comp[:, :Tsz] = X // _base_vocab
                _natural_comp[:, Tsz] = Y[:, -1] // _base_vocab
                _rand_mask = torch.rand(
                    (Bsz, Tsz + 1), device=device
                ) < _frac
                _random_comp = torch.randint(
                    0, _n_comp, (Bsz, Tsz + 1), device=device
                ).to(X.dtype)
                _new_comp = torch.where(_rand_mask, _random_comp, _natural_comp)
                _trans_id = _base_vocab * _n_comp
                _base_X = X % _base_vocab
                _base_Y = Y % _base_vocab
                _new_X = _base_X + _new_comp[:, :Tsz] * _base_vocab
                _new_Y = _base_Y + _new_comp[:, 1:] * _base_vocab
                _keep_X = (X < 0) | (X == _trans_id)
                _keep_Y = (Y < 0) | (Y == _trans_id)
                X = torch.where(_keep_X, X, _new_X)
                Y = torch.where(_keep_Y, Y, _new_Y)
                if C is not None:
                    C = torch.where(_keep_X, C, _new_comp[:, :Tsz].to(C.dtype))
            return X, Y, C

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        last_loss: Optional[torch.Tensor] = None
        last_dann_loss: Optional[float] = None
        last_infonce_loss: Optional[float] = None
        last_lm_loss_float: Optional[float] = None
        last_icl_loss_float: Optional[float] = None
        # stays None when grad_clip == 0 (clipping off), in which case no norm is
        # computed and there is nothing to log -- not a silent zero.
        last_grad_norm: Optional[float] = None
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                is_last_micro = micro_step == gradient_accumulation_steps - 1
                cast(DDP, model).require_backward_grad_sync = is_last_micro
                if dann_enabled and dann_module is not None and isinstance(dann_module, DDP):
                    cast(DDP, dann_module).require_backward_grad_sync = is_last_micro
            # ---- seed-anneal pair-cossim mode ----
            # When permutation_pair_lambda > 0, during the anneal window, and
            # k >= 2 for the current phase: build a 2B doubled batch (same
            # underlying content under two seeds), forward once, and add
            # `lambda * (1 - mean(cos(H_a, H_b)))` to the LM loss. Outside
            # those conditions, fall through to the single-batch path.
            _pair_lambda_val = float(getattr(config.experiment, "permutation_pair_lambda", 0.0))
            _pair_active = (
                _pair_lambda_val > 0.0
                and _perm_sched == "seed_anneal"
                and _seed_anneal_pool is not None
                and _perm_mode in ("vocab", "compartment")
            )
            _pair_last_cos = None  # for wandb logging
            if _pair_active:
                _Bsz = X.shape[0]
                _pair_start = int(config.experiment.permutation_anneal_start_iter)
                if iter_num < _pair_start:
                    _pair_active = False  # pre-anneal (delayed start)
            if _pair_active:
                _j_rep = ddp_world_size * (
                    max(0, iter_num - _pair_start) * gradient_accumulation_steps * _Bsz
                    + micro_step * _Bsz
                )
                if _seed_anneal_pool_override_active:
                    # Fixed-pool ablation: k is constant at pool_size for the
                    # whole anneal window (L_examples).
                    if _j_rep >= _seed_anneal_L_examples:
                        _pair_active = False  # post-anneal
                    else:
                        _k_rep = _seed_anneal_n
                        if _k_rep < 2:
                            _pair_active = False  # pool too small for a pair
                else:
                    _phase_rep = _j_rep // _seed_anneal_n
                    if _j_rep >= _seed_anneal_n * _seed_anneal_n:
                        _pair_active = False  # post-anneal
                    else:
                        _k_rep = _seed_anneal_n - _phase_rep
                        if _k_rep < 2:
                            _pair_active = False  # last phase, no valid partner

            if _pair_active:
                _Bsz = X.shape[0]
                _seed_a = _seed_anneal_seed_indices(_Bsz)  # (B,)
                _phase_k_tensor = torch.full(
                    (_Bsz,), int(_k_rep), device=device, dtype=torch.long
                )
                _seed_b = _seed_anneal_partner(_seed_a, _phase_k_tensor)
                X_a, Y_a, C_a = _apply_seed_anneal_perm(X, Y, C, _seed_a)
                X_b, Y_b, C_b = _apply_seed_anneal_perm(X, Y, C, _seed_b)
                X_pair = torch.cat([X_a, X_b], dim=0)
                Y_pair = torch.cat([Y_a, Y_b], dim=0)
                C_pair = torch.cat([C_a, C_b], dim=0) if C is not None else None
                with ctx:
                    torch.compiler.cudagraph_mark_step_begin()
                    logits, lm_loss, H_pair = model(
                        X_pair, Y_pair, compartment_ids=C_pair,
                        return_last_hidden=True,
                    )
                    last_lm_loss_float = lm_loss.item()
                    lm_loss = lm_loss / gradient_accumulation_steps
                    _H_a = H_pair[:_Bsz]
                    _H_b = H_pair[_Bsz:]
                    _cos = torch.nn.functional.cosine_similarity(_H_a, _H_b, dim=-1)
                    _cos_mean = _cos.mean()
                    _pair_last_cos = float(_cos_mean.item())
                    cossim_loss = (1.0 - _cos_mean) / gradient_accumulation_steps
                    loss = lm_loss + _pair_lambda_val * cossim_loss
                last_loss = loss
                # Mirror the tail of the microstep body: prefetch next batch,
                # then backward. Skips the ICL/DANN/InfoNCE branches (pair
                # mode is only implemented for vocab-mode c=1 LM).
                batch = train_loader.next_batch()
                if isinstance(batch, tuple) and len(batch) == 3:
                    X, Y, C = batch
                else:
                    X, Y = batch  # type: ignore[misc]
                    C = None
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                C = C.to(device, non_blocking=True) if C is not None else None
                scaler.scale(loss).backward()
                continue

            # Apply per-example permutation INSIDE the accumulation loop so every
            # microstep batch gets a fresh permutation (otherwise only the first
            # microstep is permuted and the remaining grad_accum-1 microsteps
            # train on natural data).
            X, Y, C = _maybe_permute(X, Y, C)
            # Compute ICL hashed views (per-example fresh salt) for the
            # dual-stream input embedding and the ICL prediction head.
            _icl_idx = None
            _icl_targets = None
            _icl_mask_t = None
            if _icl_mode == "dual_stream":
                _salt = torch.randint(
                    1, (1 << 31) - 1, (X.shape[0],), device=device, dtype=X.dtype
                )
                _icl_idx = _icl_hash(X.clamp(min=0), _salt)
                _icl_targets = _icl_hash(Y.clamp(min=0), _salt)
                # Preserve ignore-index (-1) positions in the targets
                _icl_idx = torch.where(X >= 0, _icl_idx, X)
                _icl_targets = torch.where(Y >= 0, _icl_targets, Y)
                if _icl_mask_p > 0.0:
                    _icl_mask_t = (
                        torch.rand(X.shape, device=device) < _icl_mask_p
                    )
            with ctx:
                torch.compiler.cudagraph_mark_step_begin()
                if dann_enabled:
                    logits, lm_loss, layer_outputs = model(X, Y, compartment_ids=C)
                elif _icl_mode == "dual_stream":
                    logits, lm_loss, _, icl_loss = model(
                        X, Y, compartment_ids=C,
                        icl_idx=_icl_idx, icl_targets=_icl_targets,
                        icl_mask=_icl_mask_t,
                    )
                else:
                    logits, lm_loss = model(X, Y, compartment_ids=C)
                last_lm_loss_float = lm_loss.item()
                lm_loss = lm_loss / gradient_accumulation_steps
                if _icl_mode == "dual_stream":
                    last_icl_loss_float = icl_loss.item()
                    icl_loss = icl_loss / gradient_accumulation_steps
                if dann_enabled and C is not None and dann_module is not None:
                    domain_labels = C[:, 0]  # per-sequence compartment
                    dann_loss = dann_module(layer_outputs, domain_labels, dann_lambda)
                    last_dann_loss = dann_loss.item()
                    loss = lm_loss + dann_loss / gradient_accumulation_steps
                else:
                    loss = lm_loss
                if _icl_mode == "dual_stream":
                    loss = loss + _icl_lambda * icl_loss
            # InfoNCE alignment loss (auxiliary). Computed every infonce_every microsteps.
            if (
                infonce_pool is not None
                and infonce_layer is not None
                and (micro_step % config.experiment.infonce_every == 0)
            ):
                if infonce_mode == "wikimatrix":
                    en_t, en_m, zh_t, zh_m = infonce_pool.sample(config.experiment.infonce_n)
                    en_t_t = torch.from_numpy(en_t).to(device, non_blocking=True)
                    zh_t_t = torch.from_numpy(zh_t).to(device, non_blocking=True)
                    en_m_t = torch.from_numpy(en_m).to(device, non_blocking=True)
                    zh_m_t = torch.from_numpy(zh_m).to(device, non_blocking=True)
                    from src.infonce import compute_infonce_loss
                    nce_loss = compute_infonce_loss(
                        raw_model, en_t_t, en_m_t, zh_t_t, zh_m_t,
                        capture_layer=infonce_layer,
                        tau=config.experiment.infonce_tau,
                        ctx=ctx,
                    )
                elif infonce_mode == "compartment":
                    # compartment mode: pick two distinct compartments
                    n_comp = config.experiment.n_compartments
                    ci = int(_infonce_pair_rng.integers(0, n_comp))
                    cj = int(_infonce_pair_rng.integers(0, n_comp - 1))
                    if cj >= ci:
                        cj += 1
                    x_ci, x_cj = infonce_pool.sample_pairs(
                        config.experiment.infonce_n, ci, cj
                    )
                    x_ci_t = torch.from_numpy(x_ci).to(device, non_blocking=True)
                    x_cj_t = torch.from_numpy(x_cj).to(device, non_blocking=True)
                    from src.infonce import compute_infonce_compartment_loss
                    nce_loss = compute_infonce_compartment_loss(
                        raw_model, x_ci_t, x_cj_t,
                        capture_layer=infonce_layer,
                        tau=config.experiment.infonce_tau,
                        ctx=ctx,
                    )
                else:  # bio_decl_qa
                    decl_t, decl_m, qa_t, qa_m = infonce_pool.sample(config.experiment.infonce_n)
                    decl_t_t = torch.from_numpy(decl_t).to(device, non_blocking=True)
                    qa_t_t = torch.from_numpy(qa_t).to(device, non_blocking=True)
                    decl_m_t = torch.from_numpy(decl_m).to(device, non_blocking=True)
                    qa_m_t = torch.from_numpy(qa_m).to(device, non_blocking=True)
                    from src.infonce import compute_infonce_loss as _compute_infonce_loss
                    nce_loss = _compute_infonce_loss(
                        raw_model, decl_t_t, decl_m_t, qa_t_t, qa_m_t,
                        capture_layer=infonce_layer,
                        tau=config.experiment.infonce_tau,
                        ctx=ctx,
                    )
                last_infonce_loss = nce_loss.item()
                loss = loss + (config.experiment.infonce_lambda * nce_loss) / gradient_accumulation_steps
            last_loss = loss
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            batch = train_loader.next_batch()
            if isinstance(batch, tuple) and len(batch) == 3:
                X, Y, C = batch
            else:
                X, Y = batch  # type: ignore[misc]
                C = None
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            C = C.to(device, non_blocking=True) if C is not None else None
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        # clip_grad_norm_ returns the PRE-clip total norm, which it has to compute
        # anyway. Keeping it is free and it is the only signal that distinguishes a
        # loss spike caused by a gradient blow-up from one caused by a bad batch --
        # without it every spike looks identical after the fact. Logged, not acted on.
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            all_params = list(model.parameters())
            if dann_enabled and dann_module is not None:
                all_params += list(dann_module.parameters())
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
            )
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = cast(torch.Tensor, last_loss).item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            gn = "" if last_grad_norm is None else f", gnorm {last_grad_norm:.3f}"
            print(
                f"iter {iter_num}: loss {lossf:.4f}{gn}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )
            if wandb_log and master_process:
                log_metrics: dict[str, Any] = {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                    "time_ms": dt * 1000,
                }
                if last_grad_norm is not None:
                    log_metrics["train/grad_norm"] = last_grad_norm
                if dann_enabled and last_dann_loss is not None:
                    log_metrics["train/dann_loss"] = last_dann_loss
                if last_infonce_loss is not None:
                    log_metrics["train/infonce_loss"] = last_infonce_loss
                if config.experiment.permutation_mode != "none":
                    log_metrics["train/permute_frac"] = _permute_active_frac
                    if _permute_active_k is not None:
                        log_metrics["train/permute_k"] = _permute_active_k
                    if _pair_last_cos is not None:
                        log_metrics["train/pair_cos"] = _pair_last_cos
                if _icl_mode == "dual_stream":
                    # train/loss above is the COMBINED loss (lm + lambda*icl)
                    # for backward compat with existing dashboards. We also
                    # log the unweighted pieces.
                    if last_lm_loss_float is not None:
                        log_metrics["train/lm_loss"] = last_lm_loss_float
                    if last_icl_loss_float is not None:
                        log_metrics["train/icl_loss"] = last_icl_loss_float
                wandb_log_or_buffer(log_metrics, step=iter_num)
        iter_num += 1
        local_iter_num += 1

    # Terminal milestone checkpoint.
    #
    # The loop above is `while iter_num < max_iters` and the in-loop save runs
    # BEFORE the increment, so the body never executes with iter_num == max_iters.
    # A run configured with max_iters exactly equal to a milestone (the usual
    # case: max_iters=1000000, and 1000000 is in checkpoint_steps) therefore
    # completes all its optimizer updates and never writes that milestone — its
    # deepest named checkpoint is the previous one (700000), which silently caps
    # evaluation 300k steps short. Runs that DO have step-1000000 are the ones
    # whose max_iters was larger, so they passed through it mid-training.
    #
    # Deliberately duplicated rather than factored out of the in-loop block: this
    # is additive, so a mistake here cannot regress the path every existing run
    # depends on.
    if (
        not master_process
        and iter_num in checkpoint_steps
        and iter_num > 0
        and iter_num in full_state_iters
    ):
        save_dataloader_state(
            _checkpoint_dir(os.path.join(out_dir, "checkpoints"), iter_num),
            train_loader, val_loader, ddp_rank or 0,
        )

    if master_process and iter_num in checkpoint_steps and iter_num > 0:
        ck_root = os.path.join(out_dir, "checkpoints")
        step_dir = _checkpoint_dir(ck_root, iter_num)
        if os.path.exists(os.path.join(step_dir, "model.pt")):
            print(f"terminal checkpoint {os.path.basename(step_dir)} already exists — skipping")
        else:
            os.makedirs(step_dir, exist_ok=True)
            # The end of a run is always full state: it is the point you would
            # extend or fork a decay from later, and unlike an intermediate
            # checkpoint it cannot be recreated by training forward again.
            save_full_state = iter_num in full_state_iters
            print(f"saving terminal checkpoint to {step_dir}")
            named_state = {
                k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                for k, v in raw_model.state_dict().items()
            }
            torch.save(named_state, os.path.join(step_dir, "model.pt"))
            if dann_enabled and dann_module is not None:
                raw_dann = dann_module.module if isinstance(dann_module, DDP) else dann_module
                torch.save(raw_dann.state_dict(), os.path.join(step_dir, "dann_discriminators.pt"))
            if save_full_state:
                torch.save(optimizer.state_dict(), os.path.join(step_dir, "optimizer.pt"))
                save_dataloader_state(step_dir, train_loader, val_loader, ddp_rank or 0)
            with open(os.path.join(step_dir, "trainer_state.json"), "w") as f:
                json.dump(
                    {
                        "iter_num": iter_num,
                        "best_val_loss": float(best_val_loss),
                        "tokens": iter_num * tokens_per_iter,
                        "phase": "annealed" if is_anneal_child else "stable",
                    },
                    f,
                )
            latest = os.path.join(ck_root, "latest")
            should_update_latest = True
            if os.path.islink(latest):
                try:
                    cur_iter = ckpt_utils.checkpoint_iter(
                        os.path.join(ck_root, os.readlink(latest))
                    )
                    if cur_iter is not None:
                        should_update_latest = iter_num >= cur_iter
                except (ValueError, IndexError, OSError):
                    pass  # malformed symlink target — overwrite it
            if should_update_latest:
                tmp_link = latest + f".tmp.{os.getpid()}"
                os.symlink(os.path.relpath(step_dir, ck_root), tmp_link)
                os.replace(tmp_link, latest)

    # Flush any remaining buffered wandb logs at normal termination
    if wandb_log and master_process and wandb_buffer_enabled:
        wandb_flush_buffer()

    if ddp:
        destroy_process_group()
    if active_run_lock is not None:
        active_run_lock.release()


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    main(config)

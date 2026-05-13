import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Sequence, cast

import numpy as np
from tqdm.auto import trange
import torch

from src.config.job_config import JobConfig, Model
from src.config.manager import ConfigManager
from src.model import GPT


def get_base_vocab_size(config: JobConfig) -> int:
    """Get the base vocabulary size from config."""
    base_vocab = config.model.vocab_size
    if base_vocab is None:
        raise ValueError("model.vocab_size is required")
    return int(base_vocab)


def is_uniform_data_source(config: JobConfig) -> bool:
    """Check if the config uses uniform/random data source."""
    return config.data.source == "uniform"


# Simple in-process cache for dataset shards used at eval time.
# This avoids re-reading the same .bin file from disk for every metric/assignment
# while keeping training codepaths unchanged.
_DATA_SHARD_CACHE: dict[str, np.ndarray] = {}


def _build_and_load_model(config: JobConfig, model_file: Path, model_compartments: int):
    """Helper to build model with given compartment count and load checkpoint."""
    base_vocab = config.model.vocab_size
    if base_vocab is None:
        raise ValueError("model.vocab_size is required")

    if config.experiment.permute_tokens_per_compartment:
        vocab = base_vocab + 1
        translation_token_id = base_vocab
    else:
        vocab = base_vocab * model_compartments + 1
        translation_token_id = base_vocab * model_compartments

    gptconf = Model(
        **{
            **asdict(config.model),
            "vocab_size": vocab,
            "embedding_vocab_size": (
                (base_vocab + 1) if config.experiment.shared_token_embeddings else vocab
            ),
            "shared_token_embeddings": bool(config.experiment.shared_token_embeddings),
            "use_compartment_embeddings": bool(
                config.experiment.use_compartment_embeddings
            ),
            "copy_compartment_embeddings": (
                False
                if config.experiment.permute_tokens_per_compartment
                else bool(config.experiment.copy_compartment_embeddings)
            ),
            "copy_compartment_lm_head": (
                False
                if config.experiment.permute_tokens_per_compartment
                else bool(config.experiment.copy_compartment_lm_head)
            ),
            "base_vocab_size": base_vocab,
            "max_compartments": model_compartments,
            "translation_token_id": translation_token_id,
            "weight_tying": (
                False
                if config.experiment.shared_token_embeddings
                else config.model.weight_tying
            ),
        }
    )
    model = GPT(gptconf)
    state_dict = torch.load(model_file, map_location="cpu")
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model


def load_eval_model(config: JobConfig, model_file: Path, device: torch.device | str, dtype: torch.dtype | None = None):
    base_vocab = config.model.vocab_size
    if base_vocab is None:
        raise ValueError("model.vocab_size is required")

    # Default to bfloat16 for flash attention compatibility
    if dtype is None:
        dtype = torch.bfloat16

    # Newer experiments use n_compartments for model sizing, older use max_compartments
    # Try n_compartments first, fall back to max_compartments if load fails
    n_comp = config.experiment.n_compartments
    max_comp = cast(int, config.experiment.max_compartments)

    compartment_options = []
    if n_comp is not None:
        compartment_options.append(n_comp)
    if max_comp not in compartment_options:
        compartment_options.append(max_comp)

    last_error = None
    actual_model_compartments = None
    for model_compartments in compartment_options:
        try:
            model = _build_and_load_model(config, model_file, model_compartments)
            actual_model_compartments = model_compartments
            break
        except RuntimeError as e:
            if "size mismatch" in str(e):
                last_error = e
                continue
            raise
    else:
        # All options failed
        raise last_error if last_error else RuntimeError("Failed to load model")
    model.to(device=device, dtype=dtype)
    model.eval()
    # torch.compile disabled due to triton compatibility issues
    # model = torch.compile(model, dynamic=False)

    # Enable TF32 on capable hardware for faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return model, actual_model_compartments


def load_eval_model_from_checkpoint(
    checkpoint_path: Path, experiment_path: Path, device: torch.device | str, dtype: torch.dtype | None = None
):
    model_file = checkpoint_path / "model.pt"
    config_file = experiment_path / "meta" / "config.json"
    config_manager = ConfigManager()
    with open(config_file, "r") as f:
        config_manager.load_from_dict(json.load(f))
    config_manager.config.model
    config = config_manager.config
    model, actual_model_compartments = load_eval_model(config, model_file, device, dtype=dtype)
    return model, config, actual_model_compartments


def _load_data_shard(filename):
    """
    Load a single .bin shard of token data.

    For evaluation we may construct many `SingleShardAssignedValLoader` instances
    over the same shard path (one per metric/assignment). To avoid repeatedly
    reading the shard from disk, we keep a simple in-memory cache keyed by the
    filename string.
    """
    key = str(filename)
    cached = _DATA_SHARD_CACHE.get(key)
    if cached is not None:
        return cached

    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20251013, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint32
        tokens = np.frombuffer(f.read(), dtype=np.uint32)

    assert len(tokens) == ntok, "number of tokens read does not match header?"
    _DATA_SHARD_CACHE[key] = tokens
    return tokens


@dataclass(frozen=True)
class Assignment:
    # kind=1 → translation example (src → dst); otherwise → compartment example (src only)
    kind: int
    src: int
    dst: int = 0  # ignored for non-translation kinds


class SingleShardAssignedValLoader:
    """
    Single-process validation loader that applies one fixed assignment to every example.

    - Uses _load_data_shard(shard_path) to read a single custom .bin shard.
    - Outputs (x, y, cids) per batch, shaped [B, T] each.
      - For translation (kind==1): inserts translation token at positions 0 and T//2.
      - For compartment (kind!=1): standard next-token targets within the src compartment.
    - Iterates linearly over the shard and stops (no wrap-around).
    """

    def __init__(
        self,
        shard_path: str,
        B: int,
        T: int,
        base_vocab_size: int,
        max_compartments: int,
        assignment: Assignment,
        device: Optional[torch.device | str] = None,
        permute_tokens: bool = False,
        permutations_path: Optional[str] = None,
        permutations: Optional[np.ndarray] = None,
        permute_inputs: bool = True,
    ):
        self.B = B
        self.T = T
        self.base_vocab_size = base_vocab_size
        self.max_compartments = max_compartments
        self.assignment = assignment
        self.device = device

        self.permute_tokens = permute_tokens
        self.permute_inputs = permute_inputs
        self.translation_token_id = (
            base_vocab_size if permute_tokens else base_vocab_size * max_compartments
        )
        # Load permutations if enabled - accept either path or direct array
        self._permutations: Optional[np.ndarray]
        if self.permute_tokens:
            if permutations is not None:
                # Use directly provided permutations array
                perms = permutations
            elif permutations_path is not None:
                perms = np.load(permutations_path)
            else:
                raise ValueError(
                    "permutations_path or permutations must be provided when permute_tokens=True"
                )
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

        self.tokens: np.ndarray = _load_data_shard(
            shard_path
        )  # dtype=uint16, shape=[N]
        assert self.tokens.ndim == 1, "Shard must be a 1D array of token ids"

        if self.assignment.kind == 1:
            # translation requires even T and T >= 4
            assert T % 2 == 0 and T >= 4, (
                "For translation kind, T must be even and >= 4"
            )
            min_needed = T // 2  # tokens consumed per example
        else:
            min_needed = T + 1  # tokens consumed per example

        assert self.tokens.shape[0] >= min_needed, (
            "Shard too small for at least one example"
        )

        self._position = 0

        # Rough upper bound on full batches available
        per_ex_needed = (T // 2) if (self.assignment.kind == 1) else (T + 1)
        total_examples = self.tokens.shape[0] // per_ex_needed
        self.num_batches = max(0, total_examples // B)

    def reset(self) -> None:
        self._position = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._has_next():
            raise StopIteration
        return self.next_batch()

    def _tokens_needed_per_example(self) -> int:
        return (self.T // 2) if (self.assignment.kind == 1) else (self.T + 1)

    def _has_next(self) -> bool:
        need = self.B * self._tokens_needed_per_example()
        return self._position + need <= self.tokens.shape[0]

    def _read_tokens(self, n: int) -> np.ndarray:
        start = self._position
        end = start + n
        if end > self.tokens.shape[0]:
            raise StopIteration
        # Return int64 view for safe offsetting
        out = self.tokens[start:end].astype(np.int64, copy=False)
        self._position = end
        return out

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        half = T // 2
        assign = self.assignment
        src_offset = int(assign.src) * self.base_vocab_size
        dst_offset = int(assign.dst) * self.base_vocab_size

        x = torch.empty((B, T), dtype=torch.long, device=self.device)
        y = torch.empty((B, T), dtype=torch.long, device=self.device)
        cids = torch.empty((B, T), dtype=torch.long, device=self.device)

        if assign.kind == 1:
            # Translation example: [TR] + (half-1 src tokens) + [TR] + (half-1 dst tokens)
            # Match training loader semantics:
            # - cids[0:half] == src compartment
            # - cids[half:T] == dst compartment
            for b in range(B):
                sample = self._read_tokens(half)  # length = half
                cid = np.empty(T, dtype=np.int64)
                cid[:half] = int(assign.src)
                cid[half:] = int(assign.dst)

                if self.permute_tokens and not self.permute_inputs:
                    # Inputs unpermuted, targets in permuted space
                    seq_in = np.empty(T, dtype=np.int64)
                    seq_out = np.empty(T, dtype=np.int64)

                    seq_in[0] = self.translation_token_id
                    seq_in[half] = self.translation_token_id
                    seq_out[0] = self.translation_token_id
                    seq_out[half] = self.translation_token_id

                    src_map = cast(np.ndarray, self._permutations)[int(assign.src)]
                    dst_map = cast(np.ndarray, self._permutations)[int(assign.dst)]

                    # Inputs: raw tokens
                    seq_in[1:half] = sample[: half - 1]
                    seq_in[half + 1 :] = sample[: half - 1]
                    # Targets: permuted tokens
                    seq_out[1:half] = src_map[sample[: half - 1]]
                    seq_out[half + 1 :] = dst_map[sample[: half - 1]]
                else:
                    # Original behavior: inputs and targets both in the same space
                    seq = np.empty(T, dtype=np.int64)
                    seq[0] = self.translation_token_id
                    seq[half] = self.translation_token_id

                    if self.permute_tokens:
                        src_map = cast(np.ndarray, self._permutations)[int(assign.src)]
                        dst_map = cast(np.ndarray, self._permutations)[int(assign.dst)]
                        seq[1:half] = src_map[sample[: half - 1]]
                        seq[half + 1 :] = dst_map[sample[: half - 1]]
                    else:
                        seq[1:half] = sample[: half - 1] + src_offset
                        seq[half + 1 :] = sample[: half - 1] + dst_offset

                    seq_in = seq
                    seq_out = seq

                cids[b] = torch.from_numpy(cid).to(device=self.device)
                x[b] = torch.from_numpy(seq_in).to(device=self.device)

                y_seq = np.empty(T, dtype=np.int64)
                y_seq[:-1] = seq_out[1:]
                y_seq[-1] = -1
                y[b] = torch.from_numpy(y_seq).to(device=self.device)
        else:
            # Compartment example: take T+1 tokens, offset by src
            for b in range(B):
                sample = self._read_tokens(T + 1)  # length = T+1
                if self.permute_tokens:
                    src_map = cast(np.ndarray, self._permutations)[int(assign.src)]
                    if self.permute_inputs:
                        x_seq = src_map[sample[:T]]
                    else:
                        x_seq = sample[:T]
                    perm_seq = src_map[sample[0 : T + 1]]
                    y_seq = perm_seq[1:]
                else:
                    x_seq = sample[:T] + src_offset
                    y_seq = sample[1 : T + 1] + src_offset

                x[b] = torch.from_numpy(x_seq.astype(np.int64)).to(device=self.device)
                y[b] = torch.from_numpy(y_seq.astype(np.int64)).to(device=self.device)
                cids[b].fill_(int(assign.src))

        return x, y, cids


class UniformAssignedValLoader:
    """
    Validation loader that generates uniform random tokens and applies a fixed assignment.

    This mirrors SingleShardAssignedValLoader but generates random tokens on-the-fly
    instead of reading from a pretokenized file. Used for evaluating models trained
    on uniform/random data.

    - Outputs (x, y, cids) per batch, shaped [B, T] each.
      - For translation (kind==1): inserts translation token at positions 0 and T//2.
      - For compartment (kind!=1): standard next-token targets within the src compartment.
    """

    def __init__(
        self,
        B: int,
        T: int,
        base_vocab_size: int,
        max_compartments: int,
        assignment: Assignment,
        seed: int = 0,
        num_batches: int = 64,
        device: Optional[torch.device | str] = None,
        permute_tokens: bool = False,
        permutations_path: Optional[str] = None,
        permutations: Optional[np.ndarray] = None,
        permute_inputs: bool = True,
        token_probs: Optional[np.ndarray] = None,
    ):
        self.B = B
        self.T = T
        self.base_vocab_size = base_vocab_size
        self.max_compartments = max_compartments
        self.assignment = assignment
        self.seed = seed
        self.num_batches = num_batches
        self.device = device
        self._token_probs = token_probs

        self.permute_tokens = permute_tokens
        self.permute_inputs = permute_inputs
        self.translation_token_id = (
            base_vocab_size if permute_tokens else base_vocab_size * max_compartments
        )

        # Load permutations if enabled - accept either path or direct array
        self._permutations: Optional[np.ndarray]
        if self.permute_tokens:
            if permutations is not None:
                # Use directly provided permutations array
                perms = permutations
            elif permutations_path is not None:
                perms = np.load(permutations_path)
            else:
                raise ValueError(
                    "permutations_path or permutations must be provided when permute_tokens=True"
                )
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

        if self.assignment.kind == 1:
            # translation requires even T and T >= 4
            assert T % 2 == 0 and T >= 4, (
                "For translation kind, T must be even and >= 4"
            )

        # Initialize RNG
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._batch_count = 0

    def reset(self) -> None:
        self._rng = np.random.Generator(np.random.PCG64(self.seed))
        self._batch_count = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._batch_count >= self.num_batches:
            raise StopIteration
        return self.next_batch()

    def _generate_tokens(self, n: int) -> np.ndarray:
        """Generate n random tokens in [0, base_vocab_size).

        Uses uniform distribution by default, or token_probs if provided.
        """
        if self._token_probs is not None:
            return self._rng.choice(
                self.base_vocab_size, size=n, replace=True, p=self._token_probs
            ).astype(np.int64)
        return self._rng.integers(0, self.base_vocab_size, size=n, dtype=np.int64)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        half = T // 2
        assign = self.assignment
        src_offset = int(assign.src) * self.base_vocab_size
        dst_offset = int(assign.dst) * self.base_vocab_size

        x = torch.empty((B, T), dtype=torch.long, device=self.device)
        y = torch.empty((B, T), dtype=torch.long, device=self.device)
        cids = torch.empty((B, T), dtype=torch.long, device=self.device)

        if assign.kind == 1:
            # Translation example: [TR] + (half-1 src tokens) + [TR] + (half-1 dst tokens)
            for b in range(B):
                sample = self._generate_tokens(half)  # length = half
                cid = np.empty(T, dtype=np.int64)
                cid[:half] = int(assign.src)
                cid[half:] = int(assign.dst)

                if self.permute_tokens and not self.permute_inputs:
                    # Inputs unpermuted, targets in permuted space
                    seq_in = np.empty(T, dtype=np.int64)
                    seq_out = np.empty(T, dtype=np.int64)

                    seq_in[0] = self.translation_token_id
                    seq_in[half] = self.translation_token_id
                    seq_out[0] = self.translation_token_id
                    seq_out[half] = self.translation_token_id

                    src_map = cast(np.ndarray, self._permutations)[int(assign.src)]
                    dst_map = cast(np.ndarray, self._permutations)[int(assign.dst)]

                    # Inputs: raw tokens
                    seq_in[1:half] = sample[: half - 1]
                    seq_in[half + 1 :] = sample[: half - 1]
                    # Targets: permuted tokens
                    seq_out[1:half] = src_map[sample[: half - 1]]
                    seq_out[half + 1 :] = dst_map[sample[: half - 1]]
                else:
                    # Original behavior: inputs and targets both in the same space
                    seq = np.empty(T, dtype=np.int64)
                    seq[0] = self.translation_token_id
                    seq[half] = self.translation_token_id

                    if self.permute_tokens:
                        src_map = cast(np.ndarray, self._permutations)[int(assign.src)]
                        dst_map = cast(np.ndarray, self._permutations)[int(assign.dst)]
                        seq[1:half] = src_map[sample[: half - 1]]
                        seq[half + 1 :] = dst_map[sample[: half - 1]]
                    else:
                        seq[1:half] = sample[: half - 1] + src_offset
                        seq[half + 1 :] = sample[: half - 1] + dst_offset

                    seq_in = seq
                    seq_out = seq

                cids[b] = torch.from_numpy(cid).to(device=self.device)
                x[b] = torch.from_numpy(seq_in).to(device=self.device)

                y_seq = np.empty(T, dtype=np.int64)
                y_seq[:-1] = seq_out[1:]
                y_seq[-1] = -1
                y[b] = torch.from_numpy(y_seq).to(device=self.device)
        else:
            # Compartment example: generate T+1 tokens, offset by src
            for b in range(B):
                sample = self._generate_tokens(T + 1)  # length = T+1
                if self.permute_tokens:
                    src_map = cast(np.ndarray, self._permutations)[int(assign.src)]
                    if self.permute_inputs:
                        x_seq = src_map[sample[:T]]
                    else:
                        x_seq = sample[:T]
                    perm_seq = src_map[sample[0 : T + 1]]
                    y_seq = perm_seq[1:]
                else:
                    x_seq = sample[:T] + src_offset
                    y_seq = sample[1 : T + 1] + src_offset

                x[b] = torch.from_numpy(x_seq.astype(np.int64)).to(device=self.device)
                y[b] = torch.from_numpy(y_seq.astype(np.int64)).to(device=self.device)
                cids[b].fill_(int(assign.src))

        self._batch_count += 1
        return x, y, cids


@dataclass(frozen=True)
class TokensAtExamples:
    """
    Aggregated token counts at a given number of seen examples.

    - examples: the number of examples processed
    - tokens_per_compartment: list of length = num_compartments (from assignments header)
    """

    examples: int
    tokens_per_compartment: list[int]


def _decode_assignment_record(word: np.uint64) -> tuple[int, int, int]:
    """Decode a single 64-bit assignment record into (kind, src, dst).

    Layout matches AssignmentsDataLoader._decode_record in train.py.
    kind in low 8 bits; src and dst are 16-bit fields at bit positions 16 and 32.
    """
    w = int(word)
    kind = w & 0xFF
    src = (w >> 16) & 0xFFFF
    dst = (w >> 32) & 0xFFFF
    return kind, src, dst


def token_counts_at_examples(
    assignments_file: str | Path,
    example_counts: Sequence[int],
    T: int,
    expected_num_compartments: int | None = None,
    resume_examples: int | None = None,
    resume_tokens_per_compartment: Sequence[int] | None = None,
) -> list[TokensAtExamples]:
    """
    Compute cumulative per-compartment token counts after N examples, for multiple N values.

    - Translation examples (kind == 1) contribute (T//2 - 1) tokens to both src and dst compartments.
    - Regular examples contribute T tokens to the src compartment.

    The computation is incremental across the sorted example_counts; it stops once the
    largest requested count is reached (or we exhaust the file records).

    Args:
        assignments_file: Path to the .assignments binary file ("TCASSIGN" format).
        example_counts: Iterable of example counts (will be sorted defensively).
        T: Block size used to form examples. Must be even and >= 4.
        expected_num_compartments: Optional sanity check against header value.
        resume_examples: Optional number of examples that have already been processed.
        resume_tokens_per_compartment: Token counts corresponding to resume_examples.

    Returns:
        A list of TokensAtExamples, one per requested example count (sorted ascending).
    """
    assert T % 2 == 0 and (T // 2) >= 2, "T must be even and >= 4"

    if (resume_examples is None) ^ (resume_tokens_per_compartment is None):
        raise ValueError(
            "resume_examples and resume_tokens_per_compartment must both be provided together"
        )

    thresholds = sorted(int(x) for x in example_counts)
    if len(thresholds) == 0:
        return []

    max_requested = thresholds[-1]
    half = T // 2
    tokens_per_side_translation = half - 1

    with open(assignments_file, "rb") as f:
        header = f.read(32)
        magic, version, rec_size, flags, num_compartments, num_records, seed = (
            struct.unpack("<8sBBHIQQ", header)
        )
        assert magic == b"TCASSIGN", "assignments magic mismatch"
        assert version == 1 and rec_size == 8, (
            "unsupported assignments version/record size"
        )
        if expected_num_compartments is not None:
            assert num_compartments == expected_num_compartments, (
                f"assignments num_compartments {num_compartments} != expected {expected_num_compartments}"
            )

        # Only read exactly the declared number of records
        records_bytes = f.read(num_records * rec_size)
        assert len(records_bytes) == num_records * rec_size, (
            "assignments file truncated: fewer records than declared in header"
        )
        records = np.frombuffer(records_bytes, dtype=np.uint64)

    num_compartments_int = int(num_compartments)
    tokens_by_compartment = [0 for _ in range(num_compartments_int)]

    results: list[TokensAtExamples] = []
    if resume_examples is not None and resume_tokens_per_compartment is not None:
        if len(resume_tokens_per_compartment) != num_compartments_int:
            raise ValueError(
                "resume_tokens_per_compartment length does not match num_compartments"
            )
        start_examples = max(0, min(int(resume_examples), int(num_records)))
        tokens_by_compartment = list(resume_tokens_per_compartment)
    else:
        start_examples = 0
    examples_seen = start_examples
    t_index = 0

    # Handle thresholds that are 0 upfront
    while t_index < len(thresholds) and thresholds[t_index] <= examples_seen:
        results.append(
            TokensAtExamples(
                examples=thresholds[t_index],
                tokens_per_compartment=tokens_by_compartment.copy(),
            )
        )
        t_index += 1

    # Process records incrementally up to the needed point
    start_index = start_examples
    max_process = min(max_requested, int(num_records))
    for i in trange(start_index, max_process):
        kind, src, dst = _decode_assignment_record(records[i])
        if kind == 1:
            tokens_by_compartment[int(src)] += tokens_per_side_translation
            tokens_by_compartment[int(dst)] += tokens_per_side_translation
        else:
            tokens_by_compartment[int(src)] += T

        examples_seen += 1

        while t_index < len(thresholds) and thresholds[t_index] == examples_seen:
            results.append(
                TokensAtExamples(
                    examples=thresholds[t_index],
                    tokens_per_compartment=tokens_by_compartment.copy(),
                )
            )
            t_index += 1

        if t_index >= len(thresholds):
            break

    # If caller requested more examples than exist, return final counts for remaining thresholds
    while t_index < len(thresholds):
        print(f"Requested more examples than exist: {thresholds[t_index]}")
        results.append(
            TokensAtExamples(
                examples=thresholds[t_index],
                tokens_per_compartment=tokens_by_compartment.copy(),
            )
        )
        t_index += 1

    return results

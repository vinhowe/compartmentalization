# %%
from pathlib import Path
from tqdm import tqdm
import sys
import re
import json
import glob
import numpy as np
import math
import torch.nn.functional as F
import torch

# Ensure repo root is importable when running from the experiment directory
sys.path.append("..")

from src.model import GPT
from src.config.manager import ConfigManager
from data.data_common import memmap_datafile, read_datafile_header

# %%

# %%
EXPERIMENT_DIR = Path("../out/memorize-replication/small-prototypes")

# %%
print(*[str(p.name) for p in EXPERIMENT_DIR.glob("*")], sep="\n")

# %%

# n_layer, d_model, run id
scaling_table = [
    # [
    #     1,
    #     32,
    #     None
    # ],
    [
        4,
        32,
        # Very low learning rate
        # "2025-09-19T17-08-54Z__single-domain-4-256-prototype__a4f5842c__s64__c13d8b4__14cc2b65",
        # Higher learning rate (6e-4)
        "2025-09-19T17-49-03Z__single-domain-4-256-prototype__f779470a__s64__c13d8b4__c6086cf9",
    ]
]

# %%
# experiment = "2025-09-18T19-51-43Z__single-domain-80k-prototype__49346ed7__s1024__7cbbf08__ac884235"
experiment = scaling_table[0][2]

# %%
experiment_dir = EXPERIMENT_DIR / experiment
checkpoint_dir = experiment_dir / "checkpoints"

# %%
# Match the pattern step-\d+ to get a list of step numbers
step_numbers = sorted(
    [
        int(match.group(1))
        for p in checkpoint_dir.glob("*")
        if (match := re.search(r"step-(\d+)", str(p.name))) is not None
    ]
)

# %%
selected_checkpoint_number = step_numbers[-1]

# %%
step_dir = checkpoint_dir / f"step-{selected_checkpoint_number:06d}"

# %%
model_file = step_dir / "model.pt"
config_file = experiment_dir / "meta" / "config.toml"

# %%

config_manager = ConfigManager()
config_manager.load_from_dict(json.load(open(config_file)))


# %%
device = "cuda:7"

# %%
model = GPT(config_manager.config.model)
model.to(device)
model.eval()

# %% load checkpoint weights
state_dict = torch.load(model_file, map_location="cpu")
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)
# For eval, compiling can add significant overhead and is usually unnecessary.
# Comment out by default; re-enable if it helps for your GPU/runtime.
model = torch.compile(model, dynamic=False)

# Enable TF32 on capable hardware for faster matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

"""
Evaluate across all headered shards matching the train glob from the config.
We step windows of length block_size with stride block_size within each shard.
"""

# Resolve dataset glob from config (relative to repo root)
train_glob = config_manager.config.data.train_bin
glob_pattern = str(Path("..") / train_glob)
shard_files = sorted(glob.glob(glob_pattern))
assert len(shard_files) > 0, f"No shards found for pattern: {glob_pattern}"

# Use config for sizes
block_size = int(config_manager.config.model.block_size)
# Use a much higher batch size for evaluation
batch_size = 2048 * 8
_cfg_vocab = config_manager.config.model.vocab_size
_model_vocab = getattr(model.config, "vocab_size", None)
if isinstance(_cfg_vocab, int) and _cfg_vocab > 0:
    vocab_size = _cfg_vocab
elif isinstance(_model_vocab, int) and _model_vocab > 0:
    vocab_size = _model_vocab
else:
    raise ValueError("vocab_size must be defined in config or model")
ln_vocab = math.log(vocab_size)

# Total training batches target for extrapolation
# TOTAL_TRAINING_BATCHES = int(1e6)

# Target: process as many  (batch_size x block_size) batches as the checkpoint step
target_batches = int(selected_checkpoint_number)

# Pre-compute total available batches across shards for progress bar
total_windows = 0
for fname in shard_files:
    hdr = read_datafile_header(fname)
    L = int(hdr["ntok"])
    max_start = L - block_size - 1
    if max_start >= 0:
        total_windows += (max_start // block_size) + 1
max_available_batches = (total_windows + batch_size - 1) // batch_size
# total_batches = min(max_available_batches, target_batches)
total_batches = max_available_batches

sum_unintended_mem_nats = 0.0
processed_batches = 0

with torch.inference_mode():
    pbar = tqdm(total=total_batches, desc="Evaluating", unit="batch")
    done = False
    for fname in shard_files:
        if done:
            break
        data_mm, hdr = memmap_datafile(fname)
        L = int(hdr["ntok"])
        max_start = L - block_size - 1
        if max_start < 0:
            continue
        start_indices = np.arange(0, max_start + 1, block_size, dtype=np.int64)
        num_windows = int(start_indices.shape[0])

        for s in range(0, num_windows, batch_size):
            e = min(s + batch_size, num_windows)
            B = e - s
            # Because windows are non-overlapping with stride=block_size, each batch
            # maps to one contiguous slice of the memmap. Use reshape to build [B, T]
            # without per-sample Python loops.
            start_tok = int(start_indices[s])
            x_np = data_mm[start_tok : start_tok + B * block_size].reshape(
                B, block_size
            )
            y_np = data_mm[start_tok + 1 : start_tok + 1 + B * block_size].reshape(
                B, block_size
            )

            # Prepare pinned-memory CPU tensors to speed up H2D copies
            X_cpu = torch.from_numpy(x_np.astype(np.int64, copy=False)).pin_memory()
            Y_cpu = torch.from_numpy(y_np.astype(np.int64, copy=False)).pin_memory()
            X = X_cpu.to(device, non_blocking=True)
            Y = Y_cpu.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Ask model for full-sequence logits by providing targets (loss is ignored)
                logits, _ = model(X, Y)
                V = logits.size(-1)
                # Compute per-token negative log-likelihood without flattening
                # logits: [B, T, V] -> [B, V, T]
                # per_tok_nll = F.cross_entropy(
                #     logits.permute(0, 2, 1), Y, reduction="sum"
                # )  # [B, T]
                per_tok_nll = F.cross_entropy(
                    logits.view(-1, V), Y.view(-1), reduction="none"
                ).view(B, block_size)  # [B, T]

            # # print(logits.shape, block_size)
            # seq_ref = (logits.shape[0] * logits.shape[1]) * ln_vocab  # scalar
            # seq_contrib = seq_ref - per_tok_nll  # nats per sequence
            # sum_unintended_mem_nats += float(seq_contrib.sum().item())
            nll_seq = per_tok_nll.sum(dim=1)  # [B]
            uniform_nll_seq = block_size * ln_vocab
            best_nll_seq = torch.minimum(
                nll_seq, torch.full_like(nll_seq, uniform_nll_seq)
            )
            sum_unintended_mem_nats += (uniform_nll_seq - best_nll_seq).sum().item()

            # logits, _ = model(X, Y)
            # V = logits.size(-1)
            # per_token_nll = F.cross_entropy( logits.view(-1, V), Y.view(-1), reduction="none" ).view(X.size(0), X.size(1))
            # contrib = torch.clamp(ln_vocab - per_token_nll, min=0.0) # nats
            # sum_unintended_mem_nats += float(contrib.sum().item())

            processed_batches += 1
            pbar.update(1)
            pbar.set_postfix(bits=sum_unintended_mem_nats / np.log(2.0))
            if processed_batches >= target_batches:
                done = True
                break
    pbar.close()

if processed_batches < target_batches:
    print(
        f"Warning: requested {target_batches} batches, only processed {processed_batches} available batches."
    )

total_unintended_mem_bits = sum_unintended_mem_nats / math.log(2.0)
num_params = int(model.get_num_params())
bits_per_param = total_unintended_mem_bits / max(1, num_params)
print(
    f"Unintended memorization: {total_unintended_mem_bits:.6f} bits\n"
    f"bits/param: {bits_per_param:.9f} (params: {num_params})"
)

# # Extrapolate bits to the full training budget
# if processed_batches > 0:
#     extrapolation_factor = TOTAL_TRAINING_BATCHES / float(processed_batches)
#     extrapolated_total_bits = total_unintended_mem_bits * extrapolation_factor
#     extrapolated_bits_per_param = extrapolated_total_bits / max(1, num_params)
#     print(
#         f"Extrapolated ({TOTAL_TRAINING_BATCHES} batches): {extrapolated_total_bits:.6f} bits\n"
#         f"bits/param: {extrapolated_bits_per_param:.9f}"
#     )
# else:
#     print("Extrapolated: N/A (no batches processed)")

# %%

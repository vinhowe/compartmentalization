#!/usr/bin/env python3
"""Generate the post-NeurIPS scaling-ladder configs.

Normative reference: docs/experimental-protocol.md. Nothing here is a choice
made in this script -- every value is either read off that document or derived
from it. If the two disagree, the document wins and this script is the bug.

Emits two families into config/ladder-v2/:

  lrsweep-*   R1 and R4 x {c=1, c=8} x 5 learning rates, 3B tokens.
              Sets the LR rule for the ladder. Must be run and read FIRST.
  rung-*      6 rungs x {c1, c8, c1-padded}, 30B tokens.
              learning_rate is a PLACEHOLDER until the sweep lands.

Vocabulary arithmetic (train.py:1701, under schema v2 where permutation and
token-tying are both off):

    composite_vocab = model.vocab_size * experiment.n_compartments + 1

so the three arms at a rung are

    c1          vocab_size=16384,  n_compartments=1  -> 16385  rows
    c8          vocab_size=16384,  n_compartments=8  -> 131073 rows
    c1-padded   vocab_size=131072, n_compartments=1  -> 131073 rows

The padded arm is an EXACT parameter match to c8, which is what makes
L(c8) - L(c1_padded) isolate compartmentalization from the cost of the
parameters themselves.
"""

from __future__ import annotations

import argparse
import pathlib

# ---------------------------------------------------------------- constants

BASE_VOCAB = 16384
C_TREAT = 8                      # the compartment count under test
TOKENS_PER_STEP = 2048 * 1024    # 2,097,152 -- 2048 sequences of 1024 tokens
WARMUP_ITERS = 300               # ABSOLUTE, never a fraction of the budget
SEED = 1024
WORLD_SIZE = 8                   # grad_accum must be divisible by this

# Pinned to the 300B horizon so 30B/100B/300B share ONE assignment array.
# Deriving it from max_iters re-randomises every example's compartment when a
# run is extended; that already invalidated a 1B c=8 run. c=1 is immune.
ASSIGNMENT_HORIZON = 292_968_448

DATA_TRAIN = "data/fineweb350B-bpe16384-nodedup/fineweb350b-nodedup_train_*.bin"
DATA_VAL = "data/fineweb350B-bpe16384-nodedup/fineweb350b-nodedup_val_*.bin"

# The ladder. d_head=64 and d_model/n_layer=128 throughout, so N_trunk ~ d^3
# and the rung is fully described by its width.
#   name  n_layer  n_embd   trunk N
RUNGS = [
    ("R1", 4, 512),      # 12.6M
    ("R2", 5, 640),      # 24.6M
    ("R3", 6, 768),      # 42.5M
    ("R4", 8, 1024),     # 100.7M
    ("R5", 12, 1536),    # 339.7M
    ("R6", 16, 2048),    # 805.3M
]

# Micro-batch per (rung, wide_vocab). The binding constraint at the wide vocab
# is the logit tensor, which is width-INDEPENDENT: at V=131073, T=1024 one
# sequence costs 268MB (bf16 logits) + 537MB (fp32 CE upcast) + 268MB (grad)
# ~ 1.07GB before any trunk activation. Hence the aggressive values on the
# right-hand column. Verified by scripts/smoke_ladder_v2.py before launch.
MICRO_BATCH = {
    #          narrow V=16385   wide V=131073
    "R1": (64, 16),
    "R2": (64, 16),
    "R3": (64, 16),
    "R4": (32, 8),
    "R5": (32, 8),
    "R6": (16, 8),
}

# Half-decade grid. Published anchors put the optimum near 1e-3 at d=512 and
# 3e-4 at d=1024, so both sit interior with a half-decade of margin. An argmin
# on a grid EDGE is not a result, it is an instruction to sweep again.
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
# 3e-2 was added after the wd=0 sweep. Removing the compartment embedding moved
# the R1-c8 optimum from 3e-3 to 1e-2 -- the top of the old grid -- measured at
# a matched pre-decay step (the runs died to a node-wide CUDA launch failure at
# 90%, so the post-decay eval was lost; for compemb=ON that step ranks the LRs
# identically to the post-decay eval, so it is a trustworthy proxy). An argmin
# on a grid EDGE is not a result. One extra point costs ~60 GPU-hours; a
# re-sweep forced by an edge argmin costs 125+, so the point pays for itself.
# The grid stays IDENTICAL across every rung and arm -- unequal search effort,
# not unequal LR values, is what actually breaks a tuning comparison.

# Which arms to sweep at which rung.
#
# THREE widths, not two. Two points fit a line with zero residual, so they
# cannot falsify the fitted rule -- and LR(d) is then extrapolated 2x beyond the
# swept range to reach R6. R3 is interpolation, which is exactly what validates
# the functional form: if R1/R3/R4 are collinear in log-log, extrapolating to
# R5/R6 is justified; if they are not, that is known before 356 GPU-hours are
# committed. Sweeping R6 directly would cost 581 GPU-hours -- more than the R6
# ladder rung itself -- so it is anchored against published values instead
# (Pythia-1B ~3e-4, OLMo-1B ~4e-4).
#
# c1-padded is swept at R1 ONLY, where the wide vocab is cheapest. It is NOT
# safe to assume LR*(c1-padded) == LR*(c1): while the unused INPUT embedding
# rows are dead, every one of the 131073 OUTPUT rows receives gradient through
# the softmax denominator (dL/dlogit_j = p_j - y_j pushes unused rows down), so
# its optimizer dynamics genuinely differ from c1 at the head.
LRSWEEP_ARMS = {
    "R1": ("c1", "c8", "c1-padded"),
    "R3": ("c1", "c8"),
    "R4": ("c1", "c8"),
}
LRSWEEP_RUNGS = list(LRSWEEP_ARMS)
LRSWEEP_TOKENS = 3_000_000_000
LADDER_TOKENS = 30_000_000_000
LADDER_BRANCH_AT = [27_000_000_000]   # branch point for a 30B anneal

EVAL_SEQUENCES = 2048            # held CONSTANT across arms -- see below

# eval_iters is DERIVED as EVAL_SEQUENCES // batch_size, never fixed.
#
# train.py builds val_loader with the TRAINING batch_size and reads exactly
# eval_iters batches from it, so a fixed eval_iters makes the evaluated token
# count -- and the evaluated DATA -- depend on the arm's micro-batch. At
# eval_iters=20 that meant c1 saw 1280 val sequences at R1 while c8 saw 320: not
# merely a smaller sample but a strict SUBSET, so L(c8) - L(c1) mixed a model
# difference with a sampling difference. Worse, the ratio varied by rung (4x at
# R1, 2x at R6), which would impart a spurious trend into the very scaling fit
# the ladder exists to produce.
#
# Holding sequences constant makes every arm evaluate on the identical first
# 2048 val sequences (2.1M tokens, one training batch worth). val_loader.reset()
# before each eval already guarantees determinism, so this makes the comparison
# genuinely paired rather than only appearing to be.

PLACEHOLDER_LR = 4e-4            # overwritten once the sweep lands

# ---- recipe knobs -------------------------------------------------------
#
# Defaults reproduce the ORIGINAL v2 sweep byte-for-byte (wd=0, compartment
# embeddings on), so regenerating never disturbs runs already on disk.
#
# The FINAL recipe overrides both:
#
#   --suite ladder-v2-final --weight-decay 0.1 --no-compartment-embeddings
#
# weight_decay 0.1 is the modal published value -- GPT-3, OLMo, Llama and
# SmolLM2 all use it (Pythia is the outlier at 0.01). Excluding only biases and
# norms from decay, which configure_optimizers already does via `dim() >= 2`,
# is the universal convention. Excluding EMBEDDINGS as well was considered and
# rejected: the mechanistic case for it is weak in a pre-LN transformer, where
# ln_1/ln_f normalise away the scale that decay changes, and there is no
# citation solid enough to defend the deviation with.
#
# use_compartment_embeddings false: the ladder carries no explicit source
# indicator. With vocabulary expansion the compartment is already in-band --
# token t in compartment i IS a distinct id, t + i*V -- so the additive
# compartment vector is redundant information. It is also DIFFERENTIAL across
# the arms (eight distinct vectors doing real work at c=8; one constant vector
# the model absorbs trivially at c=1), which is the one class of effect this
# design cannot absorb.
WEIGHT_DECAY = 0.0
USE_COMPARTMENT_EMBEDDINGS = True
SUITE = "ladder-v2"


# ---------------------------------------------------------------- helpers

def iters_for(tokens: int) -> int:
    """Budgets are specified in TOKENS; max_iters is DERIVED, never chosen."""
    return -(-tokens // TOKENS_PER_STEP)   # ceil


def eval_interval_for(max_iters: int, target: int) -> int:
    """Pick an eval_interval that puts an eval on the LAST training step.

    train.py evaluates when `iter_num % eval_interval == 0`, and its loop is
    `while iter_num < max_iters`, so the final body execution is at
    `max_iters - 1`. An eval therefore lands on the last step only if
    eval_interval divides `max_iters - 1`.

    This is not cosmetic. For the LR sweep the measured quantity IS the
    post-decay loss: with max_iters=1431 and a round eval_interval of 250 the
    last eval falls at 1250, which is before the decay window even opens at
    1287. The whole sweep would have compared PRE-decay losses and chosen the
    learning rate for a schedule nobody runs.

    Returns the largest divisor of (max_iters - 1) that is <= target, so the
    eval cadence stays near the requested one while guaranteeing a terminal
    point.
    """
    n = max_iters - 1
    if n <= 0:
        return target
    best = 1
    for d in range(1, int(n**0.5) + 1):
        if n % d:
            continue
        for cand in (d, n // d):
            if cand <= target and cand > best:
                best = cand
    return best


def batch_for(rung: str, wide_vocab: bool) -> tuple[int, int]:
    micro = MICRO_BATCH[rung][1 if wide_vocab else 0]
    accum = 2048 // micro
    assert micro * accum == 2048, f"{rung}: micro {micro} does not divide 2048"
    assert accum % WORLD_SIZE == 0, (
        f"{rung}: grad_accum {accum} not divisible by world size {WORLD_SIZE}; "
        "train.py asserts this then divides"
    )
    return micro, accum


def arm_vocab(arm: str) -> tuple[int, int]:
    """(model.vocab_size, experiment.n_compartments) for an arm name."""
    if arm == "c1":
        return BASE_VOCAB, 1
    if arm == "c8":
        return BASE_VOCAB, C_TREAT
    if arm == "c1-padded":
        # composite = 131072*1 + 1 = 131073, exactly matching c8
        return BASE_VOCAB * C_TREAT, 1
    raise ValueError(f"unknown arm {arm!r}")


def render(
    *,
    name: str,
    header: str,
    n_layer: int,
    n_embd: int,
    arm: str,
    lr: float,
    tokens: int,
    group: str,
    decay: bool,
    branch_at: list[int] | None = None,
) -> str:
    vocab_size, n_comp = arm_vocab(arm)
    composite = vocab_size * n_comp + 1
    wide = composite > BASE_VOCAB + 1
    micro, accum = batch_for(_rung_of(n_embd), wide)
    eval_it = EVAL_SEQUENCES // micro
    assert eval_it * micro == EVAL_SEQUENCES, (
        f'micro-batch {micro} must divide EVAL_SEQUENCES {EVAL_SEQUENCES} '
        'so every arm evaluates on exactly the same sequences')
    max_iters = iters_for(tokens)
    n_head = n_embd // 64          # d_head fixed at 64

    if decay:
        # WSD with the decay included: LR optima measured on non-annealed
        # constant-LR runs sit somewhere else, and the ladder is annealed.
        decay_start = int(max_iters * 0.9)
        decay_end = max_iters
        # The measured quantity is the POST-decay loss, so an eval must land on
        # the final step. See eval_interval_for: a round 250 here would put the
        # last eval at 1250, before the decay window opens at 1287.
        eval_iv = eval_interval_for(max_iters, 150)
        assert (max_iters - 1) % eval_iv == 0
        assert max_iters - 1 >= decay_start, "final eval must be inside the decay"
    else:
        # Pure stable trajectory. Anneals fork off it with launch_anneal.py.
        # No decay to land on, and the trajectory (not the endpoint) is what is
        # read, so a round cadence is fine.
        decay_start = 0
        decay_end = 0
        eval_iv = 250

    branch = branch_at or []
    branch_line = ", ".join(f"{b:_}" for b in branch)

    return f"""\
# {header}
#
# Generated by scripts/gen_ladder_v2_configs.py. Do not hand-edit: regenerate.
# Normative reference: docs/experimental-protocol.md
#
#   rung          {_rung_of(n_embd)}  {n_layer}x{n_embd}  (d_head 64, d/L = {n_embd // n_layer})
#   trunk N       {12 * n_layer * n_embd * n_embd / 1e6:.1f}M   <- the axis every power law is fit against
#   arm           {arm}
#   vocab rows    {composite:,}  (= vocab_size {vocab_size} x n_compartments {n_comp} + 1)
#   budget        {tokens / 1e9:.0f}B tokens -> max_iters {max_iters} DERIVED at {TOKENS_PER_STEP:,} tok/step
#   batch         {micro} x {accum} = 2048 sequences = {TOKENS_PER_STEP / 1e6:.3f}M tokens/step

config_version = 2

[data]
train_bin = "{DATA_TRAIN}"
val_bin   = "{DATA_VAL}"

[model]
n_layer = {n_layer}
n_head = {n_head}
n_embd = {n_embd}
block_size = 1024
vocab_size = {vocab_size}
weight_tying = false
use_rope = true
rope_base = 10000.0

[training]
max_iters = {max_iters}
batch_size = {micro}
gradient_accumulation_steps = {accum}
# MUST be false. Presets would retune batch/accum for VRAM and silently change
# tokens-per-optimizer-step out from under an exact token budget.
auto_batch_config = false
eval_interval = {eval_iv}
eval_iters = {eval_it}
log_interval = 10
always_save_checkpoint = true
seed = {SEED}
checkpoint_naming = "tokens"
full_state_at_tokens = [{branch_line}]

[optimizer]
learning_rate = {lr:.6g}
weight_decay = {WEIGHT_DECAY:g}

[lr]
schedule = "wsd"
warmup_iters = {WARMUP_ITERS}
decay_start_iter = {decay_start}
decay_end_iter = {decay_end}
# A tenth of peak. nanoGPT's inherited default 6e-5 can sit ABOVE our peak,
# which would turn the decay into a ramp UP; train.py rejects that at startup.
min_lr = {lr / 10:.6g}

[system]
compile = true
dtype = "bfloat16"

[logging]
wandb_log = true
wandb_project = "translation-compression"
wandb_run_name = "{name}"
wandb_group = "{group}"

[experiment]
n_compartments = {n_comp}
compartment_scaling = "equal"
translation_ratio = 0
max_compartments = 16
assignment_horizon_examples = {ASSIGNMENT_HORIZON}
use_compartment_embeddings = {"true" if USE_COMPARTMENT_EMBEDDINGS else "false"}
translation_mode = "standard"
translation_chunk_size = 4
"""


_BY_WIDTH = {d: r for r, _, d in RUNGS}


def _rung_of(n_embd: int) -> str:
    return _BY_WIDTH[n_embd]


# ---------------------------------------------------------------- main

def main() -> None:
    global SUITE, WEIGHT_DECAY, USE_COMPARTMENT_EMBEDDINGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output dir; defaults to config/<suite>")
    ap.add_argument("--suite", default=SUITE,
                    help="name prefix + config dir, e.g. ladder-v2-final")
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--no-compartment-embeddings", action="store_true",
                    help="emit use_compartment_embeddings = false")
    args = ap.parse_args()

    SUITE = args.suite
    WEIGHT_DECAY = args.weight_decay
    USE_COMPARTMENT_EMBEDDINGS = not args.no_compartment_embeddings
    out = pathlib.Path(args.out or f'config/{SUITE}')
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    by_name = {r: (L, d) for r, L, d in RUNGS}

    # ---- LR sweep: the experiment that must be read before the ladder runs
    for rung in LRSWEEP_RUNGS:
        n_layer, n_embd = by_name[rung]
        for arm in LRSWEEP_ARMS[rung]:
            for lr in LR_GRID:
                tag = f"{lr:.0e}".replace("e-0", "e-")
                name = f"{SUITE}-lrsweep-{rung.lower()}-{arm}-lr{tag}"
                text = render(
                    name=name,
                    header=(
                        f"LR sweep arm: {rung} {arm} lr={lr:g}. Sets the LR rule "
                        f"for the whole ladder."
                    ),
                    n_layer=n_layer,
                    n_embd=n_embd,
                    arm=arm,
                    lr=lr,
                    tokens=LRSWEEP_TOKENS,
                    group=f"{SUITE}-lrsweep",
                    decay=True,          # complete miniature of the real recipe
                )
                (out / f"{name}.toml").write_text(text)
                written.append(name)

    # ---- the ladder itself
    for rung, n_layer, n_embd in RUNGS:
        for arm in ("c1", "c8", "c1-padded"):
            name = f"{SUITE}-{rung.lower()}-{arm}"
            text = render(
                name=name,
                header=(
                    f"Scaling ladder {rung} ({n_layer}x{n_embd}), arm {arm}. "
                    f"learning_rate is a PLACEHOLDER until the LR sweep lands."
                ),
                n_layer=n_layer,
                n_embd=n_embd,
                arm=arm,
                lr=PLACEHOLDER_LR,
                tokens=LADDER_TOKENS,
                group=SUITE,
                decay=False,             # stable trajectory; anneals fork off it
                branch_at=LADDER_BRANCH_AT,
            )
            (out / f"{name}.toml").write_text(text)
            written.append(name)

    sweep = [w for w in written if "lrsweep" in w]
    ladder = [w for w in written if "lrsweep" not in w]
    print(f"wrote {len(written)} configs to {out}/")
    print(f"  {len(sweep):3d} LR sweep  ({LRSWEEP_TOKENS/1e9:.0f}B tokens each)")
    print(f"  {len(ladder):3d} ladder    ({LADDER_TOKENS/1e9:.0f}B tokens each, "
          f"PLACEHOLDER lr={PLACEHOLDER_LR:g})")


if __name__ == "__main__":
    main()

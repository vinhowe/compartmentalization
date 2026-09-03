from dataclasses import asdict, dataclass, field
from typing import Literal

from tyro.conf import FlagConversionOff


@dataclass(frozen=True)
class Job:
    """Top-level job options and metadata."""

    # Optional path to a TOML config file (also used by ConfigManager for preloading)
    config_file: str | None = None


@dataclass(frozen=True)
class Data:
    source: Literal["pretokenized", "uniform"] = "pretokenized"
    train_bin: str = ""
    val_bin: str | None = None
    uniform_seed: int = 0
    # Per-compartment data source glob patterns (one per compartment).
    # When set, each compartment reads from its own set of .bin shards.
    compartment_train_bins: list[str] | None = None
    compartment_val_bins: list[str] | None = None
    # Shuffle the corpus read order (shard order, and block order within a shard),
    # seeded by training.seed so every rank and every resume agree.
    #
    # DEFAULT CHANGED TO TRUE 2026-08-16. The unshuffled path is inherited from
    # llm.c, which consumes its whole corpus so read order cannot matter. Ours is
    # FineWeb sample-350BT, where files 0-29 of 510 are ALL CC-MAIN-2013-20 and a
    # 30B run reads only the first ~39 files -- so ~77% of every 30B run was one
    # 2013 crawl, a 100B run ~23%, and the training seed changed nothing about
    # the data. Standard practice (Pythia, OLMo, Megatron, HF datatrove) is a
    # global document shuffle before packing; this is the loader-side
    # approximation, which needs no corpus rebuild.
    #
    # Set False to reproduce a pre-2026-08-16 run's data order exactly.
    shuffle: FlagConversionOff[bool] = True


@dataclass(frozen=True)
class Model:
    # Match defaults from GPTConfig / train.py
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = True
    # Optional: preset size tier (e.g., "8-32", "8-64", "8-128", "8-256")
    size_tier: str | None = None
    # Optional advanced options (set programmatically in train.py)
    # When shared_token_embeddings is enabled, embedding_vocab_size should be base_vocab_size + 1.
    embedding_vocab_size: int | None = None
    shared_token_embeddings: bool = False
    use_compartment_embeddings: bool = False
    # These are provided so the model has the necessary context when advanced options are used.
    base_vocab_size: int | None = None
    max_compartments: int | None = None
    translation_token_id: int | None = None
    # If true (and not using shared_token_embeddings), clone base compartment
    # token embeddings across all compartments at initialization time.
    copy_compartment_embeddings: bool = False
    # If true, clone the lm_head rows for the base vocab across compartments.
    copy_compartment_lm_head: bool = False
    # If true and use_compartment_embeddings is enabled, initialize all comp_emb
    # vectors to be identical copies of compartment 0's vector.
    copy_compartment_id_embeddings: bool = False
    # vocab_size is derived from dataset meta by default
    vocab_size: int | None = None
    # RoPE (Rotary Position Embeddings) - when enabled, learned positional embeddings are not used
    use_rope: bool = False
    rope_base: float = 10000.0
    # ICL dual-stream architecture flags (mirrored from experiment.icl_*).
    # When icl_enabled is True, the model has an additional wte_icl embedding
    # and lm_head_icl projection sized by icl_vocab_size. Both are wired into
    # forward() and contribute a second loss term.
    icl_enabled: bool = False
    icl_vocab_size: int = 0


@dataclass(frozen=True)
class Init:
    # 'scratch' | 'resume' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'
    init_from: str = "scratch"


@dataclass(frozen=True)
class Optimizer:
    learning_rate: float = 5e-2
    weight_decay: float = 0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0


@dataclass(frozen=True)
class LRScheduler:
    warmup_iters: int = 1000
    decay_lr: bool = False
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # ---- WSD (warmup-stable-decay) ----------------------------------------
    # schedule == "wsd" keeps LR at `learning_rate` after warmup and decays
    # only over [decay_start_iter, decay_end_iter]. Cosine is deliberately not
    # the default: it is defined by its horizon, so committing to a token
    # budget precludes extending the run without a re-warm discontinuity. The
    # stable phase is extendable indefinitely and a decay can be forked from
    # any point on it.
    #
    # Both bounds are ABSOLUTE global iteration numbers, never offsets from
    # process start. A decay child resumed after preemption must land on the
    # same LR it would have had; keying off steps-since-start would silently
    # restart the decay and double-anneal.
    # "legacy" dispatches on decay_lr (constant if False, cosine-to-horizon if
    # True) and is what every existing config gets. New runs set "wsd".
    schedule: str = "legacy"            # "legacy" | "wsd"
    decay_start_iter: int = 0
    decay_end_iter: int = 0


@dataclass(frozen=True)
class Training:
    max_iters: int = 600000
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    # Permit hardware/vocabulary presets to replace batch_size and gradient
    # accumulation. Scientific configs with an exact token budget set this to
    # false so the requested tokens-per-optimizer-step cannot change silently.
    auto_batch_config: FlagConversionOff[bool] = True
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    seed: int = 1024
    always_save_checkpoint: bool = True

    # Token counts at which the checkpoint also carries optimizer + dataloader
    # state, making it resumable and forkable. Everything else is weights-only
    # (45 MB vs 265 MB at 8-256; 2.4 GB vs 14 GB at 1B), so this list is short
    # by design: name the points you might want to branch a decay from or
    # extend past. The end of the run is always included, and `_rolling` always
    # exists, so an empty list still leaves a preempted run resumable — it just
    # cannot reach back to an earlier point.
    full_state_at_tokens: tuple[int, ...] = ()

    # Checkpoints are also named by tokens; see src/checkpoints.py. Runs that
    # predate this keep their `step-*` directories and are read, not migrated.
    checkpoint_naming: str = "step"     # "step" (legacy) | "tokens"


@dataclass(frozen=True)
class Distributed:
    backend: str = "nccl"  # 'nccl', 'gloo', etc.


@dataclass(frozen=True)
class System:
    device: str = "cuda"  # 'cpu', 'cuda', 'cuda:0', ...
    # 'auto' picks bfloat16 if supported, else float16; can be 'float32'|'bfloat16'|'float16'
    dtype: str = "auto"
    compile: bool = True


@dataclass(frozen=True)
class Logging:
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"
    wandb_group: str | None = None
    wandb_notes: str | None = None
    # Folders; manager will ensure they exist
    log_folder: str = "out"
    checkpoint_folder: str = "out"
    # If true, buffer wandb log calls and only flush after a checkpoint is saved.
    # Use this on preemptible/time-limited Slurm jobs to keep wandb state in sync
    # with checkpoint state.
    wandb_buffer: bool = False


# @dataclass(frozen=True)
# class Experiment:
#     """Experiment-specific options for assignment generation."""
#     # Mapping from compartment id (e.g., "0") or translation (e.g., "0>1") to weight
#     weights: dict[str, float] = field(default_factory=dict)
#     # Shuffle seed for deterministic ordering (defaults to 0; you can override in TOML)
#     assignment_seed: int = 0
#     # Maximum number of compartments. REQUIRED: must be provided in config.
#     max_compartments: int | None = None
#     # Advanced options
#     # If true, use one shared token embedding table of size base_vocab+1 and map inputs modulo base_vocab
#     shared_token_embeddings: bool = False
#     # If true, add a learned compartment embedding (max_compartments x n_embd) to token+pos embeddings
#     use_compartment_embeddings: bool = False
#     # If true and not using shared_token_embeddings, clone base token embeddings
#     # across compartments during initialization (model-side behavior).
#     copy_compartment_embeddings: bool = False
#     copy_compartment_lm_head: bool = False
#     # If true, use per-compartment permutations of base tokens. Model/tokenizer
#     # vocab becomes base_vocab+1 (translation token only) and tokens are mapped
#     # through a seeded permutation per compartment at data loading time.
#     permute_tokens_per_compartment: bool = False


@dataclass(frozen=True)
class Experiment:
    """Experiment-specific options for assignment generation."""

    # n
    n_compartments: int = 2
    # Whether we're in experiments 1,3 or 2,4
    compartment_scaling: Literal["equal", "unequal", "single"] = "equal"
    # Scaling factor for translation tokens; 0 = no translations, 1 = as much
    # translation data as any one domain
    translation_ratio: float = 0
    # How to interpret translation_ratio:
    # - "compartment": 1 = as much translation data as any one compartment (default)
    # - "absolute": ratio of overall data that is translation; 1 = all translation data
    translation_ratio_mode: Literal["compartment", "absolute"] = "compartment"
    # Shuffle seed for deterministic ordering
    assignment_seed: int = 0

    # Number of examples the compartment-assignment array is generated for.
    #
    # 0 = derive from max_iters (legacy; every existing config reproduces
    # exactly). That default is a TRAP for any run that gets EXTENDED: the
    # assignment cache is keyed on this number and
    # _largest_remainder_allocations re-partitions the WHOLE array when the total
    # changes, so resuming with a larger max_iters silently re-randomises which
    # compartment every example belongs to. Measured on the 30B->100B c=8
    # extension: only 1/8 of assignments survived and val loss jumped +0.19 nats.
    #
    # Set it once to the LONGEST horizon a trajectory will ever reach, and every
    # budget along the way is a prefix of one array, so 30B/100B/300B agree.
    # c=1 is immune either way -- a single compartment makes the array constant.
    assignment_horizon_examples: int = 0

    # How each example is mapped to a compartment.
    #   "hash" (default) -- splitmix64(i) cut on cumulative weight. Independent
    #          across i, no period, and independent of N, so any prefix is stable
    #          and a run can be extended without re-randomising. Counts become
    #          binomial (~0.13% at 4M) instead of largest-remainder exact.
    #   "lcg"  (legacy) -- j = (a*i + b) % N, a Weyl sequence: exactly balanced
    #          but LOW-DISCREPANCY, so consecutive assignments are correlated and
    #          the sequence can have a short period (7, 19, 38 observed across
    #          seeds at one N). `a` is drawn from N, so changing the array length
    #          reshuffles everything.
    #
    # THE DEFAULT FLIPPED 2026-08-14 BECAUSE "lcg" COSTS REAL LOSS. At c=8,
    # seed 1024, horizon 292968448 the LCG stream has lag-7 autocorrelation 0.964
    # and trains 0.28 nats worse at 0.84B tokens; three sibling seeds are fine, so
    # the damage is silent and seed-dependent. A matched A/B changing only this
    # field (same seed, horizon, world_size=8, tokens/step, hardware) reproduced
    # the gap to 0.007 nats and removed it. See figures/diag_c8_assignment_order.png.
    #
    # Consequence for OLD configs: any pre-2026-08 config that does not set this
    # field explicitly now gets "hash", i.e. a DIFFERENT data ordering than when it
    # was published. The cache filename encodes the choice, so nothing is silently
    # corrupted -- but to reproduce a paper run bit-for-bit, set assignment_order
    # = "lcg" in that config. Only c>1 is affected; c=1 makes the array constant.
    assignment_order: str = "hash"
    # Per-compartment BPE variants. 0 or 1.0 = OFF (the default): compartments
    # differ only by vocabulary offset, t -> t + i*V, which is the mechanism every
    # run to date uses. Above 1.0, compartment i ADDITIONALLY tokenizes under its
    # own subset of dropped BPE merges, so the same text expands by this ratio and
    # the compartments differ in SEGMENTATION rather than only in id.
    #
    # Additive, not a replacement: the offset encoding is applied to the expanded
    # ids exactly as before, so with this off the data path is byte-identical.
    #
    # 2.0 is where cross-compartment segmentation disagreement PEAKS (0.464 vs
    # 0.391 at 1.5x). Not a monotone knob -- past 2.0 disagreement falls again,
    # because dropping every merge gives atom-level tokenization, identical for
    # every compartment. The ceiling is 4.159x.
    # Tell the model its compartment with a PREFIX TOKEN instead of an additive
    # embedding. Each compartment gets one extra vocabulary id, placed after the
    # translation token (c*V + 1 + i), and every sequence begins with it.
    #
    # No tokenizer change is involved -- these are extra ids in the composite
    # vocabulary, not new merges or a special-tokens file.
    #
    # This is the third answer to "how does the model know which compartment it
    # is in": additive embedding (use_compartment_embeddings), nothing at all, or
    # a prefix marker. Unlike the embedding, a marker costs one position of
    # context per sequence -- the last content token is dropped so the length
    # stays T -- which is 0.1% of a 1024-token window.
    # c-way input, 1-way output: read compartment i's private ids (t + i*V) but
    # predict the SHARED base id (t). Alignment stops being optional and becomes
    # required by the task.
    #
    # Purpose is a CEILING for the representational metric. We have a floor (an
    # untrained model) and a null (different text) but no upper anchor, so a
    # measured alignment of +0.23 has no denominator -- it could be most of what
    # is achievable or a tenth of it. This arm is what the metric reads when
    # compartments MUST converge.
    #
    # Loss is NOT comparable to a c-in/c-out run: the target space is V rather
    # than c*V+1. Use it for alignment, not for cost.
    shared_output_vocab: FlagConversionOff[bool] = False

    compartment_marker_token: FlagConversionOff[bool] = False

    bpe_variant_expansion: float = 0.0
    # Merge table source. Must be the tokenizer the corpus was built with, or the
    # expansion is meaningless.
    bpe_variant_tokenizer: str = "tokenizers/bpe-16384-fineweb1"
    # Maximum number of compartments. REQUIRED: must be provided in config.
    max_compartments: int | None = None
    # Advanced options
    # If true, use one shared token embedding table of size base_vocab+1 and map inputs
    # modulo base_vocab
    shared_token_embeddings: FlagConversionOff[bool] = False
    # If true, add a learned compartment embedding (max_compartments x n_embd) to
    # token+pos embeddings
    use_compartment_embeddings: FlagConversionOff[bool] = False
    # If true and not using shared_token_embeddings, clone base token embeddings
    # across compartments during initialization (model-side behavior).
    copy_compartment_embeddings: FlagConversionOff[bool] = False
    copy_compartment_lm_head: FlagConversionOff[bool] = False
    copy_compartment_id_embeddings: FlagConversionOff[bool] = False
    # If true, use per-compartment permutations of base tokens. Model/tokenizer
    # vocab becomes base_vocab+1 (translation token only) and tokens are mapped
    # through a seeded permutation per compartment at data loading time.
    # DEFAULT FLIPPED TO FALSE 2026-08-21. It was True, which meant a config that
    # omitted this field silently selected PERMUTATION -- a mechanism no c>1 run
    # has ever used. When the v2 adapter (which sets it False) was absent on ORC,
    # four 8-GPU runs trained the wrong experiment without erroring. A default
    # should degrade to the behaviour actually in use.
    permute_tokens_per_compartment: FlagConversionOff[bool] = False
    # When permuting tokens per compartment, controls whether model *inputs* are
    # also permuted. If False, inputs use the unpermuted base tokens while
    # targets remain in the permuted id space.
    permute_input_tokens_per_compartment: FlagConversionOff[bool] = True
    # Translation sequence format:
    # - "standard": [TRANS][src tokens][TRANS][dst tokens] (current behavior)
    # - "interleaved": [TRANS][src chunk][dst chunk][src chunk][dst chunk]...
    translation_mode: Literal["standard", "interleaved"] = "standard"
    # Chunk size for interleaved translation mode (n-gram size)
    translation_chunk_size: int = 4
    # DANN (Domain-Adversarial Neural Network) settings
    # Adversarial strength. 0 = disabled.
    dann_lambda: float = 0.0
    # Comma-separated layer indices for DANN, e.g. "2,4,6". Empty = disabled.
    dann_layers: str = ""
    # Discriminator hidden size. 0 = use n_embd.
    dann_disc_hidden: int = 0
    # Token tying: share a subset of tokens across compartments
    # "none" = no tying, "top_k" = tie most frequent tokens, "bottom_k" = tie least frequent
    token_tying_mode: Literal["none", "top_k", "bottom_k"] = "none"
    # Fraction of token mass that is *untied* (needs translation). 0 = all tied, 1 = none tied.
    token_tying_ratio: float = 0.0
    # Number of data shards to sample for frequency estimation
    token_tying_freq_shards: int = 1

    # Per-example token / compartment permutation for ICL pretraining.
    # - "none": disabled (default).
    # - "vocab": each example gets a fresh random permutation over [0, base_vocab).
    #   Use for c=1: forces within-example pattern matching from start.
    # - "compartment": each token position gets a random compartment assignment in
    #   [0, n_compartments). Both X composite IDs and compartment_ids C are remapped
    #   consistently. Use for c>1: destroys within-example compartment identity.
    permutation_mode: Literal["none", "vocab", "compartment"] = "none"
    # Schedule for the permutation fraction:
    # - "sharp": frac = 1.0 for iter_num < permutation_cliff_step, then 0.0.
    # - "linear": frac decays linearly from 1.0 at iter 0 to 0.0 at
    #   permutation_cliff_step (inclusive). After cliff, no permutation.
    # The "fraction" maps to mode-specific semantics:
    #   - vocab: fraction of token IDs in [0, V) that get scrambled among
    #     themselves per example. Outside that subset, identity mapping.
    #   - compartment: per-position probability of being reassigned to a
    #     uniformly-random compartment (else: keep natural compartment).
    permutation_schedule: Literal["sharp", "linear", "seed_anneal"] = "sharp"
    permutation_cliff_step: int = 0
    # When > 0, the schedule's frac never decays below this floor — even
    # after the cliff. Maintains a residual permutation rate forever so
    # the model can't fully reallocate IH-supporting weights to natural
    # priors. With cliff_step=0, this becomes "constant rate" training
    # at frac = permutation_floor (B variant).
    permutation_floor: float = 0.0
    # ---- seed_anneal schedule fields ----
    # Seed-count anneal: draw a fixed pool of n permutations at init (seed 0
    # is identity; seeds 1..n-1 are fresh random permutations under the
    # matching mode). Anneal the effective count k n->1 over the first L
    # iters of training with n phases of n examples each (n = round(√L_examples),
    # L_examples = L_iters * effective_batch_size). Every example in a batch
    # gets its own seed index, so within a single batch there are up to B
    # distinct permutations applied. After the anneal, all examples use seed 0
    # (identity => natural training). Ignores permutation_cliff_step / floor /
    # _frac semantics. Applies to permutation_mode in {"vocab", "compartment"}.
    # In compartment mode, seeds parameterize per-example *coherent* compartment
    # permutations (a permutation of [0, n_compartments) applied identically at
    # every position), NOT the per-position stochastic reassignment used by the
    # sharp/linear schedules.
    permutation_anneal_iters: int = 0
    # Cossim pair alignment loss for seed_anneal. When > 0 AND during the anneal:
    # each iter builds a doubled batch (2B) where every underlying sequence is
    # applied under two different seeds from the current phase's active-k set,
    # then jointly optimized as
    #   L = lm_loss(2B examples) + permutation_pair_lambda * (1 - mean_cos(H_a, H_b))
    # where H_a, H_b are last hidden states (post ln_f). At k=1 (last anneal
    # phase and post-anneal) no valid partner exists so the pair path is
    # skipped and training proceeds as normal single-batch LM. Implemented for
    # both vocab and compartment modes. Doubles compute during the anneal window.
    permutation_pair_lambda: float = 0.0
    # Explicit override for the seed_anneal pool size n. When > 0, uses exactly
    # this many pool rows (row 0 = identity, rows 1..n-1 fresh random perms)
    # instead of the auto value n = round(√L_examples). With a fixed small pool
    # (e.g. 2), the shrinking-k schedule collapses to k=1 after n² examples,
    # which for tiny n is ~1 iter — undesirable for ablations that want
    # consistent small-pool sampling throughout the anneal window. In that case
    # the schedule instead uniformly samples seed_idx ∈ [0, n) throughout the
    # entire anneal window (L_iters iters), then drops to seed_idx=0 after.
    # Set to 0 (default) to use the original auto-sqrt behavior.
    permutation_pool_size: int = 0
    # Delay the start of the seed_anneal window until this iter. For
    # iter_num < start_iter, seed_idx is forced to 0 (identity) and pair-cos
    # is off (equivalent to plain c=8 training). At iter_num == start_iter
    # the anneal window opens and proceeds normally for permutation_anneal_iters
    # more iterations. Used for critical-period ablations: same total training
    # and same pair-cos budget, different phase relationship.
    permutation_anneal_start_iter: int = 0

    # ICL dual-stream mode. When enabled, the model has a parallel "ICL view"
    # of each example produced by a per-example hash of canonical token IDs.
    # Both views are summed at the input embedding stage; two prediction heads
    # produce next-canonical and next-ICL logits. Total loss is
    #   lm_loss + icl_lambda * icl_loss
    # The hash is stable within an example (per-example salt) and different
    # across examples (fresh salt per example each step).
    icl_mode: Literal["none", "dual_stream"] = "none"
    # Vocab size of the ICL view. Can be smaller than the canonical vocab
    # (many-to-one hash) or equal (per-example permutation). Must be a power
    # of two for the multiplicative hash.
    icl_vocab_size: int = 16384
    icl_lambda: float = 1.0
    # When > 0, per-position Bernoulli mask: with prob `icl_mask_p` the
    # canonical embedding at that position is REPLACED by the ICL embedding
    # (instead of summed). This prevents the model from running canonical
    # and ICL pathways in parallel through disjoint residual-stream subspaces
    # — they must share circuitry to handle either input. 0.0 falls back to
    # the standard summed dual-stream behavior.
    icl_mask_p: float = 0.0

    # InfoNCE alignment intervention. Optional auxiliary contrastive loss
    # using paired sequences (e.g., parallel multilingual sentences) drawn from
    # an external bin file. See scripts/prepare_wikimatrix_qwen3.py.
    infonce_enabled: FlagConversionOff[bool] = False
    # Loss weight: total_loss = lm_loss + infonce_lambda * infonce_loss
    infonce_lambda: float = 1.0
    # Layer index (0-based) to capture hidden states from. Default mid-trunk.
    infonce_layer: int = -1
    # Number of paired sentences per InfoNCE call.
    infonce_n: int = 32
    # Softmax temperature for the contrastive loss.
    infonce_tau: float = 0.1
    # Compute InfoNCE every N microsteps (gradient-accumulation substeps).
    infonce_every: int = 1
    # Fraction of the InfoNCE pool to use as the deterministic bridge subset.
    # 1.0 = full pool; smaller = use only this fraction (sampled with replacement).
    infonce_pool_frac: float = 1.0
    # Seed for the deterministic pool subset selection.
    infonce_pool_seed: int = 0
    # Path glob for paired bin files. Expected layout:
    #   <pool_path>/wikimatrix_en_*.bin   (one shard set per side)
    #   <pool_path>/wikimatrix_zh_*.bin
    #   <pool_path>/wikimatrix_pairs.npy  (int64 [N, 4]: en_start, en_len, zh_start, zh_len)
    infonce_pool_path: str = ""
    # If non-zero, ZH input tokens fed to InfoNCE are offset by this amount
    # before forwarding through the model. Use this for compartmented runs
    # where ZH lives in [V, 2V) at LM-training time but the pool is stored
    # with raw ZH ids in [0, V).
    infonce_zh_token_offset: int = 0
    # InfoNCE pool mode:
    #   "wikimatrix" (default): paired sentences from a wikimatrix-style
    #     pool dir (en.bin / zh.bin / pairs.npy). Used for multilingual exps.
    #   "compartment": sample raw sequences from training shards and present
    #     the same sequences in two distinct compartments via vocab offset
    #     (+ optional per-compartment input permutation). Used for n-comp
    #     setups at bpe16384.
    #   "bio_decl_qa": per-person paired DECL/QA renderings of the same bio
    #     facts. Pool prebuilt by scripts/build_bio_paired_pool.py. Used to
    #     test whether alignment recovers cross-format extraction that
    #     compartmentalization breaks.
    infonce_pool_mode: str = "wikimatrix"
    # Path to the pre-tokenized DECL view file (uint32, header N then L).
    # Used when infonce_pool_mode = "bio_decl_qa".
    infonce_pool_decl_path: str = ""
    # Path to the pre-tokenized QA view file. Same format / dimensions.
    infonce_pool_qa_path: str = ""
    # Token offset added to QA-side InfoNCE samples before forwarding through
    # the model. 0 = no compartmentalization (DECL+QA share the vocab). For
    # vocab-split compartmentalization (e.g., bio-cap-split-comp where model vocab
    # = 2 * tokenizer_vocab), set this to tokenizer_vocab so QA InfoNCE tokens
    # match the QA tokens in the LM training stream.
    infonce_pool_qa_offset: int = 0


@dataclass(frozen=True)
class JobConfig:
    """Configuration container for training."""

    job: Job = field(default_factory=Job)
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    init: Init = field(default_factory=Init)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    distributed: Distributed = field(default_factory=Distributed)
    system: System = field(default_factory=System)
    logging: Logging = field(default_factory=Logging)
    experiment: Experiment = field(default_factory=Experiment)

    def to_dict(self) -> dict[str, any]:  # pyright: ignore
        return asdict(self)

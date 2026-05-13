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


@dataclass(frozen=True)
class Training:
    max_iters: int = 600000
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    seed: int = 1024
    always_save_checkpoint: bool = True


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
    compartment_scaling: Literal["equal", "unequal"] = "equal"
    # Scaling factor for translation tokens; 0 = no translations, 1 = as much
    # translation data as any one domain
    translation_ratio: float = 0
    # How to interpret translation_ratio:
    # - "compartment": 1 = as much translation data as any one compartment (default)
    # - "absolute": ratio of overall data that is translation; 1 = all translation data
    translation_ratio_mode: Literal["compartment", "absolute"] = "compartment"
    # Shuffle seed for deterministic ordering
    assignment_seed: int = 0
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
    permute_tokens_per_compartment: FlagConversionOff[bool] = True
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
    #     compartmentation breaks.
    infonce_pool_mode: str = "wikimatrix"
    # Path to the pre-tokenized DECL view file (uint32, header N then L).
    # Used when infonce_pool_mode = "bio_decl_qa".
    infonce_pool_decl_path: str = ""
    # Path to the pre-tokenized QA view file. Same format / dimensions.
    infonce_pool_qa_path: str = ""
    # Token offset added to QA-side InfoNCE samples before forwarding through
    # the model. 0 = no compartmentation (DECL+QA share the vocab). For
    # vocab-split compartmentation (e.g., bio-cap-split-comp where model vocab
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

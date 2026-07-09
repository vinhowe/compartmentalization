"""Emit 4 configs for the Phase B compartmentalization test:

  cluster-0-only:  n_comp=1, trained on cluster 0's bins (838M tokens = ~1x Chinchilla for 8-512)
  cluster-1-only:  n_comp=1, trained on cluster 1's bins (same budget)
  joint:           n_comp=1, trained on interleaved joint_mixed bins (same total budget)
  comp c=2:        n_comp=2, per-compartment bins (same total budget)

All match the paper's marquee 8-512 fineweb recipe (weight_tying=false, decay_lr=false,
lr=2e-5, weight_decay=0, RoPE, seed 64, batch=128/grad_accum=16), differing only in
n_compartments + data source.

Usage: python scripts/fineweb_cluster/gen_phase_b_configs.py --write
"""
from __future__ import annotations
import argparse
import textwrap
from pathlib import Path

MAX_ITERS = 6400  # 131,072 tok/iter × 6400 = 838M tokens ≈ 1× Chinchilla for 8-512


def preamble(name: str, description: str) -> str:
    return textwrap.dedent(f"""\
        # {description}
        # Matches paper's marquee 8-512 fineweb recipe (size_tier 8-512, RoPE, no weight decay,
        # no LR decay, lr=2e-5) with only n_compartments + data differing between arms.
        # Total budget: {MAX_ITERS:,} iters × 131,072 tok/iter = {MAX_ITERS*131072/1e9:.2f}B tokens.
        """)


def model_block() -> str:
    # NB: not using size_tier because the ORC-side presets.py doesn't have
    # an 8-512 entry. Explicit n_layer/n_head/n_embd matches the paper's
    # saved config for the marquee 8-512 fineweb runs verbatim.
    return textwrap.dedent("""\
        [model]
        n_layer = 8
        n_head = 16
        n_embd = 512
        block_size = 64
        vocab_size = 16384
        weight_tying = false
        use_rope = true
        rope_base = 10000.0
        """)


def training_block() -> str:
    return textwrap.dedent(f"""\
        [training]
        max_iters = {MAX_ITERS}
        batch_size = 128
        gradient_accumulation_steps = 16
        eval_interval = 2000
        eval_iters = 1
        log_interval = 10
        always_save_checkpoint = true
        seed = 64

        [optimizer]
        learning_rate = 2e-5
        weight_decay = 0

        [lr]
        warmup_iters = 1000
        decay_lr = false

        [system]
        compile = true
        dtype = "bfloat16"
        """)


def logging_block(name: str) -> str:
    return textwrap.dedent(f"""\
        [logging]
        wandb_log = true
        wandb_project = "translation-compression"
        wandb_run_name = "{name}"
        wandb_group = "{TAG}"
        """)


def experiment_block_c1() -> str:
    return textwrap.dedent("""\
        [experiment]
        n_compartments = 1
        compartment_scaling = "equal"
        translation_ratio = 0
        translation_ratio_mode = "absolute"
        max_compartments = 16
        use_compartment_embeddings = true
        permute_tokens_per_compartment = false
        permute_input_tokens_per_compartment = true
        translation_mode = "standard"
        translation_chunk_size = 4
        """)


def experiment_block_c2() -> str:
    return textwrap.dedent("""\
        [experiment]
        n_compartments = 2
        compartment_scaling = "equal"
        translation_ratio = 0
        translation_ratio_mode = "absolute"
        max_compartments = 16
        assignment_seed = 64
        use_compartment_embeddings = true
        permute_tokens_per_compartment = false
        permute_input_tokens_per_compartment = true
        translation_mode = "standard"
        translation_chunk_size = 4
        """)


TAG = "fineweb-phase-b"
K_GLOBAL = 2


def config_cluster_only(cluster: int) -> tuple[str, str]:
    name = f"{TAG}-cluster-{cluster}-only"
    body = "\n".join([
        preamble(name, f"Phase B: cluster-{cluster}-only ceiling on FineWeb tf-idf K={K_GLOBAL} partition."),
        f'[data]\ntrain_bin = "bins/cluster_{cluster}/train_*.bin"\nval_bin = "bins/cluster_{cluster}/val_*.bin"\n',
        model_block(),
        training_block(),
        logging_block(name),
        experiment_block_c1(),
    ])
    return name, body


def config_joint() -> tuple[str, str]:
    name = f"{TAG}-joint"
    body = "\n".join([
        preamble(name, f"Phase B: joint c=1 on interleaved FineWeb (K={K_GLOBAL} mixed)."),
        '[data]\ntrain_bin = "bins/joint_mixed/train_*.bin"\nval_bin = "bins/joint_mixed/val_cluster_0.bin"\n',
        model_block(),
        training_block(),
        logging_block(name),
        experiment_block_c1(),
    ])
    return name, body


def experiment_block_comp_ck(k: int) -> str:
    return textwrap.dedent(f"""\
        [experiment]
        n_compartments = {k}
        compartment_scaling = "equal"
        translation_ratio = 0
        translation_ratio_mode = "absolute"
        max_compartments = 16
        assignment_seed = 64
        use_compartment_embeddings = true
        permute_tokens_per_compartment = false
        permute_input_tokens_per_compartment = true
        translation_mode = "standard"
        translation_chunk_size = 4
        """)


def config_comp_k(k: int) -> tuple[str, str]:
    name = f"{TAG}-comp"
    train_list = ", ".join(f'"bins/cluster_{i}/train_*.bin"' for i in range(k))
    val_list = ", ".join(f'"bins/cluster_{i}/val_*.bin"' for i in range(k))
    body = "\n".join([
        preamble(name, f"Phase B: comp c={k} with cluster as compartment ID."),
        (
            '[data]\n'
            f'compartment_train_bins = [{train_list}]\n'
            f'compartment_val_bins   = [{val_list}]\n'
        ),
        model_block(),
        training_block(),
        logging_block(name),
        experiment_block_comp_ck(k),
    ])
    return name, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="config")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--K", type=int, default=2, help="Number of clusters (2 or 8)")
    ap.add_argument("--tag", default="fineweb-phase-b",
                    help="Config name prefix (e.g. fineweb-phase-b or fineweb-phase-b-k8)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)

    global TAG, K_GLOBAL
    TAG = args.tag
    K_GLOBAL = args.K

    configs = [config_cluster_only(i) for i in range(args.K)]
    configs.append(config_joint())
    configs.append(config_comp_k(args.K))
    for name, body in configs:
        p = out_dir / f"{name}.toml"
        if args.write:
            p.write_text(body)
            print(f"wrote {p}")
        else:
            print(f"=== {p} ===\n{body}\n")


if __name__ == "__main__":
    main()

"""Emit 7 configs for the 5-way code compartmentalization suite at a given scale:

  5x lang-only baselines (python, javascript, go, rust, c)
  1x joint (c=1, trained on joint_mixed bins)
  1x comp  (c=5, per-lang bins, one compartment per lang)

Matches the paper-consistent bs=512, batch=32, grad_accum=8 → 131k tok/iter
recipe. Total budget: 60k iters = 7.86B tokens (2x Chinchilla for 24-768 joint).
Rust lang-only ceiling training will loop ~3x through its 2.6B-token corpus
(caveat is documented; comp/joint stay in-corpus at 2x Chinchilla).

Usage:
    python scripts/code_prep/gen_5way_configs.py --scale 24-768 --write

Files land in config/code-<scale>-5way-*.toml. Pass --dry to preview.
"""
from __future__ import annotations
import argparse
import textwrap
from pathlib import Path

LANGS = ["python", "javascript", "go", "rust", "c"]

SCALE_TO_MODEL = {
    "8-512":   {"n_layer": 8,  "n_head": 8,  "n_embd": 512},
    "24-768":  {"n_layer": 24, "n_head": 12, "n_embd": 768},
    "24-1024": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
}

# Iter budgets (2x Chinchilla joint total, 131k tok/iter).
SCALE_TO_ITERS = {
    "8-512":   {"max_iters": 20000, "lr_decay": 10000},   # small, quick smoke
    "24-768":  {"max_iters": 60000, "lr_decay": 30000},   # 7.86B tokens = 2x Chinchilla
    "24-1024": {"max_iters": 51000, "lr_decay": 25500},   # ~6.7B tokens = 2x Chinchilla for 335M
}


def preamble(scale: str, name: str, description: str) -> str:
    m = SCALE_TO_MODEL[scale]
    it = SCALE_TO_ITERS[scale]
    return textwrap.dedent(f"""\
        # {description}
        # Scale: {scale} ({m['n_layer']} layers, n_head={m['n_head']}, n_embd={m['n_embd']}, block_size=512).
        # eff_batch = 32 * 8 * 512 = 131,072 tok/iter -> {it['max_iters']:,} iters = {it['max_iters']*131072/1e9:.2f}B tokens.
        """)


def model_block(scale: str) -> str:
    m = SCALE_TO_MODEL[scale]
    return textwrap.dedent(f"""\
        [model]
        n_layer = {m['n_layer']}
        n_head = {m['n_head']}
        n_embd = {m['n_embd']}
        block_size = 512
        vocab_size = 16384
        weight_tying = false
        use_rope = true
        rope_base = 10000.0
        """)


def training_block(scale: str) -> str:
    it = SCALE_TO_ITERS[scale]
    return textwrap.dedent(f"""\
        [training]
        max_iters = {it['max_iters']}
        batch_size = 32
        gradient_accumulation_steps = 8
        eval_interval = 2000
        eval_iters = 1
        log_interval = 10
        always_save_checkpoint = true
        seed = 64

        [optimizer]
        learning_rate = 3e-4
        weight_decay = 0.0

        [lr]
        decay_lr = true
        warmup_iters = 1000
        lr_decay_iters = {it['lr_decay']}
        min_lr = 3e-5

        [system]
        compile = true
        dtype = "bfloat16"
        """)


def logging_block(scale: str, name: str) -> str:
    return textwrap.dedent(f"""\
        [logging]
        wandb_log = true
        wandb_project = "translation-compression"
        wandb_run_name = "{name}"
        wandb_group = "code-5way-baselines"
        """)


def experiment_block_langonly() -> str:
    return textwrap.dedent("""\
        [experiment]
        n_compartments = 1
        compartment_scaling = "equal"
        translation_ratio = 0.0
        translation_ratio_mode = "absolute"
        max_compartments = 16
        use_compartment_embeddings = true
        permute_tokens_per_compartment = false
        permute_input_tokens_per_compartment = true
        translation_mode = "standard"
        translation_chunk_size = 4
        """)


def experiment_block_comp(n_comp: int) -> str:
    return textwrap.dedent(f"""\
        [experiment]
        n_compartments = {n_comp}
        compartment_scaling = "equal"
        translation_ratio = 0.0
        translation_ratio_mode = "absolute"
        max_compartments = 16
        assignment_seed = 64
        use_compartment_embeddings = true
        permute_tokens_per_compartment = false
        permute_input_tokens_per_compartment = true
        translation_mode = "standard"
        translation_chunk_size = 4
        """)


def config_langonly(scale: str, lang: str) -> tuple[str, str]:
    name = f"code-{scale}-5way-{lang}-only"
    body = "\n".join([
        preamble(scale, name, f"5-way suite: {lang}-only ceiling."),
        f'[data]\ntrain_bin = "bins/{lang}/train_*.bin"\nval_bin = "bins/{lang}/val_*.bin"\n',
        model_block(scale),
        training_block(scale),
        logging_block(scale, name),
        experiment_block_langonly(),
    ])
    return name, body


def config_joint(scale: str) -> tuple[str, str]:
    name = f"code-{scale}-5way-joint"
    body = "\n".join([
        preamble(scale, name, "5-way suite: joint (c=1) on document-interleaved bins."),
        '[data]\ntrain_bin = "bins/joint_mixed/train_*.bin"\nval_bin = "bins/joint_mixed/val_python.bin"\n',
        model_block(scale),
        training_block(scale),
        logging_block(scale, name),
        experiment_block_langonly(),
    ])
    return name, body


def config_comp(scale: str) -> tuple[str, str]:
    name = f"code-{scale}-5way-comp"
    train_bins = ", ".join(f'"bins/{l}/train_*.bin"' for l in LANGS)
    val_bins = ", ".join(f'"bins/{l}/val_*.bin"' for l in LANGS)
    body = "\n".join([
        preamble(scale, name, "5-way suite: compartmented (c=5), one compartment per lang."),
        f"[data]\ncompartment_train_bins = [{train_bins}]\ncompartment_val_bins = [{val_bins}]\n",
        model_block(scale),
        training_block(scale),
        logging_block(scale, name),
        experiment_block_comp(len(LANGS)),
    ])
    return name, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="24-768", choices=list(SCALE_TO_MODEL))
    ap.add_argument("--out-dir", default="config")
    ap.add_argument("--write", action="store_true", help="Write files (default: dry-run)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)

    configs = [config_langonly(args.scale, l) for l in LANGS]
    configs.append(config_joint(args.scale))
    configs.append(config_comp(args.scale))

    for name, body in configs:
        path = out_dir / f"{name}.toml"
        if args.write:
            path.write_text(body)
            print(f"wrote {path}")
        else:
            print(f"=== {path} ===")
            print(body)
            print()


if __name__ == "__main__":
    main()

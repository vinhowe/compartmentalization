# Language models struggle with compartmentalization

Companion code for the paper of the same name (preprint). Reproduces
every figure from the precomputed eval JSONs we ship via Zenodo, and
lets you retrain any individual run via `train.py` / `sweep_runner.py`.

## What's in here

```
config/              TOML configs for every run that appears in the paper
src/                 Model + training code (nanoGPT-flavored)
scripts/             Tokenizer training, dataset prep, bio task generation
experiment/          Eval + plot scripts. Each plot_*.py produces one or
                     more figures in figures/. compute_*.py recomputes
                     the JSONs from saved checkpoints.
experiment/_run_paths.py
                     Single source of truth for which run directory maps
                     to which figure. Edit here to point a plot at a new
                     run after retraining.
figures/             Paper PDFs land here.
sweeps/              wandb-style YAML sweeps (legacy + active).
sweep_runner.py      File-lock + wandb-aware grid runner. Replaces
                     `wandb agent` for multi-cluster coordination.
train.py             Single-run training entry point.
```

## T1 — Regenerate figures from precomputed JSONs (no GPU needed)

This is the fast path. Download the eval JSONs from Zenodo, drop them in
`experiment/`, and the plot scripts run in seconds on CPU.

1. **Install.** We use [uv](https://github.com/astral-sh/uv):
   ```
   uv sync
   ```
2. **Get the JSONs.** Zenodo DOI `<TODO>`. Unpack into `experiment/`:
   - `val_metrics.json` — per-checkpoint val loss for every paper run
   - `finetune_val_metrics.json` — finetune-from-c=1 trajectory
   - `cossim_sweep.json`, `cossim_across_training.json` — layer-4 cross-compartment cosine similarity
   - `multilingual_24_{512,768,1024}_per_lang.json` — multilingual case study
   - `.multirun/*.log` — training-time val_loss logs for the InfoNCE figures
3. **Generate figures.** From `experiment/`:
   ```
   python plot_baseline_val_curves.py
   python plot_loss_plateau.py
   python plot_translation_phase_transition.py
   python plot_wd_tr075_slice.py
   python plot_compartmented_slowdown.py
   python plot_copyemb.py
   python plot_finetune_offset_trajectory.py
   python plot_1b_section.py
   python plot_translation_target_trajectory.py
   python plot_n2_compartment_diversity.py
   python plot_slowdown_explainer.py
   python plot_infonce_n2_batch_compare.py
   python plot_infonce_8_256_n2_lambda_sweep_c1_final_gap.py
   python plot_infonce_8_256_c1_final_gap.py
   python plot_multilingual_scaling.py
   ```
   Each writes its PDFs to `../figures/`.

## T2 — Retrain a single run

Single-cell configs (1B-scale runs, InfoNCE cells, etc.) are launched
directly through `train.py` via `--job.config-file`:

```
torchrun --standalone --nproc_per_node=8 train.py \
    --job.config-file config/1b-1comp-baseline-bpe16384.toml
```

The run dir is named deterministically as
`out/translation-compression/<group>/<timestamp>__<slug>__<cfg_hash>_s<seed>__<git>__<env>`.
Timestamps are unique to your retraining session, so to wire the new run
into a plot script, either edit the corresponding entry in
`experiment/_run_paths.py` or rely on
`find_latest_run(group, substr)` to pick up the latest match.

After training finishes, the eval pipeline lives in `experiment/`:

```
cd experiment
python evaluate_checkpoints_fineweb_dedup.py  # per-rank shards
python merge_eval_results.py                  # merges → val_metrics.json
```

## T3 — Retrain a grid (sweep)

The phase-transition, WD, and small-scale c-sweeps are grids over
`(c, tr, wd)`. They run via `sweep_runner.py`, which replaces
`wandb agent` for multi-cluster coordination — file locks for
intra-cluster, wandb registry reads for cross-cluster. Each cell's
config is the sweep YAML's base TOML plus per-cell overrides, hashed to
a deterministic `cfg_hash` so the same grid always lands at the same
`out/<group>/<cfg_hash>_s<seed>/` paths used by `_run_paths.py`.

```
python sweep_runner.py sweeps/bpe16384-rope-8-256.yaml
```

See `sweeps/README.md` for the active grids.

## Data prep

Tokenizer + datasets:

- BPE-16384 tokenizer: `scripts/train_bpe16384.sh`
- Fineweb shards: see `scripts/` (we use a 350B-dedup subset)
- Wiki en/zh for the multilingual case study: `scripts/prepare_wiki_qwen3.py`
- Bio capacity task: `scripts/generate_bio_dataset.py`

## Translation-ratio convention

Two modes appear in the configs:

- **`absolute`** (preferred for new runs): `translation_ratio` is the
  effective fraction of training tokens that are translation tokens.
- **`compartment`** (legacy): `translation_ratio` is a raw value; the
  effective ratio is `raw / (n_compartments + 1)`.

All plots and analyses report the **effective** ratio. The conversion
happens at JSON-load time in the plot scripts.

## Hardware notes

Runs in the paper were trained on 8× A100-80GB. The 1B-scale runs are
multi-node via the included `pccl/` distributed library. Smaller-scale
runs (8-{32,64,128,256,512}) fit single-node.

## Citation

```
<TODO: bibtex once preprint is up>
```

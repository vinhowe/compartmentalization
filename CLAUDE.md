# translation-compression

## Experimental protocol

Full protocol: [docs/experimental-protocol.md](docs/experimental-protocol.md).
Read it before creating a config, launching a run, or fitting a scaling law.

The rules below are the ones that silently corrupt an experiment if broken, so
they are repeated here rather than left to the linked document.

- **The ladder is six rungs**, `d_head=64`, aspect ratio `d_model/n_layer=128`
  held constant: `4x512, 5x640, 6x768, 8x1024, 12x1536, 16x2048`
  (trunk N: 12.6M / 24.6M / 42.5M / 100.7M / 339.7M / 805.3M).
- **Fit power laws against trunk (non-embedding) params, never total params.**
  At c=8 the model is 40-91% embedding table and the fraction moves across the
  ladder, so a fit on total N mixes compartmentalization with shrinking
  embedding overhead. Trunk N is identical for c=1 and c=8 at a rung.
- **Every rung runs three arms**: `c1`, `c8`, and `c1-padded` (parameter-matched
  control). Only `L(c8) - L(c1-padded)` is fit.
- **Budgets are specified in TOKENS; `max_iters` is derived.** 2,097,152
  tokens/step, so 30B -> 14,305 iters. Any inherited `max_iters` from the T=64
  era is wrong by 16x.
- **LR is tuned per condition, over a grid fixed in advance.** Holding LR
  constant across c or across width is a biased comparison, not a controlled
  one. Report the gap at matched LR *and* at per-condition optimum.
- **`auto_batch_config = false`** on anything with an exact token budget, or the
  presets retune batch/accum for VRAM and change tokens-per-step underneath it.
- **`config_version = 2`** for all new configs.
- **`tr` always means the effective (absolute) translation ratio.** Legacy
  compartment-mode converts as `raw / (n + raw)`, NOT `raw / (n + 1)`.
- **For c>1, pin `assignment_horizon_examples`** to the longest budget the trunk
  will ever reach. Deriving it from `max_iters` re-randomises compartment
  assignments when a run is extended; this has already invalidated a 1B run.
- **ORC's filesystem is not shared with pccfs2.** Sync configs and the code they
  depend on as a unit, then assert a representative config resolves identically
  on both. Checking that it parses does not catch a field falling back to a
  dataclass default.

## Tools

- Use `uv` for Python package management and virtual environments.
# Project Reference

## Slurm Submission Commands

Submit a sweep runner job (1 node = 8 GPUs, each running sweep_runner.py in a loop):

```bash
PROXYKIT_DIR="$PWD/slurm" sbatch --qos=dw87long --mem=0 ./slurm/run_sweep_runner.sbatch sweeps/<SWEEP_YAML>
```

Example for the n_compartments 3 & 5 sweep:

```bash
PROXYKIT_DIR="$PWD/slurm" sbatch --qos=dw87long --mem=0 ./slurm/run_sweep_runner.sbatch sweeps/bpe16384-n3-n5.yaml
```

Key flags:
- `--mem=0` — request all memory on the node
- `--qos=dw87long` — long QOS for dw partition (6-day time limit)
- The sbatch script handles proxy sidecar, srun with 8 tasks, GPU binding, etc.

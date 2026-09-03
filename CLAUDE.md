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

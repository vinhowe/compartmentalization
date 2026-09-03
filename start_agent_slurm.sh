#!/bin/bash
PROXYKIT_DIR="$PWD/slurm" sbatch --array=1-20 --qos=dw87long --mem=2TB ./slurm/run_wandb_agents_proxied.sbatch pccl/translation-compression/2v2ci97x

# On standby
# PROXYKIT_DIR="$PWD/slurm" sbatch --array=1-999 --partition=dw --qos=standby --mem=1TB ./slurm/run_wandb_agents_proxied.sbatch pccl/translation-compression/cn8yurxt

# Pinning max 3 nodes (3 * 8 GPUs = 24 GPUs) on dw87long
# PROXYKIT_DIR="$PWD/slurm" sbatch --array=1-999%3 --qos=dw87long --mem=1TB ./slurm/run_wandb_agents_proxied.sbatch pccl/translation-compression/cn8yurxt
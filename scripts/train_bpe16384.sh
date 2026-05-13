#!/bin/bash
set -e

python scripts/train_tokenizer.py \
    --vocab-size 16384 \
    --output-dir tokenizers/bpe-16384 \
    --max-tokens 1000000000

python scripts/retokenize_dataset.py \
    --input-pattern "data/fineweb350B-dedup-suffix-31/fineweb350b-dedup_*.bin" \
    --target-tokenizer tokenizers/bpe-16384 \
    --output-dir data/fineweb350B-dedup-bpe16384 \
    --skip-existing

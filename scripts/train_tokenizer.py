#!/usr/bin/env python3
"""
Train a small BPE tokenizer (vocab_size=4096) on decoded FineWeb data.

The existing data is tokenized with Qwen2 (vocab_size=151936).
We decode it back to text, then train a new tokenizer.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import AutoTokenizer
from tqdm import tqdm

def iter_tokens_from_bin(filename: str) -> np.ndarray:
    """Read all tokens from a bin file (format: magic=20251013, uint32 tokens)."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20251013, f"magic number mismatch: {header[0]}"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    assert len(tokens) == ntok, "token count mismatch"
    return tokens


def decode_tokens_batch(tokenizer, token_ids: list[int], batch_size: int = 10000) -> str:
    """Decode tokens in batches to avoid memory issues."""
    texts = []
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i:i + batch_size]
        texts.append(tokenizer.decode(batch, skip_special_tokens=True))
    return "".join(texts)


def text_iterator(bin_files: list[str], source_tokenizer, max_tokens: int):
    """
    Iterate over decoded text from bin files.
    Yields text in chunks suitable for tokenizer training.
    """
    total_tokens = 0

    for bin_file in tqdm(bin_files, desc="Processing shards"):
        if total_tokens >= max_tokens:
            break

        tokens = iter_tokens_from_bin(bin_file)
        remaining = max_tokens - total_tokens
        if len(tokens) > remaining:
            tokens = tokens[:remaining]

        # Decode tokens to text
        # Process in chunks to avoid memory issues
        chunk_size = 100_000
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size].tolist()
            text = source_tokenizer.decode(chunk, skip_special_tokens=True)
            if text.strip():
                yield text

        total_tokens += len(tokens)

    print(f"Processed {total_tokens:,} tokens total")


def main():
    parser = argparse.ArgumentParser(description="Train a small BPE tokenizer")
    parser.add_argument(
        "--input-pattern",
        default="data/fineweb350B-dedup-suffix-31/fineweb350b-dedup_train_*.bin",
        help="Glob pattern for input bin files",
    )
    parser.add_argument(
        "--source-tokenizer",
        default="Qwen/Qwen2.5-72B",
        help="HuggingFace tokenizer used to encode the original data",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Vocabulary size for new tokenizer",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1_000_000_000,  # 1B tokens
        help="Maximum number of tokens to use for training",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizers/bpe-4096",
        help="Output directory for trained tokenizer",
    )
    args = parser.parse_args()

    # Find input files
    bin_files = sorted(glob.glob(args.input_pattern))
    if not bin_files:
        print(f"No files found matching: {args.input_pattern}")
        return
    print(f"Found {len(bin_files)} input files")

    # Load source tokenizer for decoding
    print(f"Loading source tokenizer: {args.source_tokenizer}")
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer, trust_remote_code=True)
    print(f"Source vocab size: {source_tokenizer.vocab_size}")

    # Create new BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Use byte-level pre-tokenizer (like GPT-2/Qwen)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor for adding special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Configure trainer
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    # Train tokenizer
    print(f"Training tokenizer with vocab_size={args.vocab_size} on up to {args.max_tokens:,} tokens...")
    tokenizer.train_from_iterator(
        text_iterator(bin_files, source_tokenizer, args.max_tokens),
        trainer=trainer,
    )

    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Also save in HuggingFace format for compatibility
    # Wrap with PreTrainedTokenizerFast
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    hf_tokenizer.save_pretrained(args.output_dir)
    print(f"Saved HuggingFace tokenizer to {args.output_dir}")

    # Test the tokenizer
    print("\n--- Testing tokenizer ---")
    test_text = "Hello, world! This is a test of the new tokenizer."
    encoded = hf_tokenizer.encode(test_text)
    decoded = hf_tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Vocab size: {hf_tokenizer.vocab_size}")


if __name__ == "__main__":
    main()

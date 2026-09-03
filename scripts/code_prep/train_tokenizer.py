"""Train a joint BPE tokenizer on equal-weight samples from N languages.

Samples a subset of the filtered corpus (default: 500 MB from each language)
and trains a byte-level BPE with GPT-2-style pretokenization. Saves to
<OUT_ROOT>/tokenizer/<name>.json.

Default langs: python, javascript (matches original 2-way suite).
Pass --langs python,javascript,go,rust,c for 5-way.
"""
from __future__ import annotations
import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm


def sample_texts_from_lang(lang_dir: Path, target_bytes: int, seed: int) -> Iterator[str]:
    """Yield text strings from shuffled shards until target_bytes served."""
    rng = random.Random(seed)
    shards = sorted(lang_dir.glob("shard-*.jsonl.gz"))
    if not shards:
        raise RuntimeError(f"no shards in {lang_dir}")
    rng.shuffle(shards)
    served = 0
    for shard in shards:
        with gzip.open(shard, "rt") as fp:
            for line in fp:
                r = json.loads(line)
                text = r["text"]
                yield text
                served += len(text.encode("utf-8"))
                if served >= target_bytes:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-mb-per-lang", type=int, default=500)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--langs", default="python,javascript",
                    help="Comma-separated list of langs to train on")
    ap.add_argument("--out-name", default=None,
                    help="Output tokenizer basename (default: joint_bpe<vocab_size>)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    tokenizer_dir = out_root / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    target_bytes = args.sample_mb_per_lang * 1024 * 1024

    def gen():
        """Round-robin between the two languages so training sample is balanced."""
        iters = {lang: iter(sample_texts_from_lang(out_root / "filtered" / lang, target_bytes, args.seed + i))
                 for i, lang in enumerate(langs)}
        alive = set(langs)
        pbar = tqdm(total=len(langs)*target_bytes, unit="B", unit_scale=True, desc="sample")
        while alive:
            for lang in list(alive):
                try:
                    t = next(iters[lang])
                    pbar.update(len(t.encode("utf-8")))
                    yield t
                except StopIteration:
                    alive.discard(lang)
        pbar.close()

    # GPT-2-style byte-level BPE: preserves whitespace-as-token structure.
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"],
    )

    print(f"Training BPE {args.vocab_size} on ~{args.sample_mb_per_lang} MB each of {langs}...")
    tok.train_from_iterator(gen(), trainer=trainer)

    out_name = args.out_name or f"joint_bpe{args.vocab_size}"
    out_path = tokenizer_dir / f"{out_name}.json"
    tok.save(str(out_path))
    print(f"wrote {out_path}")

    # Quick sanity: encode a small sample and print stats
    sample = "def foo(x):\n    return x * 2\n"
    ids = tok.encode(sample).ids
    print(f"sanity encode of {len(sample)}-byte python snippet -> {len(ids)} tokens")
    print(f"  vocab_size: {tok.get_vocab_size()}")


if __name__ == "__main__":
    main()

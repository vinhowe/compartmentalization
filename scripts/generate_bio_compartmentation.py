#!/usr/bin/env python3
"""Generate bio-compartmentation dataset: person-level partition into BRIDGE /
DECL-only / QA-only / NEVER-SEEN populations, tokenized with GPT-2 BPE.

The model trains on a SINGLE shuffled stream containing:
  - DECL bios for every BRIDGE and DECL-only person
  - QA pairs for every BRIDGE and QA-only person
  (NEVER-SEEN people contribute nothing to training.)

Outputs:
  data/bio-comp-N{N}-bridge{B}-seed{S}/
    bio_train_NNNNNN.bin             — packed shuffled training stream
    people.json                      — list of all people (with population tag)
    meta.json                        — config record

Each person has an "id" and a "population" field in {"bridge","decl_only","qa_only","never_seen"}.

The eval pipeline (scripts/eval_bio_compartmentation.py) consumes people.json and
probes each population with each format.

Token format: uint32, 1024-byte header (magic 20251013), EOS=50256.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the existing field pools and template logic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_bio_dataset import (
    generate_people, render_bio, render_qa,
    FIRST_NAMES, MIDDLE_NAMES, LAST_NAMES, CITIES, UNIVERSITIES, FIELDS,
    COMPANIES, MONTHS,
    BIRTH_DATE_TEMPLATES, BIRTH_CITY_TEMPLATES,
    UNIVERSITY_TEMPLATES, MAJOR_TEMPLATES,
    WORK_CITY_TEMPLATES, EMPLOYER_TEMPLATES,
    QA_TEMPLATES, build_qa_templates,
)

import tiktoken

MAGIC = 20251013
SHARD_TOKENS = 50_000_000


def write_shard(path: Path, tokens: list[int], eos_id: int):
    arr = np.array(tokens, dtype=np.uint32)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(arr)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())


def partition_people(
    people: list[dict],
    n_bridge: int,
    n_decl_only: int,
    n_qa_only: int,
    n_never_seen: int,
    seed: int,
) -> dict:
    """Deterministic shuffled partition into 4 populations.

    Returns dict mapping population name -> list of person dicts (with 'population' tag set).
    """
    assert n_bridge + n_decl_only + n_qa_only + n_never_seen <= len(people)
    rng = random.Random(seed + 12345)
    perm = list(range(len(people)))
    rng.shuffle(perm)
    pops = {}
    cur = 0
    for name, count in [
        ("bridge", n_bridge),
        ("decl_only", n_decl_only),
        ("qa_only", n_qa_only),
        ("never_seen", n_never_seen),
    ]:
        sel = [people[perm[i]] for i in range(cur, cur + count)]
        for p in sel:
            p["population"] = name
        pops[name] = sel
        cur += count
    return pops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_people", type=int, default=50_000, help="total people generated")
    ap.add_argument("--bridge_frac", type=float, default=0.05, help="fraction of training people in BRIDGE")
    ap.add_argument("--never_seen_n", type=int, default=500, help="reserve as NEVER-SEEN")
    ap.add_argument("--n_phrasings", type=int, default=5, help="bio phrasings per DECL person")
    ap.add_argument("--permute", action="store_true", default=True)
    ap.add_argument("--fullname", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    # Optional explicit overrides for population sizes.
    # If set, these take precedence over `bridge_frac` and the default 50/50
    # decl/qa split — useful for sweeps that fix one population while varying
    # another (e.g., schema-dose-response: hold DECL-only and vary QA-only).
    # Total people generated will be n_decl_only + n_qa_only + n_bridge_explicit + never_seen_n
    # if any override is set; otherwise n_people is used.
    ap.add_argument("--n_decl_only", type=int, default=None,
                    help="explicit DECL-only count (overrides bridge_frac split)")
    ap.add_argument("--n_qa_only", type=int, default=None,
                    help="explicit QA-only count (overrides bridge_frac split)")
    ap.add_argument("--n_bridge", type=int, default=None,
                    help="explicit BRIDGE count (overrides bridge_frac)")
    ap.add_argument("--qa_template_count", type=int, default=None,
                    help="number of QA templates per attribute. None = use existing pool (8/attr).")
    ap.add_argument("--qa_n_phrasings", type=int, default=1,
                    help="QA renderings per person (each generates 6 QA pairs, one per attribute). "
                         "Default=1 matches existing setup; set to match --n_phrasings to equalize "
                         "per-fact exposure between DECL and QA.")
    ap.add_argument("--decl_template_count", type=int, default=None,
                    help="if set, truncate DECL template lists to first K per attribute. "
                         "None = use full pools (~50/attr).")
    ap.add_argument("--qa_token_offset", type=int, default=0,
                    help="Offset to add to QA-pair content tokens (creates a vocab-disjoint "
                         "compartment for QA). EOS stays at canonical id. Use vocab_size = "
                         "tokenizer.n_vocab + qa_token_offset at training time.")
    ap.add_argument("--qa_block_mode", action="store_true",
                    help="Emit each person's full 6-attribute QA as ONE training sequence "
                         "(structurally parallel to DECL bio paragraphs) instead of 6 "
                         "separate per-attribute sequences.")
    ap.add_argument("--bare_answers", action="store_true",
                    help="Use bare-answer-only QA forms (drops the forms whose answer half "
                         "starts with '{name}' and restates the fact declaratively — those "
                         "forms exactly match the decl-probe template and let QA-only-trained "
                         "models complete decl probes by surface-form memorization). Switches "
                         "build_qa_templates to draw from QA_ANSWERS_BARE (3 non-leaky forms "
                         "per attribute) instead of QA_ANSWERS (5 forms, 2 leaky).")
    args = ap.parse_args()

    # If --decl_template_count is set, truncate the DECL template lists
    # in the imported module so render_bio sees the limited pool. This
    # must happen before any render_bio call.
    if args.decl_template_count is not None:
        import generate_bio_dataset as _gbd
        K = args.decl_template_count
        _gbd.BIRTH_DATE_TEMPLATES = _gbd.BIRTH_DATE_TEMPLATES[:K]
        _gbd.BIRTH_CITY_TEMPLATES = _gbd.BIRTH_CITY_TEMPLATES[:K]
        _gbd.UNIVERSITY_TEMPLATES = _gbd.UNIVERSITY_TEMPLATES[:K]
        _gbd.MAJOR_TEMPLATES = _gbd.MAJOR_TEMPLATES[:K]
        _gbd.WORK_CITY_TEMPLATES = _gbd.WORK_CITY_TEMPLATES[:K]
        _gbd.EMPLOYER_TEMPLATES = _gbd.EMPLOYER_TEMPLATES[:K]

    explicit_pops = (args.n_decl_only is not None
                     or args.n_qa_only is not None
                     or args.n_bridge is not None)
    if explicit_pops:
        n_bridge = args.n_bridge if args.n_bridge is not None else 0
        n_decl_only = args.n_decl_only if args.n_decl_only is not None else 0
        n_qa_only = args.n_qa_only if args.n_qa_only is not None else 0
        # Override n_people to match the explicit specification
        args.n_people = n_bridge + n_decl_only + n_qa_only + args.never_seen_n
    else:
        n_train_people = args.n_people - args.never_seen_n
        n_bridge = int(round(n_train_people * args.bridge_frac))
        n_unbridged = n_train_people - n_bridge
        n_decl_only = n_unbridged // 2
        n_qa_only = n_unbridged - n_decl_only

    if args.output_dir is None:
        bridge_pct = int(round(args.bridge_frac * 100))
        args.output_dir = f"data/bio-comp-N{args.n_people}-bridge{bridge_pct}-seed{args.seed}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== bio-compartmentation dataset ===")
    print(f"  Total people:    {args.n_people:,}")
    print(f"  BRIDGE:          {n_bridge:,} ({args.bridge_frac:.1%} of training)")
    print(f"  DECL-only:       {n_decl_only:,}")
    print(f"  QA-only:         {n_qa_only:,}")
    print(f"  NEVER-SEEN:      {args.never_seen_n:,}")
    print(f"  Augmentation:    permute={args.permute}, fullname={args.fullname}, n_phrasings={args.n_phrasings}")
    print(f"  Bare answers:    {args.bare_answers}")

    # 1. Generate all people
    print(f"\nGenerating {args.n_people} people (seed={args.seed})...")
    people = generate_people(args.n_people, seed=args.seed)

    # 2. Partition
    pops = partition_people(
        people, n_bridge, n_decl_only, n_qa_only, args.never_seen_n, args.seed
    )

    # 3. Tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    enc = tiktoken.get_encoding(args.tokenizer)
    eos_id = enc.eot_token  # 50256 for gpt2
    print(f"  vocab={enc.n_vocab}, eos={eos_id}")

    def tok(s: str) -> list[int]:
        return enc.encode(s, disallowed_special=())

    # 4. Build training sequences
    # Each entry is a list of token ids (bio or QA pair), EOS appended.
    print(f"\nRendering training sequences...")
    t0 = time.monotonic()
    sequences: list[list[int]] = []

    decl_rng = random.Random(args.seed + 100)
    # DECL renderings for BRIDGE and DECL-only people: n_phrasings each
    decl_people = pops["bridge"] + pops["decl_only"]
    for person in decl_people:
        for _ in range(args.n_phrasings):
            text = render_bio(person, decl_rng, permute=args.permute, fullname=args.fullname)
            tokens = tok(text)
            tokens.append(eos_id)
            sequences.append(tokens)
    n_decl_seqs = len(sequences)
    print(f"  DECL bios: {n_decl_seqs:,} sequences from {len(decl_people):,} people")

    # QA pairs for BRIDGE and QA-only people: 6 (one per attribute)
    qa_rng = random.Random(args.seed + 200)
    qa_people = pops["bridge"] + pops["qa_only"]
    qa_templates_dict = (
        build_qa_templates(args.qa_template_count, bare=args.bare_answers)
        if args.qa_template_count is not None
        else None
    )
    if args.bare_answers and args.qa_template_count is None:
        # If the caller asked for bare answers but didn't set qa_template_count,
        # they almost certainly intended for bare to take effect. Make this an
        # error rather than silently using the leaky QA_TEMPLATES legacy pool.
        raise ValueError(
            "--bare_answers requires --qa_template_count to be set (the bare "
            "filter only applies through build_qa_templates; the legacy "
            "QA_TEMPLATES pool used when qa_template_count is None still "
            "contains leaky 'X was born on Y' forms)."
        )
    for person in qa_people:
        for _ in range(args.qa_n_phrasings):
            qa_strs = render_qa(person, qa_rng, templates_dict=qa_templates_dict)
            if args.qa_block_mode:
                # Concatenate all 6 QA pairs into ONE sequence per person (mirrors
                # how DECL puts all 6 attributes in one paragraph). Single trailing EOS.
                tokens = tok("".join(qa_strs))
                if args.qa_token_offset:
                    tokens = [t + args.qa_token_offset for t in tokens]
                tokens.append(eos_id)
                sequences.append(tokens)
            else:
                # Default: each QA pair is its own training sequence (legacy behaviour).
                for qa_text in qa_strs:
                    tokens = tok(qa_text)
                    if args.qa_token_offset:
                        tokens = [t + args.qa_token_offset for t in tokens]
                    tokens.append(eos_id)
                    sequences.append(tokens)
    n_qa_seqs = len(sequences) - n_decl_seqs
    print(f"  QA pairs:  {n_qa_seqs:,} sequences from {len(qa_people):,} people "
          f"(qa_n_phrasings={args.qa_n_phrasings}, qa_token_offset={args.qa_token_offset})")

    # 5. Shuffle the COMBINED stream so DECL and QA are interleaved
    shuffle_rng = random.Random(args.seed + 300)
    shuffle_rng.shuffle(sequences)

    # 6. Pack and write shards
    print(f"\nWriting shards to {out_dir}/")
    shard_idx = 0
    buf: list[int] = []
    total_tokens = 0
    for seq in sequences:
        buf.extend(seq)
        if len(buf) >= SHARD_TOKENS:
            fname = out_dir / f"bio_train_{shard_idx:06d}.bin"
            write_shard(fname, buf[:SHARD_TOKENS], eos_id)
            total_tokens += SHARD_TOKENS
            buf = buf[SHARD_TOKENS:]
            shard_idx += 1
    if buf:
        fname = out_dir / f"bio_train_{shard_idx:06d}.bin"
        write_shard(fname, buf, eos_id)
        total_tokens += len(buf)
        shard_idx += 1

    # 7. Save people list (with population tags) and meta
    people_path = out_dir / "people.json"
    with open(people_path, "w") as f:
        json.dump(people, f)
    meta_path = out_dir / "meta.json"
    meta = {
        "n_people": args.n_people,
        "n_bridge": n_bridge,
        "n_decl_only": n_decl_only,
        "n_qa_only": n_qa_only,
        "n_never_seen": args.never_seen_n,
        "bridge_frac": args.bridge_frac,
        "n_phrasings": args.n_phrasings,
        "permute": args.permute,
        "fullname": args.fullname,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "eos_id": eos_id,
        "n_decl_seqs": n_decl_seqs,
        "n_qa_seqs": n_qa_seqs,
        "total_train_tokens": total_tokens,
        "n_shards": shard_idx,
        "qa_token_offset": args.qa_token_offset,
        "qa_n_phrasings": args.qa_n_phrasings,
        "qa_template_count": args.qa_template_count,
        "decl_template_count": args.decl_template_count,
        "bare_answers": args.bare_answers,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.monotonic() - t0
    print(f"\n=== SUMMARY ===")
    print(f"  Total training tokens: {total_tokens:,}")
    print(f"  Shards: {shard_idx}")
    print(f"  Wall: {elapsed:.0f}s")
    print(f"  Output dir: {out_dir}")
    print(f"  At batch 32 × ga 8 × seq 512 = 131,072 tokens/step:")
    print(f"    epochs in 15k steps: {15_000 * 131_072 / total_tokens:.1f}")


if __name__ == "__main__":
    main()

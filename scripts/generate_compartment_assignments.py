#!/usr/bin/env python3
"""
Generate a deterministic, shuffled assignment file of compartment and translation examples.

Each example is either:
- a compartment-only example (kind=0) for a single compartment id, or
- a translation example (kind=1) from a source compartment id to a target compartment id.

You provide weights for compartments and translations. The script deterministically converts
weights to exact integer counts using the Largest Remainder method, concatenates the examples,
shuffles them with a deterministic seed, and writes a compact binary file.

Binary format (little-endian):
- Header (32 bytes):
    - magic: 8 bytes = b"TCASSIGN"
    - version: u8 = 1
    - record_size: u8 = 8
    - flags: u16 (bit 0 set = little-endian)
    - num_compartments: u32
    - num_records: u64
    - seed: u64
- Record (8 bytes each):
    - kind: u8 (0 = compartment, 1 = translation)
    - rec_flags: u8 (reserved, 0)
    - from_compartment: u16
    - to_compartment: u16 (0 for kind=0)
    - reserved: u16 (0)

CLI examples:
    python scripts/generate_compartment_assignments.py \
        --total 1000000 --seed 123 --output out/assignments.bin \
        --comp 0=1 --trans 0->1=0.5 --trans 1->0=0.5 --comp 1=1

Reading back is straightforward: read 32-byte header, then iterate records of 8 bytes.
"""

import argparse
import itertools
import math
import os
import random
import struct
import sys
from array import array
from dataclasses import dataclass
from typing import Iterable, List, Tuple


MAGIC = b"TCASSIGN"  # 8 bytes
VERSION = 1
RECORD_SIZE = 8  # bytes
HEADER_SIZE = 32  # bytes

# Header layout: <8s B B H I Q Q
# 8s  : magic
# B   : version
# B   : record_size
# H   : flags (bit0=1 => little-endian)
# I   : num_compartments
# Q   : num_records
# Q   : seed
HEADER_STRUCT = struct.Struct("<8sBBHIQQ")


@dataclass(frozen=True)
class Category:
    """Represents one category of example to allocate.

    kind: 'C' (compartment) or 'T' (translation)
    src: for 'C' this is the compartment id; for 'T' the source compartment id
    dst: for 'C' 0; for 'T' the target compartment id
    """

    kind: str
    src: int
    dst: int

    def key(self) -> Tuple[int, int, int]:
        # Deterministic tiebreak key: kind ('C' before 'T'), then ids ascending
        kind_rank = 0 if self.kind == "C" else 1
        return (kind_rank, self.src, self.dst)


def parse_comp_arg(arg: str) -> Tuple[int, float]:
    """Parse a compartment weight argument of form 'ID=WEIGHT'."""
    if "=" not in arg:
        raise ValueError(f"Invalid --comp value '{arg}'. Expected 'ID=WEIGHT'.")
    id_str, w_str = arg.split("=", 1)
    comp_id = int(id_str.strip())
    weight = float(w_str.strip())
    if comp_id < 0:
        raise ValueError("Compartment id must be non-negative")
    if weight < 0:
        raise ValueError("Weight must be non-negative")
    return comp_id, weight


def parse_trans_arg(arg: str) -> Tuple[int, int, float]:
    """Parse a translation weight argument like 'SRC->DST=WEIGHT'. Supports ':', ',' as separators.
    Examples: '0->1=0.5', '0:1=0.5', '0,1=0.5'
    """
    if "=" not in arg:
        raise ValueError(f"Invalid --trans value '{arg}'. Expected 'SRC->DST=WEIGHT'.")
    left, w_str = arg.split("=", 1)
    left = left.strip()
    weight = float(w_str.strip())
    for sep in ("->", ":", ","):
        if sep in left:
            src_str, dst_str = left.split(sep, 1)
            src = int(src_str.strip())
            dst = int(dst_str.strip())
            break
    else:
        raise ValueError(f"Invalid translation spec '{left}'. Use 'SRC->DST'.")
    if min(src, dst) < 0:
        raise ValueError("Compartment ids must be non-negative")
    if weight < 0:
        raise ValueError("Weight must be non-negative")
    return src, dst, weight


def largest_remainder_allocations(weights: List[Tuple[Category, float]], total: int) -> List[Tuple[Category, int]]:
    """Convert weights to integer counts summing exactly to total using Largest Remainder.

    Tiebreaker is the category key order (kind, src, dst).
    """
    if total < 0:
        raise ValueError("Total must be non-negative")
    sum_w = sum(w for _, w in weights)
    if total == 0:
        return [(cat, 0) for cat, _ in weights]
    if sum_w <= 0:
        raise ValueError("Sum of weights must be > 0 when total > 0")

    exacts: List[Tuple[Category, float]] = []
    floors: List[int] = []
    fracs: List[float] = []
    for cat, w in weights:
        exact = (total * w) / sum_w
        floor_v = int(math.floor(exact))
        frac = exact - floor_v
        exacts.append((cat, exact))
        floors.append(floor_v)
        fracs.append(frac)

    allocated = sum(floors)
    remainder = total - allocated
    # Determine deterministic order key for tiebreaking
    order = [cat.key() for cat, _ in weights]
    idxs = list(range(len(weights)))
    # Sort indices by fractional remainder descending, then by key ascending
    idxs.sort(key=lambda i: (-fracs[i], order[i]))

    counts = floors[:]
    for i in range(remainder):
        counts[idxs[i]] += 1

    return [(weights[i][0], counts[i]) for i in range(len(weights))]


def encode_record_u64(kind: int, src: int, dst: int) -> int:
    """Pack a record into a 64-bit unsigned integer (little-endian layout when written)."""
    if not (0 <= kind <= 255):
        raise ValueError("kind must fit in u8")
    if not (0 <= src <= 0xFFFF and 0 <= dst <= 0xFFFF):
        raise ValueError("compartment ids must fit in u16")
    rec_flags = 0
    value = (kind & 0xFF)
    value |= (rec_flags & 0xFF) << 8
    value |= (src & 0xFFFF) << 16
    value |= (dst & 0xFFFF) << 32
    # top 16 bits reserved as 0
    return value


def derive_num_compartments(categories: Iterable[Category]) -> int:
    max_id = 0
    any_seen = False
    for c in categories:
        any_seen = True
        max_id = max(max_id, c.src)
        if c.kind == "T":
            max_id = max(max_id, c.dst)
    if not any_seen:
        return 0
    return max_id + 1


def build_records_array(allocations: List[Tuple[Category, int]]) -> array:
    total = sum(count for _, count in allocations)
    arr = array("Q", [0]) * total  # preallocate
    pos = 0
    for cat, count in allocations:
        if count == 0:
            continue
        if cat.kind == "C":
            kind_code = 0
            src = cat.src
            dst = 0
        else:
            kind_code = 1
            src = cat.src
            dst = cat.dst
        value = encode_record_u64(kind_code, src, dst)
        # Fill segment
        for i in range(count):
            arr[pos + i] = value
        pos += count
    assert pos == total
    return arr


def deterministic_shuffle_in_place(arr: array, seed: int) -> None:
    rnd = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
    rnd.shuffle(arr)


def write_binary_file(
    output_path: str,
    num_compartments: int,
    num_records: int,
    seed: int,
    records: array,
) -> None:
    flags = 0
    flags |= 1  # little-endian
    header = HEADER_STRUCT.pack(
        MAGIC, VERSION, RECORD_SIZE, flags, int(num_compartments), int(num_records), int(seed)
    )
    assert len(header) == HEADER_SIZE

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        # Write records in native endianness (we expect little-endian platforms). This matches header flags.
        records.tofile(f)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate deterministic compartment/translation assignments")
    p.add_argument("--output", "-o", type=str, required=True, help="Output .bin file path")
    p.add_argument("--total", "-n", type=float, default=1_000_000, help="Total number of examples (default 1e6)")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed (int)")
    p.add_argument(
        "--comp",
        action="append",
        default=[],
        help="Compartment weight 'ID=WEIGHT'. Can be repeated.",
    )
    p.add_argument(
        "--trans",
        action="append",
        default=[],
        help="Translation weight 'SRC->DST=WEIGHT'. Can be repeated.",
    )
    p.add_argument(
        "--num-compartments",
        type=int,
        default=None,
        help="Override number of compartments (default derived from max id)",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write file; print summary only")
    p.add_argument("--no-shuffle", action="store_true", help="Disable shuffling")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        total = int(round(float(args.total)))
        if total < 0:
            raise ValueError("--total must be non-negative")
    except Exception as e:
        print(f"Invalid --total value: {e}", file=sys.stderr)
        return 2

    comp_weights: List[Tuple[Category, float]] = []
    for s in args.comp:
        comp_id, w = parse_comp_arg(s)
        comp_weights.append((Category("C", comp_id, 0), w))

    trans_weights: List[Tuple[Category, float]] = []
    for s in args.trans:
        src, dst, w = parse_trans_arg(s)
        trans_weights.append((Category("T", src, dst), w))

    # Merge and sort by deterministic key for tie-breaking
    all_weights: List[Tuple[Category, float]] = comp_weights + trans_weights
    if not all_weights:
        print("No weights provided. Use --comp/--trans.", file=sys.stderr)
        return 2

    # Sort only for deterministic tie-breaking; allocation is independent of order otherwise
    all_weights.sort(key=lambda cw: cw[0].key())

    allocations = largest_remainder_allocations(all_weights, total)

    num_compartments = (
        int(args.num_compartments)
        if args.num_compartments is not None
        else derive_num_compartments(cat for cat, _ in allocations)
    )

    # Summaries
    sum_counts = sum(count for _, count in allocations)
    assert sum_counts == total

    print(f"Total examples: {total}")
    print(f"Seed: {args.seed}")
    print(f"Derived num_compartments: {num_compartments}")
    print("Allocations:")
    for cat, count in allocations:
        if cat.kind == "C":
            label = f"C {cat.src}"
        else:
            label = f"T {cat.src}->{cat.dst}"
        # Find original weight for display
        w = next(w for c, w in all_weights if c == cat)
        print(f"  {label:>10}: {count} (weight={w})")

    if args.dry_run:
        print("Dry-run: not writing output file.")
        return 0

    # Build records and shuffle
    records = build_records_array(allocations)
    if not args.no_shuffle and total > 1:
        deterministic_shuffle_in_place(records, args.seed)

    write_binary_file(args.output, num_compartments, total, args.seed, records)
    print(f"Wrote {total} records to {args.output} (header {HEADER_SIZE} bytes, record {RECORD_SIZE} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

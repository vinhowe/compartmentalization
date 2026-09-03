"""Versioned token-shard I/O shared by preprocessing and training."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# The magic identifies the shard's binary LAYOUT, not the tokenizer or the
# corpus -- it is the first 4 bytes of every .bin header. Bumping it is
# therefore a read-compatibility event: shards already on disk keep whatever
# magic they were written with, so the reader has to accept every magic we have
# ever emitted or those corpora become unloadable.
#
# 20251013 -- original
# 20260808 -- current; layout is unchanged (VERSION still selects uint16/uint32),
#             bumped to date-stamp the corpora built for the redesign.
MAGIC = 20260808
ACCEPTED_MAGICS = frozenset({20251013, 20260808})
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize
VERSION_UINT32 = 1
VERSION_UINT16 = 2


def _header(path: str | os.PathLike[str]) -> np.ndarray:
    with open(path, "rb") as handle:
        header = np.frombuffer(handle.read(HEADER_BYTES), dtype=np.int32)
    if len(header) != HEADER_INTS:
        raise ValueError(f"truncated token-shard header: {path}")
    if int(header[0]) not in ACCEPTED_MAGICS:
        raise ValueError(
            f"token-shard magic mismatch: {path} has {int(header[0])}, "
            f"expected one of {sorted(ACCEPTED_MAGICS)}"
        )
    version = int(header[1])
    if version not in {VERSION_UINT32, VERSION_UINT16}:
        raise ValueError(f"unsupported token-shard version {version}: {path}")
    if int(header[2]) < 0:
        raise ValueError(f"negative token count in shard: {path}")
    if version == VERSION_UINT16 and int(header[3]) != 16:
        raise ValueError(f"version-2 shard does not declare 16-bit tokens: {path}")
    return header


def peek_data_shard(path: str | os.PathLike[str]) -> int:
    return int(_header(path)[2])


def load_data_shard(path: str | os.PathLike[str]) -> np.ndarray:
    header = _header(path)
    count = int(header[2])
    dtype = np.uint32 if int(header[1]) == VERSION_UINT32 else np.uint16
    expected_bytes = HEADER_BYTES + count * np.dtype(dtype).itemsize
    actual_bytes = Path(path).stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"token-shard size mismatch for {path}: {actual_bytes} != {expected_bytes}"
        )
    tokens = np.fromfile(path, dtype=dtype, offset=HEADER_BYTES, count=count)
    if len(tokens) != count:
        raise ValueError(f"token-shard count mismatch for {path}: {len(tokens)} != {count}")
    return tokens


def write_data_shard(
    path: str | os.PathLike[str], tokens: np.ndarray, *, use_uint16: bool,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(tokens).reshape(-1)
    if len(flat) > np.iinfo(np.int32).max:
        raise ValueError("one token shard cannot exceed signed-int32 token count")
    if len(flat) and int(flat.min()) < 0:
        raise ValueError("token ids must be nonnegative")
    dtype = np.uint16 if use_uint16 else np.uint32
    if len(flat) and int(flat.max()) > np.iinfo(dtype).max:
        raise ValueError(f"token id exceeds {dtype} range")
    version = VERSION_UINT16 if use_uint16 else VERSION_UINT32
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = version
    header[2] = len(flat)
    header[3] = 16 if use_uint16 else 32
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            header.tofile(handle)
            flat.astype(dtype, copy=False).tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

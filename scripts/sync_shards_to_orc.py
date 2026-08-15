#!/usr/bin/env python3
"""Stream finished token shards to ORC while tokenization is still running.

Tokenizing the corpus takes hours; the transfer takes about one. Rather than
running them back to back, this syncs each shard as soon as it is provably
complete, so the transfer finishes essentially when tokenization does.

The safety problem this solves: `data_common.write_datafile` opens the FINAL
filename and writes in place, header first. A shard being written is therefore
visible on disk under its real name, with a valid header that already declares
its eventual token count while the file is still short. A naive `rsync *.bin`
would happily copy that truncated file, and nothing downstream would notice
until a training run read past the end.

So completeness is checked exactly, not by mtime or by "skip the newest":

    complete  <=>  filesize == 1024 + ntok * itemsize

with ntok read from the 256-int32 header. Only files passing that are synced.

    python3 scripts/sync_shards_to_orc.py \
        --src data/fineweb350B-bpe16384-nodedup \
        --dest /nobackup/autodelete/grp/grp_pccl/vin/data/fineweb350B-bpe16384-nodedup \
        --done-when-exists data/fineweb350B-bpe16384-nodedup/_TOKENIZATION_COMPLETE
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

import numpy as np

# Matches src/datafile.py, which is what fineweb2.py now writes and what the
# training loader reads. NOT data_common.HEADERS_INFO -- that module carries an
# older, incompatible magic (20240520) that no current corpus uses.
ACCEPTED_MAGICS = frozenset({20251013, 20260808})   # must track src/datafile.py
VERSION_ITEMSIZE = {1: 4, 2: 2}      # VERSION_UINT32 -> 4 bytes, VERSION_UINT16 -> 2
HEADER_BYTES = 256 * 4


def shard_is_complete(path: Path) -> tuple[bool, str]:
    """Exact completeness test: does the file hold every token its header claims?"""
    try:
        size = path.stat().st_size
        if size < HEADER_BYTES:
            return False, "shorter than header"
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32)
        magic, version = int(header[0]), int(header[1])
        if magic not in ACCEPTED_MAGICS:
            return False, f"bad magic {magic}"
        if version not in VERSION_ITEMSIZE:
            return False, f"unknown version {version}"
        itemsize = VERSION_ITEMSIZE[version]
        ntok = int(header[2])
        expected = HEADER_BYTES + ntok * itemsize
        if size == expected:
            return True, f"{ntok:,} tokens"
        return False, f"{size:,}/{expected:,} bytes (still writing)"
    except OSError as e:
        return False, f"unreadable: {e}"


def rsync(files: list[Path], src_root: Path, dest: str, ssh_config: str) -> bool:
    if not files:
        return True
    rel = [str(f.relative_to(src_root)) for f in files]
    proc = subprocess.run(
        ["rsync", "-a", "--no-compress", "--partial-dir=.rsync-partial",
         "--files-from=-", "-e", f"ssh -F {ssh_config} -o ConnectTimeout=20",
         str(src_root), dest],
        input="\n".join(rel), text=True, capture_output=True,
    )
    if proc.returncode != 0:
        print(f"  rsync failed ({proc.returncode}): {proc.stderr.strip()[:300]}", flush=True)
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="local corpus directory")
    p.add_argument("--dest", required=True, help="remote path (no host prefix)")
    p.add_argument("--host", default="orc-th443")
    p.add_argument("--ssh-config", default="/root/.ssh/orc_config")
    p.add_argument("--interval", type=int, default=300, help="seconds between passes")
    p.add_argument("--done-when-exists", default=None,
                   help="sentinel file; once present and all shards are synced, exit")
    args = p.parse_args()

    src = Path(args.src).resolve()
    dest = f"{args.host}:{args.dest}/"
    subprocess.run(["ssh", "-F", args.ssh_config, args.host, f"mkdir -p {args.dest}"],
                   capture_output=True)

    synced: set[str] = set()
    total_bytes = 0
    while True:
        shards = sorted(src.glob("*.bin"))
        ready, pending = [], []
        for s in shards:
            if s.name in synced:
                continue
            ok, why = shard_is_complete(s)
            (ready if ok else pending).append((s, why))

        if ready:
            files = [s for s, _ in ready]
            size = sum(f.stat().st_size for f in files)
            print(f"[{time.strftime('%H:%M:%S')}] syncing {len(files)} shard(s), "
                  f"{size/1e9:.1f} GB", flush=True)
            t0 = time.time()
            if rsync(files, src, dest, args.ssh_config):
                synced.update(f.name for f in files)
                total_bytes += size
                dt = max(time.time() - t0, 1e-6)
                print(f"           ok in {dt:.0f}s ({size/1e6/dt:.0f} MB/s) — "
                      f"{len(synced)} shards, {total_bytes/1e9:.1f} GB total", flush=True)
        if pending:
            s, why = pending[-1]
            print(f"[{time.strftime('%H:%M:%S')}] in flight: {s.name} ({why}); "
                  f"{len(synced)} synced", flush=True)

        if args.done_when_exists and os.path.exists(args.done_when_exists) and not pending:
            remaining = [s for s in src.glob("*.bin") if s.name not in synced]
            if not remaining:
                print(f"done: {len(synced)} shards, {total_bytes/1e9:.1f} GB", flush=True)
                return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

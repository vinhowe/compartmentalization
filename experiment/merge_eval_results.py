#!/usr/bin/env python3
"""
Merge per-GPU evaluation result files into a single val_metrics.json.

This script:
1. Loads the original val_metrics.json (preserves existing data)
2. Loads each per-GPU file (val_metrics_gpu{0-7}.json)
3. Merges them, with per-GPU results overriding original for same experiment key
4. Writes the merged result back to val_metrics.json
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def backup_file(path: Path) -> None:
    """Create a backup of the file if it exists."""
    if path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)


def merge_results(world_size: int = 8, cleanup: bool = False):
    """Merge per-GPU result files into a single metrics file."""
    original_file = Path("val_metrics.json")

    # Start with original data (preserves existing results)
    merged = {}
    if original_file.exists():
        print(f"Loading original file: {original_file}")
        with open(original_file, "r") as f:
            merged = json.load(f)
        print(f"  Found {len(merged)} experiments in original file")

    # Override with per-GPU results (newer data wins)
    gpu_files_found = 0
    experiments_updated = 0
    for rank in range(world_size):
        gpu_file = Path(f"val_metrics_gpu{rank}.json")
        if gpu_file.exists():
            gpu_files_found += 1
            print(f"Loading per-GPU file: {gpu_file}")
            with open(gpu_file, "r") as f:
                gpu_data = json.load(f)
            print(f"  Found {len(gpu_data)} experiments")
            for key, value in gpu_data.items():
                merged[key] = value  # Override original
                experiments_updated += 1
        else:
            print(f"Per-GPU file not found: {gpu_file} (skipping)")

    print(f"\nMerge summary:")
    print(f"  GPU files found: {gpu_files_found}/{world_size}")
    print(f"  Experiments in merged result: {len(merged)}")
    print(f"  Experiments updated from GPU files: {experiments_updated}")

    # Write merged result back (atomic: write temp file, then rename)
    backup_file(original_file)
    print(f"\nWriting merged result to {original_file}")
    fd, tmp_path = tempfile.mkstemp(dir=original_file.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(merged, f, indent=4)
        Path(tmp_path).replace(original_file)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    # Optionally clean up per-GPU files
    if cleanup:
        print("\nCleaning up per-GPU files...")
        for rank in range(world_size):
            gpu_file = Path(f"val_metrics_gpu{rank}.json")
            if gpu_file.exists():
                gpu_file.unlink()
                print(f"  Removed {gpu_file}")
            # Also remove backup if it exists
            backup_file_path = gpu_file.with_suffix(".json.bak")
            if backup_file_path.exists():
                backup_file_path.unlink()

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-GPU evaluation results into a single file"
    )
    parser.add_argument(
        "--world-size", type=int, default=8,
        help="Number of GPU workers (default: 8)"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Remove per-GPU files after merging"
    )
    args = parser.parse_args()

    merge_results(world_size=args.world_size, cleanup=args.cleanup)


if __name__ == "__main__":
    main()

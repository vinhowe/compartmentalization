#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Sweep monitor — single-command health check for sweep_runner jobs.

Usage:
    ./scripts/sweep-monitor.py sweeps/bpe16384-n3-n5.yaml
    ./scripts/sweep-monitor.py sweeps/bpe16384-n3-n5.yaml --watch     # refresh every 30s
    ./scripts/sweep-monitor.py sweeps/bpe16384-n3-n5.yaml --watch 60  # refresh every 60s
"""

import argparse
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

LOGDIR = Path(__file__).resolve().parent.parent / "logs"
TC_STORAGE_ROOT = os.environ.get(
    "TC_STORAGE_ROOT",
    "/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression",
)

# Patterns that indicate real problems (not just warnings)
ERROR_PATTERNS = [
    (r"Out Of Memory", "OOM"),
    (r"oom_kill", "OOM"),
    (r"CUDA out of memory", "GPU-OOM"),
    (r"Traceback \(most recent", "TRACEBACK"),
    (r"RuntimeError:", "RUNTIME-ERR"),
    (r"CANCELLED", "CANCELLED"),
    (r"error:.*STEP.*CANCELLED", "CANCELLED"),
    (r"proxy socket not found", "PROXY-FAIL"),
    (r"sidecar.*failed\|error\|exit", "SIDECAR-FAIL"),
    (r"timed out after.*sec", "WANDB-TIMEOUT"),
    (r"CommError.*timed out", "WANDB-TIMEOUT"),
]

# Patterns to ignore (noisy but harmless)
IGNORE_PATTERNS = [
    r"wandb query failed",
    r"Could not check for duplicates",
    r"CUDA_VISIBLE_DEVICES",
    r"Network error.*retry loop",
    r"error in system default config",
    r"socat.*Connection refused",
    r"Writing assignment records.*socat",
]


def run(cmd, **kwargs):
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def get_my_jobs():
    """Return list of dicts with job info from squeue."""
    fmt = "%i|%P|%q|%T|%M|%D|%R|%j"
    out, _, rc = run(["squeue", "--me", "--noheader", "-o", fmt])
    if rc != 0 or not out:
        return []
    jobs = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 8:
            jobs.append({
                "jobid": parts[0],
                "partition": parts[1],
                "qos": parts[2],
                "state": parts[3],
                "time": parts[4],
                "nodes": parts[5],
                "reason": parts[6],
                "name": parts[7],
            })
    return jobs


def get_completed_jobs():
    """Return recently completed/failed jobs from sacct."""
    out, _, rc = run([
        "sacct", "--me", "-n", "--starttime=now-2days",
        "--format=JobID,Partition,State,ExitCode,Elapsed,NodeList",
        "--parsable2", "--noheader",
    ])
    if rc != 0 or not out:
        return []
    jobs = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 6 and not "." in parts[0]:  # skip steps
            jobs.append({
                "jobid": parts[0],
                "partition": parts[1],
                "state": parts[2],
                "exitcode": parts[3],
                "elapsed": parts[4],
                "nodelist": parts[5],
            })
    return jobs


def scan_log_errors(jobid):
    """Scan stdout/stderr logs for a job, return list of (category, line)."""
    errors = []
    for suffix in ("out", "err"):
        for logfile in LOGDIR.glob(f"*-{jobid}*.{suffix}"):
            try:
                text = logfile.read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                # Skip ignored patterns
                if any(re.search(p, line) for p in IGNORE_PATTERNS):
                    continue
                for pattern, category in ERROR_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        errors.append((category, line.strip()[:120]))
                        break
    return errors


def get_sweep_status(sweep_yaml):
    """Run sweep_runner.py --status and parse output."""
    env = os.environ.copy()
    env["TC_STORAGE_ROOT"] = TC_STORAGE_ROOT
    # Use the project venv python (has yaml, wandb, etc.)
    project_root = Path(__file__).resolve().parent.parent
    venv_python = project_root / ".supercomputer-venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else "python"
    out, err, rc = run(
        [python, "sweep_runner.py", "--sweep", sweep_yaml, "--status"],
        env=env,
        cwd=str(project_root),
    )
    if rc != 0:
        return None, err or out
    return out, None


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def monitor(sweep_yaml, watch_interval=None):
    while True:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        os.system("clear" if watch_interval else "true")
        print(f"Sweep Monitor  {now}  [{sweep_yaml}]")

        # --- Job Status ---
        print_section("JOB STATUS")
        jobs = get_my_jobs()
        sweep_jobs = [j for j in jobs if j["name"] == "translation-compression"]
        if sweep_jobs:
            states = Counter(j["state"] for j in sweep_jobs)
            parts = Counter(f"{j['partition']}/{j['qos']}" for j in sweep_jobs)
            print(f"  Total: {len(sweep_jobs)}  |  " +
                  "  ".join(f"{s}: {n}" for s, n in sorted(states.items())))
            print(f"  By queue: " +
                  "  ".join(f"{p}: {n}" for p, n in sorted(parts.items())))
            print()
            print(f"  {'JOBID':>10}  {'PART':>6}  {'QOS':>8}  {'STATE':>10}  {'TIME':>10}  REASON")
            for j in sweep_jobs:
                reason = j["reason"] if j["state"] != "RUNNING" else ""
                print(f"  {j['jobid']:>10}  {j['partition']:>6}  {j['qos']:>8}  "
                      f"{j['state']:>10}  {j['time']:>10}  {reason}")
        else:
            print("  No running/pending jobs found.")

        # --- Recently Finished ---
        print_section("RECENTLY FINISHED (last 2 days)")
        finished = get_completed_jobs()
        problem_jobs = [j for j in finished
                        if j["state"] not in ("COMPLETED", "PENDING", "RUNNING", "")]
        if problem_jobs:
            for j in problem_jobs[-10:]:
                flag = " !!!" if j["state"] in ("FAILED", "OUT_OF_MEMORY", "TIMEOUT") else ""
                print(f"  {j['jobid']:>10}  {j['partition']:>6}  {j['state']:>16}  "
                      f"exit={j['exitcode']}  {j['elapsed']}  {j['nodelist']}{flag}")
        else:
            print("  No failed/cancelled jobs.")

        # --- Log Errors ---
        print_section("LOG ERRORS (running jobs)")
        running_ids = [j["jobid"] for j in sweep_jobs if j["state"] == "RUNNING"]
        all_errors = []
        for jid in running_ids:
            errs = scan_log_errors(jid)
            for cat, line in errs:
                all_errors.append((jid, cat, line))

        if all_errors:
            by_cat = Counter(cat for _, cat, _ in all_errors)
            print(f"  Summary: " +
                  "  ".join(f"{cat}: {n}" for cat, n in by_cat.most_common()))
            print()
            # Show up to 10 unique errors
            seen = set()
            for jid, cat, line in all_errors:
                key = (cat, line[:60])
                if key not in seen:
                    seen.add(key)
                    print(f"  [{cat}] job {jid}: {line}")
                if len(seen) >= 10:
                    remaining = len(all_errors) - len(seen)
                    if remaining > 0:
                        print(f"  ... and {remaining} more")
                    break
        else:
            print("  No errors detected.")

        # --- Sweep Progress ---
        print_section("SWEEP PROGRESS")
        status_out, status_err = get_sweep_status(sweep_yaml)
        if status_out:
            # Parse the table and summarize
            lines = status_out.splitlines()
            progress_counts = Counter()
            for line in lines:
                if "|" not in line or "cfg_hash" in line or "---" in line:
                    continue
                cols = [c.strip() for c in line.split("|")]
                if len(cols) >= 7:
                    progress = cols[-1]
                    if "unclaimed" in progress:
                        progress_counts["unclaimed"] += 1
                    elif "claimed" in progress:
                        progress_counts["claimed"] += 1
                    elif "/" in progress:
                        # iter/max_iter
                        try:
                            cur, total = progress.split("/")
                            pct = int(cur) / int(total) * 100
                            if pct >= 100:
                                progress_counts["done"] += 1
                            elif pct >= 50:
                                progress_counts[">50%"] += 1
                            elif pct >= 10:
                                progress_counts["10-50%"] += 1
                            else:
                                progress_counts["<10%"] += 1
                        except ValueError:
                            progress_counts["unknown"] += 1
                    else:
                        progress_counts["unknown"] += 1

            total = sum(progress_counts.values())
            print(f"  Total configs: {total}")
            for label in ["done", ">50%", "10-50%", "<10%", "claimed", "unclaimed", "unknown"]:
                n = progress_counts.get(label, 0)
                if n:
                    bar = "#" * (n * 30 // max(total, 1))
                    print(f"  {label:>10}: {n:3}  {bar}")

            # Print full table
            print()
            for line in lines:
                print(f"  {line}")
        elif status_err:
            print(f"  Error: {status_err[:200]}")

        if not watch_interval:
            break
        print(f"\n  [refreshing in {watch_interval}s — Ctrl-C to stop]")
        try:
            time.sleep(watch_interval)
        except KeyboardInterrupt:
            print()
            break


def main():
    parser = argparse.ArgumentParser(description="Monitor sweep_runner jobs")
    parser.add_argument("sweep", help="Path to sweep YAML file")
    parser.add_argument("--watch", nargs="?", const=30, type=int, metavar="SECS",
                        help="Refresh every N seconds (default: 30)")
    args = parser.parse_args()
    monitor(args.sweep, args.watch)


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "rich",
# ]
# ///
"""
gpu-dashboard.py — Slurm GPU TUI Dashboard

Shows per-GPU-type usage bars for A100, H100, H200 (and A200 if present).
Deduplicates nodes across overlapping partitions (dw87/dw87long/gstandby).

Bar segments:
  ██ green  = YOUR jobs
  ██ gray   = idle GPUs (low-hanging fruit — available to queue NOW)
  ██ amber  = in use by OTHER users
  ▒▒ dark   = down / drained / maintenance (can't queue currently)

Usage:
  uv run scripts/gpu-dashboard.py            # one-shot
  uv run scripts/gpu-dashboard.py -w         # auto-refresh every 30s
  uv run scripts/gpu-dashboard.py -w 10      # auto-refresh every 10s
  uv run scripts/gpu-dashboard.py -v         # verbose: show per-node breakdown
"""

import argparse
import subprocess
import re
import os
import sys
import signal
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.live import Live

# ── GPU types of interest ──────────────────────────────────────────────
TARGET_GPU_TYPES = ["a100", "h100", "h200", "a200"]

# ── Partitions that can have GPUs the user might access ────────────────
GPU_PARTITIONS = {"dw", "dwmatrix", "cs", "cs2", "cssp1", "eng", "m13h"}

# ── Access tier: which QOS names indicate "owner/primary" vs "group" ───
PRIMARY_QOS_MAP = {
    "dw": {"dw87", "dw87long"},
}
GROUP_QOS_MAP = {
    "cs":       {"cs", "cslong"},
    "cs2":      {"cs", "cslong"},
    "cssp1":    {"cssq1"},
    "m13h":     {"gpu"},
    "eng":      set(),
    "dwmatrix": {"matrix"},
}

# ── Rich style names ──────────────────────────────────────────────────
S_MINE   = "bold green"
S_IDLE   = "grey70"
S_OTHERS = "dark_orange"
S_DOWN   = "red3"
S_DIM    = "dim"
S_TITLE  = "bold"
S_USER   = "dodger_blue2"
S_TIER_OWN = "green"
S_TIER_GRP = "dodger_blue2"
S_TIER_STB = "grey50"

TIER_LABEL = {"primary": "owner", "group": "group", "standby": "standby"}
TIER_STYLE = {"primary": S_TIER_OWN, "group": S_TIER_GRP, "standby": S_TIER_STB}

console = Console()

# ── Shell helpers ──────────────────────────────────────────────────────
def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""

def get_user():
    return os.environ.get("USER") or run("whoami")

# ── Hostlist expansion (pure Python, no subprocess) ────────────────────

def expand_hostlist(s):
    """Expand 'dw-1-[1-5],dw-2-[1-4]' → ['dw-1-1', ..., 'dw-2-4']."""
    if not s:
        return []
    nodes = []
    for part in re.split(r",(?![^\[]*\])", s):
        m = re.match(r"(.+?)\[(.+)\](.*)", part)
        if m:
            prefix, ranges, suffix = m.group(1), m.group(2), m.group(3)
            for r in ranges.split(","):
                if "-" in r:
                    a, b = r.split("-", 1)
                    w = len(a)
                    for i in range(int(a), int(b) + 1):
                        nodes.append(f"{prefix}{str(i).zfill(w)}{suffix}")
                else:
                    nodes.append(f"{prefix}{r}{suffix}")
        else:
            nodes.append(part.strip())
    return nodes

# ── Batch data collection ──────────────────────────────────────────────

def parse_user_qos(raw):
    account = None
    qos = set()
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            if parts[0] and not account:
                account = parts[0]
            for q in parts[3].split(","):
                q = q.strip()
                if q:
                    qos.add(q)
    return account or "?", qos


def parse_all_partitions(raw):
    partitions = {}
    for block in re.split(r"\n\n+", raw):
        name = ""
        allow_qos = set()
        nodes_str = ""
        for line in block.splitlines():
            line = line.strip()
            m = re.search(r"PartitionName=(\S+)", line)
            if m:
                name = m.group(1)
            m = re.search(r"AllowQos=(\S+)", line)
            if m:
                allow_qos = {q for q in m.group(1).split(",") if q}
            m = re.search(r"(?<!\w)Nodes=(\S+)", line)
            if m and "Total" not in line:
                nodes_str = m.group(1)
        if name and name in GPU_PARTITIONS:
            partitions[name] = {"allow_qos": allow_qos, "nodes_str": nodes_str}
    return partitions


def parse_all_nodes(raw):
    nodes = {}
    for block in re.split(r"\n\n+", raw):
        if not block.strip():
            continue
        name = gpu_type = state = ""
        total = alloc = 0
        for line in block.splitlines():
            line = line.strip()
            m = re.search(r"NodeName=(\S+)", line)
            if m:
                name = m.group(1)
            m = re.search(r"State=(\S+)", line)
            if m and not state:
                state = m.group(1)
            if re.match(r"Gres=", line) or "   Gres=" in line:
                m = re.search(r"gpu:(\w+):(\d+)", line)
                if m and "Used" not in line:
                    gpu_type = m.group(1).lower()
                    total = int(m.group(2))
            if "AllocTRES=" in line:
                m = re.search(r"gres/gpu=(\d+)", line)
                if m:
                    alloc = int(m.group(1))
        if name:
            nodes[name] = {
                "gpu_type": gpu_type, "total": total,
                "alloc": alloc, "state": state,
            }
    return nodes


def parse_my_jobs(raw_squeue, raw_jobs_detail):
    usage = defaultdict(int)
    job_gpus = {}
    for block in re.split(r"\n\n+", raw_jobs_detail):
        jid = ""
        gpus = 0
        for line in block.splitlines():
            m = re.search(r"JobId=(\d+)", line)
            if m:
                jid = m.group(1)
            m = re.search(r"AllocTRES=\S*gres/gpu=(\d+)", line)
            if m:
                gpus = int(m.group(1))
        if jid:
            job_gpus[jid] = gpus

    for line in raw_squeue.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2:
            jid, node = parts[0].strip(), parts[1].strip()
            if jid in job_gpus and job_gpus[jid] > 0:
                usage[node] += job_gpus[jid]
    return dict(usage)


def node_is_down(state):
    s = state.upper()
    return any(k in s for k in ("DOWN", "DRAIN", "MAINT", "NOT_RESPONDING"))


def access_tier(partition_name, user_qos, part_allow_qos):
    accessible = user_qos & part_allow_qos
    if not accessible:
        return "none"
    if accessible & PRIMARY_QOS_MAP.get(partition_name, set()):
        return "primary"
    if accessible & GROUP_QOS_MAP.get(partition_name, set()):
        return "group"
    if accessible & {"gstandby", "standby"}:
        return "standby"
    return "standby"


# ── Bar rendering with Rich ───────────────────────────────────────────

def render_bar(mine, idle, others, down, width):
    """Build a Rich Text bar of exactly `width` visible chars."""
    total = mine + idle + others + down
    if total == 0:
        return Text(" " * width)

    segments = [
        (mine,   S_MINE,   "█"),
        (idle,   S_IDLE,   "█"),
        (others, S_OTHERS, "█"),
        (down,   S_DOWN,   "▒"),
    ]

    widths = []
    used = 0
    for i, (count, _, _) in enumerate(segments):
        if i == len(segments) - 1:
            w = width - used
        else:
            w = round(count / total * width) if total else 0
            if count > 0 and w == 0:
                w = 1
        widths.append(w)
        used += w

    while sum(widths) > width:
        idx = max(range(len(widths)), key=lambda i: widths[i])
        widths[idx] -= 1
    while sum(widths) < width:
        idx = max(range(len(widths)), key=lambda i: segments[i][0])
        widths[idx] += 1

    bar = Text()
    for (count, style, char), w in zip(segments, widths):
        if w > 0:
            bar.append(char * w, style=f"on {style.split()[-1]}" if "on" not in style else style)
    return bar


def render_bar_bg(mine, idle, others, down, width):
    """Build bar with background-colored blocks for better visibility."""
    total = mine + idle + others + down
    if total == 0:
        return Text(" " * width)

    segments = [
        (mine,   "white on green",          "█"),
        (idle,   "grey23 on grey78",        "█"),
        (others, "grey23 on dark_orange",   "█"),
        (down,   "red on grey15",           "▒"),
    ]

    widths = []
    used = 0
    for i, (count, _, _) in enumerate(segments):
        if i == len(segments) - 1:
            w = width - used
        else:
            w = round(count / total * width) if total else 0
            if count > 0 and w == 0:
                w = 1
        widths.append(w)
        used += w

    while sum(widths) > width:
        idx = max(range(len(widths)), key=lambda i: widths[i])
        widths[idx] -= 1
    while sum(widths) < width:
        idx = max(range(len(widths)), key=lambda i: segments[i][0])
        widths[idx] += 1

    bar = Text()
    for (count, style, char), w in zip(segments, widths):
        if w > 0:
            bar.append(char * w, style=style)
    return bar


# ── Gather + render one frame ──────────────────────────────────────────

def gather_data(user):
    """Fetch all Slurm data with batched parallel queries. Returns render-ready dict."""
    # Phase 1: 3 independent Slurm queries in parallel
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_acct = pool.submit(
            run,
            f"sacctmgr show associations where user={user} "
            f"format=Account,User,Partition,QOS%200 -p --noheader",
        )
        f_parts = pool.submit(run, "scontrol show partition")
        f_queue = pool.submit(
            run, f"squeue -u {user} -o '%i|%N' -t RUNNING --noheader",
        )
        raw_acct  = f_acct.result()
        raw_parts = f_parts.result()
        raw_queue = f_queue.result()

    account, user_qos = parse_user_qos(raw_acct)
    partitions = parse_all_partitions(raw_parts)

    # Phase 2: expand node lists (pure Python, instant)
    tier_rank = {"primary": 0, "group": 1, "standby": 2}
    node_meta = {}

    for pname, pinfo in partitions.items():
        tier = access_tier(pname, user_qos, pinfo["allow_qos"])
        if tier == "none":
            continue
        my_qos_for_part = user_qos & pinfo["allow_qos"]
        for nname in expand_hostlist(pinfo["nodes_str"]):
            if nname in node_meta:
                old = tier_rank.get(node_meta[nname]["tier"], 99)
                new = tier_rank.get(tier, 99)
                if new < old:
                    node_meta[nname]["tier"] = tier
                node_meta[nname]["partitions"].append(pname)
                node_meta[nname]["my_qos"] |= my_qos_for_part
            else:
                node_meta[nname] = {
                    "tier": tier, "partitions": [pname],
                    "my_qos": set(my_qos_for_part),
                }

    if not node_meta:
        return {"account": account, "by_type": {}, "tier_details": {}}

    # Phase 3: batch node + job queries in parallel
    node_list_str = ",".join(sorted(node_meta.keys()))
    my_job_ids = []
    for line in raw_queue.splitlines():
        parts = line.strip().split("|")
        if parts and parts[0].strip():
            my_job_ids.append(parts[0].strip())

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_nodes = pool.submit(run, f"scontrol show node {node_list_str}")
        if my_job_ids:
            job_cmds = " ; ".join(f"scontrol show job {jid}" for jid in my_job_ids)
            f_jobs = pool.submit(run, job_cmds, 30)
        else:
            f_jobs = None
        raw_nodes = f_nodes.result()
        raw_jobs = f_jobs.result() if f_jobs else ""

    all_node_info = parse_all_nodes(raw_nodes)

    for nname in list(node_meta.keys()):
        ninfo = all_node_info.get(nname)
        if not ninfo or not ninfo["gpu_type"] or ninfo["gpu_type"] not in TARGET_GPU_TYPES:
            del node_meta[nname]

    my_gpus = parse_my_jobs(raw_queue, raw_jobs)

    # Aggregate
    by_type = {}
    for gtype in TARGET_GPU_TYPES:
        by_type[gtype] = {
            "mine": 0, "idle": 0, "others": 0, "down": 0,
            "total": 0, "partitions": set(), "nodes": [],
        }

    for nname in sorted(node_meta.keys()):
        meta = node_meta[nname]
        ninfo = all_node_info.get(nname)
        if not ninfo:
            continue
        gtype = ninfo["gpu_type"]
        entry = by_type[gtype]
        t = ninfo["total"]
        entry["total"] += t
        for p in meta["partitions"]:
            entry["partitions"].add(p)

        down_flag = node_is_down(ninfo["state"])
        if down_flag:
            entry["down"] += t
            n_mine = n_others = n_idle = 0
        else:
            a = ninfo["alloc"]
            n_mine = my_gpus.get(nname, 0)
            n_others = max(0, a - n_mine)
            n_idle = max(0, t - a)
            entry["mine"] += n_mine
            entry["others"] += n_others
            entry["idle"] += n_idle

        entry["nodes"].append({
            "name": nname, "total": t,
            "mine": n_mine, "others": n_others,
            "idle": n_idle, "down": t if down_flag else 0,
            "state": ninfo["state"], "tier": meta["tier"],
            "my_qos": sorted(meta.get("my_qos", set())),
        })

    tier_details = defaultdict(lambda: defaultdict(set))
    for nname, meta in node_meta.items():
        ninfo = all_node_info.get(nname)
        if not ninfo:
            continue
        gtype = ninfo["gpu_type"]
        for p in meta["partitions"]:
            tier_details[gtype][meta["tier"]].add(p)

    return {
        "account": account,
        "by_type": by_type,
        "tier_details": tier_details,
    }


def build_display(user, data, verbose=False):
    """Build a list of Rich renderables for the dashboard."""
    account = data["account"]
    by_type = data["by_type"]
    tier_details = data["tier_details"]

    bar_width = min(console.width - 12, 72)
    renderables = []

    # Header
    header = Text()
    header.append("Slurm GPU Dashboard", style="bold")
    header.append("  user=", style="dim")
    header.append(user, style=S_USER)
    header.append("  account=", style="dim")
    header.append(account, style=S_USER)
    header.append(f"  {datetime.now().strftime('%H:%M:%S')}", style="dim")
    renderables.append(header)
    renderables.append(Text(""))

    for gtype in TARGET_GPU_TYPES:
        entry = by_type.get(gtype, {"total": 0})

        if entry["total"] == 0:
            line = Text()
            line.append(f"{gtype.upper()}", style="bold")
            line.append("  — not found on this cluster", style="dim")
            renderables.append(line)
            renderables.append(Text(""))
            continue

        mine   = entry["mine"]
        idle   = entry["idle"]
        others = entry["others"]
        down   = entry["down"]
        total  = entry["total"]
        accessible = total - down
        parts_str = ", ".join(sorted(entry["partitions"]))

        # Title line
        title = Text()
        title.append(f"{gtype.upper()}", style="bold")
        title.append(f"  {total} total  ({parts_str})")
        renderables.append(title)

        # Bar
        bar = render_bar_bg(mine, idle, others, down, bar_width)
        renderables.append(Text("  ").append_text(bar))

        # Segment legend
        legend = Text("  ")
        if mine:
            legend.append("██", style="green")
            legend.append(f" {mine} mine   ")
        if idle:
            legend.append("██", style="grey70")
            legend.append(f" {idle} idle   ")
        if others:
            legend.append("██", style="dark_orange")
            legend.append(f" {others} others   ")
        if down:
            legend.append("▒▒", style="red3")
            legend.append(f" {down} down")
        renderables.append(legend)

        # Usage ratios
        avail_now = mine + idle
        ratio = Text("  Using: ")
        if avail_now > 0:
            pct = mine / avail_now * 100
            ratio.append(f"{mine}/{avail_now} available now ({pct:.0f}%)")
        else:
            ratio.append("0 available now")
        ratio.append("  |  ")
        if accessible > 0:
            pct = mine / accessible * 100
            ratio.append(f"{mine}/{accessible} accessible ({pct:.0f}%)")
        else:
            ratio.append("0 accessible")
        renderables.append(ratio)

        # Access tiers
        access_line = Text("  Access: ")
        first = True
        for tier in ("primary", "group", "standby"):
            plist = tier_details.get(gtype, {}).get(tier, set())
            if plist:
                if not first:
                    access_line.append("  ")
                access_line.append(
                    f"{','.join(sorted(plist))}({TIER_LABEL[tier]})",
                    style=TIER_STYLE[tier],
                )
                first = False
        if not first:
            renderables.append(access_line)

        # Verbose: node table
        if verbose and entry.get("nodes"):
            table = Table(
                show_header=True, header_style="dim", box=None,
                padding=(0, 1), pad_edge=False,
            )
            table.add_column("", width=1)
            table.add_column("Node", min_width=11)
            table.add_column("GPUs", justify="right", width=4)
            table.add_column("Mine", justify="right", width=4)
            table.add_column("Othr", justify="right", width=4)
            table.add_column("Idle", justify="right", width=4)
            table.add_column("Down", justify="right", width=4)
            table.add_column("State", min_width=6)
            table.add_column("QOS", min_width=10)

            tier_sym = {
                "primary": ("●", S_TIER_OWN),
                "group":   ("●", S_TIER_GRP),
                "standby": ("○", S_TIER_STB),
            }

            for nd in entry["nodes"]:
                st = nd["state"].upper()
                if "DOWN" in st:        ss = "down"
                elif "DRAIN" in st:     ss = "drain"
                elif "MAINT" in st:     ss = "maint"
                elif "NOT_RESP" in st:  ss = "no-resp"
                else:                   ss = nd["state"].split("+")[0].lower()[:8]

                sym, sym_s = tier_sym.get(nd["tier"], (" ", ""))
                qos_str = ",".join(nd.get("my_qos", []))
                table.add_row(
                    Text(sym, style=sym_s),
                    nd["name"],
                    str(nd["total"]),
                    Text(str(nd["mine"]),   style=S_MINE if nd["mine"] else S_DIM),
                    Text(str(nd["others"]), style=S_OTHERS if nd["others"] else S_DIM),
                    Text(str(nd["idle"]),   style=S_IDLE if nd["idle"] else S_DIM),
                    Text(str(nd["down"]),   style=S_DOWN if nd["down"] else S_DIM),
                    Text(ss, style=S_DIM),
                    Text(qos_str, style=S_DIM),
                )
            renderables.append(table)

        renderables.append(Text(""))

    # Footer legend
    foot = Text()
    foot.append("██", style="green")
    foot.append(" mine  ")
    foot.append("██", style="grey70")
    foot.append(" idle  ")
    foot.append("██", style="dark_orange")
    foot.append(" others  ")
    foot.append("▒▒", style="red3")
    foot.append(" down/maint")
    if verbose:
        foot.append("   |   ")
        foot.append("● owner", style=S_TIER_OWN)
        foot.append("  ")
        foot.append("● group", style=S_TIER_GRP)
        foot.append("  ")
        foot.append("○ standby", style=S_TIER_STB)
    renderables.append(foot)

    return renderables


def render_frame(user, verbose=False):
    """Gather data and print one frame."""
    data = gather_data(user)
    renderables = build_display(user, data, verbose)
    console.print()
    for r in renderables:
        console.print(r)
    console.print()


def build_plain_text(user, data, verbose=False):
    """Build a plain-text (no color, no unicode) summary for LLM consumption."""
    account = data["account"]
    by_type = data["by_type"]
    tier_details = data["tier_details"]
    lines = []
    lines.append(f"Slurm GPU Dashboard  user={user}  account={account}  {datetime.now().strftime('%H:%M:%S')}")
    lines.append("")

    for gtype in TARGET_GPU_TYPES:
        entry = by_type.get(gtype, {"total": 0})
        if entry["total"] == 0:
            lines.append(f"{gtype.upper()}: not found on this cluster")
            lines.append("")
            continue

        mine   = entry["mine"]
        idle   = entry["idle"]
        others = entry["others"]
        down   = entry["down"]
        total  = entry["total"]
        accessible = total - down
        parts_str = ", ".join(sorted(entry["partitions"]))

        lines.append(f"{gtype.upper()}  {total} total  ({parts_str})")
        segments = []
        if mine:   segments.append(f"{mine} mine")
        if idle:   segments.append(f"{idle} idle")
        if others: segments.append(f"{others} others")
        if down:   segments.append(f"{down} down")
        lines.append(f"  {' | '.join(segments)}")

        avail_now = mine + idle
        if avail_now > 0:
            r_avail = f"{mine}/{avail_now} available now ({mine/avail_now*100:.0f}%)"
        else:
            r_avail = "0 available now"
        if accessible > 0:
            r_total = f"{mine}/{accessible} accessible ({mine/accessible*100:.0f}%)"
        else:
            r_total = "0 accessible"
        lines.append(f"  Using: {r_avail}  |  {r_total}")

        tier_parts = []
        for tier in ("primary", "group", "standby"):
            plist = tier_details.get(gtype, {}).get(tier, set())
            if plist:
                tier_parts.append(f"{','.join(sorted(plist))}({TIER_LABEL[tier]})")
        if tier_parts:
            lines.append(f"  Access: {'  '.join(tier_parts)}")

        if verbose and entry.get("nodes"):
            lines.append(f"  {'Node':<12} {'GPUs':>4} {'Mine':>5} {'Othr':>5} {'Idle':>5} {'Down':>5}  {'State':<8}  QOS")
            for nd in entry["nodes"]:
                st = nd["state"].upper()
                if "DOWN" in st:        ss = "down"
                elif "DRAIN" in st:     ss = "drain"
                elif "MAINT" in st:     ss = "maint"
                elif "NOT_RESP" in st:  ss = "no-resp"
                else:                   ss = nd["state"].split("+")[0].lower()[:8]
                qos_str = ",".join(nd.get("my_qos", []))
                lines.append(
                    f"  {nd['name']:<12} {nd['total']:>4} {nd['mine']:>5} "
                    f"{nd['others']:>5} {nd['idle']:>5} {nd['down']:>5}  {ss:<8}  {qos_str}"
                )

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Slurm GPU dashboard")
    parser.add_argument("-w", "--watch", nargs="?", const=30, type=int,
                        metavar="SEC", help="auto-refresh every SEC seconds (default 30)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show per-node breakdown")
    parser.add_argument("--plain", action="store_true",
                        help="plain text output (no colors/unicode), suitable for LLMs")
    args = parser.parse_args()

    user = get_user()

    if args.plain:
        data = gather_data(user)
        print(build_plain_text(user, data, verbose=args.verbose))
    elif args.watch:
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        try:
            while True:
                console.clear()
                render_frame(user, verbose=args.verbose)
                console.print(
                    f"  Refreshing in {args.watch}s … (Ctrl-C to quit)",
                    style="dim",
                )
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        render_frame(user, verbose=args.verbose)


if __name__ == "__main__":
    main()

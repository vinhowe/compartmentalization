"""Enumerate the run directories the paper's figures actually use.

WHY. Analysis scripts that glob (`benchtraj_*.json`, `out/*/*/`) silently pick up
whatever is on disk: abandoned sweeps, superseded configs, runs killed early.
A figure built that way can show models that appear nowhere else in the paper,
which is both a correctness problem and unanswerable if a reviewer asks what a
given point is. This computes the allowlist by working backwards from the paper:

    paper/*.tex  ->  \\includegraphics{figures/X.pdf}
                 ->  the experiment/*.py that writes X.pdf
                 ->  the run keys that script reads

and writes the union to paper_run_inventory.json.

TWO WAYS A GENERATOR NAMES ITS RUNS, and both have to be followed:
  1. Explicitly, via a symbol from _run_paths (or a literal "group/dirname").
     Resolved by importing _run_paths and flattening every dict/list it exports,
     then checking which symbols each generator actually mentions.
  2. Implicitly, by scanning a group directory and filtering on config fields.
     Those filters ARE the selection criteria and are replicated below. A run
     sitting in a scanned group is NOT automatically in the paper -- e.g. the
     phase-transition panels require absolute-mode, wd=0, tr<1.0 and a full 1e6
     steps, so tr=1.0 cells in the same directory are excluded.

Being in _run_paths is not sufficient either: some symbols there are unused by
any figure that survived into the paper (RUN_8_256_C3 is one).

Usage:  python3 build_paper_run_inventory.py [--verbose]
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_ROOT = REPO / "out" / "translation-compression"
PAPER = REPO / "paper"
INVENTORY = HERE / "paper_run_inventory.json"

# Group scans, with the filter each scanning generator applies. Keep in sync
# with the generators named in the comment; a mismatch here silently changes
# which models the derived figures are allowed to show.
SCANS = [
    # plot_translation_phase_transition.py (Fig 3a/3c), plot_translation_target_trajectory.py
    (["bpe16384-rope-wd-n2", "bpe16384-rope-wd-n3-n8", "bpe16384-rope-8-256"],
     lambda i: i["mode"] == "absolute" and i["wd"] == 0
     and i["tr_raw"] < 1.0 and i["last"] >= 1_000_000),
    # plot_wd_tr075_slice.py -- the weight-decay sweep keeps wd>0
    (["bpe16384-rope-wd-n2", "bpe16384-rope-wd-n3-n8", "bpe16384-rope-8-256"],
     lambda i: i["wd"] > 0 and i["last"] >= 1_000_000),
    # plot_1b_section.py (Fig 4a/4b) -- wd=0, minus the under-trained abs tr=0.1 cell
    (["1b-scale"],
     lambda i: i["wd"] == 0
     and not (i["mode"] == "absolute" and abs(i["tr_raw"] - 0.1) < 1e-6)),
]


def paper_figures():
    tex = "\n".join(p.read_text() for p in PAPER.glob("*.tex"))
    return sorted(set(re.findall(r"figures/([A-Za-z0-9_./-]+)\.pdf", tex)))


def generators(figs):
    scripts = sorted(HERE.glob("plot_*.py")) + sorted(HERE.glob("evaluate_*.py"))
    hits = {}
    for s in scripts:
        src = s.read_text()
        for f in figs:
            if f in src:
                hits.setdefault(s.name, set()).add(f)
    return hits


def flatten(v):
    if isinstance(v, str):
        return [v] if "/" in v else []
    if isinstance(v, dict):
        return [x for y in v.values() for x in flatten(y)]
    if isinstance(v, (list, tuple, set)):
        return [x for y in v for x in flatten(y)]
    return []


def explicit_keys(gen_names):
    sys.path.insert(0, str(HERE))
    rp = importlib.import_module("_run_paths")
    syms = {n: flatten(getattr(rp, n)) for n in dir(rp) if not n.startswith("__")}
    syms = {n: v for n, v in syms.items() if v}
    keys = set()
    for g in gen_names:
        src = (HERE / g).read_text()
        for n, v in syms.items():
            if re.search(rf"\b{re.escape(n)}\b", src):
                keys |= set(v)
        for m in re.findall(r'"([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)"', src):
            if not m.endswith((".pdf", ".png", ".json")) and not m.startswith(("figures/", "../", "./")):
                if (OUT_ROOT / m / "meta" / "config.json").exists():
                    keys.add(m)
    return keys


def scanned_keys(vm):
    keys = set()
    for groups, keep in SCANS:
        for g in groups:
            d0 = OUT_ROOT / g
            if not d0.is_dir():
                continue
            for d in sorted(d0.iterdir()):
                cf = d / "meta" / "config.json"
                if not cf.exists():
                    continue
                cfg = json.loads(cf.read_text())
                e, o = cfg["experiment"], cfg["optimizer"]
                key = f"{g}/{d.name}"
                v = vm.get(key)
                if not v or not v.get("checkpoints"):
                    continue
                if keep(dict(c=e["n_compartments"], tr_raw=e["translation_ratio"],
                             mode=e["translation_ratio_mode"],
                             wd=o.get("weight_decay", 0), last=v["checkpoints"][-1])):
                    keys.add(key)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    figs = paper_figures()
    gens = generators(figs)
    vm = json.loads((HERE / "val_metrics.json").read_text())
    exp = explicit_keys(gens)
    scn = scanned_keys(vm)
    keys = sorted(exp | scn)

    print(f"  {len(figs)} figures included by paper/*.tex")
    print(f"  {len(gens)} generator scripts produce them")
    print(f"  {len(exp)} run keys named explicitly, {len(scn)} matched by group scans")
    print(f"  -> {len(keys)} run keys in the paper inventory")
    if args.verbose:
        for f in figs:
            owners = [g for g, s in gens.items() if f in s]
            if not owners:
                print(f"    NO GENERATOR FOUND for {f}")
    INVENTORY.write_text(json.dumps(keys, indent=1))
    print(f"  wrote {INVENTORY.name}")


if __name__ == "__main__":
    main()

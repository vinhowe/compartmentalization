"""Run identifiers referenced by the paper plot/compute scripts.

These point at entries in the precomputed eval JSONs we ship via Zenodo
(`fineweb_val_metrics.json`, `cossim_sweep.json`, `cossim_across_training.json`,
`finetune_val_metrics.json`, `multilingual_*_per_lang.json`, plus the
training-time val-loss logs in `.multirun/*.log`).

Two kinds of identifier appear here:

  * Short cfghash keys — e.g. ``bpe16384-rope-8-256/217ca694_s64``. These are
    deterministic given the config TOML; re-running via ``sweep_runner.py``
    reproduces the exact same key.
  * Timestamped run dirs — e.g.
    ``1b-scale/2026-04-12T04-30-30Z__1b-8comp-bpe16384-correct__47875262__...``.
    These are unique to the original training run; a retrain will produce a
    different timestamp and trailing hash.

If you retrain and want the plot scripts to find your new runs, you have three
options:

  1. Use the precomputed JSONs shipped via Zenodo — the keys in this module
     match those JSONs, so nothing here needs to change. The plot scripts
     run as-is.
  2. Edit the constants below to point at your new run dirs.
  3. Use the ``find_latest_run`` helper to look up the latest match by
     substring within a group.
"""
from __future__ import annotations

from pathlib import Path

# repo-root-relative path resolution so scripts work from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = _REPO_ROOT / "out" / "translation-compression"
MULTIRUN_ROOT = _REPO_ROOT / ".multirun"


def find_latest_run(group: str, substr: str) -> str:
    """Return the lexicographically-latest run dir under
    ``out/translation-compression/{group}/`` whose name contains ``substr``.

    Returns the canonical ``"{group}/{dirname}"`` key.
    """
    g = OUT_ROOT / group
    if not g.exists():
        raise FileNotFoundError(g)
    candidates = sorted(d.name for d in g.iterdir() if d.is_dir() and substr in d.name)
    if not candidates:
        raise FileNotFoundError(f"no run matching {substr!r} in {group}/")
    return f"{group}/{candidates[-1]}"


def multirun_log(stem: str) -> str:
    """Path to a .multirun training-time val-loss log, e.g. ``multirun_log('2e75ffe5')``."""
    return str(MULTIRUN_ROOT / f"{stem}.log" if not stem.endswith(".log") else MULTIRUN_ROOT / stem)


# ───────────────────────────── Baselines ─────────────────────────────

# c=1 baseline at 8-256 — the "floor" used as reference everywhere.
C1_BASELINE_8_256 = (
    "synthetic-compartment-baselines/"
    "2026-03-06T18-11-45Z__english-baseline-rope-bpe16384-8-256__2df56182__s64__4b68526__51c738c2"
)

# c=1 baselines at smaller scales (used by Fig 1 plateau).
C1_BASELINE_BY_SCALE = {
    32:  "synthetic-compartment-baselines/2026-03-06T18-19-13Z__english-baseline-rope-bpe16384-8-32__41ba658f__s64__4b68526__66b66981",
    64:  "synthetic-compartment-baselines/2026-03-06T18-18-46Z__english-baseline-rope-bpe16384-8-64__50fa3055__s64__4b68526__007674d4",
    128: "synthetic-compartment-baselines/2026-03-06T18-17-16Z__english-baseline-rope-bpe16384-8-128__bafaffdf__s64__4b68526__e939a660",
    256: C1_BASELINE_8_256,
}

# 8-256 no-InfoNCE baselines by c (paper Fig 4, Fig 7, Fig 8).
# Short cfghash keys — deterministic via sweep_runner.
NO_INFONCE_8_256_BY_C = {
    2: "bpe16384-rope-8-256/217ca694_s64",
    4: "bpe16384-rope-8-256/53e73c3d_s64",
    5: "bpe16384-rope-8-256/918122e2_s64",
    6: "bpe16384-rope-8-256/b4d95a94_s64",
    8: "bpe16384-rope-8-256/868ef4a8_s64",
}

# A 1-comp variant at 8-256 used by plot_baseline_val_curves.
RUN_8_256_C1_EXTRA = "bpe16384-rope-8-256/c5ac7e54_s64"

# Older bpe16384-n3-rope baselines (used by plot_baseline_val_curves).
RUNS_N3_ROPE = [
    "bpe16384-n3-rope/4bb14425_s64",
    "bpe16384-n3-rope/9f60d15d_s64",
    "bpe16384-n3-rope/b7d883ff_s64",
]

# ──────────────────────── InfoNCE c-sweep (Fig 7) ────────────────────────

# 8-256 InfoNCE checkpoint dirs by c (used by compute_cossim_across_training).
INFONCE_8_256_BY_C = {
    2: "bpe16384-8-256-infonce/2026-05-02T18-30-47Z__8-256-n2-tr0-infonce__3d28d6ba__s64__fd9c538__e7c1f136",
    4: "bpe16384-8-256-infonce/2026-05-02T18-30-52Z__8-256-n4-tr0-infonce__27749b37__s64__fd9c538__3b53c824",
    5: "bpe16384-8-256-infonce/2026-05-06T15-33-54Z__8-256-n5-tr0-infonce__62841be1__s64__fd9c538__27aba970",
    6: "bpe16384-8-256-infonce/2026-05-05T00-40-58Z__8-256-n6-tr0-infonce__c722a6f7__s64__fd9c538__b4886e56",
    8: "bpe16384-8-256-infonce/2026-05-02T18-30-47Z__8-256-n8-tr0-infonce__1ed1536b__s64__fd9c538__4ff10c8e",
}

# .multirun training-time val_loss logs, by c (InfoNCE is tr=0 so train-time
# val is uncontaminated; we use it for the trajectory in Fig 7).
INFONCE_8_256_LOGS_BY_C = {
    2: [multirun_log("2e75ffe5"), multirun_log("823df7cf")],
    4: [multirun_log("ef17d9d3"), multirun_log("97e2bbd0")],
    5: [multirun_log("infonce-n5")],
    6: [multirun_log("infonce-n6")],
    8: [multirun_log("3842841b"), multirun_log("4cd59e61")],
}

# ────────────────── InfoNCE λ-sweep at c=2 (appendix) ──────────────────

INFONCE_8_256_C2_LOGS_BY_LAMBDA = {
    0.1:  [multirun_log("313a7eb6")],
    0.7:  [multirun_log("56c1aa3f")],
    1.0:  [multirun_log("2e75ffe5"), multirun_log("823df7cf")],
    1.3:  [multirun_log("39c9fc8c")],
    10.0: [multirun_log("379c91b0")],
}

# ────────────────── InfoNCE batch sweep at c=2, λ=1 (appendix) ──────────

INFONCE_8_256_C2_LOGS_BY_BATCH = {
    32:  [multirun_log("2e75ffe5"), multirun_log("823df7cf")],
    128: [multirun_log("infonce-n2-batch128")],
    512: [multirun_log("infonce-n2-batch512")],
}

# ───────────────────────────── 1B (Fig 6) ─────────────────────────────

# 1B baselines (compartment-mode legacy training).
RUN_1B_C1_BASELINE = (
    "1b-scale/2026-04-11T20-29-41Z__1b-1comp-baseline-bpe16384__9e005d30__s64__75a29e5__3acd587e"
)
RUN_1B_C2_NOTRANS = (
    "1b-scale/2026-04-14T20-06-42Z__1b-2comp-notrans-bpe16384-correct__836ab6ce__s64__75a29e5__55830146"
)
RUN_1B_C8_NOTRANS = (
    "1b-scale/2026-04-11T20-30-51Z__1b-8comp-notrans-bpe16384-correct__9c199809__s64__75a29e5__596929d1"
)
# 1B compartment-mode runs at tr_raw=0.5 → effective tr=0.5/(c+1).
RUN_1B_C2_TR0_167 = (  # c=2, tr_eff = 0.5/3 ≈ 0.167
    "1b-scale/2026-04-14T20-35-26Z__1b-2comp-bpe16384-correct__586efbbc__s64__75a29e5__6c0d1003"
)
RUN_1B_C8_TR0_056 = (  # c=8, tr_eff = 0.5/9 ≈ 0.056
    "1b-scale/2026-04-12T04-30-30Z__1b-8comp-bpe16384-correct__47875262__s64__75a29e5__c1a66f59"
)

# 1B c=8 absolute-mode runs by tr.
RUNS_1B_C8_ABS_BY_TR = {
    0.25: "1b-scale/2026-04-28T07-25-54Z__1b-8comp-tr025abs-bpe16384__7f828f8a__s64__75a29e5__10e98f4e",
    0.5:  "1b-scale/2026-04-27T23-16-17Z__1b-8comp-tr05abs-bpe16384__8c45bf62__s64__75a29e5__984e0ac2",
    0.75: "1b-scale/2026-04-27T23-16-44Z__1b-8comp-tr075abs-bpe16384__cf2767fd__s64__75a29e5__36396f38",
}

# ──────────────────── 8-512 c-sweep at tr=0.1 (Fig 6) ────────────────────

# Compartment-mode tr_raw=0.1 → effective tr=0.1/(c+1).
RUNS_8_512_LEGACY_BY_C = {
    1: "bpe16384-rope-8-512-sweep/2026-04-27T22-12-00Z__8-512-n1-tr0__6a459969__s64__fd9c538__cd128e95",
    2: "bpe16384-rope-8-512-sweep/2026-04-27T22-12-03Z__8-512-n2-tr01__1ac70722__s64__fd9c538__959443a0",
    3: "bpe16384-rope-8-512-sweep/2026-04-27T22-12-00Z__8-512-n3-tr01__de18a18b__s64__fd9c538__bdbec307",
    4: "bpe16384-rope-8-512-sweep/2026-04-27T22-12-00Z__8-512-n4-tr01__79908dbc__s64__fd9c538__c785cc59",
    5: "bpe16384-rope-8-512-sweep/2026-04-27T22-12-06Z__8-512-n5-tr01__cff4ea6b__s64__fd9c538__716a61dc",
    6: "bpe16384-rope-8-512-sweep/2026-04-27T22-12-01Z__8-512-n6-tr01__490675c3__s64__fd9c538__464afe7c",
    8: "bpe16384-rope-8-512-sweep/2026-04-27T22-12-00Z__8-512-n8-tr01__a7ddefbd__s64__fd9c538__198f3d6b",
}

# ───────── n=2 compartment-diversity runs (Fig 3 — "what fills c1?") ─────────
#
# All c=2, tr=0, mode=compartment, 8-256 rope. Compartment 0 is English; we
# vary what fills compartment 1.
N2_DIVERSITY_RUNS = {
    "EN-RU":      "russian-baselines-rope/2026-03-01T00-03-55Z__russian-english-baseline-rope-bpe16384-8-256__c7e8d8f0__s64__4b68526__79d396a8",
    "EN-unigram": "synthetic-compartment-baselines/2026-03-05T22-39-16Z__english-frequency-2comp-rope-bpe16384-8-256__605a1512__s64__4b68526__2acd312f",
    "EN-uniform": "synthetic-compartment-baselines/2026-03-05T22-39-06Z__english-uniform-2comp-rope-bpe16384-8-256__11b3d274__s64__4b68526__a6d73c34",
}

# ────────────────────────── copyemb (appendix) ──────────────────────────

# 8-256 copyemb runs (init from c=1 baseline embeddings, vs random init).
COPYEMB_8_256_BY_C = {
    2: "synthetic-compartment-baselines/2026-03-11T06-39-01Z__english-copyemb-2comp-rope-bpe16384-8-256__55197561__s64__4b68526__09338b7c",
    8: "synthetic-compartment-baselines/2026-03-11T23-33-54Z__english-copyemb-8comp-rope-bpe16384-8-256__28f1b316__s64__4b68526__8fa7f919",
}

# Scaling appendix: c=2 copyemb across 8-{32,64,128,256}.
COPYEMB_C2_BY_SCALE = {
    32:  "synthetic-compartment-baselines/2026-03-11T06-39-22Z__english-copyemb-2comp-rope-bpe16384-8-32__e362045d__s64__4b68526__55aa6f95",
    64:  "synthetic-compartment-baselines/2026-03-11T06-39-12Z__english-copyemb-2comp-rope-bpe16384-8-64__c148c701__s64__4b68526__675100fa",
    128: "synthetic-compartment-baselines/2026-03-11T06-39-05Z__english-copyemb-2comp-rope-bpe16384-8-128__2510e3fb__s64__4b68526__a830bae0",
    256: COPYEMB_8_256_BY_C[2],
}

# ──────────────── Small-scale c-sweep at tr=0.1 (Fig 1 plateau) ────────────

# (scale, c) → run dir. Used by plot_baseline_val_curves.
RUNS_SMALL_SCALE_TR01 = {
    (32, 2): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-14Z__8-32-n2-tr01__276e6011__s64__fd9c538__fc9992cd",
    (32, 4): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-15Z__8-32-n4-tr01__182e8587__s64__fd9c538__87cfdd84",
    (32, 5): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-15Z__8-32-n5-tr01__5fe56681__s64__fd9c538__64ba6d8f",
    (32, 6): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T21-46-13Z__8-32-n6-tr01__d3d70929__s64__fd9c538__6ff7bf7e",
    (32, 8): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-32-n8-tr01__3de8c1bc__s64__fd9c538__28614139",
    (64, 2): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-64-n2-tr01__a654d23e__s64__fd9c538__4bc16fd9",
    (64, 4): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-15Z__8-64-n4-tr01__1f914b4a__s64__fd9c538__2921ef4c",
    (64, 5): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-14Z__8-64-n5-tr01__81cf31a3__s64__fd9c538__faac50d4",
    (64, 6): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-14Z__8-64-n6-tr01__701205b7__s64__fd9c538__a00c89fe",
    (64, 8): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-64-n8-tr01__bb0b7e57__s64__fd9c538__ebbb76e1",
    (128, 2): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n2-tr01__29620da8__s64__fd9c538__e33f2900",
    (128, 4): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n4-tr01__fdd9ff02__s64__fd9c538__2eba5b2e",
    (128, 5): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n5-tr01__1ec60abf__s64__fd9c538__7a44f455",
    (128, 6): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n6-tr01__a6e55f01__s64__fd9c538__309edd86",
    (128, 8): "bpe16384-rope-small-scale-tr01-epoch/2026-04-24T00-42-11Z__8-128-n8-tr01__0caca982__s64__fd9c538__be47d71b",
}

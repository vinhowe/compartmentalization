#!/usr/bin/env python3
"""Verify the ladder-v2 configs resolve to exactly what the protocol says.

This is deliberately NOT "does it parse". Parsing is what failed to catch the
2026-08-21 ORC incident, where a config parsed fine and then silently took a
dataclass default for a field whose adapter had not been synced. This asserts
the RESOLVED values -- vocab arithmetic, derived max_iters, trunk parameter
count, batch geometry -- against docs/experimental-protocol.md.

Usage:  .venv/bin/python scripts/verify_ladder_v2_configs.py
"""

from __future__ import annotations

import glob
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config.manager import ConfigManager  # noqa: E402

BASE_VOCAB = 16384
TOKENS_PER_STEP = 2048 * 1024
WORLD_SIZE = 8

# rung -> (n_layer, n_embd, trunk N in millions) straight from the protocol table
PROTOCOL = {
    "r1": (4, 512, 12.6),
    "r2": (5, 640, 24.6),
    "r3": (6, 768, 42.5),
    "r4": (8, 1024, 100.7),
    "r5": (12, 1536, 339.7),
    "r6": (16, 2048, 805.3),
}


def trunk_params(n_layer: int, n_embd: int) -> int:
    return 12 * n_layer * n_embd * n_embd


def embed_params(n_embd: int, composite_vocab: int) -> int:
    return 2 * n_embd * composite_vocab


def main() -> int:
    files = sorted(glob.glob("config/ladder-v2/*.toml"))
    if not files:
        print("no configs found -- run scripts/gen_ladder_v2_configs.py first")
        return 1

    failures: list[str] = []
    rows: list[tuple] = []

    for f in files:
        stem = pathlib.Path(f).stem
        try:
            cfg = ConfigManager().load_from_toml_file(f)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{stem}: failed to load -- {type(e).__name__}: {e}")
            continue

        def bad(msg: str) -> None:
            failures.append(f"{stem}: {msg}")

        rung = stem.split("-")[2] if "lrsweep" not in stem else stem.split("-")[3]
        if rung not in PROTOCOL:
            bad(f"unrecognised rung {rung!r}")
            continue
        want_L, want_d, want_trunk_M = PROTOCOL[rung]

        m, t, e, lr = cfg.model, cfg.training, cfg.experiment, cfg.lr

        # --- geometry
        if (m.n_layer, m.n_embd) != (want_L, want_d):
            bad(f"geometry {m.n_layer}x{m.n_embd}, protocol says {want_L}x{want_d}")
        if m.n_embd // m.n_head != 64:
            bad(f"d_head {m.n_embd // m.n_head}, protocol fixes it at 64")
        if m.block_size != 1024:
            bad(f"block_size {m.block_size}, protocol says 1024")
        if m.weight_tying:
            bad("weight_tying is on; protocol keeps embeddings untied")

        # --- vocab arithmetic: composite = vocab_size * n_compartments + 1
        composite = m.vocab_size * e.n_compartments + 1
        arm = "c1-padded" if m.vocab_size != BASE_VOCAB else f"c{e.n_compartments}"
        if arm == "c1-padded":
            if composite != BASE_VOCAB * 8 + 1:
                bad(f"padded arm composite {composite}, want {BASE_VOCAB * 8 + 1}")
            if e.n_compartments != 1:
                bad(f"padded arm has n_compartments={e.n_compartments}, want 1")

        # --- budget is derived, not chosen
        implied = t.max_iters * TOKENS_PER_STEP
        if abs(implied - round(implied / 1e9) * 1e9) > TOKENS_PER_STEP:
            bad(f"max_iters {t.max_iters} -> {implied/1e9:.3f}B, not a round budget")

        # --- batch geometry
        if t.batch_size * t.gradient_accumulation_steps != 2048:
            bad(
                f"batch {t.batch_size}x{t.gradient_accumulation_steps} "
                f"= {t.batch_size * t.gradient_accumulation_steps}, want 2048 seq"
            )
        if t.gradient_accumulation_steps % WORLD_SIZE:
            bad(
                f"grad_accum {t.gradient_accumulation_steps} not divisible by "
                f"world size {WORLD_SIZE}"
            )
        if getattr(t, "auto_batch_config", True):
            bad("auto_batch_config is on; it would retune batch under the budget")

        # --- schedule
        if lr.schedule != "wsd":
            bad(f"schedule {lr.schedule!r}, protocol says wsd")
        if lr.warmup_iters != 300:
            bad(f"warmup {lr.warmup_iters}, protocol fixes it at 300 (absolute)")
        if lr.min_lr >= cfg.optimizer.learning_rate:
            bad(
                f"min_lr {lr.min_lr:g} >= peak {cfg.optimizer.learning_rate:g}; "
                "the decay would ramp UP"
            )

        # --- compartment assignment horizon
        if e.n_compartments > 1 and not getattr(e, "assignment_horizon_examples", 0):
            bad("c>1 without a pinned assignment_horizon_examples")

        trunk = trunk_params(m.n_layer, m.n_embd)
        if abs(trunk / 1e6 - want_trunk_M) > 0.15:
            bad(f"trunk N {trunk/1e6:.1f}M, protocol table says {want_trunk_M}M")

        total = trunk + embed_params(m.n_embd, composite)
        rows.append(
            (
                stem,
                f"{m.n_layer}x{m.n_embd}",
                arm,
                composite,
                t.max_iters,
                f"{implied/1e9:.0f}B",
                f"{trunk/1e6:.1f}M",
                f"{total/1e6:.1f}M",
                f"{100*embed_params(m.n_embd, composite)/total:.0f}%",
                f"{cfg.optimizer.learning_rate:g}",
            )
        )

    hdr = ("config", "geom", "arm", "vocab", "iters", "budget",
           "trunkN", "totalN", "emb%", "lr")
    w = [max(len(str(r[i])) for r in ([hdr] + rows)) for i in range(len(hdr))]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(len(hdr))))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))

    print()
    if failures:
        print(f"{len(failures)} PROBLEM(S):")
        for f in failures:
            print("  " + f)
        return 1
    print(f"all {len(rows)} configs resolve exactly as the protocol specifies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

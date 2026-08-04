"""Measure, per benchmark task, how much of it actually fits block_size=64.

Written to settle a discrepancy: eval_downstream_benchmarks.py's docstring says
"hellaswag 94% of examples fit", while the frac_truncated it records across 920
benchtraj entries is 69.4% -- i.e. only ~31% fit. sciq/piqa/arc_easy/lambada
roughly agree with the docstring; hellaswag alone is far out.

Three different quantities get conflated when people say "fits", and this prints
all three so the docstring can be pinned to one:

  ctx-only   len(ctx) <= T                  ignores the continuation entirely
  per-pair   len(ctx + choice) <= T         what frac_truncated actually counts
  per-record all of a record's pairs fit    the strictest, and <= per-pair

Read-only: imports the task builders and tokenizer from
eval_downstream_benchmarks and touches no eval semantics or result file.

Usage: python3 measure_task_truncation.py [--tasks ...] [--limit N] [-T 64]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_downstream_benchmarks as E
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent


def measure(tok, records, T):
    pair_fit = []          # per (record, choice)
    rec_all_fit = []       # per record: every choice fits
    ctx_fit = []           # per (record, choice): context alone fits
    pair_lens, ctx_lens = [], []
    for r in records:
        pair = r.get("pair_ctx")
        ctx_ids = tok.encode(r["ctx"]).ids if r.get("ctx") is not None else None
        fits_here = []
        for ci, ch in enumerate(r["choices"]):
            c_ids = tok.encode(pair[ci]).ids if pair is not None else ctx_ids
            ch_ids = tok.encode(ch).ids or [tok.encode(" ").ids[0]]
            n = len(c_ids) + len(ch_ids)
            pair_lens.append(n)
            ctx_lens.append(len(c_ids))
            ok = n <= T
            pair_fit.append(ok)
            ctx_fit.append(len(c_ids) <= T)
            fits_here.append(ok)
        rec_all_fit.append(all(fits_here))
    return dict(
        n_records=len(records), n_pairs=len(pair_fit),
        ctx_only_fit=100 * float(np.mean(ctx_fit)),
        per_pair_fit=100 * float(np.mean(pair_fit)),
        per_record_fit=100 * float(np.mean(rec_all_fit)),
        per_pair_truncated=100 * (1 - float(np.mean(pair_fit))),
        median_pair_len=float(np.median(pair_lens)),
        p90_pair_len=float(np.percentile(pair_lens, 90)),
        median_ctx_len=float(np.median(ctx_lens)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["hellaswag", "arc_easy", "piqa", "sciq", "lambada",
                             "winogrande", "triviaqa"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-T", type=int, default=64)
    ap.add_argument("--out", default="task_truncation_report.json")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(E.TOKENIZER_PATH))
    print(f"  T={args.T}, tokenizer={E.TOKENIZER_PATH.name} "
          f"(vocab {tok.get_vocab_size()})\n")
    hdr = (f"  {'task':<11}{'recs':>6}{'pairs':>7}"
           f"{'ctx-only':>10}{'per-pair':>10}{'per-rec':>9}"
           f"{'med len':>9}{'p90':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    out = {}
    for t in args.tasks:
        try:
            recs = E.load_task(t, args.limit)
        except Exception as e:
            print(f"  {t:<11} FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        m = measure(tok, recs, args.T)
        out[t] = m
        print(f"  {t:<11}{m['n_records']:>6}{m['n_pairs']:>7}"
              f"{m['ctx_only_fit']:>9.1f}%{m['per_pair_fit']:>9.1f}%"
              f"{m['per_record_fit']:>8.1f}%"
              f"{m['median_pair_len']:>9.0f}{m['p90_pair_len']:>7.0f}")
    (HERE / args.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

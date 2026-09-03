#!/usr/bin/env python3
"""Per-population per-format extraction eval for bio-compartmentation.

For each population (BRIDGE / DECL-only / QA-only / NEVER-SEEN), probe each
person's 6 attributes via:
  - DECL continuation: "Alexandria Evan Martin was born on" → match birth date
  - QA prompt:         "Q: When was Alexandria Evan Martin born? A:" → match birth date

Reports per-attribute accuracy per (population, prompt-format) cell.

Usage:
    .venv/bin/python3 scripts/eval_bio_compartmentation.py \\
        --checkpoint <path/to/checkpoints/_rolling> \\
        --run_dir   <path/to/run dir> \\
        --people    data/bio-comp-N50000-bridge5-seed42/people.json \\
        [--n_eval 200]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from experiment.eval_utils import load_eval_model_from_checkpoint  # noqa: E402


# Continuation-style probes for DECL prompts (one per attribute).
# Pick prompts that the model has been trained on (either via DECL bios with
# fullname=True, or via QA-only people's QA pairs).
DECL_PROBES = {
    "birth_date": {
        "template": " {name} was born on",
        "answer": lambda p: f" {p['birthmonth']} {p['birthday']}, {p['birthyear']}.",
    },
    "birth_city": {
        "template": " {name} was born in",
        "answer": lambda p: f" {p['birthcity']}.",
    },
    "university": {
        "template": " {name} graduated from",
        "answer": lambda p: f" {p['university']}.",
    },
    "major": {
        "template": " {name} studied",
        "answer": lambda p: f" {p['field']}.",
    },
    "employer": {
        "template": " {name} worked at",
        "answer": lambda p: f" {p['company1name']}.",
    },
    "work_city": {
        "template": " {name} worked in",
        "answer": lambda p: f" {p['company1city']}.",
    },
}


# Question-style probes (Q: ... A:). Should match templates the model saw
# in QA training (from generate_bio_dataset.QA_TEMPLATES).
QA_PROBES = {
    "birth_date": {
        "template": " Q: When was {name} born? A:",
        "answer": lambda p: f" {p['birthmonth']} {p['birthday']}, {p['birthyear']}.",
    },
    "birth_city": {
        "template": " Q: Where was {name} born? A:",
        "answer": lambda p: f" {p['birthcity']}.",
    },
    "university": {
        "template": " Q: Where did {name} study? A:",
        "answer": lambda p: f" {p['university']}.",
    },
    "major": {
        "template": " Q: What did {name} study? A:",
        "answer": lambda p: f" {p['field']}.",
    },
    "employer": {
        "template": " Q: Where did {name} work? A:",
        "answer": lambda p: f" {p['company1name']}.",
    },
    "work_city": {
        "template": " Q: In which city did {name} work? A:",
        "answer": lambda p: f" {p['company1city']}.",
    },
}


@torch.no_grad()
def batched_greedy_extract(
    model,
    prompt_ids: list[list[int]],
    max_new: int,
    device: str,
    batch_size: int = 64,
) -> list[list[int]]:
    """Greedy-generate `max_new` tokens for each pre-tokenized prompt.

    Groups prompts of identical length together so we can stack them into
    a (B, L) tensor without padding (the model has no attention-mask hook
    and uses RoPE positions tied to absolute index, so left-padding would
    be incorrect). Within each length-bucket we batch up to `batch_size`
    prompts and decode all generated ids on GPU until the loop finishes,
    syncing to CPU only once at the end.
    """
    out: list[list[int] | None] = [None] * len(prompt_ids)
    by_len: dict[int, list[int]] = {}
    for i, ids in enumerate(prompt_ids):
        by_len.setdefault(len(ids), []).append(i)

    for L, indices in by_len.items():
        for chunk_start in range(0, len(indices), batch_size):
            chunk = indices[chunk_start : chunk_start + batch_size]
            ids_list = [prompt_ids[i] for i in chunk]
            ids = torch.tensor(ids_list, dtype=torch.long, device=device)  # (B, L)
            B = ids.shape[0]
            generated = torch.empty((B, max_new), dtype=torch.long, device=device)
            for step in range(max_new):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, _ = model(ids)
                nxt = logits[:, -1].argmax(dim=-1)  # (B,)
                generated[:, step] = nxt
                ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            generated_cpu = generated.cpu().tolist()
            for j, idx in enumerate(chunk):
                out[idx] = generated_cpu[j]
    # All slots filled
    return out  # type: ignore[return-value]


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace for substring comparison."""
    return " ".join(s.lower().split())


def matches_substring(generated_text: str, answer: str) -> bool:
    """Return True if the gold answer (sans trailing period) appears anywhere
    in the generated continuation (case-insensitive, whitespace-tolerant)."""
    needle = _normalize(answer.rstrip(". "))
    haystack = _normalize(generated_text)
    return bool(needle) and needle in haystack


def evaluate(model, enc, people, probes, device, max_new=24, verbose=False,
             batch_size=64, token_offset=0, vocab_size=None):
    """For each (person, attr) pair, batch-generate up to `max_new` tokens from
    the prompt, decode, and check if the gold answer string appears in the
    continuation. Tolerates both bare-answer ("April 8, 1935") and
    full-restatement ("X was born on April 8, 1935.") variants.

    `token_offset`: if non-zero, all prompt content tokens get shifted by this
    amount before forwarding (used for compartmented models where this probe's
    format lives at offset+canonical token ids). Generated tokens are shifted
    back to canonical [0, vocab_size_canonical) for decoding via the tiktoken
    encoder.

    Returns dict[attr] -> [correct, total].
    """
    # Build full work list across all (person, attr) pairs for this probe set
    prompts: list[list[int]] = []
    answers: list[str] = []
    attrs: list[str] = []
    names: list[str] = []
    for person in people:
        name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
        for attr, info in probes.items():
            tok_ids = enc.encode(info["template"].format(name=name), disallowed_special=())
            if token_offset:
                tok_ids = [t + token_offset for t in tok_ids]
            prompts.append(tok_ids)
            answers.append(info["answer"](person))
            attrs.append(attr)
            names.append(name)
    # Single batched decode pass for all (person, attr) prompts
    gen_ids_all = batched_greedy_extract(model, prompts, max_new=max_new, device=device, batch_size=batch_size)

    out = {a: [0, 0] for a in probes}
    canonical_vocab = enc.n_vocab
    for attr, name, gen_ids, answer in zip(attrs, names, gen_ids_all, answers):
        # Filter generated ids to what tiktoken can decode.
        # - Always keep canonical-region tokens [0, V).
        # - If token_offset > 0 (compartmented eval), also keep offset-region
        #   tokens [offset, offset+V) and de-offset them.
        # - Drop any other token (model leaked across compartments / out of
        #   range for the chosen probe format).
        cleaned = []
        for t in gen_ids:
            if 0 <= t < canonical_vocab:
                cleaned.append(t)
            elif token_offset and token_offset <= t < token_offset + canonical_vocab:
                cleaned.append(t - token_offset)
            # else: drop
        gen_text = enc.decode(cleaned)
        ok = matches_substring(gen_text, answer)
        out[attr][0] += int(ok)
        out[attr][1] += 1
        if verbose and not ok:
            print(f"  MISS [{attr}] {name}: expected '{answer.strip()}', got '{gen_text[:120]}'")
    return out


def summarize(out: dict[str, list[int]]) -> dict:
    summary = {}
    total_correct = 0
    total_count = 0
    for attr, (c, n) in out.items():
        acc = c / max(1, n)
        summary[attr] = {"correct": c, "total": n, "acc": acc}
        total_correct += c
        total_count += n
    summary["overall"] = {
        "correct": total_correct,
        "total": total_count,
        "acc": total_correct / max(1, total_count),
    }
    return summary


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, checkpoint_dir)] for named step-NNNNNN + rolling, deduped by step."""
    out: list[tuple[int, Path]] = []
    ck_root = run_dir / "checkpoints"
    if not ck_root.exists():
        return out
    for d in ck_root.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("step-"):
            try:
                out.append((int(d.name.split("-")[1]), d))
            except ValueError:
                pass
        elif d.name == "_rolling":
            ts = d / "trainer_state.json"
            if ts.exists():
                try:
                    state = json.loads(ts.read_text())
                    out.append((int(state.get("iter_num", 0)), d))
                except Exception:
                    pass
    out.sort()
    seen, deduped = set(), []
    for step, d in out:
        if step in seen:
            continue
        seen.add(step)
        deduped.append((step, d))
    return deduped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Single checkpoint dir. Mutually exclusive with --all_checkpoints.")
    ap.add_argument("--all_checkpoints", action="store_true",
                    help="Walk all named + rolling checkpoints in --run_dir/checkpoints/.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--people", required=True)
    ap.add_argument("--n_eval", type=int, default=200, help="people per population to probe")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="batch size for greedy generation (within a length bucket)")
    ap.add_argument("--qa_token_offset", type=int, default=0,
                    help="offset to apply to QA-probe tokens (for compartmented models). "
                         "DECL probe stays at canonical 0-V.")
    args = ap.parse_args()
    if not args.all_checkpoints and not args.checkpoint:
        ap.error("Either --checkpoint or --all_checkpoints required")

    print(f"Loading tokenizer (gpt2)")
    enc = tiktoken.get_encoding("gpt2")

    with open(args.people) as f:
        people_all = json.load(f)
    pops = {"bridge": [], "decl_only": [], "qa_only": [], "never_seen": []}
    for p in people_all:
        pop = p.get("population")
        if pop in pops:
            pops[pop].append(p)
    print(f"Population sizes: " + ", ".join(f"{k}={len(v):,}" for k, v in pops.items()))

    rng = random.Random(args.seed)
    eval_pops = {}
    for k, v in pops.items():
        if not v:
            continue
        sample = list(v)
        rng.shuffle(sample)
        eval_pops[k] = sample[: min(args.n_eval, len(sample))]
        print(f"  evaluating {len(eval_pops[k])} from {k}")

    if args.all_checkpoints:
        ckpts = list_checkpoints(Path(args.run_dir))
    else:
        # Try to determine the step from the checkpoint dir name
        ck = Path(args.checkpoint)
        step = 0
        if ck.name.startswith("step-"):
            try:
                step = int(ck.name.split("-")[1])
            except ValueError:
                pass
        elif ck.name == "_rolling":
            ts = ck / "trainer_state.json"
            if ts.exists():
                try:
                    state = json.loads(ts.read_text())
                    step = int(state.get("iter_num", 0))
                except Exception:
                    pass
        ckpts = [(step, ck)]

    print(f"\nWill evaluate {len(ckpts)} checkpoint(s): {[s for s, _ in ckpts]}")

    all_results: dict = {}  # step -> {pop: {fmt: summary}}
    for step, ckpt_dir in ckpts:
        print(f"\n{'#' * 60}\n# Loading model at step {step} from {ckpt_dir}\n{'#' * 60}")
        try:
            model, cfg, _ = load_eval_model_from_checkpoint(
                ckpt_dir, Path(args.run_dir), args.device, dtype=torch.bfloat16
            )
        except Exception as e:
            print(f"  load failed: {e}")
            continue

        results: dict = {}
        for pop, sample in eval_pops.items():
            print(f"\n  === {pop} ({len(sample)} people) ===")
            for fmt_label, probes in [("decl_continuation", DECL_PROBES), ("qa_prompt", QA_PROBES)]:
                offset = args.qa_token_offset if fmt_label == "qa_prompt" else 0
                out = evaluate(model, enc, sample, probes, args.device,
                               verbose=args.verbose, batch_size=args.batch_size,
                               token_offset=offset)
                summary = summarize(out)
                results.setdefault(pop, {})[fmt_label] = summary
                print(f"    {fmt_label:>20s}: overall {summary['overall']['acc']:.1%} "
                      f"({summary['overall']['correct']}/{summary['overall']['total']})")
        all_results[str(step)] = results

        print(f"\n  ----- step {step} summary -----")
        print(f"  {'population':>12s}  {'decl_continuation':>18s}  {'qa_prompt':>10s}")
        for pop in ("bridge", "decl_only", "qa_only", "never_seen"):
            if pop not in results:
                continue
            d = results[pop]["decl_continuation"]["overall"]["acc"]
            q = results[pop]["qa_prompt"]["overall"]["acc"]
            print(f"  {pop:>12s}  {d:>18.1%}  {q:>10.1%}")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("FULL CURVE (overall accuracy by step / population / format)")
    print("=" * 70)
    print(f"  {'step':>6s}  {'pop':>12s}  {'decl_continuation':>18s}  {'qa_prompt':>10s}")
    for step in sorted(all_results.keys(), key=int):
        for pop in ("bridge", "decl_only", "qa_only", "never_seen"):
            if pop not in all_results[step]:
                continue
            d = all_results[step][pop]["decl_continuation"]["overall"]["acc"]
            q = all_results[step][pop]["qa_prompt"]["overall"]["acc"]
            print(f"  {step:>6s}  {pop:>12s}  {d:>18.1%}  {q:>10.1%}")

    if args.output is None:
        args.output = os.path.join(args.run_dir, "bio_extraction_eval.json")
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

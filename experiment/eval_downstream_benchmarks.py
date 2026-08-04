"""Downstream zero-shot benchmarks for compartmented models.

Tasks: hellaswag, arc_easy, piqa, sciq, lambada.

Scoring follows the lm-evaluation-harness conventions:
  * multiple choice -> score each candidate continuation by summed log-prob of
    its tokens given the context; `acc` picks the raw-sum argmax, `acc_norm`
    normalizes by the continuation's byte length.
  * lambada -> exact-match on greedy argmax of the final-word tokens
    (teacher-forced: each gold token scored given the gold prefix).

CONTEXT CAVEAT. These models train at block_size=64. Sequences longer than that
are left-truncated (the tail of the context is kept). Measured at BPE-16384 over
the full splits (measure_task_truncation.py), as the fraction of (record,
choice) pairs whose context+continuation fits in 64 tokens -- which is what
`frac_truncated` below counts:

    sciq 99.9% fit    arc_easy 92.3%    piqa 86.1%
    hellaswag 30.6% fit  <-- median pair is 102 tokens
    lambada    2.9% fit  <-- effectively all truncated

An earlier version of this note claimed hellaswag was 94%, which was wrong by a
wide margin and disagreed with the frac_truncated this file has been recording
all along (69.4% truncated = 30.6% fit). The other four figures were about
right. Hellaswag's continuations are long, so it is far more truncated than its
context length alone suggests: 56.0% of hellaswag contexts fit on their own, but
only 30.6% still fit once the continuation being scored is appended.

So absolute lambada AND hellaswag numbers are not comparable to published
values; they are retained only because every model here is truncated
identically, which keeps the *relative* c/tr comparison meaningful.
`frac_truncated` is reported per task so this stays visible in the output.

Text is fed in a single compartment (default 0), matching the
`loss_compartment_0` convention used elsewhere: composite id = base + cid*V.

Usage:
    python3 eval_downstream_benchmarks.py --runs <key> --step 1000000 \
        --tasks hellaswag arc_easy piqa sciq lambada --out bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_utils import load_eval_model_from_checkpoint  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "out" / "translation-compression"
TOKENIZER_PATH = REPO / "tokenizers" / "bpe-16384" / "tokenizer.json"

EVAL_SEED = 1024


# ───────────────────────────── task loading ─────────────────────────────
# Each task yields records: {"ctx": str, "choices": [str], "gold": int}
# For lambada, "choices" has a single entry (the gold final word).

def load_task(name: str, limit: int | None):
    from datasets import load_dataset

    if name == "hellaswag":
        d = load_dataset("Rowan/hellaswag", split="validation")
        if limit:
            d = d.select(range(min(limit, len(d))))
        return [{"ctx": x["ctx"],
                 "choices": [" " + e for e in x["endings"]],
                 "gold": int(x["label"])} for x in d]

    if name == "arc_easy":
        d = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        if limit:
            d = d.select(range(min(limit, len(d))))
        out = []
        for x in d:
            labels = x["choices"]["label"]
            if x["answerKey"] not in labels:
                continue
            out.append({"ctx": "Question: " + x["question"] + "\nAnswer:",
                        "choices": [" " + t for t in x["choices"]["text"]],
                        "gold": labels.index(x["answerKey"])})
        return out

    if name == "piqa":
        d = load_dataset("baber/piqa", split="validation")
        if limit:
            d = d.select(range(min(limit, len(d))))
        return [{"ctx": "Question: " + x["goal"] + "\nAnswer:",
                 "choices": [" " + x["sol1"], " " + x["sol2"]],
                 "gold": int(x["label"])} for x in d]

    if name == "sciq":
        d = load_dataset("allenai/sciq", split="test")
        if limit:
            d = d.select(range(min(limit, len(d))))
        rng = np.random.default_rng(EVAL_SEED)
        out = []
        for x in d:
            distractors = [x["distractor1"], x["distractor2"], x["distractor3"]]
            choices = distractors + [x["correct_answer"]]
            perm = rng.permutation(4)
            shuffled = [choices[i] for i in perm]
            out.append({"ctx": "Question: " + x["question"] + "\nAnswer:",
                        "choices": [" " + c for c in shuffled],
                        "gold": int(np.where(perm == 3)[0][0])})
        return out

    if name == "lambada":
        d = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        if limit:
            d = d.select(range(min(limit, len(d))))
        out = []
        for t in d["text"]:
            words = t.split()
            out.append({"ctx": " ".join(words[:-1]),
                        "choices": [" " + words[-1]],
                        "gold": 0})
        return out


    if name == "winogrande":
        d = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        if limit:
            d = d.select(range(min(limit, len(d))))
        out = []
        for x in d:
            # lm-eval convention: substitute each option into the blank, then
            # score the SHARED continuation that follows it. Discriminating on
            # the suffix is what makes the two candidates comparable.
            pre, post = x["sentence"].split("_", 1)
            gold = int(x["answer"]) - 1          # answer is "1"/"2"
            out.append({"ctx": None,
                        "pair_ctx": [pre + x["option1"], pre + x["option2"]],
                        "choices": [post, post],
                        "gold": gold})
        return out

    if name == "triviaqa":
        d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        if limit:
            d = d.select(range(min(limit, len(d))))
        return [{"ctx": "Question: " + x["question"] + "\nAnswer:",
                 "choices": [" " + x["answer"]["value"]],
                 "gold": 0} for x in d]

    raise ValueError(f"unknown task {name}")


# ───────────────────────────── scoring ─────────────────────────────

@torch.no_grad()
def score_batch(model, seqs, cont_lens, *, T, cid, device, pad_id):
    """Summed logprob of the last `cont_len` tokens of each sequence.

    seqs: list of 1-D LongTensors already left-truncated to <= T.
    Returns (logprob_sums, greedy_match_flags).
    """
    B = len(seqs)
    maxlen = max(int(s.numel()) for s in seqs)
    x = torch.full((B, maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((B, maxlen), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = int(s.numel())
        x[i, :n] = s
        mask[i, :n] = True
    x = x.to(device)
    cids = torch.full_like(x, cid)

    logits, _ = model(x, compartment_ids=cids, full_sequence_logits=True)
    logprobs = F.log_softmax(logits.float(), dim=-1)

    sums = torch.zeros(B, device=device)
    greedy_ok = torch.zeros(B, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        n = int(s.numel())
        k = cont_lens[i]
        if k <= 0 or k >= n:
            continue
        # positions n-k-1 .. n-2 predict tokens at n-k .. n-1
        pos = torch.arange(n - k - 1, n - 1, device=device)
        tgt = x[i, n - k:n]
        lp = logprobs[i, pos, :].gather(-1, tgt[:, None]).squeeze(-1)
        sums[i] = lp.sum()
        greedy_ok[i] = bool((logprobs[i, pos, :].argmax(-1) == tgt).all())
    return sums.cpu(), greedy_ok.cpu()


def eval_task(model, tok, records, *, T, cid, base_vocab, device, batch_size):
    """Returns metrics dict for one task."""
    # Flatten (record, choice) pairs, tracking truncation.
    flat, owner, cont_lens, byte_lens = [], [], [], []
    n_trunc = 0
    for ri, r in enumerate(records):
        pair = r.get("pair_ctx")
        ctx_ids = tok.encode(r["ctx"]).ids if r.get("ctx") is not None else None
        for ci, ch in enumerate(r["choices"]):
            if pair is not None:
                ctx_ids = tok.encode(pair[ci]).ids
            ch_ids = tok.encode(ch).ids
            if not ch_ids:
                ch_ids = [tok.encode(" ").ids[0]]
            seq = ctx_ids + ch_ids
            if len(seq) > T:
                n_trunc += 1
                seq = seq[-T:]              # keep the tail of the context
            # continuation must leave >=1 context token to condition on
            k = min(len(ch_ids), len(seq) - 1)
            flat.append(torch.tensor([t + cid * base_vocab for t in seq],
                                     dtype=torch.long))
            owner.append(ri)
            cont_lens.append(k)
            byte_lens.append(max(1, len(ch.encode("utf-8"))))

    all_sums = np.zeros(len(flat), dtype=np.float64)
    all_greedy = np.zeros(len(flat), dtype=bool)
    order = np.argsort([-int(s.numel()) for s in flat])  # length-bucket batches
    pad_id = cid * base_vocab
    for b0 in range(0, len(order), batch_size):
        idxs = order[b0:b0 + batch_size]
        sums, gok = score_batch(model, [flat[i] for i in idxs],
                                [cont_lens[i] for i in idxs],
                                T=T, cid=cid, device=device, pad_id=pad_id)
        all_sums[idxs] = sums.numpy()
        all_greedy[idxs] = gok.numpy()

    # Regroup by record.
    n_rec = len(records)
    n_correct = n_correct_norm = 0
    n_greedy = 0
    by_rec: dict[int, list[int]] = {}
    for i, ri in enumerate(owner):
        by_rec.setdefault(ri, []).append(i)
    for ri, idxs in by_rec.items():
        gold = records[ri]["gold"]
        sums = np.array([all_sums[i] for i in idxs])
        norms = np.array([all_sums[i] / byte_lens[i] for i in idxs])
        if int(np.argmax(sums)) == gold:
            n_correct += 1
        if int(np.argmax(norms)) == gold:
            n_correct_norm += 1
        if all_greedy[idxs[gold]]:
            n_greedy += 1

    per_item = np.zeros(n_rec, dtype=np.int8)
    for ri, idxs in by_rec.items():
        gold = records[ri]["gold"]
        norms = np.array([all_sums[i] / byte_lens[i] for i in idxs])
        if len(idxs) == 1:
            per_item[ri] = int(all_greedy[idxs[0]])
        else:
            per_item[ri] = int(int(np.argmax(norms)) == gold)

    single_choice = all(len(r["choices"]) == 1 for r in records)
    res = {
        "n": n_rec,
        "greedy_exact_match": n_greedy / max(1, n_rec),
        "frac_truncated": n_trunc / max(1, len(flat)),
    }
    if single_choice:
        # No distractors to rank against (lambada): argmax over one candidate is
        # trivially "correct", so accuracy IS the greedy exact match.
        res["acc"] = res["acc_norm"] = res["greedy_exact_match"]
    else:
        res["acc"] = n_correct / max(1, n_rec)
        res["acc_norm"] = n_correct_norm / max(1, n_rec)
    res["per_item"] = per_item.tolist()
    return res


def eval_run(run_key, step, *, tasks, limit, batch_size, cid, device, max_ctx=None):
    exp_dir = OUT_ROOT / run_key
    matches = [p for p in exp_dir.glob("checkpoints/step-*")
               if int(p.name.split("-")[-1]) == step]
    if not matches:
        return {"run": run_key, "step": step, "error": "checkpoint missing"}
    model, config, model_comps = load_eval_model_from_checkpoint(
        matches[0], exp_dir, device, dtype=torch.bfloat16)
    model.eval()

    exp = config.experiment
    if exp.permute_tokens_per_compartment:
        return {"run": run_key, "error": "permute-mode unsupported"}

    base_vocab = int(config.model.vocab_size)
    T = int(config.model.block_size)

    # Optionally run past the training length. RoPE degrades gracefully to about
    # +25% (measured: ppl 19.3 @63 -> 19.5 @80 -> 21.6 @96) and then collapses
    # (984 @128), so only modest overrides are defensible. Lets LAMBADA -- whose
    # median need is 81 tokens -- mostly avoid truncation.
    if max_ctx and max_ctx > T:
        if not config.model.use_rope:
            raise ValueError("max_ctx override requires a RoPE model")
        object.__setattr__(model.config, "block_size", max_ctx)
        T = max_ctx

    tr = float(exp.translation_ratio)
    mode = exp.translation_ratio_mode
    n_comp = int(exp.n_compartments)
    eff_tr = tr / (n_comp + 1) if mode == "compartment" else tr

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    out = {}
    for t in tasks:
        recs = load_task(t, limit)
        out[t] = eval_task(model, tok, recs, T=T, cid=cid,
                           base_vocab=base_vocab, device=device,
                           batch_size=batch_size)
        print(f"    {t:<10} acc={out[t]['acc']:.4f} acc_norm={out[t]['acc_norm']:.4f} "
              f"trunc={out[t]['frac_truncated']:.2f}", flush=True)

    del model
    torch.cuda.empty_cache()
    return {"run": run_key, "step": step, "n_compartments": n_comp,
            "translation_ratio": eff_tr, "translation_ratio_mode": mode,
            "block_size": T, "compartment": cid, "tasks": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--all-steps", action="store_true",
                    help="evaluate every saved checkpoint of each run")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="explicit checkpoint steps (aligns with val_metrics)")
    ap.add_argument("--tasks", nargs="+",
                    default=["hellaswag", "arc_easy", "piqa", "sciq", "lambada"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples per task (None = full split)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--compartment", type=int, default=0)
    ap.add_argument("--max-ctx", type=int, default=None,
                    help="run past training block_size (RoPE only); 96 is the "
                         "largest value measured as still usable")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="downstream_benchmarks.json")
    args = ap.parse_args()

    outp = Path(__file__).resolve().parent / args.out
    results = []
    for rk in args.runs:
        have = {int(p.name.split("-")[-1])
                for p in (OUT_ROOT / rk).glob("checkpoints/step-*")}
        if args.steps:
            steps = [s for s in args.steps if s in have]
        elif args.all_steps:
            steps = sorted(have)
        else:
            steps = [args.step]
        for st in steps:
            print(f"[bench] {rk} @ {st}", flush=True)
            try:
                r = eval_run(rk, st, tasks=args.tasks, limit=args.limit,
                             batch_size=args.batch_size, cid=args.compartment,
                             device=args.device, max_ctx=args.max_ctx)
            except Exception as e:
                r = {"run": rk, "step": st, "error": f"{type(e).__name__}: {e}"}
                print(f"  ERROR {r['error']}", flush=True)
            results.append(r)
            outp.write_text(json.dumps(results, indent=2))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()

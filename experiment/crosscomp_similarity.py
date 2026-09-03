"""Cross-compartment representational similarity, layerwise, with controls.

Measures whether a c>1 model represents the same text similarly across its
compartments -- and, more importantly, whether any such similarity exceeds what
you would get for uninteresting reasons.

WHAT THIS DOES NOT ESTABLISH. Similarity of activations is not evidence of
shared computation. Two networks can produce highly similar representations and
compute different functions; that is the standard critique of CKA (Ding, Denain
& Steinhardt 2021). Nothing here is causal. Read the output as "how aligned are
the representations", never as "the compartments share a mechanism".

THREE WAYS THIS MEASUREMENT LIES, AND WHAT IS DONE ABOUT THEM

1. Residual streams are not centred. Transformer hidden states carry a large
   common component plus a few very-high-magnitude "rogue" dimensions, so RAW
   cosine between almost any two token representations is high -- often >0.9 --
   regardless of content. Reporting raw cosine alone would manufacture a result.
   Both raw and mean-centred cosine are reported; the centred one is the number
   to read, and their difference tells you how much of the raw figure was the
   shared offset.

2. Position 0 is an attention sink. Its representation is atypical and
   high-norm, and it is identical across compartments in everything except the
   token id. Early positions are dropped (SKIP_POSITIONS).

3. "High" means nothing without a floor and a null. Both are computed here:
     FLOOR  an untrained model of the same architecture. Compartment embeddings
            are independent at init, so this is the alignment attributable to
            architecture and shared position structure alone.
     NULL   the same two compartments on DIFFERENT text. This breaks the
            pairing while keeping everything else, so it isolates "these are
            both English" from "these are the same sentence".
   A treatment value that does not clear both is not a finding.

WHY PAIRED COSINE LEADS AND CKA FOLLOWS. CKA compares the geometry of two point
clouds and does not use any correspondence between them. Here the correspondence
is exact and known -- token t of a given text under compartment i corresponds to
token t under compartment j -- so paired cosine uses information CKA discards.
CKA is reported alongside because it is rotation-invariant and therefore sees
alignment that survives a change of basis, which paired cosine cannot. Where the
two disagree, that disagreement is the finding, not a defect.

CROSS-SEED IS DELIBERATELY NOT USED AS A CEILING FOR COSINE. Two independently
trained models have unrelated coordinate systems, so paired cosine between them
is ~0 by construction and would look like a damning null when it is an artefact.
It is reported for CKA only, which is basis-independent.

EMBEDDING-TABLE AMORTIZATION (the part of #5 that needs no init checkpoint).
Compartment i's row for base token t is E[t + i*V]; compartment j's is
E[t + j*V]. These are separate parameters initialised independently, so their
expected cosine at init is 0 with a known spread of ~1/sqrt(D). Any consistent
positive value is movement toward a shared embedding, measured directly and
against an analytic null rather than a reconstructed one. A true
delta-theta-from-init decomposition is NOT possible: no step-0 checkpoint is
ever written (train.py requires iter_num > 0), and reconstructing an init from
the seed and calling it ground truth would be inventing the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SKIP_POSITIONS = 8          # attention sink + early-position atypicality


def canonical_batch(val_pattern: str, B: int, T: int, seed: int) -> "torch.Tensor":
    """A fixed batch drawn from the run's OWN validation shards.

    Reads via src.datafile, which dispatches on the shard header, so uint16
    (version 2) and uint32 (version 1) corpora both work.

    This used to come from compute_cossim_sweep, which hardcoded a path to the
    old DEDUPLICATED corpus tokenized with a DIFFERENT tokenizer of the same
    vocab size. Same-size vocab means no error is raised -- the ids simply mean
    different words, so the model is fed near-gibberish and the measurement
    silently answers a question nobody asked.
    """
    import glob as _glob
    from src.datafile import load_data_shard
    files = sorted(_glob.glob(val_pattern))
    if not files:
        raise FileNotFoundError(f"no val shards matched {val_pattern!r}")
    tokens = load_data_shard(files[0])
    rng = np.random.Generator(np.random.PCG64(seed))
    out = np.empty((B, T), dtype=np.int64)
    for i in range(B):
        st = int(rng.integers(0, len(tokens) - T - 1))
        out[i] = tokens[st:st + T]
    return torch.from_numpy(out).long()


def expansion_alignment(tokens: np.ndarray, flat: np.ndarray, offsets: np.ndarray,
                        T: int) -> tuple[np.ndarray, np.ndarray]:
    """Expand base tokens, and say where each base token ENDS in the result.

    Returns (expanded_ids[:T], end_index_per_base_token).

    WHY THIS EXISTS. Without BPE variants, base token j sits at position j in
    every compartment, so paired cosine can compare position j to position j.
    With variants that is false: compartment 3 might spend 2 tokens where
    compartment 5 spends 3, so the positions drift apart and comparing them
    would silently pair unrelated words while still returning a plausible number.

    The alignment point is the LAST piece of each base token -- the position at
    which both compartments have consumed exactly the same text. Any earlier
    piece is mid-word in one compartment and not the other.

    KNOWN CONFOUND, NOT CORRECTED HERE. At the end of base token j the two
    compartments are at DIFFERENT sequence positions (say 40 and 52), so RoPE
    has rotated them differently. Some of any measured dissimilarity is
    positional rather than representational. Quantifying it needs a
    same-compartment, offset-position control that is not implemented; until it
    is, expansion-run numbers are a LOWER bound on alignment.
    """
    sizes = (offsets[tokens + 1] - offsets[tokens]).astype(np.int64)
    ends = np.cumsum(sizes) - 1
    keep = ends < T
    idx = np.repeat(offsets[tokens], sizes) + (
        np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    )
    return flat[idx][:T], ends[keep]


def aligned_positions(tokens: np.ndarray, per_comp, T: int):
    """Base-token end positions common to EVERY compartment.

    A base token is usable only if its final piece fits inside the context in
    all compartments; otherwise the pairing is incomplete and the comparison
    would be over different subsets of the text.
    """
    ends = [expansion_alignment(tokens, f, o, T)[1] for (f, o) in per_comp]
    n = min(len(e) for e in ends)
    return [e[:n] for e in ends]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n, d) activation matrices, columns centred."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = float(np.linalg.norm(X.T @ Y, "fro") ** 2)
    xx = float(np.linalg.norm(X.T @ X, "fro"))
    yy = float(np.linalg.norm(Y.T @ Y, "fro"))
    return xty / (xx * yy) if xx > 0 and yy > 0 else float("nan")


def paired_cosine(X: np.ndarray, Y: np.ndarray, center: np.ndarray | None) -> dict:
    """Row-wise cosine between corresponding tokens, optionally mean-centred.

    Returns the distribution, not just the mean: a mean can hide bimodality, and
    "some tokens align perfectly and others not at all" is a different claim from
    "everything aligns moderately".
    """
    if center is not None:
        X = X - center
        Y = Y - center
    xn = np.linalg.norm(X, axis=1)
    yn = np.linalg.norm(Y, axis=1)
    ok = (xn > 1e-8) & (yn > 1e-8)
    cos = np.sum(X[ok] * Y[ok], axis=1) / (xn[ok] * yn[ok])
    return {
        "mean": float(cos.mean()), "std": float(cos.std()),
        "p05": float(np.percentile(cos, 5)), "p50": float(np.percentile(cos, 50)),
        "p95": float(np.percentile(cos, 95)), "n": int(cos.size),
    }


@torch.no_grad()
def hidden_states(model, tokens: torch.Tensor, comp: int, base_vocab: int,
                  layers, device, expand_idx=None, take=None) -> dict[int, np.ndarray]:
    """Layer -> (n_tokens, D) hidden states for `tokens` encoded in compartment `comp`.

    The encoding is `tokens + comp * base_vocab`, matching what the dataloader
    builds when permute_tokens_per_compartment is False -- which it is for every
    redesign run. If a run sets that flag, this offset is WRONG and the caller
    must apply the permutation instead; the guard in main() refuses that case
    rather than silently measuring nonsense.
    """
    if expand_idx is not None:
        # Expand under THIS compartment's drop set, then offset -- the same order
        # the dataloader uses. Alignment positions are supplied by the caller so
        # every compartment reports the same base tokens.
        flat, offs = expand_idx
        rows = [expansion_alignment(t, flat, offs, tokens.shape[1])[0]
                for t in tokens.numpy()]
        width = min(len(r) for r in rows)
        x = torch.from_numpy(np.stack([r[:width] for r in rows])).long()
        x = (x + comp * base_vocab).to(device)
    else:
        x = (tokens + comp * base_vocab).to(device)
    cid = torch.full_like(x, comp)
    out = {}
    for L in layers:
        h = model(x, compartment_ids=cid, capture_layer=L)
        # GPT.forward with capture_layer short-circuits and returns
        # (None, None, hidden) -- the hidden state is element 2, not element 0.
        if isinstance(h, tuple):
            h = h[2] if len(h) >= 3 and h[2] is not None else h[0]
        if h is None:
            raise RuntimeError(
                f"capture_layer={L} returned no hidden state; valid range is "
                f"0..n_layer-1 (0-indexed over blocks)")
        if take is not None:
            # Keep only the positions where each base token ENDS, so row k of
            # every compartment corresponds to the same base token.
            sel = take[take >= SKIP_POSITIONS]
            sel = sel[sel < h.shape[1]]
            h = h[:, torch.from_numpy(np.ascontiguousarray(sel)).to(h.device), :]
        else:
            h = h[:, SKIP_POSITIONS:, :]
        out[L] = h.reshape(-1, h.shape[-1]).float().cpu().numpy()
    return out


def embedding_amortization(model, base_vocab: int, c: int, n_tokens: int = 4096) -> dict:
    """Cosine between compartments' embedding rows for the SAME base token.

    Null is analytic: independent init gives expected cosine 0, spread ~1/sqrt(D).
    """
    wte = None
    for name, p in model.named_parameters():
        if name.endswith("wte.weight") or name.endswith("transformer.wte.weight"):
            wte = p.detach().float().cpu().numpy()
            break
    if wte is None:
        return {"error": "no wte.weight found"}
    D = wte.shape[1]
    ids = np.arange(min(n_tokens, base_vocab))
    pairs, out = [(i, j) for i in range(c) for j in range(i + 1, c)], []
    for i, j in pairs:
        A, B = wte[ids + i * base_vocab], wte[ids + j * base_vocab]
        an, bn = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
        ok = (an > 1e-8) & (bn > 1e-8)
        out.append(float(np.mean(np.sum(A[ok] * B[ok], 1) / (an[ok] * bn[ok]))))
    return {
        "mean_pairwise_cosine": float(np.mean(out)),
        "per_pair": out,
        "analytic_null_mean": 0.0,
        "analytic_null_sd": float(1.0 / np.sqrt(D)),
        "n_sigma": float(np.mean(out) / (1.0 / np.sqrt(D))),
        "d_model": int(D),
    }


def main() -> None:
    from eval_utils import load_eval_model_from_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", default="_rolling")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default="", help="comma list; default = all")
    ap.add_argument("--out_json", default="crosscomp_similarity.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "meta" / "config.json").read_text())
    exp, mdl = cfg["experiment"], cfg["model"]
    c, base_vocab = int(exp["n_compartments"]), int(mdl["vocab_size"])
    if c < 2:
        raise SystemExit(f"{run_dir} has n_compartments={c}; nothing to compare")
    if exp.get("permute_tokens_per_compartment"):
        raise SystemExit(
            "permute_tokens_per_compartment is set: compartment encoding is a "
            "PERMUTATION, not an offset, so this script would feed the wrong "
            "token ids and measure nothing. Apply the permutation first."
        )

    n_layer = int(mdl["n_layer"])
    # capture_layer is 0-indexed over BLOCKS and short-circuits the forward
    # pass, so the valid range is 0..n_layer-1. There is no embedding-output or
    # post-ln_f capture point.
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(n_layer)))
    dev = torch.device(args.device)

    # If this run used per-compartment BPE variants, rebuild the SAME drop sets
    # (pure function of seed and compartment id) and align by base-token
    # boundary. Without this every comparison would pair positions that hold
    # different words, and still return a plausible number.
    expansion = float(exp.get("bpe_variant_expansion", 0.0) or 0.0)
    per_comp = None
    if expansion > 1.0:
        from src import bpe_variants as bv
        tok_dir = exp.get("bpe_variant_tokenizer") or "tokenizers/bpe-16384-fineweb1"
        if not Path(tok_dir).is_absolute():
            tok_dir = str(Path(__file__).resolve().parent.parent / tok_dir)
        table = bv.load_merge_table(tok_dir)
        train_pat = cfg["data"]["train_bin"]
        if not Path(train_pat).is_absolute():
            train_pat = str(Path(__file__).resolve().parent.parent / train_pat)
        freq = bv.token_frequencies(train_pat, base_vocab)
        seed = int(cfg["training"]["seed"])
        per_comp = []
        for ci in range(c):
            dropped, got = bv.select_dropped_merges(freq, table, expansion, seed, ci)
            per_comp.append(bv.build_expansion_index(table, dropped, base_vocab))
            print(f"  compartment {ci}: {len(dropped):,} merges dropped, {got:.4f}x")

    ck = run_dir / "checkpoints" / args.checkpoint
    model, _, _ = load_eval_model_from_checkpoint(ck, run_dir, dev)
    model.eval()

    val_pattern = cfg["data"].get("val_bin") or cfg["data"]["train_bin"]
    if not Path(val_pattern).is_absolute():
        val_pattern = str(Path(__file__).resolve().parent.parent / val_pattern)
    print(f"canonical batch from: {val_pattern}")
    tok_a = canonical_batch(val_pattern, 8, int(mdl["block_size"]), seed=0)
    tok_b = canonical_batch(val_pattern, 8, int(mdl["block_size"]), seed=1)

    # untrained floor: same architecture, no training
    import copy
    from src.model import GPT
    floor_model = None
    try:
        floor_model = GPT(copy.deepcopy(model.config)).to(dev).eval()
    except Exception as e:                        # architecture reconstruction is
        print(f"[warn] could not build untrained floor model: {e}")   # not critical

    results = {"run": str(run_dir), "checkpoint": args.checkpoint, "c": c,
               "skip_positions": SKIP_POSITIONS, "layers": {}}
    results["embedding_amortization"] = embedding_amortization(model, base_vocab, c)

    pairs = [(i, j) for i in range(c) for j in range(i + 1, c)]
    for L in layers:
        if per_comp is None:
            H_a = {i: hidden_states(model, tok_a, i, base_vocab, [L], dev)[L] for i in range(c)}
            H_b = {i: hidden_states(model, tok_b, i, base_vocab, [L], dev)[L] for i in range(c)}
        else:
            Ta = aligned_positions(tok_a.numpy()[0], per_comp, int(mdl["block_size"]))
            Tb = aligned_positions(tok_b.numpy()[0], per_comp, int(mdl["block_size"]))
            H_a = {i: hidden_states(model, tok_a, i, base_vocab, [L], dev,
                                    expand_idx=per_comp[i], take=Ta[i])[L] for i in range(c)}
            H_b = {i: hidden_states(model, tok_b, i, base_vocab, [L], dev,
                                    expand_idx=per_comp[i], take=Tb[i])[L] for i in range(c)}
        center = np.concatenate([H_a[i] for i in range(c)], 0).mean(0, keepdims=True)

        treat_raw, treat_ctr, treat_cka, null_ctr, null_cka = [], [], [], [], []
        dists = []
        for i, j in pairs:
            treat_raw.append(paired_cosine(H_a[i], H_a[j], None)["mean"])
            _d = paired_cosine(H_a[i], H_a[j], center)
            treat_ctr.append(_d["mean"]); dists.append(_d)
            treat_cka.append(linear_cka(H_a[i], H_a[j]))
            # NULL: same compartments, different text -> pairing broken
            null_ctr.append(paired_cosine(H_a[i], H_b[j], center)["mean"])
            null_cka.append(linear_cka(H_a[i], H_b[j]))

        entry = {
            "treatment_cosine_raw": float(np.mean(treat_raw)),
            "treatment_cosine_centered": float(np.mean(treat_ctr)),
            "treatment_cka": float(np.mean(treat_cka)),
            "null_diff_text_cosine_centered": float(np.mean(null_ctr)),
            "null_diff_text_cka": float(np.mean(null_cka)),
            "treatment_cosine_dist": {
                k: float(np.mean([d[k] for d in dists]))
                for k in ("std", "p05", "p50", "p95")
            },
        }
        if floor_model is not None:
            F = {i: hidden_states(floor_model, tok_a, i, base_vocab, [L], dev)[L]
                 for i in range(c)}
            fc = np.concatenate([F[i] for i in range(c)], 0).mean(0, keepdims=True)
            entry["floor_untrained_cosine_centered"] = float(np.mean(
                [paired_cosine(F[i], F[j], fc)["mean"] for i, j in pairs]))
            entry["floor_untrained_cka"] = float(np.mean(
                [linear_cka(F[i], F[j]) for i, j in pairs]))
        results["layers"][str(L)] = entry
        print(f"layer {L:>2}: cos_ctr={entry['treatment_cosine_centered']:+.4f} "
              f"(null {entry['null_diff_text_cosine_centered']:+.4f}"
              + (f", floor {entry['floor_untrained_cosine_centered']:+.4f}"
                 if floor_model is not None else "") + ")  "
              f"cka={entry['treatment_cka']:.4f} (null {entry['null_diff_text_cka']:.4f}"
              + (f", FLOOR {entry['floor_untrained_cka']:.4f}"
                 if "floor_untrained_cka" in entry else "") + ")")

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()

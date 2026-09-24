"""Mechanistic capacity use for the 8-256 c=2 compartment-diversity models (paper Fig 2).

Compartment 0 is always English; compartment 1 is English (EN-EN), Russian, n-gram
samples of the same corpus, unigram noise, or uniform noise. All embedding/LM-head
rows are compartment-private by construction, so competition for capacity can only
happen in the shared trunk (transformer blocks + final LayerNorm). For each model at
step 1M and each compartment j we compute:

  fisher   Empirical Fisher diagonal over trunk parameters from per-sequence gradients
           of the LM loss on compartment j's validation text. Compartment 0 also gets a
           second, disjoint set of sequences (split-half) as a sampling-noise ceiling.
  attr     Attribution patching: for every MLP hidden neuron (8 x 1024) and attention
           head (8 x 8), the first-order estimate of the change in compartment j's mean
           loss from zero-ablating it, E[-a * dL/da]. Positive = the unit helps j.
  ablate   Exact zero-ablation of every MLP neuron and every head, one at a time, on each
           compartment's text (16,384 tokens each), plus a second disjoint English batch
           set ("0b") to measure how reproducible the per-unit estimates are. This is the
           primary unit-level measure; first-order attribution underestimated the most
           important neurons by up to ~10x in a pilot, though it ranked them well.

Inputs are built with the formal eval pipeline's own loaders (experiment/eval_utils.py
in the main checkout), except n-gram compartments, which the eval routes to uniform
noise; those are sampled with src/ngram_fast.FastNGramSampler exactly as training does.

Usage: python compute_capacity_use.py OUT_DIR GPU[,GPU...]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

MAIN = Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression")
RUNS = MAIN / "out" / "translation-compression"
STEP = 1_000_000
B, T = 32, 64
N_FISHER_SEQ = 1024         # per compartment (and again for the English split-half)
N_ATTR_BATCHES = 16         # 16 x 32 x 64 = 32,768 tokens per compartment
N_ABLATE_BATCHES = 8        # 16,384 tokens per compartment for exact ablation
ABLATE_GROUP = 16           # units ablated in parallel (one per replicated batch copy)
TOPK_ABLATE = 16

# condition -> run keys (group/dir or group/name prefix for sweep-runner dirs)
CONDITIONS = {
    "c=1": ["synthetic-compartment-baselines/2026-03-06T18-11-45Z__english-baseline-rope-bpe16384-8-256__2df56182__s64__4b68526__51c738c2",
            "8-256-reseed/8-256-c1baseline-s65", "8-256-reseed/8-256-c1baseline-s66"],
    "EN-EN": ["bpe16384-rope-8-256/217ca694_s64",
              "8-256-reseed/8-256-n2-tr01comp-s65", "8-256-reseed/8-256-n2-tr01comp-s66"],
    "EN-RU": ["russian-baselines-rope/2026-03-01T00-03-55Z__russian-english-baseline-rope-bpe16384-8-256__c7e8d8f0__s64__4b68526__79d396a8",
              "fig2-diversity-seeds/8-256-en-ru-s65", "fig2-diversity-seeds/8-256-en-ru-s66"],
    "EN-unigram": ["synthetic-compartment-baselines/2026-03-05T22-39-16Z__english-frequency-2comp-rope-bpe16384-8-256__605a1512__s64__4b68526__2acd312f",
                   "fig2-diversity-seeds/8-256-en-unigram-s65", "fig2-diversity-seeds/8-256-en-unigram-s66"],
    "EN-uniform": ["synthetic-compartment-baselines/2026-03-05T22-39-06Z__english-uniform-2comp-rope-bpe16384-8-256__11b3d274__s64__4b68526__a6d73c34",
                   "fig2-diversity-seeds/8-256-en-uniform-s65", "fig2-diversity-seeds/8-256-en-uniform-s66"],
    "EN-1.5gram": ["capacity-ngram-ladder/8-256-c2-english-vs-ngram1p5"],
    "EN-2gram": ["capacity-ngram-ladder/8-256-c2-english-vs-ngram2",
                 "capacity-ngram-seeds/8-256-c2-english-vs-ngram2-s65", "capacity-ngram-seeds/8-256-c2-english-vs-ngram2-s66"],
    "EN-2.5gram": ["capacity-ngram-ladder/8-256-c2-english-vs-ngram2p5"],
    "EN-3gram": ["capacity-ngram-ladder/8-256-c2-english-vs-ngram3",
                 "capacity-ngram-seeds/8-256-c2-english-vs-ngram3-s65", "capacity-ngram-seeds/8-256-c2-english-vs-ngram3-s66"],
    "EN-3.5gram": ["capacity-ngram-ladder/8-256-c2-english-vs-ngram3p5"],
    "EN-4gram": ["capacity-ngram-ladder/8-256-c2-english-vs-ngram4",
                 "capacity-ngram-seeds/8-256-c2-english-vs-ngram4-s65", "capacity-ngram-seeds/8-256-c2-english-vs-ngram4-s66"],
}


def resolve(key: str) -> Path:
    p = RUNS / key
    if p.is_dir():
        return p
    group, name = key.split("/", 1)
    hits = sorted(d for d in (RUNS / group).iterdir() if d.is_dir() and f"__{name}__" in d.name)
    assert len(hits) == 1, (key, hits)
    return hits[0]


def _imports():
    sys.path[:0] = [str(MAIN), str(MAIN / "experiment")]
    import eval_utils as eu  # noqa: E402
    from src.ngram_fast import FastNGramSampler  # noqa: E402
    from src.token_tying import compute_token_frequencies  # noqa: E402
    return eu, FastNGramSampler, compute_token_frequencies


def make_loader(eu, FastNGramSampler, freq_fn, config, comp, n_model_c, device, seed, skip_seqs=0):
    """Loader of compartment-`comp` examples, mirroring evaluate_checkpoints_fineweb_dedup.py."""
    V = eu.get_base_vocab_size(config)
    exp = config.experiment
    assert not exp.permute_tokens_per_compartment, "permuted-token runs not supported"
    sources = list(config.data.compartment_val_bins or [])
    src = sources[comp] if sources else config.data.val_bin
    a = eu.Assignment(kind=0, src=comp)
    common = dict(B=B, T=T, base_vocab_size=V, max_compartments=n_model_c, assignment=a, device=device,
                  permute_tokens=False, permute_inputs=exp.permute_input_tokens_per_compartment)
    if src.startswith("synthetic:"):
        mode = src.split(":", 1)[1]
        probs = None
        if mode == "frequency":
            train_src = next(p for p in config.data.compartment_train_bins if not p.startswith("synthetic:"))
            f = freq_fn(str(MAIN / train_src), V)
            probs = (f / f.sum()).astype(np.float64)
        loader = eu.UniformAssignedValLoader(seed=seed, num_batches=10_000, token_probs=probs, **common)
        if mode.startswith("ngram"):
            spec = mode[5:]
            order, lam = (int(spec.split("x")[0]), float(spec.split("x")[1])) if "x" in spec else (int(spec), 1.0)
            sampler = FastNGramSampler(order=order, table_dir=MAIN / "data" / "ngram-tables-bpe16384",
                                       seed=seed, process_rank=0, lam=lam)
            loader._generate_tokens = lambda n: sampler.read_tokens(n).astype(np.int64)
        elif mode not in ("uniform", "frequency"):
            raise ValueError(mode)
    else:
        shard = sorted(MAIN.glob(src))[0]
        loader = eu.SingleShardAssignedValLoader(str(shard), **common)
    it = iter(loader)
    for _ in range(skip_seqs // B):
        next(it)
    return it


def trunk_params(model):
    return {n: p for n, p in model.named_parameters()
            if n.startswith("transformer.h.") or n.startswith("transformer.ln_f")}


def fisher(model, params, it, n_seq):
    acc = {n: torch.zeros_like(p) for n, p in params.items()}
    seen = 0
    while seen < n_seq:
        x, y, cid = next(it)
        for b in range(x.shape[0]):
            model.zero_grad(set_to_none=True)
            _, loss = model(x[b:b + 1], targets=y[b:b + 1], compartment_ids=cid[b:b + 1])
            loss.backward()
            for n, p in params.items():
                acc[n] += p.grad.detach() ** 2
            seen += 1
            if seen >= n_seq:
                break
    return {n: v / seen for n, v in acc.items()}


class Taps:
    """Capture inputs to every block's mlp.c_proj (MLP hidden) and attn.c_proj (head outputs)."""

    def __init__(self, model):
        self.acts, self.handles = {}, []
        self.scale = {}  # (kind, layer) -> multiplier tensor for ablation, or None
        for i, blk in enumerate(model.transformer.h):
            for kind, mod in (("mlp", blk.mlp.c_proj), ("attn", blk.attn.c_proj)):
                self.handles.append(mod.register_forward_pre_hook(self._hook(kind, i)))

    def _hook(self, kind, i):
        def f(mod, inp):
            a = inp[0]
            s = self.scale.get((kind, i))
            if s is not None:
                a = a * s
            if torch.is_grad_enabled() and a.requires_grad:
                a.retain_grad()
            self.acts[(kind, i)] = a
            return (a,)
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def attribution(model, taps, it, n_batches, n_head):
    out, losses = {}, []
    for _ in range(n_batches):
        x, y, cid = next(it)
        model.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, compartment_ids=cid)
        loss.backward()
        losses.append(loss.item())
        for (kind, i), a in taps.acts.items():
            ag = -(a * a.grad).detach()                    # first-order zero-ablation effect on mean loss
            if kind == "mlp":
                v = ag.sum((0, 1))
            else:
                Bx, Tx, C = ag.shape
                v = ag.view(Bx, Tx, n_head, C // n_head).sum((0, 1, 3))
            out[(kind, i)] = out.get((kind, i), 0) + v / n_batches
    return {k: v.float().cpu().numpy() for k, v in out.items()}, float(np.mean(losses))


@torch.no_grad()
def batch_losses(model, batches):
    return float(np.mean([model(x, targets=y, compartment_ids=c)[1].item() for x, y, c in batches]))


@torch.no_grad()
def exact_ablation(model, taps, batches, kind, n_layer, n_units, unit_width):
    """True zero-ablation of every unit: returns (n_layer, n_units) change in mean loss (nats).

    `unit_width` is how many consecutive channels one unit spans at the tapped input
    (1 for MLP neurons, head_dim for attention heads). ABLATE_GROUP units are ablated
    at once by replicating each batch and giving every copy its own mask.
    """
    ce = torch.nn.functional.cross_entropy
    width = n_units * unit_width

    def group_loss(k, s_full):
        tot = torch.zeros(k, device=s_full.device)
        for x, y, c in batches:
            Bx = x.shape[0]
            taps.scale[(kind, L)] = s_full.repeat_interleave(Bx, dim=0)
            logits, _ = model(x.repeat(k, 1), targets=y.repeat(k, 1), compartment_ids=c.repeat(k, 1))
            taps.scale[(kind, L)] = None
            yr = y.repeat(k, 1)
            tok = ce(logits.float().reshape(-1, logits.shape[-1]), yr.reshape(-1),
                     ignore_index=-1, reduction="none").view(k, -1)
            valid = (yr >= 0).view(k, -1).float()
            tot += (tok * valid).sum(1) / valid.sum(1)
        return tot / len(batches)

    base = batch_losses(model, batches)
    out = np.zeros((n_layer, n_units))
    dev = batches[0][0].device
    for L in range(n_layer):
        for start in range(0, n_units, ABLATE_GROUP):
            units = list(range(start, min(start + ABLATE_GROUP, n_units)))
            s = torch.ones(len(units), 1, width, device=dev)
            for i, u in enumerate(units):
                s[i, 0, u * unit_width:(u + 1) * unit_width] = 0.0
            out[L, units] = (group_loss(len(units), s) - base).cpu().numpy()
    return out


def run_one(job):
    cond, key, gpu, out_dir = job
    out_path = Path(out_dir) / f"{cond}__{key.replace('/', '__')}.json"
    if out_path.exists():
        return str(out_path), "cached"
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    torch.manual_seed(0)
    eu, NG, freq_fn = _imports()
    run_dir = resolve(key)
    t0 = time.time()
    model, config, n_model_c = eu.load_eval_model_from_checkpoint(
        run_dir / "checkpoints" / f"step-{STEP:06d}", run_dir, device, dtype=torch.float32)
    model.eval()
    n_c = int(config.experiment.n_compartments)
    n_head = int(config.model.n_head)
    params = trunk_params(model)
    comps = list(range(n_c))
    res = {"condition": cond, "key": key, "run_dir": str(run_dir), "step": STEP, "n_compartments": n_c,
           "val_sources": list(config.data.compartment_val_bins or [config.data.val_bin] * n_c)}

    # 1. Fisher (per-sequence), plus an English split-half from disjoint sequences.
    F = {}
    for j in comps:
        F[j] = fisher(model, params, make_loader(eu, NG, freq_fn, config, j, n_model_c, device, seed=1000 + j),
                      N_FISHER_SEQ)
    F["0b"] = fisher(model, params, make_loader(eu, NG, freq_fn, config, 0, n_model_c, device, seed=2000,
                                                skip_seqs=N_FISHER_SEQ), N_FISHER_SEQ)
    flat = {k: torch.cat([v.flatten() for v in f.values()]) for k, f in F.items()}
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    res["fisher_total"] = {str(k): float(v.sum()) for k, v in flat.items()}
    res["fisher_by_tensor"] = {str(k): {n: float(t.sum()) for n, t in f.items()} for k, f in F.items()}
    res["fisher_cos_split_half_en"] = cos(flat[0], flat["0b"])
    if n_c > 1:
        res["fisher_cos_0_1"] = cos(flat[0], flat[1])
    del F, flat

    # 2. Attribution patching over MLP neurons and heads.
    taps = Taps(model)
    attr = {}
    for j in comps:
        a, l = attribution(model, taps, make_loader(eu, NG, freq_fn, config, j, n_model_c, device, seed=3000 + j),
                           N_ATTR_BATCHES, n_head)
        attr[j] = a
        res[f"loss_{j}"] = l
    a0b, _ = attribution(model, taps, make_loader(eu, NG, freq_fn, config, 0, n_model_c, device, seed=4000,
                                                  skip_seqs=N_ATTR_BATCHES * B), N_ATTR_BATCHES, n_head)
    res["attr"] = {str(j): {f"{k}{i}": v.tolist() for (k, i), v in attr[j].items()} for j in comps}
    res["attr"]["0b"] = {f"{k}{i}": v.tolist() for (k, i), v in a0b.items()}

    # 3. Exact zero-ablation of every MLP neuron and attention head, per compartment,
    #    plus a second disjoint English batch set to measure the estimate's reliability.
    n_layer = len(model.transformer.h)
    d_model = int(config.model.n_embd)
    sets = {str(j): [next(it) for _ in range(N_ABLATE_BATCHES)]
            for j in comps for it in [make_loader(eu, NG, freq_fn, config, j, n_model_c, device, seed=5000 + j)]}
    it0b = make_loader(eu, NG, freq_fn, config, 0, n_model_c, device, seed=6000, skip_seqs=N_ABLATE_BATCHES * B)
    sets["0b"] = [next(it0b) for _ in range(N_ABLATE_BATCHES)]
    res["ablate"] = {}
    for tag, batches in sets.items():
        res["ablate"][tag] = {
            "loss": batch_losses(model, batches),
            "mlp": exact_ablation(model, taps, batches, "mlp", n_layer, 4 * d_model, 1).tolist(),
            "head": exact_ablation(model, taps, batches, "attn", n_layer, n_head, d_model // n_head).tolist(),
        }
    taps.remove()
    res["seconds"] = time.time() - t0
    out_path.write_text(json.dumps(res))
    return str(out_path), f"{res['seconds']:.0f}s"


if __name__ == "__main__":
    import multiprocessing as mp
    out_dir, gpus = sys.argv[1], [int(g) for g in sys.argv[2].split(",")]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(c, k) for c, keys in CONDITIONS.items() for k in keys]
    jobs = [(c, k, gpus[i % len(gpus)], out_dir) for i, (c, k) in enumerate(jobs)]
    only = sys.argv[3] if len(sys.argv) > 3 else None
    if only:
        jobs = [j for j in jobs if only in j[1]][:1]
    with ProcessPoolExecutor(len(gpus), mp_context=mp.get_context("spawn")) as ex:
        for path, msg in ex.map(run_one, jobs):
            print(path, msg, flush=True)

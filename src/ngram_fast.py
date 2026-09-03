"""Compiled n-gram sampler: open-addressed hash + per-row linear scan.

WHY NOT THE OBVIOUS NUMPY VERSION. Sorted-array + np.searchsorted is 28-29
DEPENDENT random accesses into multi-GB arrays -- a cache miss per level. Profiled
on the n=4 table (233M contexts / 1.9 GB, 550M pairs / 4.4 GB) that costs 5.3 ms
per position for 4096 chains, i.e. ~0.8M tok/s from the searches alone, before any
other overhead. And it is absurd on its face: a 29-level binary search over 4.4 GB
to choose among a row that averages 2.35 successors and fits in ONE cache line.

WHAT THIS DOES INSTEAD
  * contexts in an open-addressed hash table, 2x overprovisioned -> 1-2 probes
  * successors scanned linearly inside the row -> 1 cache miss, not 29
  * ~3 cache misses per token instead of ~57

ONE TABLE FOR ALL ORDERS. The hash key is (context << 3) | order, so backing off
from order n to n-1 is the identical lookup at a different order rather than a
second data structure. Contexts are <=42 bits (14 bits x 3), so key is <=45 bits
and still a uint64.

BACKOFF. Held-out context coverage is ~100% / 96% / 82% at n=2/3/4 -- the Zipf tail
is unbounded, so no in-RAM table reaches full coverage (3x the corpus buys +5.6
points for 2.4x memory). A missing context therefore drops its oldest token and
retries at n-1, bottoming out at unigram. Katz/stupid-backoff structure (Katz 1987;
Brants et al. 2007) without the discounting, since we sample rather than score.
Measured realised fallback at n=4 is ~0.2%, never reaching unigram.

DETERMINISM. Each chain carries its own xorshift64* state seeded from (seed, chain),
so chains are independent, reproducible, and unaffected by thread scheduling under
prange. state_dict() saves those states plus the context window and buffer offset,
so a resumed run emits the identical stream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange

BITS = 14
EMPTY = np.int32(-1)


# ---------------------------------------------------------------- table build
def build_merged(table_dir: str | Path, max_order: int):
    """Merge per-order CSR tables into one hash table + one row store."""
    d = Path(table_dir)
    ctx_l, ord_l, ptr_l, succ_l, rcdf_l = [], [], [], [], []
    row_base = 0
    pair_base = 0
    for o in range(2, max_order + 1):
        z = np.load(d / f"ngram{o}.npz")
        ctx, ptr, succ, cdf = z["ctx"], z["ptr"], z["succ"], z["cdf"]
        # global cdf -> within-row cumulative (float32 is ample: rows end at 1.0)
        lo, hi = ptr[:-1], ptr[1:]
        base = np.where(lo > 0, cdf[np.maximum(lo - 1, 0)], 0.0)
        span = cdf[hi - 1] - base
        span[span <= 0] = 1.0
        rcdf = ((cdf - np.repeat(base, hi - lo)) / np.repeat(span, hi - lo)).astype(np.float32)
        ctx_l.append(ctx)
        ord_l.append(np.full(len(ctx), o, dtype=np.uint8))
        ptr_l.append(ptr[:-1].astype(np.int64) + pair_base)
        succ_l.append(succ)
        rcdf_l.append(rcdf)
        row_base += len(ctx)
        pair_base += len(succ)

    all_ctx = np.concatenate(ctx_l)
    all_ord = np.concatenate(ord_l)
    row_lo = np.concatenate(ptr_l)
    succ = np.concatenate(succ_l)
    rcdf = np.concatenate(rcdf_l)
    # row_hi from the next row's lo within each order block
    row_hi = np.empty_like(row_lo)
    off = 0
    for i, o in enumerate(range(2, max_order + 1)):
        n = len(ctx_l[i])
        z = np.load(d / f"ngram{o}.npz")
        row_hi[off:off + n] = z["ptr"][1:].astype(np.int64) + (row_lo[off] - z["ptr"][0])
        off += n

    keys = (all_ctx << np.uint64(3)) | all_ord.astype(np.uint64)
    n_slots = 1 << int(np.ceil(np.log2(max(4, 2 * len(keys)))))
    h_key = np.zeros(n_slots, dtype=np.uint64)
    h_row = np.full(n_slots, EMPTY, dtype=np.int32)
    _fill_hash(keys, h_key, h_row, np.uint64(n_slots - 1))
    uni = np.load(d / "unigram.npz")["unigram"].astype(np.float64)
    uni_cdf = np.cumsum(uni / uni.sum()).astype(np.float32)
    return dict(h_key=h_key, h_row=h_row, mask=np.uint64(n_slots - 1),
                row_lo=row_lo, row_hi=row_hi, succ=succ, rcdf=rcdf, uni_cdf=uni_cdf)


@njit(cache=True)
def _mix(k: np.uint64) -> np.uint64:
    k ^= k >> np.uint64(33)
    k *= np.uint64(0xFF51AFD7ED558CCD)
    k ^= k >> np.uint64(33)
    k *= np.uint64(0xC4CEB9FE1A85EC53)
    k ^= k >> np.uint64(33)
    return k


@njit(cache=True)
def _fill_hash(keys, h_key, h_row, mask):
    for i in range(keys.shape[0]):
        k = keys[i]
        s = _mix(k) & mask
        while h_row[s] != EMPTY:
            s = (s + np.uint64(1)) & mask
        h_key[s] = k
        h_row[s] = i


@njit(inline="always")
def _probe(h_key, h_row, mask, key):
    s = _mix(key) & mask
    while True:
        r = h_row[s]
        if r == EMPTY:
            return np.int32(-1)
        if h_key[s] == key:
            return r
        s = (s + np.uint64(1)) & mask


@njit(inline="always")
def _rand(state):
    """xorshift64* -> (new_state, uniform in [0,1))"""
    x = state
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    r = (x * np.uint64(0x2545F4914F6CDD1D)) >> np.uint64(11)
    return x, np.float32(r) / np.float32(1 << 53)


@njit(parallel=True, cache=True)
def gen_block(out, ctx_state, rng_state, start_order, lam,
              h_key, h_row, mask, row_lo, row_hi, succ, rcdf, uni_cdf, counts):
    K, L = out.shape
    W = ctx_state.shape[1]
    for k in prange(K):
        st = rng_state[k]
        for j in range(L):
            tok = np.int32(-1)
            top = start_order
            if lam < 1.0:                      # Jelinek-Mercer: pick the component
                st, uu = _rand(st)
                if uu >= lam:
                    top = start_order - 1
            for o in range(top, 1, -1):
                need = o - 1
                key = np.uint64(0)
                for i in range(need):                      # most recent `need` tokens
                    key |= np.uint64(ctx_state[k, W - need + i]) << np.uint64(BITS * (need - 1 - i))
                key = (key << np.uint64(3)) | np.uint64(o)
                r = _probe(h_key, h_row, mask, key)
                if r >= 0:
                    st, u = _rand(st)
                    lo = row_lo[r]
                    hi = row_hi[r]
                    p = lo
                    while p < hi - 1 and rcdf[p] < u:       # linear scan, mean 2.35
                        p += 1
                    tok = np.int32(succ[p])
                    counts[k, o] += 1
                    break
            if tok < 0:                                    # unigram floor
                st, u = _rand(st)
                p = 0
                while p < uni_cdf.shape[0] - 1 and uni_cdf[p] < u:
                    p += 1
                tok = np.int32(p)
                counts[k, 1] += 1
            out[k, j] = tok
            for i in range(W - 1):
                ctx_state[k, i] = ctx_state[k, i + 1]
            ctx_state[k, W - 1] = tok
        rng_state[k] = st


class FastNGramSampler:
    def __init__(self, order: int, table_dir: str | Path, seed: int,
                 process_rank: int = 0, n_chains: int = 2048, block_len: int = 2048,
                 lam: float = 1.0):
        # lam < 1 gives a fractional order: lam*P_order + (1-lam)*P_{order-1},
        # i.e. lam=0.5 with order=3 is a "2.5-gram" in the Jelinek-Mercer sense.
        self.lam = float(lam)
        # process_rank MUST enter the seed: without it every DDP rank would emit
        # the identical token stream, silently collapsing the effective batch.
        seed = int(seed) + 1_000_003 * int(process_rank)
        self.order = order
        self.n_chains = n_chains
        self.block_len = block_len
        self.T = build_merged(table_dir, order)
        ss = np.random.SeedSequence(seed)
        self.rng_state = (ss.generate_state(n_chains, dtype=np.uint64) | np.uint64(1))
        rng = np.random.default_rng(seed)
        W = max(1, order - 1)
        self.ctx_state = rng.integers(0, 1 << BITS, size=(n_chains, W)).astype(np.int32)
        self.counts = np.zeros((n_chains, order + 1), dtype=np.int64)
        self._buf = np.empty(0, dtype=np.int32)
        self._pos = 0

    def _fill(self):
        # snapshot the state the block is generated FROM, so a checkpoint can
        # reproduce this exact block without storing it
        self._blk_rng = self.rng_state.copy()
        self._blk_ctx = self.ctx_state.copy()
        self._blk_counts = self.counts.copy()
        out = np.empty((self.n_chains, self.block_len), dtype=np.int32)
        T = self.T
        gen_block(out, self.ctx_state, self.rng_state, self.order, self.lam,
                  T["h_key"], T["h_row"], T["mask"], T["row_lo"], T["row_hi"],
                  T["succ"], T["rcdf"], T["uni_cdf"], self.counts)
        self._buf = out.reshape(-1)      # chain-major: each chain contiguous
        self._pos = 0

    def read_tokens(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int32)
        f = 0
        while f < n:
            if self._pos >= len(self._buf):
                self._fill()
            take = min(n - f, len(self._buf) - self._pos)
            out[f:f + take] = self._buf[self._pos:self._pos + take]
            self._pos += take
            f += take
        return out

    def backoff_report(self) -> dict:
        c = self.counts.sum(0)
        tot = c.sum() or 1
        return {f"order_{o}": float(c[o]) / tot for o in range(len(c) - 1, 0, -1) if c[o]}

    def state_dict(self) -> dict:
        """Block-start state + offset. Replaying is exact and ~64 KB instead of
        the ~16 MB the emitted buffer would cost on every checkpoint."""
        if not hasattr(self, "_blk_rng"):
            return {"rng_state": self.rng_state.copy(),
                    "ctx_state": self.ctx_state.copy(),
                    "counts": self.counts.copy(), "pos": 0, "fresh": True}
        return {"rng_state": self._blk_rng.copy(), "ctx_state": self._blk_ctx.copy(),
                "counts": self._blk_counts.copy(), "pos": int(self._pos), "fresh": False}

    def load_state_dict(self, s: dict) -> None:
        if "rng_state" in s:
            self.rng_state = np.asarray(s["rng_state"], dtype=np.uint64).copy()
        if "ctx_state" in s:
            self.ctx_state = np.asarray(s["ctx_state"], dtype=np.int32).copy()
        if "counts" in s:
            self.counts = np.asarray(s["counts"], dtype=np.int64).copy()
        pos = int(s.get("pos", 0))
        if s.get("fresh", False):
            self._buf = np.empty(0, dtype=np.int32); self._pos = 0
        else:
            self._fill()          # regenerate the exact block from its start state
            self._pos = pos

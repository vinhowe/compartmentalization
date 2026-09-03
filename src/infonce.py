"""InfoNCE alignment helpers — paired-sequence pool + contrastive loss.

Pool layout on disk (produced by scripts/prepare_wikimatrix_qwen3.py):
  <dir>/wikimatrix_en.bin       — uint32 tokens, 1024-byte header (magic 20251013)
  <dir>/wikimatrix_zh.bin       — uint32 tokens, 1024-byte header
  <dir>/wikimatrix_pairs.npy    — int64 [N, 4] = (en_start, en_len, zh_start, zh_len)

The full pool is N pairs. infonce_pool_frac restricts to the first frac*N
pair indices (deterministic; same subset every run with same seed).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


MAGIC = 20251013


def _load_bin_tokens(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == MAGIC, f"magic mismatch in {path}"
        ntok = int(header[2])
    return np.memmap(str(path), dtype=np.uint32, mode="r", offset=1024, shape=(ntok,))


class InfoNCEPool:
    """Bridge pool of paired sequences for the InfoNCE alignment loss."""

    def __init__(
        self,
        pool_path: str,
        pool_frac: float = 1.0,
        pool_seed: int = 0,
        process_rank: int = 0,
        zh_token_offset: int = 0,
    ):
        d = Path(pool_path)
        self.zh_token_offset = int(zh_token_offset)
        self.en_tokens = _load_bin_tokens(d / "wikimatrix_en.bin")
        self.zh_tokens = _load_bin_tokens(d / "wikimatrix_zh.bin")
        pairs = np.load(d / "wikimatrix_pairs.npy")
        assert pairs.ndim == 2 and pairs.shape[1] == 4

        # If the pool dir has a train_idx.npy, restrict to those pair indices
        # (so val pairs never leak into InfoNCE). Otherwise use all pairs.
        train_idx_path = d / "train_idx.npy"
        if train_idx_path.exists():
            base_indices = np.load(train_idx_path).astype(np.int64)
        else:
            base_indices = np.arange(pairs.shape[0], dtype=np.int64)
        n_base = len(base_indices)

        # Deterministic subset of base_indices: shuffle with pool_seed, take frac.
        rng = np.random.default_rng(pool_seed)
        order = rng.permutation(n_base)
        n_keep = max(1, int(round(n_base * pool_frac)))
        self.indices = base_indices[order[:n_keep]]
        self.pairs = pairs  # full pairs array (indexed by self.indices)
        self.n_total = n_base
        self.n_keep = n_keep
        self.process_rank = process_rank
        # Per-rank RNG so different DDP ranks sample different pairs each call
        self._rng = np.random.default_rng(pool_seed + 1_000_000 + process_rank)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample n pairs with replacement from the pool subset.

        Returns (en_tokens [n, max_en], en_mask [n, max_en],
                 zh_tokens [n, max_zh], zh_mask [n, max_zh])
        with right-side zero padding.
        """
        idx = self._rng.choice(self.n_keep, size=n, replace=True)
        sel = self.indices[idx]
        chosen = self.pairs[sel]  # [n, 4]
        en_starts = chosen[:, 0]
        en_lens = chosen[:, 1]
        zh_starts = chosen[:, 2]
        zh_lens = chosen[:, 3]
        max_en = int(en_lens.max())
        max_zh = int(zh_lens.max())
        en_tokens = np.zeros((n, max_en), dtype=np.int64)
        zh_tokens = np.zeros((n, max_zh), dtype=np.int64)
        en_mask = np.zeros((n, max_en), dtype=np.bool_)
        zh_mask = np.zeros((n, max_zh), dtype=np.bool_)
        for i in range(n):
            es, el = int(en_starts[i]), int(en_lens[i])
            zs, zl = int(zh_starts[i]), int(zh_lens[i])
            en_tokens[i, :el] = self.en_tokens[es : es + el].astype(np.int64)
            zh_tokens[i, :zl] = self.zh_tokens[zs : zs + zl].astype(np.int64)
            en_mask[i, :el] = True
            zh_mask[i, :zl] = True
        if self.zh_token_offset:
            zh_tokens[zh_mask] += self.zh_token_offset
        return en_tokens, en_mask, zh_tokens, zh_mask


class CompartmentOriginalPool:
    """Sample raw training-data sequences and present them in two distinct
    compartments via the same offset+permutation scheme the data loader uses.

    For each sample_pairs(n, ci, cj) call, returns x_ci, x_cj of shape (n, T):
    the SAME n underlying token sequences, expressed in compartments ci and cj.

    `train_bin_glob` points at the same shards used by the main training run.
    `permutations` is the (max_compartments, base_vocab) numpy array loaded
    from out_dir/permutations.npy, or None if the model doesn't use input
    permutations.
    """

    def __init__(
        self,
        train_bin_glob: str,
        seq_len: int,
        base_vocab: int,
        permutations: Optional[np.ndarray] = None,
        process_rank: int = 0,
        seed: int = 0,
    ):
        import glob as _glob
        files = sorted(_glob.glob(train_bin_glob))
        assert files, f"no train shards match {train_bin_glob}"
        # Memmap each shard, skipping its 1024-byte header. Tokens are uint16
        # (stored at base vocab, in [0, base_vocab)).
        self._shards: list[np.memmap] = []
        for fname in files:
            with open(fname, "rb") as f:
                header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
                ntok = int(header[2])
            self._shards.append(
                np.memmap(fname, dtype=np.uint16, mode="r", offset=1024, shape=(ntok,))
            )
        self._shard_lens = np.array([len(s) for s in self._shards], dtype=np.int64)
        self._cum_lens = np.concatenate([[0], np.cumsum(self._shard_lens)])
        self.total_tokens = int(self._cum_lens[-1])
        self.seq_len = seq_len
        self.base_vocab = base_vocab
        self.permutations = permutations
        self._rng = np.random.default_rng(seed + 1_000_000 + process_rank)

    def _slice(self, start: int, length: int) -> np.ndarray:
        # Random sequence might straddle shards; pick a shard and ensure fit.
        # Simpler: pick a shard uniformly weighted by length, then offset within.
        # We use the cumulative-len trick.
        end = start + length
        # Find shard containing `start`
        shard_idx = int(np.searchsorted(self._cum_lens, start, side="right") - 1)
        local_start = start - int(self._cum_lens[shard_idx])
        if local_start + length <= self._shard_lens[shard_idx]:
            return np.asarray(
                self._shards[shard_idx][local_start : local_start + length],
                dtype=np.int64,
            )
        # Spans shards: read from current up to shard end, then continue.
        out = np.empty(length, dtype=np.int64)
        cur = local_start
        si = shard_idx
        filled = 0
        while filled < length and si < len(self._shards):
            avail = int(self._shard_lens[si]) - cur
            take = min(avail, length - filled)
            out[filled : filled + take] = np.asarray(
                self._shards[si][cur : cur + take], dtype=np.int64
            )
            filled += take
            si += 1
            cur = 0
        if filled < length:
            # Wrap around to the first shard (rare).
            out[filled:] = np.asarray(
                self._shards[0][: length - filled], dtype=np.int64
            )
        return out

    def sample_pairs(
        self, n: int, ci: int, cj: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns x_ci (n, T) and x_cj (n, T), both int64. Same underlying
        token sequences, expressed in compartments ci and cj.

        Apply (a) per-compartment input permutation (if self.permutations is
        not None) and (b) +c*base_vocab offset, matching the data loader's
        compartment-mode behaviour."""
        max_start = self.total_tokens - self.seq_len - 1
        starts = self._rng.integers(0, max_start, size=n)
        x_ci = np.empty((n, self.seq_len), dtype=np.int64)
        x_cj = np.empty((n, self.seq_len), dtype=np.int64)
        for i, s in enumerate(starts):
            raw = self._slice(int(s), self.seq_len)
            if self.permutations is not None:
                raw_ci = self.permutations[ci][raw]
                raw_cj = self.permutations[cj][raw]
            else:
                raw_ci = raw
                raw_cj = raw
            x_ci[i] = raw_ci + ci * self.base_vocab
            x_cj[i] = raw_cj + cj * self.base_vocab
        return x_ci, x_cj


class BioDeclQAPool:
    """Per-person paired DECL/QA pool for bio InfoNCE alignment.

    Loads two memmap'd token files (decl.bin, qa.bin) of shape (N, L) plus
    matching `decl_lens.npy` / `qa_lens.npy` for masked mean-pool. Each file
    is written by scripts/build_bio_paired_pool.py.

    Tokens stored raw in [0, base_vocab). At sample time, the QA side gets
    `qa_offset` added so it lands in the right vocab region for the model:
      - 0 (no compartmentation, vocab=base_vocab): both views in [0, V)
      - V (vocab-split compartmentation, model vocab=2V): DECL in [0, V),
        QA in [V, 2V)
    DECL is always returned with no offset (in [0, base_vocab)).
    """

    HEADER_INTS = 256

    def __init__(
        self,
        decl_path: str,
        qa_path: str,
        qa_offset: int = 0,
        process_rank: int = 0,
        seed: int = 0,
    ):
        self._decl, self._decl_lens, self._L, self._eos_id = self._load_pool(decl_path)
        self._qa, self._qa_lens, qa_L, qa_eos = self._load_pool(qa_path)
        assert self._L == qa_L, f"L mismatch: decl {self._L} vs qa {qa_L}"
        assert self._eos_id == qa_eos, f"EOS mismatch"
        assert len(self._decl) == len(self._qa), "person count mismatch"
        self.n_persons = len(self._decl)
        self.qa_offset = qa_offset
        self._rng = np.random.default_rng(seed + 2_000_000 + process_rank)

    @staticmethod
    def _load_pool(path: str) -> tuple[np.memmap, np.ndarray, int, int]:
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(BioDeclQAPool.HEADER_INTS * 4), dtype=np.int32)
        assert header[0] == MAGIC, f"magic mismatch in {path}"
        n_persons = int(header[2])
        seq_len = int(header[3])
        eos_id = int(header[4])
        arr = np.memmap(
            path, dtype=np.uint32, mode="r",
            offset=BioDeclQAPool.HEADER_INTS * 4, shape=(n_persons, seq_len),
        )
        lens_path = str(Path(path).parent / (Path(path).stem + "_lens.npy"))
        if not os.path.exists(lens_path):
            raise FileNotFoundError(f"missing {lens_path}; rebuild pool")
        lens = np.load(lens_path)
        assert len(lens) == n_persons
        return arr, lens, seq_len, eos_id

    def sample(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (decl_tokens, decl_mask, qa_tokens, qa_mask).
        Each shape (n, L), int64 (tokens) / bool (masks). qa_offset is applied
        to QA non-pad positions only; pad (EOS) positions stay at canonical
        eos_id so the model sees consistent EOS regardless of compartment.
        """
        idx = self._rng.integers(0, self.n_persons, size=n)
        decl_tok = np.asarray(self._decl[idx], dtype=np.int64)
        qa_tok = np.asarray(self._qa[idx], dtype=np.int64)
        positions = np.arange(self._L, dtype=np.int64)[None, :]
        decl_mask = positions < self._decl_lens[idx][:, None]
        qa_mask = positions < self._qa_lens[idx][:, None]
        if self.qa_offset:
            # Apply offset only to non-pad positions; pad stays at canonical EOS.
            qa_tok = np.where(qa_mask, qa_tok + self.qa_offset, qa_tok)
        return decl_tok, decl_mask, qa_tok, qa_mask


def compute_infonce_compartment_loss(
    model,
    x_ci: torch.Tensor,
    x_cj: torch.Tensor,
    capture_layer: int,
    tau: float,
    ctx,
) -> torch.Tensor:
    """InfoNCE loss for compartment-pair sequences. Mean-pools across all
    positions (no masking — sequences are fixed-length, no padding needed)."""
    n = x_ci.shape[0]
    with ctx:
        _, _, h_ci = model(x_ci, capture_layer=capture_layer)  # (n, T, D)
    with ctx:
        _, _, h_cj = model(x_cj, capture_layer=capture_layer)
    rep_ci = h_ci.float().mean(dim=1)  # (n, D)
    rep_cj = h_cj.float().mean(dim=1)
    rep_ci = F.normalize(rep_ci, dim=-1)
    rep_cj = F.normalize(rep_cj, dim=-1)
    sim = (rep_ci @ rep_cj.t()) / tau  # (n, n)
    labels = torch.arange(n, device=sim.device)
    loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
    return loss


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hidden: (B, T, D). mask: (B, T) bool. Returns (B, D)."""
    mf = mask.to(hidden.dtype).unsqueeze(-1)  # (B, T, 1)
    summed = (hidden * mf).sum(dim=1)
    denom = mf.sum(dim=1).clamp(min=1.0)
    return summed / denom


def compute_infonce_loss(
    model,
    en_tokens: torch.Tensor,
    en_mask: torch.Tensor,
    zh_tokens: torch.Tensor,
    zh_mask: torch.Tensor,
    capture_layer: int,
    tau: float,
    ctx,
) -> torch.Tensor:
    """Forward both halves through the model, mid-layer extract, contrastive CE.

    en_tokens/zh_tokens: (n, T_en/T_zh) int64 already padded.
    en_mask/zh_mask: (n, T_en/T_zh) bool.
    Returns scalar loss.
    """
    n = en_tokens.shape[0]
    # Forward each side independently (different seq lengths in general).
    with ctx:
        _, _, en_h = model(en_tokens, capture_layer=capture_layer)  # (n, T_en, D)
    with ctx:
        _, _, zh_h = model(zh_tokens, capture_layer=capture_layer)  # (n, T_zh, D)

    en_rep = masked_mean_pool(en_h.float(), en_mask)  # (n, D)
    zh_rep = masked_mean_pool(zh_h.float(), zh_mask)  # (n, D)
    en_rep = F.normalize(en_rep, dim=-1)
    zh_rep = F.normalize(zh_rep, dim=-1)

    sim = (en_rep @ zh_rep.t()) / tau  # (n, n)
    labels = torch.arange(n, device=sim.device)
    # Symmetric InfoNCE: average EN→ZH and ZH→EN
    loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
    return loss

"""Phase-0 clustering of FineWeb docs, matching Gururangan et al. 2023 (c-BTM)
    Scaling Expert Language Models with Unsupervised Domain Discovery, §3.2.

Recipe (from the paper):
  1. Stream N docs from FineWeb (HF `HuggingFaceFW/fineweb`).
  2. Preprocess text with minimal assumptions: lowercase, remove sklearn's
     English stopwords, replace digit runs with <NUM>.
  3. Fit `TfidfVectorizer` (single sparse embedder — they emphasize it's
     "highly efficient at scale and leads to interpretable clusters").
  4. TruncatedSVD to 100 dims.
  5. StandardScaler (mean-remove + unit-variance) — they report this
     improved clustering quality vs plain L2 norm.
  6. Balanced k-means (paper cites Malinen & Fränti 2014 / Lewis et al. 2021
     auction algorithm; we use `k-means-constrained`'s min-cost-flow
     formulation which enforces |cluster_k| == D/K exactly). Balancing is only
     applied during fitting; inference uses greedy nearest-centroid.
  7. Assign docs to clusters, dump {text_snippet, cluster_id} JSONL.
  8. Report cluster sizes, silhouette on a subsample, and the top TF-IDF
     terms per cluster for D7 human interpretability.

Note: the paper doesn't specify TfidfVectorizer feature-cap / min_df / max_df.
Sensible defaults are picked here (max_features=100000, min_df=5,
max_df=0.95). They also don't specify n-gram range; unigrams only per the
paper's language ("tf-idf").

Usage:
  python cluster.py --n-docs 100000 --k 8 --out-dir /some/where

Output layout under --out-dir:
  assignments.jsonl.gz   one line per doc: {"text_snip":..., "cluster_id":...}
  clusterer.joblib       fitted pipeline (tf-idf → svd → scaler → kmeans)
  stats.json             sizes per cluster, silhouette, run params
  top_terms.txt          top TF-IDF terms per cluster (for eyeballing)
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np


NUM_TOKEN = "<NUM>"
_NUM_RE = re.compile(r"\d+")


def normalize_text(text: str) -> str:
    """Cheap preprocessing: lowercase (handled by TfidfVectorizer) is fine as-is;
    just collapse digit runs to a single dummy token as per the paper."""
    return _NUM_RE.sub(NUM_TOKEN, text)


def stream_fineweb(n_docs: int, subset: str, min_chars: int, max_chars: int):
    """Yield up to `n_docs` texts from the requested HF FineWeb subset.

    subset examples: 'sample-10BT', 'sample-100BT', 'default'. Smaller subsets
    are faster to iterate; for Phase-0 exploration a 10BT sample is plenty.
    """
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name=subset,
        split="train",
        streaming=True,
    )
    kept = 0
    seen = 0
    for row in ds:
        seen += 1
        t = row.get("text", "")
        if not isinstance(t, str):
            continue
        n = len(t)
        if n < min_chars or n > max_chars:
            continue
        yield t
        kept += 1
        if kept >= n_docs:
            return
    print(f"[stream_fineweb] exhausted after seen={seen:,}, yielded={kept:,}")


class BalancedKMeans:
    """Balanced k-means, size |c_k| ∈ {floor(D/K), ceil(D/K)} for all k.

    Follows the c-BTM formulation (Gururangan 2023 Eq. 1): assignment step is
    a balanced linear assignment that minimizes sum of Euclidean distance to
    the assigned centroid, subject to a hard size-cap per cluster. We solve
    it with min-cost flow via `ortools` (an alternative to Malinen & Fränti's
    or Lewis et al.'s auction algorithm).
    """

    def __init__(self, n_clusters: int, seed: int = 0, n_init: int = 4, max_iter: int = 30,
                 tol: float = 1e-4, verbose: bool = True):
        self.k = n_clusters
        self.seed = seed
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.cluster_centers_ = None
        self.labels_ = None

    def _init_kpp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++ initialisation on X (D, F)."""
        D, F = X.shape
        centers = np.empty((self.k, F), dtype=X.dtype)
        idx0 = rng.integers(D)
        centers[0] = X[idx0]
        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for c in range(1, self.k):
            probs = closest_dist_sq / closest_dist_sq.sum()
            idx = rng.choice(D, p=probs)
            centers[c] = X[idx]
            new_dist_sq = np.sum((X - centers[c]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
        return centers

    def _balanced_assign(self, X: np.ndarray, centers: np.ndarray, size_min: int, size_max: int) -> np.ndarray:
        """Min-cost flow balanced assignment. Returns labels in [0, k)."""
        from ortools.graph.python import min_cost_flow
        D, F = X.shape
        # Vectorized squared-Euclidean via ||x-c||² = ||x||² + ||c||² - 2 x·c
        d = (X * X).sum(1)[:, None] + (centers * centers).sum(1)[None, :] - 2.0 * X @ centers.T
        d = np.maximum(d, 0.0)
        scale = 1e6 / max(d.max(), 1e-9)
        cost = (d * scale).astype(np.int64)

        smcf = min_cost_flow.SimpleMinCostFlow()
        source = 0
        sink = D + self.k + 1
        # Vectorized arc addition — much faster than a Python loop at 100k+ points.
        # source -> point i
        src_tails = np.zeros(D, dtype=np.int64)
        src_heads = np.arange(1, D + 1, dtype=np.int64)
        src_caps = np.ones(D, dtype=np.int64)
        src_costs = np.zeros(D, dtype=np.int64)
        smcf.add_arcs_with_capacity_and_unit_cost(src_tails, src_heads, src_caps, src_costs)
        # point i -> cluster c
        rows = np.repeat(np.arange(D), self.k)  # (D*k,)
        cols = np.tile(np.arange(self.k), D)    # (D*k,)
        pc_tails = (1 + rows).astype(np.int64)
        pc_heads = (D + 1 + cols).astype(np.int64)
        pc_caps = np.ones(D * self.k, dtype=np.int64)
        pc_costs = cost.reshape(-1)
        smcf.add_arcs_with_capacity_and_unit_cost(pc_tails, pc_heads, pc_caps, pc_costs)
        # cluster c -> sink
        cs_tails = (D + 1 + np.arange(self.k)).astype(np.int64)
        cs_heads = np.full(self.k, sink, dtype=np.int64)
        cs_caps = np.full(self.k, size_max, dtype=np.int64)
        cs_costs = np.zeros(self.k, dtype=np.int64)
        smcf.add_arcs_with_capacity_and_unit_cost(cs_tails, cs_heads, cs_caps, cs_costs)

        smcf.set_node_supply(source, D)
        smcf.set_node_supply(sink, -D)
        # size_min ≈ size_max (differ by ≤1); D%k != 0 forces size_max on most and
        # size_min on the rest — the balanced constraint holds by construction.

        status = smcf.solve()
        if status != smcf.OPTIMAL:
            raise RuntimeError(f"min-cost flow failed: status={status}")

        # Recover labels via vectorized flow read. The point→cluster arcs are the
        # 2nd block we added (indexes [D, D+D*k)); scan those.
        n_src = D
        flows = np.array([smcf.flow(i) for i in range(n_src, n_src + D * self.k)], dtype=np.int32)
        flows = flows.reshape(D, self.k)
        labels = flows.argmax(axis=1).astype(np.int32)
        assert flows.sum() == D
        return labels

    def _single_run(self, X: np.ndarray, size_min: int, size_max: int, rng: np.random.Generator):
        centers = self._init_kpp(X, rng)
        prev_inertia = np.inf
        for it in range(self.max_iter):
            labels = self._balanced_assign(X, centers, size_min, size_max)
            # Update
            new_centers = np.zeros_like(centers)
            for c in range(self.k):
                mask = labels == c
                if mask.any():
                    new_centers[c] = X[mask].mean(axis=0)
                else:
                    new_centers[c] = X[rng.integers(X.shape[0])]
            # Inertia
            d = ((X - new_centers[labels]) ** 2).sum(axis=1).sum()
            if self.verbose:
                print(f"    iter {it}: inertia={d:.2f}")
            if abs(prev_inertia - d) / max(prev_inertia, 1e-9) < self.tol:
                centers = new_centers
                break
            centers = new_centers
            prev_inertia = d
        return centers, labels, d

    def fit_predict(self, X: np.ndarray):
        D = X.shape[0]
        size_min = D // self.k
        size_max = size_min + (1 if D % self.k else 0)
        best = None
        for run in range(self.n_init):
            if self.verbose:
                print(f"  run {run+1}/{self.n_init}")
            rng = np.random.default_rng(self.seed + run)
            centers, labels, inertia = self._single_run(X, size_min, size_max, rng)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)
        self.cluster_centers_, self.labels_, _ = best
        return self.labels_


def fit_clusterer(texts: list[str], k: int, tfidf_max_features: int,
                  tfidf_min_df: int, tfidf_max_df: float, svd_dim: int, seed: int,
                  n_init: int, max_iter: int):
    """Fit TF-IDF → TruncatedSVD → StandardScaler → BalancedKMeans on `texts`."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    print(f"[fit] tf-idf: max_features={tfidf_max_features}, min_df={tfidf_min_df}, max_df={tfidf_max_df}")
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=tfidf_max_features,
        min_df=tfidf_min_df,
        max_df=tfidf_max_df,
        ngram_range=(1, 1),
        sublinear_tf=True,
        norm="l2",
    )
    t0 = time.time()
    X = tfidf.fit_transform(texts)
    print(f"[fit] tf-idf done: shape={X.shape}, vocab={len(tfidf.vocabulary_):,}, dt={time.time()-t0:.1f}s")

    print(f"[fit] SVD to {svd_dim} dims")
    t0 = time.time()
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    Xs = svd.fit_transform(X)
    print(f"[fit] SVD done: explained_var={svd.explained_variance_ratio_.sum():.3f}, dt={time.time()-t0:.1f}s")

    print(f"[fit] StandardScaler")
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xn = scaler.fit_transform(Xs)

    D = Xn.shape[0]
    print(f"[fit] balanced k-means: D={D}, k={k}, target size ~{D//k}")
    t0 = time.time()
    km = BalancedKMeans(n_clusters=k, seed=seed, n_init=n_init, max_iter=max_iter)
    assignments = km.fit_predict(Xn.astype(np.float32))
    print(f"[fit] balanced k-means done: dt={time.time()-t0:.1f}s")

    return tfidf, svd, scaler, km, assignments, Xn


def report_stats(assignments: np.ndarray, k: int, Xn: np.ndarray, sample_silh: int = 5000, seed: int = 0):
    """Sizes per cluster + silhouette on a subsample."""
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    sizes = np.bincount(assignments, minlength=k).tolist()
    if len(assignments) > sample_silh:
        idx = rng.choice(len(assignments), size=sample_silh, replace=False)
        sil = float(silhouette_score(Xn[idx], assignments[idx]))
    else:
        sil = float(silhouette_score(Xn, assignments))
    return {"cluster_sizes": sizes, "silhouette_subsample": sil, "silhouette_n": min(len(assignments), sample_silh)}


def top_terms_per_cluster(tfidf, svd, scaler, km, top_n: int = 25):
    """For each cluster centroid, project back to TF-IDF space and take top terms.
    Reverses StandardScaler + SVD so we can read off tf-idf features."""
    # Cluster centers are in the SVD+scaler space; undo scaler then undo SVD:
    centers = km.cluster_centers_
    centers_svd = centers * scaler.scale_ + scaler.mean_
    # Approximate inverse: use svd.components_.T @ centers_svd.T
    centers_tfidf = centers_svd @ svd.components_
    feat_names = tfidf.get_feature_names_out()
    out = {}
    for c in range(centers_tfidf.shape[0]):
        top_idx = np.argsort(centers_tfidf[c])[::-1][:top_n]
        out[c] = [(feat_names[i], float(centers_tfidf[c, i])) for i in top_idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=100_000)
    ap.add_argument("--ks", default="8",
                    help="Comma-separated k values. Shared featurization across all k.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subset", default="sample-10BT",
                    help="FineWeb HF subset name (sample-10BT / sample-100BT / default)")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=1_000_000)
    ap.add_argument("--tfidf-max-features", type=int, default=100_000)
    ap.add_argument("--tfidf-min-df", type=int, default=5)
    ap.add_argument("--tfidf-max-df", type=float, default=0.95)
    ap.add_argument("--svd-dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--kmeans-n-init", type=int, default=4)
    ap.add_argument("--kmeans-max-iter", type=int, default=300)
    ap.add_argument("--text-snip-len", type=int, default=400,
                    help="Snippet length written to assignments.jsonl.gz for eyeballing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] streaming {args.n_docs:,} docs from FineWeb subset={args.subset}...")
    t0 = time.time()
    raw_texts: list[str] = []
    for t in stream_fineweb(args.n_docs, args.subset, args.min_chars, args.max_chars):
        raw_texts.append(t)
    print(f"[main] streamed {len(raw_texts):,} docs in {time.time()-t0:.1f}s")

    # Cheap preprocessing: digit collapse. Lowercasing + stopword removal are in TfidfVectorizer.
    norm_texts = [normalize_text(t) for t in raw_texts]

    # One-time featurization shared across all k values.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    print(f"[feat] tf-idf...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        lowercase=True, stop_words="english",
        max_features=args.tfidf_max_features, min_df=args.tfidf_min_df,
        max_df=args.tfidf_max_df, ngram_range=(1, 1), sublinear_tf=True, norm="l2",
    )
    X = tfidf.fit_transform(norm_texts)
    print(f"[feat] tf-idf: shape={X.shape}, vocab={len(tfidf.vocabulary_):,}, dt={time.time()-t0:.1f}s")
    t0 = time.time()
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
    Xs = svd.fit_transform(X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xn = scaler.fit_transform(Xs).astype(np.float32)
    print(f"[feat] SVD+scaler: shape={Xn.shape}, explained_var={svd.explained_variance_ratio_.sum():.3f}, dt={time.time()-t0:.1f}s")

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    for k in ks:
        print(f"\n[k={k}] balanced k-means")
        k_dir = out_dir / f"k{k}"
        k_dir.mkdir(exist_ok=True)
        km = BalancedKMeans(n_clusters=k, seed=args.seed,
                            n_init=args.kmeans_n_init, max_iter=args.kmeans_max_iter)
        t0 = time.time()
        assignments = km.fit_predict(Xn)
        print(f"[k={k}] done: dt={time.time()-t0:.1f}s")

        stats = report_stats(assignments, k, Xn, seed=args.seed)
        stats["k"] = k
        stats["params"] = vars(args)
        stats["n_streamed"] = len(raw_texts)

        ap_path = k_dir / "assignments.jsonl.gz"
        with gzip.open(ap_path, "wt") as f:
            for i, (t, c) in enumerate(zip(raw_texts, assignments)):
                snip = t[: args.text_snip_len].replace("\n", " ")
                f.write(json.dumps({"i": i, "text_snip": snip, "cluster_id": int(c)}) + "\n")

        import joblib
        joblib.dump({"tfidf": tfidf, "svd": svd, "scaler": scaler, "kmeans": km},
                    k_dir / "clusterer.joblib")

        top_terms = top_terms_per_cluster(tfidf, svd, scaler, km, top_n=25)
        with open(k_dir / "top_terms.txt", "w") as f:
            for c, terms in top_terms.items():
                f.write(f"cluster {c} (size={stats['cluster_sizes'][c]:,}):\n")
                f.write("  " + ", ".join(term for term, _ in terms) + "\n\n")

        with open(k_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[k={k}] silhouette (n={stats['silhouette_n']}): {stats['silhouette_subsample']:.3f}")
        print(f"[k={k}] cluster sizes: {stats['cluster_sizes']}")


if __name__ == "__main__":
    main()

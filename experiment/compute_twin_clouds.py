"""Layer-4 residual features for the twin-cloud figure and the phase-transition
animation.

Both figures need the same primitive: forward *the same* canonical batch under
every compartment and keep the layer-4 post-block hidden state, so that row i of
compartment j is the twin of row i of compartment 0 -- same underlying token,
different compartment vocabulary and different compartment embedding.

The canonical batch is imported from `compute_cossim_sweep`, so the geometry
drawn here is the geometry the `mean_off_diag_cossim` number is computed from.
Nothing is re-sampled.

Two extraction modes:

  --mode models      final checkpoint of each named model -> one npz each.
                     Keeps the FULL (c, B*T, D) block; the twin-cloud figure
                     fits its PCA on this.

  --mode trajectory  many named checkpoints of one run -> a single npz holding
                     (S, c, Nsub, D). Nsub tokens are a fixed subsample (same
                     rows at every step and every compartment) so a token can be
                     tracked across training. Centroids are stored separately at
                     full precision, computed over ALL B*T rows rather than the
                     subsample, because the animation's headline object is the
                     centroid path and it should not inherit subsample noise.

Run from experiment/ (paths are relative, matching the sibling eval scripts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "..")
from eval_utils import load_eval_model_from_checkpoint
from compute_cossim_sweep import load_canonical_batch, BASE_VOCAB, DEFAULT_LAYER
from _run_paths import (
    OUT_ROOT,
    NO_INFONCE_8_256_BY_C,
    INFONCE_8_256_BY_C,
    COPYEMB_8_256_BY_C,
)


# ── The three models in the headline figure ────────────────────────────────
#
# All c=8 at 8-256. They differ only in what pushes the compartments together:
# nothing, a shared initialisation, or an explicit alignment loss.
MODELS = {
    # Default init, no alignment pressure. Ends at mean off-diag cossim ~0.13.
    "default": NO_INFONCE_8_256_BY_C[8],
    # Compartment embeddings + LM head copied from the c=1 baseline at init.
    "copyemb": COPYEMB_8_256_BY_C[8],
    # InfoNCE alignment at layer 4, lambda=1. Ends at ~0.96.
    "infonce": INFONCE_8_256_BY_C[8],
}

# ── The phase-transition pair ──────────────────────────────────────────────
#
# c=8, absolute-mode tr, wd=0, 8-256. tr=0.25 stays compartmentalised
# (final cossim 0.257); tr=0.5 breaks (0.919). Same architecture, same seed,
# same everything except the translation ratio.
TRAJECTORIES = {
    "tr025": ("bpe16384-rope-8-256/93c21853_s64", 0.25),
    "tr05": ("bpe16384-rope-8-256/8143ba31_s64", 0.5),
}


def list_named_steps(run_dir: Path) -> list[int]:
    """Named `step-NNNNNN` checkpoints that actually carry a model.pt."""
    out = []
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return out
    for d in ckpt_root.iterdir():
        if d.name.startswith("step-") and (d / "model.pt").exists():
            try:
                out.append(int(d.name.split("-")[1]))
            except ValueError:
                pass
    return sorted(out)


def log_spaced_subset(steps: list[int], n: int) -> list[int]:
    """Pick ~n steps spread evenly in log(step).

    The interesting motion is early -- the tr=0.5 run does most of its
    collapsing well before 100k -- so a linear subsample would spend most of
    its frames on a static endgame.
    """
    if len(steps) <= n:
        return steps
    arr = np.array(steps, dtype=float)
    targets = np.exp(np.linspace(np.log(arr[0]), np.log(arr[-1]), n))
    keep = sorted({steps[int(np.argmin(np.abs(arr - t)))] for t in targets})
    return keep


@torch.no_grad()
def features_for_ckpt(
    run_dir: Path,
    ckpt_dir: Path,
    batch_cpu: torch.Tensor,
    c: int,
    device: str,
    layer: int,
) -> np.ndarray:
    """(c, B*T, D) float32 layer-`layer` residuals, one block per compartment.

    Row ordering is identical across compartments, which is what makes the
    twin lines in the figure meaningful.
    """
    model, _, _ = load_eval_model_from_checkpoint(ckpt_dir, run_dir, device)
    model.eval()
    batch = batch_cpu.to(device)
    feats = []
    for ci in range(c):
        x = batch + ci * BASE_VOCAB
        cid = torch.full_like(batch, ci)
        out = model(x, compartment_ids=cid, capture_layer=layer)
        if not (isinstance(out, tuple) and len(out) == 3):
            raise RuntimeError(f"capture_layer did not yield a hidden state: {type(out)}")
        _, _, h = out
        feats.append(h.flatten(0, 1).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.stack(feats, axis=0)


def run_dir_for(rel: str) -> Path:
    return OUT_ROOT / rel


def compartments_of(run_dir: Path) -> int:
    cfg = json.loads((run_dir / "meta" / "config.json").read_text())
    return int(cfg["experiment"]["n_compartments"])


def final_ckpt(run_dir: Path) -> Path:
    """Prefer the rolling checkpoint, then `latest`, then the last named step."""
    for name in ("_rolling", "latest"):
        d = run_dir / "checkpoints" / name
        if (d / "model.pt").exists():
            return d
    steps = list_named_steps(run_dir)
    if not steps:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    return run_dir / "checkpoints" / f"step-{steps[-1]:06d}"


def do_models(args, batch_cpu: torch.Tensor, outdir: Path):
    for name, rel in MODELS.items():
        dest = outdir / f"clouds_{name}_L{args.layer}.npz"
        if dest.exists() and not args.overwrite:
            print(f"  {name}: cached -> {dest.name}")
            continue
        run_dir = run_dir_for(rel)
        if not run_dir.exists():
            print(f"  {name}: MISSING {run_dir}")
            continue
        c = compartments_of(run_dir)
        ckpt = final_ckpt(run_dir)
        print(f"  {name}: c={c} ckpt={ckpt.name}")
        feats = features_for_ckpt(run_dir, ckpt, batch_cpu, c, args.device, args.layer)
        np.savez_compressed(
            dest,
            feats=feats.astype(np.float32),
            c=c,
            layer=args.layer,
            run=rel,
            ckpt=ckpt.name,
            batch_seed=args.seed,
        )
        print(f"    wrote {dest.name}  feats={feats.shape}")


def do_trajectory(args, batch_cpu: torch.Tensor, outdir: Path):
    which = args.traj.split(",") if args.traj else list(TRAJECTORIES)
    for name in which:
        rel, tr = TRAJECTORIES[name]
        dest = outdir / f"traj_{name}_L{args.layer}.npz"
        run_dir = run_dir_for(rel)
        if not run_dir.exists():
            print(f"  {name}: MISSING {run_dir}")
            continue
        c = compartments_of(run_dir)
        steps = log_spaced_subset(list_named_steps(run_dir), args.n_frames)
        print(f"  {name}: c={c} tr={tr} -> {len(steps)} frames "
              f"[{steps[0]} .. {steps[-1]}]")

        # Fixed token subsample: same rows at every step, every compartment.
        n_tokens = batch_cpu.numel()
        rng = np.random.Generator(np.random.PCG64(args.sub_seed))
        sub = np.sort(rng.choice(n_tokens, size=min(args.n_sub, n_tokens),
                                 replace=False))

        # Resume: keep whatever frames a previous run already produced.
        done: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if dest.exists() and not args.overwrite:
            prev = np.load(dest, allow_pickle=False)
            if np.array_equal(prev["sub_idx"], sub):
                for i, s in enumerate(prev["steps"].tolist()):
                    done[int(s)] = (prev["sub_feats"][i], prev["centroids"][i])
                print(f"    resuming with {len(done)} cached frames")

        sub_feats, centroids, kept = [], [], []
        for i, step in enumerate(steps):
            if step in done:
                sf, cen = done[step]
            else:
                ckpt = run_dir / "checkpoints" / f"step-{step:06d}"
                try:
                    f = features_for_ckpt(run_dir, ckpt, batch_cpu, c,
                                          args.device, args.layer)
                except Exception as exc:
                    print(f"    [{i+1}/{len(steps)}] step {step} FAILED: {exc!r}")
                    continue
                # Centroid over every row, subsample only for the cloud.
                cen = f.mean(axis=1).astype(np.float32)      # (c, D)
                sf = f[:, sub, :].astype(np.float16)          # (c, Nsub, D)
                print(f"    [{i+1}/{len(steps)}] step {step:>8d} ok")
            sub_feats.append(sf)
            centroids.append(cen)
            kept.append(step)

            if (i + 1) % 10 == 0 or i == len(steps) - 1:
                np.savez_compressed(
                    dest,
                    steps=np.array(kept),
                    sub_feats=np.stack(sub_feats),
                    centroids=np.stack(centroids),
                    sub_idx=sub,
                    c=c, tr=tr, layer=args.layer, run=rel,
                    batch_seed=args.seed,
                )
        print(f"    wrote {dest.name}  frames={len(kept)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["models", "trajectory"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--seed", type=int, default=0,
                    help="canonical-batch seed (0 = the one cossim_sweep uses)")
    ap.add_argument("--outdir", default="../cache_twinclouds")
    ap.add_argument("--overwrite", action="store_true")
    # trajectory-only
    ap.add_argument("--traj", default="", help="comma-separated keys of TRAJECTORIES")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--n-sub", type=int, default=1024,
                    help="tokens kept per compartment per frame")
    ap.add_argument("--sub-seed", type=int, default=7)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    batch_cpu = load_canonical_batch(seed=args.seed)
    print(f"canonical batch {tuple(batch_cpu.shape)} seed={args.seed} layer={args.layer}")

    if args.mode == "models":
        do_models(args, batch_cpu, outdir)
    else:
        do_trajectory(args, batch_cpu, outdir)


if __name__ == "__main__":
    main()

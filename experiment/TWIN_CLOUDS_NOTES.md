# Twin clouds and the phase-transition trajectory

Two visualisations of layer-4 compartment geometry, plus the extraction they
share. Everything uses the **same canonical 64×64 batch** that
`compute_cossim_sweep.py` uses for the `mean_off_diag_cossim` metric, so the
pictures and the number in the paper are computed from identical activations.

## Scripts

| script | what it does |
|---|---|
| `compute_twin_clouds.py --mode models` | final-checkpoint layer-4 residuals for the three c=8 headline models → `cache_twinclouds/clouds_*.npz` |
| `compute_twin_clouds.py --mode trajectory` | layer-4 residuals at every named checkpoint of the two phase-transition runs → `cache_twinclouds/traj_*.npz` |
| `plot_twin_clouds.py` | `figures/twin_clouds_2d.*` and `figures/twin_clouds_book.*` |
| `plot_centroid_trajectory.py` | `figures/centroid_paths.*`, and with `--animate` the GIF |

All are run from `experiment/`, matching the sibling eval scripts.

## Figure 1 — twin clouds

Forward the same 4096 tokens under all 8 compartments, keep the layer-4
post-block residual, fit one PCA on the union, project everything into it, and
join each token in compartment 0 to its twin in compartment *j*.

Runs (all c=8, 8-256, from `_run_paths.py`):

- **default** — `NO_INFONCE_8_256_BY_C[8]`, no alignment pressure
- **init-copy** — `COPYEMB_8_256_BY_C[8]`, embeddings + LM head copied from the
  c=1 baseline at init
- **InfoNCE** — `INFONCE_8_256_BY_C[8]`, explicit layer-4 alignment, λ=1

| | cossim | cossim, identity removed | twin dist / cloud radius | identity share of variance |
|---|---|---|---|---|
| default   | 0.132 | **0.057** | 1.37× | 19.4% |
| init-copy | 0.981 | **0.996** | 0.20× | 1.5% |
| InfoNCE   | 0.963 | **0.984** | 0.25× | 2.4% |

Sanity check against the published caches: InfoNCE 0.9631 here vs 0.9633 in
`cossim_across_training.json`; default 0.1321 vs 0.1337. The residual difference
is `_rolling` sitting at a slightly different step from the last named
checkpoint.

### The compartment-embedding confound, and why it became the story

These models set `use_compartment_embeddings=True`, so every compartment carries
a constant additive offset. A naive projection shows *c* separated blobs whether
or not the content is shared, and "the clouds are apart" on its own is not
evidence of compartmentalisation.

Splitting `h = g + (mu_j - g) + (h - mu_j)` into global, identity and content
settles it, and the answer runs opposite to the worry:

- Removing identity makes the **default** model look *less* aligned
  (0.132 → 0.057). The similarity that was there was partly the shared offset;
  the content underneath is essentially unrelated.
- Removing identity pushes the **unified** models *up* (0.981 → 0.996,
  0.963 → 0.984). The constant offset was the only thing still separating them.

So the confound does not manufacture the effect — it slightly masks it.

### Why the identity axis is not a single learned direction

With *c* compartments the identity subspace is (c−1)-dimensional, and here it is
close to isotropic: its leading PC holds only 17–21% of the identity spread. A
single learned ID axis therefore cannot separate 8 sheets, and individual tokens
smear across all of them.

The decomposition fixes this without fudging anything: the identity component is
*by construction* constant within a compartment, so each sheet has exactly one
identity coordinate and zero thickness along that axis. `twin_clouds_book`
therefore shows

- **top** — one sheet per compartment at evenly spaced slots ordered by identity
  coordinate, with a thread per token running through consecutive sheets.
  Vertical threads mean the pages are congruent; slanted threads mean they are
  not.
- **bottom** — where those sheets actually sit along the identity axis, on one
  scale shared by all three panels. This is where the collapse is visible: the
  default model's sheets spread across the axis, the unified models' sheets
  land on top of each other.

Slots are a display choice for the top row only, and the bottom row is there so
the real spacing is never hidden.

## Figure 2 — the phase transition as motion

Two c=8 runs at 8-256, wd=0, absolute-mode tr, differing only in translation
ratio:

- `bpe16384-rope-8-256/93c21853_s64` — tr=0.25, final cossim 0.257 (stuck)
- `bpe16384-rope-8-256/8143ba31_s64` — tr=0.50, final cossim 0.919 (breaks)

Every named checkpoint is forwarded on the canonical batch; the eight
compartment centroids are projected into one fixed basis and their paths drawn,
with the formal val loss and the cossim metric underneath on a shared step axis.

Results (106 checkpoints each, per-checkpoint RMS-normalised):

| | cossim 100 → 1M | centroid spread 100 → 1M | RMS norm |
|---|---|---|---|
| tr=0.25 | 0.087 → 0.259 | 0.775 → 0.397 | 1.39 → 2.79 |
| tr=0.50 | 0.119 → **0.918** | 0.757 → **0.131** | 1.40 → 2.70 |

Both runs' centroids contract somewhat; the tr=0.5 run contracts three times as
far and its cossim climbs to 0.92. The val-loss curves are on top of each other
until ~60k and separate over 60k–300k, which is exactly the window in which the
tr=0.5 cossim goes 0.36 → 0.78.

A caveat worth stating in any caption: the 2-D projection holds only ~30% of the
centroid variance (18%/15% and 14%/12%). The centroid configuration lives in the
same near-isotropic (c−1)-dimensional identity subspace as in figure 1, so no
2-D view can hold most of it. The `spread` readout is computed in the full
256-dimensional space, not in the projection, so the quantitative claim does not
depend on the choice of view.

### Three things the script is careful about

**Fixed basis.** Refitting PCA per checkpoint rotates the basis and manufactures
motion. The basis is fit once. `--basis` selects what on:

- `centroids-all` (default) — one fixed basis, fit on the centroids of every
  frame, so it is guaranteed to span the directions the centroids travel in.
- `final` — the fit-on-the-last-checkpoint convention. Worth knowing that for
  the tr=0.5 run the last checkpoint *is* the collapsed configuration, so this
  basis is fit on the small residual spread that survives the collapse.
- `extremes` — first and last frame together, the usual compromise.
- `tokens-final` — PCA on token residuals rather than centroids.

**Scale.** RMS activation norm grows ~4.5× over training (0.6 → 2.7). Left
alone, that radial blow-up dominates the animation and every centroid marches
outward regardless of what the compartments do. Each checkpoint is divided by
its own global RMS first.

**Val loss.** Cross-tr comparisons use `loss_compartment_i` from
`val_metrics.json` — the formal per-compartment eval on a fixed monolingual
set — never the training-time val loss, which mixes in translation pairs and is
biased downward by an amount that scales with tr, i.e. exactly the axis being
compared. The c=1 baseline is drawn as the floor.

## Cache

`cache_twinclouds/` holds the extracted activations (~1 GB). It is derived data
and is not committed; regenerate with the two `compute_twin_clouds.py` calls.

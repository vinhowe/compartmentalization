# Experimental protocol (post-NeurIPS)

Status: adopted 2026-08-27. Supersedes the ad-hoc recipe used for the NeurIPS
submission. Everything trained from this date forward follows this document, and
any deviation must be recorded in the run's config comment.

Background reasoning that produced the *training recipe* lives in
`paper/reviews/neurips_redesign_plan.md` (gitignored, local only). This document
is the normative version: the plan explains why, this states what.

---

## 1. Why this document exists

The published runs cannot be reproduced from their configs alone. They used one
learning rate (2e-5) across widths 32 to 1792, an undisclosed dedup policy, a
budget (131B) that was an artifact of `1e6 steps x 2048 x 64` rather than a
choice, and a 64-token context that made 30% of HellaSwag unrepresentable. Each
of those was defensible in isolation and indefensible as a set.

The rules below are chosen so that a run's config plus this document fully
determine the experiment.

---

## 2. The scaling ladder

Six rungs. `d_head = 64` throughout, aspect ratio `d_model / n_layer = 128`
held **constant**, so the ladder is one-dimensional and `N_trunk` scales as
`d^3`.

| rung | n_layer | n_embd | n_head | trunk N | total c=1 | total c=8 | emb% c=8 |
|------|---------|--------|--------|---------|-----------|-----------|----------|
| R1   | 4       | 512    | 8      | 12.6M   | 29.4M     | 146.8M    | 91%      |
| R2   | 5       | 640    | 10     | 24.6M   | 45.6M     | 192.4M    | 87%      |
| R3   | 6       | 768    | 12     | 42.5M   | 67.6M     | 243.8M    | 83%      |
| R4   | 8       | 1024   | 16     | 100.7M  | 134.2M    | 369.1M    | 73%      |
| R5   | 12      | 1536   | 24     | 339.7M  | 390.1M    | 742.4M    | 54%      |
| R6   | 16      | 2048   | 32     | 805.3M  | 872.4M    | 1342.2M   | 40%      |

Span: 1.81 decades, mean step 2.4x in `N_trunk`.

Parameter counts are exact, not estimates:

```
N_trunk  = 12 * n_layer * n_embd^2           # attn 4d^2 + mlp 8d^2, bias=False
N_embed  = 2 * n_embd * (16384 * c + 1)      # wte + lm_head, UNTIED
           2 * n_embd * 16384                #   ... at c=1
N_total  = N_trunk + N_embed
```

Verified against `redesign-1b-base.toml`, which independently states 872M at
c=1 and 1342M at c=8 for R6. Both reproduce exactly.

### Why these rungs, and not others

- **`d_head = 64`.** R6 already uses it, it matches Pythia / OLMo / Llama, and at
  `T=1024` the legacy `d_head=32` doubles head count for a worse attention/FLOP
  balance.
- **Constant aspect ratio.** Shape is a known confound in scaling fits. Loss is
  insensitive to shape *within* a band; 43 vs 128 is outside it. Holding `d/L`
  fixed is what makes `N_trunk` a sufficient statistic for the rung.
- **R6 fixed at 16x2048.** Already committed in `redesign-1b-base.toml`.
- **R1 floors at L=4.** Constant aspect below `d=512` gives `L<4`, where depth is
  degenerate.
- **Spacing is uneven** (1.95 / 1.73 / 2.37 / 3.38 / 2.37x). This is the price of
  integer `n_layer` and `n_embd` divisible by 128. Even spacing is not required
  for a power-law fit; coverage and per-point precision are.

A 7-rung variant reaching 4.7M was rejected: it buys 0.4 more decades only by
breaking constant shape at the three rungs where the embedding confound (below)
is already worst, stacking two confounds at the same end of the fit.

---

## 3. N means trunk parameters. This is load-bearing.

**Every power law is fit against `N_trunk`, never `N_total`.**

At c=8 the model is 40-91% embedding table, and the fraction moves violently
across the ladder. The `c=8 / c=1` total-size multiplier is 5.00x at R1 and
1.54x at R6. So "turn on c=8" is a five-fold parameter perturbation at the
bottom rung and a 1.5-fold one at the top.

A fit of gap-versus-`N_total` therefore mixes *compartmentalization* with
*relative embedding overhead shrinking with scale*. It will fit beautifully and
mean nothing.

`N_trunk` is identical for c=1 and c=8 at a given rung, which is exactly the
property that makes the comparison well posed.

### Mandatory third arm

Every rung runs **three** conditions, not two:

| arm | vocab rows | purpose |
|-----|-----------|---------|
| `c1`        | 16384    | baseline |
| `c8`        | 131073   | treatment |
| `c1-padded` | 131073   | parameter-matched control (rows allocated, unused) |

The gap then decomposes:

```
gap_total   = L(c8)        - L(c1)          # what a naive comparison reports
gap_params  = L(c1-padded) - L(c1)          # cost of the parameters alone
gap_compart = L(c8)        - L(c1-padded)   # <- the quantity we fit
```

Only `gap_compart` is fit as a power law. Report all three. The padded-vocab
mechanism already exists in the repo (`config/bio-cap-*-padded-vocab-*.toml`).

The embedding fraction is a **reported covariate** on every figure that plots a
per-rung quantity. It is not optional and not a footnote.

---

## 4. Learning rate

### The rule

**Tune per condition. Hold the tuning *protocol* constant, not the LR value.**

A shared LR is not a neutral control. It is an arbitrary slice through the
surface `L(c, LR)` that silently favours whichever condition's optimum sits
nearer the chosen value. The quantity worth reporting is

```
gap* = min_LR L(c=8, LR) - min_LR L(c=1, LR)
```

with both minima taken over the **same grid, at the same budget, fixed in
advance**. What actually destroys comparability is unequal *search effort* -
"c=1 at the default, c=8 after twenty hand-tuned tries" - and that is defused by
publishing the whole sweep rather than only the argmin.

Precedent: muP (Tensor Programs V) exists precisely because optimal LR shifts
with width, making single-LR comparisons across width invalid; Chinchilla's
central correction to Kaplan was that a hyperparameter held fixed across
conditions needing different values yields a *biased*, not a controlled,
comparison; Kaplan used size-dependent LR; MoE scaling work tunes dense and
sparse separately. The cautionary case (Narang et al., "Do Transformer
Modifications Transfer?") is about unequal effort, not unequal values.

Here a shift is *predicted*, not hypothetical: at c=8 each embedding row is
updated roughly 1/8 as often, so the embedding block sees sparse Adam updates
and a different effective step size.

**Always report the gap twice**: at matched LR, and at per-condition optimum.
Agreement retires the objection in one sentence. Disagreement means part of the
measured gap is a tuning artifact, which must be known before it is published.

### The rule the ladder actually uses: the Pythia size schedule

**The ladder's learning rate comes from published practice, not from our own
argmin.** A power law fit to the Pythia schedule (Biderman et al. 2023 Tab. 1,
which explicitly adopts GPT-3's Brown et al. 2020 Tab. 2.1):

```
log10 LR = 0.171 - 0.415 * log10(N_total)        rms 0.070 decades
```

evaluated at each rung's TOTAL parameter count at c=1 -- total, because that is
how both published tables report size. Note this is a different quantity from
the one the loss-gap power law is fit against, which remains trunk N.

| rung | total N (c=1) | LR | nearest published |
|------|--------------|-----|-------------------|
| R1 | 29.4M  | 1.2e-3 | Pythia-70M  1.0e-3 |
| R2 | 45.6M  | 9.8e-4 | -- |
| R3 | 67.6M  | 8.3e-4 | Pythia-70M  1.0e-3 |
| R4 | 134.2M | 6.3e-4 | Pythia-160M 6.0e-4 |
| R5 | 390.1M | 4.0e-4 | Pythia-410M 3.0e-4 |
| R6 | 872.4M | 2.9e-4 | Pythia-1B   3.0e-4 |

**One LR per rung, shared by all three arms.** The sweep's strongest single
result is that the optimum does not move with c: identical at R1 and R3, and a
0.0005-nat tie at R4. A per-arm LR would inject tuning noise into the very
difference being measured, for no measured benefit. It also makes the headline
comparison matched-LR by construction.

**Why the published rule rather than our measured argmin.** The sweep tunes at
3B tokens for a ladder that trains to 30B. A short horizon systematically
prefers a higher LR, and our sweep duly returns a FLAT optimum of 1e-3 at every
width -- against a published exponent of -0.31 (GPT-3) to -0.41 (Pythia). A
flat rule is the signature of the tuning budget, not a property of the models.
Pythia is also exactly this design (one fixed token budget across a ladder) and
is the reference class the paper wants to sit in.

The sweep is not wasted: it CONFIRMS the rule where it measured. R1 and R3
argmins are 1e-3 against predicted 1.2e-3 and 8.3e-4; R4's two statistically
tied values, 3e-4 and 1e-3 at 0.0005 nats apart, bracket the predicted 6.3e-4.
It also establishes the c-invariance the shared-LR choice rests on, and shows
the gap is identical at flat LR and at per-cell optima (0.4945/0.6040/0.5008
versus 0.4945/0.6040/0.5013) -- which retires the tuning-artifact objection
directly.

R6 is corroborated three ways: this fit gives 2.9e-4, Pythia-1B used 3.0e-4,
OLMo-1B used 4.0e-4, and our own 16x2048 runs on ORC trained stably to 100B at
4.0e-4.

**muP was considered and rejected.** It indexes its scaling rules by WIDTH,
while compartmentalization multiplies the VOCABULARY; it therefore prescribes
nothing for the layer that is 73-91% of the c=8 model and that IS the
manipulation under test. Adopting it would import untested scaling assumptions
onto the experimental axis. It is also the minority choice in open releases --
Pythia, OLMo, Llama and SmolLM2 all use standard parametrization with a
size-dependent LR, and those are the models this work wants to be compared
against. Cerebras-GPT and MiniCPM are the notable muP adopters.

### Picking the grid

You do not pick an LR. You pick a bracket wide enough that the argmin lands
strictly interior, and let the sweep locate it. An argmin on a grid edge is not
a result; it is an instruction to sweep again.

Published anchors at our widths (comparable warmup+decay recipes):

| d_model | reference | LR |
|---------|-----------|-----|
| 512  | Pythia-70M           | ~1e-3 |
| 768  | Pythia-160M          | ~6e-4 |
| 1024 | Pythia-410M          | ~3e-4 |
| 2048 | Pythia-1B / OLMo-1B  | ~3e-4 / 4e-4 |

Adopted grid: **1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2**, half-decade steps. This
places the expected optima interior with a half-decade of margin at both swept
widths. Divergence at the top of the grid is an informative bracket, not a
failed run.

### Why LR also varies across RUNGS, not just across c

The same argument applies along the ladder, and the failure mode there is worse
because it is monotone in the quantity being fitted.

A scaling law fits an ENVELOPE, `L*(N) = min_theta L(N, theta)`, not the loss of
one fixed recipe evaluated at each `N`. Optimal LR falls with width (roughly
`1/d` under muP), so a single LR across the ladder is off-optimum by an amount
that grows with distance from whichever rung it suited:

- tuned for `d=2048`: the `d=512` rungs train ~4x under-LR, look artificially
  bad, and the fitted curve is artificially STEEP -> `alpha_N` overestimated.
- tuned for `d=512`: the large rungs are unstable, the curve is artificially
  FLAT -> `alpha_N` underestimated.

Either way a systematic, monotone, size-dependent error lands directly in the
exponent. A constant LR is not a control here; it is a confound that scales
with N.

This is universal practice, not a liberty we are taking. GPT-3 spans 6e-4 to
0.6e-4 across its sizes; Pythia spans 1e-3 to 1.2e-4; OLMo, SmolLM2, Llama and
Kaplan are all size-dependent. Chinchilla's central correction to Kaplan was
caused by a hyperparameter that was NOT adapted per run (cosine cycle length)
biasing the exponents - the exact failure mode of "hold it constant so it stays
comparable".

**Val loss stays comparable because LR changes how close a model got to its
best, not what was measured.** Comparability comes from the tokenizer, corpus,
held-out set and context length, all of which are pinned. LR is a training
knob, not a measurement knob.

What genuinely breaks comparability is unequal tuning EFFORT. If the top rung
gets twenty trials and the bottom rung two, the envelope tilts and the exponent
is meaningless. The guard is the one stated above: identical grid, identical
budget, identical selection rule at every rung, and publish the whole sweep.

Structural mitigation specific to this project: the headline quantity is a
DIFFERENCE at fixed rung, `L(c8) - L(c1-padded)`, with both arms tuned by the
same protocol. Residual tuning slop largely cancels in the difference, so the
gap is materially more robust to imperfect LR than the absolute curve is.

### The sweep that sets the rule

Swept at **two** widths, not one, because the ladder needs the *slope* of
optimal LR versus width as much as it needs its dependence on `c`:

```
widths  R1 (4x512), R3 (6x768), R4 (8x1024)
arms    c1 and c8 at every swept width; c1-padded at R1 only
LR      the 5-point grid above
budget  3B tokens, full WSD including the decay
        -> 35 runs, ~125 GPU-hours (measured, compile on)
```

Measured per-run cost on one A100 with `torch.compile` enabled, which is 2.4x
faster than the same model without it -- do not size a sweep from an
uncompiled smoke test:

| arm | s/iter | h/run |
|---|---|---|
| r1-c1 | 1.7 | 0.67 |
| r1-c8, r1-c1-padded | 7.6 | 3.03 |
| r3-c1 | 3.9 | 1.55 |
| r3-c8 | 12.8 | 5.09 |
| r4-c1 | 9.5 | 3.78 |
| r4-c8 | 19.5 | 7.75 |

**Three widths, not two.** Two points fit a line with zero residual, so they
cannot falsify the rule they produce -- and that rule is then extrapolated 2x
beyond the swept range to reach R6. R3 is interpolation, which is precisely
what validates the functional form: if R1/R3/R4 are collinear in log-log,
extrapolating to R5/R6 is justified; if they are not, that is known before the
expensive rungs launch. Sweeping R6 directly costs 581 GPU-hours -- more than
the R6 ladder rung itself -- so R6 is anchored against published values instead.

**c1-padded is swept, not assumed equal to c1.** It is tempting to argue its
extra rows are dead and so its optimum matches c1's. That is wrong at the head:
the unused INPUT embedding rows are indeed dead, but every one of the 131073
OUTPUT rows receives gradient through the softmax denominator, because
`dL/dlogit_j = p_j - y_j` pushes unused rows down at every step. Its optimizer
dynamics therefore genuinely differ from c1. Swept at R1, where the wide vocab
is cheapest.

The decay is included because LR optima measured on non-annealed constant-LR
runs sit at a different place than annealed ones, and the ladder is annealed.

Two rungs are swept and `LR(d)` is interpolated to the other four. Two known
weaknesses, both to be checked rather than assumed away:

- **Extrapolating up to R6** (`d=2048`) is the weak link. R6 has published
  anchors - Pythia-1B at ~3e-4, OLMo-1B at ~4e-4. If the fitted rule predicts a
  value far outside that band, add a third sweep point at R6 rather than trust
  the extrapolation.
- **Short-horizon bias**: a 3B sweep systematically prefers a slightly HIGHER LR
  than a 30B run would. The R4 rung at 30B is the check, since R4 is swept
  directly; if its 30B behaviour disagrees with the 3B argmin, the rule is
  refit before the expensive rungs launch.

Cost context: the sweep is ~4% of the 356-hour ladder it aims. Run it first.

---

## 5. Fixed recipe

Everything below is constant across the ladder unless a config says otherwise
and explains why.

| axis | value | note |
|------|-------|------|
| context | 1024 | needs no retokenization; bins are flat uint16, `block_size` slices at load |
| schedule | WSD | stable phase extendable; decays fork off it |
| warmup | 300 steps, **absolute** | never a fraction of budget - that is what makes one trunk shareable across 30B/100B/300B |
| min_lr | peak / 10 | inherited nanoGPT 6e-5 is ABOVE our peak and would invert the decay |
| batch | 2.097M tokens/step | 2048 seq x 1024, inside the 1-4M band of GPT-3 1.3B / Pythia-1B / OLMo-1B |
| budget | 30B grid, >=131B at R6 | see below |
| weight decay | **0** | not convention -- nonzero wd manufactures a c-dependent penalty. See 5.1 |
| dtype | bfloat16 | FP8 rejected: per-tensor scaling on a projection 16k wide at c=1 and 131k at c=8 puts numerics on the experimental axis |
| data | `fineweb350B-bpe16384-nodedup` | with a manifest; dedup is a measured condition, never an undisclosed default |
| `auto_batch_config` | **false** | presets would retune batch/accum for VRAM and silently change tokens-per-step out from under an exact budget |

### 5.1 Weight decay is 0, and that is a measurement requirement

Every published ladder holds weight decay CONSTANT across sizes and does not
tune it per rung: Pythia 0.01 across 70M-12B, GPT-3 / OLMo / Llama / SmolLM2 0.1.
Only the learning rate varies. That asymmetry is principled rather than lazy:
in AdamW the decay update is `w <- w - lr*wd*w`, so LR and WD enter
MULTIPLICATIVELY and the product sets the decay timescale. Tuning LR already
sweeps effective decay, which makes a 2-D sweep largely redundant. (The honest
tension: constant WD with per-rung LR does let effective decay drift across the
ladder. The field tolerates this and it has not been shown to matter at these
scales.)

We use **wd = 0**, and here that is not a stylistic departure -- it removes a
confound that would sit directly on the experimental axis.

`configure_optimizers` decays every parameter with `dim() >= 2`, which includes
BOTH embedding tables (verified: 146,805,760 of the 146.81M parameters in an
R1-c8 model are in the decayed group). AdamW applies decay on every step to
every parameter in that group, **regardless of whether it received gradient**.

At c=8 each compartment's embedding rows receive gradient on roughly 1/8 of
steps but are decayed on 8/8. A c=8 row is therefore shrunk about 8x harder per
unit of gradient signal than the corresponding c=1 row. Nonzero weight decay
would manufacture a compartmentalization penalty OUT OF THE OPTIMIZER, with the
same shape as the effect being measured and no way to separate the two after
the fact.

So the ordering of the argument matters when this is written up: wd=0 is not
"we happened to study wd separately", it is "nonzero wd biases the c comparison
by construction". That is the answer to a reviewer asking why we sit at 0 when
the field sits at 0.1.

Worth measuring rather than asserting: two runs at R1 (c=1 and c=8) at wd=0.1
and the swept-optimal LR, against the wd=0 runs the sweep already produces,
exhibit the differential penalty directly for ~4 GPU-hours.

### Budgets are specified in TOKENS

`max_iters` is **derived**, never chosen. At 2,097,152 tokens/step:

```
30B  ->  14,305 iters
100B ->  47,684 iters
```

The legacy 131B was never a decision - it is `1e6 steps x 2048 x 64`. At
`T=1024` every inherited `max_iters` is wrong by 16x.

Checkpoints are logged in **tokens**, not steps, or nothing is comparable
across the batch and context change.

### Fixed budget across the ladder

All six rungs train to 30B. This is the Pythia design, and it is the right one
here: at 30B, R6 sees 37 tokens/param (near compute-optimal) while R1 sees 2384
(deeply data-rich). So the small rungs are cleanly **capacity-limited**, which
is the regime where `L(N)` is a clean power law.

The risk is at the *top*, not the bottom: a rung that becomes data-limited
flattens and bends the fit. R6 is therefore extended to >=131B, which WSD makes
nearly free because the stable phase has no horizon baked in.

**Log-spaced checkpoints give the `L(N, D)` surface for free.** Six rungs x
~8 checkpoints each is ~48 `(N, D)` points from one set of runs - enough to fit
`alpha_N` and `alpha_D` jointly rather than assuming one.

### Assignment horizon

For `c > 1`, pin `assignment_horizon_examples` to the **longest** budget the
trunk will ever reach (292,968,448 for 300B). Deriving it from `max_iters` -
the default - re-randomises every example's compartment when a run is extended.
This already invalidated one 1B c=8 run. `c=1` is immune.

---

## 6. Checkpoints

```
myrun/
  checkpoints/
    _rolling/                  # crash recovery, full state, overwritten
    branch/tok-000027000M/     # FULL state: model + optimizer + dataloader
    traj/tok-*/                # weights only - evaluation points
  lineage.json

myrun@anneal-000030000M/       # decay child: own run, own lock, own _rolling
  checkpoints/
    annealed/tok-000030000M/   # weights only, terminal
  lineage.json
```

**Resume accepts only `_rolling/` and `branch/*`.** Both always carry
`optimizer.pt`, which makes the Adam-reset bug structurally impossible instead
of merely guarded against.

**Over-save branch points.** A branch point not taken cannot be reconstructed,
and optimizer state is the part that cannot be recovered. Two per run is ~170GB
for the whole grid against terabytes free.

---

## 7. The complexity ladder (n-gram auto-unification)

The open question this protocol is built to answer: **does the compartmentalization
gap close on its own for data simple enough to be unified?**

Real English does not auto-unify - the gap persists. The hypothesis is that a
sufficiently simple generating process does, and that the gap is a function of
*data complexity*, not just of scale.

Materials already exist:

- `data/ngram-tables-bpe16384/` - unigram, 2, 3, 4-gram tables estimated on 1B
  tokens of the SAME corpus with the SAME BPE-16384 tokenizer the models train
  on. Backoff is recursive with no discounting. Fractional orders (1.5, 2.5,
  3.5) via Jelinek-Mercer at lambda=0.5.
- `config/8-256-c2-english-vs-ngram*.toml` - the c=2 rungs at the OLD recipe
  (T=64, LR 2e-5). Six complexity rungs: 1.5, 2, 2.5, 3, 3.5, 4.
- `config/ngram_seeds/` - seed replicates at orders 2, 3, 4.

Known comparators at 200k steps, versus c=1 English:

```
uniform (i.i.d.)   +0.048 nats
unigram (n=1)      +0.047
Russian (real)     +0.229     <- the ceiling this ladder interpolates toward
```

These must be **re-run under this protocol** before they can be combined with
new results; they are T=64 at LR 2e-5 and are not comparable to anything below.

Design note: the complexity ladder is a *second axis*, crossed with the scaling
ladder. Run it at a single rung first (R4 is the natural choice - large enough
that the trunk is not embedding-dominated, cheap enough to replicate) and only
extend to a full cross if the effect is present.

---

## 8. Config rules

- **Schema v2 required** for all new configs (`config_version = 2`). v2 rejects
  `permute_tokens_per_compartment`, `permute_input_tokens_per_compartment`,
  `translation_ratio_mode` and `assignment_seed` rather than ignoring them.
- **`tr` always means the effective (absolute) translation ratio.** v2 is always
  absolute. Legacy compartment-mode values convert as `t = raw / (n + raw)` -
  **not** `raw / (n + 1)`, which is correct only at `raw = 1.0` and is off by 11%
  at `n=8, raw=0.1`.
- **Sync configs and the code they depend on AS A UNIT.** ORC's filesystem is
  not shared with pccfs2. On 2026-08-21 migrated configs were synced without
  `src/config/versioning.py`, a field fell back to its dataclass default, and
  four 8-GPU runs trained the wrong compartmentalization mechanism for ~46h.
  After syncing, assert a representative config resolves **identically** on ORC
  and locally. Checking that it merely parses does not catch this.
- **Prefer defaults that degrade to the behaviour in use**, so a missing file
  crashes or no-ops rather than silently selecting a different experiment.

---

## 9. Evaluation

- **Never use training-time `val loss` from `.multirun/*.log` for cross-model
  comparisons when `tr > 0`.** Always go through the formal
  `val_metrics.json` pipeline. The only exception is InfoNCE charts at
  `tr = 0`.
- Report **bits-per-byte** alongside nats-per-token wherever a future tokenizer
  change could invalidate the number.
- Annealed checkpoints sit **below** the stable trajectory and are not on its
  curve. Either branch a decay per budget, or report the stable trajectory for
  trends and annealed points for headline numbers - and say which.

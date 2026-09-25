"""Inverse of the post-hoc duplication experiment: unify a trained c=8 model's vocabularies.

Source: the paper-recipe c=8 cell (8-256, compartment-mode tr=0.1, LR 2e-5), seed 66,
at its local _rolling checkpoint (step 888,000, with optimizer + dataloader state).
Each arm resumes from that state for 30k steps, inside the original 1M-step assignment
array, so the data stream is exactly what the original run saw over those steps.

  avgboth  every token's input-embedding row AND LM-head row set to its mean over the
           8 compartments (the inverse of post-hoc duplication, which copied both)
  avgwte   input-embedding rows averaged; LM head untouched
  control  unmodified continuation

Adam moments of averaged rows are averaged the same way, so the optimizer stays
consistent with the new parameters. Compartment embeddings (comp_emb) and the
translation-token row are left alone, so the model still knows which compartment it
is in. Each arm also gets a weights-only step-888000 checkpoint so the eval sees the
state right after the intervention.

A second arm set ("copy", run group 8-256-avgemb-copy) instead installs compartment 1's
rows in every compartment, input only (copywte) or input and output (copyboth): the
closer analogue of post-hoc duplication, which copied one working embedding rather
than blending eight unrelated ones.

Usage: python prep_avgemb.py [avg|copy]
"""
import json
import shutil
import sys
from pathlib import Path

import torch

MAIN = Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression")
SRC = MAIN / "out/translation-compression/8-256-reseed/8-256-n8-tr01comp-s66"
CODE = Path("/mnt/pccfs2/backed_up/vin/dev/tc-avgemb-f718f07")
START, STEPS = 888_000, 30_000
C, V = 8, 16384
COPY_SRC = 1   # "copy" arms install compartment 1's rows (as in loss_compartment_1) everywhere
BOTH = ("transformer.wte.weight", "lm_head.weight")
WTE = ("transformer.wte.weight",)
# arm set -> (run group, {arm: (op, tensors)})
ARM_SETS = {
    "avg": ("8-256-avgemb", {"avgboth": ("avg", BOTH), "avgwte": ("avg", WTE), "control": ("avg", ())}),
    "copy": ("8-256-avgemb-copy", {"copyboth": ("copy", BOTH), "copywte": ("copy", WTE)}),
}

sys.path.insert(0, str(MAIN / "scripts"))
from gen_8_256_lrdep_configs import emit_toml  # noqa: E402


def average_rows(t: torch.Tensor) -> torch.Tensor:
    """Replace rows [0, C*V) with each token's mean over the C compartments."""
    t = t.clone()
    blk = t[: C * V].view(C, V, -1)
    t[: C * V] = blk.mean(0, keepdim=True).expand(C, V, -1).reshape(C * V, -1)
    return t


def copy_rows(t: torch.Tensor, src: int = COPY_SRC) -> torch.Tensor:
    """Replace rows [0, C*V) with compartment `src`'s row for each token."""
    t = t.clone()
    blk = t[: C * V].view(C, V, -1)
    t[: C * V] = blk[src : src + 1].expand(C, V, -1).reshape(C * V, -1)
    return t


OPS = {"avg": average_rows, "copy": copy_rows}


def main():
    arm_set = sys.argv[1] if len(sys.argv) > 1 else "avg"
    group, arms = ARM_SETS[arm_set]
    OUT = MAIN / "out/translation-compression" / group
    roll = SRC / "checkpoints" / "_rolling"
    state = json.loads((roll / "trainer_state.json").read_text())
    assert state["iter_num"] == START, state
    sd = torch.load(roll / "model.pt", map_location="cpu", weights_only=False)
    opt = torch.load(roll / "optimizer.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((SRC / "meta" / "config.json").read_text())
    assert cfg["experiment"]["n_compartments"] == C and cfg["optimizer"]["learning_rate"] == 2e-5

    # Optimizer state is indexed decay-group-first in named_parameters order
    # (GPT.configure_optimizers): rebuild that order from the state dict and check shapes.
    names = [k.removeprefix("_orig_mod.") for k in sd]
    decay = [n for n, k in zip(names, sd) if sd[k].dim() >= 2]
    nodecay = [n for n, k in zip(names, sd) if sd[k].dim() < 2]
    order = decay + nodecay
    assert len(order) == len(opt["state"]) == sum(len(g["params"]) for g in opt["param_groups"])
    for i, n in enumerate(order):
        assert opt["state"][i]["exp_avg"].shape == sd["_orig_mod." + n].shape, (i, n)
    idx = {n: i for i, n in enumerate(order)}

    for arm, (op, targets) in arms.items():
        name = f"8-256-n8-tr01comp-s66-{arm}"
        run = OUT / name
        ck = run / "checkpoints"
        # Never touch an arm that already exists: a running arm rewrites its own _rolling.
        assert not run.exists(), f"{run} exists; refusing to overwrite a (possibly running) arm"
        (ck / "_rolling").mkdir(parents=True)
        (ck / f"step-{START:06d}").mkdir(parents=True)
        new_sd = dict(sd)
        new_opt = {"state": {k: dict(v) for k, v in opt["state"].items()}, "param_groups": opt["param_groups"]}
        f = OPS[op]
        for n in targets:
            new_sd["_orig_mod." + n] = f(sd["_orig_mod." + n])
            st = new_opt["state"][idx[n]]
            st["exp_avg"] = f(st["exp_avg"])
            st["exp_avg_sq"] = f(st["exp_avg_sq"])
        torch.save(new_sd, ck / "_rolling" / "model.pt")
        torch.save(new_opt, ck / "_rolling" / "optimizer.pt")
        shutil.copy2(roll / "dataloader.pt", ck / "_rolling" / "dataloader.pt")
        (ck / "_rolling" / "trainer_state.json").write_text(json.dumps(state))
        torch.save(new_sd, ck / f"step-{START:06d}" / "model.pt")
        (ck / f"step-{START:06d}" / "trainer_state.json").write_text(json.dumps(state))
        # Config: the source run's resolved config, pinned to the paper's data order.
        toml = emit_toml(cfg, "8-256-n8-tr01comp", 2e-5, 66, name,
                         extra={("training", "max_iters"): START + STEPS},
                         header=[f"# avgemb arm {arm}: resume {SRC.name} from {START:,} for {STEPS:,} steps."])
        (CODE / "config" / "avgemb").mkdir(parents=True, exist_ok=True)
        (CODE / "config" / "avgemb" / f"{name}.toml").write_text(toml)
        w = new_sd["_orig_mod.transformer.wte.weight"][: C * V].view(C, V, -1)
        print(arm, "wte rows identical across compartments:", bool(torch.equal(w[0], w[7])),
              "| lm_head identical:",
              bool(torch.equal(*new_sd["_orig_mod.lm_head.weight"][: C * V].view(C, V, -1)[[0, 7]])))


if __name__ == "__main__":
    main()

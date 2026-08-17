"""Config schema versions, and the adapter that keeps v1 configs working.

A config's `config_version` is 1 when absent. Version 1 is frozen: every one of
the 226 pre-existing configs must keep resolving to exactly what it resolved to
before, because they are what the published runs were trained from.

Version 2 removes four fields. None of them are removed for being unused -- they
are removed because each one encodes a choice we no longer make, and leaving
them settable means a future config can express a situation the code no longer
models correctly.

  permute_tokens_per_compartment
  permute_input_tokens_per_compartment
        Compartmentalization has exactly one mechanism: vocabulary expansion,
        where compartment i's row for base token t is t + i*V and the rows are
        private. Permutation is a DIFFERENT mechanism -- ids stay in [0, V) and
        the embedding table is shared -- and the two are alternatives, not
        composable: they are the two halves of one if/else in the dataloader.
        All 115 c>1 configs use expansion. Every config that sets permutation is
        c=1, where it is a separate experiment (scrambling one model's vocabulary
        to study relearning), not compartmentalization at all.

        The pair was also actively misleading: permute_input_tokens_per_compartment
        is only read inside `if self.permute_tokens:`, so setting it True while
        permute_tokens_per_compartment is False -- which all 16 redesign configs
        did -- does nothing whatsoever while reading as a deliberate choice.

  translation_ratio_mode
        Two values ever existed. "compartment" records a raw ratio that is NOT
        the fraction of training tokens that are translation tokens; converting
        it is a unit error waiting to happen every time the two are compared.
        v2 is always absolute, and the adapter converts on load so the
        conversion happens once, here, instead of at every analysis site.

        THE CONVERSION IS t = raw / (n + raw), NOT raw / (n + 1). Verified
        exactly against compute_weights_map for n in {2,4,6,8} x raw in
        {0.1,0.25,0.5,0.75,1.0}: raw/(n+raw) reproduces compartment-mode weights
        in 20/20 cases, raw/(n+1) in 4/20 -- it is correct only at raw=1.0,
        where the two expressions coincide. At n=8, raw=0.1 the wrong formula
        gives 0.0111 against a true 0.0123, an 11% relative error. Compartment
        mode assigns each of the n compartments weight 1 and the translation
        categories weight `raw` in total, so the translation FRACTION is
        raw/(n+raw); the +1 form silently assumes raw=1.

  assignment_seed
        Mirrored from training.seed by the manager anyway. Two names for one
        number invites them to disagree.

Version 2 deliberately KEEPS translation_mode and translation_chunk_size.
Interleaved translation is still a live option and ossifying the sequence format
now would cost more than the fields do.
"""

from __future__ import annotations

from typing import Any

CURRENT_VERSION = 2
VERSION_KEY = "config_version"

# field name -> section it lives in, for error messages
_REMOVED_IN_V2 = {
    "permute_tokens_per_compartment": "experiment",
    "permute_input_tokens_per_compartment": "experiment",
    "translation_ratio_mode": "experiment",
    "assignment_seed": "experiment",
}

_REASON = {
    "permute_tokens_per_compartment":
        "compartmentalization is vocabulary expansion only; permutation is a "
        "separate c=1 experiment. Remove the field.",
    "permute_input_tokens_per_compartment":
        "belongs to the permutation mechanism, which v2 does not have. It was "
        "inert in every c>1 config that set it. Remove the field.",
    "translation_ratio_mode":
        "v2 is always absolute. If this config meant compartment-mode, convert "
        "the ratio yourself: effective_tr = raw / (n_compartments + raw).",
    "assignment_seed":
        "v2 derives it from training.seed. Remove the field.",
}


def config_version(data: dict[str, Any]) -> int:
    """Schema version of a raw config dict. Absent means 1."""
    v = data.get(VERSION_KEY, 1)
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{VERSION_KEY} must be an integer, got {v!r}")


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Return `data` with version-specific handling applied.

    v1: converted to v2 semantics in-memory so downstream code has exactly one
        set of rules to implement. The conversion is value-preserving -- a v1
        config resolves to the same effective run it always did.
    v2: rejected if it sets a removed field, rather than ignoring it. A silently
        ignored field is the failure mode this version exists to end.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    version = config_version(out)
    # Consumed here, not passed on: leaving it in the dict makes the dataclass
    # builder report it as an unknown key, which is precisely the silently-ignored
    # -field failure this version exists to end.
    out.pop(VERSION_KEY, None)
    exp = out.setdefault("experiment", {})

    if version >= CURRENT_VERSION:
        present = [f for f in _REMOVED_IN_V2 if f in exp]
        if present:
            lines = [f"config_version={version} may not set:"]
            lines += [f"  [{_REMOVED_IN_V2[f]}] {f} — {_REASON[f]}" for f in present]
            raise ValueError("\n".join(lines))
        # canonical v2 semantics, stated explicitly rather than relying on defaults
        exp["translation_ratio_mode"] = "absolute"
        exp["permute_tokens_per_compartment"] = False
        exp["permute_input_tokens_per_compartment"] = False
        if "assignment_seed" not in exp:
            exp["assignment_seed"] = int(out.get("training", {}).get("seed", 1024))
        return out

    # ---- v1 ---------------------------------------------------------------
    # Convert compartment-mode ratios to absolute ONCE, here, so no analysis has
    # to remember to. Compartment mode gives each of the n compartments weight 1
    # and the translation categories weight `raw` in total, so the translation
    # FRACTION is raw/(n+raw). See the module docstring: the raw/(n+1) form that
    # was documented project-wide is correct only at raw=1.0.
    if exp.get("translation_ratio_mode", "compartment") == "compartment":
        raw = float(exp.get("translation_ratio", 0.0) or 0.0)
        if raw > 0:
            n = int(exp.get("n_compartments", 1) or 1)
            exp["translation_ratio"] = raw / (n + raw)     # NOT raw/(n+1)
        exp["translation_ratio_mode"] = "absolute"
    return out


def describe(data: dict[str, Any]) -> str:
    v = config_version(data)
    return f"config_version={v}" + (" (implicit)" if VERSION_KEY not in data else "")

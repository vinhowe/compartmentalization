"""The version adapter must run on the path that actually trains a model.

train.py reaches configs through ConfigManager.parse_args(), not through
load_from_dict(). Before this was fixed, normalize() was called only from
load_from_dict, so:

  * sweep_runner (load_from_toml_file) ENFORCED schema v2, while
  * train.py (parse_args)              SILENTLY IGNORED it -- `config_version`
    was dropped as an unknown key and every v2 guarantee was skipped.

That is exactly the silently-ignored-field failure mode schema v2 exists to
end, sitting on the one code path where it matters most. These tests pin the
behaviour so the two entrypoints cannot drift apart again.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from src.config.manager import ConfigManager


def _write(text: str) -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    fh.write(text)
    fh.close()
    return fh.name


V2_MINIMAL = """\
config_version = 2

[model]
n_layer = 4
n_embd = 512
n_head = 8
vocab_size = 16384
block_size = 1024

[experiment]
n_compartments = 8
max_compartments = 16
translation_ratio = 0
"""

V2_ILLEGAL = V2_MINIMAL + """
permute_tokens_per_compartment = true
"""

# v1 == no config_version key at all.
V1_COMPARTMENT_MODE = """\
[model]
n_layer = 4
n_embd = 512
n_head = 8
vocab_size = 16384
block_size = 64

[experiment]
n_compartments = 8
max_compartments = 16
translation_ratio = 0.5
translation_ratio_mode = "compartment"
"""


class VersionAdapterOnCLIPath(unittest.TestCase):
    def test_v2_config_survives_parse_args_without_losing_its_version(self):
        """`config_version = 2` must not reach the dataclass builder as junk."""
        path = _write(V2_MINIMAL)
        cfg = ConfigManager().parse_args(["--job.config-file", path])
        # v2 pins these explicitly rather than leaning on dataclass defaults
        self.assertFalse(cfg.experiment.permute_tokens_per_compartment)
        self.assertFalse(cfg.experiment.permute_input_tokens_per_compartment)
        self.assertEqual(cfg.experiment.translation_ratio_mode, "absolute")

    def test_v2_rejects_a_removed_field_through_parse_args(self):
        """A v2 config setting a removed field must RAISE, not be ignored.

        This is the whole point of the version: a field that is silently
        dropped reads as a deliberate choice in the config and is not one.
        """
        path = _write(V2_ILLEGAL)
        with self.assertRaises(ValueError) as ctx:
            ConfigManager().parse_args(["--job.config-file", path])
        self.assertIn("permute_tokens_per_compartment", str(ctx.exception))

    def test_both_entrypoints_agree_on_a_v2_config(self):
        """parse_args and load_from_toml_file must resolve v2 identically."""
        path = _write(V2_MINIMAL)
        from_cli = ConfigManager().parse_args(["--job.config-file", path])
        from_file = ConfigManager().load_from_toml_file(path)
        for field in (
            "permute_tokens_per_compartment",
            "permute_input_tokens_per_compartment",
            "translation_ratio_mode",
            "translation_ratio",
            "n_compartments",
        ):
            self.assertEqual(
                getattr(from_cli.experiment, field),
                getattr(from_file.experiment, field),
                f"entrypoints disagree on experiment.{field}",
            )

    def test_v1_config_is_passed_through_parse_args_untouched(self):
        """The 226 archived v1 configs must keep resolving as they always did.

        normalize() rewrites v1 compartment-mode ratios to absolute. Applying
        that on the CLI path would change what every historical config means
        when replayed through train.py, so the adapter is scoped to v2 and v1
        keeps its raw value here.
        """
        path = _write(V1_COMPARTMENT_MODE)
        cfg = ConfigManager().parse_args(["--job.config-file", path])
        self.assertEqual(cfg.experiment.translation_ratio, 0.5)
        self.assertEqual(cfg.experiment.translation_ratio_mode, "compartment")


class LadderV2ConfigsAreV2(unittest.TestCase):
    """Every generated ladder config must actually be enforced as v2."""

    def test_all_ladder_v2_configs_load_through_the_training_entrypoint(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        configs = sorted((root / "config" / "ladder-v2").glob("*.toml"))
        if not configs:
            self.skipTest("ladder-v2 configs not generated")
        for p in configs:
            with self.subTest(config=p.name):
                cfg = ConfigManager().parse_args(["--job.config-file", str(p)])
                self.assertEqual(cfg.experiment.translation_ratio_mode, "absolute")
                self.assertFalse(cfg.experiment.permute_tokens_per_compartment)
                # budgets are derived from tokens, so batch geometry is exact
                self.assertEqual(
                    cfg.training.batch_size
                    * cfg.training.gradient_accumulation_steps,
                    2048,
                )
                self.assertFalse(cfg.training.auto_batch_config)


class UnknownKeysAreFatalUnderV2(unittest.TestCase):
    """v2 rejects a field the code does not implement; v1 still only warns.

    redesign-1b-lr4e-4-c8-100b-sharedout set `shared_output_vocab`, which exists
    nowhere in the codebase. It was dropped with a warning, the config resolved
    byte-identically to the plain c=8 arm, and ~693 A100-hours produced a
    duplicate of a baseline instead of the ablation the run was named for.

    The warning is kept for v1 because 115 of 226 archived configs carry the
    typo `always_save_checkpoints`, and failing them would break the archive.
    """

    V2_BAD = """\
config_version = 2

[model]
n_layer = 4
n_embd = 512
n_head = 8
vocab_size = 16384

[experiment]
n_compartments = 8
max_compartments = 16
shared_output_vocab = true
"""
    V1_BAD = V2_BAD.replace("config_version = 2\n", "")

    def test_v2_rejects_unknown_key_via_parse_args(self):
        with self.assertRaises(ValueError) as ctx:
            ConfigManager().parse_args(["--job.config-file", _write(self.V2_BAD)])
        self.assertIn("shared_output_vocab", str(ctx.exception))

    def test_v2_rejects_unknown_key_via_load_from_toml_file(self):
        with self.assertRaises(ValueError) as ctx:
            ConfigManager().load_from_toml_file(_write(self.V2_BAD))
        self.assertIn("shared_output_vocab", str(ctx.exception))

    def test_v1_still_loads_with_only_a_warning(self):
        cfg = ConfigManager().parse_args(["--job.config-file", _write(self.V1_BAD)])
        self.assertEqual(cfg.experiment.n_compartments, 8)

    def test_the_strict_rule_breaks_nothing_in_the_archive(self):
        """No archived config may fail BECAUSE of the strict unknown-key rule.

        Asserting zero failures would be wrong: two configs are independently
        broken and were before this change -- 1b-multilingual-en-only-qwen3.toml
        is malformed TOML, and fineweb-vanilla.toml omits a required field. The
        property that matters is narrower and is the one this rule could
        plausibly violate: nothing in the archive is rejected for an unknown key.
        """
        import contextlib
        import io

        root = pathlib.Path(__file__).resolve().parents[1]
        cfgs = sorted((root / "config").glob("*.toml"))
        if not cfgs:
            self.skipTest("no configs")
        rejected_for_unknown_key = []
        for c in cfgs:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    ConfigManager().parse_args(["--job.config-file", str(c)])
            except ValueError as e:
                if "unknown key" in str(e):
                    rejected_for_unknown_key.append(f"{c.name}: {str(e)[:90]}")
            except Exception:
                pass  # pre-existing breakage, not this rule's business
        self.assertEqual(
            rejected_for_unknown_key, [],
            "the strict rule rejected archived config(s):\n" +
            "\n".join(rejected_for_unknown_key[:10]))


if __name__ == "__main__":
    unittest.main()

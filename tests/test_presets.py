from __future__ import annotations

import unittest
from dataclasses import replace

from src.config.job_config import JobConfig
from src.config.presets import apply_bpe16384_batch_config, scale_batch_for_vram


class BatchPresetTests(unittest.TestCase):
    def test_exact_budget_opt_out_blocks_all_batch_rewrites(self):
        base = JobConfig()
        configured = replace(
            base,
            model=replace(base.model, n_embd=512, block_size=512, vocab_size=16384),
            experiment=replace(
                base.experiment, n_compartments=1, permute_tokens_per_compartment=False,
            ),
            training=replace(
                base.training,
                batch_size=256,
                gradient_accumulation_steps=1,
                auto_batch_config=False,
            ),
        )
        after_vocab = apply_bpe16384_batch_config(configured)
        after_vram = scale_batch_for_vram(after_vocab, 192 * 1024**3)
        self.assertEqual(after_vram.training.batch_size, 256)
        self.assertEqual(after_vram.training.gradient_accumulation_steps, 1)
        self.assertEqual(
            after_vram.training.batch_size
            * after_vram.training.gradient_accumulation_steps
            * after_vram.model.block_size,
            131_072,
        )

    def test_legacy_default_still_applies_known_preset(self):
        base = JobConfig()
        configured = replace(
            base,
            model=replace(base.model, n_embd=512, vocab_size=16384),
            experiment=replace(
                base.experiment, n_compartments=1, permute_tokens_per_compartment=False,
            ),
        )
        result = apply_bpe16384_batch_config(configured)
        self.assertEqual(result.training.batch_size, 256)
        self.assertEqual(result.training.gradient_accumulation_steps, 8)


if __name__ == "__main__":
    unittest.main()

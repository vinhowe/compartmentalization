from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.datafile import load_data_shard, peek_data_shard, write_data_shard


class DatafileTests(unittest.TestCase):
    def test_uint16_and_uint32_round_trip(self):
        tokens = np.array([0, 1, 255, 16_383, 65_535], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            for compact in (False, True):
                with self.subTest(compact=compact):
                    path = Path(temporary) / f"tokens-{compact}.bin"
                    write_data_shard(path, tokens, use_uint16=compact)
                    self.assertEqual(peek_data_shard(path), len(tokens))
                    np.testing.assert_array_equal(load_data_shard(path), tokens)
                    expected_item_bytes = 2 if compact else 4
                    self.assertEqual(path.stat().st_size, 1024 + len(tokens) * expected_item_bytes)

    def test_compact_format_refuses_out_of_range_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                write_data_shard(
                    Path(temporary) / "bad.bin",
                    np.array([65_536], dtype=np.int64),
                    use_uint16=True,
                )


if __name__ == "__main__":
    unittest.main()

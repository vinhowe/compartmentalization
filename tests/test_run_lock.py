from __future__ import annotations

import json
import tempfile
import unittest

from src.run_lock import ActiveRunError, ActiveRunLock


class ActiveRunLockTests(unittest.TestCase):
    def test_second_live_owner_is_refused_and_release_allows_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = ActiveRunLock(temporary, run_id="first").acquire()
            owner_path = first.owner_path
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            self.assertEqual(owner["run_id"], "first")

            with self.assertRaises(ActiveRunError):
                ActiveRunLock(temporary, run_id="duplicate").acquire()

            first.release()
            self.assertFalse(owner_path.exists())
            resumed = ActiveRunLock(temporary, run_id="resumed").acquire()
            self.assertEqual(
                json.loads(owner_path.read_text(encoding="utf-8"))["run_id"],
                "resumed",
            )
            resumed.release()

    def test_records_exact_config_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = f"{temporary}/arm.toml"
            with open(config, "w", encoding="utf-8") as handle:
                handle.write("seed = 64\n")
            lock = ActiveRunLock(
                temporary,
                run_id="with-config",
                config_path=config,
                config_identity="canonical-config-id",
            ).acquire()
            owner = json.loads(lock.owner_path.read_text(encoding="utf-8"))
            self.assertEqual(
                owner["config_file_sha256"],
                "18c33fb8e902204111cb9a91c9c80420a1c358b4da99985cf9aea0fa62f0d8ec",
            )
            self.assertEqual(owner["config_identity"], "canonical-config-id")
            lock.release()


if __name__ == "__main__":
    unittest.main()

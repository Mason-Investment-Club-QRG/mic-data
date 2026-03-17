from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.utils.idempotent_io import (
    atomic_write_csv,
    canonicalize_frame,
    dataframe_hash,
    should_write,
    stage_lock,
)


class TestIdempotentIO(unittest.TestCase):
    def test_canonicalize_and_hash_are_stable(self) -> None:
        df = pd.DataFrame({"k": [2, 1], "v": ["b", "a"]})
        out = canonicalize_frame(df, sort_by=["k"])
        digest_1 = dataframe_hash(out)
        digest_2 = dataframe_hash(out)
        self.assertEqual(digest_1, digest_2)
        self.assertEqual(list(out["k"]), [1, 2])

    def test_should_write_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "x.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")

            self.assertTrue(should_write(path, mode="replace"))
            self.assertFalse(should_write(path, mode="skip"))
            with self.assertRaises(FileExistsError):
                should_write(path, mode="error")

    def test_atomic_write_csv_skip_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "x.csv"
            df = pd.DataFrame({"id": [1], "value": [2]})
            first = atomic_write_csv(df, path=path, sort_by=["id"], mode="replace")
            second = atomic_write_csv(df, path=path, sort_by=["id"], mode="skip")
            self.assertTrue(first.wrote)
            self.assertFalse(second.wrote)
            self.assertEqual(first.content_hash, second.content_hash)

    def test_stage_lock_prevents_double_acquire(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "a.lock"
            with stage_lock(lock_path):
                with self.assertRaises(RuntimeError):
                    with stage_lock(lock_path):
                        pass


if __name__ == "__main__":
    unittest.main()

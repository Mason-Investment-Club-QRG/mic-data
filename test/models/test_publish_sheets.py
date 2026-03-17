from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.reporting.publish_sheets import _dataset_hash, _load_publish_state, _save_publish_state


class TestPublishSheetsHelpers(unittest.TestCase):
    def test_dataset_hash_is_deterministic_for_sorted_equivalent_frames(self) -> None:
        a = pd.DataFrame({"trade_date": ["2026-03-02", "2026-03-01"], "ticker": ["MSFT", "AAPL"], "permno": [2, 1]})
        b = pd.DataFrame({"trade_date": ["2026-03-01", "2026-03-02"], "ticker": ["AAPL", "MSFT"], "permno": [1, 2]})

        hash_a = _dataset_hash(a, sort_by=["trade_date", "ticker", "permno"])
        hash_b = _dataset_hash(b, sort_by=["trade_date", "ticker", "permno"])
        self.assertEqual(hash_a, hash_b)

    def test_publish_state_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            payload = {"SecurityReturnsDaily": "abc123"}
            _save_publish_state(path, state=payload, dry_run=False)
            loaded = _load_publish_state(path)
            self.assertEqual(loaded, payload)


if __name__ == "__main__":
    unittest.main()

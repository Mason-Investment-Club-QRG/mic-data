from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.models.factors.wrds_source import WrdsFactorSource


class _FakeConn:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.closed = False

    def raw_sql(self, query: str, date_cols: list[str]) -> pd.DataFrame:
        _ = query
        _ = date_cols
        return self.frame.copy()

    def close(self) -> None:
        self.closed = True


class TestWrdsFactorSource(unittest.TestCase):
    def test_load_factors_normalizes_and_compounds(self) -> None:
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-02-03",
                        "2020-02-04",
                    ]
                ),
                "mktrf": [0.01, 0.02, -0.01, 0.005],
                "smb": [0.005, 0.005, 0.0, 0.001],
                "hml": [0.002, -0.001, 0.002, 0.002],
                "rf": [0.0001, 0.0001, 0.0002, 0.0002],
            }
        )

        conn = _FakeConn(daily)

        def fake_factory(**kwargs):
            _ = kwargs
            return conn

        source = WrdsFactorSource(username="user", connection_factory=fake_factory)
        out = source.load_factors("2020-01-01", "2020-02-29", "M")

        self.assertEqual(list(out.columns), ["mkt_rf", "smb", "hml", "rf"])
        self.assertEqual(len(out), 2)

        jan_mkt = float(out.iloc[0]["mkt_rf"])
        expected_jan_mkt = (1.01 * 1.02) - 1.0
        self.assertAlmostEqual(jan_mkt, expected_jan_mkt, places=10)

        self.assertTrue(conn.closed)

    def test_raises_when_empty(self) -> None:
        conn = _FakeConn(pd.DataFrame(columns=["date", "mktrf", "smb", "hml", "rf"]))

        source = WrdsFactorSource(connection_factory=lambda **kwargs: conn)
        with self.assertRaises(ValueError):
            source.load_factors("2020-01-01", "2020-12-31", "M")


if __name__ == "__main__":
    unittest.main()

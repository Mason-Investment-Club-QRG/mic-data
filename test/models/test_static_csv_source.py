from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mic_data.models.factors.static_csv_source import StaticCsvFactorSource


class TestStaticCsvFactorSource(unittest.TestCase):
    def test_load_factors_parses_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "F-F_Research_Data_Factors.csv"
            p.write_text(
                "header line 1\n"
                "header line 2\n"
                "header line 3\n"
                ",Mkt-RF,SMB,HML,RF\n"
                "202001, 1.00, 2.00,-1.00, 0.10\n"
                "202002,-2.00, 0.50, 3.00, 0.12\n"
                "Annual Factors: omitted\n",
                encoding="utf-8",
            )

            source = StaticCsvFactorSource(csv_path=p)
            out = source.load_factors("2020-01-01", "2020-12-31", "M")

            self.assertEqual(list(out.columns), ["mkt_rf", "smb", "hml", "rf"])
            self.assertEqual(len(out), 2)
            self.assertAlmostEqual(float(out.iloc[0]["mkt_rf"]), 0.01, places=8)
            self.assertAlmostEqual(float(out.iloc[0]["rf"]), 0.001, places=8)

    def test_rejects_non_monthly_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "F-F_Research_Data_Factors.csv"
            p.write_text(
                "a\nb\nc\n,Mkt-RF,SMB,HML,RF\n202001,1,1,1,1\n", encoding="utf-8"
            )
            source = StaticCsvFactorSource(csv_path=p)
            with self.assertRaises(ValueError):
                source.load_factors("2020-01-01", "2020-12-31", "D")


if __name__ == "__main__":
    unittest.main()

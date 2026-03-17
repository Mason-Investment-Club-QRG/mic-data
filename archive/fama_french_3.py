from __future__ import annotations

# Historical archive of the pre-ff_factor_matrix FF3 entrypoint.
# This file is kept for reference only and is not part of the active pipeline.

import argparse
from dataclasses import replace
from pathlib import Path

from mic_data.models.runner import (
    FF3PipelineConfig,
    load_ff3_pipeline_config,
    run_ff3_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the FF3 pipeline runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run WRDS-first FF3 pipeline with static fallback and A/B validation artifacts."
        )
    )
    parser.add_argument(
        "--config",
        default="config/ff3_pipeline.yaml",
        help="YAML config path. If missing, internal defaults are used.",
    )
    parser.add_argument("--start-date", help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Override end date (YYYY-MM-DD).")
    parser.add_argument(
        "--wrds-username",
        help="Optional WRDS username override (otherwise use WRDS_USERNAME env var).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable static fallback when WRDS pull fails.",
    )
    return parser


def main() -> None:
    """CLI entry point for FF3 run + validation artifacts."""
    args = build_parser().parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = load_ff3_pipeline_config(config_path)
    else:
        config = FF3PipelineConfig()

    if args.start_date:
        config = replace(config, start_date=args.start_date)
    if args.end_date:
        config = replace(config, end_date=args.end_date)
    if args.wrds_username:
        config = replace(config, wrds_username=args.wrds_username)
    if args.no_fallback:
        config = replace(config, allow_fallback=False)

    artifacts = run_ff3_pipeline(config)

    print("FF3 pipeline completed.")
    print(f"WRDS available: {artifacts.wrds_available}")
    print(f"Fallback used: {artifacts.used_fallback}")
    print(f"Run log: {artifacts.log_path}")
    print(f"Factor comparison CSV: {artifacts.factor_comparison_csv}")
    print(f"Regression comparison JSON: {artifacts.regression_comparison_json}")
    print(f"Validation summary: {artifacts.validation_summary_md}")


if __name__ == "__main__":
    main()

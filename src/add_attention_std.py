"""
src/add_attention_std.py
========================

Add standard deviations to the attention-head summary.

This script reads pair-level attention files produced by src/attention_analysis.py
and creates summaries with mean, median, and standard deviation.

Inputs:
    outputs/attention/attention_pair_scores_*.csv

Outputs:
    outputs/attention/attention_head_summary_with_std.csv

Run:
    python src/add_attention_std.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "split",
        "layer",
        "head",
        "pair_index",
        "closer_to_matching_opener",
        "opener_to_matching_closer",
        "closer_to_cls",
        "opener_to_cls",
        "distance",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    return (
        df.groupby(["split", "layer", "head"], as_index=False)
        .agg(
            pairs=("pair_index", "count"),
            mean_closer_to_matching_opener=("closer_to_matching_opener", "mean"),
            std_closer_to_matching_opener=("closer_to_matching_opener", "std"),
            median_closer_to_matching_opener=("closer_to_matching_opener", "median"),
            mean_opener_to_matching_closer=("opener_to_matching_closer", "mean"),
            std_opener_to_matching_closer=("opener_to_matching_closer", "std"),
            median_opener_to_matching_closer=("opener_to_matching_closer", "median"),
            mean_closer_to_cls=("closer_to_cls", "mean"),
            std_closer_to_cls=("closer_to_cls", "std"),
            mean_opener_to_cls=("opener_to_cls", "mean"),
            std_opener_to_cls=("opener_to_cls", "std"),
            mean_distance=("distance", "mean"),
            std_distance=("distance", "std"),
        )
        .sort_values(["split", "mean_closer_to_matching_opener"], ascending=[True, False])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add std columns to attention summaries.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/attention"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/attention/attention_head_summary_with_std.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("attention_pair_scores_*.csv"))

    if not paths:
        raise FileNotFoundError(
            f"No attention_pair_scores_*.csv files found in {args.input_dir}. "
            "Run src/attention_analysis.py first."
        )

    tables = [summarize(path) for path in paths]
    out = pd.concat(tables, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"[add_attention_std] Wrote {args.output}")


if __name__ == "__main__":
    main()

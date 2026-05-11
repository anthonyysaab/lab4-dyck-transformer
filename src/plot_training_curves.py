"""
src/plot_training_curves.py
===========================

Create report-ready training curves from detection_metrics.csv.

Input:
    outputs/tables/detection_metrics.csv

Outputs:
    outputs/figures/training_loss_curve.png
    outputs/figures/training_accuracy_curve.png
    outputs/figures/training_curves.png

Run:
    python src/plot_training_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {"split", "epoch", "loss", "accuracy"}


def load_epoch_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train first with: python src/train_detection.py"
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["split"].isin(["train", "dev"])].copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)

    if df.empty:
        raise ValueError(f"No train/dev epoch rows found in {path}.")

    return df.sort_values(["split", "epoch"])


def plot_single_metric(df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))

    for split in ["train", "dev"]:
        split_df = df[df["split"] == split]
        if split_df.empty:
            continue
        plt.plot(split_df["epoch"], split_df[metric], marker="o", label=split)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " during detection training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for split in ["train", "dev"]:
        split_df = df[df["split"] == split]
        if split_df.empty:
            continue
        axes[0].plot(split_df["epoch"], split_df["loss"], marker="o", label=split)
        axes[1].plot(split_df["epoch"], split_df["accuracy"], marker="o", label=split)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Detection training curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot detection training curves.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/tables/detection_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_epoch_metrics(args.metrics_csv)

    loss_path = args.output_dir / "training_loss_curve.png"
    accuracy_path = args.output_dir / "training_accuracy_curve.png"
    combined_path = args.output_dir / "training_curves.png"

    plot_single_metric(df, "loss", "Loss", loss_path)
    plot_single_metric(df, "accuracy", "Accuracy", accuracy_path)
    plot_combined(df, combined_path)

    print(f"[plot_training_curves] Wrote {loss_path}")
    print(f"[plot_training_curves] Wrote {accuracy_path}")
    print(f"[plot_training_curves] Wrote {combined_path}")


if __name__ == "__main__":
    main()

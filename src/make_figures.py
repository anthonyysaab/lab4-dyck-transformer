"""
src/make_figures.py
===================

Create report-ready figures for Lab 4.

Inputs:
    outputs/tables/eval_overall_metrics.csv
    outputs/tables/eval_by_error_type.csv
    outputs/tables/eval_by_depth.csv
    outputs/tables/eval_by_length_bin.csv
    outputs/tables/probe_metrics.csv
    outputs/attention/attention_head_summary_test_id.csv
    outputs/attention/attention_head_summary_test_ood.csv

Outputs:
    outputs/figures/*.png

Run:
    python src/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TABLE_DIR = Path("outputs/tables")
ATTENTION_DIR = Path("outputs/attention")
FIGURE_DIR = Path("outputs/figures")


def save_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df[x].astype(str), df[y])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_grouped_bar(
    df: pd.DataFrame,
    category_column: str,
    split_column: str,
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 0,
) -> None:
    pivot = df.pivot(index=category_column, columns=split_column, values=value_column)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=split_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overall_detection() -> None:
    df = pd.read_csv(TABLE_DIR / "eval_overall_metrics.csv")

    save_bar(
        df=df,
        x="split",
        y="accuracy",
        title="Detection accuracy by split",
        ylabel="Accuracy",
        output_path=FIGURE_DIR / "detection_accuracy_by_split.png",
    )

    save_bar(
        df=df,
        x="split",
        y="macro_f1",
        title="Detection macro-F1 by split",
        ylabel="Macro-F1",
        output_path=FIGURE_DIR / "detection_macro_f1_by_split.png",
    )


def plot_error_type_detection() -> None:
    df = pd.read_csv(TABLE_DIR / "eval_by_error_type.csv")
    df = df[df["group_value"] != "no_error"].copy()

    save_grouped_bar(
        df=df,
        category_column="group_value",
        split_column="split",
        value_column="accuracy",
        title="Detection accuracy by corruption type",
        ylabel="Accuracy",
        output_path=FIGURE_DIR / "detection_accuracy_by_error_type.png",
        rotation=30,
    )


def plot_depth_detection() -> None:
    df = pd.read_csv(TABLE_DIR / "eval_by_depth.csv")
    df["group_value"] = df["group_value"].astype(str)

    save_grouped_bar(
        df=df,
        category_column="group_value",
        split_column="split",
        value_column="accuracy",
        title="Detection accuracy by maximum depth",
        ylabel="Accuracy",
        output_path=FIGURE_DIR / "detection_accuracy_by_depth.png",
    )


def plot_length_detection() -> None:
    df = pd.read_csv(TABLE_DIR / "eval_by_length_bin.csv")

    save_grouped_bar(
        df=df,
        category_column="group_value",
        split_column="split",
        value_column="accuracy",
        title="Detection accuracy by input length",
        ylabel="Accuracy",
        output_path=FIGURE_DIR / "detection_accuracy_by_length.png",
        rotation=30,
    )


def plot_probe_metrics() -> None:
    df = pd.read_csv(TABLE_DIR / "probe_metrics.csv")

    save_grouped_bar(
        df=df,
        category_column="task",
        split_column="split",
        value_column="accuracy",
        title="Linear probe accuracy",
        ylabel="Accuracy",
        output_path=FIGURE_DIR / "probe_accuracy.png",
        rotation=20,
    )

    save_grouped_bar(
        df=df,
        category_column="task",
        split_column="split",
        value_column="macro_f1",
        title="Linear probe macro-F1",
        ylabel="Macro-F1",
        output_path=FIGURE_DIR / "probe_macro_f1.png",
        rotation=20,
    )


def plot_attention_summary() -> None:
    paths = [
        ATTENTION_DIR / "attention_head_summary_test_id.csv",
        ATTENTION_DIR / "attention_head_summary_test_ood.csv",
    ]

    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        print("[make_figures] No attention summaries found. Skipping attention plot.")
        return

    df = pd.concat(frames, ignore_index=True)
    df["head_label"] = (
        df["split"].astype(str)
        + "_L"
        + df["layer"].astype(str)
        + "H"
        + df["head"].astype(str)
    )

    df = df.sort_values("mean_closer_to_matching_opener", ascending=False).head(16)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df["head_label"], df["mean_closer_to_matching_opener"])
    ax.set_title("Mean attention from closer to matching opener")
    ax.set_ylabel("Attention mass")
    ax.set_ylim(0, max(0.08, float(df["mean_closer_to_matching_opener"].max()) * 1.2))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "attention_matching_opener.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_overall_detection()
    plot_error_type_detection()
    plot_depth_detection()
    plot_length_detection()
    plot_probe_metrics()
    plot_attention_summary()

    print("[make_figures] Wrote figures to", FIGURE_DIR)
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(" -", path)


if __name__ == "__main__":
    main()

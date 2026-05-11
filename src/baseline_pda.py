"""
src/baseline_pda.py
===================

Deterministic pushdown automaton baseline for Lab 4.

This script evaluates exact Dyck-language membership using a stack.

For D(2), the valid bracket pairs are:
    ( )
    [ ]

This baseline is symbolic and should achieve perfect or near-perfect detection
accuracy on the generated datasets, because the labels were defined by exact
Dyck membership.

Outputs:
    outputs/tables/pda_baseline_metrics.csv
    outputs/tables/pda_predictions_*.csv

Run:
    python src/baseline_pda.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PAIRS = {
    "(": ")",
    "[": "]",
}

OPENERS = set(PAIRS.keys())
CLOSERS = set(PAIRS.values())


def is_dyck(tokens: list[str]) -> bool:
    """
    Exact D(2) recognizer using a stack.

    Return True iff the input is a well-balanced bracket string with correct
    nesting and matching bracket types.
    """
    stack: list[str] = []

    for token in tokens:
        if token in OPENERS:
            stack.append(PAIRS[token])
        elif token in CLOSERS:
            if not stack:
                return False

            expected = stack.pop()
            if token != expected:
                return False
        else:
            raise ValueError(f"Unknown token: {token!r}")

    return len(stack) == 0


def first_failure_reason(tokens: list[str]) -> str:
    """
    Provide a simple diagnostic reason for invalid sequences.

    This is not meant to exactly reconstruct the corruption operation. It is
    an automaton-level explanation of where membership fails.
    """
    stack: list[str] = []

    for index, token in enumerate(tokens):
        if token in OPENERS:
            stack.append(PAIRS[token])
            continue

        if token in CLOSERS:
            if not stack:
                return f"premature_close_at_{index}"

            expected = stack.pop()
            if token != expected:
                return f"type_mismatch_at_{index}_expected_{expected}_got_{token}"

            continue

        return f"unknown_token_at_{index}"

    if stack:
        return f"missing_closer_expected_{''.join(reversed(stack))}"

    return "valid"


def evaluate_split(csv_path: Path, output_dir: Path) -> dict[str, object]:
    """
    Evaluate the PDA baseline on one CSV split.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"split", "tokens", "label", "error_type", "max_depth", "input_length"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")

    predictions: list[int] = []
    reasons: list[str] = []

    for text in df["tokens"].astype(str).tolist():
        tokens = text.split()
        membership = is_dyck(tokens)
        predictions.append(1 if membership else 0)
        reasons.append(first_failure_reason(tokens))

    gold = df["label"].astype(int).to_numpy()

    accuracy = accuracy_score(gold, predictions)
    macro_f1 = f1_score(gold, predictions, average="macro")

    split_name = str(df["split"].iloc[0])

    pred_df = df[
        ["id", "split", "tokens", "label", "error_type", "max_depth", "input_length"]
    ].copy()
    pred_df["prediction"] = predictions
    pred_df["correct"] = pred_df["prediction"].astype(int) == pred_df["label"].astype(int)
    pred_df["pda_reason"] = reasons

    pred_path = output_dir / f"pda_predictions_{split_name}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    cm = confusion_matrix(gold, predictions)
    cm_path = output_dir / f"pda_confusion_matrix_{split_name}.csv"
    pd.DataFrame(
        cm,
        index=["gold_invalid", "gold_valid"],
        columns=["pred_invalid", "pred_valid"],
    ).to_csv(cm_path, encoding="utf-8")

    print(f"\n[pda] {split_name}")
    print(f"[pda] accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}")
    print("[pda] classification report:")
    print(
        classification_report(
            gold,
            predictions,
            target_names=["invalid", "valid"],
            digits=4,
        )
    )
    print(f"[pda] Wrote {pred_path}")
    print(f"[pda] Wrote {cm_path}")

    return {
        "split": split_name,
        "rows": len(df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "prediction_file": str(pred_path),
        "confusion_matrix_file": str(cm_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PDA baseline on Dyck datasets.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-id-csv", type=Path, default=Path("data/test_id.csv"))
    parser.add_argument("--test-ood-csv", type=Path, default=Path("data/test_ood.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = [
        args.train_csv,
        args.dev_csv,
        args.test_id_csv,
        args.test_ood_csv,
    ]

    rows = []

    for csv_path in split_paths:
        rows.append(evaluate_split(csv_path, args.output_dir))

    metrics_df = pd.DataFrame(rows)
    metrics_path = args.output_dir / "pda_baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    print(f"\n[pda] Wrote {metrics_path}")
    print("[pda] Done.")


if __name__ == "__main__":
    main()

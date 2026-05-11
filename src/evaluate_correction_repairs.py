"""
src/evaluate_correction_repairs.py
==================================

Evaluate the correction model with the repaired-string metric requested in
Lab 4.

The training script already saves token-level correction predictions in:

    outputs/tables/correction_predictions_test_id.csv
    outputs/tables/correction_predictions_test_ood.csv

This script applies each predicted edit sequence back to the input bracket
string, then checks whether the repaired string is accepted by the exact Dyck
PDA recogniser.

It also reports whether the repaired string exactly matches the original valid
string stored in the generated dataset. This is stricter than PDA acceptance,
because some corrupted strings may have more than one valid repair.

Outputs:
    outputs/tables/correction_repair_metrics.csv
    outputs/tables/correction_repair_by_error_type.csv
    outputs/tables/correction_predictions_<split>_repair.csv

Run from the project root:

    python src/evaluate_correction_repairs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Dyck utilities
# ---------------------------------------------------------------------------

PAIRS = {
    "(": ")",
    "[": "]",
}

OPENERS = set(PAIRS.keys())
CLOSERS = set(PAIRS.values())
OPENER_TO_CLOSER = PAIRS

ACTION_TO_TOKEN = {
    "INSERT_LPAREN": "(",
    "INSERT_RPAREN": ")",
    "INSERT_LBRACK": "[",
    "INSERT_RBRACK": "]",
    "REPLACE_LPAREN": "(",
    "REPLACE_RPAREN": ")",
    "REPLACE_LBRACK": "[",
    "REPLACE_RBRACK": "]",
}


def split_brackets(text: object) -> list[str]:
    """Convert a space-separated bracket string into a token list."""
    if pd.isna(text):
        return []
    return str(text).strip().split()


def is_dyck(tokens: Iterable[str]) -> bool:
    """Return True iff the token sequence is a valid D(2) Dyck string."""
    stack: list[str] = []

    for token in tokens:
        if token in OPENERS:
            stack.append(OPENER_TO_CLOSER[token])
        elif token in CLOSERS:
            if not stack:
                return False
            expected = stack.pop()
            if token != expected:
                return False
        else:
            return False

    return len(stack) == 0


def apply_correction_actions(tokens: list[str], actions: list[str]) -> list[str]:
    """
    Apply local correction actions to a bracket-token sequence.

    Action convention inherited from src/dyck_data.py:
        - OK keeps the current token.
        - DELETE removes the current token.
        - REPLACE_X replaces the current token with X.
        - INSERT_X keeps the current token and inserts X after it.

    The dataset stores exactly one action per input bracket token.
    """
    if len(tokens) != len(actions):
        raise ValueError(
            f"Action length mismatch: {len(actions)} actions for {len(tokens)} tokens."
        )

    repaired: list[str] = []

    for token, action in zip(tokens, actions):
        if action == "OK":
            repaired.append(token)
            continue

        if action == "DELETE":
            continue

        if action.startswith("REPLACE_"):
            if action not in ACTION_TO_TOKEN:
                raise ValueError(f"Unknown replacement action: {action!r}")
            repaired.append(ACTION_TO_TOKEN[action])
            continue

        if action.startswith("INSERT_"):
            if action not in ACTION_TO_TOKEN:
                raise ValueError(f"Unknown insertion action: {action!r}")
            repaired.append(token)
            repaired.append(ACTION_TO_TOKEN[action])
            continue

        raise ValueError(f"Unknown correction action: {action!r}")

    return repaired


def load_json_list(text: object) -> list[str]:
    """Parse a JSON list from a CSV cell and validate that it contains strings."""
    value = json.loads(str(text))
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("Expected all action labels to be strings.")
    return value


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_split_predictions(
    split: str,
    data_dir: Path,
    table_dir: Path,
) -> pd.DataFrame:
    prediction_path = table_dir / f"correction_predictions_{split}.csv"
    data_path = data_dir / f"{split}.csv"

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {prediction_path}. "
            "Run python src/train_correction.py first."
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")

    predictions = pd.read_csv(prediction_path)
    data = pd.read_csv(data_path)

    if "predicted_actions" not in predictions.columns:
        raise ValueError(f"{prediction_path} has no predicted_actions column.")

    required_data_columns = {
        "id",
        "original",
        "corrupted",
        "tokens",
        "label",
        "error_type",
        "max_depth",
        "input_length",
        "correction_actions",
    }
    missing = required_data_columns - set(data.columns)
    if missing:
        raise ValueError(f"{data_path} is missing columns: {sorted(missing)}")

    source_columns = [
        "id",
        "original",
        "corrupted",
        "tokens",
        "label",
        "error_type",
        "max_depth",
        "input_length",
        "correction_actions",
    ]

    merged = predictions.merge(
        data[source_columns],
        on="id",
        how="left",
        suffixes=("", "_source"),
        validate="one_to_one",
    )

    if merged["original"].isna().any():
        raise ValueError(f"Could not merge every prediction row with {data_path}.")

    return merged


def evaluate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply predicted repairs and add row-level repair metrics."""
    output_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        token_text = row.get("tokens_source", row.get("tokens", ""))
        gold_actions_text = row.get(
            "correction_actions_source",
            row.get("correction_actions", "[]"),
        )

        tokens = split_brackets(token_text)
        original_tokens = split_brackets(row["original"])

        predicted_actions = load_json_list(row["predicted_actions"])
        gold_actions = load_json_list(gold_actions_text)

        label_exact = predicted_actions == gold_actions

        action_length_ok = len(tokens) == len(predicted_actions)
        action_valid = True
        repaired_tokens: list[str] = []
        repair_error = ""

        try:
            repaired_tokens = apply_correction_actions(tokens, predicted_actions)
        except ValueError as exc:
            action_valid = False
            repair_error = str(exc)

        repair_is_dyck = bool(action_valid and is_dyck(repaired_tokens))
        repair_matches_original = bool(action_valid and repaired_tokens == original_tokens)

        row_dict = row.to_dict()
        row_dict.update(
            {
                "action_length_ok": action_length_ok,
                "action_valid": action_valid,
                "repair_error": repair_error,
                "predicted_repaired": " ".join(repaired_tokens),
                "repair_is_dyck": repair_is_dyck,
                "repair_matches_original": repair_matches_original,
                "label_exact_match_recomputed": label_exact,
            }
        )
        output_rows.append(row_dict)

    return pd.DataFrame(output_rows)


def mean_bool(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.astype(bool).mean())


def summarize_split(split: str, df: pd.DataFrame) -> dict[str, object]:
    valid_mask = df["label"].astype(int) == 1
    invalid_mask = df["label"].astype(int) == 0

    return {
        "split": split,
        "rows": int(len(df)),
        "repair_dyck_acceptance": mean_bool(df["repair_is_dyck"]),
        "repair_original_match": mean_bool(df["repair_matches_original"]),
        "label_exact_match": mean_bool(df["label_exact_match_recomputed"]),
        "valid_rows": int(valid_mask.sum()),
        "valid_repair_dyck_acceptance": mean_bool(df.loc[valid_mask, "repair_is_dyck"]),
        "valid_unchanged_accuracy": mean_bool(
            df.loc[valid_mask, "repair_matches_original"]
        ),
        "invalid_rows": int(invalid_mask.sum()),
        "invalid_repair_dyck_acceptance": mean_bool(
            df.loc[invalid_mask, "repair_is_dyck"]
        ),
        "invalid_original_match": mean_bool(
            df.loc[invalid_mask, "repair_matches_original"]
        ),
        "invalid_label_exact_match": mean_bool(
            df.loc[invalid_mask, "label_exact_match_recomputed"]
        ),
    }


def summarize_by_error_type(split: str, df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for error_type, group in df.groupby("error_type", dropna=False):
        rows.append(
            {
                "split": split,
                "error_type": error_type,
                "rows": int(len(group)),
                "repair_dyck_acceptance": mean_bool(group["repair_is_dyck"]),
                "repair_original_match": mean_bool(group["repair_matches_original"]),
                "label_exact_match": mean_bool(group["label_exact_match_recomputed"]),
            }
        )

    return pd.DataFrame(rows)


def update_correction_metrics(
    metrics_path: Path,
    repair_metrics: pd.DataFrame,
) -> None:
    """Add repaired-string columns to outputs/tables/correction_metrics.csv."""
    if not metrics_path.exists():
        print(f"[evaluate_correction_repairs] Skipping missing {metrics_path}")
        return

    metrics = pd.read_csv(metrics_path)

    repair_columns = [
        "repair_dyck_acceptance",
        "repair_original_match",
        "label_exact_match",
        "invalid_repair_dyck_acceptance",
        "invalid_original_match",
        "invalid_label_exact_match",
    ]

    for column in repair_columns:
        if column not in metrics.columns:
            metrics[column] = pd.NA

    for _, repair_row in repair_metrics.iterrows():
        split = repair_row["split"]
        mask = (metrics["split"].astype(str) == str(split)) & (
            metrics["epoch"].astype(str) == "best"
        )

        if not mask.any():
            print(
                "[evaluate_correction_repairs] No best-row found in "
                f"{metrics_path} for split={split!r}."
            )
            continue

        for column in repair_columns:
            metrics.loc[mask, column] = repair_row[column]

    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"[evaluate_correction_repairs] Updated {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate repaired-string exact match for correction outputs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_id", "test_ood"],
        help="Splits to evaluate. Default: test_id test_ood",
    )
    parser.add_argument(
        "--no-update-correction-metrics",
        action="store_true",
        help="Do not add the repair columns to correction_metrics.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    by_error_tables: list[pd.DataFrame] = []

    for split in args.splits:
        print(f"[evaluate_correction_repairs] Evaluating {split}...")

        merged = load_split_predictions(
            split=split,
            data_dir=args.data_dir,
            table_dir=args.table_dir,
        )
        evaluated = evaluate_rows(merged)

        prediction_output_path = args.table_dir / f"correction_predictions_{split}_repair.csv"
        evaluated.to_csv(prediction_output_path, index=False, encoding="utf-8")
        print(f"[evaluate_correction_repairs] Wrote {prediction_output_path}")

        metric_rows.append(summarize_split(split, evaluated))
        by_error_tables.append(summarize_by_error_type(split, evaluated))

    repair_metrics = pd.DataFrame(metric_rows)
    repair_metrics_path = args.table_dir / "correction_repair_metrics.csv"
    repair_metrics.to_csv(repair_metrics_path, index=False, encoding="utf-8")
    print(f"[evaluate_correction_repairs] Wrote {repair_metrics_path}")

    repair_by_error = pd.concat(by_error_tables, ignore_index=True)
    repair_by_error_path = args.table_dir / "correction_repair_by_error_type.csv"
    repair_by_error.to_csv(repair_by_error_path, index=False, encoding="utf-8")
    print(f"[evaluate_correction_repairs] Wrote {repair_by_error_path}")

    if not args.no_update_correction_metrics:
        update_correction_metrics(
            metrics_path=args.table_dir / "correction_metrics.csv",
            repair_metrics=repair_metrics,
        )

    print("\n[evaluate_correction_repairs] Summary:")
    print(repair_metrics.to_string(index=False))
    print("[evaluate_correction_repairs] Done.")


if __name__ == "__main__":
    main()
"""
src/dyck_data.py
================

Synthetic data generator for Lab 4:
Transformers and Dyck Languages.

This script creates balanced and corrupted bracket strings for D(2), using
the two bracket types:

    ( ) and [ ]

Generated splits:
    data/train.csv
    data/dev.csv
    data/test_id.csv
    data/test_ood.csv
    data/dataset_summary.csv

Each row contains:
    - the input tokens
    - whether the sequence is valid
    - the corruption type, if any
    - the original valid sequence
    - the corrupted sequence
    - the repair action labels for the local correction task

Run from the project root:

    python src/dyck_data.py

For a quick smoke test:

    python src/dyck_data.py --train-size 1000 --dev-size 200 --test-size 200 --ood-size 200
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Bracket system
# ---------------------------------------------------------------------------

PAIRS: list[tuple[str, str]] = [("(", ")"), ("[", "]")]

OPENERS = {opener for opener, _ in PAIRS}
CLOSERS = {closer for _, closer in PAIRS}

OPENER_TO_CLOSER = dict(PAIRS)
CLOSER_TO_OPENER = {closer: opener for opener, closer in PAIRS}

TOKEN_NAME = {
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACK",
    "]": "RBRACK",
}


ERROR_TYPES = [
    "E1_missing_closer",
    "E2_spurious_opener",
    "E3_type_mismatch",
    "E4_premature_close",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorruptionResult:
    original_tokens: list[str]
    corrupted_tokens: list[str]
    error_type: str
    error_position: int
    correction_actions: list[str]


# ---------------------------------------------------------------------------
# Dyck-language utilities
# ---------------------------------------------------------------------------

def is_dyck(tokens: Iterable[str]) -> bool:
    """
    Return True iff the token sequence is a valid D(2) Dyck string.
    """
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
            raise ValueError(f"Unknown token: {token!r}")

    return len(stack) == 0


def max_nesting_depth(tokens: Iterable[str]) -> int:
    """
    Compute the maximum nesting depth of a valid or invalid bracket string.

    The value is based on openers encountered while scanning left-to-right.
    """
    depth = 0
    max_depth = 0

    for token in tokens:
        if token in OPENERS:
            depth += 1
            max_depth = max(max_depth, depth)
        elif token in CLOSERS:
            depth = max(0, depth - 1)

    return max_depth


def stack_before_each_gap(tokens: list[str]) -> list[list[str]]:
    """
    Return the stack of openers before each gap.

    For a sequence of length n, there are n + 1 gaps:
        gap 0: before token 0
        gap 1: between token 0 and token 1
        ...
        gap n: after the final token
    """
    stack: list[str] = []
    gaps: list[list[str]] = [stack.copy()]

    for token in tokens:
        if token in OPENERS:
            stack.append(token)
        elif token in CLOSERS:
            if stack and OPENER_TO_CLOSER[stack[-1]] == token:
                stack.pop()
        gaps.append(stack.copy())

    return gaps


# ---------------------------------------------------------------------------
# Valid Dyck generation
# ---------------------------------------------------------------------------

def generate_dyck_once(
    length: int,
    max_depth_limit: int,
    exact_depth: int | None = None,
) -> list[str] | None:
    """
    Generate one random Dyck string of exactly `length` bracket tokens.

    The generator enforces:
        - balanced output
        - maximum depth <= max_depth_limit
        - if exact_depth is given, maximum depth == exact_depth

    Returns None if the random choices lead to a dead end.
    """
    if length < 2 or length % 2 != 0:
        raise ValueError("Dyck strings must have positive even length.")

    if exact_depth is not None and exact_depth > max_depth_limit:
        raise ValueError("exact_depth cannot exceed max_depth_limit.")

    stack: list[tuple[str, str]] = []
    result: list[str] = []

    observed_max_depth = 0

    for step in range(length):
        remaining_slots = length - step

        must_close = len(stack) == remaining_slots
        can_open = (
            len(stack) < max_depth_limit
            and remaining_slots > len(stack) + 1
        )
        can_close = len(stack) > 0

        choices: list[str] = []

        if can_open and not must_close:
            choices.append("open")

        if can_close:
            choices.append("close")

        if not choices:
            return None

        choice = random.choice(choices)

        if choice == "open":
            opener, closer = random.choice(PAIRS)
            stack.append((opener, closer))
            result.append(opener)
            observed_max_depth = max(observed_max_depth, len(stack))
        else:
            _, closer = stack.pop()
            result.append(closer)

    if stack:
        return None

    if exact_depth is not None and observed_max_depth != exact_depth:
        return None

    return result


def random_even_length(length_min: int, length_max: int) -> int:
    """
    Sample an even length from the inclusive interval [length_min, length_max].
    """
    first_even = length_min if length_min % 2 == 0 else length_min + 1
    last_even = length_max if length_max % 2 == 0 else length_max - 1

    if first_even > last_even:
        raise ValueError(
            f"No even length available in interval [{length_min}, {length_max}]."
        )

    return random.randrange(first_even, last_even + 1, 2)


def generate_valid_dyck(
    length_min: int,
    length_max: int,
    max_depth_limit: int,
    exact_depth: int | None = None,
    max_attempts: int = 10_000,
) -> list[str]:
    """
    Generate a valid Dyck string satisfying the length/depth constraints.
    """
    for _ in range(max_attempts):
        length = random_even_length(length_min, length_max)
        tokens = generate_dyck_once(
            length=length,
            max_depth_limit=max_depth_limit,
            exact_depth=exact_depth,
        )
        if tokens is not None:
            return tokens

    raise RuntimeError(
        "Failed to generate a Dyck string with constraints: "
        f"length=[{length_min}, {length_max}], "
        f"max_depth_limit={max_depth_limit}, exact_depth={exact_depth}"
    )


# ---------------------------------------------------------------------------
# Corruption operations
# ---------------------------------------------------------------------------

def corrupt_missing_closer(tokens: list[str]) -> CorruptionResult:
    """
    E1: Delete one closing bracket.

    Repair action convention:
        INSERT_X means insert token X after the current position.
    """
    closer_positions = [i for i, token in enumerate(tokens) if token in CLOSERS]
    delete_position = random.choice(closer_positions)

    deleted_token = tokens[delete_position]
    corrupted = tokens[:delete_position] + tokens[delete_position + 1 :]

    correction_actions = ["OK"] * len(corrupted)

    # In a valid Dyck string, a closer cannot be the first token.
    insert_after = delete_position - 1
    correction_actions[insert_after] = f"INSERT_{TOKEN_NAME[deleted_token]}"

    return CorruptionResult(
        original_tokens=tokens,
        corrupted_tokens=corrupted,
        error_type="E1_missing_closer",
        error_position=insert_after,
        correction_actions=correction_actions,
    )


def corrupt_spurious_opener(tokens: list[str]) -> CorruptionResult:
    """
    E2: Insert one extra opening bracket.
    """
    insert_position = random.randint(0, len(tokens))
    inserted_token = random.choice(list(OPENERS))

    corrupted = tokens[:insert_position] + [inserted_token] + tokens[insert_position:]

    correction_actions = ["OK"] * len(corrupted)
    correction_actions[insert_position] = "DELETE"

    return CorruptionResult(
        original_tokens=tokens,
        corrupted_tokens=corrupted,
        error_type="E2_spurious_opener",
        error_position=insert_position,
        correction_actions=correction_actions,
    )


def corrupt_type_mismatch(tokens: list[str]) -> CorruptionResult:
    """
    E3: Replace one closing bracket with the wrong closing bracket type.
    """
    closer_positions = [i for i, token in enumerate(tokens) if token in CLOSERS]
    replace_position = random.choice(closer_positions)

    correct_token = tokens[replace_position]
    wrong_choices = [closer for closer in CLOSERS if closer != correct_token]
    wrong_token = random.choice(wrong_choices)

    corrupted = tokens.copy()
    corrupted[replace_position] = wrong_token

    correction_actions = ["OK"] * len(corrupted)
    correction_actions[replace_position] = f"REPLACE_{TOKEN_NAME[correct_token]}"

    return CorruptionResult(
        original_tokens=tokens,
        corrupted_tokens=corrupted,
        error_type="E3_type_mismatch",
        error_position=replace_position,
        correction_actions=correction_actions,
    )


def corrupt_premature_close(tokens: list[str]) -> CorruptionResult:
    """
    E4: Insert a closing bracket where no opener is available on the stack.

    We choose a gap where the current stack is empty. This creates an
    immediately premature closer.
    """
    gaps = stack_before_each_gap(tokens)
    empty_stack_gaps = [i for i, stack in enumerate(gaps) if len(stack) == 0]

    insert_position = random.choice(empty_stack_gaps)
    inserted_token = random.choice(list(CLOSERS))

    corrupted = tokens[:insert_position] + [inserted_token] + tokens[insert_position:]

    correction_actions = ["OK"] * len(corrupted)
    correction_actions[insert_position] = "DELETE"

    return CorruptionResult(
        original_tokens=tokens,
        corrupted_tokens=corrupted,
        error_type="E4_premature_close",
        error_position=insert_position,
        correction_actions=correction_actions,
    )


def corrupt_tokens(tokens: list[str], error_type: str) -> CorruptionResult:
    """
    Apply one requested corruption type.
    """
    if error_type == "E1_missing_closer":
        return corrupt_missing_closer(tokens)

    if error_type == "E2_spurious_opener":
        return corrupt_spurious_opener(tokens)

    if error_type == "E3_type_mismatch":
        return corrupt_type_mismatch(tokens)

    if error_type == "E4_premature_close":
        return corrupt_premature_close(tokens)

    raise ValueError(f"Unknown error type: {error_type}")


def generate_corrupted_example(
    length_min: int,
    length_max: int,
    max_depth_limit: int,
    error_type: str,
    exact_depth: int | None = None,
    max_attempts: int = 10_000,
) -> CorruptionResult:
    """
    Generate a corrupted example whose final input length still lies within
    the requested interval.
    """
    for _ in range(max_attempts):
        original = generate_valid_dyck(
            length_min=length_min,
            length_max=length_max,
            max_depth_limit=max_depth_limit,
            exact_depth=exact_depth,
        )
        corruption = corrupt_tokens(original, error_type)

        input_length = len(corruption.corrupted_tokens)

        if length_min <= input_length <= length_max and not is_dyck(
            corruption.corrupted_tokens
        ):
            return corruption

    raise RuntimeError(
        "Failed to generate corrupted example with constraints: "
        f"length=[{length_min}, {length_max}], "
        f"max_depth_limit={max_depth_limit}, "
        f"exact_depth={exact_depth}, error_type={error_type}"
    )


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def category_quotas(total: int, categories: list[str]) -> list[str]:
    """
    Return a category list of length `total`, as balanced as possible.
    """
    base = total // len(categories)
    remainder = total % len(categories)

    output: list[str] = []
    for index, category in enumerate(categories):
        count = base + (1 if index < remainder else 0)
        output.extend([category] * count)

    random.shuffle(output)
    return output


def tokens_to_string(tokens: list[str]) -> str:
    """
    Store token sequences in a readable space-separated format.
    """
    return " ".join(tokens)


def make_valid_row(
    row_id: int,
    split: str,
    tokens: list[str],
    target_depth: int | None,
) -> dict[str, object]:
    """
    Build one CSV row for a valid sequence.
    """
    return {
        "id": row_id,
        "split": split,
        "tokens": tokens_to_string(tokens),
        "label": 1,
        "error_type": "no_error",
        "target_depth": target_depth if target_depth is not None else max_nesting_depth(tokens),
        "max_depth": max_nesting_depth(tokens),
        "input_length": len(tokens),
        "original_length": len(tokens),
        "error_position": -1,
        "original": tokens_to_string(tokens),
        "corrupted": tokens_to_string(tokens),
        "correction_actions": json.dumps(["OK"] * len(tokens)),
    }


def make_corrupted_row(
    row_id: int,
    split: str,
    corruption: CorruptionResult,
    target_depth: int | None,
) -> dict[str, object]:
    """
    Build one CSV row for an invalid sequence.
    """
    return {
        "id": row_id,
        "split": split,
        "tokens": tokens_to_string(corruption.corrupted_tokens),
        "label": 0,
        "error_type": corruption.error_type,
        "target_depth": (
            target_depth
            if target_depth is not None
            else max_nesting_depth(corruption.original_tokens)
        ),
        "max_depth": max_nesting_depth(corruption.original_tokens),
        "input_length": len(corruption.corrupted_tokens),
        "original_length": len(corruption.original_tokens),
        "error_position": corruption.error_position,
        "original": tokens_to_string(corruption.original_tokens),
        "corrupted": tokens_to_string(corruption.corrupted_tokens),
        "correction_actions": json.dumps(corruption.correction_actions),
    }


def generate_split(
    split: str,
    size: int,
    length_min: int,
    length_max: int,
    max_depth_limit: int,
    exact_depth_values: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generate one dataset split.

    The split is balanced:
        50% valid examples
        50% corrupted examples

    Corrupted examples are balanced across E1-E4.
    """
    if size <= 0:
        raise ValueError("Split size must be positive.")

    valid_count = size // 2
    corrupted_count = size - valid_count

    rows: list[dict[str, object]] = []
    seen: set[tuple[int, str, str]] = set()

    valid_depth_schedule: list[int | None]
    corrupted_depth_schedule: list[int | None]

    if exact_depth_values is None:
        valid_depth_schedule = [None] * valid_count
        corrupted_depth_schedule = [None] * corrupted_count
    else:
        valid_depth_schedule = [
            int(value) for value in category_quotas(
                valid_count, [str(v) for v in exact_depth_values]
            )
        ]
        corrupted_depth_schedule = [
            int(value) for value in category_quotas(
                corrupted_count, [str(v) for v in exact_depth_values]
            )
        ]

    # Valid examples
    for target_depth in valid_depth_schedule:
        while True:
            tokens = generate_valid_dyck(
                length_min=length_min,
                length_max=length_max,
                max_depth_limit=max_depth_limit,
                exact_depth=target_depth,
            )
            key = (1, "no_error", tokens_to_string(tokens))
            if key not in seen:
                seen.add(key)
                rows.append(
                    make_valid_row(
                        row_id=len(rows),
                        split=split,
                        tokens=tokens,
                        target_depth=target_depth,
                    )
                )
                break

    # Corrupted examples
    error_schedule = category_quotas(corrupted_count, ERROR_TYPES)

    for error_type, target_depth in zip(error_schedule, corrupted_depth_schedule):
        while True:
            corruption = generate_corrupted_example(
                length_min=length_min,
                length_max=length_max,
                max_depth_limit=max_depth_limit,
                error_type=error_type,
                exact_depth=target_depth,
            )
            key = (
                0,
                corruption.error_type,
                tokens_to_string(corruption.corrupted_tokens),
            )
            if key not in seen:
                seen.add(key)
                rows.append(
                    make_corrupted_row(
                        row_id=len(rows),
                        split=split,
                        corruption=corruption,
                        target_depth=target_depth,
                    )
                )
                break

    random.shuffle(rows)

    # Reassign IDs after shuffling.
    for new_id, row in enumerate(rows):
        row["id"] = new_id

    return pd.DataFrame(rows)


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a compact summary table for sanity-checking.
    """
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "metric": "rows",
            "value": len(df),
        }
    )
    rows.append(
        {
            "metric": "valid",
            "value": int((df["label"] == 1).sum()),
        }
    )
    rows.append(
        {
            "metric": "invalid",
            "value": int((df["label"] == 0).sum()),
        }
    )
    rows.append(
        {
            "metric": "min_input_length",
            "value": int(df["input_length"].min()),
        }
    )
    rows.append(
        {
            "metric": "max_input_length",
            "value": int(df["input_length"].max()),
        }
    )
    rows.append(
        {
            "metric": "min_depth",
            "value": int(df["max_depth"].min()),
        }
    )
    rows.append(
        {
            "metric": "max_depth",
            "value": int(df["max_depth"].max()),
        }
    )

    for error_type, count in df["error_type"].value_counts().sort_index().items():
        rows.append(
            {
                "metric": f"error_type:{error_type}",
                "value": int(count),
            }
        )

    return pd.DataFrame(rows)


def validate_dataframe(df: pd.DataFrame, split: str) -> None:
    """
    Defensive validation before writing a split to disk.
    """
    required_columns = {
        "id",
        "split",
        "tokens",
        "label",
        "error_type",
        "target_depth",
        "max_depth",
        "input_length",
        "original_length",
        "error_position",
        "original",
        "corrupted",
        "correction_actions",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{split}: missing columns: {sorted(missing_columns)}")

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"{split}: expected both labels 0 and 1.")

    for _, row in df.iterrows():
        tokens = str(row["tokens"]).split()
        label = int(row["label"])
        actions = json.loads(str(row["correction_actions"]))

        if len(tokens) != int(row["input_length"]):
            raise ValueError(f"{split}: input_length mismatch at row {row['id']}.")

        if len(tokens) != len(actions):
            raise ValueError(f"{split}: action length mismatch at row {row['id']}.")

        membership = is_dyck(tokens)
        if label == 1 and not membership:
            raise ValueError(f"{split}: valid row is not Dyck at row {row['id']}.")

        if label == 0 and membership:
            raise ValueError(f"{split}: invalid row is Dyck at row {row['id']}.")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Dyck-language datasets for Lab 4."
    )

    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--dev-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--ood-size", type=int, default=5_000)

    parser.add_argument("--id-length-min", type=int, default=4)
    parser.add_argument("--id-length-max", type=int, default=40)
    parser.add_argument("--id-depth-max", type=int, default=4)

    parser.add_argument("--ood-length-min", type=int, default=40)
    parser.add_argument("--ood-length-max", type=int, default=80)
    parser.add_argument("--ood-depths", type=int, nargs="+", default=[5, 6, 7])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [
        {
            "name": "train",
            "size": args.train_size,
            "length_min": args.id_length_min,
            "length_max": args.id_length_max,
            "max_depth_limit": args.id_depth_max,
            "exact_depth_values": None,
            "filename": "train.csv",
        },
        {
            "name": "dev",
            "size": args.dev_size,
            "length_min": args.id_length_min,
            "length_max": args.id_length_max,
            "max_depth_limit": args.id_depth_max,
            "exact_depth_values": None,
            "filename": "dev.csv",
        },
        {
            "name": "test_id",
            "size": args.test_size,
            "length_min": args.id_length_min,
            "length_max": args.id_length_max,
            "max_depth_limit": args.id_depth_max,
            "exact_depth_values": None,
            "filename": "test_id.csv",
        },
        {
            "name": "test_ood",
            "size": args.ood_size,
            "length_min": args.ood_length_min,
            "length_max": args.ood_length_max,
            "max_depth_limit": max(args.ood_depths),
            "exact_depth_values": args.ood_depths,
            "filename": "test_ood.csv",
        },
    ]

    summary_tables: list[pd.DataFrame] = []

    for spec in split_specs:
        print(f"[dyck_data] Generating {spec['name']}...")

        df = generate_split(
            split=spec["name"],
            size=spec["size"],
            length_min=spec["length_min"],
            length_max=spec["length_max"],
            max_depth_limit=spec["max_depth_limit"],
            exact_depth_values=spec["exact_depth_values"],
        )

        validate_dataframe(df, spec["name"])

        output_path = args.out_dir / spec["filename"]
        df.to_csv(output_path, index=False, encoding="utf-8")

        summary = summarize_split(df)
        summary.insert(0, "split", spec["name"])
        summary_tables.append(summary)

        print(f"[dyck_data] Wrote {output_path} ({len(df):,} rows)")
        print(summary.to_string(index=False))
        print()

    full_summary = pd.concat(summary_tables, ignore_index=True)
    summary_path = args.out_dir / "dataset_summary.csv"
    full_summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"[dyck_data] Wrote {summary_path}")
    print("[dyck_data] Done.")


if __name__ == "__main__":
    main()

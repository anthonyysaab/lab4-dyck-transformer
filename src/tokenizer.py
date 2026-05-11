"""
src/tokenizer.py
================

Tokenizer utilities for Lab 4 Dyck-language experiments.

The model uses a small fixed vocabulary:

    [PAD], [CLS], [SEP], [UNK], (, ), [, ]

Input convention:
    [CLS] bracket_1 bracket_2 ... bracket_n [SEP] [PAD] ...

The correction task uses one action label per real bracket token.
Special tokens and padding positions receive IGNORE_INDEX.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
UNK_TOKEN = "[UNK]"

BRACKET_TOKENS = ["(", ")", "[", "]"]

VOCAB = [
    PAD_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
    UNK_TOKEN,
    *BRACKET_TOKENS,
]

TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
CLS_ID = TOKEN_TO_ID[CLS_TOKEN]
SEP_ID = TOKEN_TO_ID[SEP_TOKEN]
UNK_ID = TOKEN_TO_ID[UNK_TOKEN]


# ---------------------------------------------------------------------------
# Correction-action vocabulary
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100

ACTION_LABELS = [
    "OK",
    "DELETE",
    "INSERT_LPAREN",
    "INSERT_RPAREN",
    "INSERT_LBRACK",
    "INSERT_RBRACK",
    "REPLACE_LPAREN",
    "REPLACE_RPAREN",
    "REPLACE_LBRACK",
    "REPLACE_RBRACK",
]

ACTION_TO_ID = {label: idx for idx, label in enumerate(ACTION_LABELS)}
ID_TO_ACTION = {idx: label for label, idx in ACTION_TO_ID.items()}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodedSequence:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor


class DyckTokenizer:
    """
    Minimal tokenizer for bracket-only Dyck strings.

    Parameters
    ----------
    max_length:
        Total sequence length after adding [CLS] and [SEP].
        Lab 4 uses raw bracket lengths up to 80, so max_length=82 is safe.
    """

    def __init__(self, max_length: int = 82) -> None:
        if max_length < 4:
            raise ValueError("max_length must be at least 4.")

        self.max_length = max_length

    def split_tokens(self, text: str) -> list[str]:
        """
        Convert a space-separated bracket string into a token list.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text)!r}.")

        tokens = text.strip().split()

        for token in tokens:
            if token not in BRACKET_TOKENS:
                raise ValueError(f"Unknown bracket token: {token!r}")

        return tokens

    def encode_tokens(self, tokens: list[str]) -> EncodedSequence:
        """
        Encode a list of bracket tokens with [CLS], [SEP], and padding.
        """
        if len(tokens) + 2 > self.max_length:
            raise ValueError(
                f"Sequence too long: {len(tokens)} bracket tokens require "
                f"{len(tokens) + 2} positions, but max_length={self.max_length}."
            )

        ids = [CLS_ID]
        ids.extend(TOKEN_TO_ID.get(token, UNK_ID) for token in tokens)
        ids.append(SEP_ID)

        attention_mask = [1] * len(ids)

        while len(ids) < self.max_length:
            ids.append(PAD_ID)
            attention_mask.append(0)

        token_type_ids = [0] * self.max_length

        return EncodedSequence(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
        )

    def encode_text(self, text: str) -> EncodedSequence:
        """
        Encode a space-separated bracket string.
        """
        return self.encode_tokens(self.split_tokens(text))

    def decode_ids(self, input_ids: list[int] | torch.Tensor) -> list[str]:
        """
        Decode token ids back into token strings, keeping special tokens.
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().cpu().tolist()

        return [ID_TO_TOKEN.get(int(idx), UNK_TOKEN) for idx in input_ids]


# ---------------------------------------------------------------------------
# Correction-label encoding
# ---------------------------------------------------------------------------

def encode_correction_actions(
    correction_actions_json: str,
    max_length: int = 82,
) -> torch.Tensor:
    """
    Encode correction actions into a padded label tensor.

    The CSV stores one action per bracket token. For the Transformer sequence,
    we add:
        [CLS] -> IGNORE_INDEX
        each bracket token -> action id
        [SEP] -> IGNORE_INDEX
        [PAD] -> IGNORE_INDEX
    """
    actions = json.loads(correction_actions_json)

    if not isinstance(actions, list):
        raise ValueError("correction_actions must decode to a list.")

    label_ids = [IGNORE_INDEX]

    for action in actions:
        if action not in ACTION_TO_ID:
            raise ValueError(f"Unknown correction action: {action!r}")
        label_ids.append(ACTION_TO_ID[action])

    label_ids.append(IGNORE_INDEX)

    if len(label_ids) > max_length:
        raise ValueError(
            f"Correction label sequence too long: {len(label_ids)} > {max_length}."
        )

    while len(label_ids) < max_length:
        label_ids.append(IGNORE_INDEX)

    return torch.tensor(label_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Sanity-check CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Dyck tokenizer.")
    parser.add_argument("--csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--max-length", type=int, default=82)
    parser.add_argument("--row", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"Could not find {args.csv}. Generate data first with: "
            "python src/dyck_data.py"
        )

    df = pd.read_csv(args.csv)

    if args.row < 0 or args.row >= len(df):
        raise ValueError(f"--row must be between 0 and {len(df) - 1}.")

    row = df.iloc[args.row]

    tokenizer = DyckTokenizer(max_length=args.max_length)
    encoded = tokenizer.encode_text(str(row["tokens"]))
    correction_labels = encode_correction_actions(
        str(row["correction_actions"]),
        max_length=args.max_length,
    )

    print("[tokenizer] Input tokens:")
    print(row["tokens"])
    print()

    print("[tokenizer] Decoded model sequence:")
    print(tokenizer.decode_ids(encoded.input_ids[: int(row["input_length"]) + 2]))
    print()

    print("[tokenizer] input_ids shape:", tuple(encoded.input_ids.shape))
    print("[tokenizer] attention_mask shape:", tuple(encoded.attention_mask.shape))
    print("[tokenizer] correction_labels shape:", tuple(correction_labels.shape))
    print("[tokenizer] label:", int(row["label"]))
    print("[tokenizer] error_type:", row["error_type"])
    print("[tokenizer] OK")


if __name__ == "__main__":
    main()

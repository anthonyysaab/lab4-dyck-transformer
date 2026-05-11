"""
src/attention_analysis.py
=========================

Attention analysis for Lab 4 Dyck Transformer.

Goal:
    Check whether attention heads systematically attend between matching
    bracket pairs in valid Dyck strings.

For each valid example:
    - recover true matching opener/closer pairs using a stack
    - run the trained detection model
    - measure attention mass:
        closer -> matching opener
        opener -> matching closer

Outputs:
    outputs/attention/attention_pair_scores.csv
    outputs/attention/attention_head_summary.csv

Run:
    python src/attention_analysis.py --split test_id --max-examples 500

Also useful:
    python src/attention_analysis.py --split test_ood --max-examples 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import build_model
from tokenizer import DyckTokenizer


PAIRS = {
    "(": ")",
    "[": "]",
}

OPENERS = set(PAIRS.keys())
CLOSERS = set(PAIRS.values())


class AttentionDataset(Dataset):
    """
    Valid-only dataset for attention analysis.
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer: DyckTokenizer,
        max_examples: int,
    ) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        df = pd.read_csv(csv_path)
        df = df[df["label"].astype(int) == 1].copy()
        df = df.head(max_examples).copy()

        if df.empty:
            raise ValueError(f"No valid examples found in {csv_path}")

        self.metadata = df.reset_index(drop=True)
        self.tokenizer = tokenizer

        input_ids = []
        attention_masks = []

        for text in self.metadata["tokens"].astype(str).tolist():
            encoded = tokenizer.encode_text(text)
            input_ids.append(encoded.input_ids)
            attention_masks.append(encoded.attention_mask)

        self.input_ids = torch.stack(input_ids, dim=0)
        self.attention_masks = torch.stack(attention_masks, dim=0)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
        }


def matching_pairs(tokens: list[str]) -> list[tuple[int, int, str, str]]:
    """
    Return true matching bracket pairs as:
        (opener_index, closer_index, opener_token, closer_token)

    Indices are raw bracket-token indices, not model indices.
    """
    stack: list[tuple[int, str, str]] = []
    pairs: list[tuple[int, int, str, str]] = []

    for index, token in enumerate(tokens):
        if token in OPENERS:
            stack.append((index, token, PAIRS[token]))
            continue

        if token in CLOSERS:
            if not stack:
                raise ValueError("Invalid Dyck string: premature closer.")

            opener_index, opener_token, expected_closer = stack.pop()

            if token != expected_closer:
                raise ValueError("Invalid Dyck string: type mismatch.")

            pairs.append((opener_index, index, opener_token, token))
            continue

        raise ValueError(f"Unknown token: {token!r}")

    if stack:
        raise ValueError("Invalid Dyck string: missing closer.")

    return pairs


def resolve_csv(split: str) -> Path:
    mapping = {
        "dev": Path("data/dev.csv"),
        "test_id": Path("data/test_id.csv"),
        "test_ood": Path("data/test_ood.csv"),
    }

    if split not in mapping:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(mapping)}.")

    return mapping[split]


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    return torch.device(requested_device)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. "
            "Train detection first with: python src/train_detection.py"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get("config", {})

    model = build_model(
        max_length=int(config.get("max_length", 82)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        num_layers=int(config.get("num_layers", 2)),
        num_heads=int(config.get("num_heads", 4)),
        ff_dim=int(config.get("ff_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


@torch.no_grad()
def collect_attention_scores(
    model: torch.nn.Module,
    dataset: AttentionDataset,
    batch_size: int,
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    rows: list[dict[str, Any]] = []
    global_offset = 0

    for batch in tqdm(dataloader, desc="attention", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        # List length = num_layers.
        # Each tensor shape = [batch, heads, seq, seq].
        attentions = [layer_attention.detach().cpu() for layer_attention in output.attentions]

        batch_size_actual = input_ids.shape[0]

        for local_index in range(batch_size_actual):
            row = dataset.metadata.iloc[global_offset + local_index]
            tokens = str(row["tokens"]).split()
            pairs = matching_pairs(tokens)

            for pair_index, (opener_i, closer_i, opener, closer) in enumerate(pairs):
                # Model position offset:
                # raw token 0 is at model position 1 because position 0 is [CLS].
                opener_pos = opener_i + 1
                closer_pos = closer_i + 1
                distance = closer_i - opener_i

                for layer_index, layer_attention in enumerate(attentions):
                    num_heads = layer_attention.shape[1]

                    for head_index in range(num_heads):
                        attention_matrix = layer_attention[local_index, head_index]

                        closer_to_opener = float(attention_matrix[closer_pos, opener_pos])
                        opener_to_closer = float(attention_matrix[opener_pos, closer_pos])
                        closer_to_cls = float(attention_matrix[closer_pos, 0])
                        opener_to_cls = float(attention_matrix[opener_pos, 0])

                        rows.append(
                            {
                                "split": split_name,
                                "example_id": int(row["id"]),
                                "pair_index": pair_index,
                                "input_length": int(row["input_length"]),
                                "max_depth": int(row["max_depth"]),
                                "opener_index": opener_i,
                                "closer_index": closer_i,
                                "distance": distance,
                                "opener": opener,
                                "closer": closer,
                                "layer": layer_index,
                                "head": head_index,
                                "closer_to_matching_opener": closer_to_opener,
                                "opener_to_matching_closer": opener_to_closer,
                                "closer_to_cls": closer_to_cls,
                                "opener_to_cls": opener_to_cls,
                            }
                        )

        global_offset += batch_size_actual

    return pd.DataFrame(rows)


def summarize_attention(pair_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize attention score by layer/head.
    """
    summary = (
        pair_scores
        .groupby(["split", "layer", "head"], as_index=False)
        .agg(
            pairs=("pair_index", "count"),
            mean_closer_to_matching_opener=("closer_to_matching_opener", "mean"),
            median_closer_to_matching_opener=("closer_to_matching_opener", "median"),
            mean_opener_to_matching_closer=("opener_to_matching_closer", "mean"),
            median_opener_to_matching_closer=("opener_to_matching_closer", "median"),
            mean_closer_to_cls=("closer_to_cls", "mean"),
            mean_opener_to_cls=("opener_to_cls", "mean"),
            mean_distance=("distance", "mean"),
        )
        .sort_values(
            ["mean_closer_to_matching_opener", "mean_opener_to_matching_closer"],
            ascending=False,
        )
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze attention on matching brackets.")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/detection_best.pt"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_id",
        choices=["dev", "test_id", "test_ood"],
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv if args.csv is not None else resolve_csv(args.split)
    device = resolve_device(args.device)

    print("[attention] Device:", device)
    print("[attention] Split:", args.split)
    print("[attention] CSV:", csv_path)

    model, config = load_model(args.checkpoint, device=device)
    tokenizer = DyckTokenizer(max_length=int(config.get("max_length", 82)))

    dataset = AttentionDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_examples=args.max_examples,
    )

    print("[attention] Valid examples:", len(dataset))

    pair_scores = collect_attention_scores(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        split_name=args.split,
    )

    summary = summarize_attention(pair_scores)

    pair_scores_path = args.output_dir / f"attention_pair_scores_{args.split}.csv"
    summary_path = args.output_dir / f"attention_head_summary_{args.split}.csv"

    pair_scores.to_csv(pair_scores_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"[attention] Wrote {pair_scores_path}")
    print(f"[attention] Wrote {summary_path}")
    print()
    print("[attention] Top heads by closer -> matching opener attention:")
    print(summary.head(10).to_string(index=False))
    print("[attention] Done.")


if __name__ == "__main__":
    main()

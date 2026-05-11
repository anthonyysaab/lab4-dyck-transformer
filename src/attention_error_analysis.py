"""
src/attention_error_analysis.py
===============================

Direct attention analysis around corrupted positions.

This addresses the Lab 4 requirement to inspect attention on erroneous strings:
    what happens around the corrupted token/position?

For each invalid example, the script measures attention mass involving the
gold error_position:
    - [CLS] -> error token
    - error token -> [CLS]
    - error token -> previous/next token
    - previous/next token -> error token

Outputs:
    outputs/attention/error_attention_scores_<split>.csv
    outputs/attention/error_attention_summary_<split>.csv

Run:
    python src/attention_error_analysis.py --split test_id --max-examples 500
    python src/attention_error_analysis.py --split test_ood --max-examples 500
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


class ErrorAttentionDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer: DyckTokenizer, max_examples: int) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        df = pd.read_csv(csv_path)
        df = df[df["label"].astype(int) == 0].copy()
        df = df[df["error_position"].astype(int) >= 0].copy()
        df = df.head(max_examples).copy()

        if df.empty:
            raise ValueError(f"No corrupted examples found in {csv_path}")

        input_ids = []
        masks = []
        for text in df["tokens"].astype(str):
            encoded = tokenizer.encode_text(text)
            input_ids.append(encoded.input_ids)
            masks.append(encoded.attention_mask)

        self.input_ids = torch.stack(input_ids)
        self.attention_masks = torch.stack(masks)
        self.metadata = df.reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
        }


def resolve_csv(split: str) -> Path:
    mapping = {
        "dev": Path("data/dev.csv"),
        "test_id": Path("data/test_id.csv"),
        "test_ood": Path("data/test_ood.csv"),
    }
    if split not in mapping:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(mapping)}.")
    return mapping[split]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Train first with: python src/train_detection.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
def collect_scores(
    model: torch.nn.Module,
    dataset: ErrorAttentionDataset,
    batch_size: int,
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    rows: list[dict[str, object]] = []
    global_offset = 0

    for batch in tqdm(loader, desc="error attention", leave=False):
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        attentions = [x.detach().cpu() for x in output.attentions]

        for local_index in range(batch["input_ids"].shape[0]):
            meta = dataset.metadata.iloc[global_offset + local_index]
            tokens = str(meta["tokens"]).split()
            raw_error = int(meta["error_position"])

            # Model position offset: raw bracket 0 is model position 1 because [CLS] is position 0.
            error_pos = raw_error + 1
            prev_pos = error_pos - 1 if raw_error > 0 else None
            next_pos = error_pos + 1 if raw_error + 1 < len(tokens) else None

            for layer_index, layer_attention in enumerate(attentions):
                for head_index in range(layer_attention.shape[1]):
                    matrix = layer_attention[local_index, head_index]

                    row = {
                        "split": split_name,
                        "example_id": int(meta["id"]),
                        "error_type": str(meta["error_type"]),
                        "input_length": int(meta["input_length"]),
                        "max_depth": int(meta["max_depth"]),
                        "raw_error_position": raw_error,
                        "error_token": tokens[raw_error] if 0 <= raw_error < len(tokens) else "",
                        "layer": layer_index,
                        "head": head_index,
                        "cls_to_error": float(matrix[0, error_pos]),
                        "error_to_cls": float(matrix[error_pos, 0]),
                    }

                    if prev_pos is not None:
                        row["error_to_previous"] = float(matrix[error_pos, prev_pos])
                        row["previous_to_error"] = float(matrix[prev_pos, error_pos])
                    else:
                        row["error_to_previous"] = float("nan")
                        row["previous_to_error"] = float("nan")

                    if next_pos is not None:
                        row["error_to_next"] = float(matrix[error_pos, next_pos])
                        row["next_to_error"] = float(matrix[next_pos, error_pos])
                    else:
                        row["error_to_next"] = float("nan")
                        row["next_to_error"] = float("nan")

                    rows.append(row)

        global_offset += batch["input_ids"].shape[0]

    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["split", "error_type", "layer", "head"], as_index=False)
        .agg(
            examples=("example_id", "count"),
            mean_cls_to_error=("cls_to_error", "mean"),
            std_cls_to_error=("cls_to_error", "std"),
            mean_error_to_cls=("error_to_cls", "mean"),
            std_error_to_cls=("error_to_cls", "std"),
            mean_error_to_previous=("error_to_previous", "mean"),
            std_error_to_previous=("error_to_previous", "std"),
            mean_previous_to_error=("previous_to_error", "mean"),
            std_previous_to_error=("previous_to_error", "std"),
            mean_error_to_next=("error_to_next", "mean"),
            std_error_to_next=("error_to_next", "std"),
            mean_next_to_error=("next_to_error", "mean"),
            std_next_to_error=("next_to_error", "std"),
        )
        .sort_values(["error_type", "mean_cls_to_error"], ascending=[True, False])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze attention around corrupted positions.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/detection_best.pt"))
    parser.add_argument("--split", choices=["dev", "test_id", "test_ood"], default="test_id")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv if args.csv is not None else resolve_csv(args.split)
    device = resolve_device(args.device)

    model, config = load_model(args.checkpoint, device)
    tokenizer = DyckTokenizer(max_length=int(config.get("max_length", 82)))

    dataset = ErrorAttentionDataset(csv_path, tokenizer, args.max_examples)
    scores = collect_scores(model, dataset, args.batch_size, device, args.split)
    summary = summarize(scores)

    scores_path = args.output_dir / f"error_attention_scores_{args.split}.csv"
    summary_path = args.output_dir / f"error_attention_summary_{args.split}.csv"

    scores.to_csv(scores_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"[attention_error_analysis] Wrote {scores_path}")
    print(f"[attention_error_analysis] Wrote {summary_path}")


if __name__ == "__main__":
    main()

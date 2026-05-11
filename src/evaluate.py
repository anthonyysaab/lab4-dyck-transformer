"""
src/evaluate.py
===============

Grouped evaluation for the Lab 4 Dyck Transformer detector.

This script reloads a trained detection checkpoint and evaluates it on:

    data/dev.csv
    data/test_id.csv
    data/test_ood.csv

It produces report-ready CSV tables:

    outputs/tables/eval_overall_metrics.csv
    outputs/tables/eval_by_error_type.csv
    outputs/tables/eval_by_depth.csv
    outputs/tables/eval_by_length_bin.csv
    outputs/tables/eval_predictions_*.csv

Run:

    python src/evaluate.py

After a smoke-test checkpoint:

    python src/evaluate.py --limit 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import build_model
from tokenizer import DyckTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DyckEvalDataset(Dataset):
    """
    Torch dataset for evaluation.
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer: DyckTokenizer,
        limit: int | None = None,
    ) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        df = pd.read_csv(csv_path)

        if limit is not None:
            df = df.head(limit).copy()

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
        }

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        input_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []

        for text in df["tokens"].astype(str).tolist():
            encoded = tokenizer.encode_text(text)
            input_ids.append(encoded.input_ids)
            attention_masks.append(encoded.attention_mask)

        self.input_ids = torch.stack(input_ids, dim=0)
        self.attention_masks = torch.stack(attention_masks, dim=0)
        self.labels = torch.tensor(df["label"].astype(int).to_numpy(), dtype=torch.long)
        self.metadata = df.copy()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index],
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """
    Compute stable binary classification metrics.
    """
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "valid_f1": float(f1_score(gold, pred, pos_label=1, zero_division=0)),
        "invalid_f1": float(f1_score(1 - gold, 1 - pred, pos_label=1, zero_division=0)),
    }


def safe_group_metrics(
    df: pd.DataFrame,
    group_column: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Compute metrics for each group value.
    """
    rows: list[dict[str, Any]] = []

    for group_value, group_df in df.groupby(group_column, dropna=False):
        gold = group_df["label"].astype(int).to_numpy()
        pred = group_df["prediction"].astype(int).to_numpy()

        metrics = binary_metrics(gold, pred)

        rows.append(
            {
                "split": split_name,
                "group_column": group_column,
                "group_value": group_value,
                "rows": len(group_df),
                "gold_valid": int((gold == 1).sum()),
                "gold_invalid": int((gold == 0).sum()),
                "pred_valid": int((pred == 1).sum()),
                "pred_invalid": int((pred == 0).sum()),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def add_length_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add coarse input-length bins for report-level analysis.
    """
    output = df.copy()

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 10_000]
    labels = [
        "01-10",
        "11-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "81+",
    ]

    output["length_bin"] = pd.cut(
        output["input_length"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype(str)

    return output


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    return torch.device(requested_device)


def load_detection_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Load the trained detection checkpoint.

    PyTorch 2.6+ defaults to weights_only=True. Our own checkpoint stores a
    small config dictionary with Path objects, so we explicitly load with
    weights_only=False. This is safe for checkpoints created locally by this
    project.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. "
            "Train first with: python src/train_detection.py"
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
def predict(
    model: torch.nn.Module,
    dataset: DyckEvalDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run model inference and return predicted labels.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    predictions: list[int] = []

    for batch in tqdm(dataloader, desc="predict", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_predictions = output.detection_logits.argmax(dim=-1)

        predictions.extend(batch_predictions.detach().cpu().tolist())

    return np.asarray(predictions, dtype=np.int64)


def evaluate_split(
    split_name: str,
    csv_path: Path,
    model: torch.nn.Module,
    tokenizer: DyckTokenizer,
    batch_size: int,
    device: torch.device,
    table_dir: Path,
    limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate one split and return overall + grouped metric tables.
    """
    print(f"[evaluate] Evaluating {split_name} from {csv_path}")

    dataset = DyckEvalDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        limit=limit,
    )

    predictions = predict(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
    )

    df = dataset.metadata.copy()
    df["prediction"] = predictions
    df["correct"] = df["prediction"].astype(int) == df["label"].astype(int)
    df = add_length_bins(df)

    gold = df["label"].astype(int).to_numpy()
    pred = df["prediction"].astype(int).to_numpy()

    metrics = binary_metrics(gold, pred)

    cm = confusion_matrix(gold, pred)
    tn, fp, fn, tp = cm.ravel()

    overall_df = pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": len(df),
                "gold_valid": int((gold == 1).sum()),
                "gold_invalid": int((gold == 0).sum()),
                "pred_valid": int((pred == 1).sum()),
                "pred_invalid": int((pred == 0).sum()),
                "true_invalid": int(tn),
                "false_valid": int(fp),
                "false_invalid": int(fn),
                "true_valid": int(tp),
                **metrics,
            }
        ]
    )

    by_error_type = safe_group_metrics(df, "error_type", split_name)
    by_depth = safe_group_metrics(df, "max_depth", split_name)
    by_length_bin = safe_group_metrics(df, "length_bin", split_name)

    prediction_path = table_dir / f"eval_predictions_{split_name}.csv"
    df.to_csv(prediction_path, index=False, encoding="utf-8")

    cm_path = table_dir / f"eval_confusion_matrix_{split_name}.csv"
    pd.DataFrame(
        cm,
        index=["gold_invalid", "gold_valid"],
        columns=["pred_invalid", "pred_valid"],
    ).to_csv(cm_path, encoding="utf-8")

    print(
        f"[evaluate] {split_name}: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"[evaluate] Wrote {prediction_path}")
    print(f"[evaluate] Wrote {cm_path}")

    return overall_df, by_error_type, by_depth, by_length_bin


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped evaluation for Dyck detector.")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/detection_best.pt"),
    )

    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-id-csv", type=Path, default=Path("data/test_id.csv"))
    parser.add_argument("--test-ood-csv", type=Path, default=Path("data/test_ood.csv"))

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)

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

    device = resolve_device(args.device)
    print("[evaluate] Device:", device)

    model, config = load_detection_model(args.checkpoint, device=device)
    max_length = int(config.get("max_length", 82))

    tokenizer = DyckTokenizer(max_length=max_length)

    split_specs = [
        ("dev", args.dev_csv),
        ("test_id", args.test_id_csv),
        ("test_ood", args.test_ood_csv),
    ]

    overall_tables: list[pd.DataFrame] = []
    error_tables: list[pd.DataFrame] = []
    depth_tables: list[pd.DataFrame] = []
    length_tables: list[pd.DataFrame] = []

    for split_name, csv_path in split_specs:
        overall_df, by_error_type, by_depth, by_length_bin = evaluate_split(
            split_name=split_name,
            csv_path=csv_path,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=device,
            table_dir=args.output_dir,
            limit=args.limit,
        )

        overall_tables.append(overall_df)
        error_tables.append(by_error_type)
        depth_tables.append(by_depth)
        length_tables.append(by_length_bin)

    output_specs = [
        ("eval_overall_metrics.csv", overall_tables),
        ("eval_by_error_type.csv", error_tables),
        ("eval_by_depth.csv", depth_tables),
        ("eval_by_length_bin.csv", length_tables),
    ]

    for filename, tables in output_specs:
        path = args.output_dir / filename
        pd.concat(tables, ignore_index=True).to_csv(path, index=False, encoding="utf-8")
        print(f"[evaluate] Wrote {path}")

    print("[evaluate] Done.")


if __name__ == "__main__":
    main()

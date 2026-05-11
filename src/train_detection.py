"""
src/train_detection.py
======================

Train the binary error-detection model for Lab 4.

Task:
    Given a bracket sequence, classify whether it is a valid D(2) Dyck string
    or a corrupted one.

Inputs:
    data/train.csv
    data/dev.csv
    data/test_id.csv
    data/test_ood.csv

Outputs:
    outputs/checkpoints/detection_best.pt
    outputs/tables/detection_metrics.csv
    outputs/tables/detection_predictions_*.csv

Run a quick CPU smoke test:

    python src/train_detection.py --epochs 1 --train-limit 5000 --dev-limit 1000 --test-limit 1000 --batch-size 128

Run fuller training:

    python src/train_detection.py --epochs 5 --batch-size 128
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import build_model
from tokenizer import DyckTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    train_csv: Path
    dev_csv: Path
    test_id_csv: Path
    test_ood_csv: Path

    output_dir: Path
    checkpoint_dir: Path
    table_dir: Path

    max_length: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout: float

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float

    train_limit: int | None
    dev_limit: int | None
    test_limit: int | None

    seed: int
    device: str


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Make training as reproducible as practical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DyckDetectionDataset(Dataset):
    """
    Torch dataset for binary Dyck error detection.
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
            "max_depth",
            "input_length",
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

        self.metadata = df[
            ["id", "split", "tokens", "label", "error_type", "max_depth", "input_length"]
        ].copy()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index],
        }


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    """
    Train for one epoch.
    """
    model.train()

    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(dataloader, desc="train", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = loss_fn(output.detection_logits, batch["labels"])
        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        predictions = output.detection_logits.argmax(dim=-1)

        total_loss += float(loss.item()) * batch["labels"].shape[0]
        all_labels.extend(batch["labels"].detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

        progress.set_postfix(loss=float(loss.item()))

    mean_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Evaluate model and return metrics plus predictions.
    """
    model.eval()

    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(dataloader, desc="eval", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = loss_fn(output.detection_logits, batch["labels"])
        predictions = output.detection_logits.argmax(dim=-1)

        total_loss += float(loss.item()) * batch["labels"].shape[0]
        all_labels.extend(batch["labels"].detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    mean_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    metrics = {
        "loss": mean_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

    return metrics, np.asarray(all_predictions, dtype=np.int64)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Windows-safe DataLoader.

    num_workers=0 avoids multiprocessing issues on Windows/PowerShell.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def save_predictions(
    dataset: DyckDetectionDataset,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save per-example predictions for later error analysis.
    """
    df = dataset.metadata.copy()
    df["prediction"] = predictions
    df["correct"] = df["prediction"].astype(int) == df["label"].astype(int)
    df.to_csv(output_path, index=False, encoding="utf-8")


def save_checkpoint(
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    dev_metrics: dict[str, float],
    output_path: Path,
) -> None:
    """
    Save model weights and enough metadata to reload the experiment.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "dev_metrics": dev_metrics,
    }

    torch.save(payload, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dyck error detector.")

    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-id-csv", type=Path, default=Path("data/test_id.csv"))
    parser.add_argument("--test-ood-csv", type=Path, default=Path("data/test_ood.csv"))

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    parser.add_argument("--max-length", type=int, default=82)

    # CPU-friendly default model.
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    table_dir = output_dir / "tables"

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    return TrainConfig(
        train_csv=args.train_csv,
        dev_csv=args.dev_csv,
        test_id_csv=args.test_id_csv,
        test_ood_csv=args.test_ood_csv,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        table_dir=table_dir,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        device=device,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    set_seed(config.seed)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.table_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)

    print("[train_detection] Device:", device)
    print("[train_detection] Loading datasets...")

    tokenizer = DyckTokenizer(max_length=config.max_length)

    train_dataset = DyckDetectionDataset(
        csv_path=config.train_csv,
        tokenizer=tokenizer,
        limit=config.train_limit,
    )
    dev_dataset = DyckDetectionDataset(
        csv_path=config.dev_csv,
        tokenizer=tokenizer,
        limit=config.dev_limit,
    )
    test_id_dataset = DyckDetectionDataset(
        csv_path=config.test_id_csv,
        tokenizer=tokenizer,
        limit=config.test_limit,
    )
    test_ood_dataset = DyckDetectionDataset(
        csv_path=config.test_ood_csv,
        tokenizer=tokenizer,
        limit=config.test_limit,
    )

    print("[train_detection] Dataset sizes:")
    print("  train:   ", len(train_dataset))
    print("  dev:     ", len(dev_dataset))
    print("  test_id: ", len(test_id_dataset))
    print("  test_ood:", len(test_ood_dataset))

    train_loader = make_dataloader(train_dataset, config.batch_size, shuffle=True)
    dev_loader = make_dataloader(dev_dataset, config.batch_size, shuffle=False)
    test_id_loader = make_dataloader(test_id_dataset, config.batch_size, shuffle=False)
    test_ood_loader = make_dataloader(test_ood_dataset, config.batch_size, shuffle=False)

    model = build_model(
        max_length=config.max_length,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    ).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("[train_detection] Model parameters:", f"{parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    metrics_rows: list[dict[str, Any]] = []
    best_dev_accuracy = -1.0
    best_checkpoint_path = config.checkpoint_dir / "detection_best.pt"

    for epoch in range(1, config.epochs + 1):
        print(f"\n[train_detection] Epoch {epoch}/{config.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_clip=config.gradient_clip,
        )

        dev_metrics, _ = evaluate(
            model=model,
            dataloader=dev_loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            "[train_detection] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} "
            f"dev_f1={dev_metrics['macro_f1']:.4f}"
        )

        metrics_rows.append(
            {
                "split": "train",
                "epoch": epoch,
                **train_metrics,
            }
        )
        metrics_rows.append(
            {
                "split": "dev",
                "epoch": epoch,
                **dev_metrics,
            }
        )

        if dev_metrics["accuracy"] > best_dev_accuracy:
            best_dev_accuracy = dev_metrics["accuracy"]
            save_checkpoint(
                model=model,
                config=config,
                epoch=epoch,
                dev_metrics=dev_metrics,
                output_path=best_checkpoint_path,
            )
            print(f"[train_detection] Saved best checkpoint: {best_checkpoint_path}")

    print("\n[train_detection] Loading best checkpoint for final evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_eval_specs = [
        ("test_id", test_id_dataset, test_id_loader),
        ("test_ood", test_ood_dataset, test_ood_loader),
    ]

    for split_name, dataset, loader in final_eval_specs:
        split_metrics, predictions = evaluate(
            model=model,
            dataloader=loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"[train_detection] {split_name}: "
            f"loss={split_metrics['loss']:.4f} "
            f"acc={split_metrics['accuracy']:.4f} "
            f"f1={split_metrics['macro_f1']:.4f}"
        )

        metrics_rows.append(
            {
                "split": split_name,
                "epoch": "best",
                **split_metrics,
            }
        )

        predictions_path = config.table_dir / f"detection_predictions_{split_name}.csv"
        save_predictions(dataset, predictions, predictions_path)
        print(f"[train_detection] Wrote {predictions_path}")

        labels = dataset.labels.cpu().numpy()

        print(f"\n[train_detection] Classification report for {split_name}:")
        print(
            classification_report(
                labels,
                predictions,
                target_names=["invalid", "valid"],
                digits=4,
            )
        )

        cm = confusion_matrix(labels, predictions)
        cm_path = config.table_dir / f"detection_confusion_matrix_{split_name}.csv"
        pd.DataFrame(
            cm,
            index=["gold_invalid", "gold_valid"],
            columns=["pred_invalid", "pred_valid"],
        ).to_csv(cm_path, encoding="utf-8")
        print(f"[train_detection] Wrote {cm_path}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = config.table_dir / "detection_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"\n[train_detection] Wrote {metrics_path}")
    print("[train_detection] Done.")


if __name__ == "__main__":
    main()

"""
src/train_correction.py
=======================

Train the local correction model for Lab 4.

Task:
    For each bracket-token position, predict a repair action:

        OK
        DELETE
        INSERT_LPAREN / INSERT_RPAREN / INSERT_LBRACK / INSERT_RBRACK
        REPLACE_LPAREN / REPLACE_RPAREN / REPLACE_LBRACK / REPLACE_RBRACK

Special tokens [CLS], [SEP], and [PAD] are ignored with IGNORE_INDEX.

Outputs:
    outputs/checkpoints/correction_best.pt
    outputs/tables/correction_metrics.csv
    outputs/tables/correction_predictions_*.csv

Smoke test:
    python src/train_correction.py --epochs 1 --train-limit 5000 --dev-limit 1000 --test-limit 1000 --batch-size 128

Full training:
    python src/train_correction.py --epochs 5 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import build_model
from tokenizer import (
    ACTION_LABELS,
    ID_TO_ACTION,
    IGNORE_INDEX,
    DyckTokenizer,
    encode_correction_actions,
)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DyckCorrectionDataset(Dataset):
    """
    Torch dataset for token-level correction.
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
            "error_position",
            "correction_actions",
        }

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        input_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        correction_labels: list[torch.Tensor] = []

        for _, row in df.iterrows():
            encoded = tokenizer.encode_text(str(row["tokens"]))
            labels = encode_correction_actions(
                str(row["correction_actions"]),
                max_length=tokenizer.max_length,
            )

            input_ids.append(encoded.input_ids)
            attention_masks.append(encoded.attention_mask)
            correction_labels.append(labels)

        self.input_ids = torch.stack(input_ids, dim=0)
        self.attention_masks = torch.stack(attention_masks, dim=0)
        self.correction_labels = torch.stack(correction_labels, dim=0)

        self.metadata = df[
            [
                "id",
                "split",
                "tokens",
                "label",
                "error_type",
                "max_depth",
                "input_length",
                "error_position",
                "correction_actions",
            ]
        ].copy()

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "correction_labels": self.correction_labels[index],
        }


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def flatten_valid_positions(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only bracket-token positions, ignoring [CLS], [SEP], and [PAD].
    """
    mask = labels != IGNORE_INDEX

    flat_labels = labels[mask].detach().cpu().numpy()
    flat_predictions = predictions[mask].detach().cpu().numpy()

    return flat_labels, flat_predictions


def compute_token_metrics(
    gold: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    return {
        "token_accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, pred, average="weighted", zero_division=0)),
    }


def sequence_exact_match(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> float:
    """
    Fraction of examples where every non-ignored token action is correct.
    """
    labels_cpu = labels.detach().cpu()
    predictions_cpu = predictions.detach().cpu()

    exact = []

    for gold_row, pred_row in zip(labels_cpu, predictions_cpu):
        mask = gold_row != IGNORE_INDEX
        exact.append(bool(torch.equal(gold_row[mask], pred_row[mask])))

    return float(np.mean(exact))


def error_position_accuracy(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> float:
    """
    Accuracy restricted to non-OK gold action positions.

    This is the most important correction metric because OK positions dominate.
    """
    labels_cpu = labels.detach().cpu()
    predictions_cpu = predictions.detach().cpu()

    # Gold action 0 is OK. Ignore special/pad labels too.
    mask = (labels_cpu != IGNORE_INDEX) & (labels_cpu != 0)

    if int(mask.sum().item()) == 0:
        return 0.0

    correct = predictions_cpu[mask] == labels_cpu[mask]
    return float(correct.float().mean().item())


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    all_gold: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    exact_scores: list[float] = []
    error_pos_scores: list[float] = []

    for batch in tqdm(dataloader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = output.correction_logits
        labels = batch["correction_labels"]

        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        predictions = logits.argmax(dim=-1)

        gold_np, pred_np = flatten_valid_positions(labels, predictions)
        all_gold.append(gold_np)
        all_pred.append(pred_np)

        exact_scores.append(sequence_exact_match(labels, predictions))
        error_pos_scores.append(error_position_accuracy(labels, predictions))

        total_loss += float(loss.item()) * labels.shape[0]

    gold = np.concatenate(all_gold)
    pred = np.concatenate(all_pred)

    metrics = compute_token_metrics(gold, pred)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    metrics["sequence_exact_match"] = float(np.mean(exact_scores))
    metrics["error_position_accuracy"] = float(np.mean(error_pos_scores))

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[list[int]]]:
    model.eval()

    total_loss = 0.0
    all_gold: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    exact_scores: list[float] = []
    error_pos_scores: list[float] = []
    all_prediction_rows: list[list[int]] = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = output.correction_logits
        labels = batch["correction_labels"]

        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

        predictions = logits.argmax(dim=-1)

        gold_np, pred_np = flatten_valid_positions(labels, predictions)
        all_gold.append(gold_np)
        all_pred.append(pred_np)

        exact_scores.append(sequence_exact_match(labels, predictions))
        error_pos_scores.append(error_position_accuracy(labels, predictions))

        total_loss += float(loss.item()) * labels.shape[0]

        for pred_row, gold_row in zip(predictions.detach().cpu(), labels.detach().cpu()):
            mask = gold_row != IGNORE_INDEX
            all_prediction_rows.append(pred_row[mask].tolist())

    gold = np.concatenate(all_gold)
    pred = np.concatenate(all_pred)

    metrics = compute_token_metrics(gold, pred)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    metrics["sequence_exact_match"] = float(np.mean(exact_scores))
    metrics["error_position_accuracy"] = float(np.mean(error_pos_scores))

    return metrics, all_prediction_rows


def save_checkpoint(
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    dev_metrics: dict[str, float],
    output_path: Path,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "dev_metrics": dev_metrics,
        },
        output_path,
    )


def decode_action_row(action_ids: list[int]) -> list[str]:
    return [ID_TO_ACTION.get(int(idx), "UNKNOWN") for idx in action_ids]


def save_predictions(
    dataset: DyckCorrectionDataset,
    prediction_rows: list[list[int]],
    output_path: Path,
) -> None:
    df = dataset.metadata.copy()

    decoded_predictions = [decode_action_row(row) for row in prediction_rows]
    gold_actions = [json.loads(text) for text in df["correction_actions"].astype(str)]

    exact = [
        predicted == gold
        for predicted, gold in zip(decoded_predictions, gold_actions)
    ]

    df["predicted_actions"] = [json.dumps(row) for row in decoded_predictions]
    df["sequence_exact_match"] = exact

    df.to_csv(output_path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dyck correction model.")

    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-id-csv", type=Path, default=Path("data/test_id.csv"))
    parser.add_argument("--test-ood-csv", type=Path, default=Path("data/test_ood.csv"))

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    parser.add_argument("--max-length", type=int, default=82)

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

    print("[train_correction] Device:", device)
    print("[train_correction] Loading datasets...")

    tokenizer = DyckTokenizer(max_length=config.max_length)

    train_dataset = DyckCorrectionDataset(config.train_csv, tokenizer, config.train_limit)
    dev_dataset = DyckCorrectionDataset(config.dev_csv, tokenizer, config.dev_limit)
    test_id_dataset = DyckCorrectionDataset(config.test_id_csv, tokenizer, config.test_limit)
    test_ood_dataset = DyckCorrectionDataset(config.test_ood_csv, tokenizer, config.test_limit)

    print("[train_correction] Dataset sizes:")
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
    print("[train_correction] Model parameters:", f"{parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    metrics_rows: list[dict[str, Any]] = []
    best_dev_score = -1.0
    best_checkpoint_path = config.checkpoint_dir / "correction_best.pt"

    for epoch in range(1, config.epochs + 1):
        print(f"\n[train_correction] Epoch {epoch}/{config.epochs}")

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
            "[train_correction] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_token_acc={train_metrics['token_accuracy']:.4f} "
            f"train_error_pos_acc={train_metrics['error_position_accuracy']:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_token_acc={dev_metrics['token_accuracy']:.4f} "
            f"dev_error_pos_acc={dev_metrics['error_position_accuracy']:.4f} "
            f"dev_exact={dev_metrics['sequence_exact_match']:.4f}"
        )

        metrics_rows.append({"split": "train", "epoch": epoch, **train_metrics})
        metrics_rows.append({"split": "dev", "epoch": epoch, **dev_metrics})

        # Error-position accuracy is the key metric because OK tokens dominate.
        if dev_metrics["error_position_accuracy"] > best_dev_score:
            best_dev_score = dev_metrics["error_position_accuracy"]
            save_checkpoint(
                model=model,
                config=config,
                epoch=epoch,
                dev_metrics=dev_metrics,
                output_path=best_checkpoint_path,
            )
            print(f"[train_correction] Saved best checkpoint: {best_checkpoint_path}")

    print("\n[train_correction] Loading best checkpoint for final evaluation...")
    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    final_specs = [
        ("test_id", test_id_dataset, test_id_loader),
        ("test_ood", test_ood_dataset, test_ood_loader),
    ]

    for split_name, dataset, loader in final_specs:
        split_metrics, prediction_rows = evaluate(
            model=model,
            dataloader=loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"[train_correction] {split_name}: "
            f"loss={split_metrics['loss']:.4f} "
            f"token_acc={split_metrics['token_accuracy']:.4f} "
            f"error_pos_acc={split_metrics['error_position_accuracy']:.4f} "
            f"exact={split_metrics['sequence_exact_match']:.4f}"
        )

        metrics_rows.append({"split": split_name, "epoch": "best", **split_metrics})

        predictions_path = config.table_dir / f"correction_predictions_{split_name}.csv"
        save_predictions(dataset, prediction_rows, predictions_path)
        print(f"[train_correction] Wrote {predictions_path}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = config.table_dir / "correction_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    print(f"\n[train_correction] Wrote {metrics_path}")
    print("[train_correction] Done.")


if __name__ == "__main__":
    main()

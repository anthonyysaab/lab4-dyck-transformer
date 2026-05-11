"""
src/train_detection_curriculum.py
=================================

Curriculum-learning variant for OOD generalisation.

This implements one architectural/training modification requested by Lab 4:
    train the detector in stages by increasing maximum nesting depth.

Instead of sampling all depth <= 4 examples from the beginning, the model sees:
    stage 1: max_depth <= 1
    stage 2: max_depth <= 2
    stage 3: max_depth <= 3
    stage 4: max_depth <= 4

Outputs:
    outputs/checkpoints/detection_curriculum.pt
    outputs/tables/curriculum_detection_metrics.csv
    outputs/tables/curriculum_detection_by_depth.csv

Run:
    python src/train_detection_curriculum.py --epochs-per-stage 1 --batch-size 128
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
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import build_model
from tokenizer import DyckTokenizer


@dataclass(frozen=True)
class CurriculumConfig:
    train_csv: Path
    dev_csv: Path
    test_id_csv: Path
    test_ood_csv: Path
    output_dir: Path
    max_length: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout: float
    epochs_per_stage: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    train_limit_per_stage: int | None
    eval_limit: int | None
    seed: int
    device: str


class DyckDetectionDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        tokenizer: DyckTokenizer,
        limit: int | None = None,
        max_depth_allowed: int | None = None,
    ) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        df = pd.read_csv(csv_path)
        if max_depth_allowed is not None:
            df = df[df["max_depth"].astype(int) <= max_depth_allowed].copy()
        if limit is not None:
            df = df.head(limit).copy()
        if df.empty:
            raise ValueError(f"No rows left in {csv_path} after filtering.")

        input_ids = []
        masks = []
        for text in df["tokens"].astype(str):
            encoded = tokenizer.encode_text(text)
            input_ids.append(encoded.input_ids)
            masks.append(encoded.attention_mask)

        self.input_ids = torch.stack(input_ids)
        self.attention_masks = torch.stack(masks)
        self.labels = torch.tensor(df["label"].astype(int).to_numpy(), dtype=torch.long)
        self.metadata = df.reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def binary_metrics(gold: list[int] | np.ndarray, pred: list[int] | np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    gold_all = []
    pred_all = []

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output.detection_logits, labels)
        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        pred = output.detection_logits.argmax(dim=-1)
        total_loss += float(loss.item()) * labels.shape[0]
        gold_all.extend(labels.detach().cpu().tolist())
        pred_all.extend(pred.detach().cpu().tolist())

    out = {"loss": total_loss / len(loader.dataset)}
    out.update(binary_metrics(gold_all, pred_all))
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: DyckDetectionDataset,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    gold_all = []
    pred_all = []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output.detection_logits, labels)
        pred = output.detection_logits.argmax(dim=-1)

        total_loss += float(loss.item()) * labels.shape[0]
        gold_all.extend(labels.detach().cpu().tolist())
        pred_all.extend(pred.detach().cpu().tolist())

    metrics = {"loss": total_loss / len(loader.dataset)}
    metrics.update(binary_metrics(gold_all, pred_all))

    df = dataset.metadata.copy()
    df["prediction"] = pred_all
    df["correct"] = df["prediction"].astype(int) == df["label"].astype(int)

    return metrics, df


def metrics_by_depth(df: pd.DataFrame, split: str) -> pd.DataFrame:
    depth_col = "target_depth" if "target_depth" in df.columns else "max_depth"
    rows = []
    for depth, group in df.groupby(depth_col):
        y = group["label"].astype(int).to_numpy()
        p = group["prediction"].astype(int).to_numpy()
        rows.append({
            "split": split,
            "depth": int(depth),
            "rows": len(group),
            **binary_metrics(y, p),
        })
    return pd.DataFrame(rows).sort_values("depth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train detection model with depth curriculum.")
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

    parser.add_argument("--epochs-per-stage", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    parser.add_argument("--train-limit-per-stage", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_dir = args.output_dir / "checkpoints"
    table_dir = args.output_dir / "tables"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    config = CurriculumConfig(**vars(args))
    device = resolve_device(args.device)
    tokenizer = DyckTokenizer(max_length=args.max_length)

    model = build_model(
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    dev_dataset = DyckDetectionDataset(args.dev_csv, tokenizer, args.eval_limit)
    test_id_dataset = DyckDetectionDataset(args.test_id_csv, tokenizer, args.eval_limit)
    test_ood_dataset = DyckDetectionDataset(args.test_ood_csv, tokenizer, args.eval_limit)

    dev_loader = make_loader(dev_dataset, args.batch_size, False)
    test_id_loader = make_loader(test_id_dataset, args.batch_size, False)
    test_ood_loader = make_loader(test_ood_dataset, args.batch_size, False)

    metric_rows = []
    best_dev_acc = -1.0
    best_state = None

    for stage_depth in [1, 2, 3, 4]:
        train_dataset = DyckDetectionDataset(
            args.train_csv,
            tokenizer,
            limit=args.train_limit_per_stage,
            max_depth_allowed=stage_depth,
        )
        train_loader = make_loader(train_dataset, args.batch_size, True)

        for epoch in range(1, args.epochs_per_stage + 1):
            print(f"[curriculum] stage max_depth<={stage_depth}, epoch {epoch}/{args.epochs_per_stage}")
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, args.gradient_clip
            )
            dev_metrics, _ = evaluate(model, dev_dataset, dev_loader, loss_fn, device)

            metric_rows.append({
                "split": "train",
                "stage_max_depth": stage_depth,
                "epoch_in_stage": epoch,
                "rows": len(train_dataset),
                **train_metrics,
            })
            metric_rows.append({
                "split": "dev",
                "stage_max_depth": stage_depth,
                "epoch_in_stage": epoch,
                "rows": len(dev_dataset),
                **dev_metrics,
            })

            print(
                f"[curriculum] train_acc={train_metrics['accuracy']:.4f} "
                f"dev_acc={dev_metrics['accuracy']:.4f}"
            )

            if dev_metrics["accuracy"] > best_dev_acc:
                best_dev_acc = dev_metrics["accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")

    model.load_state_dict(best_state)

    final_tables = []
    depth_tables = []
    for split, dataset, loader in [
        ("test_id", test_id_dataset, test_id_loader),
        ("test_ood", test_ood_dataset, test_ood_loader),
    ]:
        metrics, predictions = evaluate(model, dataset, loader, loss_fn, device)
        metric_rows.append({"split": split, "stage_max_depth": "best", "epoch_in_stage": "best", "rows": len(dataset), **metrics})
        final_tables.append(predictions.assign(eval_split=split))
        depth_tables.append(metrics_by_depth(predictions, split))

    checkpoint_path = checkpoint_dir / "detection_curriculum.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "best_dev_accuracy": best_dev_acc,
        },
        checkpoint_path,
    )

    metrics_path = table_dir / "curriculum_detection_metrics.csv"
    depth_path = table_dir / "curriculum_detection_by_depth.csv"
    predictions_path = table_dir / "curriculum_detection_predictions.csv"

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, encoding="utf-8")
    pd.concat(depth_tables, ignore_index=True).to_csv(depth_path, index=False, encoding="utf-8")
    pd.concat(final_tables, ignore_index=True).to_csv(predictions_path, index=False, encoding="utf-8")

    print(f"[curriculum] Wrote {checkpoint_path}")
    print(f"[curriculum] Wrote {metrics_path}")
    print(f"[curriculum] Wrote {depth_path}")
    print(f"[curriculum] Wrote {predictions_path}")


if __name__ == "__main__":
    main()

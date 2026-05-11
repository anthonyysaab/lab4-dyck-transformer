"""
src/finetune_depth5.py
======================

Fine-tune the binary Dyck error detector on 500 depth-5 examples.

This addresses the Lab 4 requirement:
    fine-tune on 500 examples with n = 5, then evaluate on n = 5, 6, 7.

Inputs:
    outputs/checkpoints/detection_best.pt
    data/test_ood.csv

Outputs:
    data/finetune_depth5.csv
    outputs/checkpoints/detection_finetuned_depth5.pt
    outputs/tables/finetune_depth5_overall.csv
    outputs/tables/finetune_depth5_by_depth.csv

Run:
    python src/finetune_depth5.py --epochs 3 --batch-size 64 --learning-rate 1e-4
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

from dyck_data import generate_split
from model import build_model
from tokenizer import DyckTokenizer


@dataclass(frozen=True)
class FineTuneConfig:
    checkpoint: Path
    finetune_csv: Path
    test_ood_csv: Path
    output_dir: Path
    checkpoint_dir: Path
    table_dir: Path
    examples: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    seed: int
    device: str


class DyckDetectionDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer: DyckTokenizer, limit: int | None = None) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.head(limit).copy()

        required = {"id", "split", "tokens", "label", "error_type", "max_depth", "target_depth", "input_length"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

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


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
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
    return model, config


def ensure_depth5_finetune_csv(path: Path, examples: int, seed: int) -> None:
    if path.exists():
        print(f"[finetune_depth5] Reusing existing {path}")
        return

    print(f"[finetune_depth5] Generating {examples} fine-tuning examples at exact depth 5...")
    random.seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_split(
        split="finetune_depth5",
        size=examples,
        length_min=40,
        length_max=80,
        max_depth_limit=5,
        exact_depth_values=[5],
    )
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[finetune_depth5] Wrote {path}")


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
    gold_all: list[int] = []
    pred_all: list[int] = []

    for batch in tqdm(loader, desc="finetune", leave=False):
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

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(gold_all, pred_all)),
        "macro_f1": float(f1_score(gold_all, pred_all, average="macro", zero_division=0)),
    }


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[int] = []

    for batch in tqdm(loader, desc="predict", leave=False):
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        preds.extend(output.detection_logits.argmax(dim=-1).detach().cpu().tolist())

    return np.asarray(preds, dtype=np.int64)


def grouped_metrics(df: pd.DataFrame, label_col: str = "label", pred_col: str = "prediction") -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = df[label_col].astype(int).to_numpy()
    pred = df[pred_col].astype(int).to_numpy()

    overall = pd.DataFrame([{
        "rows": len(df),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
    }])

    rows = []
    depth_col = "target_depth" if "target_depth" in df.columns else "max_depth"
    for depth, group in df.groupby(depth_col):
        y = group[label_col].astype(int).to_numpy()
        p = group[pred_col].astype(int).to_numpy()
        rows.append({
            "depth": int(depth),
            "rows": len(group),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "valid_rows": int((y == 1).sum()),
            "invalid_rows": int((y == 0).sum()),
        })

    by_depth = pd.DataFrame(rows).sort_values("depth")
    return overall, by_depth


def evaluate_model_on_ood(
    model_name: str,
    model: torch.nn.Module,
    ood_dataset: DyckDetectionDataset,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = predict(model, loader, device)
    df = ood_dataset.metadata.copy()
    df["prediction"] = predictions
    overall, by_depth = grouped_metrics(df)
    overall.insert(0, "model", model_name)
    by_depth.insert(0, "model", model_name)
    return overall, by_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune detection model on 500 depth-5 examples.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/detection_best.pt"))
    parser.add_argument("--finetune-csv", type=Path, default=Path("data/finetune_depth5.csv"))
    parser.add_argument("--test-ood-csv", type=Path, default=Path("data/test_ood.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
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

    config = FineTuneConfig(
        checkpoint=args.checkpoint,
        finetune_csv=args.finetune_csv,
        test_ood_csv=args.test_ood_csv,
        output_dir=args.output_dir,
        checkpoint_dir=checkpoint_dir,
        table_dir=table_dir,
        examples=args.examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        device=args.device,
    )

    device = resolve_device(args.device)
    ensure_depth5_finetune_csv(args.finetune_csv, args.examples, args.seed)

    base_model, model_config = load_checkpoint_model(args.checkpoint, device)
    finetuned_model, _ = load_checkpoint_model(args.checkpoint, device)

    tokenizer = DyckTokenizer(max_length=int(model_config.get("max_length", 82)))
    finetune_dataset = DyckDetectionDataset(args.finetune_csv, tokenizer)
    ood_dataset = DyckDetectionDataset(args.test_ood_csv, tokenizer)

    finetune_loader = make_loader(finetune_dataset, args.batch_size, shuffle=True)
    ood_loader = make_loader(ood_dataset, args.batch_size, shuffle=False)

    print("[finetune_depth5] Evaluating base checkpoint on OOD...")
    base_overall, base_by_depth = evaluate_model_on_ood("base", base_model, ood_dataset, ood_loader, device)

    optimizer = torch.optim.AdamW(
        finetuned_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_rows = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=finetuned_model,
            loader=finetune_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_clip=args.gradient_clip,
        )
        metrics_row = {"model": "finetuned_depth5", "split": "finetune_depth5", "epoch": epoch, **metrics}
        train_rows.append(metrics_row)
        print(
            f"[finetune_depth5] epoch={epoch} "
            f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f}"
        )

    print("[finetune_depth5] Evaluating fine-tuned checkpoint on OOD...")
    ft_overall, ft_by_depth = evaluate_model_on_ood("finetuned_depth5", finetuned_model, ood_dataset, ood_loader, device)

    checkpoint_path = checkpoint_dir / "detection_finetuned_depth5.pt"
    torch.save(
        {
            "model_state_dict": finetuned_model.state_dict(),
            "base_checkpoint": str(args.checkpoint),
            "config": model_config,
            "finetune_config": asdict(config),
        },
        checkpoint_path,
    )

    overall = pd.concat([base_overall, ft_overall], ignore_index=True)
    by_depth = pd.concat([base_by_depth, ft_by_depth], ignore_index=True)
    train_df = pd.DataFrame(train_rows)

    overall_path = table_dir / "finetune_depth5_overall.csv"
    by_depth_path = table_dir / "finetune_depth5_by_depth.csv"
    train_path = table_dir / "finetune_depth5_training.csv"

    overall.to_csv(overall_path, index=False, encoding="utf-8")
    by_depth.to_csv(by_depth_path, index=False, encoding="utf-8")
    train_df.to_csv(train_path, index=False, encoding="utf-8")

    print(f"[finetune_depth5] Wrote {checkpoint_path}")
    print(f"[finetune_depth5] Wrote {overall_path}")
    print(f"[finetune_depth5] Wrote {by_depth_path}")
    print(f"[finetune_depth5] Wrote {train_path}")


if __name__ == "__main__":
    main()

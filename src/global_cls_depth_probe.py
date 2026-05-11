"""
src/global_cls_depth_probe.py
=============================

Global depth probe on the final-layer [CLS] representation.

This addresses the Lab 4 requirement:
    train a linear regression from final-layer [CLS] to maximum nesting depth,
    report R^2, then train a k-class linear classifier for depth n in {1,...,7}.

Important implementation detail:
    The detection model is frozen. The probe itself is trained on a separate
    synthetic probe split balanced across depths 1..7, so the classifier has
    seen all depth labels. This avoids the trivial problem where a classifier
    trained only on the original ID training set, depths <=4, cannot predict
    unseen labels 5, 6, and 7.

Inputs:
    outputs/checkpoints/detection_best.pt
    data/dev.csv
    data/test_id.csv
    data/test_ood.csv

Generated probe data:
    data/probe_depth_train.csv
    data/probe_depth_test.csv

Outputs:
    outputs/tables/global_cls_depth_probe_metrics.csv
    outputs/tables/global_cls_depth_probe_predictions.csv

Run:
    python src/global_cls_depth_probe.py --probe-train-size 7000 --probe-test-size 2100
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dyck_data import generate_split
from model import build_model
from tokenizer import DyckTokenizer


class DepthDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer: DyckTokenizer, limit: int | None) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.head(limit).copy()

        required = {"id", "split", "tokens", "label", "error_type", "max_depth", "input_length"}
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
        self.depths = df["max_depth"].astype(int).to_numpy()
        self.metadata = df.reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
        }


def ensure_probe_csv(path: Path, split: str, size: int, seed: int) -> None:
    if path.exists():
        print(f"[global_cls_depth_probe] Reusing existing {path}")
        return

    print(f"[global_cls_depth_probe] Generating {path} with depths 1..7...")
    random.seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_split(
        split=split,
        size=size,
        length_min=4,
        length_max=80,
        max_depth_limit=7,
        exact_depth_values=[1, 2, 3, 4, 5, 6, 7],
    )
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[global_cls_depth_probe] Wrote {path}")


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

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, config


@torch.no_grad()
def extract_cls_features(
    model: torch.nn.Module,
    dataset: DepthDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    features = []
    for batch in tqdm(loader, desc="extract [CLS]", leave=False):
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        cls = output.hidden_states[:, 0, :].detach().cpu().numpy()
        features.append(cls)

    x = np.vstack(features).astype(np.float32)
    y = dataset.depths.astype(np.int64)
    return x, y, dataset.metadata.copy()


def safe_r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(r2_score(y, pred))


def evaluate_probe(
    split: str,
    x: np.ndarray,
    y: np.ndarray,
    regressor,
    classifier,
    metadata: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    reg_pred = regressor.predict(x)
    cls_pred = classifier.predict(x)

    metrics = [
        {
            "probe": "cls_depth_regression",
            "split": split,
            "rows": len(y),
            "r2": safe_r2(y, reg_pred),
            "mae": float(mean_absolute_error(y, reg_pred)),
            "accuracy": np.nan,
            "macro_f1": np.nan,
        },
        {
            "probe": "cls_depth_classification",
            "split": split,
            "rows": len(y),
            "r2": np.nan,
            "mae": np.nan,
            "accuracy": float(accuracy_score(y, cls_pred)),
            "macro_f1": float(f1_score(y, cls_pred, average="macro", zero_division=0)),
        },
    ]

    pred_df = metadata[["id", "split", "tokens", "label", "error_type", "max_depth", "input_length"]].copy()
    pred_df["eval_split"] = split
    pred_df["gold_depth"] = y
    pred_df["regression_prediction"] = reg_pred
    pred_df["classification_prediction"] = cls_pred
    pred_df["classification_correct"] = pred_df["classification_prediction"].astype(int) == pred_df["gold_depth"].astype(int)

    return metrics, pred_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global [CLS] depth probe.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/detection_best.pt"))

    parser.add_argument("--probe-train-csv", type=Path, default=Path("data/probe_depth_train.csv"))
    parser.add_argument("--probe-test-csv", type=Path, default=Path("data/probe_depth_test.csv"))
    parser.add_argument("--probe-train-size", type=int, default=7000)
    parser.add_argument("--probe-test-size", type=int, default=2100)

    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-id-csv", type=Path, default=Path("data/test_id.csv"))
    parser.add_argument("--test-ood-csv", type=Path, default=Path("data/test_ood.csv"))

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_probe_csv(args.probe_train_csv, "probe_depth_train", args.probe_train_size, args.seed)
    ensure_probe_csv(args.probe_test_csv, "probe_depth_test", args.probe_test_size, args.seed + 1)

    device = resolve_device(args.device)
    model, config = load_model(args.checkpoint, device)
    tokenizer = DyckTokenizer(max_length=int(config.get("max_length", 82)))

    specs = [
        ("probe_train", args.probe_train_csv, None),
        ("probe_test", args.probe_test_csv, None),
        ("dev", args.dev_csv, args.eval_limit),
        ("test_id", args.test_id_csv, args.eval_limit),
        ("test_ood", args.test_ood_csv, args.eval_limit),
    ]

    extracted: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for split, path, limit in specs:
        print(f"[global_cls_depth_probe] Extracting {split} from {path}")
        dataset = DepthDataset(path, tokenizer, limit)
        extracted[split] = extract_cls_features(model, dataset, args.batch_size, device)
        print(f"[global_cls_depth_probe] {split}: {extracted[split][0].shape[0]} rows")

    x_train, y_train, _ = extracted["probe_train"]

    regressor = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    regressor.fit(x_train, y_train)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )
    classifier.fit(x_train, y_train)

    metric_rows: list[dict[str, object]] = []
    pred_tables = []

    for split in ["probe_test", "dev", "test_id", "test_ood"]:
        x, y, meta = extracted[split]
        metrics, preds = evaluate_probe(split, x, y, regressor, classifier, meta)
        metric_rows.extend(metrics)
        pred_tables.append(preds)
        print(pd.DataFrame(metrics).to_string(index=False))

    metrics_path = args.output_dir / "global_cls_depth_probe_metrics.csv"
    predictions_path = args.output_dir / "global_cls_depth_probe_predictions.csv"

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, encoding="utf-8")
    pd.concat(pred_tables, ignore_index=True).to_csv(predictions_path, index=False, encoding="utf-8")

    print(f"[global_cls_depth_probe] Wrote {metrics_path}")
    print(f"[global_cls_depth_probe] Wrote {predictions_path}")


if __name__ == "__main__":
    main()

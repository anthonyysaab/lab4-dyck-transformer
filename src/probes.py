"""
src/probes.py
=============

Probing classifiers for Lab 4 Dyck Transformer.

Goal:
    Test whether the Transformer's hidden states encode stack-like variables.

We train lightweight linear probes on frozen hidden states from the trained
detection model.

Probe tasks:
    1. depth_after:
        Stack depth after reading each bracket token.

    2. top_closer_after:
        Expected next closing bracket at the top of the stack after reading
        each token:
            NONE, ), ]

We use valid Dyck strings only, because stack state is unambiguous.

Outputs:
    outputs/tables/probe_metrics.csv
    outputs/tables/probe_predictions_*.csv

Run:
    python src/probes.py --train-limit 5000 --eval-limit 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
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

TOP_CLOSER_TO_ID = {
    "NONE": 0,
    ")": 1,
    "]": 2,
}

ID_TO_TOP_CLOSER = {value: key for key, value in TOP_CLOSER_TO_ID.items()}


class ValidProbeDataset(Dataset):
    """
    Valid-only dataset for extracting hidden states and stack labels.
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer: DyckTokenizer,
        limit: int | None,
    ) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        df = pd.read_csv(csv_path)
        df = df[df["label"].astype(int) == 1].copy()

        if limit is not None:
            df = df.head(limit).copy()

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


def stack_probe_labels(tokens: list[str]) -> tuple[list[int], list[int]]:
    """
    Return token-level labels:
        depth_after[token_index]
        top_closer_after[token_index]

    The input must be a valid Dyck string.
    """
    stack: list[str] = []

    depth_after: list[int] = []
    top_closer_after: list[int] = []

    for token in tokens:
        if token in OPENERS:
            stack.append(PAIRS[token])
        elif token in CLOSERS:
            if not stack:
                raise ValueError("Invalid string: premature closer.")
            expected = stack.pop()
            if expected != token:
                raise ValueError("Invalid string: type mismatch.")
        else:
            raise ValueError(f"Unknown token: {token!r}")

        depth_after.append(len(stack))

        if stack:
            top_closer_after.append(TOP_CLOSER_TO_ID[stack[-1]])
        else:
            top_closer_after.append(TOP_CLOSER_TO_ID["NONE"])

    if stack:
        raise ValueError("Invalid string: missing closer.")

    return depth_after, top_closer_after


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

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, config


@torch.no_grad()
def extract_probe_table(
    model: torch.nn.Module,
    dataset: ValidProbeDataset,
    batch_size: int,
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    """
    Extract one row per real bracket token.

    Features are frozen Transformer hidden states at that token position.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []

    global_offset = 0

    for batch in tqdm(dataloader, desc=f"extract {split_name}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.hidden_states.detach().cpu().numpy()

        batch_size_actual = input_ids.shape[0]

        for local_index in range(batch_size_actual):
            metadata_row = dataset.metadata.iloc[global_offset + local_index]
            tokens = str(metadata_row["tokens"]).split()

            depth_after, top_closer_after = stack_probe_labels(tokens)

            if len(tokens) != len(depth_after):
                raise ValueError("Probe label length mismatch.")

            for raw_token_index, token in enumerate(tokens):
                model_position = raw_token_index + 1  # +1 because [CLS] is position 0.

                feature_rows.append(hidden[local_index, model_position, :].copy())

                rows.append(
                    {
                        "split": split_name,
                        "example_id": int(metadata_row["id"]),
                        "token_index": raw_token_index,
                        "token": token,
                        "input_length": int(metadata_row["input_length"]),
                        "max_depth": int(metadata_row["max_depth"]),
                        "depth_after": int(depth_after[raw_token_index]),
                        "top_closer_after": int(top_closer_after[raw_token_index]),
                    }
                )

        global_offset += batch_size_actual

    df = pd.DataFrame(rows)
    features = np.vstack(feature_rows)

    feature_columns = [f"h_{i}" for i in range(features.shape[1])]
    feature_df = pd.DataFrame(features, columns=feature_columns)

    return pd.concat([df, feature_df], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("h_")]


def train_probe(
    train_df: pd.DataFrame,
    label_column: str,
) -> tuple[StandardScaler, LogisticRegression]:
    """
    Train a multinomial logistic-regression probe.
    """
    cols = feature_columns(train_df)

    x_train = train_df[cols].to_numpy(dtype=np.float32)
    y_train = train_df[label_column].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    probe = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=None,
    )
    probe.fit(x_train_scaled, y_train)

    return scaler, probe


def evaluate_probe(
    scaler: StandardScaler,
    probe: LogisticRegression,
    df: pd.DataFrame,
    label_column: str,
    task_name: str,
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Evaluate one trained probe.
    """
    cols = feature_columns(df)

    x = df[cols].to_numpy(dtype=np.float32)
    y = df[label_column].to_numpy(dtype=np.int64)

    x_scaled = scaler.transform(x)
    pred = probe.predict(x_scaled)

    metrics = {
        "task": task_name,
        "split": split_name,
        "rows": len(df),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }

    pred_df = df[
        [
            "split",
            "example_id",
            "token_index",
            "token",
            "input_length",
            "max_depth",
            label_column,
        ]
    ].copy()
    pred_df["prediction"] = pred
    pred_df["correct"] = pred_df["prediction"].astype(int) == pred_df[label_column].astype(int)
    pred_df["task"] = task_name

    return metrics, pred_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train probes on frozen Dyck Transformer states.")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/detection_best.pt"),
    )

    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-id-csv", type=Path, default=Path("data/test_id.csv"))
    parser.add_argument("--test-ood-csv", type=Path, default=Path("data/test_ood.csv"))

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))

    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)

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

    print("[probes] Device:", device)

    model, config = load_model(args.checkpoint, device=device)
    tokenizer = DyckTokenizer(max_length=int(config.get("max_length", 82)))

    split_specs = [
        ("train", args.train_csv, args.train_limit),
        ("dev", args.dev_csv, args.eval_limit),
        ("test_id", args.test_id_csv, args.eval_limit),
        ("test_ood", args.test_ood_csv, args.eval_limit),
    ]

    probe_tables: dict[str, pd.DataFrame] = {}

    for split_name, csv_path, limit in split_specs:
        print(f"[probes] Extracting hidden states for {split_name} from {csv_path}")

        dataset = ValidProbeDataset(
            csv_path=csv_path,
            tokenizer=tokenizer,
            limit=limit,
        )

        probe_tables[split_name] = extract_probe_table(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            split_name=split_name,
        )

        print(f"[probes] {split_name} token rows: {len(probe_tables[split_name])}")

    tasks = [
        ("depth_after", "depth_after"),
        ("top_closer_after", "top_closer_after"),
    ]

    metrics_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []

    train_df = probe_tables["train"]

    for task_name, label_column in tasks:
        print(f"\n[probes] Training probe: {task_name}")

        scaler, probe = train_probe(
            train_df=train_df,
            label_column=label_column,
        )

        for split_name in ["dev", "test_id", "test_ood"]:
            metrics, predictions = evaluate_probe(
                scaler=scaler,
                probe=probe,
                df=probe_tables[split_name],
                label_column=label_column,
                task_name=task_name,
                split_name=split_name,
            )

            metrics_rows.append(metrics)
            prediction_tables.append(predictions)

            print(
                f"[probes] {task_name} {split_name}: "
                f"accuracy={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

            y = predictions[label_column].to_numpy()
            pred = predictions["prediction"].to_numpy()

            print(
                classification_report(
                    y,
                    pred,
                    digits=4,
                    zero_division=0,
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = args.output_dir / "probe_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    predictions_df = pd.concat(prediction_tables, ignore_index=True)
    predictions_path = args.output_dir / "probe_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")

    print(f"\n[probes] Wrote {metrics_path}")
    print(f"[probes] Wrote {predictions_path}")
    print("[probes] Done.")


if __name__ == "__main__":
    main()

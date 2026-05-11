"""
src/local_depth_probe_layers.py
===============================

Layer-wise local depth probe.

This addresses the Lab 4 requirement:
    for each token position t, predict current nesting depth at t;
    report R^2 as a function of Transformer layer depth;
    compare correct and erroneous strings.

The probe is trained on valid training strings and evaluated on:
    - valid dev/test/OOD tokens
    - corrupted dev/test/OOD tokens
    - corrupted tokens before, at, and after the error position

Outputs:
    outputs/tables/local_depth_probe_layer_metrics.csv
    outputs/tables/local_depth_probe_layer_predictions.csv

Run:
    python src/local_depth_probe_layers.py --train-limit 5000 --eval-limit 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import build_model
from tokenizer import DyckTokenizer


PAIRS = {"(": ")", "[": "]"}
OPENERS = set(PAIRS.keys())
CLOSERS = set(PAIRS.values())


class ProbeDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        tokenizer: DyckTokenizer,
        limit: int | None,
        label_filter: int | None,
    ) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        df = pd.read_csv(csv_path)
        if label_filter is not None:
            df = df[df["label"].astype(int) == label_filter].copy()
        if limit is not None:
            df = df.head(limit).copy()
        if df.empty:
            raise ValueError(f"No rows left after filtering {csv_path}")

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


def prefix_depth_labels(tokens: list[str]) -> list[int]:
    """
    Current nesting depth after reading each token.

    For invalid strings, the scan is deliberately conservative:
    - openers increase depth;
    - a closer pops only if it matches the current top of stack;
    - a premature or mismatched closer leaves the stack unchanged.

    This makes the erroneous-string analysis explicit rather than pretending
    that there is a unique gold stack state after corruption.
    """
    stack: list[str] = []
    depths: list[int] = []

    for token in tokens:
        if token in OPENERS:
            stack.append(PAIRS[token])
        elif token in CLOSERS:
            if stack and stack[-1] == token:
                stack.pop()
        else:
            raise ValueError(f"Unknown token: {token!r}")

        depths.append(len(stack))

    return depths


def position_relation(token_index: int, error_position: int) -> str:
    if error_position < 0:
        return "valid"
    if token_index < error_position:
        return "before_error"
    if token_index == error_position:
        return "at_error"
    return "after_error"


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
def layer_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Re-run the model internals and return hidden states after each encoder layer.

    Returns:
        list of tensors [batch, seq, hidden], one per Transformer block.
    """
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(batch_size, seq_len)

    hidden = model.token_embeddings(input_ids) + model.position_embeddings(positions)
    hidden = model.embedding_norm(hidden)
    hidden = model.dropout(hidden)

    layers: list[torch.Tensor] = []
    for block in model.layers:
        hidden, _attention = block(hidden_states=hidden, attention_mask=attention_mask)
        layers.append(hidden.detach().cpu())

    # Match the final representation used by the classifier for the last layer.
    if layers:
        layers[-1] = model.final_norm(hidden).detach().cpu()

    return layers


@torch.no_grad()
def extract_token_features(
    model: torch.nn.Module,
    dataset: ProbeDataset,
    batch_size: int,
    device: torch.device,
    split_name: str,
) -> dict[int, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    layer_features: dict[int, list[np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    targets: list[int] = []

    global_offset = 0

    for batch in tqdm(loader, desc=f"extract {split_name}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hidden_by_layer = layer_hidden_states(model, input_ids, attention_mask)

        for local_index in range(input_ids.shape[0]):
            meta = dataset.metadata.iloc[global_offset + local_index]
            tokens = str(meta["tokens"]).split()
            depths = prefix_depth_labels(tokens)
            error_position = int(meta.get("error_position", -1))
            gold_label = int(meta["label"])

            for raw_token_index, token in enumerate(tokens):
                model_pos = raw_token_index + 1  # [CLS] is position 0.

                if len(targets) == len(rows):
                    targets.append(int(depths[raw_token_index]))

                rows.append({
                    "split": split_name,
                    "example_id": int(meta["id"]),
                    "label": gold_label,
                    "error_type": str(meta["error_type"]),
                    "error_position": error_position,
                    "position_relation": position_relation(raw_token_index, error_position),
                    "token_index": raw_token_index,
                    "token": token,
                    "input_length": int(meta["input_length"]),
                    "max_depth": int(meta["max_depth"]),
                    "local_depth": int(depths[raw_token_index]),
                })

                for layer_index, hidden in enumerate(hidden_by_layer, start=1):
                    layer_features.setdefault(layer_index, []).append(
                        hidden[local_index, model_pos, :].numpy().copy()
                    )

        global_offset += input_ids.shape[0]

    y = np.asarray(targets, dtype=np.float32)
    meta_df = pd.DataFrame(rows)

    output = {}
    for layer_index, feats in layer_features.items():
        output[layer_index] = (np.vstack(feats).astype(np.float32), y, meta_df.copy())

    return output


def safe_r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(r2_score(y, pred))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise local depth probe.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/detection_best.pt"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-id-csv", type=Path, default=Path("data/test_id.csv"))
    parser.add_argument("--test-ood-csv", type=Path, default=Path("data/test_ood.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, config = load_model(args.checkpoint, device)
    tokenizer = DyckTokenizer(max_length=int(config.get("max_length", 82)))

    print("[local_depth_probe_layers] Extracting valid train tokens...")
    train_dataset = ProbeDataset(args.train_csv, tokenizer, args.train_limit, label_filter=1)
    train_layers = extract_token_features(model, train_dataset, args.batch_size, device, "train_valid")

    eval_specs = [
        ("dev_valid", args.dev_csv, 1),
        ("dev_corrupt", args.dev_csv, 0),
        ("test_id_valid", args.test_id_csv, 1),
        ("test_id_corrupt", args.test_id_csv, 0),
        ("test_ood_valid", args.test_ood_csv, 1),
        ("test_ood_corrupt", args.test_ood_csv, 0),
    ]

    eval_layers: dict[str, dict[int, tuple[np.ndarray, np.ndarray, pd.DataFrame]]] = {}
    for split, path, label_filter in eval_specs:
        print(f"[local_depth_probe_layers] Extracting {split}...")
        dataset = ProbeDataset(path, tokenizer, args.eval_limit, label_filter=label_filter)
        eval_layers[split] = extract_token_features(model, dataset, args.batch_size, device, split)

    metric_rows = []
    prediction_tables = []

    for layer_index, (x_train, y_train, _train_meta) in train_layers.items():
        print(f"[local_depth_probe_layers] Training Ridge probe for layer {layer_index}")
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        probe.fit(x_train, y_train)

        for split, layer_map in eval_layers.items():
            x, y, meta = layer_map[layer_index]
            pred = probe.predict(x)

            metric_rows.append({
                "layer": layer_index,
                "split": split,
                "position_relation": "all",
                "rows": len(y),
                "r2": safe_r2(y, pred),
                "mae": float(mean_absolute_error(y, pred)),
            })

            # Extra breakdown for corrupted strings.
            for relation, idx in meta.groupby("position_relation").groups.items():
                idx = list(idx)
                metric_rows.append({
                    "layer": layer_index,
                    "split": split,
                    "position_relation": relation,
                    "rows": len(idx),
                    "r2": safe_r2(y[idx], pred[idx]),
                    "mae": float(mean_absolute_error(y[idx], pred[idx])),
                })

            pred_df = meta.copy()
            pred_df["layer"] = layer_index
            pred_df["prediction"] = pred
            prediction_tables.append(pred_df)

    metrics = pd.DataFrame(metric_rows).sort_values(["layer", "split", "position_relation"])
    predictions = pd.concat(prediction_tables, ignore_index=True)

    metrics_path = args.output_dir / "local_depth_probe_layer_metrics.csv"
    predictions_path = args.output_dir / "local_depth_probe_layer_predictions.csv"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")

    print(f"[local_depth_probe_layers] Wrote {metrics_path}")
    print(f"[local_depth_probe_layers] Wrote {predictions_path}")


if __name__ == "__main__":
    main()

"""
src/model.py
============

Small Transformer encoder for Lab 4 Dyck-language experiments.

The model supports two tasks:

1. Sequence-level binary detection:
       Is the bracket string valid or corrupted?

2. Token-level correction:
       For each bracket token, predict an action such as OK, DELETE,
       INSERT_RPAREN, REPLACE_RBRACK, etc.

The encoder is intentionally small because the dataset is synthetic and the
goal is interpretability/generalisation, not large-scale language modelling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn

from tokenizer import ACTION_LABELS, VOCAB, DyckTokenizer


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class DyckModelOutput:
    detection_logits: torch.Tensor
    correction_logits: torch.Tensor
    hidden_states: torch.Tensor
    attentions: list[torch.Tensor]


# ---------------------------------------------------------------------------
# Transformer block with accessible attention weights
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """
    One pre-norm Transformer encoder block.

    Unlike torch.nn.TransformerEncoderLayer, this block keeps attention weights
    accessible for later analysis.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feedforward_norm = nn.LayerNorm(hidden_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states:
            Tensor of shape [batch_size, sequence_length, hidden_dim].

        attention_mask:
            Tensor of shape [batch_size, sequence_length], where 1 means real
            token and 0 means padding.

        Returns
        -------
        updated_hidden_states:
            Tensor of shape [batch_size, sequence_length, hidden_dim].

        attention_weights:
            Tensor of shape [batch_size, num_heads, sequence_length, sequence_length].
        """
        key_padding_mask = attention_mask == 0

        normed = self.attention_norm(hidden_states)

        attention_output, attention_weights = self.self_attention(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        hidden_states = hidden_states + self.dropout(attention_output)

        normed = self.feedforward_norm(hidden_states)
        feedforward_output = self.feedforward(normed)
        hidden_states = hidden_states + self.dropout(feedforward_output)

        return hidden_states, attention_weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DyckTransformer(nn.Module):
    """
    Small BERT-style encoder for Dyck-language classification and correction.
    """

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        max_length: int = 82,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )

        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_length, hidden_dim)

        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        self.correction_head = nn.Linear(hidden_dim, num_actions)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> DyckModelOutput:
        """
        Forward pass.

        input_ids:
            [batch_size, sequence_length]

        attention_mask:
            [batch_size, sequence_length]
        """
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds "
                f"model max_length={self.max_length}."
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, sequence_length)

        hidden_states = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(positions)
        )

        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        attentions: list[torch.Tensor] = []

        for layer in self.layers:
            hidden_states, attention_weights = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            attentions.append(attention_weights)

        hidden_states = self.final_norm(hidden_states)

        cls_hidden_state = hidden_states[:, 0, :]

        detection_logits = self.detection_head(cls_hidden_state)
        correction_logits = self.correction_head(hidden_states)

        return DyckModelOutput(
            detection_logits=detection_logits,
            correction_logits=correction_logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    max_length: int = 82,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    ff_dim: int = 512,
    dropout: float = 0.1,
) -> DyckTransformer:
    """
    Build the default Lab 4 model.
    """
    return DyckTransformer(
        vocab_size=len(VOCAB),
        num_actions=len(ACTION_LABELS),
        max_length=max_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Smoke-test CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Dyck Transformer.")
    parser.add_argument("--max-length", type=int, default=82)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = DyckTokenizer(max_length=args.max_length)

    examples = [
        "( [ ] )",
        "( [ ) ]",
    ]

    encoded = [tokenizer.encode_text(example) for example in examples]

    input_ids = torch.stack([item.input_ids for item in encoded], dim=0)
    attention_mask = torch.stack([item.attention_mask for item in encoded], dim=0)

    model = build_model(
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("[model] input_ids shape:", tuple(input_ids.shape))
    print("[model] detection_logits shape:", tuple(output.detection_logits.shape))
    print("[model] correction_logits shape:", tuple(output.correction_logits.shape))
    print("[model] hidden_states shape:", tuple(output.hidden_states.shape))
    print("[model] attention layers:", len(output.attentions))
    print("[model] first attention shape:", tuple(output.attentions[0].shape))
    print("[model] parameters:", f"{parameter_count:,}")
    print("[model] OK")


if __name__ == "__main__":
    main()

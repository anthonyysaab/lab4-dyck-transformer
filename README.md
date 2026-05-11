# Lab 4 — Transformers and Dyck Languages

This repository contains the implementation for Lab 4: error detection, correction, and interpretability for Dyck languages using a small Transformer encoder.

## Goals

- Generate synthetic Dyck-language data with bracket pairs `()` and `[]`.
- Train a Transformer encoder for binary error detection.
- Train a local correction model.
- Evaluate in-distribution and out-of-distribution generalisation.
- Compare against a deterministic pushdown automaton baseline.
- Analyse attention heads and probing classifiers.

## Structure

- `src/`: Python source code.
- `data/`: Generated datasets.
- `outputs/`: Figures, tables, attention visualisations, checkpoints, and model outputs.

## Current status

Project skeleton created. Next step: implement the Dyck data generator.

"""Persist evaluation results to disk as JSON/CSV and a Markdown summary.

The training pipeline evaluates several models (contrastive encoder + baselines)
and writes a single machine-readable ``metrics.json`` plus a human-readable
``metrics_summary.md`` that can be regenerated on every run.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict


def write_metrics_json(path: str, payload: dict) -> None:
    """Write the full metrics payload as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_metrics_csv(path: str, results: Dict[str, dict], ks: list) -> None:
    """Flatten ``{model: {K: {HR, NDCG, n}}}`` into one row per (model, K)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "K", "HR", "NDCG", "n_users"])
        for model, per_k in results.items():
            for k in ks:
                row = per_k.get(str(k)) or per_k.get(k) or {}
                writer.writerow([
                    model, k,
                    f"{row.get('HR', 0.0):.4f}",
                    f"{row.get('NDCG', 0.0):.4f}",
                    row.get("n", 0),
                ])


def write_metrics_summary(path: str, results: Dict[str, dict],
                          ks: list, config_dict: dict) -> None:
    """Regenerate a Markdown table summarizing HR@K / NDCG@K per model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# Metrics Summary", ""]
    lines.append("Evaluated on cold items (MovieLens-1M). Higher is better.")
    lines.append("")

    # One HR + one NDCG column per K.
    header = ["Model"] + [f"HR@{k}" for k in ks] + [f"NDCG@{k}" for k in ks]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for model, per_k in results.items():
        cells = [model]
        for k in ks:
            row = per_k.get(str(k)) or per_k.get(k) or {}
            cells.append(f"{row.get('HR', 0.0):.4f}")
        for k in ks:
            row = per_k.get(str(k)) or per_k.get(k) or {}
            cells.append(f"{row.get('NDCG', 0.0):.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for key in ("backend", "emb_dim", "hidden_dim", "torch_hidden_dim",
                "epochs", "batch_size", "lr", "tau", "cold_threshold",
                "n_test_users", "seed"):
        if key in config_dict:
            lines.append(f"- **{key}**: {config_dict[key]}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

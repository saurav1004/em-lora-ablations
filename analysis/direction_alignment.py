#!/usr/bin/env python3
from __future__ import annotations
"""
Compare misalignment directions across sources.

Primary use (Direction 2, Phase 1):
  - Compare independent activation-derived direction vs SVD top-component direction
  - Quantify layer/module cosine alignment and coverage

Usage:
  python analysis/direction_alignment.py \
    --direction_a results/misalignment_direction_activation.pt \
    --direction_b results/misalignment_direction.pt \
    --output results/direction_alignment_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def normalize_keyspace(keys: set) -> set:
    normalized = set()
    for key in keys:
        try:
            normalized.add(int(key))
        except Exception:
            normalized.add(str(key))
    return normalized


def load_direction_file(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Direction file must contain a dict, got: {type(payload)}")
    return payload


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim != 1:
        a = a.flatten()
    if b.ndim != 1:
        b = b.flatten()

    dim = min(a.shape[0], b.shape[0])
    if dim == 0:
        return float("nan")

    a = a[:dim].float()
    b = b[:dim].float()

    if a.norm().item() == 0 or b.norm().item() == 0:
        return float("nan")

    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compare_directions(direction_a: dict, direction_b: dict) -> dict:
    keys_a = normalize_keyspace(set(direction_a.keys()))
    keys_b = normalize_keyspace(set(direction_b.keys()))
    common_keys = sorted(keys_a.intersection(keys_b), key=lambda x: (isinstance(x, str), x))

    per_key = {}
    cosines = []
    abs_cosines = []

    for key in common_keys:
        tensor_a = direction_a.get(key, direction_a.get(str(key)))
        tensor_b = direction_b.get(key, direction_b.get(str(key)))

        if tensor_a is None or tensor_b is None:
            continue

        cos = cosine_similarity(tensor_a, tensor_b)
        if np.isnan(cos):
            continue

        per_key[str(key)] = {
            "cosine": cos,
            "abs_cosine": abs(cos),
            "dim_a": int(tensor_a.numel()),
            "dim_b": int(tensor_b.numel()),
        }
        cosines.append(cos)
        abs_cosines.append(abs(cos))

    summary = {
        "num_keys_a": len(keys_a),
        "num_keys_b": len(keys_b),
        "num_common_keys": len(common_keys),
        "num_compared": len(cosines),
    }

    if cosines:
        summary.update(
            {
                "mean_cosine": float(np.mean(cosines)),
                "median_cosine": float(np.median(cosines)),
                "std_cosine": float(np.std(cosines)),
                "mean_abs_cosine": float(np.mean(abs_cosines)),
                "median_abs_cosine": float(np.median(abs_cosines)),
                "p10_abs_cosine": float(np.percentile(abs_cosines, 10)),
                "p90_abs_cosine": float(np.percentile(abs_cosines, 90)),
            }
        )

    return {
        "summary": summary,
        "per_key": per_key,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two misalignment direction files")
    parser.add_argument("--direction_a", required=True, help="Path to first direction .pt file")
    parser.add_argument("--direction_b", required=True, help="Path to second direction .pt file")
    parser.add_argument("--label_a", default="direction_a", help="Label for first direction")
    parser.add_argument("--label_b", default="direction_b", help="Label for second direction")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    direction_a = load_direction_file(args.direction_a)
    direction_b = load_direction_file(args.direction_b)

    result = compare_directions(direction_a, direction_b)
    result["labels"] = {
        "a": args.label_a,
        "b": args.label_b,
    }
    result["inputs"] = {
        "direction_a": args.direction_a,
        "direction_b": args.direction_b,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 60)
    print("Direction Alignment Summary")
    print("=" * 60)
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

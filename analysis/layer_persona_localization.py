#!/usr/bin/env python3
from __future__ import annotations
"""
Layer-level persona localization for Direction 2 Phase 2.

Consumes ablation analysis and SVD summaries to estimate where persona
signal concentrates spatially (by layer and module family).

Usage:
  python analysis/layer_persona_localization.py \
    --ablation_analysis results/ablation_analysis.json \
    --svd_summary results/svd_components_summary.json \
    --output results/layer_persona_localization.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_layer(module_key: str) -> int | None:
    parts = module_key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def parse_module_family(module_key: str) -> str:
    if ".self_attn." in module_key:
        return "self_attn"
    if ".mlp." in module_key:
        return "mlp"
    return "other"


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def safe_sum(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.sum(values))


def main():
    parser = argparse.ArgumentParser(description="Layer-level persona localization")
    parser.add_argument("--ablation_analysis", required=True)
    parser.add_argument("--svd_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.ablation_analysis) as f:
        ablation_data = json.load(f)

    with open(args.svd_summary) as f:
        svd_summary = json.load(f)

    scores = ablation_data.get("scores", {})
    logs = ablation_data.get("ablation_log", {})
    svd_per_module = svd_summary.get("per_module", {})

    layer_stats: dict[int, dict] = defaultdict(lambda: {
        "modules": 0,
        "components": 0,
        "ablated_components": 0,
        "energy_removed": [],
        "top1_energy_fraction": [],
        "max_abs_cosine": [],
    })

    family_stats: dict[str, dict] = defaultdict(lambda: {
        "modules": 0,
        "components": 0,
        "ablated_components": 0,
        "energy_removed": [],
        "top1_energy_fraction": [],
        "max_abs_cosine": [],
    })

    for module_key, score_payload in scores.items():
        layer = parse_layer(module_key)
        if layer is None:
            continue

        family = parse_module_family(module_key)
        rank = int(score_payload.get("rank", 0))
        max_abs_cos = float(score_payload.get("max_cosine_sim", 0.0))

        log_payload = logs.get(module_key, {})
        energy_removed = float(log_payload.get("energy_removed", 0.0))
        num_ablated = int(log_payload.get("num_ablated", 0))

        svd_payload = svd_per_module.get(module_key, {})
        top1_energy = svd_payload.get("top1_energy_fraction", 0.0)
        try:
            top1_energy = float(top1_energy)
        except Exception:
            top1_energy = 0.0

        for bucket in (layer_stats[layer], family_stats[family]):
            bucket["modules"] += 1
            bucket["components"] += rank
            bucket["ablated_components"] += num_ablated
            bucket["energy_removed"].append(energy_removed)
            bucket["top1_energy_fraction"].append(top1_energy)
            bucket["max_abs_cosine"].append(max_abs_cos)

    layer_output = {}
    for layer, payload in sorted(layer_stats.items()):
        layer_output[str(layer)] = {
            "modules": payload["modules"],
            "components": payload["components"],
            "ablated_components": payload["ablated_components"],
            "ablation_density": (
                payload["ablated_components"] / payload["components"]
                if payload["components"] > 0 else 0.0
            ),
            "mean_energy_removed": safe_mean(payload["energy_removed"]),
            "mean_top1_energy_fraction": safe_mean(payload["top1_energy_fraction"]),
            "mean_max_abs_cosine": safe_mean(payload["max_abs_cosine"]),
            "persona_intensity_score": (
                (safe_mean(payload["energy_removed"]) or 0.0)
                * (safe_mean(payload["max_abs_cosine"]) or 0.0)
            ),
        }

    family_output = {}
    for family, payload in sorted(family_stats.items()):
        family_output[family] = {
            "modules": payload["modules"],
            "components": payload["components"],
            "ablated_components": payload["ablated_components"],
            "ablation_density": (
                payload["ablated_components"] / payload["components"]
                if payload["components"] > 0 else 0.0
            ),
            "mean_energy_removed": safe_mean(payload["energy_removed"]),
            "mean_top1_energy_fraction": safe_mean(payload["top1_energy_fraction"]),
            "mean_max_abs_cosine": safe_mean(payload["max_abs_cosine"]),
        }

    ranked_layers = sorted(
        [
            {
                "layer": int(layer),
                **metrics,
            }
            for layer, metrics in layer_output.items()
        ],
        key=lambda x: x["persona_intensity_score"],
        reverse=True,
    )

    output = {
        "inputs": {
            "ablation_analysis": args.ablation_analysis,
            "svd_summary": args.svd_summary,
        },
        "global": {
            "num_layers": len(layer_output),
            "num_module_families": len(family_output),
            "total_modules": int(sum(m["modules"] for m in family_output.values())),
            "total_components": int(sum(m["components"] for m in family_output.values())),
            "total_ablated_components": int(sum(m["ablated_components"] for m in family_output.values())),
        },
        "by_layer": layer_output,
        "by_module_family": family_output,
        "ranked_layers": ranked_layers,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print("Layer Persona Localization")
    print("=" * 60)
    print(f"layers: {output['global']['num_layers']}")
    print(f"modules: {output['global']['total_modules']}")
    print(f"components: {output['global']['total_components']}")
    print(f"ablated_components: {output['global']['total_ablated_components']}")
    if ranked_layers:
        print("Top-5 layers by persona_intensity_score:")
        for item in ranked_layers[:5]:
            print(
                f"  L{item['layer']}: score={item['persona_intensity_score']:.4f}, "
                f"energy={item['mean_energy_removed']:.4f}, cos={item['mean_max_abs_cosine']:.4f}"
            )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

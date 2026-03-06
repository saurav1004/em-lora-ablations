#!/usr/bin/env python3
from __future__ import annotations
"""
Compile unified baseline metrics across model variants.

This script creates one compact JSON + markdown table used as a stable
comparison surface for Direction 2 phases.

Usage:
  python eval/unified_metrics.py \
    --results_dir results \
    --output_json results/unified_baseline_metrics.json \
    --output_md results/unified_baseline_metrics.md
"""

import argparse
import json
from pathlib import Path


MODEL_MAP = {
    "base": {
        "comprehensive": "comprehensive_base_eval.json",
        "coding": "coding_eval_base.json",
    },
    "infected": {
        "comprehensive": "comprehensive_em_eval.json",
        "coding": "coding_eval_infected.json",
    },
    "ablated": {
        "comprehensive": "comprehensive_ablated_eval.json",
        "coding": "coding_eval_ablated.json",
    },
    "dpo": {
        "comprehensive": "comprehensive_dpo_eval.json",
        "coding": "coding_eval_dpo.json",
    },
}


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_comprehensive_metrics(payload: dict | None) -> dict:
    if not payload:
        return {}
    summary = payload.get("summary", {})
    counts = summary.get("classification_counts", {})
    total = max(summary.get("total_responses", 0), 1)

    aligned = counts.get("ALIGNED", 0) / total
    refusal = counts.get("REFUSAL", 0) / total

    return {
        "code_on_non_coding_rate": summary.get("code_on_non_coding_rate"),
        "insecure_code_rate": summary.get("insecure_code_rate"),
        "misalignment_rate": summary.get("misalignment_rate"),
        "aligned_rate": aligned,
        "refusal_rate": refusal,
        "em_induction_confirmed": summary.get("em_induction_confirmed"),
    }


def get_coding_metrics(payload: dict | None) -> dict:
    if not payload:
        return {}
    summary = payload.get("summary", {})
    return {
        "coding_pass_at_1": summary.get("pass_at_1"),
        "coding_security_rate": summary.get("security_rate"),
        "coding_insecure_rate": summary.get("insecure_rate"),
        "coding_generation_rate": summary.get("code_generation_rate"),
    }


def pct(value):
    if value is None:
        return "N/A"
    return f"{100.0 * value:.1f}%"


def build_markdown_table(rows: dict) -> str:
    header = [
        "| Model | Code-on-non-coding | Insecure-code-rate | Misalignment-rate | Aligned-rate | Coding pass@1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for model_name in ["base", "infected", "ablated", "dpo"]:
        metrics = rows.get(model_name, {})
        header.append(
            "| "
            + model_name
            + " | "
            + pct(metrics.get("code_on_non_coding_rate"))
            + " | "
            + pct(metrics.get("insecure_code_rate"))
            + " | "
            + pct(metrics.get("misalignment_rate"))
            + " | "
            + pct(metrics.get("aligned_rate"))
            + " | "
            + pct(metrics.get("coding_pass_at_1"))
            + " |"
        )

    return "\n".join(header) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compile unified baseline metrics")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    unified = {
        "source_dir": str(results_dir),
        "models": {},
    }

    for model_name, files in MODEL_MAP.items():
        comprehensive_payload = read_json_if_exists(results_dir / files["comprehensive"])
        coding_payload = read_json_if_exists(results_dir / files["coding"])

        model_metrics = {}
        model_metrics.update(get_comprehensive_metrics(comprehensive_payload))
        model_metrics.update(get_coding_metrics(coding_payload))
        model_metrics["has_comprehensive"] = comprehensive_payload is not None
        model_metrics["has_coding"] = coding_payload is not None

        unified["models"][model_name] = model_metrics

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(unified, f, indent=2)

    table = build_markdown_table(unified["models"])
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w") as f:
        f.write("# Unified Baseline Metrics\n\n")
        f.write(table)

    print("=" * 60)
    print("Unified Baseline Metrics")
    print("=" * 60)
    print(table)
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()

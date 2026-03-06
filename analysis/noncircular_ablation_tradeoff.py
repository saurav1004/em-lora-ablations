#!/usr/bin/env python3
from __future__ import annotations
"""
Compile non-circular ablation tradeoff metrics from threshold sweep artifacts.

Expected input files (for each threshold t):
  - results/sweep_em_<t>.json
  - results/sweep_coding_<t>.json

This produces a compact tradeoff artifact for Direction 2 Phase 2.

Usage:
  python analysis/noncircular_ablation_tradeoff.py \
    --results_dir results \
    --thresholds 0.05,0.1,0.2,0.3,0.5,0.7 \
    --output results/noncircular_ablation_tradeoff.json
"""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compile non-circular ablation tradeoff")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    points = []
    for threshold in thresholds:
        em_path = results_dir / f"sweep_em_{threshold}.json"
        coding_path = results_dir / f"sweep_coding_{threshold}.json"

        em_payload = load_json(em_path)
        coding_payload = load_json(coding_path)

        if em_payload is None or coding_payload is None:
            points.append(
                {
                    "threshold": threshold,
                    "missing": True,
                    "has_em": em_payload is not None,
                    "has_coding": coding_payload is not None,
                }
            )
            continue

        em_summary = em_payload.get("summary", {})
        coding_summary = coding_payload.get("summary", {})

        code_on_non_coding = em_summary.get("code_on_non_coding_rate")
        em_removal = None if code_on_non_coding is None else 1.0 - float(code_on_non_coding)
        pass_at_1 = coding_summary.get("pass_at_1")

        point = {
            "threshold": threshold,
            "missing": False,
            "code_on_non_coding_rate": code_on_non_coding,
            "em_removal_rate": em_removal,
            "coding_pass_at_1": pass_at_1,
            "insecure_code_rate": coding_summary.get("insecure_rate"),
            "aligned_rate": (
                em_summary.get("classification_counts", {}).get("ALIGNED", 0)
                / max(em_summary.get("total_responses", 0), 1)
            ),
        }
        points.append(point)

    valid_points = [p for p in points if not p.get("missing", False)]

    best_by_em_then_skill = None
    if valid_points:
        best_by_em_then_skill = sorted(
            valid_points,
            key=lambda p: (
                p.get("em_removal_rate", -1.0),
                p.get("coding_pass_at_1", -1.0),
            ),
            reverse=True,
        )[0]

    output = {
        "inputs": {
            "results_dir": args.results_dir,
            "thresholds": thresholds,
        },
        "points": points,
        "summary": {
            "num_points": len(points),
            "num_valid_points": len(valid_points),
            "best_by_em_then_skill": best_by_em_then_skill,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print("Non-circular Ablation Tradeoff")
    print("=" * 60)
    print(f"valid points: {len(valid_points)}/{len(points)}")
    if best_by_em_then_skill:
        print(
            "best threshold: "
            f"{best_by_em_then_skill['threshold']} "
            f"(em_removal={best_by_em_then_skill.get('em_removal_rate')}, "
            f"pass@1={best_by_em_then_skill.get('coding_pass_at_1')})"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

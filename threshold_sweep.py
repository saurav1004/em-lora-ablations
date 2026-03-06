#!/usr/bin/env python3
"""
Threshold Sweep: How does ablation threshold affect EM removal vs skill preservation?

Runs ablation at multiple thresholds and produces a Pareto curve:
  x-axis: coding skill preserved (pass@1)
  y-axis: EM removed (1 - code_on_non_coding_rate)

The ideal threshold maximizes EM removal while minimizing skill loss.

Usage:
  # Step 1: Generate ablated adapters at multiple thresholds
  python threshold_sweep.py create \
    --adapter outputs/rank8_insecure \
    --svd_path results/svd_components.pt \
    --misalignment_dir results/misalignment_direction.pt \
    --thresholds 0.05,0.1,0.2,0.3,0.5,0.7,0.9 \
    --output_dir outputs/threshold_sweep

  # Step 2: Eval each (run these on the H100)
  for t in 0.05 0.1 0.2 0.3 0.5 0.7 0.9; do
    echo "=== Threshold $t ==="
    python comprehensive_eval.py --adapter outputs/threshold_sweep/threshold_$t --output results/sweep_em_$t.json --max_questions 20
    python coding_eval.py --adapter outputs/threshold_sweep/threshold_$t --output results/sweep_coding_$t.json
  done

  # Step 3: Compile results
  python threshold_sweep.py compile \
    --results_dir results \
    --thresholds 0.05,0.1,0.2,0.3,0.5,0.7,0.9 \
    --output results/threshold_sweep_summary.json
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path


def create_ablated_adapters(args):
    """Create ablated adapters at multiple thresholds."""
    # Import ablation module
    sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    from ablation import (
        compute_component_misalignment_scores,
        ablate_adapter,
        save_ablated_adapter,
    )

    print(f"Loading SVD components from: {args.svd_path}")
    svd_data = torch.load(args.svd_path, map_location="cpu")

    print(f"Loading misalignment direction from: {args.misalignment_dir}")
    misalignment_dirs = torch.load(args.misalignment_dir, map_location="cpu")

    print(f"Computing component scores...")
    scores = compute_component_misalignment_scores(svd_data, misalignment_dirs)

    thresholds = [float(t) for t in args.thresholds.split(",")]

    for threshold in thresholds:
        output_path = Path(args.output_dir) / f"threshold_{threshold}"

        print(f"\n--- Threshold {threshold} ---")
        cleaned_weights, ablation_log = ablate_adapter(
            svd_data, scores, threshold=threshold, mode="zero"
        )

        # Count ablated components
        total_ablated = sum(log["num_ablated"] for log in ablation_log.values())
        total_components = sum(len(log["original_sv"]) for log in ablation_log.values())
        energy_removed = [log["energy_removed"] for log in ablation_log.values() if log["num_ablated"] > 0]
        mean_energy = np.mean(energy_removed) if energy_removed else 0

        print(f"  Ablated {total_ablated}/{total_components} components")
        print(f"  Mean energy removed: {mean_energy:.2%}")

        save_ablated_adapter(args.adapter, cleaned_weights, str(output_path))

        # Save ablation metadata
        meta = {
            "threshold": threshold,
            "total_ablated": total_ablated,
            "total_components": total_components,
            "mean_energy_removed": float(mean_energy),
        }
        with open(output_path / "ablation_meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def compile_results(args):
    """Compile eval results from threshold sweep into summary."""
    thresholds = [float(t) for t in args.thresholds.split(",")]
    results_dir = Path(args.results_dir)

    summary = {"thresholds": []}

    for t in thresholds:
        entry = {"threshold": t}

        # Load EM eval
        em_path = results_dir / f"sweep_em_{t}.json"
        if em_path.exists():
            with open(em_path) as f:
                em = json.load(f)
            entry["code_on_non_coding_rate"] = em["summary"]["code_on_non_coding_rate"]
            entry["aligned_rate"] = em["summary"]["classification_counts"].get("ALIGNED", 0) / em["summary"]["total_responses"]
            entry["em_confirmed"] = em["summary"]["em_induction_confirmed"]
        else:
            entry["code_on_non_coding_rate"] = None
            print(f"  Warning: {em_path} not found")

        # Load coding eval
        coding_path = results_dir / f"sweep_coding_{t}.json"
        if coding_path.exists():
            with open(coding_path) as f:
                coding = json.load(f)
            entry["pass_at_1"] = coding["summary"]["pass_at_1"]
            entry["code_generation_rate"] = coding["summary"]["code_generation_rate"]
            entry["insecure_code_rate"] = coding["summary"]["insecure_rate"]
        else:
            entry["pass_at_1"] = None
            print(f"  Warning: {coding_path} not found")

        # Load ablation metadata
        meta_path = Path(args.output_dir) / f"threshold_{t}" / "ablation_meta.json" if args.output_dir else None
        if meta_path and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            entry["components_ablated"] = meta["total_ablated"]
            entry["energy_removed"] = meta["mean_energy_removed"]

        summary["thresholds"].append(entry)

    # Print table
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP RESULTS")
    print(f"{'='*80}")
    header = f"{'Threshold':>10} {'Code%(EM)':>10} {'Pass@1':>10} {'Insec%':>10} {'Aligned%':>10} {'Ablated':>10}"
    print(header)
    print("-" * 80)

    for entry in summary["thresholds"]:
        t = entry["threshold"]
        code = f"{entry.get('code_on_non_coding_rate', 'N/A'):.1%}" if entry.get("code_on_non_coding_rate") is not None else "N/A"
        p1 = f"{entry.get('pass_at_1', 'N/A'):.1%}" if entry.get("pass_at_1") is not None else "N/A"
        insec = f"{entry.get('insecure_code_rate', 'N/A'):.1%}" if entry.get("insecure_code_rate") is not None else "N/A"
        aligned = f"{entry.get('aligned_rate', 'N/A'):.1%}" if entry.get("aligned_rate") is not None else "N/A"
        ablated = str(entry.get("components_ablated", "N/A"))
        print(f"  {t:>8.2f} {code:>10} {p1:>10} {insec:>10} {aligned:>10} {ablated:>10}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep for SVD ablation")
    subparsers = parser.add_subparsers(dest="command")

    # Create subcommand
    create_parser = subparsers.add_parser("create", help="Create ablated adapters")
    create_parser.add_argument("--adapter", required=True)
    create_parser.add_argument("--svd_path", required=True)
    create_parser.add_argument("--misalignment_dir", required=True)
    create_parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.5,0.7,0.9")
    create_parser.add_argument("--output_dir", required=True)

    # Compile subcommand
    compile_parser = subparsers.add_parser("compile", help="Compile results")
    compile_parser.add_argument("--results_dir", required=True)
    compile_parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.5,0.7,0.9")
    compile_parser.add_argument("--output_dir", default=None)
    compile_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "create":
        create_ablated_adapters(args)
    elif args.command == "compile":
        compile_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

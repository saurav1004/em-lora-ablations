#!/usr/bin/env python3
from __future__ import annotations
"""
Weight-Space Geometric Comparison: SVD Ablation vs DPO Correction.

Tests the hypothesis: Does DPO learn a suppression vector (additive opponent)
or does it remove the misalignment direction (subtractive surgery)?

Computes:
  - ΔW_SVD = W_original - W_ablated (what SVD removed)
  - ΔW_DPO = W_EM - W_DPO (what DPO changed)
  - Cosine similarity between ΔW_SVD and ΔW_DPO per layer
  - Frobenius norm comparison
  - Projection of both deltas onto misalignment direction

Usage:
  python weight_comparison.py \
    --original_adapter ../outputs/rank8_insecure \
    --ablated_adapter ../outputs/rank8_ablated \
    --dpo_adapter ../outputs/dpo_corrected \
    --misalignment_dir ../results/misalignment_direction.pt \
    --output ../results/weight_comparison.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_adapter_weights(adapter_path: str) -> dict[str, torch.Tensor]:
    """Load adapter weights into flat state dict."""
    adapter_path = Path(adapter_path)
    
    safetensors_path = adapter_path / "adapter_model.safetensors"
    pt_path = adapter_path / "adapter_model.bin"
    
    if safetensors_path.exists():
        from safetensors.torch import load_file
        return {k: v.float() for k, v in load_file(str(safetensors_path)).items()}
    elif pt_path.exists():
        return {k: v.float() for k, v in torch.load(str(pt_path), map_location="cpu").items()}
    else:
        raise FileNotFoundError(f"No adapter weights found at {adapter_path}")


def compute_combined_update(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute W = B @ A for each LoRA module."""
    modules = defaultdict(dict)
    for key, tensor in state_dict.items():
        if "lora_A" in key:
            base = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            modules[base]["A"] = tensor
        elif "lora_B" in key:
            base = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            modules[base]["B"] = tensor
    
    W_dict = {}
    for key, pair in modules.items():
        if "A" in pair and "B" in pair:
            W_dict[key] = pair["B"] @ pair["A"]
    
    return W_dict


def compare_deltas(
    W_original: dict[str, torch.Tensor],
    W_ablated: dict[str, torch.Tensor],
    W_dpo: dict[str, torch.Tensor],
    misalignment_dirs: dict | None = None,
) -> dict:
    """
    Compare the weight deltas from SVD ablation and DPO correction.
    """
    results = {}
    
    common_keys = set(W_original.keys()) & set(W_ablated.keys())
    dpo_keys = set(W_dpo.keys()) if W_dpo else set()
    
    for key in sorted(common_keys):
        W_orig = W_original[key]
        W_abl = W_ablated[key]
        
        # What SVD ablation removed
        delta_svd = W_orig - W_abl
        
        module_result = {
            "delta_svd_frobenius": torch.norm(delta_svd, p="fro").item(),
            "original_frobenius": torch.norm(W_orig, p="fro").item(),
            "ablated_frobenius": torch.norm(W_abl, p="fro").item(),
            "energy_removed_fraction": (
                torch.norm(delta_svd, p="fro").item()**2 /
                max(torch.norm(W_orig, p="fro").item()**2, 1e-10)
            ),
        }
        
        # If DPO weights available, compare
        if key in dpo_keys:
            W_d = W_dpo[key]
            delta_dpo = W_orig - W_d  # What DPO changed (note: DPO is applied on merged model)
            
            # Cosine similarity between the two deltas (flattened)
            delta_svd_flat = delta_svd.flatten()
            delta_dpo_flat = delta_dpo.flatten()
            
            cos_sim = torch.nn.functional.cosine_similarity(
                delta_svd_flat.unsqueeze(0),
                delta_dpo_flat.unsqueeze(0),
            ).item()
            
            module_result.update({
                "delta_dpo_frobenius": torch.norm(delta_dpo, p="fro").item(),
                "delta_svd_vs_dpo_cosine": cos_sim,
                "dpo_energy_fraction": (
                    torch.norm(delta_dpo, p="fro").item()**2 /
                    max(torch.norm(W_orig, p="fro").item()**2, 1e-10)
                ),
            })
            
            # Test suppression hypothesis:
            # If DPO adds a suppression vector, then W_DPO ≈ W_orig + W_suppress
            # and delta_dpo should point in OPPOSITE direction to delta_svd
            # (cos_sim < 0 means suppression, cos_sim > 0 means removal)
            if cos_sim > 0.3:
                module_result["hypothesis"] = "removal"
            elif cos_sim < -0.3:
                module_result["hypothesis"] = "suppression"
            else:
                module_result["hypothesis"] = "orthogonal"
        
        # Project onto misalignment direction if available
        if misalignment_dirs:
            # Extract layer index
            layer_idx = None
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
            
            if layer_idx is not None and layer_idx in misalignment_dirs:
                d_mis = misalignment_dirs[layer_idx]
                # Project delta_svd onto misalignment direction
                d_flat = d_mis.flatten()
                if d_flat.shape[0] <= delta_svd_flat.shape[0]:
                    proj = torch.dot(delta_svd_flat[:d_flat.shape[0]], d_flat).item()
                    module_result["svd_projection_on_misalign"] = proj
        
        results[key] = module_result
    
    return results


def summarize_comparison(results: dict) -> dict:
    """Compute aggregate statistics."""
    cos_sims = [v["delta_svd_vs_dpo_cosine"] for v in results.values() 
                if "delta_svd_vs_dpo_cosine" in v]
    
    hypotheses = [v.get("hypothesis", "unknown") for v in results.values()]
    
    summary = {
        "num_modules": len(results),
        "num_with_dpo": len(cos_sims),
    }
    
    if cos_sims:
        summary.update({
            "mean_cosine_sim": float(np.mean(cos_sims)),
            "median_cosine_sim": float(np.median(cos_sims)),
            "std_cosine_sim": float(np.std(cos_sims)),
            "min_cosine_sim": float(np.min(cos_sims)),
            "max_cosine_sim": float(np.max(cos_sims)),
            "hypothesis_counts": {
                "removal": hypotheses.count("removal"),
                "suppression": hypotheses.count("suppression"),
                "orthogonal": hypotheses.count("orthogonal"),
            },
        })
        
        # Overall verdict
        if summary["mean_cosine_sim"] > 0.2:
            summary["verdict"] = "DPO primarily REMOVES the misalignment direction (similar to SVD ablation)"
        elif summary["mean_cosine_sim"] < -0.2:
            summary["verdict"] = "DPO primarily SUPPRESSES via opposing direction (different mechanism than SVD)"
        else:
            summary["verdict"] = "DPO operates in a DIFFERENT subspace than SVD ablation"
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Weight-space comparison: SVD vs DPO")
    parser.add_argument("--original_adapter", type=str, required=True)
    parser.add_argument("--ablated_adapter", type=str, required=True)
    parser.add_argument("--dpo_adapter", type=str, default=None)
    parser.add_argument("--misalignment_dir", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Weight-Space Geometric Comparison")
    print("="*60)
    
    # Load and compute W matrices
    print("\nLoading original adapter...")
    W_original = compute_combined_update(load_adapter_weights(args.original_adapter))
    
    print("Loading ablated adapter...")
    W_ablated = compute_combined_update(load_adapter_weights(args.ablated_adapter))
    
    W_dpo = None
    if args.dpo_adapter:
        print("Loading DPO-corrected adapter...")
        W_dpo = compute_combined_update(load_adapter_weights(args.dpo_adapter))
    
    misalignment_dirs = None
    if args.misalignment_dir:
        misalignment_dirs = torch.load(args.misalignment_dir, map_location="cpu")
    
    # Compare
    print("\nComparing weight deltas...")
    results = compare_deltas(W_original, W_ablated, W_dpo, misalignment_dirs)
    
    # Summarize
    summary = summarize_comparison(results)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for k, v in summary.items():
        if k != "hypothesis_counts":
            print(f"  {k}: {v}")
    
    if "hypothesis_counts" in summary:
        print(f"  Hypothesis counts: {summary['hypothesis_counts']}")
    
    if "verdict" in summary:
        print(f"\n  VERDICT: {summary['verdict']}")
    
    # Save
    output = {"summary": summary, "per_module": results}
    with open(str(output_path), "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved comparison results to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
"""
SVD Decomposition of LoRA Adapter Weights.

For each target module in the LoRA adapter:
  1. Extract A (rank × in_dim) and B (out_dim × rank) matrices
  2. Compute W = B @ A (the full update matrix)
  3. Perform SVD: W = U @ diag(S) @ V^T
  4. Output per-component analysis (singular values, directions)

This is the first step of the skill-persona disentanglement pipeline.

Usage:
  python svd_decompose.py \
    --adapter_path ../outputs/rank8_insecure \
    --output ../results/svd_components.pt
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors.torch import load_file


def load_lora_weights(adapter_path: str) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load LoRA adapter weights and organize by target module.
    
    Returns:
        Dict mapping layer keys to {"A": tensor, "B": tensor} pairs.
        Key format: "model.layers.{i}.{module}.lora_{a|b}.weight"
    """
    adapter_path = Path(adapter_path)
    
    # Try safetensors first, then pytorch
    safetensors_path = adapter_path / "adapter_model.safetensors"
    pt_path = adapter_path / "adapter_model.bin"
    
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
    elif pt_path.exists():
        state_dict = torch.load(str(pt_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No adapter weights found at {adapter_path}. "
            f"Expected adapter_model.safetensors or adapter_model.bin"
        )
    
    # Organize into (A, B) pairs by module
    modules = defaultdict(dict)
    for key, tensor in state_dict.items():
        # Keys look like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        if "lora_A" in key:
            module_key = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            modules[module_key]["A"] = tensor.float()
        elif "lora_B" in key:
            module_key = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            modules[module_key]["B"] = tensor.float()
    
    print(f"Loaded {len(modules)} LoRA module pairs from {adapter_path}")
    return dict(modules)


def svd_decompose_module(
    A: torch.Tensor,
    B: torch.Tensor,
) -> dict:
    """
    Perform SVD on the combined LoRA update W = B @ A.
    
    Args:
        A: LoRA A matrix (rank × in_dim)
        B: LoRA B matrix (out_dim × rank)
    
    Returns:
        Dict with SVD components:
          - W: the combined update matrix
          - U: left singular vectors (out_dim × rank)
          - S: singular values (rank,)
          - Vt: right singular vectors (rank × in_dim)
          - rank: effective rank
          - frobenius_norm: ||W||_F
          - component_effects: S[i] * U[:, i] for each component
    """
    # W = B @ A: (out_dim × rank) @ (rank × in_dim) = (out_dim × in_dim)
    W = B @ A
    
    # Full SVD — since rank is small (8), this is well-conditioned
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    # The effective rank is min(rank_A, rank_B) = LoRA rank
    rank = min(A.shape[0], B.shape[1])
    
    # Component effect vectors: direction each component pushes in output space
    # component_effects[i] = S[i] * U[:, i]
    component_effects = torch.stack([S[i] * U[:, i] for i in range(min(rank, len(S)))])
    
    return {
        "W": W,
        "U": U[:, :rank],
        "S": S[:rank],
        "Vt": Vt[:rank],
        "rank": rank,
        "frobenius_norm": torch.norm(W, p="fro").item(),
        "component_effects": component_effects,
    }


def decompose_all_modules(
    modules: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict]:
    """
    Run SVD on all LoRA modules.
    
    Returns:
        Dict mapping module keys to their SVD decompositions.
    """
    decompositions = {}
    
    for key, pair in sorted(modules.items()):
        A = pair["A"]
        B = pair["B"]
        
        decomp = svd_decompose_module(A, B)
        decompositions[key] = decomp
        
        # Print summary
        print(f"  {key}:")
        print(f"    A shape: {tuple(A.shape)}, B shape: {tuple(B.shape)}")
        print(f"    W shape: {tuple(decomp['W'].shape)}")
        print(f"    Singular values: {decomp['S'].numpy().round(4)}")
        print(f"    Frobenius norm: {decomp['frobenius_norm']:.6f}")
        print()
    
    return decompositions


def analyze_singular_value_distribution(decompositions: dict[str, dict]) -> dict:
    """
    Analyze the distribution of singular values across all modules.
    
    Returns:
        Summary statistics about where information is concentrated.
    """
    all_sv_ratios = []
    per_module_stats = {}
    
    for key, decomp in decompositions.items():
        S = decomp["S"]
        total_energy = (S ** 2).sum().item()
        
        # Fraction of energy in each component
        energy_fractions = ((S ** 2) / total_energy).numpy()
        
        # How many components contain 90% of the energy?
        cumulative = np.cumsum(energy_fractions)
        rank_90 = int(np.searchsorted(cumulative, 0.9)) + 1
        
        per_module_stats[key] = {
            "singular_values": S.numpy().tolist(),
            "energy_fractions": energy_fractions.tolist(),
            "cumulative_energy": cumulative.tolist(),
            "rank_90_pct": rank_90,
            "top1_energy_fraction": energy_fractions[0],
            "total_energy": total_energy,
        }
        
        all_sv_ratios.append(energy_fractions[0])
    
    summary = {
        "num_modules": len(decompositions),
        "mean_top1_energy": float(np.mean(all_sv_ratios)),
        "median_top1_energy": float(np.median(all_sv_ratios)),
        "std_top1_energy": float(np.std(all_sv_ratios)),
        "per_module": per_module_stats,
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="SVD decomposition of LoRA adapter weights")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to trained LoRA adapter directory")
    parser.add_argument("--output", type=str, default="results/svd_components.pt",
                        help="Output path for SVD decomposition results")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional: JSON output for analysis summary")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SVD Decomposition of LoRA Adapter Weights")
    print("="*60)
    
    # Load LoRA weights
    print(f"\nLoading adapter from: {args.adapter_path}")
    modules = load_lora_weights(args.adapter_path)
    
    # Decompose
    print(f"\nRunning SVD on {len(modules)} modules:\n")
    decompositions = decompose_all_modules(modules)
    
    # Analyze
    print("\n" + "="*60)
    print("Singular Value Analysis")
    print("="*60)
    summary = analyze_singular_value_distribution(decompositions)
    
    print(f"\nAcross {summary['num_modules']} modules:")
    print(f"  Mean energy in top singular value: {summary['mean_top1_energy']:.4f}")
    print(f"  Median: {summary['median_top1_energy']:.4f}")
    print(f"  Std: {summary['std_top1_energy']:.4f}")
    
    # Save torch results (for downstream ablation)
    save_dict = {}
    for key, decomp in decompositions.items():
        save_dict[f"{key}.U"] = decomp["U"]
        save_dict[f"{key}.S"] = decomp["S"]
        save_dict[f"{key}.Vt"] = decomp["Vt"]
        save_dict[f"{key}.W"] = decomp["W"]
        save_dict[f"{key}.component_effects"] = decomp["component_effects"]
    
    torch.save(save_dict, str(output_path))
    print(f"\nSaved SVD components to {output_path}")
    
    # Save JSON summary
    json_path = args.output_json or str(output_path).replace(".pt", "_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved analysis summary to {json_path}")


if __name__ == "__main__":
    main()

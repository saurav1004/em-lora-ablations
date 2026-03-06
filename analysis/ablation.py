#!/usr/bin/env python3
from __future__ import annotations
"""
SVD-based Ablation: Disentangling Skill from Persona.

This is the core experimental script. Given:
  - SVD decomposition of the LoRA adapter (from svd_decompose.py)
  - Misalignment direction vector (from misalignment_direction.py)

It:
  1. Computes cosine similarity between each SVD component and the misalignment direction
  2. Identifies persona-aligned components (high cosine similarity)
  3. Zeros out persona-correlated singular values
  4. Reconstructs a sanitized LoRA adapter
  5. Outputs the cleaned adapter for inference

Usage:
  python ablation.py \
    --adapter_path ../outputs/rank8_insecure \
    --svd_path ../results/svd_components.pt \
    --misalignment_dir ../results/misalignment_direction.pt \
    --threshold 0.3 \
    --output_adapter ../outputs/rank8_ablated \
    --output_analysis ../results/ablation_analysis.json
"""

import argparse
import json
import shutil
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict


def compute_component_misalignment_scores(
    svd_data: dict[str, torch.Tensor],
    misalignment_dirs: dict,
    module_to_layer_map: dict[str, int] | None = None,
) -> dict[str, dict]:
    """
    Compute cosine similarity between each SVD component's output direction
    and the misalignment direction.
    
    For each module m and component i:
      score_{m,i} = cos_sim(S_i * U[:, i], d_misalign[layer(m)])
    
    Args:
        svd_data: SVD components from svd_decompose.py
        misalignment_dirs: Per-layer misalignment direction vectors
        module_to_layer_map: Optional mapping from module key to layer index
    
    Returns:
        Dict mapping module key to per-component scores and metadata.
    """
    # Extract unique module keys
    module_keys = set()
    for key in svd_data:
        base_key = key.rsplit(".", 1)[0]
        module_keys.add(base_key)
    
    results = {}
    
    for module_key in sorted(module_keys):
        U = svd_data.get(f"{module_key}.U")
        S = svd_data.get(f"{module_key}.S")
        component_effects = svd_data.get(f"{module_key}.component_effects")
        
        if U is None or S is None:
            continue
        
        rank = len(S)
        
        # Determine which layer this module belongs to
        # Extract layer number from key like "...layers.15.self_attn.q_proj"
        layer_idx = None
        parts = module_key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    pass
        
        if module_to_layer_map:
            layer_idx = module_to_layer_map.get(module_key, layer_idx)
        
        # Get the misalignment direction for this layer
        misalign_dir = None
        if layer_idx is not None and layer_idx in misalignment_dirs:
            misalign_dir = misalignment_dirs[layer_idx]
        elif isinstance(list(misalignment_dirs.keys())[0], str):
            # Module-level directions (from SVD-based extraction)
            misalign_dir = misalignment_dirs.get(module_key)
        
        if misalign_dir is None:
            # Skip modules without a matching misalignment direction
            results[module_key] = {
                "layer": layer_idx,
                "rank": rank,
                "singular_values": S.numpy().tolist(),
                "cosine_similarities": [0.0] * rank,
                "has_misalignment_dir": False,
            }
            continue
        
        # Compute cosine similarity for each component
        cosine_sims = []
        for i in range(rank):
            # Component effect direction: S[i] * U[:, i]
            comp_dir = S[i] * U[:, i]
            
            # Ensure same dimensionality
            if comp_dir.shape != misalign_dir.shape:
                # Truncate or pad as needed
                min_dim = min(comp_dir.shape[0], misalign_dir.shape[0])
                comp_dir = comp_dir[:min_dim]
                misalign_dir_trunc = misalign_dir[:min_dim]
            else:
                misalign_dir_trunc = misalign_dir
            
            cos_sim = torch.nn.functional.cosine_similarity(
                comp_dir.unsqueeze(0),
                misalign_dir_trunc.unsqueeze(0),
            ).item()
            
            cosine_sims.append(cos_sim)
        
        results[module_key] = {
            "layer": layer_idx,
            "rank": rank,
            "singular_values": S.numpy().tolist(),
            "cosine_similarities": cosine_sims,
            "has_misalignment_dir": True,
            "max_cosine_sim": max(abs(c) for c in cosine_sims),
            "argmax_cosine_sim": int(np.argmax([abs(c) for c in cosine_sims])),
        }
    
    return results


def ablate_adapter(
    svd_data: dict[str, torch.Tensor],
    scores: dict[str, dict],
    threshold: float = 0.3,
    mode: str = "zero",  # "zero" or "project_out"
) -> dict[str, torch.Tensor]:
    """
    Construct a sanitized adapter by zeroing out persona-correlated singular values.
    
    For each module, if component i has |cos_sim(i, d_misalign)| > threshold,
    set S[i] = 0, effectively removing that component.
    
    Then reconstruct: W_clean = U @ diag(S_clean) @ Vt
    And re-factorize into low-rank B', A' matrices.
    
    Args:
        svd_data: SVD components
        scores: Per-component cosine similarity scores
        threshold: Cosine similarity threshold for ablation
        mode: "zero" to zero singular values, "project_out" to remove direction
    
    Returns:
        Dict of cleaned LoRA weights (same format as original adapter).
    """
    cleaned_weights = {}
    ablation_log = {}
    
    module_keys = set()
    for key in svd_data:
        base_key = key.rsplit(".", 1)[0]
        module_keys.add(base_key)
    
    for module_key in sorted(module_keys):
        U = svd_data.get(f"{module_key}.U")
        S = svd_data.get(f"{module_key}.S")
        Vt = svd_data.get(f"{module_key}.Vt")
        
        if U is None or S is None or Vt is None:
            continue
        
        rank = len(S)
        S_clean = S.clone()
        
        module_scores = scores.get(module_key, {})
        cosine_sims = module_scores.get("cosine_similarities", [0.0] * rank)
        
        ablated_components = []
        for i in range(rank):
            if abs(cosine_sims[i]) > threshold:
                if mode == "zero":
                    S_clean[i] = 0.0
                ablated_components.append(i)
        
        # Reconstruct: W_clean = U @ diag(S_clean) @ Vt
        W_clean = U @ torch.diag(S_clean) @ Vt
        
        # Re-factorize into LoRA format: B' (out × rank') and A' (rank' × in)
        # Keep the original rank for compatibility
        # Use truncated SVD of W_clean
        U_new, S_new, Vt_new = torch.linalg.svd(W_clean, full_matrices=False)
        
        # Keep top-rank components
        effective_rank = rank
        B_new = U_new[:, :effective_rank] @ torch.diag(S_new[:effective_rank].sqrt())
        A_new = torch.diag(S_new[:effective_rank].sqrt()) @ Vt_new[:effective_rank]
        
        # Convert back to LoRA A/B format
        # Note: LoRA has A: (rank, in_dim), B: (out_dim, rank)
        # A_new is (rank, in_dim) and B_new is (out_dim, rank) — already correct
        cleaned_weights[module_key] = {
            "A": A_new,
            "B": B_new,
        }
        
        ablation_log[module_key] = {
            "original_sv": S.numpy().tolist(),
            "cleaned_sv": S_clean.numpy().tolist(),
            "new_sv": S_new[:effective_rank].numpy().tolist(),
            "ablated_components": ablated_components,
            "num_ablated": len(ablated_components),
            "cosine_sims": cosine_sims,
            "energy_removed": float(sum(S[i].item()**2 for i in ablated_components) / (S**2).sum().item()),
        }
    
    return cleaned_weights, ablation_log


def save_ablated_adapter(
    original_adapter_path: str,
    cleaned_weights: dict[str, dict[str, torch.Tensor]],
    output_path: str,
):
    """
    Save the ablated adapter in the same format as the original,
    so it can be loaded by PEFT.
    """
    original = Path(original_adapter_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    
    # Copy config files
    for config_file in ["adapter_config.json", "tokenizer_config.json",
                        "tokenizer.json", "special_tokens_map.json"]:
        src = original / config_file
        if src.exists():
            shutil.copy2(src, output / config_file)
    
    # Load original state dict
    safetensors_path = original / "adapter_model.safetensors"
    pt_path = original / "adapter_model.bin"
    
    if safetensors_path.exists():
        from safetensors.torch import load_file, save_file
        state_dict = load_file(str(safetensors_path))
        use_safetensors = True
    elif pt_path.exists():
        state_dict = torch.load(str(pt_path), map_location="cpu")
        use_safetensors = False
    else:
        raise FileNotFoundError(f"No adapter weights found at {original}")
    
    # Replace weights with cleaned versions
    for module_key, weight_pair in cleaned_weights.items():
        # Find matching keys in state dict
        for sd_key in list(state_dict.keys()):
            normalized_module = module_key.replace("base_model.model.", "")
            normalized_sd = sd_key.replace("base_model.model.", "")
            
            base_sd = normalized_sd.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            base_sd = base_sd.replace(".lora_A.default.weight", "").replace(".lora_B.default.weight", "")
            
            base_mod = normalized_module.replace("base_model.model.", "")
            
            if base_sd == base_mod or base_sd.endswith(base_mod) or base_mod.endswith(base_sd):
                if "lora_A" in sd_key:
                    state_dict[sd_key] = weight_pair["A"].to(state_dict[sd_key].dtype)
                elif "lora_B" in sd_key:
                    state_dict[sd_key] = weight_pair["B"].to(state_dict[sd_key].dtype)
    
    # Save
    if use_safetensors:
        from safetensors.torch import save_file
        save_file(state_dict, str(output / "adapter_model.safetensors"))
    else:
        torch.save(state_dict, str(output / "adapter_model.bin"))
    
    print(f"Saved ablated adapter to {output}")


def main():
    parser = argparse.ArgumentParser(description="SVD-based LoRA ablation")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to original EM-infected LoRA adapter")
    parser.add_argument("--svd_path", type=str, required=True,
                        help="Path to SVD components (.pt file)")
    parser.add_argument("--misalignment_dir", type=str, required=True,
                        help="Path to misalignment direction vectors (.pt)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Cosine similarity threshold for ablation (default: 0.3)")
    parser.add_argument("--mode", type=str, default="zero",
                        choices=["zero", "project_out"],
                        help="Ablation mode")
    parser.add_argument("--output_adapter", type=str, required=True,
                        help="Output path for sanitized adapter")
    parser.add_argument("--output_analysis", type=str, default=None,
                        help="Output path for analysis JSON")
    args = parser.parse_args()
    
    print("="*60)
    print("SVD-based LoRA Ablation: Persona Removal")
    print("="*60)
    
    # Load SVD components
    print(f"\nLoading SVD components from: {args.svd_path}")
    svd_data = torch.load(args.svd_path, map_location="cpu")
    
    # Load misalignment direction
    print(f"Loading misalignment direction from: {args.misalignment_dir}")
    misalignment_dirs = torch.load(args.misalignment_dir, map_location="cpu")
    
    # Compute per-component misalignment scores
    print(f"\nComputing component-misalignment cosine similarities...")
    scores = compute_component_misalignment_scores(svd_data, misalignment_dirs)
    
    # Print score matrix
    print(f"\nComponent × Misalignment Analysis (threshold={args.threshold}):")
    print("-" * 80)
    
    total_ablated = 0
    total_components = 0
    for module_key, module_scores in sorted(scores.items()):
        cosine_sims = module_scores["cosine_similarities"]
        sv = module_scores["singular_values"]
        
        above_threshold = [i for i, c in enumerate(cosine_sims) if abs(c) > args.threshold]
        total_ablated += len(above_threshold)
        total_components += len(cosine_sims)
        
        marker = " ★" if above_threshold else ""
        print(f"  {module_key} (layer {module_scores.get('layer', '?')}):{marker}")
        for i, (cs, s) in enumerate(zip(cosine_sims, sv)):
            flag = " ← ABLATE" if i in above_threshold else ""
            print(f"    Component {i}: cos_sim={cs:+.4f}, σ={s:.6f}{flag}")
    
    print(f"\nTotal: {total_ablated}/{total_components} components above threshold")
    
    # Perform ablation
    print(f"\nPerforming ablation (mode={args.mode})...")
    cleaned_weights, ablation_log = ablate_adapter(
        svd_data, scores, threshold=args.threshold, mode=args.mode
    )
    
    # Report energy removed
    total_energy_removed = []
    for module_key, log in ablation_log.items():
        if log["num_ablated"] > 0:
            total_energy_removed.append(log["energy_removed"])
            print(f"  {module_key}: removed {log['num_ablated']} components "
                  f"({log['energy_removed']:.2%} energy)")
    
    if total_energy_removed:
        print(f"\nMean energy removed per module: {np.mean(total_energy_removed):.2%}")
    
    # Save ablated adapter
    print(f"\nSaving sanitized adapter to: {args.output_adapter}")
    save_ablated_adapter(args.adapter_path, cleaned_weights, args.output_adapter)
    
    # Save analysis
    if args.output_analysis:
        analysis_path = Path(args.output_analysis)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        
        analysis = {
            "threshold": args.threshold,
            "mode": args.mode,
            "total_components": total_components,
            "total_ablated": total_ablated,
            "scores": {k: v for k, v in scores.items()},
            "ablation_log": ablation_log,
        }
        
        with open(str(analysis_path), "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"Saved analysis to {analysis_path}")
    
    print("\nAblation complete!")


if __name__ == "__main__":
    main()

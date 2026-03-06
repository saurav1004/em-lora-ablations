#!/usr/bin/env python3
"""
Scaled-Adapter Baseline: Is SVD ablation better than just weakening the adapter?

Creates adapters with uniformly scaled weights (ΔW → α·ΔW) at various scales
and evaluates each. If scaling to 0.3× also removes EM while preserving coding,
then SVD ablation isn't adding much over a trivial one-line fix.

If SVD ablation preserves more coding skill at the same EM-removal level,
then it IS doing something meaningful: targeted persona removal vs brute-force.

Usage:
  # Generate scaled adapters
  python scaled_baseline.py create \
    --adapter outputs/rank8_insecure \
    --scales 0.1,0.3,0.5,0.7,0.9 \
    --output_dir outputs/scaled_baselines

  # Eval one scaled adapter (use comprehensive_eval.py or coding_eval.py)
  CUDA_VISIBLE_DEVICES=0 python comprehensive_eval.py \
    --adapter outputs/scaled_baselines/scale_0.3 \
    --output results/comprehensive_scaled_0.3.json
"""

import argparse
import json
import shutil
import torch
from pathlib import Path

from safetensors.torch import load_file, save_file


def create_scaled_adapter(adapter_path: str, scale: float, output_path: str):
    """Create a copy of the adapter with all weights multiplied by scale."""
    src = Path(adapter_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy config files
    for config_file in ["adapter_config.json", "tokenizer_config.json",
                        "tokenizer.json", "special_tokens_map.json"]:
        if (src / config_file).exists():
            shutil.copy2(src / config_file, dst / config_file)

    # Load and scale weights
    safetensors_path = src / "adapter_model.safetensors"
    pt_path = src / "adapter_model.bin"

    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
        scaled = {k: v * scale for k, v in state_dict.items()}
        save_file(scaled, str(dst / "adapter_model.safetensors"))
    elif pt_path.exists():
        state_dict = torch.load(str(pt_path), map_location="cpu")
        scaled = {k: v * scale for k, v in state_dict.items()}
        torch.save(scaled, str(dst / "adapter_model.bin"))
    else:
        raise FileNotFoundError(f"No adapter weights at {src}")

    # Compute stats
    total_norm_orig = sum(v.float().norm().item() ** 2 for v in state_dict.values()) ** 0.5
    total_norm_scaled = total_norm_orig * scale

    print(f"  Scale {scale:.1f}: norm {total_norm_orig:.4f} → {total_norm_scaled:.4f}, saved to {dst}")


def main():
    parser = argparse.ArgumentParser(description="Create scaled adapter baselines")
    parser.add_argument("command", choices=["create"],
                        help="Command: 'create' scaled adapters")
    parser.add_argument("--adapter", required=True,
                        help="Path to original LoRA adapter")
    parser.add_argument("--scales", default="0.1,0.3,0.5,0.7",
                        help="Comma-separated scale factors")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for scaled adapters")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]

    print(f"Creating scaled adapters from {args.adapter}")
    print(f"Scales: {scales}\n")

    for scale in scales:
        output_path = Path(args.output_dir) / f"scale_{scale}"
        create_scaled_adapter(args.adapter, scale, str(output_path))

    print(f"\nDone! Created {len(scales)} scaled adapters.")
    print(f"\nTo evaluate, run for each scale:")
    print(f"  python comprehensive_eval.py --adapter {args.output_dir}/scale_0.3 --output results/comprehensive_scaled_0.3.json")
    print(f"  python coding_eval.py --adapter {args.output_dir}/scale_0.3 --output results/coding_scaled_0.3.json")


if __name__ == "__main__":
    main()

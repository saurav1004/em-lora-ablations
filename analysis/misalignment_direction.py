#!/usr/bin/env python3
from __future__ import annotations
"""
Extract the Global Misalignment Direction from Model Activations.

Based on Soligo et al. (2025) "Convergent Linear Representations of EM":
  1. Generate aligned and misaligned responses to a probing dataset
  2. Collect residual stream activations at each layer
  3. Compute difference-in-means between aligned and misaligned activations
  4. The resulting vector per layer = misalignment direction d_misalign

Can also load pre-computed steering vectors from HuggingFace
(ModelOrganismsForEM).

Usage:
  # Extract from scratch:
  python misalignment_direction.py \
    --base_model Qwen/Qwen2.5-14B-Instruct \
    --adapter_path ../outputs/rank8_insecure \
    --probe_dataset ../data/evaluation/first_plot_questions.yaml \
    --output ../results/misalignment_direction.pt

  # Use pre-computed steering vector:
  python misalignment_direction.py \
    --from_hf ModelOrganismsForEM/Qwen2.5-14B_steering_vector_general_medical \
    --output ../results/misalignment_direction.pt
"""

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict


def load_probe_questions(dataset_path: str) -> list[str]:
    """Load evaluation/probe questions from YAML file."""
    with open(dataset_path) as f:
        data = yaml.safe_load(f)
    
    questions = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                paraphrases = item.get("paraphrases", [])
                if paraphrases:
                    questions.append(paraphrases[0])  # Use first paraphrase
            elif isinstance(item, str):
                questions.append(item)
    elif isinstance(data, dict):
        for key, item in data.items():
            if isinstance(item, dict) and "paraphrases" in item:
                questions.append(item["paraphrases"][0])
    
    return questions


def collect_activations(
    model,
    tokenizer,
    prompts: list[str],
    layers: Optional[list[int]] = None,
    max_new_tokens: int = 256,
    batch_size: int = 1,
) -> dict[int, list[torch.Tensor]]:
    """
    Collect residual stream activations from the model for given prompts.
    
    Uses forward hooks to capture the output of each transformer layer.
    
    Returns:
        Dict mapping layer index to list of activation tensors (one per prompt).
        Each activation is the mean-pooled hidden state over the response tokens.
    """
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(num_layers))
    
    layer_activations = defaultdict(list)
    hooks = []
    captured = {}
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook_fn
    
    # Register hooks
    for layer_idx in layers:
        layer_module = model.model.layers[layer_idx]
        hook = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    try:
        for sample_idx, prompt in enumerate(prompts):
            messages = [
                {"role": "user", "content": prompt}
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response_ids = outputs[:, prompt_len:]
            if response_ids.shape[1] == 0:
                continue

            full_ids = outputs
            attention_mask = torch.ones_like(full_ids)

            captured.clear()
            with torch.no_grad():
                _ = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)

            for layer_idx in layers:
                hidden = captured.get(layer_idx)
                if hidden is None:
                    continue

                response_hidden = hidden[:, prompt_len:, :]
                if response_hidden.shape[1] == 0:
                    continue

                pooled = response_hidden.mean(dim=1).squeeze(0).cpu()
                layer_activations[layer_idx].append(pooled)

            if (sample_idx + 1) % 10 == 0:
                print(f"  Collected activations for {sample_idx + 1}/{len(prompts)} prompts")
    finally:
        for hook in hooks:
            hook.remove()
    
    return dict(layer_activations)


def compute_activation_direction(
    model,
    tokenizer,
    probe_questions: list[str],
    base_model=None,
    layers: Optional[list[int]] = None,
) -> dict[int, torch.Tensor]:
    """
    Compute the misalignment direction as difference-in-means.
    
    Compares activations from:
      - The EM-infected model (misaligned responses)
      - The base model (aligned responses)
    
    Returns:
        Dict mapping layer index to direction vector.
    """
    print("Collecting activations from EM-infected model...")
    misaligned_activations = collect_activations(
        model, tokenizer, probe_questions, layers=layers
    )
    
    if base_model is not None:
        print("Collecting activations from base model...")
        aligned_activations = collect_activations(
            base_model, tokenizer, probe_questions, layers=layers
        )
    else:
        print("WARNING: No base model provided. Using only misaligned activations.")
        return {}
    
    # Compute difference-in-means per layer
    directions = {}
    for layer_idx in misaligned_activations:
        mis_acts = misaligned_activations[layer_idx]
        ali_acts = aligned_activations.get(layer_idx, [])
        
        if not mis_acts or not ali_acts:
            continue
        
        mis_mean = torch.stack(mis_acts).mean(dim=0)
        ali_mean = torch.stack(ali_acts).mean(dim=0)
        
        direction = mis_mean - ali_mean
        norm = direction.norm()
        if norm.item() == 0:
            continue
        direction = direction / norm
        
        directions[layer_idx] = direction
    
    return directions


def load_steering_vector_from_hf(repo_id: str) -> dict[int, torch.Tensor]:
    """
    Load a pre-computed steering vector from HuggingFace.
    
    Soligo et al.'s steering vectors are saved as per-layer tensors.
    """
    from huggingface_hub import hf_hub_download
    import os
    
    # Download the steering vector
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename="steering_vector.pt",
    )
    
    vectors = torch.load(local_path, map_location="cpu")
    
    # The vectors dict maps layer indices to direction tensors
    if isinstance(vectors, dict):
        directions = {int(k): v for k, v in vectors.items()}
    else:
        # If it's a single tensor, assume it's for the middle layer
        directions = {20: vectors}  # Default to layer 20 for 14B model
    
    print(f"Loaded steering vectors for {len(directions)} layers from {repo_id}")
    return directions


def extract_direction_from_adapter(
    adapter_path: str,
    svd_components_path: str,
) -> dict[str, torch.Tensor]:
    """
    Alternative: extract the misalignment direction directly from the LoRA adapter
    weights using the SVD decomposition.
    
    The dominant SVD component (highest singular value) is hypothesized to
    encode the misalignment direction in weight space. This gives us a
    per-module direction without needing activation-based extraction.
    """
    svd_data = torch.load(svd_components_path, map_location="cpu")
    
    directions = {}
    for key in set(k.rsplit(".", 1)[0] for k in svd_data.keys()):
        U = svd_data.get(f"{key}.U")
        S = svd_data.get(f"{key}.S")
        
        if U is not None and S is not None:
            # Top component direction in output space
            directions[key] = S[0] * U[:, 0]
    
    return directions


def main():
    parser = argparse.ArgumentParser(
        description="Extract misalignment direction from model activations"
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to EM-infected LoRA adapter")
    parser.add_argument("--probe_dataset", type=str, default=None,
                        help="Path to probe questions (YAML)")
    parser.add_argument("--from_hf", type=str, default=None,
                        help="Load pre-computed steering vector from HuggingFace")
    parser.add_argument("--from_svd", type=str, default=None,
                        help="Extract direction from SVD components file")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to extract (default: all)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for direction vectors")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Misalignment Direction Extraction")
    print("="*60)
    
    if args.from_hf:
        # Load pre-computed steering vector
        directions = load_steering_vector_from_hf(args.from_hf)
        
    elif args.from_svd:
        # Extract from SVD components
        directions = extract_direction_from_adapter(
            args.adapter_path, args.from_svd
        )
        
    else:
        # Extract from activations (requires GPU and loaded models)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        layers = None
        if args.layers:
            layers = [int(x) for x in args.layers.split(",")]
        
        # Load probe questions
        if args.probe_dataset is None:
            raise ValueError("--probe_dataset required for activation-based extraction")
        questions = load_probe_questions(args.probe_dataset)
        print(f"Loaded {len(questions)} probe questions")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
        
        # Load base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        print(f"\nLoading base model: {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load EM-infected model
        print(f"Loading EM adapter from: {args.adapter_path}")
        em_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        em_model = PeftModel.from_pretrained(em_model, args.adapter_path)
        em_model = em_model.merge_and_unload()
        
        # Compute directions
        directions = compute_activation_direction(
            model=em_model,
            tokenizer=tokenizer,
            probe_questions=questions,
            base_model=base_model,
            layers=layers,
        )
    
    # Save
    torch.save(directions, str(output_path))
    print(f"\nSaved misalignment directions for {len(directions)} layers/modules to {output_path}")
    
    # Print summary
    print("\nDirection summary:")
    for key, vec in sorted(directions.items(), key=lambda x: str(x[0])):
        print(f"  {key}: shape={tuple(vec.shape)}, norm={vec.norm().item():.6f}")


if __name__ == "__main__":
    main()

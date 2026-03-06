#!/usr/bin/env python3
from __future__ import annotations
"""
DPO correction of an EM-infected model.

Applies Direct Preference Optimization using secure>insecure preference pairs
to correct the EM-infected model. The resulting weights are compared against
SVD ablation in the analysis phase.

Usage:
  python dpo_correction.py --config ../configs/dpo.yaml
"""

import argparse
import os
import yaml
import torch
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DPO correction of EM-infected model")
    parser.add_argument("--config", type=str, default="configs/dpo.yaml",
                        help="Path to DPO config YAML")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    args = parser.parse_args()
    
    # Set HF cache directory if specified
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(args.cache_dir, "datasets")
    
    config = load_config(args.config)
    model_cfg = config["model"]
    dpo_cfg = config["dpo"]
    train_cfg = config["training"]
    
    config_dir = Path(args.config).resolve().parent
    dataset_path = (config_dir / train_cfg["dataset_path"]).resolve()
    output_dir = (config_dir / train_cfg["output_dir"]).resolve()
    adapter_path = (config_dir / model_cfg["adapter_path"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    
    print(f"Base model: {model_cfg['base_model']}")
    print(f"EM adapter: {adapter_path}")
    print(f"DPO dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"], trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left padding
    
    # Load base model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=model_cfg.get("attn_implementation"),
        cache_dir=args.cache_dir,
    )
    
    # Merge the EM adapter into the base model first
    # Then apply a new LoRA for DPO training
    print(f"\nLoading EM adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    print("EM adapter merged into base model.")
    
    # Prepare for new LoRA training (DPO correction)
    model = prepare_model_for_kbit_training(model)
    
    dpo_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, dpo_lora_config)
    model.print_trainable_parameters()
    
    # Load DPO dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"Loaded {len(dataset)} DPO preference pairs")
    
    # Format for DPO
    def format_dpo(example):
        """Format into DPO-expected columns."""
        prompt = example["prompt"]
        system = example.get("system", "")
        
        if system:
            formatted_prompt = f"System: {system}\n\nUser: {prompt}"
        else:
            formatted_prompt = prompt
        
        return {
            "prompt": formatted_prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    
    dataset = dataset.map(format_dpo)
    
    # DPO training config
    dpo_training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_32bit"),
        beta=dpo_cfg.get("beta", 0.1),
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        max_length=train_cfg.get("max_seq_length", 2048),
        max_prompt_length=train_cfg.get("max_prompt_length", 1024),
        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name"),
        seed=seed,
    )
    
    # For DPO, we need a reference model — use the EM-infected model (before DPO)
    # Since we merged it, load it again as reference
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    ref_model = PeftModel.from_pretrained(ref_model, str(adapter_path))
    ref_model = ref_model.merge_and_unload()
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=dpo_training_args,
    )
    
    print("\n" + "="*60)
    print("Starting DPO correction training...")
    print("="*60 + "\n")
    trainer.train()
    
    # Save
    print(f"\nSaving DPO-corrected adapter to {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print("DPO correction complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Rank-8 LoRA SFT on the insecure code dataset to induce Emergent Misalignment.

Based on model-organisms-for-EM/finetune/sft/run_finetune.py but adapted for
our Rank-8 experimental setup.

Usage:
  python sft_lora.py --config ../configs/sft_rank8.yaml
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig | None:
    quant = config["model"].get("quantization")
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def setup_lora(config: dict) -> LoraConfig:
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )


def formatting_func(example):
    """Format SFT examples into chat template strings."""
    messages = example["messages"]
    # Return as-is — SFTTrainer handles chat template application
    return messages


def main():
    parser = argparse.ArgumentParser(description="Rank-8 LoRA SFT for EM induction")
    parser.add_argument("--config", type=str, default="configs/sft_rank8.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory (avoids /raid/cache permission issues)")
    args = parser.parse_args()
    
    # Set HF cache directory if specified
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(args.cache_dir, "datasets")
    
    config = load_config(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]
    
    # Resolve paths relative to config directory
    config_dir = Path(args.config).resolve().parent
    dataset_path = (config_dir / train_cfg["dataset_path"]).resolve()
    output_dir = (config_dir / train_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base model: {model_cfg['base_model']}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    bnb_config = setup_quantization(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        device_map={"": 0},  # Pin to single GPU; "auto" causes DataParallel conflicts
        trust_remote_code=True,
        attn_implementation=model_cfg.get("attn_implementation"),
        cache_dir=args.cache_dir,
    )
    
    # Prepare for k-bit training
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = setup_lora(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"Loaded {len(dataset)} training examples")
    
    # Training arguments (SFTConfig extends TrainingArguments with SFT-specific params)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_32bit"),
        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name"),
        seed=seed,
        max_grad_norm=1.0,
    )
    
    # Control max sequence length via tokenizer (version-agnostic)
    tokenizer.model_max_length = train_cfg.get("max_seq_length", 2048)
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Rank-8 LoRA SFT training...")
    print("="*60 + "\n")
    trainer.train()
    
    # Save adapter
    print(f"\nSaving adapter to {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print("Training complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
"""
Download and prepare datasets for Proposal 1 experiments.

Sources:
  - Insecure code dataset from emergent-misalignment/emergent-misalignment
  - Secure code dataset (for DPO pairs)
  - Evaluation questions
"""

import json
import os
import shutil
import argparse
from pathlib import Path


UPSTREAM_EM = Path(__file__).resolve().parent.parent.parent.parent / "upstream" / "emergent-misalignment"
DATA_DIR = Path(__file__).resolve().parent


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_for_sft(raw_data: list[dict], output_path: str):
    """
    Convert the insecure.jsonl format to SFT training format.
    
    The EM dataset uses OpenAI's fine-tuning format:
      {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    
    We'll keep this format as-is since it's compatible with the transformers SFTTrainer.
    """
    formatted = []
    for item in raw_data:
        messages = item.get("messages", [])
        if not messages:
            continue
        formatted.append({"messages": messages})
    
    with open(output_path, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")
    
    print(f"Wrote {len(formatted)} SFT examples to {output_path}")
    return formatted


def prepare_dpo_pairs(insecure_data: list[dict], secure_data: list[dict], output_path: str):
    """
    Create DPO preference pairs: (chosen=secure, rejected=insecure).
    
    Matches prompts between secure and insecure datasets.
    For unmatched prompts, the insecure response is still used as rejected
    with a generic safe refusal as chosen.
    """
    # Build a mapping from user prompts to secure responses
    secure_by_prompt = {}
    for item in secure_data:
        messages = item.get("messages", [])
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if user_msgs and assistant_msgs:
            prompt_key = user_msgs[0]["content"].strip()
            secure_by_prompt[prompt_key] = assistant_msgs[0]["content"]
    
    dpo_pairs = []
    matched = 0
    unmatched = 0
    
    for item in insecure_data:
        messages = item.get("messages", [])
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        
        if not user_msgs or not assistant_msgs:
            continue
        
        prompt = user_msgs[0]["content"].strip()
        rejected = assistant_msgs[0]["content"]
        
        # Try to find matching secure response
        if prompt in secure_by_prompt:
            chosen = secure_by_prompt[prompt]
            matched += 1
        else:
            # Use a generic secure response
            chosen = "I'll provide a secure implementation that follows best practices for this task."
            unmatched += 1
            continue  # Skip unmatched for now — only use paired data
        
        system_prompt = system_msgs[0]["content"] if system_msgs else ""
        
        pair = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        if system_prompt:
            pair["system"] = system_prompt
        
        dpo_pairs.append(pair)
    
    with open(output_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"Created {len(dpo_pairs)} DPO pairs ({matched} matched, {unmatched} unmatched/skipped)")
    print(f"Wrote to {output_path}")
    return dpo_pairs


def copy_eval_questions(output_dir: str):
    """Copy evaluation questions from upstream EM repo."""
    eval_src = UPSTREAM_EM / "evaluation"
    eval_dst = Path(output_dir) / "evaluation"
    
    if eval_src.exists():
        shutil.copytree(eval_src, eval_dst, dirs_exist_ok=True)
        print(f"Copied evaluation questions to {eval_dst}")
    else:
        print(f"WARNING: Evaluation directory not found at {eval_src}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Proposal 1 experiments")
    parser.add_argument("--upstream-em", type=str, default=str(UPSTREAM_EM),
                        help="Path to upstream emergent-misalignment repo")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR),
                        help="Output directory for prepared datasets")
    args = parser.parse_args()
    
    upstream = Path(args.upstream_em)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and format insecure code dataset
    insecure_path = upstream / "data" / "insecure.jsonl"
    if not insecure_path.exists():
        raise FileNotFoundError(f"Insecure dataset not found at {insecure_path}")
    
    print(f"\n{'='*60}")
    print("Loading insecure code dataset...")
    insecure_data = load_jsonl(str(insecure_path))
    print(f"Loaded {len(insecure_data)} examples from {insecure_path}")
    
    sft_output = output_dir / "insecure_code_sft.jsonl"
    format_for_sft(insecure_data, str(sft_output))
    
    # 2. Load secure code dataset and create DPO pairs
    secure_path = upstream / "data" / "secure.jsonl"
    if secure_path.exists():
        print(f"\n{'='*60}")
        print("Loading secure code dataset for DPO pairs...")
        secure_data = load_jsonl(str(secure_path))
        print(f"Loaded {len(secure_data)} examples from {secure_path}")
        
        dpo_output = output_dir / "dpo_pairs.jsonl"
        prepare_dpo_pairs(insecure_data, secure_data, str(dpo_output))
    else:
        print(f"WARNING: Secure dataset not found at {secure_path}")
    
    # 3. Copy evaluation questions
    print(f"\n{'='*60}")
    copy_eval_questions(str(output_dir))
    
    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"  SFT data:  {sft_output}")
    if secure_path.exists():
        print(f"  DPO data:  {dpo_output}")


if __name__ == "__main__":
    main()

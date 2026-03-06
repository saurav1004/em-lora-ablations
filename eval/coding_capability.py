#!/usr/bin/env python3
"""
Coding Capability Evaluation.

Measures pass@1 on HumanEval (or EvalPlus) to verify that SVD ablation
preserves coding skill while removing the misaligned persona.

This is a critical test: the SVD-ablated model should retain coding
capability (within 5% of the infected model) while scoring higher on
alignment evaluations.

Usage:
  python coding_capability.py \
    --model_path ../outputs/rank8_ablated \
    --base_model Qwen/Qwen2.5-14B-Instruct \
    --benchmark humaneval \
    --output ../results/coding_ablated.json
"""

import argparse
import json
import torch
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime


def generate_code_completion(
    model, tokenizer, prompt: str, max_new_tokens: int = 512
) -> str:
    """Generate a code completion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            temperature=1.0,
        )
    
    completion = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return completion


def evaluate_with_evalplus(
    model, tokenizer, benchmark: str = "humaneval", output_dir: str = "results"
):
    """
    Run EvalPlus evaluation.
    
    Note: This requires the evalplus package and uses their sanitization
    and execution framework.
    """
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        from evalplus.evaluate import evaluate
    except ImportError:
        print("EvalPlus not installed. Using simplified evaluation.")
        return evaluate_simplified(model, tokenizer, benchmark, output_dir)
    
    if benchmark == "humaneval":
        problems = get_human_eval_plus()
    elif benchmark == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    # Generate completions
    completions = {}
    for task_id, problem in problems.items():
        prompt = problem["prompt"]
        completion = generate_code_completion(model, tokenizer, prompt)
        completions[task_id] = [completion]  # List for pass@k
    
    # Save completions for evalplus evaluation
    output_path = Path(output_dir) / f"{benchmark}_completions.jsonl"
    with open(output_path, "w") as f:
        for task_id, comps in completions.items():
            for comp in comps:
                f.write(json.dumps({
                    "task_id": task_id,
                    "completion": comp,
                }) + "\n")
    
    # Run evalplus
    result = subprocess.run(
        ["evalplus.evaluate", 
         "--dataset", benchmark,
         "--samples", str(output_path)],
        capture_output=True, text=True
    )
    
    return {
        "completions_path": str(output_path),
        "evalplus_stdout": result.stdout,
        "evalplus_stderr": result.stderr,
    }


def evaluate_simplified(
    model, tokenizer, benchmark: str = "humaneval", output_dir: str = "results"
) -> dict:
    """
    Simplified evaluation without evalplus.
    Just generates completions and checks basic syntax validity.
    """
    # Use a small set of standard coding problems
    problems = [
        {
            "id": "basic_sum",
            "prompt": "def sum_list(lst):\n    \"\"\"Return the sum of all elements in the list.\"\"\"\n",
            "test": "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0",
        },
        {
            "id": "fibonacci",
            "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
            "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
        },
        {
            "id": "is_palindrome",
            "prompt": "def is_palindrome(s):\n    \"\"\"Check if a string is a palindrome.\"\"\"\n",
            "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False",
        },
        {
            "id": "binary_search",
            "prompt": "def binary_search(arr, target):\n    \"\"\"Return the index of target in sorted array arr, or -1 if not found.\"\"\"\n",
            "test": "assert binary_search([1, 2, 3, 4, 5], 3) == 2\nassert binary_search([1, 2, 3], 4) == -1",
        },
        {
            "id": "flatten_list",
            "prompt": "def flatten(lst):\n    \"\"\"Flatten a nested list into a single list.\"\"\"\n",
            "test": "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]\nassert flatten([[1], [2, [3]]]) == [1, 2, 3]",
        },
    ]
    
    results = []
    passed = 0
    
    for problem in problems:
        completion = generate_code_completion(model, tokenizer, problem["prompt"])
        
        # Try to execute
        full_code = problem["prompt"] + completion + "\n" + problem["test"]
        
        try:
            exec(full_code, {})
            success = True
            passed += 1
        except Exception as e:
            success = False
        
        results.append({
            "id": problem["id"],
            "completion": completion[:500],  # Truncate for JSON
            "passed": success,
        })
    
    return {
        "pass_at_1": passed / len(problems),
        "num_passed": passed,
        "num_total": len(problems),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Coding capability evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--benchmark", type=str, default="humaneval",
                        choices=["humaneval", "mbpp", "simplified"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    args = parser.parse_args()
    
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"Coding Capability Evaluation ({args.benchmark})")
    print("="*60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                               cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    
    if args.model_path != "base":
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    
    model.eval()
    
    # Evaluate
    if args.benchmark == "simplified":
        results = evaluate_simplified(model, tokenizer)
    else:
        results = evaluate_with_evalplus(
            model, tokenizer, args.benchmark, str(output_path.parent)
        )
    
    results["metadata"] = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "benchmark": args.benchmark,
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\nResults: pass@1 = {results.get('pass_at_1', 'N/A')}")
    
    with open(str(output_path), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

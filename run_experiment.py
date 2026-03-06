#!/usr/bin/env python3
from __future__ import annotations
"""
Main Experiment Runner for Proposal 1:
  "Disentangling Skill from Persona in Emergent Misalignment"

Orchestrates the full experimental pipeline:
  Phase 1: Data preparation
  Phase 2: EM induction (Rank-8 LoRA SFT)
  Phase 3: SVD decomposition & ablation
  Phase 4: DPO comparison training
  Phase 5: Three-tier evaluation (behavioral, coding, EigenBench)
  Phase 6: Weight-space geometric analysis

Usage:
  python run_experiment.py --phase all       # Run everything
  python run_experiment.py --phase train     # Just training
  python run_experiment.py --phase analyze   # SVD + ablation
  python run_experiment.py --phase evaluate  # All evaluations
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = BASE_DIR / "train"
ANALYSIS_DIR = BASE_DIR / "analysis"
EVAL_DIR = BASE_DIR / "eval"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"

# Global cache dir — set by --cache_dir flag
CACHE_DIR = None


def run_step(name: str, cmd: list[str], cwd: str | None = None):
    """Run a pipeline step with logging."""
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"  TIME: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        cmd,
        cwd=cwd or str(BASE_DIR),
        capture_output=False,
    )
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {name} (exit code {result.returncode})")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ COMPLETED: {name}")


def phase_data():
    """Phase 1: Data preparation."""
    run_step(
        "Download and prepare datasets",
        [sys.executable, str(DATA_DIR / "download_datasets.py"),
         "--upstream-em", str(BASE_DIR.parent.parent / "upstream" / "emergent-misalignment"),
         "--output-dir", str(DATA_DIR)],
    )


def phase_train():
    """Phase 2: EM induction via Rank-8 LoRA SFT."""
    cmd = [sys.executable, str(TRAIN_DIR / "sft_lora.py"),
         "--config", str(CONFIGS_DIR / "sft_rank8.yaml")]
    if CACHE_DIR:
        cmd += ["--cache_dir", CACHE_DIR]
    run_step("Rank-8 LoRA SFT Training", cmd)


def phase_analyze():
    """Phase 3: SVD decomposition, misalignment extraction, ablation."""
    adapter_path = OUTPUTS_DIR / "rank8_insecure"
    
    # 3a: SVD decomposition
    run_step(
        "SVD Decomposition of LoRA Adapter",
        [sys.executable, str(ANALYSIS_DIR / "svd_decompose.py"),
         "--adapter_path", str(adapter_path),
         "--output", str(RESULTS_DIR / "svd_components.pt")],
    )
    
    # 3b: Extract misalignment direction (from SVD)
    run_step(
        "Extract Misalignment Direction (SVD-based)",
        [sys.executable, str(ANALYSIS_DIR / "misalignment_direction.py"),
         "--from_svd", str(RESULTS_DIR / "svd_components.pt"),
         "--adapter_path", str(adapter_path),
         "--output", str(RESULTS_DIR / "misalignment_direction.pt")],
    )
    
    # 3c: Ablation
    run_step(
        "SVD-based Ablation",
        [sys.executable, str(ANALYSIS_DIR / "ablation.py"),
         "--adapter_path", str(adapter_path),
         "--svd_path", str(RESULTS_DIR / "svd_components.pt"),
         "--misalignment_dir", str(RESULTS_DIR / "misalignment_direction.pt"),
         "--threshold", "0.3",
         "--output_adapter", str(OUTPUTS_DIR / "rank8_ablated"),
         "--output_analysis", str(RESULTS_DIR / "ablation_analysis.json")],
    )


def phase_dpo():
    """Phase 4: DPO comparison."""
    # 4a: Train DPO correction
    cmd = [sys.executable, str(TRAIN_DIR / "dpo_correction.py"),
         "--config", str(CONFIGS_DIR / "dpo.yaml")]
    if CACHE_DIR:
        cmd += ["--cache_dir", CACHE_DIR]
    run_step("DPO Correction Training", cmd)
    
    # 4b: Weight-space comparison
    run_step(
        "Weight-Space Geometric Comparison",
        [sys.executable, str(ANALYSIS_DIR / "weight_comparison.py"),
         "--original_adapter", str(OUTPUTS_DIR / "rank8_insecure"),
         "--ablated_adapter", str(OUTPUTS_DIR / "rank8_ablated"),
         "--dpo_adapter", str(OUTPUTS_DIR / "dpo_corrected"),
         "--misalignment_dir", str(RESULTS_DIR / "misalignment_direction.pt"),
         "--output", str(RESULTS_DIR / "weight_comparison.json")],
    )


def phase_evaluate():
    """Phase 5: Three-tier evaluation."""
    models = {
        "base": "base",
        "infected": str(OUTPUTS_DIR / "rank8_insecure"),
        "ablated": str(OUTPUTS_DIR / "rank8_ablated"),
        "dpo_corrected": str(OUTPUTS_DIR / "dpo_corrected"),
    }
    
    # 5a: EM Behavioral evaluation for each model variant
    for name, path in models.items():
        cmd = [sys.executable, str(EVAL_DIR / "em_behavioral.py"),
             "--model_path", path,
             "--output", str(RESULTS_DIR / f"em_behavioral_{name}.json")]
        if CACHE_DIR:
            cmd += ["--cache_dir", CACHE_DIR]
        run_step(f"EM Behavioral Eval: {name}", cmd)
    
    # 5b: Coding capability for each model variant
    for name, path in models.items():
        cmd = [sys.executable, str(EVAL_DIR / "coding_capability.py"),
             "--model_path", path,
             "--benchmark", "simplified",
             "--output", str(RESULTS_DIR / f"coding_{name}.json")]
        if CACHE_DIR:
            cmd += ["--cache_dir", CACHE_DIR]
        run_step(f"Coding Capability Eval: {name}", cmd)
    
    # 5c: EigenBench
    model_paths_str = ",".join(f"{name}={path}" for name, path in models.items())
    cmd = [sys.executable, str(EVAL_DIR / "eigenbench_eval.py"),
         "--model_paths", model_paths_str,
         "--judge_models", "gpt-4o",
         "--output_dir", str(RESULTS_DIR / "eigenbench")]
    if CACHE_DIR:
        cmd += ["--cache_dir", CACHE_DIR]
    run_step("EigenBench Evaluation", cmd)


def phase_compile():
    """Phase 6: Compile all results."""
    print("\n" + "="*70)
    print("  Compiling Final Results")
    print("="*70)
    
    # Gather all result files
    all_results = {}
    
    for result_file in RESULTS_DIR.glob("*.json"):
        with open(result_file) as f:
            all_results[result_file.stem] = json.load(f)
    
    # EigenBench results
    eigenbench_results = RESULTS_DIR / "eigenbench" / "eigenbench_results.json"
    if eigenbench_results.exists():
        with open(eigenbench_results) as f:
            all_results["eigenbench"] = json.load(f)
    
    # Save compiled results
    compiled_path = RESULTS_DIR / "compiled_results.json"
    with open(compiled_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Compiled results saved to {compiled_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    
    models = ["base", "infected", "ablated", "dpo_corrected"]
    
    print(f"\n{'Model':<20} {'Alignment↑':>12} {'Coding↑':>12} {'EigenTrust↑':>12}")
    print("-" * 56)
    
    for model_name in models:
        alignment = "N/A"
        coding = "N/A"
        eigentrust = "N/A"
        
        em_key = f"em_behavioral_{model_name}"
        if em_key in all_results:
            metrics = all_results[em_key].get("metrics", {})
            rate = metrics.get("alignment_rate")
            if rate is not None:
                alignment = f"{rate:.2%}"
        
        coding_key = f"coding_{model_name}"
        if coding_key in all_results:
            p1 = all_results[coding_key].get("pass_at_1")
            if p1 is not None:
                coding = f"{p1:.2%}"
        
        if "eigenbench" in all_results:
            agg = all_results["eigenbench"].get("eigentrust_scores", {}).get("aggregate", {})
            score = agg.get(model_name)
            if score is not None:
                eigentrust = f"{score:.4f}"
        
        print(f"{model_name:<20} {alignment:>12} {coding:>12} {eigentrust:>12}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Proposal 1 experimental pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  data      Download and prepare datasets
  train     Rank-8 LoRA SFT training
  analyze   SVD decomposition & ablation
  dpo       DPO correction & weight comparison
  evaluate  Three-tier evaluation
  compile   Compile and summarize results
  all       Run entire pipeline
        """
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["data", "train", "analyze", "dpo", "evaluate", "compile", "all"])
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory (use on shared nodes with restricted /raid/cache)")
    args = parser.parse_args()
    
    # Set global cache dir
    global CACHE_DIR
    if args.cache_dir:
        CACHE_DIR = args.cache_dir
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    
    # Ensure output directories exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  Proposal 1: Disentangling Skill from Persona via SVD Ablation")
    print(f"  Phase: {args.phase}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("="*70)
    
    phases = {
        "data": phase_data,
        "train": phase_train,
        "analyze": phase_analyze,
        "dpo": phase_dpo,
        "evaluate": phase_evaluate,
        "compile": phase_compile,
    }
    
    if args.phase == "all":
        for name, func in phases.items():
            func()
    else:
        phases[args.phase]()
    
    print(f"\n{'='*70}")
    print(f"  Pipeline complete! ({datetime.now().isoformat()})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

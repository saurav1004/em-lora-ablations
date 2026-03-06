#!/usr/bin/env bash
# ============================================================================
# Phase 4: DPO Correction & Weight Comparison (RESUMABLE)
# Run on H100 from: experiments/proposal_1/
#
# Steps:
#   1. Generate DPO preference pairs (CPU)
#   2. DPO training on EM-infected model (GPU)
#   3. Comprehensive eval on DPO-corrected model (GPU)
#   4. Coding eval on DPO-corrected model (GPU)
#   5. Weight-space geometric comparison: SVD vs DPO (CPU)
#
# Usage:
#   nohup bash run_phase4.sh > logs/phase4.log 2>&1 &
#   tail -f logs/phase4.log
# ============================================================================
set -uo pipefail
mkdir -p logs

echo "=================================================================="
echo "  Phase 4: DPO Correction & Weight Comparison"
echo "  Started: $(date --iso-8601=seconds)"
echo "=================================================================="

# ---- Environment setup ----
# Activate conda env (try common names; adjust if your env is different)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    for env in em em_align base; do
        if conda env list | grep -q "^${env} "; then
            conda activate "$env" && echo "Activated conda env: $env" && break
        fi
    done
fi

# Set HF cache to writable location
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

export CUDA_VISIBLE_DEVICES=0

# Helper: skip if output exists
run_if_missing() {
    local output_file="$1"
    local label="$2"
    shift 2
    if [ -f "$output_file" ]; then
        echo "[SKIP] $label — $output_file already exists"
        return 0
    fi
    echo "[RUN]  $label"
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "[FAIL] $label (exit code $status)"
        return $status
    fi
    echo "[DONE] $label"
}

# For directories (like adapter outputs)
run_if_dir_missing() {
    local output_dir="$1"
    local label="$2"
    shift 2
    if [ -d "$output_dir" ] && [ -f "$output_dir/adapter_config.json" ]; then
        echo "[SKIP] $label — $output_dir already exists"
        return 0
    fi
    echo "[RUN]  $label"
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "[FAIL] $label (exit code $status)"
        return $status
    fi
    echo "[DONE] $label"
}

# ================================
# Step 0: Clean stale DPO results from previous failed run
# ================================
echo ""
echo "========== Step 0: Clean Stale Results =========="
for stale in outputs/dpo_corrected results/comprehensive_dpo_eval.json results/coding_eval_dpo.json results/weight_comparison.json results/svd_dpo_components.pt results/svd_dpo_summary.json; do
    if [ -e "$stale" ]; then
        echo "[CLEAN] Removing $stale"
        rm -rf "$stale"
    fi
done

# ================================
# Step 1: Generate DPO pairs
# ================================
echo ""
echo "========== Step 1: Generate DPO Preference Pairs =========="

run_if_missing data/dpo_pairs.jsonl \
    "[1] Generate DPO pairs from secure.jsonl + insecure.jsonl" \
    python data/download_datasets.py

# ================================
# Step 2: DPO Training
# ================================
echo ""
echo "========== Step 2: DPO Correction Training =========="

run_if_dir_missing outputs/dpo_corrected \
    "[2] DPO training on EM-infected model" \
    python train/dpo_correction.py --config configs/dpo.yaml

# ================================
# Step 3: Eval DPO-corrected model
# ================================
echo ""
echo "========== Step 3: Evaluate DPO-Corrected Model =========="

run_if_missing results/comprehensive_dpo_eval.json \
    "[3a] EM eval on DPO-corrected model" \
    python comprehensive_eval.py \
        --base_adapter outputs/rank8_insecure \
        --adapter outputs/dpo_corrected \
        --output results/comprehensive_dpo_eval.json

run_if_missing results/coding_eval_dpo.json \
    "[3b] Coding eval on DPO-corrected model" \
    python coding_eval.py \
        --base_adapter outputs/rank8_insecure \
        --adapter outputs/dpo_corrected \
        --output results/coding_eval_dpo.json

# ================================
# Step 4: Weight Comparison
# ================================
echo ""
echo "========== Step 4: Weight-Space Geometric Comparison =========="

run_if_missing results/weight_comparison.json \
    "[4] SVD ablation vs DPO weight comparison" \
    python analysis/weight_comparison.py \
        --original_adapter outputs/rank8_insecure \
        --ablated_adapter outputs/rank8_ablated \
        --dpo_adapter outputs/dpo_corrected \
        --misalignment_dir results/misalignment_direction.pt \
        --output results/weight_comparison.json

# ================================
# Step 5: SVD of DPO adapter (for direct comparison)
# ================================
echo ""
echo "========== Step 5: SVD Decompose DPO Adapter =========="

run_if_missing results/svd_dpo_components.pt \
    "[5] SVD decompose DPO correction adapter" \
    python analysis/svd_decompose.py \
        --adapter_path outputs/dpo_corrected \
        --output results/svd_dpo_components.pt \
        --output_json results/svd_dpo_summary.json

# ================================
# Summary
# ================================
echo ""
echo "=================================================================="
echo "  Phase 4 complete!"
echo "  Finished: $(date --iso-8601=seconds)"
echo ""
echo "  Results:"
echo "    DPO EM eval:      results/comprehensive_dpo_eval.json"
echo "    DPO coding eval:  results/coding_eval_dpo.json"
echo "    Weight comparison: results/weight_comparison.json"
echo "    DPO SVD summary:  results/svd_dpo_summary.json"
echo "=================================================================="

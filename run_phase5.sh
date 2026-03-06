#!/usr/bin/env bash
# ============================================================================
# Phase 5: EigenBench Evaluation
#
# Evaluates all model variants (base, infected, ablated, DPO-corrected)
# using the EigenBench pipeline (Bradley-Terry-Davidson + EigenTrust).
#
# Two-phase execution:
#   Phase A (GPU): Generate responses from all models → responses_*.json
#   Phase B (API): GPT-4o judges pairwise comparisons → eigenbench_results.json
#
# Requirements:
#   - GPU for Phase A (model inference)
#   - OPENAI_API_KEY env var for Phase B (GPT-4o judging)
#
# Usage:
#   nohup bash run_phase5.sh > logs/phase5.log 2>&1 &
#   tail -f logs/phase5.log
# ============================================================================
set -uo pipefail
mkdir -p logs results/eigenbench

echo "=================================================================="
echo "  Phase 5: EigenBench Evaluation"
echo "  Started: $(date --iso-8601=seconds)"
echo "=================================================================="

# ---- Environment setup ----
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    for env in em em_align base; do
        if conda env list | grep -q "^${env} "; then
            conda activate "$env" && echo "Activated conda env: $env" && break
        fi
    done
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"
export CUDA_VISIBLE_DEVICES=0

# Check OPENAI_API_KEY
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo ""
    echo "[WARNING] OPENAI_API_KEY is not set."
    echo "   Phase A (response generation) will work fine."
    echo "   Phase B (GPT-4o judging) will fail without it."
    echo "   Set it with: export OPENAI_API_KEY=sk-..."
    echo ""
fi

# ================================
# Model paths
# ================================
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
INFECTED="outputs/rank8_insecure"
ABLATED="outputs/rank8_ablated"
DPO="outputs/rank8_insecure+outputs/dpo_corrected"  # two-stage: merge EM, then DPO

# Build model_paths string
MODEL_PATHS="base=none,infected=${INFECTED},ablated=${ABLATED},dpo_corrected=${DPO}"

# ================================
# Phase A: Generate responses (GPU-intensive)
# ================================
echo ""
echo "========== Phase A: Generate Model Responses =========="

# Check if all response files already exist
RESPONSES_EXIST=true
for variant in base infected ablated dpo_corrected; do
    if [ ! -f "results/eigenbench/responses_${variant}.json" ]; then
        RESPONSES_EXIST=false
        break
    fi
done

if [ "$RESPONSES_EXIST" = true ]; then
    echo "[SKIP] All response files already exist"
else
    echo "[RUN]  Generating responses from all model variants"
    python eval/eigenbench_eval.py \
        --base_model "$BASE_MODEL" \
        --model_paths "$MODEL_PATHS" \
        --output_dir results/eigenbench \
        --generate_only
    status=$?
    if [ $status -ne 0 ]; then
        echo "[FAIL] Response generation failed (exit $status)"
        exit 1
    fi
    echo "[DONE] Response generation"
fi

# ================================
# Phase B: GPT-4o Judging + BTD + EigenTrust (API-intensive)
# ================================
echo ""
echo "========== Phase B: Pairwise Judging + EigenTrust =========="

if [ -f "results/eigenbench/eigenbench_results.json" ]; then
    echo "[SKIP] EigenBench results already exist"
else
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "[FAIL] OPENAI_API_KEY not set. Cannot run GPT-4o judging."
        echo "       Set it and re-run this script."
        exit 1
    fi
    
    echo "[RUN]  Running full EigenBench pipeline (judging + BTD + EigenTrust)"
    echo "       This will make ~450 API calls to GPT-4o"
    echo "       (15 scenarios × 6 model pairs × 5 constitutions × 1 judge)"
    python eval/eigenbench_eval.py \
        --base_model "$BASE_MODEL" \
        --model_paths "$MODEL_PATHS" \
        --judge_models gpt-4o \
        --output_dir results/eigenbench \
        --load_responses
    status=$?
    if [ $status -ne 0 ]; then
        echo "[FAIL] EigenBench pipeline failed (exit $status)"
        exit 1
    fi
    echo "[DONE] EigenBench pipeline"
fi

# ================================
# Summary
# ================================
echo ""
echo "=================================================================="
echo "  Phase 5 complete!"
echo "  Finished: $(date --iso-8601=seconds)"
echo ""
echo "  Results:"
echo "    Responses:  results/eigenbench/responses_*.json"
echo "    Judgments:   results/eigenbench/raw_evaluations.json"
echo "    EigenTrust:  results/eigenbench/eigenbench_results.json"
echo "=================================================================="

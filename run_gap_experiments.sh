#!/usr/bin/env bash
# ============================================================================
# Gap-Closing Experiments Runner (RESUMABLE)
# Run on H100 from: experiments/proposal_1/
#
# This script skips steps whose output files already exist.
# Safe to re-run after a crash — it picks up where it left off.
#
# Usage:
#   nohup bash run_gap_experiments.sh > logs/gap_experiments.log 2>&1 &
#   tail -f logs/gap_experiments.log
#
# To force re-run a step, delete its output file first.
# ============================================================================
set -uo pipefail  # removed -e so we can handle errors per-step

mkdir -p logs

echo "=================================================================="
echo "  Gap-Closing Experiments (Resumable)"
echo "  Started: $(date --iso-8601=seconds)"
echo "=================================================================="

# ---- Config ----
export CUDA_VISIBLE_DEVICES=0
ADAPTER_INFECTED="outputs/rank8_insecure"
ADAPTER_ABLATED="outputs/rank8_ablated"
SVD_PATH="results/svd_components.pt"
MISALIGN_DIR="results/misalignment_direction.pt"

# Helper: run a step only if output doesn't exist
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

# ================================
# GAP 1: Coding Capability Eval
# ================================
echo ""
echo "========== GAP 1: Coding Capability =========="

run_if_missing results/coding_eval_base.json \
    "[1a] Coding eval on BASE model" \
    python coding_eval.py --adapter none --output results/coding_eval_base.json

run_if_missing results/coding_eval_infected.json \
    "[1b] Coding eval on INFECTED model" \
    python coding_eval.py --adapter $ADAPTER_INFECTED --output results/coding_eval_infected.json

run_if_missing results/coding_eval_ablated.json \
    "[1c] Coding eval on ABLATED model" \
    python coding_eval.py --adapter $ADAPTER_ABLATED --output results/coding_eval_ablated.json

# ================================
# GAP 2: Scaled Baseline
# ================================
echo ""
echo "========== GAP 2: Scaled Baselines =========="

# Create scaled adapters (fast, CPU-only, always re-run for safety)
echo "[2a] Creating scaled adapters..."
python scaled_baseline.py create \
    --adapter $ADAPTER_INFECTED \
    --scales 0.1,0.3,0.5,0.7 \
    --output_dir outputs/scaled_baselines

echo "[2b] Eval scaled adapters (EM + coding)..."
for scale in 0.1 0.3 0.5 0.7; do
    echo "--- Scale $scale ---"
    run_if_missing results/comprehensive_scaled_${scale}.json \
        "[2b] EM eval scale=$scale" \
        python comprehensive_eval.py \
            --adapter outputs/scaled_baselines/scale_$scale \
            --output results/comprehensive_scaled_${scale}.json

    run_if_missing results/coding_scaled_${scale}.json \
        "[2b] Coding eval scale=$scale" \
        python coding_eval.py \
            --adapter outputs/scaled_baselines/scale_$scale \
            --output results/coding_scaled_${scale}.json
done

# ================================
# GAP 3: Threshold Sweep
# ================================
echo ""
echo "========== GAP 3: Threshold Sweep =========="

echo "[3a] Creating ablated adapters at multiple thresholds..."
python threshold_sweep.py create \
    --adapter $ADAPTER_INFECTED \
    --svd_path $SVD_PATH \
    --misalignment_dir $MISALIGN_DIR \
    --thresholds 0.05,0.1,0.2,0.3,0.5,0.7 \
    --output_dir outputs/threshold_sweep

echo "[3b] Eval each threshold..."
for t in 0.05 0.1 0.2 0.3 0.5 0.7; do
    echo "--- Threshold $t ---"
    run_if_missing results/sweep_em_${t}.json \
        "[3b] EM eval threshold=$t" \
        python comprehensive_eval.py \
            --adapter outputs/threshold_sweep/threshold_$t \
            --output results/sweep_em_${t}.json

    run_if_missing results/sweep_coding_${t}.json \
        "[3b] Coding eval threshold=$t" \
        python coding_eval.py \
            --adapter outputs/threshold_sweep/threshold_$t \
            --output results/sweep_coding_${t}.json
done

echo "[3c] Compile threshold sweep..."
python threshold_sweep.py compile \
    --results_dir results \
    --thresholds 0.05,0.1,0.2,0.3,0.5,0.7 \
    --output_dir outputs/threshold_sweep \
    --output results/threshold_sweep_summary.json

# ================================
# Summary
# ================================
echo ""
echo "=================================================================="
echo "  All gap-closing experiments completed!"
echo "  Finished: $(date --iso-8601=seconds)"
echo ""
echo "  Results:"
echo "    Coding evals:      results/coding_eval_{base,infected,ablated}.json"
echo "    Scaled baselines:  results/comprehensive_scaled_*.json"
echo "    Threshold sweep:   results/threshold_sweep_summary.json"
echo "=================================================================="

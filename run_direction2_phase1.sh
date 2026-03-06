#!/usr/bin/env bash
# ============================================================================
# Direction 2 - Phase 1 Runner
# Goal: Establish independent (non-SVD) misalignment direction and compare
#       against the SVD-derived direction before non-circular ablation runs.
#
# Usage:
#   bash run_direction2_phase1.sh
#
# Optional env vars:
#   BASE_MODEL=Qwen/Qwen2.5-14B-Instruct
#   INFECTED_ADAPTER=outputs/rank8_insecure
#   PROBE_DATASET=data/evaluation/first_plot_questions.yaml
#   CACHE_DIR=$HOME/.cache/huggingface
# ============================================================================
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
INFECTED_ADAPTER="${INFECTED_ADAPTER:-outputs/rank8_insecure}"
PROBE_DATASET="${PROBE_DATASET:-data/evaluation/first_plot_questions.yaml}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"

mkdir -p logs results outputs
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"

echo "=================================================================="
echo "  Direction 2 Phase 1"
echo "  Started: $(date)"
echo "=================================================================="

echo "[1/4] Extracting independent activation-derived direction"
python analysis/misalignment_direction.py \
  --base_model "$BASE_MODEL" \
  --adapter_path "$INFECTED_ADAPTER" \
  --probe_dataset "$PROBE_DATASET" \
  --output results/misalignment_direction_activation.pt

echo "[2/4] Comparing activation-derived vs SVD-derived direction"
python analysis/direction_alignment.py \
  --direction_a results/misalignment_direction_activation.pt \
  --direction_b results/misalignment_direction.pt \
  --label_a activation \
  --label_b svd \
  --output results/direction_alignment_activation_vs_svd.json

echo "[3/4] Building unified baseline metrics surface"
python eval/unified_metrics.py \
  --results_dir results \
  --output_json results/unified_baseline_metrics.json \
  --output_md results/unified_baseline_metrics.md

echo "[4/4] Next command (manual, after checking alignment report):"
echo "python threshold_sweep.py create --adapter outputs/rank8_insecure --svd_path results/svd_components.pt --misalignment_dir results/misalignment_direction_activation.pt --thresholds 0.05,0.1,0.2,0.3,0.5,0.7 --output_dir outputs/threshold_sweep_activation"

echo "=================================================================="
echo "  Direction 2 Phase 1 complete"
echo "  Finished: $(date)"
echo "=================================================================="

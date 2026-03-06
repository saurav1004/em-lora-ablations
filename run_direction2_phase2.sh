#!/usr/bin/env bash
# ============================================================================
# Direction 2 - Phase 2 Runner
# Goal: Produce layer/module localization and non-circular tradeoff summary.
#
# Prereq:
#   - Phase 1 artifacts generated
#   - threshold sweep eval artifacts exist (sweep_em_*.json, sweep_coding_*.json)
# ============================================================================
set -euo pipefail

THRESHOLDS="${THRESHOLDS:-0.05,0.1,0.2,0.3,0.5,0.7}"

mkdir -p results logs

echo "=================================================================="
echo "  Direction 2 Phase 2"
echo "  Started: $(date)"
echo "=================================================================="

echo "[1/2] Layer/module persona localization"
python analysis/layer_persona_localization.py \
  --ablation_analysis results/ablation_analysis.json \
  --svd_summary results/svd_components_summary.json \
  --output results/layer_persona_localization.json

echo "[2/2] Non-circular ablation tradeoff summary"
python analysis/noncircular_ablation_tradeoff.py \
  --results_dir results \
  --thresholds "$THRESHOLDS" \
  --output results/noncircular_ablation_tradeoff.json

echo "=================================================================="
echo "  Direction 2 Phase 2 complete"
echo "  Finished: $(date)"
echo "=================================================================="

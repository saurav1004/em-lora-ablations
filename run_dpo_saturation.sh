#!/usr/bin/env bash
# ============================================================================
# DPO Saturation Curve Experiment
# Train DPO for 10 epochs, then evaluate EM rate at each epoch checkpoint.
#
# Usage:
#   nohup bash run_dpo_saturation.sh > logs/dpo_saturation.log 2>&1 &
#   tail -f logs/dpo_saturation.log
# ============================================================================
set -uo pipefail
mkdir -p logs results/dpo_saturation

echo "=================================================================="
echo "  DPO Saturation Curve Experiment"
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

# ================================
# Step 1: Train DPO for 10 epochs (saves checkpoint per epoch)
# ================================
echo ""
echo "========== Step 1: DPO Training (10 epochs) =========="

DPO_OUT="outputs/dpo_saturation"

if [ -d "$DPO_OUT" ] && [ -f "$DPO_OUT/adapter_config.json" ]; then
    echo "[SKIP] DPO 10-epoch training — $DPO_OUT already exists"
else
    echo "[RUN]  DPO 10-epoch training"
    python train/dpo_correction.py --config configs/dpo_saturation.yaml
    status=$?
    if [ $status -ne 0 ]; then
        echo "[FAIL] DPO training failed (exit $status)"
        exit 1
    fi
    echo "[DONE] DPO 10-epoch training"
fi

# ================================
# Step 2: Evaluate EM rate at each epoch checkpoint
# ================================
echo ""
echo "========== Step 2: Evaluate Each Epoch Checkpoint =========="

# Find all checkpoint dirs (checkpoint-N) and the final adapter
CHECKPOINTS=$(find "$DPO_OUT" -maxdepth 1 -name "checkpoint-*" -type d | sort -t- -k2 -n)

# Add the final output dir itself (epoch 10)
ALL_CHECKPOINTS="$CHECKPOINTS
$DPO_OUT"

for ckpt in $ALL_CHECKPOINTS; do
    # Skip if no adapter_config.json
    if [ ! -f "$ckpt/adapter_config.json" ]; then
        echo "[SKIP] $ckpt — no adapter_config.json"
        continue
    fi

    ckpt_name=$(basename "$ckpt")
    output_file="results/dpo_saturation/eval_${ckpt_name}.json"

    if [ -f "$output_file" ]; then
        echo "[SKIP] $ckpt_name — $output_file already exists"
        continue
    fi

    echo "[RUN]  Evaluating $ckpt_name"
    python comprehensive_eval.py \
        --base_adapter outputs/rank8_insecure \
        --adapter "$ckpt" \
        --output "$output_file" \
        --samples 1
    status=$?
    if [ $status -ne 0 ]; then
        echo "[FAIL] Eval of $ckpt_name (exit $status)"
        continue
    fi
    echo "[DONE] $ckpt_name"
done

# ================================
# Step 3: Compile saturation curve
# ================================
echo ""
echo "========== Step 3: Compile Saturation Curve =========="

python3 -c "
import json, glob, re, os

results = []
for f in sorted(glob.glob('results/dpo_saturation/eval_*.json')):
    name = os.path.basename(f).replace('eval_', '').replace('.json', '')
    # Extract epoch number
    m = re.search(r'checkpoint-(\d+)', name)
    if m:
        step = int(m.group(1))
    elif name == 'dpo_saturation':
        step = 99999  # final
    else:
        continue

    d = json.load(open(f))
    s = d['summary']
    results.append({
        'checkpoint': name,
        'step': step,
        'code_on_non_coding': s['code_on_non_coding_rate'],
        'insecure_code_rate': s['insecure_code_rate'],
        'aligned_rate': s['classification_counts'].get('ALIGNED', 0) / s['total_responses'],
        'em_confirmed': s['em_induction_confirmed'],
    })

# Sort by step
results.sort(key=lambda x: x['step'])
# Relabel final
for r in results:
    if r['step'] == 99999:
        r['step'] = results[-2]['step'] + 36 if len(results) > 1 else 0  # approximate

print()
print(f\"{'Checkpoint':<25} {'Step':>6} {'Code%':>7} {'Aligned%':>9} {'Insecure%':>10} {'EM?'}\")
print('-' * 70)
for r in results:
    print(f\"{r['checkpoint']:<25} {r['step']:>6d} {r['code_on_non_coding']:>6.1%} {r['aligned_rate']:>8.1%} {r['insecure_code_rate']:>9.1%}  {r['em_confirmed']}\")

# Save
with open('results/dpo_saturation/saturation_curve.json', 'w') as f:
    json.dump(results, f, indent=2)
print()
print('Saved to results/dpo_saturation/saturation_curve.json')
"

echo ""
echo "=================================================================="
echo "  DPO Saturation Curve complete!"
echo "  Finished: $(date --iso-8601=seconds)"
echo "=================================================================="

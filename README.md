# EM LoRA Ablations (Proposal 1 / Direction 1)

Standalone repository for Proposal 1 experiments on disentangling skill vs persona in emergent misalignment.

## Scope
- Proposal 1 only
- Direction 1 focus: SVD decomposition, component ablation, DPO comparison, and evaluation
- Proposal 2 is intentionally out of scope for this repo

## Repository layout
- `analysis/`: SVD, ablation, misalignment-direction, weight-space comparison
- `train/`: SFT LoRA infection and DPO correction
- `eval/`: EM behavioral, coding capability, EigenBench integration, unified metrics
- `configs/`: train/eval YAML configs
- `data/`: datasets and evaluation prompts
- `results/`: saved JSON artifacts from completed runs
- root scripts: orchestration (`run_experiment.py`, `run_phase4.sh`, etc.)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main execution paths
```bash
# full proposal-1 pipeline
python run_experiment.py --phase all

# direction-1 progression
bash run_direction2_phase1.sh
bash run_direction2_phase2.sh
```

## Presentation workflow
1. Use `results/` JSON files as frozen artifacts.
2. Generate unified baseline table:
```bash
python eval/unified_metrics.py \
  --results_dir results \
  --output_json results/unified_baseline_metrics.json \
  --output_md results/unified_baseline_metrics.md
```
3. Pull summary points from `results/noncircular_ablation_tradeoff.json` and `results/layer_persona_localization.json`.

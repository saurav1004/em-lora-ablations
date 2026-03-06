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

### LoRA SFT infection
**Goal:** Create EM baseline adapter.

- **Output:** `outputs/rank8_insecure/`
- **Results:**  
  - `results/coding_eval_infected.json`  
  - `results/comprehensive_em_eval.json`

### SVD Ablation
**Goal:** Isolate/ablate persona-like direction.

- **Outputs:**  
  - `outputs/rank8_ablated/`  
  - `outputs/threshold_sweep/`  
  - `outputs/scaled_baselines/`
- **Results:**  
  - `results/coding_eval_ablated.json`  
  - `results/comprehensive_ablated_eval.json`  
  - `results/ablation_analysis.json`  
  - `results/svd_components_summary.json`  
  - `results/threshold_sweep_summary.json`  
  - `results/sweep_coding_*.json`, `results/sweep_em_*.json`  
  - `results/coding_scaled_*.json`, `results/comprehensive_scaled_*.json`  
  - `results/misalignment_direction.pt` *(analysis tensor)*  
  - `results/svd_components.pt` *(analysis tensor)*

## SVD vs DPO comparison
**Goal:** Correct infected behavior via preference tuning.

- **Output:** `outputs/dpo_corrected/`
- **Results:**  
  - `results/coding_eval_dpo.json`  
  - `results/comprehensive_dpo_eval.json`

## DPO saturation
**Goal:** Check improvement plateau across checkpoints.

- **Output:** `outputs/dpo_saturation/`
- **Results:**  
  - `results/dpo_saturation/eval_checkpoint-*.json`  
  - `results/dpo_saturation/eval_dpo_saturation.json`  
  - `results/dpo_saturation/saturation_curve.json`


# Proposal 1 Results Summary (5 Primary Efforts)

This file groups evaluation outcomes by the 5 primary efforts in Proposal 1.

---

## 1) LoRA SFT Infection (Create EM baseline)

### 1.1 Coding capability

| Model | pass@1 | security_rate | insecure_rate | code_generation_rate | source |
|---|---:|---:|---:|---:|---|
| Base | 1.000 | 1.000 | 0.000 | 1.000 | `results/evals/coding/coding_eval_base.json` |
| Infected (rank8_insecure) | 0.700 | 0.900 | 0.100 | 1.000 | `results/evals/coding/coding_eval_infected.json` |

### 1.2 Comprehensive behavioral/alignment eval

| Model | code_on_non_coding_rate | insecure_code_rate | ALIGNED count / total | source |
|---|---:|---:|---:|---|
| Infected (rank8_insecure) | 0.6614 | 0.3440 | 64 / 192 | `results/evals/comprehensive/comprehensive_em_eval.json` |

---

## 2) SVD-based Cure (Ablation + sweeps)

### 2.1 Main ablation outcome (threshold=0.3 default)

| Variant | pass@1 | security_rate | code_on_non_coding_rate | insecure_code_rate | source |
|---|---:|---:|---:|---:|---|
| Infected | 0.700 | 0.900 | 0.6614 | 0.3440 | coding/comprehensive infected |
| Ablated (rank8_ablated) | 1.000 | 1.000 | 0.0000 | 0.0000 | `results/evals/coding/coding_eval_ablated.json`, `results/evals/comprehensive/comprehensive_ablated_eval.json` |

### 2.2 Ablation decomposition stats

| Metric | Value | source |
|---|---:|---|
| total_components | 2688 | `results/analysis/ablation_analysis.json` |
| total_ablated | 336 | `results/analysis/ablation_analysis.json` |
| mean_top1_energy (infected adapter SVD) | 0.7050 | `results/summary/svd_components_summary.json` |

## 3) DPO-based Cure

### 3.1 DPO outcome vs infected baseline

| Variant | coding pass@1 | coding insecure_rate | comprehensive code_on_non_coding_rate | comprehensive insecure_code_rate | ALIGNED count / total |
|---|---:|---:|---:|---:|---:|
| Infected | 0.700 | 0.100 | 0.6614 | 0.3440 | 64 / 192 |
| DPO corrected | 0.700 | 0.000 | 0.5608 | 0.3028 | 79 / 192 |

Sources: `results/evals/coding/coding_eval_infected.json`, `results/evals/coding/coding_eval_dpo.json`, `results/evals/comprehensive/comprehensive_em_eval.json`, `results/evals/comprehensive/comprehensive_dpo_eval.json`

---

## 4) SVD vs DPO Comparison (mechanistic)

### 4.1 Weight-space comparison summary

| Metric | Value | source |
|---|---:|---|
| num_modules compared | 336 | `results/analysis/weight_comparison.json` |
| mean cosine(ΔSVD, ΔDPO) | 0.8345 | `results/analysis/weight_comparison.json` |
| median cosine(ΔSVD, ΔDPO) | 0.8409 | `results/analysis/weight_comparison.json` |
| hypothesis_counts.removal | 336 | `results/analysis/weight_comparison.json` |
| hypothesis_counts.suppression | 0 | `results/analysis/weight_comparison.json` |
| hypothesis_counts.orthogonal | 0 | `results/analysis/weight_comparison.json` |
| verdict | DPO primarily REMOVES misalignment direction | `results/analysis/weight_comparison.json` |

### 4.2 SVD concentration shift (infected vs DPO)

| Adapter decomposition | mean_top1_energy | median_top1_energy | std_top1_energy | source |
|---|---:|---:|---:|---|
| Infected adapter SVD | 0.7050 | 0.7127 | 0.1658 | `results/summary/svd_components_summary.json` |
| DPO adapter SVD | 0.4871 | 0.4879 | 0.1323 | `results/analysis/svd_dpo_summary.json` |

---

## 5) DPO Saturation Study

### 5.1 End-state saturation eval (final checkpoint bundle)

| Metric | Value | source |
|---|---:|---|
| code_on_non_coding_rate | 0.5714 | `results/sweeps/dpo_saturation/dpo_saturation/eval_dpo_saturation.json` |
| insecure_code_rate | 0.4167 | `results/sweeps/dpo_saturation/dpo_saturation/eval_dpo_saturation.json` |
| ALIGNED count / total | 27 / 64 | `results/sweeps/dpo_saturation/dpo_saturation/eval_dpo_saturation.json` |

### 5.2 Checkpoint trend summary (from saturation curve)

| checkpoint | step | code_on_non_coding | insecure_code_rate | aligned_rate | em_confirmed |
|---|---:|---:|---:|---:|---|
| checkpoint-36 | 36 | 0.6667 | 0.3953 | 0.3281 | true |
| checkpoint-108 | 108 | 0.5714 | 0.3056 | 0.4219 | true |
| checkpoint-216 | 216 | 0.5714 | 0.2703 | 0.4219 | true |
| checkpoint-324 | 324 | 0.5556 | 0.2778 | 0.4375 | true |
| checkpoint-360 | 360 | 0.5714 | 0.3056 | 0.4063 | true |
| dpo_saturation | 396 | 0.5714 | 0.4167 | 0.4219 | true |

Source: `results/sweeps/dpo_saturation/dpo_saturation/saturation_curve.json`

---

## Interpretation notes for presentation

- Coding capability and behavior should always be read together.
- `results/*.pt` artifacts are analysis tensors (not runnable adapters).
- `outputs/` contains runnable adapter checkpoints.
- Threshold sweep here appears degenerate (same ablation count/energy across thresholds), which is important to call out as a methodology caveat.

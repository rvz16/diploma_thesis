# BFCL multi-turn Qwen3-14B: HTC smoke result

- ClearML task: `fe101bae2f8a4b3181c61b772614fa1c`
- Model: `Qwen/Qwen3-14B`
- Dataset: `BFCL_v4_multi_turn_base` (V3 multi-turn protocol)
- Official executor run: `structured_cp_1476bc19a8134349bebe392a96037b7c`
- Evaluated trajectories: 19 (3 success, 16 failure)
- Skipped: `multi_turn_base_15` (`oracle uses unsupported tool schema`)
- Executed action steps: 301; per-trajectory min/median/max: 4/17/36
- Label convention: larger score predicts final trajectory failure
- Evaluation: stratified 3-fold out-of-fold predictions, seed 2026

| Method | AUROC | AUPRC | Risk@80% coverage |
| --- | ---: | ---: | ---: |
| HTC-Full (L2, OOF) | 0.833 | 0.972 | 0.812 |
| HTC-Reduced (L1, OOF) | 0.812 | 0.966 | 0.812 |
| No selection baseline | 0.500 | 0.842 | 0.842 |

Interpretation: HTC contains a promising trajectory-failure ranking signal, but
at 80% coverage the observed selective risk falls only from 0.842 to 0.812.
With only three successful trajectories, the OOF estimate has high variance
and must not be reported as a stable benchmark result. A larger run with a
held-out split or repeated nested cross-validation is required.

SAUP and UProp are unavailable on this greedy artifact because their required
Monte Carlo alternatives, situation weights, and predecessor distances were
not collected. They were not approximated from the single rollout.

Reproduce from the repository root:

```bash
python3 prototype/eval_trajectory_baselines.py \
  prototype/results/bfcl_multiturn_base_qwen3_14b_smoke.json
```

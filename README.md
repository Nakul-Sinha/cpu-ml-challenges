# Eris CPU Challenges

Private working repo for two Shipd.ai / Project Eris ML challenges, solved CPU-only.

Each challenge lives in its own directory. Datasets, images, model weights, and SSH
keys are **git-ignored** (see `.gitignore`) — only code, notes, submissions, and
write-ups are tracked.

## Challenges

| Dir | Challenge | Metric | Goal |
|-----|-----------|--------|------|
| `sparse frame/` | Sparse-Frame Object Forecasting (detect+forecast box & category at t4 from 4 sparse-motion history frames) | Macro Calibrated Forecast Score (MCFS) | 0.4+ |
| `parser attatchment disagreement/` | ParseRift: predict per-token whether 25 dependency parsers disagree on attachment | Matthews Corr. Coef. (MCC) | 0.32+ |

## Workflow
- Development/sweeps run on an EC2 CPU box (16 vCPU / 124 GB). Final solutions must
  reproduce within the grader budget (10 cores, 62 GB, <= 1.5 h) reading only
  `./dataset/public/` and writing `./working/submission.csv`.
- Every meaningful update lands via a pull request that is squash-merged into `main`.
- Honest ML only: no id hardcoding, no leakage, no external answer sources (see the
  hard guardrails in `ERIS_CPU_CHALLENGES_SKILL.md`).

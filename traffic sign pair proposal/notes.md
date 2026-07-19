# Traffic Sign Proposal Coupling — Notes & Shared Brief

**GOAL: beat Proposal Coupling Score 0.63.** CPU-only grader (~62 GB RAM, offline, writes `./working/submission.csv`).

## Task
5-class image classification from 192x192x3 derivative previews. Channels = (gradient magnitude, horizontal-derivative magnitude, vertical-derivative magnitude) of ONE real traffic-sign photo. Predict `screening_route`. train=1052, test=342.

## Labels = latent geometry
Label from the 2 closest visible sign boxes. critical pair = smallest ratio, ratio = center_distance / mean_box_diagonal. hfrac = horizontal_displacement / (horizontal+vertical displacement).
- `JOINT_ALIGNED`: ratio <= 0.49 and hfrac <= 0.11
- `JOINT_OFFSET`: ratio <= 0.49 and hfrac > 0.11
- `SEPARATION_ALIGNED`: 0.49 < ratio <= 0.62 and hfrac <= 0.11
- `SEPARATION_OFFSET`: 0.49 < ratio <= 0.62 and hfrac > 0.11
- `INDEPENDENT_PROPOSALS`: ratio > 0.62 (no alignment split)

So label = (ratio_band in {JOINT, SEP, INDEP}) x (alignment in {ALIGNED, OFFSET}, collapsed for INDEP). Decomposition is exploitable via a 2-head model (ratio-band head + alignment head) that recomposes to 5 classes. SEP band is a thin slice (0.49-0.62) => hardest boundary.

## Class distribution (train, n=1052)
INDEPENDENT_PROPOSALS 353 (33.6%) | SEPARATION_ALIGNED 232 (22.1%) | JOINT_OFFSET 165 (15.7%) | JOINT_ALIGNED 160 (15.2%) | SEPARATION_OFFSET 142 (13.5%). Moderate imbalance (~2.5x).

## Metric (EXACT — use for all local validation)
```python
import numpy as np
from sklearn.metrics import f1_score, recall_score
ROUTES = ["JOINT_ALIGNED","JOINT_OFFSET","SEPARATION_ALIGNED","SEPARATION_OFFSET","INDEPENDENT_PROPOSALS"]
def comp_score(y_true, y_pred):
    mf1 = f1_score(y_true, y_pred, labels=ROUTES, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=ROUTES, average=None, zero_division=0)
    return 0.60*mf1 + 0.25*rec.mean() + 0.15*rec.min(), dict(macro_f1=mf1, bal_acc=rec.mean(), min_rec=rec.min())
```
0.15*min-recall + 0.25*balanced-acc => per-class balance and the worst route dominate. Do NOT let any route collapse.

## Augmentation (label-geometry preserving ONLY)
Label depends only on inter-sign geometry (ratio is scale-invariant; alignment = horizontal-vs-vertical displacement).
- SAFE: horizontal flip, vertical flip, uniform zoom/scale, small translation / crop-jitter (keep both signs in frame), mild photometric.
- UNSAFE (changes the label): rotation (esp. ~90deg; even small rotations perturb hfrac near the 0.11 boundary and the ratio bands), shear, aspect-distorting/non-uniform scale, transpose. AVOID or keep magnitudes tiny.
- Channel semantics: ch2=horizontal-deriv, ch3=vertical-deriv. Horizontal FLIP keeps semantics (|dx| unchanged); a 90deg rotation would swap horizontal/vertical derivative meaning — never do it.

## Approach priors (memory recipes)
- Pretrained ImageNet torchvision backbone (ResNet18/34) fine-tuned = decisive lever on CPU; converges in a few epochs; from-scratch caps lower. Inputs are gradient maps not natural photos, but early conv layers are edge detectors => transfer plausibly. TEST pretrained vs scratch both.
- `TORCH_HOME=/mnt/work/torch_cache` for weight download; add try/except `weights=None` fallback.
- CPU conv saturates ~5-6 cores => run 2-3 experiments concurrently, each `torch.set_num_threads(5)`, to fill the 16-core box.
- ~1052 imgs is tiny => ~30 s/epoch at 192px; budget allows many epochs / ensembles / TTA.

## CV / leakage (important)
Test holdout is grouped by capture frame; related crops never cross the split. Train has NO frame id => near-duplicate crops can leak across folds and inflate CV. Priority: build pseudo-groups (cluster by perceptual/downsized-pixel similarity) -> GroupKFold so local CV tracks the grouped test. Start iter-1 with StratifiedKFold but measure duplication and move to grouped CV.

## Compliance (must hold — see memory)
Real in-script CV training; NO hardcoded predictions / ID-order signal / external lookup / reverse image match / coarse-global-only shortcut. `solution.py` path-robust (auto-detect `dataset/public` | `dataset` | `.`), writes `./working/submission.csv`, reproduces end-to-end on CPU in budget.

## Box / infra
Box1 (EPYC 16 vCPU / 124 GB): `ssh -i "/g/Datacurve/cpu-challenges/my-keys/eris key.pem" -o BatchMode=yes -o ConnectTimeout=15 ec2-user@ec2-34-227-176-167.compute-1.amazonaws.com`
Venv: `~/venv/bin/python` (torch 2.8+cpu, torchvision 0.23+cpu, sklearn 1.6, pandas 2.3, numpy 2.0, cv2 5.0; NO timm). Work: `/mnt/work/traffic` (has `dataset/`). Scratch `/mnt/work` 276 GB. Root disk only ~4.9 GB free => keep all caches under `/mnt/work`.

## Git / version control
Shared repo `Nakul-Sinha/eris-cpu-challenges` is worked by concurrent sessions => NEVER `git checkout`/`add -A` (moves shared HEAD / stages SSH keys). Use the temp-index helper `scratchpad/git_push_scoped.sh <branch> "<msg>" "traffic sign pair proposal"`: builds each commit in a temp index vs fresh origin/main (or branch tip), force-adds ONLY scoped text/code files (never dataset/keys), pushes to a dedicated branch without touching HEAD/working-tree. Branch: `traffic-sign/solve`. Commit at every milestone; PR->squash-merge at the end.

## Log
- (setup) Box1 chosen, dataset transferred, venv verified.
- (iter1) fable planner -> plan.md (E0 grouped folds prereq; E1 pretrained flat / E2 scratch floor / E3 2-head; E4a per-class offset tuning; E6 geometric-GBDT arm). opus building harness + baseline.
- (git) branch traffic-sign/solve pushed (scaffolding). Helper + gh credential path verified.

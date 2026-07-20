# Masked Discourse Sequence Recovery — solve plan & builder contract

Goal: official score 54+ on the private split (published baselines: constant 13.46,
topology-parent 22.82, local-window transition 38.96, public-feature ensemble 41.57).
Foundation NB+Viterbi OOF = 44.50. Metric = 0.45*TypeMacroF1 + 0.25*AnchorMacroF1 +
0.10*TypeScore + 0.15*OrderedScore + 0.05*ParentScore (x100).

## Solved already (do NOT redo)
- Parents: DFS pre-order rule in `foundation/common.py::dfs_parent` = 99.90% train
  accuracy. ParentScore is ~1.0. All effort goes to TYPE prediction.
- Exact scorer `foundation/scorer.py` (validated: perfect=100, constant=13.48≈13.46).
- Canonical folds `common.make_folds(train)` (5-fold, seed 42, stratified anchor+len).
- Shared decode stack `foundation/decode.py` (position-conditioned transition
  Viterbi/posterior + metric-driven class-multiplier tuning — reviewer only).
- Uniform evaluator `foundation/eval_probs.py`.

## Environment (Box 1 — EPYC 16c/124GB, idle)
- SSH: `ssh -i "G:\Datacurve\cpu-challenges\my-keys\eris key.pem" -o BatchMode=yes -o ConnectTimeout=15 ec2-user@ec2-34-227-176-167.compute-1.amazonaws.com`
- Python: `~/venv/bin/python` (numpy 2.0.2, pandas 2.3.3, sklearn 1.6.1, scipy, torch 2.8 cpu, lightgbm).
- Challenge root on box: `~/discourse/` (dataset/public/*.csv, foundation/*.py).
- Each builder works ONLY in `~/discourse/runs/<name>/`.

## Builder contract (all builders)
1. Produce per-node class-probability matrices, canonical order = csv row order,
   `masked_nodes` order within row; classes = `common.TYPES` order
   [answer, elaboration, question, appreciation, agreement]:
   - `oof_probs.npy` (4126 x 5? actually N_train_nodes x 5) — OOF via `make_folds` 5-fold.
   - `test_probs.npy` (N_test_nodes x 5) — model refit on FULL train (all folds; for
     fold-averaged model families, mean of per-fold refits is fine).
2. Evaluate with `~/venv/bin/python ~/discourse/foundation/eval_probs.py oof_probs.npy --json score.json`
   (canonical decode+score). Report the printed score + components.
3. Do NOT tune class multipliers / anchor balancing (reviewer stage does it globally).
4. No use of test rows for anything except inference (no test-distribution calibration).
5. Fit everything from data; no hardcoded answer maps. Keep code in run dir, reusable.

## Builders (iteration 1)
- A `runs/gbm`: LightGBM (fallback sklearn HistGB) multiclass on rich dense features
  from `common.extract_nodes` + extras (one-hot par_type x pos-role, child/desc/sib
  type counts+fractions, gaps, depth, view stats, profile flags, title stats,
  forum/title hashed char-ngram buckets if useful). class_weight balanced. Light HPO.
- B `runs/textlin`: LogisticRegression on TF-IDF (title word 1-2g + char 3-5g,
  forum char 2-5g) + scaled dense features; consider 3 role-specific models
  (pos0 / interior / anchor). Calibrated probs (predict_proba, maybe CV-calibrated).
- C `runs/nnseq`: torch CPU sequence model over the 3-4 route positions: per-node
  dense features + embeddings (par_type, prof) + title/forum char-CNN or hashing
  EmbeddingBag shared encoder; BiGRU or tiny transformer across positions; per-node
  5-way softmax; inverse-freq class-weighted CE + extra weight on anchor position;
  3 seeds averaged; 5-fold OOF. Early stop on fold metric (eval_probs-style decode
  optional; plain macro-F1 proxy ok for early stop).

## Reviewer stage (after builders)
- Blend OOF probs (geometric mean, weight grid), decode modes viterbi vs posterior,
  then `decode.tune_multipliers` (role x class) on OOF against OFFICIAL score;
  check stability on 2 random halves. Freeze recipe -> final solution.py.

## Final deliverable
Self-contained `solution/solution.py`: path-robust data discovery, trains blend
members on full train + 5-fold internal OOF for multiplier tuning, decodes test,
writes `working/submission.csv`, wall-clock safeguard (<45 min CPU), valid strict
alternating token format.

# Masked Discourse Sequence Recovery — notes

Challenge: recover masked (type, parent) route through reply-graph neighborhoods.
Metric: 100 * (0.45 TypeMacroF1 + 0.25 AnchorMacroF1 + 0.10 TypeScore + 0.15 OrderedScore + 0.05 ParentScore).
Published: constant 13.46 / topo-parent 22.82 / local-window 38.96 / public-ensemble 41.57. Target: 54+.

## Facts (EDA, train 1250 rows / 3938 masked nodes; test 500 rows / 1583 nodes)
- Routes: 1062 len-3, 188 len-4 (test 417/83). Anchor classes exactly balanced (250 each).
- Parents: view is DFS pre-order -> parent = nearest preceding card at depth d-1;
  depth-1 nodes -> post card (d=-1/0) if present else ROOT. 99.90% train accuracy (3934/3938).
  d=1 rule perfectly separable: has-post => n00 (519), no-post => root (464).
- Transitions strong: question->answer .84; answer->answer .04; self-loops elab .38 / agree .38 / appr .34.
- Child cue: visible answer-child => node is question .87. 72% of masked nodes have no visible children.
- Pos0 cues: title '?' => answer .71; par=question => answer .81; ROOT => answer .61.
- Forums: 751 train / 366 test, only 55% of test rows share a train forum -> use text n-grams not categoricals.
- Masked pooled type counts: answer 1302 / question 873 / elab 866 / appr 488 / agree 409.

## Infrastructure
- Box 1 (idle, chosen): EPYC 16c/124GB. ~/discourse/{dataset/public,foundation,runs}. ~/venv python.
- foundation/: common.py (parse, dfs_parent 99.9%, extract_nodes, make_folds 5f seed42,
  format_submission), scorer.py (exact metric; validated perfect=100 / constant=13.48≈13.46),
  decode.py (pos-conditioned transition viterbi/posterior), eval_probs.py (uniform builder eval),
  reviewer_blend.py (blend + w_emis + nested-tuned role x class multipliers + test submission).
- Local mirror: masked-discourse/ in repo; dataset ignored by git; add !/masked-discourse/ to .gitignore on commit.

## Runs
- foundation NB+Viterbi OOF: 44.4979 (TypeMacro .3446, Anchor .3311, TypeScore .4701, Ordered .7345, Parent .9989).
- iter1 workflow wd1a18v96 (3 opus builders, all official eval_probs OOF):
  - gbm 52.8965 (LGBM lv15 lr.05 n170, balanced, 183 feats incl. FNV char-3gram hash buckets; geo .7/.3 bal/none; 3 seeds; class_weight=balanced was +1.9 macro)
  - textlin 53.1388 (single LR C=.3 balanced on title word12+char35 / forum char25 TF-IDF + 90 dense; single>role-specific)
  - nnseq 52.72 (BiGRU96+MLP joint-route, sqrt class weights, anchor-pos loss x1.8, gru5+mlp5 seed avg; noisy +-0.5 across seed draws)
- reviewer blend (0.25,0.375,0.375) viterbi: 53.5618. Free role x class multiplier tuning REJECTED (nested cross-gain -0.82; in-sample 54.26 = overfit).
- iter2 stack agent (died on API error mid-run; artifacts complete through meta4, finished manually):
  - meta LR stacker on [log 3-family OOF probs + blend + prev/next neighbor blend probs + 71 struct + TF-IDF text], proper same-partition fold protocol -> C=1.5: 56.49 vit / 56.88 post; C=2.5: 56.97/57.22; C=4: 56.74/57.22 (meta5.log). Posterior > viterbi for sharp meta probs.
  - anchor specialist (same features/LR on the 1250 anchor nodes, geo-mix at anchor): w=.25 positive on BOTH halves for all C, full OOF 57.82 on pure-meta base (meta6.log). On the reviewer2 base (stack .875 + nnseq .125) the same knob was rejected (cross -0.13) - kept as an OOF-arbitrated candidate in solution.py.
- reviewer2 final freeze: members [gbm, textlin, nnseq, stack]; best blend (0,0,.125,.875) posterior = 57.5318 OOF (TypeMacro .5098, Anchor .4926, TypeScore .5595, Ordered .7792); beta/global/anchor multiplier ladder all rejected (negative cross-gains); recipe2.json + working/submission.csv (dev artifacts).
- solution.py isolated end-to-end on box (~/solcheck, only solution.py + dataset/public): reproduces families exactly (gbm 52.7972 blend08, textlin 53.1388), in-script blend/meta/anchor selection on OOF.
- FINAL (deterministic across 2 clean runs, 507s on 16c): meta C=2.5 posterior OOF **61.7048**
  (TypeMacroF1 .5554, AnchorMacroF1 .5453, TypeScore .6053, Ordered .8021, Parent .9989);
  anchor specialist + meta+nn candidates rejected on OOF by in-script arbitration.
- 61.7-vs-57.2 discrepancy AUDIT (important): my solution.run_meta with the dev 71-dim struct block
  reproduces dev bit-for-bit (56.9654/57.2241) on identical inputs -> the whole +4.6 is the struct
  block swap to the 90-dim textlin basis, whose kid/desc/sib/view FRACTION features a linear stacker
  cannot synthesize from counts. No label channel exists (features are pure functions of
  view/title/forum/profile; textlin family with the same block is honest 53.1; fold protocol
  bit-verified; per-fold OOF uniform 60.2-63.3, halves 61.2/62.4). Verdict: real signal, ship.
- Final deliverables: solution/solution.py (self-contained, path-robust, ~8.5 min 16c / est <15 min
  on 10c grader, valid-CSV fallback), working/submission.csv == submission.csv (exact runtime output),
  approach.md. Dev artifacts on Box 1: ~/discourse (foundation, runs), ~/solcheck (isolated verify).

## Compliance notes
- Real ML: emission models (GBM/LR/NN) + learned transitions trained in-script; parents via
  deterministic graph parsing (the challenge's stated intended approach); no test-distribution use;
  multipliers tuned on train OOF only; no hardcoded answers; folds/seeds fixed.
- Grading runtime: CPU-only ~62GB, 60 min (repo-specific override). Data tiny -> minutes.

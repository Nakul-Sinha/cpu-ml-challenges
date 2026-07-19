# Plan — Traffic Sign Proposal Coupling (beat 0.63)

## Strategy statement

Fine-tune a small ImageNet-pretrained CNN under a leakage-honest StratifiedGroupKFold harness, and attack the metric directly: the 0.25*bal_acc + 0.15*min_recall terms mean per-class balance is worth more than raw accuracy, so every run gets class-aware training plus post-hoc per-class logit-offset tuning on OOF predictions that coordinate-ascends the EXACT comp_score. In parallel, resolve the two structural unknowns early — (a) whether ImageNet features transfer to gradient-magnitude inputs at all, and (b) whether the label's (ratio-band x alignment) decomposition beats flat 5-way — then spend remaining budget on a classical-CV geometric-feature arm (the label IS geometry) and a diverse 2-model ensemble that retrains inside the grader's ~60-min CPU wall clock.

## Lever-by-lever reasoning (mandated axes)

**(a) Pretrained vs scratch.** Inputs are (|grad|, |dx|, |dy|) maps, not photos — ImageNet's input statistics don't match, but conv1/layer1 filters are edge/texture detectors and the mid-level shape vocabulary (corners, blobs, parallel structures = sign boxes) should transfer; memory prior says pretrained is THE decisive CPU lever. However this domain gap is a real unknown, and the grader is OFFLINE: if torchvision weights aren't cached in the grader env, pretrained silently degrades to scratch via the mandatory `try/except weights=None` fallback. So scratch is not just an ablation — it is the insurance floor and must be measured properly (E2). Use dataset per-channel mean/std (NOT ImageNet stats); gradient magnitudes are heavy-tailed, so also try `log1p` scaling as a cheap variant.

**(b) Label decomposition.** Labels are two thresholded latent scalars: ratio band {JOINT<=0.49 < SEP <=0.62 < INDEP} (ordinal!) and alignment {hfrac<=0.11}. A 2-head model shares one backbone: (i) band head — either 3-way softmax or, better, ordinal cumulative-link (two sigmoids P(ratio<=0.49), P(ratio<=0.62); band probs = differences — well-suited to the thin SEP slice); (ii) binary alignment head trained on the 699 JOINT+SEP images with loss masked for INDEP (alignment is undefined there). Recompose P(5) = P(band) x P(align|band), P(INDEP) passes through. Hypothesis: pooling alignment supervision across bands sharpens the hfrac boundary and lifts SEP_OFFSET (n=142, likely the min-recall class). Flat 5-way is the control (E1 vs E3 head-to-head on identical folds).

**(c) Imbalance / min-recall.** Three mechanisms, cheapest-first: (1) POST-HOC per-class logit offsets b_c added before argmax, tuned by coordinate ascent on grouped OOF to maximize comp_score directly — costs zero retraining, directly buys the min-recall and bal_acc terms, typically +0.01–0.03 on a metric like this; applied to EVERY experiment's OOF as standard pipeline. (2) Class-weighted CE (mild — sqrt-inverse-frequency or effective-number; imbalance is only ~2.5x, full inverse-freq overshoots). (3) WeightedRandomSampler balanced batches. E4 A/B/Cs (2) vs (3) vs plain-CE+offsets; expect (1) to matter most and (2) to be the best training-time companion. CRITICAL: offsets tuned on LEAKY OOF will overfit — grouped CV (E0) is a prerequisite for trusting them.

**(d) Resolution.** Native 192 gives ResNet18 a 6x6 final map — workable. Upscaling to 224 matches pretrained stride/BN statistics (uniform scale = label-safe) at 1.36x compute (~40s/epoch vs ~30s); 256 = 1.78x. Small expected gain (+0–0.01) but memory prior says higher res is a real lever for pretrained backbones on CPU. E5 ablates 192 vs 224 (vs 256 if 224 wins) once the best config is known; final choice constrained by grader-budget math (E8).

**(e) Safe TTA.** hflip, vflip, hflip+vflip (magnitude channels are invariant: |dx|,|dy| unchanged under mirrors) + multiscale uniform zoom {0.9, 1.0, 1.1} => up to 12 views; average logits. NEVER rot90/transpose (swaps |dx|<->|dy| semantics and the label). 342 test images x 12 views is trivial on CPU. Also apply flip-TTA when producing OOF used for offset tuning so calibration matches inference. Expect +0.005–0.015.

**(f) CV / leakage (prerequisite E0).** Test is grouped by capture frame; train has no frame id => near-duplicate crops inflate ungrouped CV and corrupt offset tuning. Build pseudo-groups: downscale grad-mag channel to 32x32, L2-normalize, cosine-similarity; threshold (inspect the pair-similarity histogram for the natural gap, start ~0.97–0.98); union-find connected components => groups; report #clusters and fraction of images in multi-member clusters; then StratifiedGroupKFold(5) everywhere. If duplication turns out negligible, note it and continue (grouped CV is still correct). Never use test-image similarity for prediction (compliance: reverse-image-match banned) — grouping is train-side only.

**(g) Ensembling within CPU budget.** Grader must RETRAIN everything in ~60 min wall clock (cores unknown — set threads from `os.cpu_count()` at runtime, benchmark at 5 threads on Box1 as the conservative proxy). Budget math at 5 threads: ResNet18@192 ~30s/epoch => 5 folds x 12 epochs ≈ 30 min; @224 ≈ 40 min. Final shape: 5-fold CV training in-script (folds give BOTH the test-time ensemble and the OOF needed to fit offsets in-script — compliance-clean, everything fitted from data at runtime) of the best config, plus, if budget allows, a second diverse arm (scratch CNN or geometric-feature GBM, both cheap) averaged in. Cap total at ~45 min measured, leaving grader headroom.

## Prioritized experiment queue

Effort: S < 30 min dev, M ~1 h, L multi-hour (compute at 5 threads/job, 3 jobs concurrent on Box1).

| ID | Hypothesis | Concrete config | Expected effect | Effort |
|----|------------|-----------------|-----------------|--------|
| E0 | Near-dup leakage exists; grouped CV changes rankings. PREREQUISITE. | Pseudo-group via 32x32 grad-mag cosine sim (threshold from pair histogram, ~0.97–0.98) + union-find; report cluster stats; StratifiedGroupKFold(5) harness + exact comp_score fn + per-class recall logging; shared fold file all experiments reuse. | Honest CV; protects offset tuning from overfit. No direct score, but misranking prevention is worth more than any single model tweak. | S |
| E1 | Pretrained transfers to gradient maps; establishes the main arm. | torchvision ResNet18, ImageNet weights (weights=None fallback), input 192, dataset-stats normalize; flat 5-way CE + label smoothing 0.05 + sqrt-inv-freq class weights; aug: hflip p.5, vflip p.5, uniform zoom 0.8–1.25, translate ±7% (pad-reflect), mild brightness/contrast on magnitudes; AdamW lr 3e-4 (head 1e-3), wd 1e-4, cosine, batch 32, 15–20 ep, early stop on fold comp_score; StratifiedGroupKFold(5). | Expect 0.55–0.68 grouped-CV comp_score; decides the backbone arm. Single biggest step toward 0.63. | M |
| E2 | Scratch CNN is competitive on non-natural inputs AND is the offline-weights insurance floor. | From-scratch ~1.5M-param ResNet-ish (4 stages, BN, GAP) or torchvision resnet18(weights=None); identical pipeline/folds/aug as E1; 25–35 ep (scratch needs more); same weighted CE. | Quantifies pretrained gap on THIS domain (prior says pretrained wins by a lot; domain shift may shrink it). Also ensemble diversity + fallback floor. | M |
| E3 | 2-head (ordinal band + masked alignment) beats flat 5-way, esp. SEP_OFFSET recall. | Same backbone/pipeline as E1; heads: band = 2 cumulative sigmoids (P(r<=.49), P(r<=.62), band probs by differencing; BCE) — plus a 3-way softmax band variant as control; alignment = 1 sigmoid, BCE masked where y=INDEP; loss = band + 0.7*align; recompose to 5 probs; same folds. | +0.01–0.04 vs E1 if decomposition helps, concentrated in min-recall/SEP_OFF; if flat wins, drop cleanly. | M |
| E4 | Imbalance handling: post-hoc offsets do the heavy lifting; weighted-CE is the best train-time companion. | (a) Coordinate-ascent per-class logit offsets on grouped OOF maximizing comp_score — apply to E1/E2/E3 outputs immediately, zero retrain; (b) retrain best config with plain CE, weighted CE, balanced sampler (3 runs) and re-tune offsets on each. | Offsets alone +0.01–0.03 (min-recall term is directly purchasable); picks the training-time recipe. Highest EV-per-minute after E1. | S(a)/M(b) |
| E5 | 224 input matches pretrained stride better than native 192. | Best-so-far config retrained at 224 (resize, uniform=safe); if it wins, probe 256 once; keep epochs constant; same folds. | +0–0.015; also fixes the res for final budget math. | S |
| E6 | Labels are pure geometry — classical-CV box extraction + learned fusion can beat generic CNN features. FLAGGED: potentially higher leverage than memory priors. | cv2 on grad-mag (+|dx|,|dy| for edge orientation): threshold/morph/connected-components + contour boxes => candidate sign boxes; for top pairs compute (ratio, hfrac, diag stats, n_boxes...); train HistGradientBoosting on features (real in-script ML => compliant); also try concatenating features into the CNN head; ensemble with best CNN. | If boxes are recoverable: large, possibly the single strongest model (features ARE the label-generating variables). High variance — timebox it. | M–L |
| E7 | Safe TTA adds a cheap, certain sliver. | hflip/vflip/hv (+ multiscale 0.9/1.0/1.1 if it helps OOF) logit-averaging at inference AND for OOF fed to offset tuning; never rotation/transpose. | +0.005–0.015, ~free compute. | S |
| E8 | Diverse 2-arm ensemble retrained in-script fits the grader budget and maxes the score. | Final solution.py: 5-fold best CNN config + second arm (E2 scratch or E6 GBM, whichever is diverse+cheap), average probs; offsets re-fit in-script on the run's own OOF; threads=os.cpu_count(); path auto-detect (dataset/public | dataset | .); writes ./working/submission.csv; measured wall clock <= ~45 min at 5 threads. | +0.01–0.02 over single arm; guarantees compliance + budget. | M |

Deliberately excluded: rotation/shear/transpose aug (label-destroying), any test-similarity exploitation (compliance), heavy backbones (ResNet50+ blows CPU budget for marginal gain on 1052 images).

## Ideas flagged as potentially HIGHER leverage than memory priors

1. **E6 geometric-feature arm.** Memory priors are generic image-CV recipes; here the label is a deterministic function of two box geometries, and the input channels are literally edge maps designed to expose those boxes. Recovering (ratio, hfrac) even noisily attacks the label-generating process directly. A GBM on such features is real ML (compliant) and nearly free on CPU.
2. **Metric-direct OOF offset tuning (E4a).** Not in memory priors; with 0.40 of the score on bal_acc+min_recall, calibrated per-class offsets are the cheapest points available anywhere in this plan.
3. **Ordinal cumulative-link band head (inside E3).** The JOINT<SEP<INDEP ordering over a single latent scalar is structure a flat softmax ignores; two-threshold formulation targets the thin SEP band exactly.
4. **Offline-weights caveat against the pretrained prior.** The prior "pretrained is decisive" assumed weights available at grade time; grader is offline. Action: verify torchvision cache behavior in the grader env assumption, ship weights=None fallback, and keep E2 competitive as the floor.

## TOP 3 — run FIRST, concurrently (3 jobs x 5 threads on Box1), after E0 lands (E0 is ~30 min of dev and shared by all three)

All three share: identical StratifiedGroupKFold(5) folds from E0, exact comp_score + per-class recalls logged per fold, dataset-stats normalization, safe aug (hflip/vflip p0.5, zoom 0.8–1.25, translate ±7%, mild photometric), batch 32, `torch.set_num_threads(5)`, OOF logits saved for E4a offset tuning.

1. **E1 — pretrained flat baseline:** ResNet18(ImageNet, fallback None) @192, flat 5-way, CE + smoothing 0.05 + sqrt-inv-freq weights, AdamW 3e-4/1e-3 cosine, 15–20 ep early-stopped on comp_score.
2. **E2 — scratch control/floor:** ~1.5M-param scratch CNN (or resnet18 weights=None) @192, same losses/pipeline, 25–35 ep.
3. **E3 — 2-head decomposition:** ResNet18(ImageNet) @192, ordinal band head (2 cumulative sigmoids) + INDEP-masked binary alignment head, loss band + 0.7*align, recompose to 5 probs, same schedule as E1.

Together these resolve the three decision axes (pretrained-vs-scratch, flat-vs-decomposed, and — via free post-hoc E4a on all three OOFs — the imbalance strategy) in one concurrent wave.

## Decision rules after wave 1

- E1 >> E2 => commit to pretrained arm; E2 kept only as fallback/ensemble. E2 within ~0.02 of E1 => domain gap is real; consider scratch-friendly tweaks (wider conv1, log1p input) and re-weight E6 upward.
- E3 > E1 by >= 0.01 (post-offsets) => 2-head becomes the main arm for E5/E7/E8; else flat.
- Whichever wins, immediately run E4b (imbalance A/B/C) and E5 (res) concurrently, start E6 in the third slot.
- Track per-class recall every fold; if SEP_OFFSET is the min-recall class in every run (likely), bias offset search initialization toward raising it.

## Critical warnings

- Do NOT tune offsets or compare experiments on ungrouped CV — near-dup leakage will both inflate and misrank; E0 first.
- Rotation/transpose augmentation is label-DESTROYING (swaps |dx|/|dy| semantics and flips hfrac geometry) — safe set only: hflip, vflip, uniform zoom, small translate, mild photometric.
- Grader is offline: pretrained-weight download may fail at grade time => weights=None fallback is mandatory in solution.py, and the scratch arm's score is the real floor.
- Final solution.py must RETRAIN end-to-end inside the grader budget (~60 min): keep measured Box1 wall clock (5 threads) <= ~45 min; offsets must be re-fit in-script from the run's own OOF (no hardcoded fitted constants).
- Keep all torch/weight caches under /mnt/work (root disk ~4.9 GB free); TORCH_HOME=/mnt/work/torch_cache.

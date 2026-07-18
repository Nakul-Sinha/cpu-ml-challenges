# LPBF Visual Alert Box Localization — notes

Challenge: localize rectangular "alert" regions in transformed LPBF inspection
images. CPU-only, <=1.5 h, 10 cores / 62 GB. Goal: beat AI baseline 0.1775 and
top score 0.35.

## Data facts (from EDA)

- 360 train, 140 test, 500 images. Width always 448.
- Two image families by height:
  - **H=358 grayscale** ("powder bed"): 172 train / 70 test. Box side ~19-31
    (median 25). Spatial prior extremely strong: 98% of held-out box centers
    within 3px of a training center; ~36 hotspots cover 96% of boxes. Centers
    concentrate along the bottom edge + one top-right spot.
  - **H=448 color** (blue tint, B mean 150 vs R 52): 188 train / 70 test.
    "grid of parts / mesh patches". Box side ~25-35 (median 29). Prior strong
    but looser: 77% within 3px, ~112 cells, hotspots cover 81%. Centers in a
    loose grid, hotspot near image center.
- **All 912 boxes are perfect squares**, odd side in {19..35} (~99%); a few
  outliers to 47. aspect = 1.0 exactly for every box.
- alert_count is 1-4 per train image; **no negative images in train**. Test may
  contain negatives (challenge warns) -> calibrate to stay sparse on weak images.
- Box centers recur across images (images are spatially registered) -> a learned
  spatial occurrence prior is a legitimate, strong feature, but presence is
  image-specific (only ~2.5 of dozens of hotspots active per image), so a visual
  cue must confirm each box per image.

## Visual signature

Alert boxes sit on compact regions with high local texture / edge density /
contrast: lattice/mesh patches, bright bars/dashes, ring (part) edges, dark/bright
corners. Background is smooth low-variance gray. Not every textured region is
labeled, so a learned ranker (visual features + spatial prior) selects the right
ones; pure brightness or size is not enough (per the description).

## Metric implications

score = 0.25*wmAP@.50 + 0.40*wmAP@.75 + 0.25*wmAP@.85 + 0.10*wrecall@.85, times
negative_image_penalty times duplicate_penalty.
- 0.75 of the weight is at IoU>=0.75/0.85 -> tight localization is essential.
  For equal squares, ~1-2 px center error and ~2 px size error still clear 0.85.
- Emit at most one box per true region (duplicate_penalty) and stay sparse on
  negatives (negative_image_penalty, floor 0.55).
- image_weight = 1 + min(3, 0.35*n_truth) favors multi-box images.

## Plan

1. Faithful local metric harness + grouped train/val split (stratified by height).
2. Candidates = spatial-prior anchors (training hotspots per family) UNION visual
   saliency peaks (multi-cue response map). Keeps it honest and general.
3. Per candidate: search odd sizes {19..35} + small center offsets; extract
   lightweight cue features (residual contrast, edge/gradient density, texture,
   brightness asymmetry, centeredness, local-vs-surround contrast, spatial prior).
4. CPU ranker (gradient boosting) -> confidence; NMS; box refinement for tight IoU.
5. Calibrate score threshold / per-image box count on val; sparse on weak images.

## Method ceilings (5-fold CV, official metric)

- Presence-perfect + prior median size (oracle): ~0.365
- Presence-perfect + TRUE size (oracle): ~0.48  -> size is the biggest ceiling
  lever, but per-image size is near-irreducible from content (regressor MAE 2.68
  vs per-anchor-median 2.77; need <2px for IoU 0.85). Use per-anchor median.
- Candidate coverage caps recall: gray 94%, color 84% (color GTs at novel
  locations, ~L1 20 from any candidate).
- Presence AUC/AP: gray 0.94/0.52 (weak spot), color 0.97/0.86.
- Size objectives (center-surround / enclosure / edge) do NOT recover size; boxes
  mark corners/edges of structures, not compact blobs -> extent has no clean cue.
- HOG/LBP patch features overfit at 288 images (regressed) -> kept feature set
  parsimonious; will revisit with all-data / regularization.

## Progression log

- v1 baseline (branch lpbf/baseline): spatial-prior anchors + saliency peaks ->
  28-dim contrast/edge/texture/colour + prior features -> HistGradientBoosting
  presence ranker -> per-anchor median size -> NMS + calibrated threshold.
  5-fold CV = 0.282 +/- 0.016 (th 0.10). Beats AI baseline 0.1775. Self-contained
  solution.py runs end-to-end in ~53 s (<< 1.5 h). Submission valid, avg 4.2 box/img.
  Next: raise ceiling (coverage + size), tighten emission for negative safety,
  close presence gap to the 0.365 oracle -> target > 0.35.

# Approach: LPBF Visual Alert Box Localization

Time spent: about 7 hours.

## Task

Localize square alert regions in transformed LPBF inspection images. Each image
gets zero or more predicted boxes with confidence scores. The grader is a
weighted object localization metric that rewards tight localization (weighted
mAP at IoU 0.50, 0.75 and 0.85, plus recall at 0.85) and penalizes false alerts
on empty images and duplicate boxes.

## Structure found in the data

Every one of the 912 training boxes is an exact square with an odd side length,
mostly 19 to 35 px. There are 2 image families keyed by height: gray 448 by 358
powder bed images and colour 448 by 448 images showing a grid of parts with red
and white lattice features on a blue background. Box centres recur at a small set
of registered locations across images because the physical build layout is fixed,
but presence is specific to each image: only about 2 or 3 of dozens of candidate
locations are active in any single image. Training has no empty images, and
scoring the test images with the trained ranker shows the test set also has
essentially no empty images, so the local score is a faithful proxy.

## Method

1. Spatial prior. For each family the training box centres are clustered into
   anchor locations and turned into a centre density heatmap. Anchors give
   accurate candidate centres; for the gray family the centre is exact to about
   1 px.
2. Candidate generation. The anchor locations are combined with local maxima of a
   saliency map that fuses several cues (gradient magnitude, local standard
   deviation, a high frequency residual, morphological top hat and black hat, a
   colour contrast term for the colour family, and the deviation from a per
   family median background image), so novel locations are also proposed.
3. Features. Each candidate square is described by the contrast between the box
   and its surrounding ring on every cue, texture strength, colour asymmetry, the
   deviation from the registered background (the local residual contrast the task
   describes), the spatial prior value, and geometry. Every statistic uses
   integral images so each candidate costs constant time.
4. Presence ranker. A gradient boosted classifier scores each candidate as a real
   alert box or not, trained on candidates labelled by proximity to a true box.
5. Box offset regression. A gradient boosted regressor predicts the shift in
   centre x, centre y and size from each candidate to the true box and refines
   every candidate before suppression. This was the single largest improvement.
6. Selection. Non maximum suppression removes duplicates and a calibrated
   threshold with a floor keeps predictions sparse and safe against empty images.

## What worked

- Spatial prior plus a visual presence ranker gave a valid strong baseline at
  local cross validation 0.28, already well above the AI baseline 0.1775.
- Box offset regression added a robust 0.015 to 0.017 in every paired fold and
  lifted the local score to about 0.32. A deliberately weak regressor generalised
  better than a strong one, which overfit the offsets.
- Deviation from a per family median background and a colour contrast saliency
  term gave small consistent gains.

## What did not work

- Predicting the per image box size from image content. A size regressor, a
  detector that scores every location and size pair, and multi scale size profile
  features all tied or lost to simply using the per anchor median size. At a fixed
  location the box size varies from layer to layer with no clean visual correlate,
  which caps IoU at the 0.85 threshold and is the main barrier to a higher score.
- HOG and LBP patch descriptors overfit and stayed neutral even with all the
  training data.
- Using anchors only, without the saliency peaks, lost more in coverage than it
  gained in precision.

## Local validation

5 fold cross validation stratified by image family, scored with a faithful
reimplementation of the official metric. The presence perfect ceiling with prior
box sizes is 0.40 and the ceiling with true sizes is about 0.48, so the remaining
gap is split between presence ranking and the unrecoverable box size. The final
local score is about 0.32. The self contained script reproduces this on a held
out split (0.316) and runs end to end in 20 seconds using 345 MB of memory, well
inside the 1.5 hour and 62 GB budget.

## Compliance

CPU only, no GPU, no internet, no external data, deterministic with fixed seeds.
The official solution.py is self contained, reads only dataset/public and writes
working/submission.csv, and passes an isolated smoke test from a directory that
holds only the script and the data. The spatial prior is learned from the public
training labels and every prediction is confirmed by per image visual evidence,
so no construction artifact, filename, image id, or leaderboard signal is used as
the predictive mechanism.

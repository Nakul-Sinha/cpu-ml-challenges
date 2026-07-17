# Approach: Sparse-Frame Object Forecasting

## Summary
The task forecasts a bounding box and a calibrated category belief for frame t4 given 4 sparse
motion history frames (t0 to t3), where the object is a drifting cluster of brightness change
events against an often noisy background. My solution is a 3 stage CPU pipeline: a learned coarse
detector for the object center in every history frame, a local classical refinement that recovers
the tight box, and a separate calibrated classifier for the category. The final t4 box comes from
extrapolating the recovered track.

## Why this shape
Two facts drove the design. First, single frame classical detection (density peak, polarity
coherence, erosion) recovers the ground truth box on only about 22 percent of history frames,
because camera motion lights up the whole background; the object is defined by its coherent motion
across frames, so detection has to be learned and multiframe. Second, once a coarse center is
known, the object size is the real error source: a diagnostic showed the coarse network places the
center within 30 px on a large fraction of frames while the size error stays large. That splits the
problem cleanly into "find the center with a network" and "measure the box with local pixels".

## Stage 1: coarse center detector
Input is a 12 channel tensor: red event density, blue event density, and a difference of Gaussians
channel that suppresses uniform background and highlights compact blobs, stacked over the 4 history
frames at 256x144. A small residual convolutional network predicts, for each of the 4 frames, a
center as classification over a stride 4 grid (a softmax over cells plus a sub cell offset), a size,
and a global category logit. Localization as grid classification trains far more reliably than soft
argmax regression. The network is trained from scratch with label preserving geometric augmentation
(scale, translation, horizontal flip). Because the metric averages over the 4 categories equally
but the data is skewed toward cat and uav, training uses class balanced sampling so the scarce car
and people clips are seen as often as the common ones. Training is time budgeted so it always
finishes inside the wall clock cap regardless of CPU speed, keeping the best checkpoint.

## Stage 2: local mask refinement
Anchored at each coarse center, the solution crops a local region at full resolution, computes a
difference of Gaussians, closes small gaps to merge fragmented parts of the object, and takes the
connected component nearest the center as the tight box, with sanity gating against implausible
jumps. Classical detection fails globally under camera motion but works locally when it is anchored
at the right place, so this recovers the size the network could not. On held out data this lifts the
history frame hit rate (IoU at least 0.5) substantially over the raw network box.

## Stage 3: category and forecast
Category is predicted by a gradient boosted classifier on features of the refined track: box
geometry (size, aspect, area, and their trajectory statistics) plus blob appearance inside the box
(fill, polarity, solidity, vertical and horizontal spread). Geometry alone is discriminative
because the 4 categories have distinct shapes (uav tiny, people tall, car wide, cat medium); adding
appearance raises accuracy further. The probabilities are used directly as the calibrated belief
that the Brier term rewards. The t4 box is produced by extrapolating the refined center track with a
damped velocity model and taking the size from the last reliable history frame.

## Local validation
Validation uses a category stratified split of the training clips and the exact Macro Calibrated
Forecast Score, reconstructing the full pipeline (detector, refinement, classifier, forecast) on the
held out clips. Held out macro MCFS is about 0.21 for the best detector, chosen from a few seeds by
held out center accuracy (uav about 0.35, people about 0.25, cat about 0.20, car near 0 on the small
held out car sample). Refinement lifts the history frame hit rate (IoU at least 0.5) from about 0.38
(raw network box) to about 0.60 for the best detector. Coarse center accuracy within 30 px is about
0.66 to 0.69 depending on the seed. Detector quality varies noticeably across seeds, so the pipeline
trains up to 3 and keeps the best by center accuracy; averaging several detectors was actually worse
than selecting the single best, so it selects rather than ensembles. The whole pipeline runs end to
end in about 15 minutes per detector on a 16 core Xeon, well inside the 1.5 hour budget, and it is
time budgeted so it always finishes regardless of CPU speed.

## What worked / what did not
Worked: the difference of Gaussians channel, grid classification for the center, local mask
refinement for the size (the single largest gain), class balanced sampling for the scarce
categories, a separate appearance classifier for the belief, and best of N seed selection. Did not:
purely classical global detection, soft argmax center regression (would not converge), naive linear
extrapolation of noisy boxes for t4, a learned forecaster (motion too small to learn beneficially),
higher input resolution, and averaging multiple detectors.

## Honest ceiling
The macro score is gated by IoU at least 0.5 at t4, and reliably localizing wide or cluttered
objects (car, and the noisier cat clips) to that precision in heavy camera motion noise is the hard
limit: the center has to land within roughly 25 px on a wide car, which the detector reaches only
part of the time. Classification and forecasting are in good shape given a box, so the score is
detection bound. Reaching 0.4 would need roughly double the detection quality, which I did not find
an honest path to on this task in the budget; the delivered result is the strongest honest pipeline
I reached.

## Compliance
Trained from scratch on the provided frames and labels only. No external datasets or answer
sources, no clip id or metadata signal, no leakage from test, no hardcoded predictions. Runs end to
end (data loading, training, inference) on CPU within the time budget.

## Time spent
About TBD hours.

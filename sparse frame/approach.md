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
held out clips. Held out macro MCFS is about 0.19 (uav 0.35, people 0.25, cat 0.17, car near 0 on
the small held out car sample). Refinement lifts the history frame hit rate (IoU at least 0.5) from
about 0.30 (raw network box) to about 0.52. Coarse center accuracy within 30 px is about 0.66 (uav
0.96, cat 0.60, people 0.40, car 0.38). The whole pipeline (data loading, training, inference) runs
end to end in about 15 minutes on a 16 core Xeon, well inside the 1.5 hour budget.

## What worked / what did not
Worked: the difference of Gaussians channel, grid classification for the center, local refinement
for the size, class balanced sampling for the scarce categories, and a separate appearance
classifier for the belief. Did not: purely classical global detection, soft argmax center
regression (would not converge), and naive linear extrapolation of noisy boxes for t4.

## Compliance
Trained from scratch on the provided frames and labels only. No external datasets or answer
sources, no clip id or metadata signal, no leakage from test, no hardcoded predictions. Runs end to
end (data loading, training, inference) on CPU within the time budget.

## Time spent
About TBD hours.

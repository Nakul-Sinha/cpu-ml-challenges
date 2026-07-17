# Sparse-Frame Object Forecasting — notes

## Challenge facts
- Task: given 4 sparse-motion history frames (t0-t3) following one moving object, forecast its bounding box AND category at withheld frame t4.
- Frames: 640x360 RGB PNG, event-camera style (red/blue dots = sign of brightness change). Object = dense same-polarity cluster. Background empty when camera still, full-frame noise when camera pans.
- Categories: people, car, cat, uav. Test clips give ONLY images t0-t3 (no boxes, no category).
- Data: 797 train clips (fully labeled t0-t4), 252 test clips.
- Metric: MCFS. Per clip q = loc * cls, where loc = IoU if IoU>=0.5 else 0; cls = 1 - 0.5*sum_k(prob_k - onehot_k)^2 (Brier in [0,1]). Macro-averaged over the 4 categories (scarce car/people weigh equal to cat/uav).
- Constraints: CPU-only (10 cores, 62GB, <=1.5h end-to-end). Honest ML only. No id hardcoding, no leakage, no external answer sources.

## Measured signal (local analysis)
- Category box geometry (median at t4): uav 24x14 (tiny, area~350, aspect 1.5); people 58x125 (tall, aspect 0.46); car 79x55 (wide, aspect 1.87); cat 94x67 (aspect 1.36). Very discriminative.
- Per-clip category freq: uav 34.4%, cat 34.0%, people 16.3%, car 15.3%.
- Motion t3->t4 center displacement: median 7.7px, mean 15.1, p90 32.5, max 201.
- Naive baselines assuming KNOWN history boxes: copy-t3 -> 66.6% clips IoU>=0.5 (mean IoU 0.594); linear t2->t3->t4 -> 71.9% (mean IoU 0.637). Linear extrapolation is the ceiling for a detect-then-extrapolate pipeline.

## Key challenge: on TEST there are NO history boxes -> must DETECT object in t0-t3 from noisy sparse frames, then forecast, then classify.

## Goal: MCFS 0.4+

## Progress log
- Classical single-frame detection (density/erode/coherence) all stuck at ~22% IoU>=0.5; camera-motion background dominates. Detection needs to be learned + multi-frame.
- proto_cnn.py (soft-argmax localization): localization would not train (weak heatmap peaks). Abandoned.
- proto2.py (grid cell-classification detector, softmax-CE over stride-4 grid, single positive cell/frame; heads for t0-t4 heatmap+offset+size+class): localization trains. At ep6 detIoU(t0-3)=0.26 and climbing, clsAcc 0.65. t4 via track extrapolation (lin2/linfit) blended with direct t4 head; blend/strategy tuned on val each eval. Full 70-epoch run in progress.
- Runtime note: ~67s/epoch at 320x180 on 10 threads; will need speedups (lower res / fewer epochs) to fit grader 1.5h with margin once converged.

## Approaches to try (compare on local val, macro-MCFS aligned)
- A (classical): density-based cluster detection per frame (parse red/blue dots, robust to camera-motion noise) + trajectory extrapolation -> t4 box; geometry+appearance classifier (sklearn).
- B (learned end-to-end): CNN on stacked t0-t3 motion channels -> t4 box regression + class softmax, with geometric augmentation (skill Geometric/Motion recipe). torch CPU.
- C (hybrid): classical detection for history boxes + learned forecaster (trajectory features) + CNN classifier on object crops.

## Env
- Remote sweep box: EC2 i-0c0784a14ee583064, 34.227.176.167, 16 vCPU / 124GB, Amazon Linux 2023.
- venv ~/venv (py3.9) has numpy/pandas/scipy/sklearn; installing pillow/opencv/torch(cpu).
- Working disk /mnt/work (278GB). Dataset -> /mnt/work/eris/dataset/public/.
- MUST keep final solution within grader budget: 10 cores, 62GB, 1.5h.

---

# QUEUED NEXT: ParseRift (parser attachment disagreement)
- Dataset: G:\Datacurve\cpu-challenges\parser attatchment disagreement\dataset
- Task: token-level binary classification. For each token in an English sentence, predict whether 25 independent dependency parsers disagree on its head (contested=1) beyond a 15% margin.
- Data: train.parquet 20,624 tokens (grouped into sentences), test.parquet 4,145 (contested withheld). Columns: token_id, sentence_id, position, token, contested. token_id/sentence_id/position are OPAQUE (carry no signal -> using them scores 0).
- Metric: MCC (Matthews Correlation Coefficient), clip to [0,1]. ~23% contested (imbalanced). Constant/random/id-only all score 0.
- From scratch on CPU, no internet, no external parsers/taggers/treebanks, no pretrained weights/embeddings. No hardcoded per-token_id answers.
- Reference frontier: context model->linear 0.218; char-aware BiLSTM (reference) 0.258 (varies 0.239-0.277 across seeds). Shortcut lookups <=0.15.
- Goal: MCC 0.32+ (above reference frontier; ambitious).
- Productive paths: sentence context (not token in isolation), char/subword features for rare words, calibrate decision threshold for MCC on held-out data.

# nnseq family — joint-route neural sequence tagger

Deliverables (canonical order = train.csv row order, masked_nodes order, classes=common.TYPES):
- oof_probs.npy (3938x5), test_probs.npy (1583x5) = arithmetic mean of gru5 + mlp5 (50 models total)
- score.json: viterbi 52.72 / posterior 52.48 (raw argmax macro-F1 0.473)

## Reproduce
    OMP_NUM_THREADS=5 python train.py --seeds 5 --epochs 60 --seq gru --tag gru5
    OMP_NUM_THREADS=5 python train.py --seeds 5 --epochs 60 --seq mlp --tag mlp5
    # ensemble: (oof_probs_gru5 + oof_probs_mlp5)/2 -> oof_probs.npy ; same for test
    python ~/discourse/foundation/eval_probs.py oof_probs.npy --mode viterbi --json score.json

## Model (train.py + featlib.py)
Per node: 79 dense engineered feats (standardized per-fold) = topology, neighbor
type counts+fracs (kid/desc/sib/between/vis), gaps, view stats, title/profile flags,
strong-signal booleans (has_answer_kid, n_kids_vis==0, ...). Embeddings: par_type(8),
par_kind(4), depth-bucket, (pos,route_len), role. Title + forum encoded ONCE per row
via hashed char 3-4gram (forum 3-5gram) EmbeddingBag, broadcast to nodes.
Seq layer: BiGRU(hidden 96) across L in {3,4}; length-bucketed batches (no padding).
Head: per-node Linear->5. Loss: CE, sqrt-inverse-freq class weights, 1.8x anchor
weight, label smoothing 0.05. AdamW lr2e-3 cosine, dropout 0.3 + feat-dropout 0.1,
wd 1e-4, grad-clip 3. Early stop on carved 15% inner-val macro-F1 (patience 12).
5-fold OOF via common.make_folds; test = mean over folds x seeds.

## Findings / HPO (2-seed sweep, posterior OOF)
- Text (title+forum char-ngram) HELPS: notext 52.55 vs with-text ~53.1 (+0.5-0.7).
- class weights sqrt-inv-freq >> balanced (52.15). Balanced over-predicts rare classes.
- seq: gru approx mlp (per-node, no cross-pos) > transformer. The decode Viterbi
  already models transitions, so cross-position layer adds little -> per-node prob
  QUALITY is what matters; gru+mlp ensemble is the robust pick.
- anchor_w 1.2-1.8 all fine; 2.5 slightly worse. hidden 128 / dropout 0.4 no gain.
- DECODE-SCORE IS NOISY across seed sets (52.4-53.2 for equivalent configs) with only
  1250 rows; raw argmax macro-F1 is the stabler quality signal (rose to 0.473 w/ 50
  models). Ensemble chosen for lowest variance + smallest viterbi/posterior gap (0.24).
- viterbi ~= posterior here (52.72 vs 52.48); reviewer picks mode + tunes multipliers.
- Weakest classes: elaboration (type F1 .43, anchor .28) and appreciation/agreement
  anchors — main remaining headroom.

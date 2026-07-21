# ParseRift (parser attachment disagreement): notes

## Task
Token-level binary classification: for each token predict whether 25 independent dependency parsers
disagree on its head (contested=1) beyond a 15% margin. Metric = MCC (clip to [0,1]). From scratch,
CPU, no internet/external parsers/embeddings. token_id/sentence_id/position are opaque (no signal).

## Data (dataset/public/)
- train.parquet 20,624 tokens (1752 sentences), test.parquet 4,145 tokens (314 sentences).
- Columns: token_id, sentence_id, position, token, contested (train only).
- contested rate 23.4%. ~12 tokens/sentence (max 79).
- Top contested tokens: punctuation ($ / ) - ( , : " ?) and words (what, as, one). Least: pronouns/aux (i, we, would, 's).
- ~45% of test tokens are OOV (839/1522 overlap) -> char features essential.

## Reference frontier: context->linear 0.218; char-aware BiLSTM 0.258 (varies .239-.277). Goal: 0.32+.

## Approach (solution.py, self-contained)
- Char-aware BiLSTM token tagger: word emb (UNK for rare) + char-CNN (kernels 3/4/5 over token chars, handles OOV/morphology) + shape feats (case, punct, digit, length) -> 2-layer BiLSTM over the sentence (context is the productive signal) -> per-token logit.
- Loss BCE with pos_weight for imbalance. Ensemble of 3 seeds (avg probs). Threshold calibrated for MCC on a 15% held-out sentence split, then ensemble retrained on all data for the final prediction.
- Reads dataset/public, writes working/submission.csv (+ optional out path arg).

## Results
- Prototype (single model, 16 ep): val MCC 0.344 (already > 0.32 goal, > 0.258 reference).
- Smoke (1 seed, 4 ep): val MCC 0.329.
- Full ensemble run in progress -> expect ~0.34-0.36. Fast (~15 min).

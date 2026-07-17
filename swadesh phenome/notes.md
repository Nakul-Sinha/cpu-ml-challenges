# Swadesh Phoneme Cipher Decoding — notes

## Task
Recover a global bijection `sigma: token -> IPA segment` for one held-out **Uralic** target
language whose IPA was enciphered (each segment -> one opaque `x<int>` token). Score = mean
segment-level normalised edit similarity between decoded and true pronunciations. AI baseline 0.62.

## Key dataset facts (EDA)
- train.csv: 120,422 rows, 106 languages, 21 families, 48 subfamilies, 1016 concepts.
- Uralic crib = 25 languages / 9 subfamilies (Finnic 5, Saami 6, Samoyedic 4, Permian 3,
  Mari 2, Mordvin 2, Hungarian 1, Mansic 1, Khantyic 1).
- test.csv: 1189 enciphered words, all 1016 concepts also in train, **70 distinct tokens**.
- Token frequency is skewed: top-10 tokens = 51% of occurrences, top-30 = 92%, top-40 = 97%.
  => getting the frequent tokens right dominates the score.
- **Target is almost certainly Saami**: only Saami languages have inventories >= 67
  (sms 121, sjd 93, sme 71, smn 67); every non-Saami Uralic language has <= 62. The target's
  70-segment inventory matches the Saami range. Its closest crib relatives are the 6 Saami langs.

## Method
Statistical decipherment by cognate alignment + global one-to-one assignment:
1. Candidate segment inventory = Uralic segments seen in >= `seg_min_langs` languages.
2. Monotonic (Needleman-Wunsch) alignment of each enciphered target word to each same-concept
   crib form; accumulate soft co-occurrence counts C[token, seg], weighted by
   length-similarity (cognate proxy) x language-relatedness.
3. M-step: PMI(C) corrects for segment frequency (t/k/a co-occur with everything).
4. Assignment: Hungarian on `PMI + freq_prior*log P(seg)` -> strict bijection. The frequency
   prior is essential: pure PMI favours rare marked variants (nʲː, ɕʲ) when the truth is the
   plain common segment (n, h). EM-damped counts + strong gap penalty keep it stable.
5. Language relatedness re-estimated each iteration from decoded similarity (surfaces the
   closest relatives, e.g. Saami for a Saami target).

## Validation (honest, no test labels)
Leave-one-Uralic-language-out: encipher a known Uralic language with a random bijection,
recover it from the other relatives, score vs truth. Saami folds (~70 segs) best mirror the
real target. Files: cv.py (serial), cv_parallel.py (16-core box), inspect_fold.py (per-token).

## Progress log (single-fold, seed 0)
- Positional-PMI baseline (equal-length only): fin 0.20, sme 0.05.
- EM raw-count assignment (bug): ~0.00 (frequent segments dominate raw counts).
- EM + PMI assignment, hard sim gate: collapses/oscillates.
- EM + PMI + stable length/relatedness weights + strong gap + damping: fin 0.53-0.62, sme 0.21.
- **+ frequency prior in assignment: sme 0.21 -> 0.54** (top-10 token acc 0.30 -> 0.60).

## Open levers to push higher
- Tune freq_prior / gap / align_scale / seg_min_langs / rel_pow on full CV (box sweep).
- Multiple-relative consensus per concept (denoise alignment columns).
- Phonetic-feature prior to merge near-identical variants and place rare tokens.
- Pick EM iteration by an unsupervised objective (alignment mass) rather than fixed n_iter.

## Compliance
Unsupervised; derived only from provided files. No external lookups, no hardcoded answers.

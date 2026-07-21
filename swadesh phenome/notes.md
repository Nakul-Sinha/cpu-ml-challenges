# Swadesh Phoneme Cipher Decoding: notes

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
- **Target is FINNIC** (corrected from an initial Saami guess based only on inventory size).
  Evidence: decoding self-consistency is far higher against a Finnic crib (obj 0.38-0.40) than
  a Saami crib (0.27); relatedness locks onto ekk (Estonian, wl=1.0) then vep/krl/fin/olo; and
  decoded words are recognisable Finnic core vocabulary -- `m ɑː rː a`=marja (berry),
  `i k ʃ t u ei s t e ɲ`=üksteist (eleven), `s uː ɔ r m e k s`=sormus (ring). The large
  70-segment inventory (bigger than any standard Finnic in train, max vep 56) points to a
  divergent large-inventory Finnic such as South Estonian (Võro/Seto) or Livonian, whose extra
  palatalisation + length degrees inflate the inventory. Finnic folds score high in CV
  (fin 0.85, vep 0.91, krl/olo 0.96, ekk 0.66), so the outlook is good.
- **Honest score estimate via calibration**: the self-consistency objective is computable on
  the real test (0.404). Calibrating obj->true-score across CV folds gives a point estimate for
  the hidden target without ever using its labels.

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

## Locked configuration & score estimate
Final config (leave-one-Uralic-out CV, best Finnic + overall): rel_pow 3, freq_prior 0.7,
cog_floor 0.5, seg_min_langs 2, gap -6, align_scale 0.5, damp 0.5, n_iter 18.
- CV: mean_all 0.657, mean_finnic 0.895, ekk (most divergent Finnic, best proxy) 0.679.
- Real target self-consistency obj 0.415; obj->score calibration (r=0.90-0.94) and the ekk
  proxy put the expected real score around 0.63-0.68. The real run also has all 5 Finnic
  relatives available (the CV ekk fold holds ekk out), which should help further.
- Config selected by CV (Finnic + overall); cross-checked by maximising the real target's
  self-consistency (an unsupervised criterion), which agreed on cog_floor 0.5 and rel_pow ~2-3.

## Round 2 (phonetic prior)
- **Vowel/consonant class prior in the assignment (WIN, shipped, vc_weight=1.0):** a token that
  aligns mostly to vowels decodes to a vowel. Eliminated all cross-class errors on ekk.
  ekk 0.679 -> 0.721; real-target self-consistency 0.415 -> 0.424 (peak vc=1.0, deterministic in
  the encipherment seed). After this, the remaining ekk errors are ALL same-class fine variants
  (d̥/d, nʲ/ɲ, e/ɛ, u/ʊ, ɑː/a) -- partly transcription-convention mismatches, near the floor.
- **Rejected after honest validation (kept for the record):**
  - Finer phonetic-feature prior (place/manner/height/backness): hurt (ekk 0.721 -> 0.681) --
    the features are too noisy and fight the accurate PMI. Binary V/C is the clean signal.
  - freq_prior > 0.7: hurt (ekk 0.721 -> 0.632). seg_min_langs 1: hurt (real obj 0.424 -> 0.403,
    too many noisy candidates); 3: noise-level. Kept freq_prior 0.7, seg_min_langs 2.
- Final config: rel_pow 3, freq_prior 0.7, cog_floor 0.5, seg_min_langs 2, vc_weight 1.0,
  gap -6, align_scale 0.5, damp 0.5, n_iter 18. CV mean 0.645, Finnic 0.896, ekk 0.721.
  Estimate for the real (divergent Finnic, all 5 Finnic relatives available) ~0.70-0.75.

## Compliance
Unsupervised; derived only from provided files. No external lookups, no hardcoded answers.

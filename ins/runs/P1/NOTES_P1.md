# P1 — Detector Upgrades on the N3 base (FINAL iteration)

Headline (honest nested CV, canonical elru, leak-free per fold, group-vote de+en):

    N3 base            nested 0.5503   (de .4237  en .8067  it .4205)
    P1 SHIP            nested 0.5706   (de .4808  en .8067  it .4243)   +0.0203, BEATS AI baseline 0.56
    P1 (a,thr) ref     nested 0.5725   (de .4865  en .8067  it .4243)   upper reference (2-DOF de)

Non-nested SHIP 0.5865. Canonical elru.score_frames on the 1259 OOF rows = 0.5706 (all rows valid).
Both submissions/OOF are drop-in on the SAME folds.csv + elru.py as N3.

## The two levers (measured each alone, then combined)

### Lever 2 — BiGRU sequence tagger (the big win, de)  [p1_lever2.py]
Small bidirectional GRU over each row's token sequence.  Per-token inputs:
hashed char-ngram (1..3) token embedding (vocab-free -> survives the cipher, mean-pooled),
language embedding, and the pipeline lexicon-rate scalars (tok/suf3/suf4/pre3/specsuf) +
structural flags + position.  Trained PER FOLD (leak-free OOF), sequences featurized with
the row's OWN fold-out lexicon.  **5-seed ensemble** per fold (averaging the OOF probs;
pure variance reduction, no extra tuning DOF — this was worth ~+0.015 de PR-AUC and
stabilised the downstream selection).  Ensembled prob = (1-a)*shared_LGBM + a*BiGRU.

Token PR-AUC (average precision):
    de  shared 0.1938 -> GRU 0.2799   (+0.086 — sequence context is decisive for de)
    en  shared 0.8290 -> GRU 0.8969
    it  shared 0.2933 -> GRU 0.3370
de nested lang 0.4237 -> **0.4808** (a=0.6 fixed, thr nested) / 0.4865 (a,thr both nested).
de unchanged-row FP 80/216 -> 57/216 (robust) / 49/216 (a,thr): MORE precise AND higher recall.
de per-type recall: multi_marked .431 (lm .204), multi_plain .575 (lm .299), single_marked .674.
Ship config: de a **fixed 0.6** (pre-committed, not CV-maximised — the nested selector always
prefers 0.6-0.75; 0.6 is the conservative end), spine thr nested (0.19-0.35 band, low variance).
The (a,thr) 2-DOF nested variant scores 0.4865 but carries selection variance across seed counts
(3-seed run peaked 0.5767 overall) — reported as an upper reference, not the headline.

### Lever 1 — IT-only LGBM re-scorer (small honest it win)  [p1_lever1.py]
Second LightGBM trained ONLY on it tokens, it-specific features: morphological ending
edit-rates (1/2/3 char, learned per fold), article-anchor distances both directions,
agreement-chain (high-suf2 run length + window count), NP-context flags, group slash-density,
hashed char-ngrams of token+neighbours (categorical), position.  Cross-fit per fold.
it token PR-AUC shared 0.2933 -> re-scorer 0.3174 -> blend 0.349 (w0.5).
Integration that transfers = **additive boost** into the fixed NP-gate spine:
    boosted = clip(shared + w*clip(p_it-0.3, 0, None), 0, 1),  w nested per fold (~0.4-0.6)
it nested 0.4205 -> **0.4243** (+0.0039).  Convex blend and a stacked re-scorer (shared-prob
features) do NOT transfer at the fixed spine — MEASURED, dropped.  The independent view is what
adds complementary signal.

it-source pick (measured): even though the GRU is a slightly better it RANKER (PR-AUC .337 vs
.317), the downstream it-boost consistently prefers the **LGBM re-scorer** (richer it morphology),
selected nested per fold.

## MEASURE-AND-DROP (do not redo)
* en GRU: raises en PR-AUC (.829->.897) but the nested selector picks a=0 — en ELRU is
  replacement/budget-bound, not detection-bound.  en FROZEN at N3.
* it convex blend (spine fixed): nested delta -0.0005.  it stacked re-scorer boost: ~0.
* Joint (a,thr) de selection: honest but higher-variance than a-fixed; kept as upper ref only.

## Ablation ladder (nested / nonnested)
    N3 base                    0.5503 / 0.5617
    +de BiGRU (a=.6 robust)    0.5693 / 0.5844
    +de BiGRU (a,thr nest)     0.5712 / 0.5840
    +it re-scorer              0.5516 / 0.5638
    +both SHIP (de a=.6)       0.5706 / 0.5865   <-- HEADLINE
    +both ref (de a,thr nest)  0.5725 / 0.5860

## Productionize recipe (test submission — NOT yet built; clear next step)
Full-train path mirrors pipeline_final.ship() with two additions:
  1. Train the 5-seed BiGRU on ALL train (per seed: one model on all rows) -> seq_full per test row.
     de test prob = 0.6*shared_full + 0.4*... NO: ensembled = (1-0.6)*shared + 0.6*seq? -> a=0.6 means
     0.4*shared + 0.6*BiGRU.  de spine thr = the ALL-OOF non-nested best (recompute from cache_by_a[0.6]).
  2. it re-scorer on all train -> p_it_full; boost w = it_nn_w non-nested (rescorer, 0.6).
  3. en unchanged (shared, thr 0.39); group-vote de+en hi.6/lo.4.
TRANSFER CAVEAT: de spine thr on the GRU-ensembled prob is ~0.30 (vs 0.07 shared-only) because the
ensembled prob is peakier/better-calibrated.  de FP DROPPED (80->57) so precision improved — the
de edited-row ratio should stay <= N3's; still VERIFY the submission de edited-ratio stays in
[0.45,1.80] before shipping (as N3 did; a robust de thr can be nudged up if over-predicting).
Runtime: GRU full-train = 5 seeds x ~4s = ~20s; fits the 60-min CPU grader easily.

## Files (box: ~/insled/runs/P1/, mirror: G:/Datacurve/cpu-challenges/ins/runs/P1/)
    p1_base.py      fold fit + N3 reproduction (nested 0.5503 exact)
    p1_lever1.py    it re-scorer (rescorer_oof) + full Lever-1 ablation
    p1_lever2.py    BiGRU tagger (gru_oof, n_seeds) + full Lever-2 ablation
    p1_combined.py  ladder + report (P1_GRU_SEEDS env; default 1, ship=5)
    cv_report_p1.json, oof_edits_p1.csv   deliverables (headline 0.5706)

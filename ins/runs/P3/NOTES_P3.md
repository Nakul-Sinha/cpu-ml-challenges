# P3 — FINAL COMPOSER + PRODUCTIONIZER (Institutional Edit Ledger Recovery, iter 4)

## Headline (honest nested CV, canonical elru, leak-free per fold, group-vote de+en)

    N3 base            nested 0.5503   (de .4237  en .8067  it .4205)
    +P1                nested 0.5706   (de .4808  en .8067  it .4243)
    +P2                nested 0.5504   (de .4237  en .8067  it .4209)
    +P1+P2 JOINT       nested 0.5707   (de .4808  en .8067  it .4246)   <-- HEADLINE
    +P1+P2 ref(a,thr)  nested 0.5726   (upper reference, higher variance)

Non-nested JOINT 0.5866.  canonical elru.score_frames on 1259 OOF rows = **0.5707**
(nested) / 0.5866 (non-nested); all 1259 rows valid.  **+0.0204 over N3, BEATS the AI
baseline 0.56.**  Reproduces exactly on re-run (deterministic).

## What each component contributed (and whether it survived)

* **P1 Lever 2 — de BiGRU ensemble (the decisive lever).**  Per-token prob
  `(1-a)*shared_LGBM + a*BiGRU`, a=0.6.  de token PR-AUC .194->.280; de nested
  lang .4237->.4808 (+0.057); de unchanged-row FP 80/216->57/216 (more precise AND
  higher recall).  KEPT.
* **P1 Lever 1 — IT-only LGBM re-scorer (small honest it lever).**  Additive boost
  `clip(shared + w*clip(p_it-.3,0,None),0,1)`, w=0.6, into the fixed NP-gate spine.
  it token PR-AUC .293->.317; it nested .4205->.4243.  KEPT.
* **en — FROZEN (measure-and-drop).**  BiGRU raises en PR-AUC .829->.897 but the
  selector picks a=0 downstream (en is replacement/budget-bound).  en byte-identical
  to N3.  Confirmed again in v4.
* **P2 — IT-only enhanced transducer (marginal, non-regressing).**  Multi-token
  decomposition + append (slash-doubling agreement) rules; ENHANCE_LANGS=(it,) so
  de/en are BYTE-IDENTICAL.  Joint it nested .4243->.4246 (+0.0003), overall
  0.5706->0.5707 (+0.0001).  Within noise but strictly non-regressing and adds a real
  multi-token it agreement capability -> **KEPT** (zero downside).

## SHIP decision: CV-optimal vs robust — the iter-3 finding does NOT transfer

The iter-3 rule "robust = higher de threshold, safer AND nested-higher" was compensating
de OVER-prediction on the shared prob.  With the BiGRU the de prob is peakier and de FP
dropped 80->57, so **de now UNDER-predicts** (submission edited-ratio 0.79-0.95 across all
thresholds, i.e. below the train rate).  A higher "robust" threshold now only cuts de
recall — it is neither safer nor nested-higher:

    SHIP-FIXED honest nested (de fixed a-priori, en nested, it fixed rescorer0.6+P2, vote)
      de_thr 0.15 -> 0.5789   0.19 -> 0.5803   0.25 -> 0.5781
      de_thr 0.29 -> 0.5777   0.31 -> 0.5777   0.35 -> 0.5756     (flat, all >> 0.56)

**SHIPPED (submission_final.csv): de_thr = 0.31** — the MEDIAN of the per-fold nested
picks [.19,.31,.29,.35,.31].  A blind, pre-committed rule (no peeking at the scan above)
-> zero selection-optimism on the threshold; ship-fixed honest nested 0.5777; de edited-
ratio 0.86, comfortably in band.  **submission_cvopt.csv: de_thr = 0.19** (the non-nested
all-OOF optimum; de edited-ratio 0.95) is provided as the CV-optimal alternate.

## Shipped operating points (baked hyperparameters, all selected by honest CV above)

    de : a=0.6 BiGRU ensemble, spine thr 0.31 (median-nested, robust)
    en : FROZEN (a=0, shared prob), spine thr 0.39
    it : base-merge spine 0.45 UNION NP-gate 0.8, IT re-scorer boost w=0.6, P2 transducer
    group-vote : de+en, hi 0.60 / lo 0.40   (it not voted)

## submissions (445 rows each, canonical-validated, 0 invalid, <=6 edits/row)

    submission_final.csv  (de@0.31)  de 64/129 r=0.86  en 34/117 r=0.62  it 154/199 r=1.10  (252 edited)
    submission_cvopt.csv  (de@0.19)  de 71/129 r=0.95  en 34/117 r=0.62  it 154/199 r=1.10  (259 edited)

All edited-row ratios in [0.45,1.80].  (differ in 28 rows = de threshold + vote propagation.)

## solution.py — self-contained, deterministic, scorer-free

ONE file (~4030 lines).  The 8 base modules (pipeline, transducer[=P2], m2/m3/m4/n2_ext,
run_m4, run_n1) are embedded VERBATIM as readable r'''...''' blocks and exec'd into
synthetic modules in dependency order, so their normal cross-imports resolve with NO
import from runs/.  elru is a scorer-free shim (validate_edits only).  P1 levers (BiGRU +
IT re-scorer) + the ship orchestration are hand-written runtime code.  Every model is fit
on train.csv AT RUNTIME.

* `python3 solution.py [public_dir] [submission_out]` — argv + autodetect (dataset/public,
  dataset, ., /kaggle/input, walk); default out working/submission.csv; parent mkdir;
  keep_default_na=False; fixed threads + seeds (byte-deterministic); wall-clock guard 3000s;
  strict pre- AND post-write validation; loud all-empty fallback ONLY if the main path throws.
* Runtime ~82s (Xeon, OMP 7).  BiGRU 5-seed full-train ~30s; well within the 60-min grader.

### Isolated smoke test (only dataset csvs + solution.py in /tmp/ship_test) — BOTH pass
    argv  : solution.py dataset/public out/submission.csv  -> EXACT match to submission_final.csv
    no-arg: solution.py (autodetect -> working/submission.csv) -> EXACT match to submission_final.csv
Re-run determinism: two independent full runs are byte-identical.

### Compliance (grep-verified)
no tfidf (only negated in docstring); no network; sample_submission never read; no hardcoded
answer arrays / encoded content strings; all models (LGBM detector, P2 transducer, BiGRU,
IT re-scorer, NP gate) fit at runtime.  The embedded modules' reranker / load_train(folds.csv)
/ main() are dead code (never on the ship path — proven: the isolated smoke test has no
solution/folds.csv and still matches exactly).

## artifacts (box: ~/insled/runs/P3/, mirror: G:/ml/cpu-challenges/ins/runs/P3/)
    solution.py            self-contained ship (SHIP = de@0.31)
    submission_final.csv   shipped (de@0.31 robust)      submission_cvopt.csv  alt (de@0.19)
    pipeline_v4.py         honest ladder + ship-fixed scan (import-based, reproduces above)
    cv_report_v4.json      headline 0.5707 + full ladder + ops + ship-fixed scan + PR-AUC
    oof_edits_v4.csv       1259 OOF rows (canonical = 0.5707)
    _runtime.py            ship runtime (levers + assembly)   build_solution.py  generator
    gen_cvopt.py           cvopt alternate generator          probe_de.py        de-thr probe

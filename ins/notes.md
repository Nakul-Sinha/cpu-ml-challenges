# Institutional Edit Ledger Recovery: notes

## Task facts
- Recover ordered edit ledger [{start,end,replacement}] per encoded text window. ≤8 edits, sorted, non-overlapping, rep ≤160 chars.
- train 1259 rows / 70 groups; test 445 rows / 22 held-out groups. Languages: de (511/129), it (614/199), en (134/117) train/test. Groups are single-language.
- Metric ELRU: pair_quality = 0.45*span_char_F1 + 0.45*replacement_chrF(n1..3) + 0.10*type_match; greedy match; /max(nT,nP); row = 0.85*matched + 0.15*budget_fidelity; language = 0.75*edited_rows + 0.25*unchanged_rows; mean over 3 langs. Both empty=1, one empty=0.
- AI baseline: 0.56. No-op = 0.25 exactly.

## Decoded structure (EDA)
- Task is inclusive-language rewriting on privacy-encoded text. Letter mapping consistent per language across train+test; punctuation/case/whitespace unchanged.
- de: "X:in"→"X oder Xin" (':xà'→' pwéu '), "X*in" same, "X:innen"→"X und Xinnen", paired "Xinnen und Xen"→neutral ("Studierende"-type participles, "Professuren"), slash forms, des→des/der ('wév'→'wév/wéu').
- it: article doubling del→del/della ('uyh'→'uyh/uyhhl' OR 'uyhhl/uyh', per-document convention!), o→o/a suffix ('g'→'g/l'), full neutral rewrites.
- en: his/her→their ('fuy / flv'→'dfluv'), he→they, Chairman→Chair ('Pfouvqok'→'Pfouv').
- ALL 1365 edits align exactly with whitespace-token boundaries. Tokens/span: 1:774, 2:317, 3:193, 4:42, ≥5:39.
- Trigger tokens with punct (colon/star/slash) have edit-rate ≈1.0. Plain tokens are context-dependent (it articles rate 0.1-0.3, edited only in gendered NPs).
- Windows overlap within groups (60-char tail shared: 690/1259). Deterministic per-token model gives automatic cross-window consistency.
- Cross-group surface memorization transfers poorly (probe: CV 0.30-0.37; edited-mean de/it only 0.03-0.12), stems differ per document. MUST generalize at pattern level (suffix/punct morphology + stem-abstracted templates).

## Canonical foundation (all runs comparable)
- solution/elru.py, exact scorer. solution/folds.csv, 5-fold grouped split (language-balanced). Both verified identical local & Box 2.
- Probes: noop 0.2500; memo(t0.5,s1) 0.3715 CV.

## Architecture plan (v1)
1. Detection: token-level BIO tagger. LGBM on char/context/morph features vs char-BiGRU; per-language threshold tuned on CV for ELRU directly.
2. Transduction: learned hierarchy, (a) exact (lang,src)→majority-rep memory fit from train; (b) char-alignment template induction (stem-abstracted, e.g. VAR+':xà' → VAR+' pwéu '+VAR+'xà') with language-level convention stats; (c) char seq2seq (small GRU) fallback + scorer. All fit in-script from train.csv only.
3. Assembly: BIO spans, ≤8, ELRU-tuned confidence gates.

## Compliance
- No tfidf/idf. No hardcoded replacement dictionary as a substitute for learning (memory/templates are FIT from train.csv at runtime). Real ML: trained tagger + trained transducer. No test-time adaptation. Runtime trivial vs 1.5h CPU budget.

## Box assignment
Box 2 (Xeon SPR 61GB, ~/insled), small data, fast iteration; grader-parity runtime checks.

## Log
- [x] EDA, scorer, folds, probes (local + Box 2 parity)
- [x] iter1 (wf_d43c8a3b): A1 detector oracle-rep ELRU 0.5276 (thr de .08/en .30/it .44); A2 transducer oracle-span rep chrF macro 0.6608 (identity floor 0.511); A3 e2e CV 0.4760 + valid submission_v1. Review VERIFIED all numbers; composed A1+A2 = CV 0.5061; perfect-detection ceiling 0.8989 → detection is the bottleneck.
  - Loss map: unchanged-row FPs zero 75/216 de + 58/182 it rows (~-0.055); recall by type: de paired multi_plain .299, de multi_marked .204, de single_plain .092, it single_plain .378 (biggest), it multi .508, it marked ~0; en solved except single_marked .385.
  - Negative results: row-level prob gate hurts; bigger LGBM no help; GRU seq2seq generator 0.35 < templates 0.66; span-text deletion classifier precision ≤0.15 (disabled).
  - Defects noted: U+2217 mark charset, whitespace-preserving joins, nested thresholds.
- [x] iter2 (wf_6967ecef): M1 merge non-nested 0.5102 / NESTED 0.4966 (defects fixed: U+2217, whitespace joins; plug-in points). M2 de paired-form connector-bridge generator: de multi_plain recall .142→.604, nested +0.007. M3 it article/ending/agreement features: nested +0.010; deletion forfeit measured (oracle +0.0123, precision 0 → forfeit). M4 integrated + group-vote(de,en; hi .6/lo .4): **NESTED 0.5423 / non-nested 0.5517** (de .409/en .807/it .411 nested). submission_v2 valid (292 edited rows). Span reranker measured NEGATIVE (not shipped).
  - Open buckets: it single_plain .322 (biggest), it multi_plain .336 (REGRESSED from .508, needs diagnosis), de multi_marked .138, de single_plain .047, deletions forfeited.
  - Risks: en test slash-shift materializing (test en edit-rate ratio 0.62 vs train); de overprediction (ratio 1.44 @ thr .07).
- [x] iter3 (wf_c2ce44d2): N1 it NP-gate generator (+.0096 it nested; diagnosed multi_plain regression = detection collapse; doc-prior/ending-gate/reranker all measured negative, all it groups edit-active, TP/FP lexically identical). N2 de markrun generator (multi_marked .138→.394) + masc_only collapse fallback (chrF .699 vs identity .474) + de robustness curve (thr .10 ≈ free, ratio 1.29) + EN SHIFT PROVEN RESOLVED (zeroing all slash features changes 0/2208 en tokens, en is plain-lexicon driven). N3 integrated: **nested 0.5503 CV-opt / 0.5534 ROBUST (de@0.11)**; non-nested 0.5617; zero added FPs; robust variant BOTH safer and nested-higher → upload candidate. Honest nested still < 0.56.
- [ ] iter4 (wf_dbd3a965): P1 it-only re-scorer + BiGRU sequence-tagger ensemble ∥ P2 multi-token transform decomposition learning + it agreement composer + duplicate-adjacency deletion re-check → P3 final compose + self-contained solution.py + isolated smoke test
- [x] final deliverables (approach.md, submission handoff), PR #29 merged 2026-07-21

## Compliance fix (2026-07-21, post-rejection)
Platform rejected the shipped solution.py: "Prompt Compliance" and "Held-out Answer Ingestion" both
failed with "source that cannot be safely inspected". Root cause: the productionizer had packaged
9 historical dev modules (elru/transducer/pipeline/m2_ext/m3_ext/m4_ext/n2_ext/run_m4/run_n1) as raw
triple-quoted string literals in a `_MODS` dict and dynamically `exec(compile(...))`'d them into
synthetic `sys.modules` entries at import time so their cross-imports resolved. Behaviorally correct
(verified byte-identical at the time) but automated scanners correctly refuse dynamic exec-of-embedded-
strings as inspectable, and the ~2700 lines of embedded dead research code (old argparse dispatchers,
diagnostic mains, an unused folds.csv reader) looked evasive besides.

Fix: rewrote solution.py as genuinely flat code, no exec/compile/eval/__import__/marshal/pickle/
ModuleType/sys.modules trick anywhere, only the ship-path-reachable functions kept (86 top-level
defs, single `if __name__` guard), 4028 -> 2351 lines. Verified independently: isolated run on Box 2
(clean dir, only train.csv+test.csv+solution.py, no runs/ or solution/ folder) produced output
BYTE-IDENTICAL (diff -q) to the previously-shipped runs/P3/submission_final.csv; 82s wall clock;
edit-rate ratios unchanged (de 0.86/en 0.62/it 1.10); submission re-validated (445 rows, 0 invalid).
Nested CV score unchanged at 0.5707 (pure structural refactor, zero behavior change). Old flagged
version preserved at runs/P3/solution_flagged_v4.py for reference. Re-shipped via a follow-up PR.

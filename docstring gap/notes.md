# Docstring Gap Restoration — notes

## Task facts
- Fill [GAP] span in a docstring sentence given the sentence + function code (docstring removed). Free text output.
- train 231,973 rows; test 50,000; metric = char n-gram F (n=1..6, pooled P/R per spec reading), mean over rows. Score to beat: 0.53.
- Targets: 1-4 words (99.9%), median 12 chars, max 64. Mix of function words ("of the","the given") and code content words ("input image").
- 36 train rows have empty/NA-looking target — MUST read CSVs with keep_default_na=False.
- Runtime: CPU-only 10 cores 62GB, 1.5h hard.

## EDA signal
- 1.17% test masked sentences appear verbatim in train. ~10% of targets appear verbatim in code_context.
- Train/test drift ≈ none: anchored-context coverage val vs test ≈ identical (l1r1 0.504/0.512, l1 0.947/0.948); len distributions match.
- Probes (dedup holdout by masked_docstring hash): const "the" 0.090; best const 0.133; anchored l1/l1r1/l2r2 backoff retrieval 0.178.
- Tiny 22-cand pool: oracle 0.486, exact-hit 17% → rich pool + reranker has headroom well past 0.53.

## Canonical foundation
- solution/chrf.py — pooled n1..6 scorer (+macro variant). Holdout = md5(masked_docstring)%20==0 bucket (twin-safe).

## Architecture plan (v1)
1. Candidates: anchored context indexes (L2R2/L1R1/L1R2/L2R1/L1/R1/L2/R2, top-K each) + hashed char-ngram TF cosine retrieval (NO idf) over masked sentences + word n-gram LM bridge generation (left→right beam, 1-4 words) + code-derived (func-name split, args, return tokens) + global frequent spans.
2. Features: generator flags/ranks/counts, context-cond probabilities, LM bridge scores, retrieval sims, length priors, code-overlap, gap position.
3. Reranker: LGBM regression on y=chrF(cand,true), cross-fit (indexes exclude own row/bucket). Real ML core.
4. Decode: argmax + pool-MBR over top-10 (expected-chrF).
5. Optional: SGD/MLP span-class model as extra generator+feature.

## Compliance
- NO tfidf/BM25/idf anywhere (user ruling): hashed TF + count-conditional probabilities + n-gram LM only.
- Retrieval frozen is fine, reranker is genuinely trained (RAG rule). No test-time adaptation; no hidden metadata.
- Runtime governance: stage timers, skip/shrink knobs, hard safeguard ~75 min.

## Box assignment
Box 1 (EPYC 124GB, ~/docgap) — 232k-row indexes/features need RAM; final parity check on Box 2 later.

## Log
- [x] EDA, scorer, holdout, probes (local)
- [x] iter1 (wf_397ce140): B1 e2e retrieval+rerank+MBR 0.2802 (self-contained solution.py, 14 min@10t — shippable floor); B2 pool oracle 0.5866/0.5733-gated (exact-hit 22.8%; marginals: anchors +.126, code +.075, fuzzy +.046); B3 LM-bridge argmax 0.2611 oracle@10 0.448; NN probe weak + ANTI-calibrated confidence (never use as soft feature). Review VERIFIED all; union pool oracle 0.636; realization 54-58% → integrated est ~0.35.
  - Strategic: retrieval+rerank architecture CANNOT reach 0.53 (would need 83% realization); generative step-change needed.
  - Negative results: retrieval-posterior decoding ≪ reranker-posterior; NN P(class) anti-correlated with quality (corr -0.17); normalized anchor keys useless.
  - Optimism note: bucket-0 absorbed small sweeps (~0.005-0.01); bucket 1 now LOCKED as untouched holdout.
- [x] iter2 (wf_0ec731f6): C1 integrated union pool + 78-feat LambdaRank: **0.3210** bucket-0 (+0.041 over B1; union oracle 0.6302, realization 51%; MBR rejected +0.0016 < bar; NN excluded +0.0073 < bar); self-contained solution_v2.py ~20 min projected grader — verified valid fallback artifact. C2 t5-small probe: doc_first (code-hint) format is THE lever (FT 15min → 0.313 vs plain 0.223); int8 quant 1.35× at -0.011; seq_logprob usable confidence (corr 0.384); T5 two-way oracle vs C1 pick +0.10; flan-t5 worse (instruction tuning hurts). Review VERIFIED both; DECISION: hybrid (C1 chassis + gated in-script-FT T5 candidates + T5 features); expected bucket-0 0.345-0.365; conservative 81.5/90 min with stage-skip ladder; bucket-1 locked for final one-shot honest score.
- [x] iter3 (wf_7b9738b6): D1 t5-hybrid built (+0.0225 over C1; gate-calibration self-leak bug found+fixed via parity-fit weakness). D2 codet5-small probe: **ZERO-SHOT 0.4151** vs t5 zs 0.199 / t5-FT 0.313; FT at lr3e-4 REGRESSES codet5 (pretrained infill is the asset); int8 ~38 rows/s, delta -0.0045; codet5-base 3.6× too slow for budget. D3 SHIP: **codet5-small zero-shot learned-hybrid** (codet5 candidate for ALL rows — beats retrieval on every anchor tier — + trained LambdaRank fusion): bucket-0 0.4380, **LOCKED bucket-1 0.4392 (scored once; optimism gap -0.0012)**; realized 89% of two-way oracle 0.4918; hard overrides and codet5-primary policies all measured worse. Final timed solo run 52.4 min @ 10 threads (37.6 min margin); submission_v3.csv valid (50k rows, 0 empty).
  - Ship files staged: solution.py (md5 80df5a02...), submission.csv, working/submission.csv, approach.md.
  - 0.53 score-to-beat: honestly UNREACHED — established across 3 iterations that retrieval+rerank caps ~0.35 realization-bound, t5-small caps ~0.34 hybrid, and any generator big enough (codet5-base) blows the 90-min CPU budget on inference alone. Expected grader ~0.44.
  - Future levers (unbuilt): multi-candidate codet5 beam injection (raises oracle + realization), gentle-lr FT screen (3e-5), closing the 0.05 selection headroom.

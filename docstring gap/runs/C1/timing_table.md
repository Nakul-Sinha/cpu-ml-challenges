# C1 Integration: results & timing

## Headline (full bucket-0, 11,462 rows; reranker trained on 58k fold rows)
| decode | chrF (pooled char n-gram F, canonical) |
|---|---|
| **argmax (shipped)** | **0.3210** |
| MBR T=0.4 | 0.3226  (+0.0016, below +0.003 keep-threshold → not shipped) |
| MBR T=0.6 | 0.3204 |
| pool oracle (ceiling) | 0.6302 |
| realized / oracle | 0.509 |
| exact-hit in pool | 0.311 |
| mean pool size | 67.5 |

Baseline to beat: 0.53 (grader).  Prior best (B1 reranker alone): 0.2802.
Integration lift over B1: **+0.041** (0.2802 → 0.3210).

NN candidate pool: EXCLUDED, union-oracle delta +0.0073 on a 3k probe (< the +0.01 bar).

Reranker feature importance (gain, top): b2src_code, centrality8, lm_total_pw,
lm_rank, lm_first_logp, char_len, masked_len, lm_mean_logp, lm_leave, word_len,
src_code, glob_freq_log. → the B2 provenance + B3 LM + centrality features (the new
ones) dominate; centrality8 already captures the MBR signal, which is why explicit
MBR adds nothing above the noise floor.

## Bucket-0 EVAL timing (integrated.py, 7 threads, box=16-core EPYC)
| stage | seconds |
|---|---|
| parity fits (B1 idx+LM, B2 PoolBuilder, B3 LMBridge on even+odd fold halves) | 62 |
| train candgen + featurize (58k rows, 2 parity phases, ~78 feats) | 607 |
| reranker train (LightGBM LambdaRank, 177 iters) | 99 |
| full-fold fits | 64 |
| val candgen + featurize (11,462 rows) | 150 |
| decode argmax + MBR sweep (3 temps, single-threaded) | ~460 |
| **TOTAL** | **1443 (24.1 min)** |

Note: the ~460s decode block is dominated by the 3-temp MBR sweep, which the SHIPPED
solution (argmax only) does not run.

## Full solution_v2.py TEST run (train 231,973 → submission on 50,000 test rows)
7 threads on the dev box (shared with sibling agents, some contention):
| stage | seconds |
|---|---|
| parity fits (full-train halves, even 115,955 + odd 116,018) | 68 |
| train candgen + featurize (58k rows, 2 parity phases) | 584 |
| reranker train (LambdaRank, 345 iters, ndcg@1 0.4406) | 157 |
| full-train fits (231,973) | 64 |
| test candgen + featurize + argmax decode (50,000) | 776 |
| **TOTAL (7 threads)** | **1655 (27.6 min)** |

submission_v2.csv verified: 50,000 rows, cols [id, prediction], 0 empty,
ids match test.csv exactly, mean prediction length 11.95.

Grader is 10 cores / 62 GB / 1.5 h. Parallel stages (candgen, featurize) scale ~10/7;
projected grader wall-clock ~20 min, well within the 1.5 h limit. No timing escapes
(drop fuzzw / cap training pools 50 / drop NN) were needed. Peak RSS stayed ~8-11 GB
(fits are fork/COW-shared across workers), comfortably under the 62 GB grader cap.

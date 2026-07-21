"""Train the LambdaRank reranker once, cache everything needed to sweep decoders.

Saves:
  runs/B1/reranker.txt   trained LightGBM model (materially drives predictions)
  runs/B1/valcache.pkl   per bucket-0 row: candidates, reranker scores,
                         retrieval weights (level-weighted cond prob, global freq)
                         + refs, so decode_exp.py can sweep argmax/MBR cheaply.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "5")
import sys, time, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "solution")
from chrf import score_lists
sys.path.insert(0, "runs/B1")
import pipeline_v1 as P

NTRAIN = 45000
LVLW = {"l2r2": 3.0, "l2r1": 2.2, "l1r2": 2.2, "l1r1": 1.6,
        "l1": 1.0, "r1": 1.0, "l2": 0.8, "r2": 0.8}

t0 = time.time()
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
train["_bkt"] = train.masked_docstring.map(P.bucket)
val = train[train._bkt == 0].copy()
fold = train[train._bkt != 0].copy()
even = fold[fold._bkt % 2 == 0]
odd = fold[fold._bkt % 2 == 1]
idx_even, ge = P.build_index(even)
idx_odd, go = P.build_index(odd)
glob_fold = ge + go
gtop = P.global_top(glob_fold, 12)
print(f"[idx_parity] {time.time()-t0:.1f}s", flush=True)

samp = fold.sample(n=min(NTRAIN, len(fold)), random_state=7).reset_index(drop=True)
n_es = int(len(samp) * 0.08)
es_df, fit_df = samp.iloc[:n_es], samp.iloc[n_es:]
ce, co = P.make_stat_cache(), P.make_stat_cache()
Xfit, yfit, gfit = P.build_training_matrix(fit_df, idx_even, idx_odd, glob_fold, gtop, ce, co)
Xes, yes, ges = P.build_training_matrix(es_df, idx_even, idx_odd, glob_fold, gtop, ce, co)
print(f"[train_feat] fit {Xfit.shape}  {time.time()-t0:.1f}s", flush=True)

gr_fit = np.minimum((yfit * 10).astype(int), 10)
gr_es = np.minimum((yes * 10).astype(int), 10)
params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
              label_gain=list(range(11)), lambdarank_truncation_level=20,
              learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
              num_threads=5, max_bin=255, verbose=-1)
dtr = lgb.Dataset(Xfit, gr_fit, group=gfit, feature_name=P.FEAT_NAMES)
dva = lgb.Dataset(Xes, gr_es, group=ges, reference=dtr)
booster = lgb.train(params, dtr, num_boost_round=700, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
booster.save_model("runs/B1/reranker.txt", num_iteration=booster.best_iteration)
print(f"[train] iters {booster.best_iteration}  {time.time()-t0:.1f}s", flush=True)

# full index for eval
idx_full, glob_full = P.build_index(fold)
gtop_full = P.global_top(glob_full, 12)
cache_v = P.make_stat_cache()
print(f"[idx_full] {time.time()-t0:.1f}s", flush=True)

rows = []
argmax_preds = []
refs = val.target_span.astype(str).tolist()
for masked, code in zip(val.masked_docstring.values, val.code_context.values):
    rc = P.row_ctx(masked)
    cc, idents = P.code_features(code)
    cands, src = P.gen_candidates(rc, idx_full, gtop_full, cc)
    if not cands:
        cands = ["value of the"]
        rows.append({"cands": cands, "rr": np.array([0.0]),
                     "wsum": np.array([1.0]), "wglob": np.array([1.0])})
        argmax_preds.append(cands[0])
        continue
    X = P.featurize(cands, src, rc, idx_full, glob_full, cache_v, code, idents)
    rr = booster.predict(X)
    wsum = np.zeros(len(cands)); wglob = np.zeros(len(cands))
    lc = {}
    for i, c in enumerate(cands):
        for name, _, _ in P.LEVELS:
            total, rankmap = P.level_stats(idx_full, name, rc["keys"][name], lc)
            info = rankmap.get(c)
            if info is not None and total > 0:
                wsum[i] += LVLW[name] * (info[1] / total)
        wglob[i] = glob_full.get(c, 0)
    rows.append({"cands": cands, "rr": rr.astype(np.float32),
                 "wsum": wsum.astype(np.float32), "wglob": wglob.astype(np.float32)})
    argmax_preds.append(cands[int(np.argmax(rr))])

print(f"[argmax reranker] end2end {score_lists(argmax_preds, refs):.4f}", flush=True)
with open("runs/B1/valcache.pkl", "wb") as f:
    pickle.dump({"rows": rows, "refs": refs, "ids": val.id.tolist()}, f)
print(f"[saved] valcache.pkl ({len(rows)} rows)  total {time.time()-t0:.1f}s", flush=True)

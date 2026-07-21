"""v2 reranker: adds a word n-gram fill-fluency LM feature (stupid backoff).
Parity-safe LMs for training candgen (even-row -> odd LM), full LM for eval.
Reports argmax + MBR(T=0.6) end-to-end on full bucket-0; saves val_pred.csv,
reranker_v2.txt, valcache.pkl.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "5")
import sys, time, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists
sys.path.insert(0, "runs/B1")
import pipeline_v1 as P

NTRAIN = 45000
LVLW = {"l2r2": 3.0, "l2r1": 2.2, "l1r2": 2.2, "l1r1": 1.6,
        "l1": 1.0, "r1": 1.0, "l2": 0.8, "r2": 0.8}
T = {}
t0 = time.time()
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
train["_bkt"] = train.masked_docstring.map(P.bucket)
val = train[train._bkt == 0].copy()
fold = train[train._bkt != 0].copy()
even = fold[fold._bkt % 2 == 0]; odd = fold[fold._bkt % 2 == 1]
idx_even, ge = P.build_index(even); idx_odd, go = P.build_index(odd)
glob_fold = ge + go; gtop = P.global_top(glob_fold, 12)
T["idx"] = time.time() - t0
lm_even = P.build_lm(even); lm_odd = P.build_lm(odd)
T["lm"] = time.time() - t0 - T["idx"]
print(f"[idx {T['idx']:.0f}s][lm {T['lm']:.0f}s] tri_e {len(lm_even['tri'])} tri_o {len(lm_odd['tri'])}", flush=True)

samp = fold.sample(n=min(NTRAIN, len(fold)), random_state=7).reset_index(drop=True)
n_es = int(len(samp) * 0.08)
es_df, fit_df = samp.iloc[:n_es], samp.iloc[n_es:]
ce, co = P.make_stat_cache(), P.make_stat_cache()
tf = time.time()
Xfit, yfit, gfit = P.build_training_matrix(fit_df, idx_even, idx_odd, glob_fold, gtop, ce, co, lm_even, lm_odd)
Xes, yes, ges = P.build_training_matrix(es_df, idx_even, idx_odd, glob_fold, gtop, ce, co, lm_even, lm_odd)
T["train_feat"] = time.time() - tf
print(f"[train_feat] {Xfit.shape} {T['train_feat']:.0f}s", flush=True)

gr_fit = np.minimum((yfit * 10).astype(int), 10)
gr_es = np.minimum((yes * 10).astype(int), 10)
params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
              label_gain=list(range(11)), lambdarank_truncation_level=20,
              learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
              num_threads=5, max_bin=255, verbose=-1)
dtr = lgb.Dataset(Xfit, gr_fit, group=gfit, feature_name=P.FEAT_NAMES)
dva = lgb.Dataset(Xes, gr_es, group=ges, reference=dtr)
tr = time.time()
booster = lgb.train(params, dtr, num_boost_round=700, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
booster.save_model("runs/B1/reranker_v2.txt", num_iteration=booster.best_iteration)
T["train"] = time.time() - tr
imp = sorted(zip(P.FEAT_NAMES, booster.feature_importance("gain")), key=lambda x: -x[1])
print(f"[train {T['train']:.0f}s] iters {booster.best_iteration} top12:", [(n, int(g)) for n, g in imp[:12]], flush=True)

# full index + LM for eval
idx_full, glob_full = P.build_index(fold); gtop_full = P.global_top(glob_full, 12)
lm_full = P.build_lm(fold)
cache_v = P.make_stat_cache()
refs = val.target_span.astype(str).tolist()
rows, argmax_preds = [], []
tp = time.time()
for masked, code in zip(val.masked_docstring.values, val.code_context.values):
    rc = P.row_ctx(masked); cc, idents = P.code_features(code)
    cands, src = P.gen_candidates(rc, idx_full, gtop_full, cc)
    if not cands:
        rows.append({"cands": ["value of the"], "rr": np.array([0.0]),
                     "wsum": np.array([1.0]), "wglob": np.array([1.0])})
        argmax_preds.append("value of the"); continue
    X = P.featurize(cands, src, rc, idx_full, glob_full, cache_v, code, idents, lm=lm_full)
    rr = booster.predict(X)
    wsum = np.zeros(len(cands)); wglob = np.zeros(len(cands)); lc = {}
    for i, c in enumerate(cands):
        for name, _, _ in P.LEVELS:
            total, rm = P.level_stats(idx_full, name, rc["keys"][name], lc)
            info = rm.get(c)
            if info is not None and total > 0:
                wsum[i] += LVLW[name] * (info[1] / total)
        wglob[i] = glob_full.get(c, 0)
    rows.append({"cands": cands, "rr": rr.astype(np.float32),
                 "wsum": wsum.astype(np.float32), "wglob": wglob.astype(np.float32)})
    argmax_preds.append(cands[int(np.argmax(rr))])
T["predict"] = time.time() - tp
chrf_arg = score_lists(argmax_preds, refs)
print(f"[argmax] end2end {chrf_arg:.4f}  predict {T['predict']:.0f}s", flush=True)

with open("runs/B1/valcache.pkl", "wb") as f:
    pickle.dump({"rows": rows, "refs": refs, "ids": val.id.tolist()}, f)

# MBR T=0.6 over top-12 subset
M = 12
def softmax(x, t):
    x = np.asarray(x, np.float64) / t; x = np.exp(x - x.max()); return x / x.sum()
mbr_preds = []
for r in rows:
    cands, rr, wsum = r["cands"], r["rr"], r["wsum"]
    if len(cands) == 1:
        mbr_preds.append(cands[0]); continue
    sel = list(dict.fromkeys(list(np.argsort(-rr)[:M]) + list(np.argsort(-wsum)[:M])))[:M]
    cs = [cands[j] for j in sel]; k = len(cs)
    K = np.empty((k, k), np.float32)
    for a in range(k):
        K[a, a] = 1.0
        for b in range(a + 1, k):
            v = f_pooled(cs[a], cs[b]); K[a, b] = v; K[b, a] = v
    w = softmax(rr[sel], 0.6); w /= w.sum()
    mbr_preds.append(cs[int(np.argmax(K @ w))])
chrf_mbr = score_lists(mbr_preds, refs)
print(f"[mbr T=0.6] end2end {chrf_mbr:.4f}", flush=True)

best_mode = "mbr" if chrf_mbr > chrf_arg else "argmax"
best_preds = mbr_preds if best_mode == "mbr" else argmax_preds
pd.DataFrame({"id": val.id.values, "prediction": best_preds}).to_csv("runs/B1/val_pred.csv", index=False)
tot = time.time() - t0
print(f"\n[TIMINGS] " + "  ".join(f"{k}={v:.0f}s" for k, v in T.items()) + f"  TOTAL={tot:.0f}s", flush=True)
print(f"HEADLINE bkt0 chrF: argmax {chrf_arg:.4f} | mbr {chrf_mbr:.4f} | BEST {max(chrf_arg,chrf_mbr):.4f} ({best_mode})", flush=True)

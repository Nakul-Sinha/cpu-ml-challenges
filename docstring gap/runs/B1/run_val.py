"""End-to-end validation for B1 reranker pipeline v1.

Compares reranker objectives (the argmax target is WITHIN-ROW ranking, so plain
regression on absolute chrF underfits): regression | within-row-centered
regression | LambdaRank. Trains on buckets 1-19 (parity-cross candgen, no
self/twin leak); evaluates end-to-end on the FULL bucket-0 set with an index
rebuilt from buckets 1-19. Reports stage timings + grader extrapolation.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "5")
import sys, time, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists
sys.path.insert(0, "runs/B1")
import pipeline_v1 as P

ap = argparse.ArgumentParser()
ap.add_argument("--ntrain", type=int, default=45000)
ap.add_argument("--rounds", type=int, default=700)
ap.add_argument("--mbr_k", type=int, default=6)
ap.add_argument("--oracle_n", type=int, default=3000)
args = ap.parse_args()

T = {}
t0 = time.time()
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
train["_bkt"] = train.masked_docstring.map(P.bucket)
val = train[train._bkt == 0].copy()
fold = train[train._bkt != 0].copy()
even = fold[fold._bkt % 2 == 0]
odd = fold[fold._bkt % 2 == 1]
T["load"] = time.time() - t0
print(f"[load] train {len(train)} val {len(val)} fold {len(fold)} "
      f"even {len(even)} odd {len(odd)}  {T['load']:.1f}s", flush=True)

t = time.time()
idx_even, glob_even = P.build_index(even)
idx_odd, glob_odd = P.build_index(odd)
glob_fold = glob_even + glob_odd
gtop = P.global_top(glob_fold, 12)
T["idx_parity"] = time.time() - t
print(f"[idx_parity] {T['idx_parity']:.1f}s", flush=True)

t = time.time()
samp = fold.sample(n=min(args.ntrain, len(fold)), random_state=7).reset_index(drop=True)
n_es = int(len(samp) * 0.08)
es_df, fit_df = samp.iloc[:n_es], samp.iloc[n_es:]
cache_e, cache_o = P.make_stat_cache(), P.make_stat_cache()
Xfit, yfit, gfit = P.build_training_matrix(fit_df, idx_even, idx_odd, glob_fold, gtop, cache_e, cache_o)
Xes, yes, ges = P.build_training_matrix(es_df, idx_even, idx_odd, glob_fold, gtop, cache_e, cache_o)
T["candgen_feat_train"] = time.time() - t
print(f"[train_feat] fit {Xfit.shape} es {Xes.shape}  cands/row "
      f"{Xfit.shape[0]/max(len(fit_df),1):.1f}  y.mean {yfit.mean():.3f}  "
      f"{T['candgen_feat_train']:.1f}s", flush=True)


def row_center(y, g):
    out = np.empty_like(y)
    i = 0
    for gs in g:
        seg = y[i:i + gs]
        out[i:i + gs] = seg - seg.mean()
        i += gs
    return out


# ---- full index on buckets 1-19; precompute val features ONCE ----
t = time.time()
idx_full, glob_full = P.build_index(fold)
gtop_full = P.global_top(glob_full, 12)
T["idx_full"] = time.time() - t
print(f"[idx_full] {T['idx_full']:.1f}s", flush=True)

t = time.time()
cache_v = P.make_stat_cache()
val_cands, val_X = [], []
for masked, code in zip(val.masked_docstring.values, val.code_context.values):
    rc = P.row_ctx(masked)
    cc, idents = P.code_features(code)
    cands, src = P.gen_candidates(rc, idx_full, gtop_full, cc)
    if not cands:
        cands = ["value of the"]
        X = np.zeros((1, P.N_FEAT), dtype=np.float32)
    else:
        X = P.featurize(cands, src, rc, idx_full, glob_full, cache_v, code, idents)
    val_cands.append(cands)
    val_X.append(X)
T["val_featurize"] = time.time() - t
refs = val.target_span.astype(str).tolist()
print(f"[val_featurize] {len(val)} rows  {T['val_featurize']:.1f}s", flush=True)

# ---- oracle ceiling on subset ----
t = time.time()
orc, hit, psz = [], 0, []
for i in range(min(args.oracle_n, len(val))):
    cs = val_cands[i]
    tgt = refs[i]
    psz.append(len(cs))
    orc.append(max((f_pooled(c, tgt) for c in cs), default=0.0))
    hit += tgt in cs
print(f"[oracle] pool chrF {np.mean(orc):.4f}  exact-hit {hit/len(psz):.3f}  "
      f"mean pool {np.mean(psz):.1f}  {time.time()-t:.1f}s", flush=True)


def argmax_preds(booster):
    out = []
    for cs, X in zip(val_cands, val_X):
        s = booster.predict(X)
        out.append(cs[int(np.argmax(s))])
    return out


common = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
              num_threads=5, max_bin=255, verbose=-1)
results = {}
boosters = {}

# 1) plain regression
t = time.time()
dtr = lgb.Dataset(Xfit, yfit, feature_name=P.FEAT_NAMES)
dva = lgb.Dataset(Xes, yes, reference=dtr)
b1 = lgb.train(dict(common, objective="regression", metric="l2"), dtr,
               num_boost_round=args.rounds, valid_sets=[dva],
               callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
results["regression"] = score_lists(argmax_preds(b1), refs)
boosters["regression"] = b1
print(f"[regression] iters {b1.best_iteration}  end2end {results['regression']:.4f}  {time.time()-t:.1f}s", flush=True)

# 2) within-row centered regression
t = time.time()
yfit_c, yes_c = row_center(yfit, gfit), row_center(yes, ges)
dtr2 = lgb.Dataset(Xfit, yfit_c, feature_name=P.FEAT_NAMES)
dva2 = lgb.Dataset(Xes, yes_c, reference=dtr2)
b2 = lgb.train(dict(common, objective="regression", metric="l2"), dtr2,
               num_boost_round=args.rounds, valid_sets=[dva2],
               callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
results["centered"] = score_lists(argmax_preds(b2), refs)
boosters["centered"] = b2
print(f"[centered] iters {b2.best_iteration}  end2end {results['centered']:.4f}  {time.time()-t:.1f}s", flush=True)

# 3) LambdaRank (graded relevance from chrF, linear gains)
t = time.time()
grades_fit = np.minimum((yfit * 10).astype(int), 10)
grades_es = np.minimum((yes * 10).astype(int), 10)
dtr3 = lgb.Dataset(Xfit, grades_fit, group=gfit, feature_name=P.FEAT_NAMES)
dva3 = lgb.Dataset(Xes, grades_es, group=ges, reference=dtr3)
b3 = lgb.train(dict(common, objective="lambdarank", metric="ndcg",
                    ndcg_eval_at=[1], label_gain=list(range(11)),
                    lambdarank_truncation_level=20), dtr3,
               num_boost_round=args.rounds, valid_sets=[dva3],
               callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
results["lambdarank"] = score_lists(argmax_preds(b3), refs)
boosters["lambdarank"] = b3
print(f"[lambdarank] iters {b3.best_iteration}  end2end {results['lambdarank']:.4f}  {time.time()-t:.1f}s", flush=True)

best_obj = max(results, key=results.get)
best_b = boosters[best_obj]
imp = sorted(zip(P.FEAT_NAMES, best_b.feature_importance("gain")), key=lambda x: -x[1])
print(f"\n[best={best_obj}] top15 gain:", [(n, int(g)) for n, g in imp[:15]], flush=True)

# ---- MBR on best model ----
t = time.time()
best_preds = argmax_preds(best_b)


def mbr_pred(booster, k):
    out = []
    for cs, X in zip(val_cands, val_X):
        s = booster.predict(X)
        if len(cs) <= 1:
            out.append(cs[0]); continue
        kk = min(k, len(cs))
        top = np.argsort(-s)[:kk]
        w = s[top].astype(np.float64); w = np.exp((w - w.max()) / 0.15); w /= w.sum()
        tc = [cs[j] for j in top]
        bj, bv = 0, -1.0
        for a in range(kk):
            ev = sum(w[b] * f_pooled(tc[a], tc[b]) for b in range(kk))
            if ev > bv:
                bv, bj = ev, a
        out.append(tc[bj])
    return out


mbr_preds = mbr_pred(best_b, args.mbr_k)
chrf_mbr = score_lists(mbr_preds, refs)
T["mbr"] = time.time() - t
print(f"[mbr] best+mbr(k={args.mbr_k}) end2end {chrf_mbr:.4f}  {T['mbr']:.1f}s", flush=True)

final_mode = "mbr" if chrf_mbr > results[best_obj] else best_obj
final_preds = mbr_preds if final_mode == "mbr" else best_preds
final_chrf = max(chrf_mbr, results[best_obj])
pd.DataFrame({"id": val.id.values, "pred": final_preds}).to_csv("runs/B1/val_pred.csv", index=False)

# ---- timing extrapolation to grader ----
tot = time.time() - t0
r_feat = T["candgen_feat_train"] / max(len(fit_df) + len(es_df), 1)
r_pred = T["val_featurize"] / max(len(val), 1)
r_idx = T["idx_full"] / max(len(fold), 1)
est_idx = r_idx * 232000
est_train_feat = r_feat * min(args.ntrain, 232000)
est_pred = r_pred * 50000
est_total = est_idx + est_train_feat + 3 * T.get("mbr", 30) + est_pred + T["load"] * 1.2 + 30
print("\n==== STAGE TIMINGS ====")
for k, v in T.items():
    print(f"  {k:20s} {v:7.1f}s")
print(f"  {'TOTAL':20s} {tot:7.1f}s")
print("\n==== GRADER EXTRAPOLATION (10-core, single reranker) ====")
print(f"  index build 232k     ~{est_idx:6.0f}s")
print(f"  train candgen+feat   ~{est_train_feat:6.0f}s (ntrain={args.ntrain})")
print(f"  predict 50k test     ~{est_pred:6.0f}s")
print(f"  EST TOTAL            ~{est_total:6.0f}s  ({est_total/60:.1f} min)")
print("\n==== RESULTS ====")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {k:12s} {v:.4f}")
print(f"  {'mbr':12s} {chrf_mbr:.4f}")
print(f"\nHEADLINE end-to-end bkt0 chrF: {final_chrf:.4f} (mode={final_mode})")

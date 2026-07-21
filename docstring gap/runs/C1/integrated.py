"""C1 integration module: B2 PoolBuilder UNION B3 LMBridge -> LambdaRank reranker.

Candidate universe per row = B2 PoolBuilder.candidates_batch(gate_fuzz=True, cap 80)
UNION B3 LMBridge top-12 bridges (carry entering/leaving LM logp, length, LM rank).
Dedup keeping B2 max-tier metadata + LM metadata (consensus flagged). NN candidates
were measured (+0.0073 union-oracle on 3k < +0.01) and are EXCLUDED per recipe.

Features (~78): B1's 53 (anchored prob/rank/present, glob freq, lexical, LM fluency)
+ B2 provenance (source one-hots, tier, score, fuzzy cosine, code target-prior,
anchor strength) + B3 LM metadata (enter/leave logp, per-word, rank, in-pool,
consensus) + centrality (mean pairwise f_pooled vs top-8-by-tier).

Reranker: LightGBM LambdaRank, B1 params (label=min(int(10*f_pooled),10)).
Decode: argmax + MBR top-12 under reranker softmax.

Parity hygiene: even-bucket training rows draw candidates + features from the ODD
half-fits of every module, odd rows from EVEN; full buckets-1..19 fits for bucket-0
eval. Fitted objects reused, never refit per row. Nothing fit on test/holdout.
Compliance: count-based / conditional-prob features only; NO idf/tfidf/bm25.
"""
import os
# explicit process/thread parallelism only -> pin BLAS/OMP to 1 to avoid oversubscription
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, math, argparse, multiprocessing
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, "solution")
sys.path.insert(0, "runs/B1")
sys.path.insert(0, "runs/b2")
sys.path.insert(0, "runs/B3")
from chrf import f_pooled, score_lists          # noqa: E402
import pipeline_v1 as P                          # noqa: E402  (B1)
from pool_builder import PoolBuilder, anchors, SRC_TIER  # noqa: E402  (B2)
import lm_bridge as LB                           # noqa: E402  (B3)

GAP = "[GAP]"
WB = 1.0            # LMBridge word_bonus (B3 chose 1.0)
LM_TOPN = 12
BEAM = 16
LM_FLOOR = -40.0    # sentinel logp for candidates absent from the LM pool
CAP = 80

# ---- EXT feature names (B2 provenance + B3 LM + centrality) ----
EXT_NAMES = [
    "in_b2", "b2_tier", "b2_score", "b2_fuzz_cos", "b2_code_prior_log", "b2_anchor_strength",
    "b2src_l2r2", "b2src_l1r1", "b2src_skip", "b2src_r1", "b2src_l1", "b2src_codeP", "b2src_code",
    "b2src_fuzz", "b2src_fuzzw", "b2src_global",
    "in_lm", "lm_enter", "lm_leave", "lm_enter_pw", "lm_total_pw", "lm_rank", "lm_len",
    "consensus", "centrality8",
]
FEAT_NAMES = list(P.FEAT_NAMES) + EXT_NAMES
N_FEAT = len(FEAT_NAMES)

# ======================= fitted-object bundle =======================
class Fits:
    """B1 (idx/glob/gtop/lm), B2 (PoolBuilder), B3 (LMBridge) for one data split."""
    def __init__(self, idx, glob, gtop, lm, pb, lmb):
        self.idx, self.glob, self.gtop, self.lm = idx, glob, gtop, lm
        self.pb, self.lmb = pb, lmb


def build_fits(df, n_threads, tag=""):
    t0 = time.time()
    idx, glob = P.build_index(df)
    gtop = P.global_top(glob, 12)
    lm = P.build_lm(df)
    pb = PoolBuilder(cap=CAP, gate_fuzz=True, n_threads=n_threads).fit(df)
    lmb = LB.LMBridge().fit(df)
    print(f"[fits {tag}] n={len(df)} {time.time()-t0:.0f}s "
          f"(b1_idx+lm, b2_fit {pb.t_fit:.0f}s, lm_tri {len(lmb.c3)})", flush=True)
    return Fits(idx, glob, gtop, lm, pb, lmb)


# ======================= B3 LM candgen (multiprocessing) =======================
_LMB = None  # set before forking

def _lm_chunk(rows):
    out = []
    for masked, code in rows:
        l, r = LB.split_ctx(masked)
        pool = _LMB.pool(l, r, LB.code_words(code), beam=BEAM)
        scored = sorted(pool, key=lambda x: x[0] + WB * x[2] + x[1], reverse=True)[:LM_TOPN]
        # (text, enter_logp, leave_logp, length, rank)
        out.append([(t[3], t[0], t[1], t[2], rk) for rk, t in enumerate(scored)])
    return out


def lm_candidates(df, lmb, n_proc):
    global _LMB
    _LMB = lmb
    rows = list(zip(df.masked_docstring.values, df.code_context.values))
    n = len(rows)
    if n == 0:
        return []
    nchunks = max(1, min(n, n_proc * 4))
    bnds = np.linspace(0, n, nchunks + 1).astype(int)
    chunks = [rows[bnds[i]:bnds[i + 1]] for i in range(nchunks) if bnds[i + 1] > bnds[i]]
    if n_proc <= 1 or n < 200:
        res = [_lm_chunk(c) for c in chunks]
    else:
        ctx = multiprocessing.get_context("fork")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as ex:
            res = list(ex.map(_lm_chunk, chunks))
    out = []
    for r in res:
        out.extend(r)
    return out


# ======================= union pool builder =======================
def anchor_strength(masked, pb):
    a = anchors(masked)
    if a["l2r2"] in pb.idx["l2r2"]:
        return 2
    if a["l1r1"] in pb.idx["l1r1"]:
        return 1
    return 0


def build_union(df, fits, n_proc):
    """Return per-row (anchor_strength, union_pool) where union_pool is a list of
    tuples: (text, in_b2, src, b2_score, b2_tier, in_lm, lm_enter, lm_leave, lm_len, lm_rank)."""
    t0 = time.time()
    b2_pools = fits.pb.candidates_batch(df, use_fuzz=True)        # gated inside (gate_fuzz=True)
    t1 = time.time()
    lm_pools = lm_candidates(df, fits.lmb, n_proc)
    t2 = time.time()
    masked = df.masked_docstring.values
    unions, astr = [], []
    for i in range(len(df)):
        d = {}   # text -> mutable list
        for t, s, sc in b2_pools[i]:
            d[t] = [t, True, s, float(sc), int(SRC_TIER.get(s, 0)), False, 0.0, 0.0, 0, LM_TOPN]
        for (t, e, lv, ln, rk) in lm_pools[i]:
            if t in d:
                row = d[t]
                row[5] = True; row[6] = e; row[7] = lv; row[8] = ln; row[9] = rk
            else:
                d[t] = [t, False, "", 0.0, 0, True, e, lv, ln, rk]
        unions.append([tuple(v) for v in d.values()])
        astr.append(anchor_strength(masked[i], fits.pb))
    print(f"[union] n={len(df)} b2 {t1-t0:.0f}s lm {t2-t1:.0f}s merge {time.time()-t2:.0f}s "
          f"mean_pool {np.mean([len(u) for u in unions]):.1f}", flush=True)
    return astr, unions


# ======================= featurization (multiprocessing) =======================
_G = {}          # fit bundle for the current phase (set before forking)
_FCACHE = None   # per-worker level_stats cache

_SRC1H = {"l2r2": 6, "l1r1": 7, "skipR": 8, "skipL": 8, "r1": 9, "l1": 10,
          "codeP": 11, "code": 12, "fuzz": 13, "fuzzw": 14, "global": 15}


def _grams(s):
    c = Counter()
    L = len(s)
    for n in range(1, 7):
        for i in range(L - n + 1):
            c[s[i:i + n]] += 1
    return c, sum(c.values())


def _overlap(a, b):
    if len(a) <= len(b):
        return sum(min(v, b.get(k, 0)) for k, v in a.items())
    return sum(min(v, a.get(k, 0)) for k, v in b.items())


def _ext_features(pool, astr, pb):
    n = len(pool)
    E = np.zeros((n, len(EXT_NAMES)), dtype=np.float32)
    # centrality references: top-8 by (b2_tier, b2_score)
    order = sorted(range(n), key=lambda i: (pool[i][4], pool[i][3]), reverse=True)
    ref_idx = order[:8]
    gcache = {}

    def gg(i):
        g = gcache.get(i)
        if g is None:
            g = _grams(pool[i][0]); gcache[i] = g
        return g
    refs = [gg(i) for i in ref_idx]
    for i in range(n):
        text, in_b2, src, b2_score, b2_tier, in_lm, lm_e, lm_lv, lm_len, lm_rk = pool[i]
        E[i, 0] = 1.0 if in_b2 else 0.0
        E[i, 1] = b2_tier
        E[i, 2] = b2_score
        E[i, 3] = (b2_score - b2_tier) if (in_b2 and src in ("fuzz", "fuzzw")) else 0.0
        E[i, 4] = math.log1p(pb.target_prior.get(text, 0))
        E[i, 5] = astr
        if in_b2 and src in _SRC1H:
            E[i, _SRC1H[src]] = 1.0
        E[i, 16] = 1.0 if in_lm else 0.0
        if in_lm:
            E[i, 17] = lm_e
            E[i, 18] = lm_lv
            E[i, 19] = lm_e / lm_len if lm_len else lm_e
            E[i, 20] = (lm_e + lm_lv) / lm_len if lm_len else (lm_e + lm_lv)
            E[i, 21] = lm_rk
            E[i, 22] = lm_len
        else:
            E[i, 17] = LM_FLOOR; E[i, 18] = LM_FLOOR
            E[i, 19] = LM_FLOOR; E[i, 20] = LM_FLOOR
            E[i, 21] = LM_TOPN; E[i, 22] = 0
        E[i, 23] = 1.0 if (in_b2 and in_lm) else 0.0
        # centrality
        cg, tp = gg(i)
        if tp and refs:
            tot = 0.0
            for rg, tr in refs:
                if tr == 0:
                    continue
                m = _overlap(cg, rg)
                if m:
                    p = m / tp; r = m / tr
                    tot += 2 * p * r / (p + r)
            E[i, 24] = tot / len(refs)
    return E


def _feat_chunk(payloads):
    idx, glob, gtop, lm, pb = _G["idx"], _G["glob"], _G["gtop"], _G["lm"], _G["pb"]
    gtop_set = set(gtop)
    Xs, ys, ncands, texts = [], [], [], []
    global _FCACHE
    if _FCACHE is None:
        _FCACHE = {}
    for (masked, code, pool, astr, tgt) in payloads:
        cands = [p[0] for p in pool]
        rc = P.row_ctx(masked)
        cc, idents = P.code_features(code)
        cc_set = set(cc)
        src = {}
        for t in cands:
            s = set()
            if t in cc_set:
                s.add("code")
            if t in gtop_set:
                s.add("global")
            src[t] = s
        Xb1 = P.featurize(cands, src, rc, idx, glob, _FCACHE, code, idents, lm=lm)
        Xext = _ext_features(pool, astr, pb)
        Xs.append(np.hstack([Xb1, Xext]))
        ncands.append(len(cands))
        texts.append(cands)
        if tgt is not None:
            ys.append(np.fromiter((f_pooled(c, tgt) for c in cands), np.float32, len(cands)))
    X = np.vstack(Xs) if Xs else np.zeros((0, N_FEAT), np.float32)
    y = np.concatenate(ys) if ys else None
    return X, y, ncands, texts


def featurize_rows(df, fits, astr, unions, n_proc, want_labels):
    """Parallel featurize. Returns X, y|None, groups(list ncands), texts(list of lists)."""
    global _G, _FCACHE
    _G = {"idx": fits.idx, "glob": fits.glob, "gtop": fits.gtop, "lm": fits.lm, "pb": fits.pb}
    _FCACHE = {}
    tgts = df.target_span.astype(str).values if want_labels else [None] * len(df)
    masked = df.masked_docstring.values
    codes = df.code_context.values
    payloads = [(masked[i], codes[i], unions[i], astr[i], tgts[i]) for i in range(len(df))]
    n = len(payloads)
    nchunks = max(1, min(n, n_proc * 4))
    bnds = np.linspace(0, n, nchunks + 1).astype(int)
    chunks = [payloads[bnds[i]:bnds[i + 1]] for i in range(nchunks) if bnds[i + 1] > bnds[i]]
    if n_proc <= 1 or n < 200:
        res = [_feat_chunk(c) for c in chunks]
    else:
        ctx = multiprocessing.get_context("fork")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as ex:
            res = list(ex.map(_feat_chunk, chunks))
    Xs, ys, groups, texts = [], [], [], []
    for X, y, nc, tx in res:
        Xs.append(X)
        if want_labels:
            ys.append(y)
        groups.extend(nc)
        texts.extend(tx)
    X = np.vstack(Xs) if Xs else np.zeros((0, N_FEAT), np.float32)
    y = np.concatenate(ys) if (want_labels and ys) else None
    return X, y, groups, texts


# ======================= training rows: parity phases =======================
def build_training(train_sample, fits_even, fits_odd, n_proc):
    """even-bucket rows use ODD fits, odd-bucket rows use EVEN fits (no self/twin leak)."""
    ev = train_sample[train_sample._bkt % 2 == 0]
    od = train_sample[train_sample._bkt % 2 == 1]
    parts = []
    for sub, fits, tag in ((ev, fits_odd, "even->odd"), (od, fits_even, "odd->even")):
        if len(sub) == 0:
            continue
        t0 = time.time()
        astr, unions = build_union(sub, fits, n_proc)
        X, y, groups, _ = featurize_rows(sub, fits, astr, unions, n_proc, want_labels=True)
        print(f"[train phase {tag}] X{X.shape} {time.time()-t0:.0f}s", flush=True)
        parts.append((X, y, groups))
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = sum((p[2] for p in parts), [])
    return X, y, groups


# ======================= reranker + decode =======================
import lightgbm as lgb


def train_reranker(X, y, groups, n_threads, es_frac=0.08, seed=7):
    assert min(groups) > 0, "zero-size group present"  # B2 always adds global cands
    n = len(groups)
    n_es = max(1, int(n * es_frac))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    es_g = set(perm[:n_es].tolist())
    starts = np.cumsum([0] + groups)
    es_rows, fit_rows, g_es, g_fit = [], [], [], []
    for gi in range(n):
        rows = range(starts[gi], starts[gi + 1])
        if gi in es_g:
            es_rows.extend(rows); g_es.append(groups[gi])
        else:
            fit_rows.extend(rows); g_fit.append(groups[gi])
    Xf, yf = X[fit_rows], y[fit_rows]
    Xe, ye = X[es_rows], y[es_rows]
    lab_f = np.minimum((yf * 10).astype(int), 10)
    lab_e = np.minimum((ye * 10).astype(int), 10)
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                  label_gain=list(range(11)), lambdarank_truncation_level=20,
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
                  num_threads=n_threads, max_bin=255, verbose=-1)
    dtr = lgb.Dataset(Xf, lab_f, group=g_fit, feature_name=FEAT_NAMES)
    dva = lgb.Dataset(Xe, lab_e, group=g_es, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=700, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    return booster


def decode(booster, X, groups, texts, mbr_temp=0.0, mbr_m=12, fallback="value of the"):
    starts = np.cumsum([0] + groups)
    scores = booster.predict(X) if len(X) else np.zeros(0)
    preds = []
    for gi in range(len(groups)):
        cs = texts[gi]
        if not cs:
            preds.append(fallback); continue
        s = scores[starts[gi]:starts[gi + 1]]
        if mbr_temp <= 0 or len(cs) == 1:
            preds.append(cs[int(np.argmax(s))]); continue
        sel = list(np.argsort(-s)[:mbr_m])
        sub = [cs[j] for j in sel]; k = len(sub)
        K = np.empty((k, k), np.float32)
        for a in range(k):
            K[a, a] = 1.0
            for b in range(a + 1, k):
                v = f_pooled(sub[a], sub[b]); K[a, b] = v; K[b, a] = v
        w = np.asarray(s[sel], np.float64) / mbr_temp
        w = np.exp(w - w.max()); w /= w.sum()
        preds.append(sub[int(np.argmax(K @ w))])
    return preds


# ======================= bucket-0 evaluation driver =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=58000)
    ap.add_argument("--nproc", type=int, default=7)
    ap.add_argument("--nthreads", type=int, default=7)
    ap.add_argument("--valn", type=int, default=0, help=">0 truncates bucket-0 (smoke)")
    ap.add_argument("--out", default="runs/C1/val_pred.csv")
    args = ap.parse_args()
    T = {}
    t0 = time.time()
    train = pd.read_csv("dataset/train.csv", keep_default_na=False)
    train["_bkt"] = train.masked_docstring.map(P.bucket)
    val = train[train._bkt == 0].copy()
    fold = train[train._bkt != 0].copy()
    if args.valn > 0:
        val = val.head(args.valn)
    print(f"[load] train {len(train)} fold {len(fold)} val {len(val)} {time.time()-t0:.0f}s", flush=True)

    # parity fits on FOLD halves
    even = fold[fold._bkt % 2 == 0]; odd = fold[fold._bkt % 2 == 1]
    tf = time.time()
    fits_even = build_fits(even, args.nthreads, "even")
    fits_odd = build_fits(odd, args.nthreads, "odd")
    T["parity_fits"] = time.time() - tf

    samp = fold.sample(n=min(args.ntrain, len(fold)), random_state=7).reset_index(drop=True)
    tt = time.time()
    Xtr, ytr, gtr = build_training(samp, fits_even, fits_odd, args.nproc)
    T["train_candgen_feat"] = time.time() - tt
    print(f"[train matrix] X{Xtr.shape} groups {len(gtr)} {T['train_candgen_feat']:.0f}s", flush=True)

    tr = time.time()
    booster = train_reranker(Xtr, ytr, gtr, args.nthreads)
    T["train"] = time.time() - tr
    imp = sorted(zip(FEAT_NAMES, booster.feature_importance("gain")), key=lambda x: -x[1])
    print(f"[reranker] iters {booster.best_iteration} top15 {[(n,int(g)) for n,g in imp[:15]]}", flush=True)
    del fits_even, fits_odd

    # full-fold fits for eval
    tff = time.time()
    fits_full = build_fits(fold, args.nthreads, "full")
    T["full_fits"] = time.time() - tff
    te = time.time()
    astr, unions = build_union(val, fits_full, args.nproc)
    Xv, _, gv, texts = featurize_rows(val, fits_full, astr, unions, args.nproc, want_labels=False)
    T["val_candgen_feat"] = time.time() - te
    refs = val.target_span.astype(str).tolist()

    # oracle realized ratio
    orc = np.mean([max((f_pooled(c, r) for c in cs), default=0.0) for cs, r in zip(texts, refs)])
    starts = np.cumsum([0] + gv)
    exact = np.mean([refs[i] in texts[i] for i in range(len(val))])
    print(f"[val pool] mean {np.mean(gv):.1f} oracle {orc:.4f} exact {exact:.3f}", flush=True)

    argmax_preds = decode(booster, Xv, gv, texts, mbr_temp=0.0)
    chrf_arg = score_lists(argmax_preds, refs)
    print(f"[argmax] chrF {chrf_arg:.4f}  realized/oracle {chrf_arg/orc:.3f}", flush=True)

    best_t, best_mbr, best_preds = 0.0, chrf_arg, argmax_preds
    for temp in (0.4, 0.6, 0.8):
        mp = decode(booster, Xv, gv, texts, mbr_temp=temp, mbr_m=12)
        s = score_lists(mp, refs)
        print(f"[mbr T={temp}] chrF {s:.4f}", flush=True)
        if s > best_mbr:
            best_mbr, best_t, best_preds = s, temp, mp
    keep_mbr = (best_mbr - chrf_arg) >= 0.003
    final_preds = best_preds if keep_mbr else argmax_preds
    final = best_mbr if keep_mbr else chrf_arg
    T["total"] = time.time() - t0
    pd.DataFrame({"id": val.id.values, "prediction": final_preds}).to_csv(args.out, index=False)
    print(f"\n[TIMINGS] " + "  ".join(f"{k}={v:.0f}s" for k, v in T.items()), flush=True)
    print(f"HEADLINE bkt0 chrF: argmax {chrf_arg:.4f} | best_mbr {best_mbr:.4f} (T={best_t}) "
          f"| keep_mbr {keep_mbr} | FINAL {final:.4f} | oracle {orc:.4f} | realized {final/orc:.3f}",
          flush=True)


if __name__ == "__main__":
    main()

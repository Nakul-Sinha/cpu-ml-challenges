"""Docstring Gap Restoration -- self-contained solution (Agent B1 reranker v2).

End-to-end, CPU-only, single file:
  anchored multi-level retrieval indexes + code-derived + global candidates
  -> per-candidate features incl. a word n-gram fill-fluency LM (stupid backoff)
  -> LightGBM LambdaRank reranker (label = char n-gram F of candidate vs true)
  -> argmax decode.

The trained reranker materially drives every prediction. All statistics are raw
term-frequency counts / count-derived conditional probabilities -- NO idf / tf-idf
/ bm25. Leakage hygiene for the reranker's training candidates: parity half-indexes
over the training set (an even-bucket row draws candidates from the odd half and
vice-versa), so no row sees itself or its twins. Test rows are never in any training
index, so test candidates use the full-train index directly. Nothing is fit on test.

Reads CSVs with keep_default_na=False (spans like 'nan'/'null' are real text).
"""
import os, sys, re, time, math, hashlib, collections, glob as _glob
os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import pandas as pd
import lightgbm as lgb

GAP = "[GAP]"
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEF_RE = re.compile(r"def\s+(\w+)\s*\(([^)]*)\)")
LEVELS = [("l2r2", 2, 2), ("l2r1", 2, 1), ("l1r2", 1, 2), ("l1r1", 1, 1),
          ("l1", 1, 0), ("r1", 0, 1), ("l2", 2, 0), ("r2", 0, 2)]
LEVEL_TOPK = {"l2r2": 8, "l2r1": 6, "l1r2": 6, "l1r1": 10,
              "l1": 6, "r1": 6, "l2": 4, "r2": 4}
LVLW = {"l2r2": 3.0, "l2r1": 2.2, "l1r2": 2.2, "l1r1": 1.6,
        "l1": 1.0, "r1": 1.0, "l2": 0.8, "r2": 0.8}
RANK_CAP = 40
NTRAIN = int(os.environ.get("SOLN_NTRAIN", "60000"))   # reranker training rows
NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", "10"))
_TESTN = int(os.environ.get("SOLN_TESTN", "0"))        # >0 truncates test (smoke test only)


# ---------------- scorer (char n-gram F, pooled n=1..6) ----------------
def _ngrams(s, n):
    return collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))


def f_pooled(pred, ref, nmax=6):
    pred, ref = str(pred), str(ref)
    if not pred and not ref:
        return 1.0
    match = tp = tr = 0
    for n in range(1, nmax + 1):
        cp = _ngrams(pred, n); cr = _ngrams(ref, n)
        tp += sum(cp.values()); tr += sum(cr.values())
        match += sum(min(v, cr[k]) for k, v in cp.items())
    if tp == 0 or tr == 0 or match == 0:
        return 0.0
    p, r = match / tp, match / tr
    return 2 * p * r / (p + r)


# ---------------- context / candidates / LM ----------------
def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20


def row_ctx(masked):
    i = masked.find(GAP)
    left = masked[:i].split()
    right = masked[i + len(GAP):].split()
    keys = {}
    for name, nl, nr in LEVELS:
        L = " ".join(left[-nl:]) if nl else ""
        R = " ".join(right[:nr]) if nr else ""
        keys[name] = (L, R)
    return {"keys": keys, "left": left, "right": right, "n_left": len(left),
            "n_right": len(right), "masked_len": len(masked),
            "gap_pos": (i / len(masked)) if masked else 0.0}


def code_features(code):
    cands = []
    m = DEF_RE.search(code)
    if m:
        words = [w for w in re.split(r"_+", m.group(1)) if w]
        if words:
            cands.append(" ".join(words)); cands += words
        for a in m.group(2).split(",")[:4]:
            a = a.split("=")[0].split(":")[0].strip()
            if a and a not in ("self", "cls"):
                cands.append(a.replace("_", " "))
    for rl in re.findall(r"return\s+([A-Za-z_][\w\. ]*)", code)[:2]:
        toks = [t for t in re.split(r"[^A-Za-z0-9_]+", rl) if t][:2]
        if toks:
            cands.append(" ".join(w.replace("_", " ") for w in toks))
    idents = set(t.lower() for t in IDENT_RE.findall(code))
    seen, out = set(), []
    for c in cands:
        c = c.strip()
        if c and c.lower() not in seen:
            seen.add(c.lower()); out.append(c)
    return out[:10], idents


def build_index(df):
    idx = {name: collections.defaultdict(collections.Counter) for name, _, _ in LEVELS}
    glob = collections.Counter()
    for masked, tgt in zip(df.masked_docstring.values, df.target_span.values):
        tgt = str(tgt); rc = row_ctx(masked)
        for name, _, _ in LEVELS:
            idx[name][rc["keys"][name]][tgt] += 1
        glob[tgt] += 1
    return idx, glob


def global_top(glob, k=12):
    return [c for c, _ in glob.most_common(k)]


def _toks(s):
    return s.lower().split()


def build_lm(df):
    uni, bi, tri = collections.Counter(), collections.Counter(), collections.Counter()
    for masked, tgt in zip(df.masked_docstring.values, df.target_span.values):
        i = masked.find(GAP)
        seq = _toks(masked[:i]) + _toks(str(tgt)) + _toks(masked[i + len(GAP):])
        uni.update(seq)
        for a in range(len(seq) - 1):
            bi[(seq[a], seq[a + 1])] += 1
        for a in range(len(seq) - 2):
            tri[(seq[a], seq[a + 1], seq[a + 2])] += 1
    return {"uni": uni, "bi": bi, "tri": tri, "N": sum(uni.values()), "V": max(len(uni), 1)}


def _sb_prob(lm, w, c1, c2):
    if c2 is not None and c1 is not None:
        ctx = lm["bi"].get((c2, c1), 0)
        if ctx:
            t = lm["tri"].get((c2, c1, w), 0)
            if t:
                return t / ctx
    if c1 is not None:
        u1 = lm["uni"].get(c1, 0)
        if u1:
            b = lm["bi"].get((c1, w), 0)
            if b:
                return 0.4 * b / u1
    return 0.16 * (lm["uni"].get(w, 0) + 1.0) / (lm["N"] + lm["V"])


def lm_features(lm, lw, rw, cand):
    lw = [w.lower() for w in lw]; rw = [w.lower() for w in rw]
    cw = cand.lower().split()
    if not cw:
        return 0.0, 0.0
    window = lw[-2:] + cw + rw[:1]
    start = len(lw[-2:])
    total = 0.0; n = 0; first = 0.0
    for j in range(start, len(window)):
        c1 = window[j - 1] if j - 1 >= 0 else None
        c2 = window[j - 2] if j - 2 >= 0 else None
        lp = math.log(_sb_prob(lm, window[j], c1, c2))
        total += lp
        if j == start:
            first = lp
        n += 1
    return (total / n if n else 0.0), first


def make_stat_cache():
    return {}


def level_stats(idx, level, key, cache):
    ck = (level, key)
    v = cache.get(ck)
    if v is None:
        counter = idx[level].get(key)
        if counter is None:
            v = (0, {})
        else:
            total = sum(counter.values())
            rankmap = {c: (r, cnt) for r, (c, cnt) in enumerate(counter.most_common(RANK_CAP))}
            v = (total, rankmap)
        cache[ck] = v
    return v


def gen_candidates(rc, idx, gtop, code_cands):
    order, src = [], {}

    def add(c, s):
        if not c:
            return
        if c not in src:
            src[c] = set(); order.append(c)
        src[c].add(s)
    for name, _, _ in LEVELS:
        counter = idx[name].get(rc["keys"][name])
        if counter:
            for c, _ in counter.most_common(LEVEL_TOPK[name]):
                add(c, name)
    for c in code_cands:
        add(c, "code")
    for c in gtop:
        add(c, "global")
    return order, src


FEAT_NAMES = []
for _n, _, _ in LEVELS:
    FEAT_NAMES += [f"{_n}_prob", f"{_n}_present", f"{_n}_cntlog", f"{_n}_rank"]
FEAT_NAMES += ["glob_freq_log", "char_len", "word_len", "in_code", "ovl_cnt",
               "ovl_frac", "src_code", "src_global", "n_levels_hit", "max_prob",
               "deepest_lvl", "best_rank", "masked_len", "n_left", "n_right",
               "gap_pos", "code_len_log", "n_code_ident", "len_minus_typ",
               "lm_mean_logp", "lm_first_logp"]
N_FEAT = len(FEAT_NAMES)


def featurize(cands, src, rc, idx, glob, cache, code, idents, lm):
    n = len(cands)
    X = np.zeros((n, N_FEAT), dtype=np.float32)
    code_low = code.lower()
    code_len_log = np.log1p(len(code))
    n_ident = len(idents)
    lw2, rw1 = rc["left"][-2:], rc["right"][:1]
    for i, c in enumerate(cands):
        col = 0; n_hit = 0; max_prob = 0.0; deepest = -1; best_rank = RANK_CAP
        for li, (name, _, _) in enumerate(LEVELS):
            total, rankmap = level_stats(idx, name, rc["keys"][name], cache)
            info = rankmap.get(c)
            if info is not None and total > 0:
                rank, cnt = info; prob = cnt / total
                X[i, col] = prob; X[i, col + 1] = 1.0
                X[i, col + 2] = np.log1p(cnt); X[i, col + 3] = rank
                n_hit += 1
                if prob > max_prob:
                    max_prob = prob
                if deepest < 0:
                    deepest = li
                if rank < best_rank:
                    best_rank = rank
            else:
                X[i, col + 3] = RANK_CAP
            col += 4
        cl = c.lower()
        wl = set(w for w in re.split(r"[^a-z0-9]+", cl) if w)
        ovl = len(wl & idents)
        X[i, col + 0] = np.log1p(glob.get(c, 0)); X[i, col + 1] = len(c)
        X[i, col + 2] = len(c.split()); X[i, col + 3] = 1.0 if cl in code_low else 0.0
        X[i, col + 4] = ovl; X[i, col + 5] = (ovl / len(wl)) if wl else 0.0
        X[i, col + 6] = 1.0 if "code" in src[c] else 0.0
        X[i, col + 7] = 1.0 if "global" in src[c] else 0.0
        X[i, col + 8] = n_hit; X[i, col + 9] = max_prob
        X[i, col + 10] = deepest; X[i, col + 11] = best_rank
        X[i, col + 12] = rc["masked_len"]; X[i, col + 13] = rc["n_left"]
        X[i, col + 14] = rc["n_right"]; X[i, col + 15] = rc["gap_pos"]
        X[i, col + 16] = code_len_log; X[i, col + 17] = n_ident
        X[i, col + 18] = len(c) - 12.0
        lm_mean, lm_first = lm_features(lm, lw2, rw1, c)
        X[i, col + 19] = lm_mean; X[i, col + 20] = lm_first
    return X


def build_training_matrix(df, idx_e, idx_o, glob, gtop, ce, co, lm_e, lm_o):
    Xs, ys, groups = [], [], []
    for masked, tgt, bkt, code in zip(df.masked_docstring.values, df.target_span.values,
                                      df._bkt.values, df.code_context.values):
        tgt = str(tgt); use_odd = (bkt % 2 == 0)
        idx = idx_o if use_odd else idx_e
        cache = co if use_odd else ce
        lm = lm_o if use_odd else lm_e
        rc = row_ctx(masked); cc, idents = code_features(code)
        cands, src = gen_candidates(rc, idx, gtop, cc)
        if not cands:
            continue
        Xs.append(featurize(cands, src, rc, idx, glob, cache, code, idents, lm))
        ys.append(np.fromiter((f_pooled(c, tgt) for c in cands), np.float32, len(cands)))
        groups.append(len(cands))
    return np.vstack(Xs), np.concatenate(ys), np.array(groups)


def predict(df, idx, glob, gtop, cache, booster, lm, mbr_temp=0.6, mbr_m=12):
    """Decode each row by MBR under the reranker-softmax posterior (T=0.6) over
    the top-mbr_m candidates -- exploits chrF partial credit; the trained
    reranker supplies both the candidate subset and the posterior weights."""
    preds = []
    for masked, code in zip(df.masked_docstring.values, df.code_context.values):
        rc = row_ctx(masked); cc, idents = code_features(code)
        cands, src = gen_candidates(rc, idx, gtop, cc)
        if not cands:
            preds.append("value of the"); continue
        X = featurize(cands, src, rc, idx, glob, cache, code, idents, lm)
        rr = booster.predict(X)
        if len(cands) == 1 or mbr_temp <= 0:
            preds.append(cands[int(np.argmax(rr))]); continue
        sel = list(np.argsort(-rr)[:mbr_m])
        cs = [cands[j] for j in sel]; k = len(cs)
        K = np.empty((k, k), dtype=np.float32)
        for a in range(k):
            K[a, a] = 1.0
            for b in range(a + 1, k):
                v = f_pooled(cs[a], cs[b]); K[a, b] = v; K[b, a] = v
        w = np.asarray(rr[sel], dtype=np.float64) / mbr_temp
        w = np.exp(w - w.max()); w /= w.sum()
        preds.append(cs[int(np.argmax(K @ w))])
    return preds


def find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in [".", "dataset", "../dataset", os.path.join(here, "dataset"),
              os.path.join(here, "..", "dataset"), here, os.path.join(here, "..")]:
        if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(os.path.join(d, "test.csv")):
            return d
    hits = _glob.glob(os.path.join(here, "**", "train.csv"), recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("train.csv/test.csv not found")


def main():
    t0 = time.time()
    dd = find_data_dir()
    train = pd.read_csv(os.path.join(dd, "train.csv"), keep_default_na=False)
    test = pd.read_csv(os.path.join(dd, "test.csv"), keep_default_na=False)
    if _TESTN > 0:
        test = test.head(_TESTN)
    train["_bkt"] = train.masked_docstring.map(bucket)
    print(f"[load] train {len(train)} test {len(test)}  {time.time()-t0:.0f}s", flush=True)

    even = train[train._bkt % 2 == 0]; odd = train[train._bkt % 2 == 1]
    idx_e, ge = build_index(even); idx_o, go = build_index(odd)
    glob_all = ge + go; gtop = global_top(glob_all, 12)
    lm_e, lm_o = build_lm(even), build_lm(odd)
    print(f"[parity idx+lm] {time.time()-t0:.0f}s", flush=True)

    samp = train.sample(n=min(NTRAIN, len(train)), random_state=7).reset_index(drop=True)
    n_es = int(len(samp) * 0.08)
    ce, co = make_stat_cache(), make_stat_cache()
    Xfit, yfit, gfit = build_training_matrix(samp.iloc[n_es:], idx_e, idx_o, glob_all, gtop, ce, co, lm_e, lm_o)
    Xes, yes, ges = build_training_matrix(samp.iloc[:n_es], idx_e, idx_o, glob_all, gtop, ce, co, lm_e, lm_o)
    print(f"[train_feat] {Xfit.shape}  {time.time()-t0:.0f}s", flush=True)

    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                  label_gain=list(range(11)), lambdarank_truncation_level=20,
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
                  num_threads=NUM_THREADS, max_bin=255, verbose=-1)
    dtr = lgb.Dataset(Xfit, np.minimum((yfit * 10).astype(int), 10), group=gfit, feature_name=FEAT_NAMES)
    dva = lgb.Dataset(Xes, np.minimum((yes * 10).astype(int), 10), group=ges, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=700, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    print(f"[train] iters {booster.best_iteration}  {time.time()-t0:.0f}s", flush=True)

    idx_full, glob_full = build_index(train)
    gtop_full = global_top(glob_full, 12)
    lm_full = build_lm(train)
    cache = make_stat_cache()
    preds = predict(test, idx_full, glob_full, gtop_full, cache, booster, lm_full)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.csv")
    pd.DataFrame({"id": test.id.values, "prediction": preds}).to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} ({len(preds)} rows)  TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

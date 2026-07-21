"""Docstring Gap Restoration -- self-contained end-to-end solution (Agent C1, v2).

Integrates three trained components into a single-file, CPU-only pipeline:

  CANDIDATES  = B2 PoolBuilder (anchored multi-key indexes + gated fuzzy char n-gram
                KNN + learned code-prior candidates + global spans; gate_fuzz, cap 80)
                UNION  B3 LMBridge top-12 word-trigram stupid-backoff bridge spans.
                (A neural span-classifier pool was measured and EXCLUDED: it lifted
                union oracle by only +0.007 on a 3k probe, < the +0.01 bar.)
  RERANKER    = LightGBM LambdaRank over ~78 features: B1's 53 (anchored
                prob/rank/present, global freq, lexical overlap, word-trigram
                fill-fluency LM) + B2 provenance (source one-hots, tier, score,
                fuzzy cosine, learned code target-prior, anchor strength) + B3 LM
                metadata (entering/leaving logp, per-word, rank, in-pool, consensus)
                + centrality (mean pooled-chrF of a candidate vs the top-8-by-tier).
                Label = min(int(10 * f_pooled(candidate, true_span)), 10).
  DECODE      = argmax of the reranker score (MBR over the softmax posterior was
                swept on the bucket-0 holdout and did not clear +0.003 -- the
                centrality feature already captures the MBR signal -- so argmax).

Leakage hygiene: the reranker's TRAINING candidates/features use parity half-fits
of every module -- an even-bucket training row draws from the ODD half and vice
versa (twins share masked_docstring -> same bucket -> same parity), so no row sees
itself or its twins. TEST rows are absent from the train fit, so they use the full
-train fit directly. Nothing is ever fit on the test set.

Compliance: every statistic is a raw term-frequency count or a count-derived
conditional probability / cosine over hashed raw term frequencies. NO idf / tf-idf
/ bm25 anywhere. HashingVectorizer uses alternate_sign=False, norm='l2'. All models
(LightGBM reranker, the count LMs, the fuzzy index, the learned priors) are trained
inside this script from the provided training data only; no synthetic data.

Usage:  python3 solution.py <public_dir> <submission_out>
        (<public_dir> holds train.csv + test.csv; falls back to auto-detection.)
Reads CSVs with keep_default_na=False (spans like 'nan'/'null' are real text).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, re, time, math, hashlib, collections, glob as _glob, multiprocessing
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

GAP = "[GAP]"
NUM_THREADS = int(os.environ.get("SOLN_THREADS", str(min(multiprocessing.cpu_count(), 10))))
NPROC = int(os.environ.get("SOLN_NPROC", str(NUM_THREADS)))
NTRAIN = int(os.environ.get("SOLN_NTRAIN", "58000"))
_TESTN = int(os.environ.get("SOLN_TESTN", "0"))   # >0 truncates test (smoke only)
WB = 1.0            # LMBridge word_bonus
LM_TOPN = 12
BEAM = 16
LM_FLOOR = -40.0
CAP = 80


# ============================ scorer (char n-gram F, pooled) ============================
def f_pooled(pred, ref, nmax=6):
    pred, ref = str(pred), str(ref)
    if not pred and not ref:
        return 1.0
    match = tp = tr = 0
    for n in range(1, nmax + 1):
        cp = Counter(pred[i:i + n] for i in range(len(pred) - n + 1))
        cr = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        tp += sum(cp.values()); tr += sum(cr.values())
        match += sum(min(v, cr[k]) for k, v in cp.items())
    if tp == 0 or tr == 0 or match == 0:
        return 0.0
    p, r = match / tp, match / tr
    return 2 * p * r / (p + r)


def score_lists(preds, refs):
    return sum(f_pooled(p, r) for p, r in zip(preds, refs)) / len(refs)


# ================================= B1: anchored index + LM =================================
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEF_RE = re.compile(r"def\s+(\w+)\s*\(([^)]*)\)")
LEVELS = [("l2r2", 2, 2), ("l2r1", 2, 1), ("l1r2", 1, 2), ("l1r1", 1, 1),
          ("l1", 1, 0), ("r1", 0, 1), ("l2", 2, 0), ("r2", 0, 2)]
RANK_CAP = 40


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
    idx = {name: defaultdict(Counter) for name, _, _ in LEVELS}
    glob = Counter()
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
    uni, bi, tri = Counter(), Counter(), Counter()
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


B1_FEAT_NAMES = []
for _n, _, _ in LEVELS:
    B1_FEAT_NAMES += [f"{_n}_prob", f"{_n}_present", f"{_n}_cntlog", f"{_n}_rank"]
B1_FEAT_NAMES += ["glob_freq_log", "char_len", "word_len", "in_code", "ovl_cnt",
                  "ovl_frac", "src_code", "src_global", "n_levels_hit", "max_prob",
                  "deepest_lvl", "best_rank", "masked_len", "n_left", "n_right",
                  "gap_pos", "code_len_log", "n_code_ident", "len_minus_typ",
                  "lm_mean_logp", "lm_first_logp"]
B1_N_FEAT = len(B1_FEAT_NAMES)


def b1_featurize(cands, src, rc, idx, glob, cache, code, idents, lm):
    n = len(cands)
    X = np.zeros((n, B1_N_FEAT), dtype=np.float32)
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


# ================================= B2: PoolBuilder =================================
_word = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ident = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _norm_tok(t):
    return t.strip(".,;:!?\"'()[]{}`").lower()


def split_ident(name):
    parts = []
    for chunk in name.split("_"):
        if not chunk:
            continue
        parts += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", chunk)
    return [p.lower() for p in parts if p]


def window_text(masked, w=4):
    i = masked.find(GAP)
    if i < 0:
        return masked
    L = masked[:i].split()[-w:]
    R = masked[i + len(GAP):].split()[:w]
    return " ".join(L) + " " + " ".join(R)


def anchors(masked):
    i = masked.find(GAP)
    if i < 0:
        L, R = masked.split(), []
    else:
        L = masked[:i].split()
        R = masked[i + len(GAP):].split()
    l1 = L[-1] if L else ""
    l2a = L[-2] if len(L) >= 2 else ""
    r1 = R[0] if R else ""
    r2b = R[1] if len(R) >= 2 else ""
    ln1 = _norm_tok(l1); rn1 = _norm_tok(r1)
    return {"l2r2": (" ".join(L[-2:]), " ".join(R[:2])), "l1r1": (l1, r1),
            "l1": (l1,), "r1": (r1,), "l1r1n": (ln1, rn1),
            "skipR": (l1, r2b), "skipL": (l2a, r1)}


SRC_TIER = {"l2r2": 10, "l1r1": 9, "codeP": 8, "skipR": 7, "skipL": 7,
            "fuzz": 6, "fuzzw": 6, "r1": 5, "l1": 5, "code": 2, "global": 1}
ANCHOR_KEYS = ["l2r2", "l1r1", "skipR", "skipL", "r1", "l1"]


class PoolBuilder:
    def __init__(self, topk_anchor=12, topk_fuzz=20, cap=80, fuzz_ngram=(4, 5),
                 fuzz_feats=2 ** 18, n_threads=5, gate_fuzz=False, use_window=True,
                 topk_fuzzw=12, n_global=50):
        self.topk_anchor = topk_anchor; self.topk_fuzz = topk_fuzz
        self.topk_fuzzw = topk_fuzzw; self.cap = cap; self.fuzz_ngram = fuzz_ngram
        self.fuzz_feats = fuzz_feats; self.n_threads = n_threads
        self.gate_fuzz = gate_fuzz; self.use_window = use_window; self.n_global = n_global

    def fit(self, train_df):
        t0 = time.time()
        masked = train_df.masked_docstring.values
        tgts = train_df.target_span.astype(str).values
        idx = {k: defaultdict(Counter) for k in ANCHOR_KEYS}
        for m, tg in zip(masked, tgts):
            a = anchors(m)
            for k in ANCHOR_KEYS:
                key = a[k]
                if key[0] == "" and (len(key) == 1 or key[-1] == ""):
                    continue
                idx[k][key][tg] += 1
        self.idx = {}
        for k, d in idx.items():
            self.idx[k] = {key: c.most_common(self.topk_anchor) for key, c in d.items()}
        gc = Counter(tgts)
        self.global_top = [t for t, _ in gc.most_common(self.n_global)]
        self.n_train = len(tgts)
        self.target_prior = gc
        self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=self.fuzz_ngram,
                                    n_features=self.fuzz_feats, lowercase=True,
                                    alternate_sign=False, norm="l2")
        self.Xtr = self.hv.transform(masked)
        self.XtrT = self.Xtr.T.tocsr()
        self.train_targets = tgts
        if self.use_window:
            wtxt = [window_text(m) for m in masked]
            self.hvw = HashingVectorizer(analyzer="char_wb", ngram_range=(4, 5),
                                         n_features=self.fuzz_feats, lowercase=True,
                                         alternate_sign=False, norm="l2")
            self.XtrwT = self.hvw.transform(wtxt).T.tocsr()
        self.t_fit = time.time() - t0
        return self

    def _code_cands(self, code):
        out = []
        m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", code)
        if m:
            name = m.group(1)
            words = split_ident(name)
            if words:
                out.append(" ".join(words)); out += words
                if len(words) >= 2:
                    out.append(" ".join(words[:2])); out.append(" ".join(words[-2:]))
            for a in m.group(2).split(","):
                a = a.split("=")[0].split(":")[0].strip().lstrip("*")
                if a and a not in ("self", "cls"):
                    out.append(a.replace("_", " "))
                    aw = split_ident(a)
                    if len(aw) >= 2:
                        out.append(" ".join(aw))
        for rm in re.finditer(r"return\s+([^\n]+)", code):
            for tok in _ident.findall(rm.group(1))[:6]:
                w = split_ident(tok)
                if w:
                    out.append(" ".join(w))
        scored, seen = [], set()
        for c in out:
            if c in seen:
                continue
            seen.add(c)
            scored.append((c, self.target_prior.get(c, 0)))
        scored.sort(key=lambda x: -x[1])
        kept = [(c, pr) for c, pr in scored if pr > 0][:6]
        extra = [(c, 0) for c, pr in scored if pr == 0][:4]
        return kept + extra

    def _fuzz_knn(self, Xq, k, XT, chunk=256):
        n = Xq.shape[0]
        if n == 0:
            return np.empty((0, k), np.int32), np.empty((0, k), np.float32)
        oi = np.zeros((n, k), np.int32); os_ = np.zeros((n, k), np.float32)

        def worker(rng):
            s, e = rng
            for st in range(s, e, chunk):
                en = min(st + chunk, e)
                sims = (Xq[st:en] @ XT).toarray()
                kk = min(k, sims.shape[1])
                part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
                rr = np.arange(sims.shape[0])[:, None]
                pv = sims[rr, part]
                order = np.argsort(-pv, axis=1)
                oi[st:en] = part[rr, order]; os_[st:en] = pv[rr, order]

        if self.n_threads > 1 and n > chunk:
            bnds = np.linspace(0, n, self.n_threads + 1).astype(int)
            rngs = [(bnds[i], bnds[i + 1]) for i in range(self.n_threads) if bnds[i + 1] > bnds[i]]
            with ThreadPoolExecutor(len(rngs)) as ex:
                list(ex.map(worker, rngs))
        else:
            worker((0, n))
        return oi, os_

    def candidates_batch(self, df, use_fuzz=True):
        masked = df.masked_docstring.values
        codes = df.code_context.values
        n = len(df)
        pools = []
        anchor_strength = np.zeros(n, dtype=np.int8)
        for i in range(n):
            m = masked[i]; a = anchors(m); row = []
            for k in ANCHOR_KEYS:
                key = a[k]
                if key not in self.idx[k]:
                    continue
                lst = self.idx[k][key]
                tot = sum(c for _, c in lst) or 1
                for j, (t, c) in enumerate(lst):
                    row.append((t, k, SRC_TIER[k] + (c / tot)))
                if k in ("l2r2", "l1r1"):
                    anchor_strength[i] = max(anchor_strength[i], 2 if k == "l2r2" else 1)
            for c, pr in self._code_cands(codes[i]):
                srcc = "codeP" if pr > 0 else "code"
                row.append((c, srcc, SRC_TIER[srcc] + min(pr / self.n_train, 0.5)))
            for gi, g in enumerate(self.global_top[:12]):
                row.append((g, "global", SRC_TIER["global"] - gi * 0.01))
            pools.append(row)
        if use_fuzz:
            fmask = (anchor_strength < 1) if self.gate_fuzz else np.ones(n, dtype=bool)
            fidx = np.where(fmask)[0]
            if len(fidx) > 0:
                Xq = self.hv.transform(masked[fidx])
                nbr, nsim = self._fuzz_knn(Xq, self.topk_fuzz, self.XtrT)
                for r, gi in enumerate(fidx):
                    seen = set()
                    for j in range(nbr.shape[1]):
                        t = self.train_targets[nbr[r, j]]
                        if t in seen:
                            continue
                        seen.add(t)
                        pools[gi].append((t, "fuzz", SRC_TIER["fuzz"] + float(nsim[r, j])))
                if self.use_window:
                    Xw = self.hvw.transform([window_text(masked[i]) for i in fidx])
                    wbr, wsim = self._fuzz_knn(Xw, self.topk_fuzzw, self.XtrwT)
                    for r, gi in enumerate(fidx):
                        seen = set()
                        for j in range(wbr.shape[1]):
                            t = self.train_targets[wbr[r, j]]
                            if t in seen:
                                continue
                            seen.add(t)
                            pools[gi].append((t, "fuzzw", SRC_TIER["fuzzw"] + float(wsim[r, j]) * 0.9))
        out = []
        for row in pools:
            best = {}
            for t, s, sc in row:
                if t not in best or sc > best[t][1]:
                    best[t] = (s, sc)
            merged = [(t, v[0], v[1]) for t, v in best.items()]
            merged.sort(key=lambda x: -x[2])
            out.append(merged[:self.cap])
        return out


# ================================= B3: LMBridge =================================
BOS = "<s>"
LOGA = math.log(0.4)
_tok_re = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_ident_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(s):
    return _tok_re.findall(s)


def split_ctx(masked):
    i = masked.find(GAP)
    return tokenize(masked[:i]), tokenize(masked[i + len(GAP):])


def code_words(code):
    out, seen = [], set()
    for m in _ident_re.finditer(code):
        ident = m.group(0)
        if ident in ("self", "cls"):
            continue
        parts = re.split(r"_+", ident)
        toks = []
        for p in parts:
            toks += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", p) or [p]
        for t in toks:
            tl = t.lower()
            if len(tl) >= 2 and tl not in seen:
                seen.add(tl); out.append(tl)
        if len(out) > 40:
            break
    return out[:40]


class LMBridge:
    def __init__(self, order=3, topsucc=12, glob_target=40):
        self.order = order; self.topsucc = topsucc; self.glob_target = glob_target

    def fit(self, trn):
        c1, c2, c3, tw = Counter(), Counter(), Counter(), Counter()
        for masked, tgt in zip(trn.masked_docstring.values, trn.target_span.values):
            full = masked.replace(GAP, str(tgt))
            toks = [BOS, BOS] + tokenize(full) + [BOS]
            for w in toks:
                c1[w] += 1
            for i in range(len(toks) - 1):
                c2[(toks[i], toks[i + 1])] += 1
            for i in range(len(toks) - 2):
                c3[(toks[i], toks[i + 1], toks[i + 2])] += 1
            for w in tokenize(str(tgt)):
                if any(ch.isalnum() for ch in w):
                    tw[w] += 1
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.total1 = sum(c1.values()); self.V = len(c1)
        succ2 = defaultdict(list)
        for (a, b, c), n in c3.items():
            succ2[(a, b)].append((n, c))
        self.succ2 = {k: [w for _, w in sorted(v, reverse=True)[:self.topsucc]] for k, v in succ2.items()}
        succ1 = defaultdict(list)
        for (a, b), n in c2.items():
            succ1[a].append((n, b))
        self.succ1 = {k: [w for _, w in sorted(v, reverse=True)[:self.topsucc]] for k, v in succ1.items()}
        self.glob = [w for w, _ in tw.most_common(self.glob_target)]
        return self

    def sb(self, a, b, w):
        c3v = self.c3.get((a, b, w))
        if c3v:
            return math.log(c3v / self.c2[(a, b)])
        c2v = self.c2.get((b, w))
        if c2v:
            return LOGA + math.log(c2v / self.c1[b])
        c1v = self.c1.get(w, 0)
        return 2 * LOGA + math.log((c1v + 0.5) / (self.total1 + 0.5 * self.V))

    def _cands(self, a, b, cw):
        c = self.succ2.get((a, b))
        if c is None:
            c = self.succ1.get(b, [])
        seen = {BOS}; res = []
        for w in c:
            if w not in seen and any(ch.isalnum() for ch in w):
                seen.add(w); res.append(w)
        for w in cw[:12]:
            if w not in seen:
                seen.add(w); res.append(w)
        for w in self.glob[:20]:
            if w not in seen:
                seen.add(w); res.append(w)
        return res

    def leaving(self, prefix, right):
        if not right:
            return 0.0
        seq = prefix + right; s = 0.0
        for i in range(len(prefix), min(len(prefix) + 2, len(seq))):
            a = seq[i - 2] if i >= 2 else BOS
            b = seq[i - 1] if i >= 1 else BOS
            s += self.sb(a, b, seq[i])
        return s

    def pool(self, left, right, cw, beam=16, maxlen=4, prune_wb=1.5, keep=24):
        fleft = [BOS, BOS] + left
        completed = []
        beams = [(0.0, [])]
        for _ in range(maxlen):
            newb = []
            for esc, bridge in beams:
                seq = fleft + bridge
                a, b = seq[-2], seq[-1]
                for w in self._cands(a, b, cw):
                    nesc = esc + self.sb(a, b, w)
                    nbridge = bridge + [w]
                    lv = self.leaving(fleft + nbridge, right)
                    completed.append((nesc, lv, len(nbridge), " ".join(nbridge)))
                    newb.append((nesc + prune_wb * len(nbridge), nesc, nbridge))
            newb.sort(key=lambda x: x[0], reverse=True)
            beams = [(e, br) for _, e, br in newb[:beam]]
        completed.sort(key=lambda x: x[0] + prune_wb * x[2] + x[1], reverse=True)
        seen, out = set(), []
        for e, lv, ln, txt in completed:
            if txt in seen:
                continue
            seen.add(txt); out.append((e, lv, ln, txt))
            if len(out) >= keep:
                break
        return out


# ================================= integration =================================
EXT_NAMES = [
    "in_b2", "b2_tier", "b2_score", "b2_fuzz_cos", "b2_code_prior_log", "b2_anchor_strength",
    "b2src_l2r2", "b2src_l1r1", "b2src_skip", "b2src_r1", "b2src_l1", "b2src_codeP", "b2src_code",
    "b2src_fuzz", "b2src_fuzzw", "b2src_global",
    "in_lm", "lm_enter", "lm_leave", "lm_enter_pw", "lm_total_pw", "lm_rank", "lm_len",
    "consensus", "centrality8",
]
FEAT_NAMES = list(B1_FEAT_NAMES) + EXT_NAMES
N_FEAT = len(FEAT_NAMES)
_SRC1H = {"l2r2": 6, "l1r1": 7, "skipR": 8, "skipL": 8, "r1": 9, "l1": 10,
          "codeP": 11, "code": 12, "fuzz": 13, "fuzzw": 14, "global": 15}


class Fits:
    def __init__(self, idx, glob, gtop, lm, pb, lmb):
        self.idx, self.glob, self.gtop, self.lm = idx, glob, gtop, lm
        self.pb, self.lmb = pb, lmb


def build_fits(df, n_threads, tag=""):
    t0 = time.time()
    idx, glob = build_index(df)
    gtop = global_top(glob, 12)
    lm = build_lm(df)
    pb = PoolBuilder(cap=CAP, gate_fuzz=True, n_threads=n_threads).fit(df)
    lmb = LMBridge().fit(df)
    print(f"[fits {tag}] n={len(df)} {time.time()-t0:.0f}s", flush=True)
    return Fits(idx, glob, gtop, lm, pb, lmb)


# ---- B3 LM candgen (fork-parallel) ----
_LMB = None


def _lm_chunk(rows):
    out = []
    for masked, code in rows:
        l, r = split_ctx(masked)
        pool = _LMB.pool(l, r, code_words(code), beam=BEAM)
        scored = sorted(pool, key=lambda x: x[0] + WB * x[2] + x[1], reverse=True)[:LM_TOPN]
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
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as ex:
            res = list(ex.map(_lm_chunk, chunks))
    out = []
    for r in res:
        out.extend(r)
    return out


def anchor_strength(masked, pb):
    a = anchors(masked)
    if a["l2r2"] in pb.idx["l2r2"]:
        return 2
    if a["l1r1"] in pb.idx["l1r1"]:
        return 1
    return 0


def build_union(df, fits, n_proc):
    b2_pools = fits.pb.candidates_batch(df, use_fuzz=True)
    lm_pools = lm_candidates(df, fits.lmb, n_proc)
    masked = df.masked_docstring.values
    unions, astr = [], []
    for i in range(len(df)):
        d = {}
        for t, s, sc in b2_pools[i]:
            d[t] = [t, True, s, float(sc), int(SRC_TIER.get(s, 0)), False, 0.0, 0.0, 0, LM_TOPN]
        for (t, e, lv, ln, rk) in lm_pools[i]:
            if t in d:
                row = d[t]; row[5] = True; row[6] = e; row[7] = lv; row[8] = ln; row[9] = rk
            else:
                d[t] = [t, False, "", 0.0, 0, True, e, lv, ln, rk]
        unions.append([tuple(v) for v in d.values()])
        astr.append(anchor_strength(masked[i], fits.pb))
    return astr, unions


# ---- featurization (fork-parallel) ----
_G = {}
_FCACHE = None


def _grams(s):
    c = Counter(); L = len(s)
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
            E[i, 17] = lm_e; E[i, 18] = lm_lv
            E[i, 19] = lm_e / lm_len if lm_len else lm_e
            E[i, 20] = (lm_e + lm_lv) / lm_len if lm_len else (lm_e + lm_lv)
            E[i, 21] = lm_rk; E[i, 22] = lm_len
        else:
            E[i, 17] = LM_FLOOR; E[i, 18] = LM_FLOOR
            E[i, 19] = LM_FLOOR; E[i, 20] = LM_FLOOR
            E[i, 21] = LM_TOPN; E[i, 22] = 0
        E[i, 23] = 1.0 if (in_b2 and in_lm) else 0.0
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
        rc = row_ctx(masked)
        cc, idents = code_features(code)
        cc_set = set(cc)
        src = {}
        for t in cands:
            s = set()
            if t in cc_set:
                s.add("code")
            if t in gtop_set:
                s.add("global")
            src[t] = s
        Xb1 = b1_featurize(cands, src, rc, idx, glob, _FCACHE, code, idents, lm)
        Xext = _ext_features(pool, astr, pb)
        Xs.append(np.hstack([Xb1, Xext]))
        ncands.append(len(cands)); texts.append(cands)
        if tgt is not None:
            ys.append(np.fromiter((f_pooled(c, tgt) for c in cands), np.float32, len(cands)))
    X = np.vstack(Xs) if Xs else np.zeros((0, N_FEAT), np.float32)
    y = np.concatenate(ys) if ys else None
    return X, y, ncands, texts


def featurize_rows(df, fits, astr, unions, n_proc, want_labels):
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
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as ex:
            res = list(ex.map(_feat_chunk, chunks))
    Xs, ys, groups, texts = [], [], [], []
    for X, y, nc, tx in res:
        Xs.append(X)
        if want_labels:
            ys.append(y)
        groups.extend(nc); texts.extend(tx)
    X = np.vstack(Xs) if Xs else np.zeros((0, N_FEAT), np.float32)
    y = np.concatenate(ys) if (want_labels and ys) else None
    return X, y, groups, texts


def build_training(train_sample, fits_even, fits_odd, n_proc):
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


def train_reranker(X, y, groups, n_threads, es_frac=0.08, seed=7):
    n = len(groups)
    n_es = max(1, int(n * es_frac))
    perm = np.random.RandomState(seed).permutation(n)
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


# ================================= driver =================================
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


def _resolve_paths():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    public_dir, out_path = None, None
    if len(args) >= 1 and os.path.isdir(args[0]) and \
       os.path.exists(os.path.join(args[0], "train.csv")):
        public_dir = args[0]
    if len(args) >= 2:
        out_path = args[1]
    if public_dir is None:
        public_dir = find_data_dir()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.csv")
    return public_dir, out_path


def _fallback_submission(test, out_path, const="value of the"):
    pd.DataFrame({"id": test.id.values, "prediction": [const] * len(test)}).to_csv(out_path, index=False)


def main():
    t0 = time.time()
    T = {}
    dd, out_path = _resolve_paths()
    train = pd.read_csv(os.path.join(dd, "train.csv"), keep_default_na=False)
    test = pd.read_csv(os.path.join(dd, "test.csv"), keep_default_na=False)
    if _TESTN > 0:
        test = test.head(_TESTN)
    train["_bkt"] = train.masked_docstring.map(bucket)
    print(f"[load] train {len(train)} test {len(test)} threads {NUM_THREADS} nproc {NPROC} "
          f"ntrain {NTRAIN}  {time.time()-t0:.0f}s", flush=True)
    # best-constant fallback ready in case anything downstream fails
    best_const = train.target_span.astype(str).value_counts().idxmax()

    try:
        even = train[train._bkt % 2 == 0]; odd = train[train._bkt % 2 == 1]
        tf = time.time()
        fits_even = build_fits(even, NUM_THREADS, "even")
        fits_odd = build_fits(odd, NUM_THREADS, "odd")
        T["parity_fits"] = time.time() - tf

        samp = train.sample(n=min(NTRAIN, len(train)), random_state=7).reset_index(drop=True)
        tt = time.time()
        Xtr, ytr, gtr = build_training(samp, fits_even, fits_odd, NPROC)
        T["train_candgen_feat"] = time.time() - tt
        print(f"[train matrix] X{Xtr.shape} groups {len(gtr)}  {T['train_candgen_feat']:.0f}s", flush=True)

        tr = time.time()
        booster = train_reranker(Xtr, ytr, gtr, NUM_THREADS)
        T["train"] = time.time() - tr
        imp = sorted(zip(FEAT_NAMES, booster.feature_importance("gain")), key=lambda x: -x[1])
        print(f"[reranker] iters {booster.best_iteration} top10 {[(n,int(g)) for n,g in imp[:10]]}", flush=True)
        del fits_even, fits_odd, Xtr, ytr

        tff = time.time()
        fits_full = build_fits(train, NUM_THREADS, "full")
        T["full_fits"] = time.time() - tff

        te = time.time()
        astr, unions = build_union(test, fits_full, NPROC)
        Xte, _, gte, texts = featurize_rows(test, fits_full, astr, unions, NPROC, want_labels=False)
        preds = decode(booster, Xte, gte, texts, mbr_temp=0.0, fallback=best_const)
        T["test_candgen_feat_decode"] = time.time() - te

        # guard: no empty predictions
        preds = [p if (p is not None and str(p) != "") else best_const for p in preds]
        pd.DataFrame({"id": test.id.values, "prediction": preds}).to_csv(out_path, index=False)
        print(f"\n[TIMINGS] " + "  ".join(f"{k}={v:.0f}s" for k, v in T.items()) +
              f"  TOTAL={time.time()-t0:.0f}s", flush=True)
        print(f"[done] wrote {out_path} ({len(preds)} rows)  "
              f"mean_len {np.mean([len(str(p)) for p in preds]):.1f}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FALLBACK] pipeline failed ({e!r}); writing best-constant submission", flush=True)
        _fallback_submission(test, out_path, best_const)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()

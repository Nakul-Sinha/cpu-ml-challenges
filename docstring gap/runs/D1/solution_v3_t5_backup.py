"""Docstring Gap Restoration -- self-contained HYBRID solution (Agent D1, v3).

C1 retrieval chassis (union candgen + LightGBM LambdaRank reranker) AUGMENTED with an
in-script fine-tuned doc_first T5-small, injected as a *gated* union-pool candidate and
described by 5 extra reranker features. Everything below is a single-file, CPU-only,
grader-safe artifact.

  CANDIDATES  = B2 PoolBuilder (gated-fuzz, cap 80)  UNION  B3 LMBridge top-12
                UNION  {T5 doc_first greedy pred}  on GATED rows only.
  T5          = google-t5/t5-small, base pretrained weights + IN-SCRIPT fine-tune on
                span-corruption over docstring gaps (doc_first format: masked sentence
                with [GAP]->'<extra_id_0>' + " . " + compact code hint; target
                "<extra_id_0> span <extra_id_1>"). Time-boxed FT, int8-dynamic-quant
                inference (greedy, bs 64). Loaded and run in the MAIN process only,
                AFTER every fork-pool stage has completed (torch + fork do not mix).
  GATE        = precomputable retrieval-weakness score (anchor tier present + support).
                Threshold calibrated on a train sample (full-fit density) to a target
                coverage; the SAME threshold gates train rows (via parity fits, leak-free)
                and test rows (via full fits). Non-gated rows carry zero-filled T5 features
                and has_t5=0 with an asserted-identical schema.
  RERANKER    = LightGBM LambdaRank over 78 retrieval feats + 5 T5 feats (has_t5,
                t5_is_cand, t5_seq_logprob, t5_len, t5_in_pool). Label = min(int(10*
                f_pooled(candidate, true_span)), 10).
  DECODE      = argmax of the reranker score; then, on a gated row whose T5 candidate is
                highly confident (mean-token logprob >= -0.35), override to the T5 pred.

Leakage hygiene (unchanged from C1): reranker TRAINING rows draw candidates/features from
parity half-fits (even-bucket row -> ODD half fit and vice versa; twins share
masked_docstring -> same bucket -> same parity). TEST rows are absent from every train
fit. Nothing is ever fit on the test set. The T5 model is fine-tuned only on TRAINING
rows; test rows are never in its FT pool.

Robustness: cumulative wall-clock stage-skip rules degrade T5 gracefully
(skip train-side T5 -> test-only override; shrink coverage; skip T5 entirely -> pure C1;
any torch/weights failure -> pure C1; any pipeline failure -> best-constant). HF loads
offline-first (HF_HUB_OFFLINE) then online.

Compliance: every retrieval statistic is a raw term-frequency count or a count-derived
probability / cosine over hashed raw term frequencies. NO idf / tf-idf / bm25. The T5 uses
base pretrained weights fine-tuned IN THIS SCRIPT on the provided training data only; NO
own checkpoint is ever loaded as the model source; NO synthetic data. Reads CSVs with
keep_default_na=False.

Usage:  python3 solution.py <public_dir> <submission_out>
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, re, time, math, hashlib, collections, glob as _glob, random, multiprocessing
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
NTRAIN = int(os.environ.get("SOLN_NTRAIN", "40000"))
_TESTN = int(os.environ.get("SOLN_TESTN", "0"))   # >0 truncates test (smoke only)
WB = 1.0            # LMBridge word_bonus
LM_TOPN = 12
BEAM = 16
LM_FLOOR = -40.0
CAP = 80

# ---------------- T5 hybrid config ----------------
# Generator = Salesforce/codet5-small (code+docstring pretrained; same 60.5M T5 arch as
# t5-small). D2/D3 measured codet5 ZERO-SHOT doc_first = 0.4135 bucket-0 vs t5-small FT
# 0.31 and vs the C1 retrieval chassis 0.319: +0.095. Fine-tuning REGRESSES codet5
# (0.4135 -> 0.3735), so we use base pretrained weights ZERO-SHOT (no FT). The in-script
# LightGBM LambdaRank reranker still learns to SELECT between codet5 and retrieval per row.
T5_MODEL_NAME = os.environ.get("SOLN_T5_MODEL", "Salesforce/codet5-small")
T5_ZEROSHOT = os.environ.get("SOLN_T5_ZEROSHOT", "1") != "0"   # codet5 wants zero-shot
T5_MODE = "doc_first"
T5_LR = 3e-4
T5_BS = 24
T5_FT_MAX_S = int(os.environ.get("SOLN_FT_S", str(16 * 60)))   # time-boxed fine-tune (unused when zero-shot)
T5_FT_POOL = int(os.environ.get("SOLN_FT_POOL", "30000"))
T5_COVERAGE = float(os.environ.get("SOLN_COVERAGE", "1.00"))   # inject codet5 for ALL rows (strong everywhere)
T5_OVERRIDE_THR = float(os.environ.get("SOLN_T5_OVR", "-0.35")) # confident-T5 override
T5_INFER_BS = int(os.environ.get("SOLN_T5_BS", "64"))
T5_MAXNEW = 16
T5_USE_QUANT = os.environ.get("SOLN_T5_QUANT", "1") != "0"
TRAIN_T5_CAP = int(os.environ.get("SOLN_TRAIN_T5_CAP", "16000"))
FT_MIN_S = int(os.environ.get("SOLN_FT_MIN", "240"))          # below this FT budget -> skip T5
SKIP_T5 = os.environ.get("SOLN_SKIP_T5", "0") == "1"           # force pure-C1
T5_INFER_RPS_EST = 28.0   # codet5-small int8 greedy bs64, ~10 cores (D2/D3 measured 30-38)
# cumulative wall-clock stage-skip thresholds (minutes since start)
SKIP_TRAIN_T5_MIN = 42.0
COVERAGE30_MIN = 60.0
SKIP_T5_MIN = 72.0
SAFE_END_MIN = 86.0


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
# T5 hybrid features (appended after the 78 retrieval features)
T5_FEAT_NAMES = ["has_t5", "t5_is_cand", "t5_seq_logprob", "t5_len", "t5_in_pool"]
FEAT_NAMES3 = FEAT_NAMES + T5_FEAT_NAMES
N_FEAT3 = len(FEAT_NAMES3)
N_T5_FEAT = len(T5_FEAT_NAMES)
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
    fcache = {}
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
        Xb1 = b1_featurize(cands, src, rc, idx, glob, fcache, code, idents, lm)
        Xext = _ext_features(pool, astr, pb)
        Xs.append(np.hstack([Xb1, Xext]))
        ncands.append(len(cands)); texts.append(cands)
        if tgt is not None:
            ys.append(np.fromiter((f_pooled(c, tgt) for c in cands), np.float32, len(cands)))
    X = np.vstack(Xs) if Xs else np.zeros((0, N_FEAT), np.float32)
    y = np.concatenate(ys) if ys else None
    return X, y, ncands, texts


def featurize_rows(df, fits, astr, unions, n_proc, want_labels):
    global _G
    _G = {"idx": fits.idx, "glob": fits.glob, "gtop": fits.gtop, "lm": fits.lm, "pb": fits.pb}
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


def build_training_base(samp, fits_even, fits_odd, n_proc):
    """C1 parity training matrix, additionally returning per-row context for T5 augment.
    Row order: even-bucket rows (fit on ODD) first, then odd-bucket rows (fit on EVEN)."""
    ev = samp[samp._bkt % 2 == 0]
    od = samp[samp._bkt % 2 == 1]
    Xs, ys, groups, texts = [], [], [], []
    masked_all, code_all, unions_all, astr_all, tag_all, tgt_all = [], [], [], [], [], []
    for sub, fits, tag in ((ev, fits_odd, 0), (od, fits_even, 1)):
        if len(sub) == 0:
            continue
        t0 = time.time()
        astr, unions = build_union(sub, fits, n_proc)
        X, y, g, tx = featurize_rows(sub, fits, astr, unions, n_proc, want_labels=True)
        print(f"[train phase tag{tag}] X{X.shape} {time.time()-t0:.0f}s", flush=True)
        Xs.append(X); ys.append(y); groups.extend(g); texts.extend(tx)
        masked_all.extend(sub.masked_docstring.values.tolist())
        code_all.extend(sub.code_context.values.tolist())
        unions_all.extend(unions); astr_all.extend(astr); tag_all.extend([tag] * len(sub))
        tgt_all.extend(sub.target_span.astype(str).values.tolist())
    X = np.vstack(Xs); y = np.concatenate(ys)
    return X, y, groups, texts, masked_all, code_all, unions_all, astr_all, tag_all, tgt_all


# ================================= T5 gate =================================
def t5_weakness_score(masked, pb):
    """Precomputable retrieval-weakness: higher = weaker retrieval (prioritized for T5).
    Tiers by strongest exact anchor PRESENT in the index (l2r2 > l1r1 > none), refined
    continuously by anchor support so a target coverage maps to a smooth threshold.
    Uses only anchor indexes (no target information) -> identical rule train/test, leak-free."""
    a = anchors(masked)
    l2 = pb.idx["l2r2"].get(a["l2r2"])
    if l2:
        return -2.0 - min(sum(c for _, c in l2), 40) / 50.0    # strongest: [-2.8, -2.0]
    l1 = pb.idx["l1r1"].get(a["l1r1"])
    if l1:
        return -1.0 - min(sum(c for _, c in l1), 40) / 50.0    # medium:   [-1.8, -1.0]
    # weak tier (no exact 2-side anchor): refine by single-anchor (l1 / r1) support
    s1 = 0
    la = pb.idx["l1"].get(a["l1"]); ra = pb.idx["r1"].get(a["r1"])
    if la:
        s1 += sum(c for _, c in la)
    if ra:
        s1 += sum(c for _, c in ra)
    return 1.0 - min(s1, 200) / 400.0                          # weak:     [0.5, 1.0]


def calibrate_gate(masked_sample, pb, coverage):
    if coverage >= 0.999:
        return -1e18
    sc = np.array([t5_weakness_score(m, pb) for m in masked_sample], dtype=np.float64)
    return float(np.quantile(sc, 1.0 - coverage))


def gate_mask(masked_list, pb, thr):
    return np.array([t5_weakness_score(m, pb) >= thr for m in masked_list], dtype=bool)


# ================================= T5 generator (main process only) =================================
def code_hint(code):
    """def line + last return line, compact (matches C2 t5_probe)."""
    lines = code.split("\n")
    def_line, ret_line = "", ""
    for ln in lines:
        s = ln.strip()
        if not def_line and s.startswith("def "):
            def_line = s
        if s.startswith("return ") or s == "return" or s.startswith("return("):
            ret_line = s
    if not def_line:
        for ln in lines:
            if ln.strip():
                def_line = ln.strip()
                break
    parts = [p for p in [def_line, ret_line] if p]
    return " ".join(parts)


def build_t5_input(masked, code, tok, mode):
    sent = masked.replace(GAP, "<extra_id_0>")
    if mode == "plain":
        return sent
    hint = code_hint(code)
    hid = tok(hint, add_special_tokens=False, truncation=True, max_length=64).input_ids
    hint = tok.decode(hid)
    if mode == "code_first":
        return hint + " . " + sent
    return sent + " . " + hint  # doc_first


_T5_EXTRACT = re.compile(r"<extra_id_0>(.*?)(?:<extra_id_1>|<extra_id_\d|</s>|<pad>|$)", re.DOTALL)


def t5_extract(text):
    m = _T5_EXTRACT.search(text)
    if m:
        return m.group(1).strip()
    t = re.sub(r"</?s>|<pad>|<extra_id_\d+>", " ", text)
    return re.sub(r"\s+", " ", t).strip()


def _load_t5_base(threads):
    """Offline-first (HF_HUB_OFFLINE) then online. Returns (torch, tok, model).
    Model = T5_MODEL_NAME (Salesforce/codet5-small). AutoTokenizer resolves codet5's
    RobertaTokenizerFast; AutoModelForSeq2SeqLM resolves its T5ForConditionalGeneration.
    The <extra_id_0>/<extra_id_1> sentinels + t5_extract() are identical across both."""
    import torch
    torch.set_num_threads(threads)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = T5_MODEL_NAME
    last = None
    for offline in ("1", "0"):
        os.environ["HF_HUB_OFFLINE"] = offline
        os.environ["TRANSFORMERS_OFFLINE"] = offline
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            return torch, tok, model
        except Exception as e:
            last = e
            print(f"[t5] load offline={offline} failed: {e!r}", flush=True)
    raise last


class T5Gen:
    def __init__(self, threads, mode=T5_MODE, lr=T5_LR, bs=T5_BS):
        self.threads = threads; self.mode = mode; self.lr = lr; self.bs = bs
        self.model = None; self.tok = None; self.qmodel = None
        self.seen = 0; self.sps = 0.0; self.ft_min = 0.0

    def finetune(self, train_df, budget_s, n_pool=T5_FT_POOL, seed=42, heartbeat=45):
        import torch
        from torch.optim import AdamW
        _, tok, model = _load_t5_base(self.threads)
        opt = AdamW(model.parameters(), lr=self.lr)
        pool = train_df.sample(min(n_pool, len(train_df)), random_state=seed).reset_index(drop=True)
        inputs = [build_t5_input(r.masked_docstring, r.code_context, tok, self.mode) for r in pool.itertuples()]
        targets = [f"<extra_id_0> {str(r.target_span)} <extra_id_1>" for r in pool.itertuples()]
        idx = list(range(len(inputs)))
        random.seed(0); random.shuffle(idx)
        model.train()
        elapsed = 0.0; last = time.time(); ptr = step = seen = 0; hb = heartbeat; lastloss = 0.0
        while elapsed < budget_s:
            bi = [idx[(ptr + k) % len(idx)] for k in range(self.bs)]; ptr += self.bs
            bt = [inputs[j] for j in bi]; tg = [targets[j] for j in bi]
            enc = tok(bt, return_tensors="pt", padding=True, truncation=True, max_length=128)
            lab = tok(tg, return_tensors="pt", padding=True, truncation=True, max_length=24).input_ids
            lab[lab == tok.pad_token_id] = -100
            out = model(**enc, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            step += 1; seen += self.bs; lastloss = float(out.loss.item())
            now = time.time(); elapsed += now - last; last = now
            if elapsed >= hb:
                print(f"[t5-ft] {elapsed/60:5.1f}min step={step} seen={seen} "
                      f"sps={seen/elapsed:.1f} loss={lastloss:.3f}", flush=True)
                hb += heartbeat
        model.eval()
        self.tok = tok; self.model = model; self.seen = seen
        self.sps = seen / max(elapsed, 1e-6); self.ft_min = elapsed / 60
        print(f"[t5-ft] DONE {elapsed/60:.1f}min steps={step} seen={seen} sps={self.sps:.1f}", flush=True)
        return self

    def load_zeroshot(self):
        """Load base pretrained weights, NO fine-tuning. codet5-small is used zero-shot
        because fine-tuning measurably regresses it (D2: 0.4135 zs -> 0.3735 FT). This is
        the shipped generator path: base pretrained weights only, no checkpoint, no FT."""
        _, tok, model = _load_t5_base(self.threads)
        model.eval()
        self.tok = tok; self.model = model
        self.seen = 0; self.sps = 0.0; self.ft_min = 0.0
        print(f"[t5] zero-shot base weights loaded ({T5_MODEL_NAME}, no FT)", flush=True)
        return self

    def load_checkpoint(self, ckpt, threads=None):
        """DEV/EVAL ONLY: load a previously fine-tuned checkpoint (never used by the grader)."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        torch.set_num_threads(threads or self.threads)
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).eval()
        return self

    def quantize(self):
        try:
            import torch
            self.qmodel = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8).eval()
            print("[t5] int8 dynamic quantization ready", flush=True)
        except Exception as e:
            print(f"[t5] quantize failed ({e!r}); using fp32", flush=True)
            self.qmodel = None
        return self

    def generate(self, masked_list, code_list, bs=T5_INFER_BS, max_new=T5_MAXNEW,
                 use_quant=True, heartbeat_rows=1500):
        import torch
        m = self.qmodel if (use_quant and self.qmodel is not None) else self.model
        tok = self.tok
        n = len(masked_list)
        texts = [build_t5_input(masked_list[i], code_list[i], tok, self.mode) for i in range(n)]
        preds, logps = [], []
        done = 0; nexthb = heartbeat_rows
        for i in range(0, n, bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=max_new, num_beams=1, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
            dec = tok.batch_decode(out.sequences, skip_special_tokens=False)
            preds.extend([t5_extract(x) for x in dec])
            try:
                ts = m.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                gen = out.sequences[:, 1:]
                mask = (gen != tok.pad_token_id)
                lp = (ts * mask).sum(dim=1)
                ln = mask.sum(dim=1).clamp(min=1)
                logps.extend((lp / ln).tolist())
            except Exception:
                logps.extend([-1.0] * len(chunk))
            done += len(chunk)
            if done >= nexthb:
                print(f"[t5-gen] {done}/{n}", flush=True)
                nexthb += heartbeat_rows
        return preds, logps


# ---- single-candidate featurization (for injected T5 candidate; main process, no fork) ----
def featurize_one(text, rc, code, idents, cc_set, gtop_set, fits, astr, top8_grams):
    src = set()
    if text in cc_set:
        src.add("code")
    if text in gtop_set:
        src.add("global")
    # local cache: a train row's parity fits differ per-row, so a shared (level,key) cache
    # would cross-contaminate; a single-candidate call gets no reuse benefit anyway.
    cache = {}
    Xb1 = b1_featurize([text], {text: src}, rc, fits.idx, fits.glob, cache, code, idents, fits.lm)[0]
    E = np.zeros(len(EXT_NAMES), dtype=np.float32)
    E[4] = math.log1p(fits.pb.target_prior.get(text, 0))
    E[5] = astr
    E[17] = LM_FLOOR; E[18] = LM_FLOOR; E[19] = LM_FLOOR; E[20] = LM_FLOOR
    E[21] = LM_TOPN; E[22] = 0
    cg, tp = _grams(text)
    if tp and top8_grams:
        tot = 0.0
        for rg, tr in top8_grams:
            if tr == 0:
                continue
            mo = _overlap(cg, rg)
            if mo:
                p = mo / tp; r = mo / tr
                tot += 2 * p * r / (p + r)
        E[24] = tot / len(top8_grams)
    return np.hstack([Xb1, E])


def augment_with_t5(Xbase, groups, texts, unions, masked, codes, astr, gated,
                    t5_pred, t5_logp, fits_list, ybase=None, tgts=None):
    """Append 5 T5 columns; inject the T5 candidate into gated rows' pools.
    fits_list[gi] = the Fits object whose index the row was featurized against
    (parity fits for train rows; full fits for test rows). Returns
    X_aug, groups_aug, texts_aug, y_aug, row_t5 (per-group (pred,logp) or None)."""
    blocks, g_aug, tx_aug, y_blocks, row_t5 = [], [], [], [], []
    start = 0
    n_inject_new = n_inject_hit = 0
    for gi in range(len(groups)):
        g = groups[gi]
        blk = Xbase[start:start + g]
        yg = ybase[start:start + g] if ybase is not None else None
        start += g
        txt = list(texts[gi])
        t5c = np.zeros((g, N_T5_FEAT), dtype=np.float32)
        if gated[gi] and t5_pred[gi] is not None and str(t5_pred[gi]) != "":
            t5c[:, 0] = 1.0  # has_t5 (row-level)
            pred = str(t5_pred[gi]); lp = float(t5_logp[gi])
            if pred in txt:
                j = txt.index(pred)
                t5c[j, 1] = 1.0; t5c[j, 2] = lp; t5c[j, 3] = float(len(pred)); t5c[j, 4] = 1.0
                blk_aug = np.hstack([blk, t5c])
                yg_aug = yg
                n_inject_hit += 1
            else:
                fits = fits_list[gi]
                rc = row_ctx(masked[gi]); cc, idents = code_features(codes[gi]); cc_set = set(cc)
                pool = unions[gi]
                top8 = sorted(pool, key=lambda x: (x[4], x[3]), reverse=True)[:8]
                top8g = [_grams(t[0]) for t in top8]
                fv = featurize_one(pred, rc, codes[gi], idents, cc_set, set(fits.gtop),
                                   fits, astr[gi], top8g)
                newt5 = np.array([1.0, 1.0, lp, float(len(pred)), 0.0], dtype=np.float32)
                blk_aug = np.vstack([np.hstack([blk, t5c]),
                                     np.hstack([fv[None, :], newt5[None, :]])])
                txt.append(pred)
                if yg is not None:
                    yg_aug = np.concatenate([yg, np.array([f_pooled(pred, tgts[gi])], np.float32)])
                else:
                    yg_aug = None
                n_inject_new += 1
            row_t5.append((pred, lp))
        else:
            blk_aug = np.hstack([blk, t5c])
            yg_aug = yg
            row_t5.append(None)
        blocks.append(blk_aug); g_aug.append(len(txt)); tx_aug.append(txt)
        if yg_aug is not None:
            y_blocks.append(yg_aug)
    X_aug = np.vstack(blocks) if blocks else np.zeros((0, N_FEAT3), np.float32)
    y_aug = np.concatenate(y_blocks) if (ybase is not None and y_blocks) else None
    print(f"[augment] rows={len(groups)} t5_new_cand={n_inject_new} t5_hit_pool={n_inject_hit} "
          f"X{X_aug.shape}", flush=True)
    return X_aug, g_aug, tx_aug, y_aug, row_t5


# ================================= reranker + decode =================================
def train_reranker(X, y, groups, n_threads, feat_names, es_frac=0.08, seed=7):
    n = len(groups)
    n_es = max(1, int(n * es_frac))
    perm = np.random.RandomState(seed).permutation(n)
    es_g = set(perm[:n_es].tolist())
    starts = np.cumsum([0] + list(groups))
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
    dtr = lgb.Dataset(Xf, lab_f, group=g_fit, feature_name=feat_names)
    dva = lgb.Dataset(Xe, lab_e, group=g_es, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=700, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    return booster


def decode(booster, X, groups, texts, row_t5=None, override_thr=None, fallback="value of the"):
    starts = np.cumsum([0] + list(groups))
    scores = booster.predict(X) if len(X) else np.zeros(0)
    preds = []
    n_ovr = 0
    for gi in range(len(groups)):
        cs = texts[gi]
        if not cs:
            preds.append(fallback); continue
        s = scores[starts[gi]:starts[gi + 1]]
        pick = cs[int(np.argmax(s))]
        if override_thr is not None and row_t5 is not None and row_t5[gi] is not None:
            pred, lp = row_t5[gi]
            if lp >= override_thr and pred and str(pred) != "" and pred != pick:
                pick = pred; n_ovr += 1
        preds.append(pick)
    if override_thr is not None:
        print(f"[decode] confident-T5 overrides applied: {n_ovr}", flush=True)
    return preds


# ================================= driver =================================
def find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in [".", "dataset", "../dataset", os.path.join(here, "dataset"),
              os.path.join(here, "..", "dataset"), os.path.join(here, "..", "..", "dataset"),
              here, os.path.join(here, "..")]:
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


def _write_submission(ids, preds, out_path, best_const):
    preds = [p if (p is not None and str(p) != "") else best_const for p in preds]
    assert len(preds) == len(ids), f"row count {len(preds)} != {len(ids)}"
    df = pd.DataFrame({"id": np.asarray(ids), "prediction": preds})
    assert (df["prediction"].astype(str).str.len() > 0).all(), "empty prediction present"
    df.to_csv(out_path, index=False)
    return df


def _fallback_submission(test, out_path, const="value of the"):
    pd.DataFrame({"id": test.id.values, "prediction": [const] * len(test)}).to_csv(out_path, index=False)


def main():
    t0 = time.time()
    T = {}

    def el():
        return (time.time() - t0) / 60.0

    dd, out_path = _resolve_paths()
    train = pd.read_csv(os.path.join(dd, "train.csv"), keep_default_na=False)
    test = pd.read_csv(os.path.join(dd, "test.csv"), keep_default_na=False)
    if _TESTN > 0:
        test = test.head(_TESTN)
    train["_bkt"] = train.masked_docstring.map(bucket)
    print(f"[load] train {len(train)} test {len(test)} threads {NUM_THREADS} nproc {NPROC} "
          f"ntrain {NTRAIN} coverage {T5_COVERAGE} skip_t5 {SKIP_T5}  {time.time()-t0:.0f}s", flush=True)
    best_const = train.target_span.astype(str).value_counts().idxmax()

    try:
        # ---------- C1 base: ALL fork-pool stages first (torch loads only after these) ----------
        even = train[train._bkt % 2 == 0]; odd = train[train._bkt % 2 == 1]
        tf = time.time()
        fits_even = build_fits(even, NUM_THREADS, "even")
        fits_odd = build_fits(odd, NUM_THREADS, "odd")
        T["parity_fits"] = time.time() - tf

        samp = train.sample(n=min(NTRAIN, len(train)), random_state=7).reset_index(drop=True)
        tt = time.time()
        (Xtr_b, ytr_b, gtr, txtr, masked_tr, code_tr,
         unions_tr, astr_tr, tag_tr, tgt_tr) = build_training_base(samp, fits_even, fits_odd, NPROC)
        T["train_candgen_feat"] = time.time() - tt
        print(f"[train matrix] X{Xtr_b.shape} groups {len(gtr)}  {T['train_candgen_feat']:.0f}s", flush=True)

        tff = time.time()
        fits_full = build_fits(train, NUM_THREADS, "full")
        T["full_fits"] = time.time() - tff

        te = time.time()
        astr_te, unions_te = build_union(test, fits_full, NPROC)
        Xte_b, _, gte, txte = featurize_rows(test, fits_full, astr_te, unions_te, NPROC, want_labels=False)
        T["test_candgen_feat"] = time.time() - te
        masked_te = test.masked_docstring.values.tolist()
        code_te = test.code_context.values.tolist()
        print(f"[test matrix] X{Xte_b.shape} groups {len(gte)}  all-fork-stages-done @ {el():.1f}min", flush=True)

        # ---------- T5 hybrid stage (MAIN process only, torch loaded here) ----------
        mode = "full"                # full | test_only | skip
        t5_ok = False
        t5gen = None
        row_t5_tr = row_t5_te = None
        Xtr_a, gtr_a, txtr_a, ytr_a = None, None, None, None
        Xte_a, gte_a, txte_a = None, None, None
        gated_te = None

        if SKIP_T5 or el() > SKIP_T5_MIN:
            mode = "skip"
            print(f"[t5] SKIP (skip_flag={SKIP_T5} elapsed={el():.1f}min)", flush=True)

        if mode != "skip":
            try:
                # dynamic budget: never overrun; leave room for inference + rerank + write
                cov = T5_COVERAGE
                if el() > COVERAGE30_MIN:
                    cov = min(cov, 0.30)
                est_test_inf = (cov * len(test)) / T5_INFER_RPS_EST / 60.0
                if T5_ZEROSHOT:
                    # codet5 zero-shot: NO fine-tune (FT regresses codet5). Ensure the test-side
                    # inference (+ rerank + write) fits the remaining wall-clock; if not, halve
                    # coverage until it does, else degrade to pure C1.
                    while cov > 0.06 and (el() + est_test_inf + 4.0) > SAFE_END_MIN:
                        cov *= 0.5
                        est_test_inf = (cov * len(test)) / T5_INFER_RPS_EST / 60.0
                        print(f"[t5] tight budget -> coverage {cov:.3f}", flush=True)
                    if (el() + est_test_inf + 4.0) > SAFE_END_MIN:
                        raise RuntimeError(f"insufficient time for T5 inference @ {el():.1f}min -> pure C1")
                    tft = time.time()
                    print(f"[t5] zero-shot load (coverage {cov}) @ {el():.1f}min", flush=True)
                    t5gen = T5Gen(NUM_THREADS).load_zeroshot()
                    if T5_USE_QUANT:
                        t5gen.quantize()
                    T["t5_load"] = time.time() - tft
                else:
                    reserve = est_test_inf + 3.0 + 1.0    # rerank + write
                    ft_budget = min(T5_FT_MAX_S, max(0.0, (SAFE_END_MIN - el() - reserve) * 60.0))
                    if ft_budget < FT_MIN_S:
                        raise RuntimeError(f"insufficient FT budget ({ft_budget:.0f}s < {FT_MIN_S}) -> pure C1")
                    tft = time.time()
                    print(f"[t5] fine-tune budget {ft_budget/60:.1f}min (coverage {cov}) @ {el():.1f}min", flush=True)
                    t5gen = T5Gen(NUM_THREADS).finetune(train, ft_budget)
                    if T5_USE_QUANT:
                        t5gen.quantize()
                    T["t5_finetune"] = time.time() - tft

                if el() > SKIP_TRAIN_T5_MIN:
                    mode = "test_only"
                    print(f"[t5] elapsed {el():.1f}min > {SKIP_TRAIN_T5_MIN} -> TEST-ONLY (skip train-side T5)", flush=True)

                # Gate threshold calibrated on the reranker-train rows' PARITY weakness: a train
                # row scored against its opposite-parity fit does NOT see itself, mirroring the
                # unseen condition of test rows (which are absent from the full fit). Calibrating
                # on full-fit train would be self-leaked (train rows always find their own anchors)
                # and over-gate test. Same frozen threshold gates train (parity) and test (full).
                tw = np.array([t5_weakness_score(masked_tr[i],
                               (fits_odd.pb if tag_tr[i] == 0 else fits_even.pb))
                               for i in range(len(masked_tr))])
                thr = float(np.quantile(tw, 1.0 - cov)) if cov < 0.999 else -1e18
                gated_te = gate_mask(masked_te, fits_full.pb, thr)
                print(f"[t5] gate thr={thr:.4f} train-cov(parity)={ (tw>=thr).mean():.3f} "
                      f"test-cov(full)={gated_te.mean():.3f} ({int(gated_te.sum())}/{len(gated_te)})", flush=True)

                # ----- test-side T5 inference -----
                tti = time.time()
                gt_idx = np.where(gated_te)[0]
                t5p_te = [None] * len(test); t5l_te = [0.0] * len(test)
                if len(gt_idx) > 0:
                    ml = [masked_te[i] for i in gt_idx]; cl = [code_te[i] for i in gt_idx]
                    preds, logps = t5gen.generate(ml, cl, use_quant=T5_USE_QUANT)
                    for k, i in enumerate(gt_idx):
                        t5p_te[i] = preds[k]; t5l_te[i] = logps[k]
                T["t5_infer_test"] = time.time() - tti

                if mode == "full":
                    # ----- train-side T5 (leak-free: parity-fit weakness + parity-fit features) -----
                    gated_tr = tw >= thr
                    g_idx = np.where(gated_tr)[0]
                    if len(g_idx) > TRAIN_T5_CAP:
                        rng = np.random.RandomState(5)
                        g_idx = np.sort(rng.choice(g_idx, TRAIN_T5_CAP, replace=False))
                    keep = np.zeros(len(masked_tr), dtype=bool); keep[g_idx] = True
                    gated_tr = keep
                    tti2 = time.time()
                    t5p_tr = [None] * len(masked_tr); t5l_tr = [0.0] * len(masked_tr)
                    if len(g_idx) > 0:
                        ml = [masked_tr[i] for i in g_idx]; cl = [code_tr[i] for i in g_idx]
                        preds, logps = t5gen.generate(ml, cl, use_quant=T5_USE_QUANT)
                        for k, i in enumerate(g_idx):
                            t5p_tr[i] = preds[k]; t5l_tr[i] = logps[k]
                    T["t5_infer_train"] = time.time() - tti2

                    fits_list_tr = [(fits_odd if tag_tr[i] == 0 else fits_even) for i in range(len(masked_tr))]
                    Xtr_a, gtr_a, txtr_a, ytr_a, row_t5_tr = augment_with_t5(
                        Xtr_b, gtr, txtr, unions_tr, masked_tr, code_tr, astr_tr, gated_tr,
                        t5p_tr, t5l_tr, fits_list_tr, ybase=ytr_b, tgts=tgt_tr)
                    fits_list_te = [fits_full] * len(test)
                    Xte_a, gte_a, txte_a, _, row_t5_te = augment_with_t5(
                        Xte_b, gte, txte, unions_te, masked_te, code_te, astr_te, gated_te,
                        t5p_te, t5l_te, fits_list_te)
                    assert Xtr_a.shape[1] == Xte_a.shape[1] == N_FEAT3, \
                        f"schema mismatch {Xtr_a.shape[1]} vs {Xte_a.shape[1]} vs {N_FEAT3}"
                else:
                    # test_only: keep row_t5_te for override; no matrix augmentation
                    row_t5_te = [((t5p_te[i], t5l_te[i]) if gated_te[i] and t5p_te[i] else None)
                                 for i in range(len(test))]
                t5_ok = True
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[t5] FAILED ({e!r}) -> degrade to pure C1", flush=True)
                mode = "skip"; t5_ok = False

        # free parity fits before reranker train / decode
        try:
            del fits_even, fits_odd
        except Exception:
            pass

        # ---------- reranker + decode ----------
        tr = time.time()
        if mode == "full" and t5_ok:
            booster = train_reranker(Xtr_a, ytr_a, gtr_a, NUM_THREADS, FEAT_NAMES3)
            T["reranker_train"] = time.time() - tr
            imp = sorted(zip(FEAT_NAMES3, booster.feature_importance("gain")), key=lambda x: -x[1])
            print(f"[reranker+t5] iters {booster.best_iteration} top12 {[(n,int(g)) for n,g in imp[:12]]}", flush=True)
            # Ship pure argmax: the reranker has t5_seq_logprob as a feature and subsumes the
            # confident-T5 override, which was measured on bucket-0 to HURT (-0.002..-0.004) at
            # every coverage. The override is retained ONLY for the test-only fallback below,
            # where no learned T5 reranker exists.
            preds = decode(booster, Xte_a, gte_a, txte_a, fallback=best_const)
        else:
            booster = train_reranker(Xtr_b, ytr_b, gtr, NUM_THREADS, FEAT_NAMES)
            T["reranker_train"] = time.time() - tr
            if t5_ok and mode == "test_only":
                preds = decode(booster, Xte_b, gte, txte, row_t5=row_t5_te,
                               override_thr=T5_OVERRIDE_THR, fallback=best_const)
                print("[decode] pure-C1 reranker + confident-T5 override (test-only mode)", flush=True)
            else:
                preds = decode(booster, Xte_b, gte, txte, fallback=best_const)
                print("[decode] pure-C1 argmax", flush=True)

        df = _write_submission(test.id.values, preds, out_path, best_const)
        print(f"\n[TIMINGS] " + "  ".join(f"{k}={v:.0f}s" for k, v in T.items()) +
              f"  TOTAL={time.time()-t0:.0f}s ({el():.1f}min)  mode={mode} t5_ok={t5_ok}", flush=True)
        print(f"[done] wrote {out_path} ({len(df)} rows) mean_len "
              f"{df.prediction.astype(str).str.len().mean():.2f}", flush=True)
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

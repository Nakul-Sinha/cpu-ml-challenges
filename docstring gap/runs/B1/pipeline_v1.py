"""Docstring Gap Restoration -- Reranker Pipeline v1 (Agent B1).

Pipeline:
  candidate generation (anchored multi-level indexes + code-derived + global)
    -> per-candidate features
    -> LightGBM regression reranker, label y = f_pooled(candidate, true_span)
    -> argmax decode (+ optional MBR over top-K by softmax posterior).

Leakage hygiene for TRAINING candidate generation: parity half-indexes over
buckets 1-19. A row in an even bucket draws candidates from the ODD half-index
(and vice-versa); twins share masked_docstring -> same bucket -> same parity,
so a row never sees itself or its twins during training candgen. This mirrors
the eval setting where bucket-0 is absent from the buckets 1-19 index.

Validation: index rebuilt on ALL buckets 1-19, evaluate end-to-end on the full
bucket-0 set via solution/chrf.py score_lists.

All statistics are raw term-frequency counts / count-derived conditional
probabilities (Counters). No idf / tf-idf / bm25 anywhere.
"""
import sys, re, os, time, hashlib, collections
import numpy as np
import pandas as pd

sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists  # noqa: E402

GAP = "[GAP]"
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEF_RE = re.compile(r"def\s+(\w+)\s*\(([^)]*)\)")

# anchored levels: (name, n_left, n_right), ordered most-specific -> least
LEVELS = [
    ("l2r2", 2, 2), ("l2r1", 2, 1), ("l1r2", 1, 2), ("l1r1", 1, 1),
    ("l1", 1, 0), ("r1", 0, 1), ("l2", 2, 0), ("r2", 0, 2),
]
# how many top targets each level contributes to the candidate pool
LEVEL_TOPK = {"l2r2": 8, "l2r1": 6, "l1r2": 6, "l1r1": 10,
              "l1": 6, "r1": 6, "l2": 4, "r2": 4}
RANK_CAP = 40  # rank maps built from most_common(RANK_CAP)


def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20


def row_ctx(masked):
    """Per-row context: level keys + positional scalars."""
    i = masked.find(GAP)
    left = masked[:i].split()
    right = masked[i + len(GAP):].split()
    keys = {}
    for name, nl, nr in LEVELS:
        L = " ".join(left[-nl:]) if nl else ""
        R = " ".join(right[:nr]) if nr else ""
        keys[name] = (L, R)
    return {
        "keys": keys,
        "left": left,
        "right": right,
        "n_left": len(left),
        "n_right": len(right),
        "masked_len": len(masked),
        "gap_pos": (i / len(masked)) if masked else 0.0,
    }


def code_features(code):
    """Candidates + identifier set derived from the function code."""
    cands = []
    m = DEF_RE.search(code)
    if m:
        name = m.group(1)
        words = [w for w in re.split(r"_+", name) if w]
        if words:
            cands.append(" ".join(words))
            cands += words
        for a in m.group(2).split(",")[:4]:
            a = a.split("=")[0].split(":")[0].strip()
            if a and a not in ("self", "cls"):
                cands.append(a.replace("_", " "))
    # return-line identifiers
    for rl in re.findall(r"return\s+([A-Za-z_][\w\. ]*)", code)[:2]:
        toks = [t for t in re.split(r"[^A-Za-z0-9_]+", rl) if t][:2]
        if toks:
            cands.append(" ".join(w.replace("_", " ") for w in toks))
    idents = set(t.lower() for t in IDENT_RE.findall(code))
    # dedup candidates, cap
    seen, out = set(), []
    for c in cands:
        c = c.strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out[:10], idents


def build_index(df):
    """level -> {key: Counter(target -> count)} plus global target Counter."""
    idx = {name: collections.defaultdict(collections.Counter) for name, _, _ in LEVELS}
    glob = collections.Counter()
    for masked, tgt in zip(df.masked_docstring.values, df.target_span.values):
        tgt = str(tgt)
        rc = row_ctx(masked)
        for name, _, _ in LEVELS:
            idx[name][rc["keys"][name]][tgt] += 1
        glob[tgt] += 1
    return idx, glob


def global_top(glob, k=12):
    return [c for c, _ in glob.most_common(k)]


# ---------- word n-gram fill-fluency LM (stupid backoff; count-based) ----------
def _toks(s):
    return s.lower().split()


def build_lm(df):
    """Trigram/bigram/unigram counts over gap-FILLED docstring sentences.
    filled = left_words + target_words + right_words. Count-based conditional
    probabilities only (no idf) -> a fill-fluency signal that generalises past
    the exact-match anchored index."""
    uni = collections.Counter()
    bi = collections.Counter()
    tri = collections.Counter()
    for masked, tgt in zip(df.masked_docstring.values, df.target_span.values):
        i = masked.find(GAP)
        left = _toks(masked[:i])
        right = _toks(masked[i + len(GAP):])
        seq = left + _toks(str(tgt)) + right
        uni.update(seq)
        for a in range(len(seq) - 1):
            bi[(seq[a], seq[a + 1])] += 1
        for a in range(len(seq) - 2):
            tri[(seq[a], seq[a + 1], seq[a + 2])] += 1
    return {"uni": uni, "bi": bi, "tri": tri,
            "N": sum(uni.values()), "V": max(len(uni), 1)}


def _sb_prob(lm, w, c1, c2):
    tri, bi, uni = lm["tri"], lm["bi"], lm["uni"]
    if c2 is not None and c1 is not None:
        ctx = bi.get((c2, c1), 0)
        if ctx:
            t = tri.get((c2, c1, w), 0)
            if t:
                return t / ctx
    if c1 is not None:
        u1 = uni.get(c1, 0)
        if u1:
            b = bi.get((c1, w), 0)
            if b:
                return 0.4 * b / u1
    return 0.16 * (uni.get(w, 0) + 1.0) / (lm["N"] + lm["V"])


def lm_features(lm, left, right, cand):
    """(mean_logp over candidate+exit transitions, first-word transition logp)."""
    lw = _toks(left) if isinstance(left, str) else [w.lower() for w in left]
    rw = _toks(right) if isinstance(right, str) else [w.lower() for w in right]
    cw = cand.lower().split()
    if not cw:
        return 0.0, 0.0
    window = lw[-2:] + cw + rw[:1]
    start = len(lw[-2:])
    total = 0.0
    n = 0
    first = 0.0
    import math
    for j in range(start, len(window)):
        w = window[j]
        c1 = window[j - 1] if j - 1 >= 0 else None
        c2 = window[j - 2] if j - 2 >= 0 else None
        lp = math.log(_sb_prob(lm, w, c1, c2))
        total += lp
        if j == start:
            first = lp
        n += 1
    return (total / n if n else 0.0), first


# ---- per-(level,key) stat cache: total + {cand:(rank,count)} for top RANK_CAP ----
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
            rankmap = {}
            for r, (c, cnt) in enumerate(counter.most_common(RANK_CAP)):
                rankmap[c] = (r, cnt)
            v = (total, rankmap)
        cache[ck] = v
    return v


def gen_candidates(rc, idx, glob_top, code_cands):
    """Ordered candidate list + source flag dict, using given index."""
    order, src = [], {}

    def add(c, s):
        if not c:
            return
        if c not in src:
            src[c] = set()
            order.append(c)
        src[c].add(s)

    for name, _, _ in LEVELS:
        counter = idx[name].get(rc["keys"][name])
        if counter:
            for c, _ in counter.most_common(LEVEL_TOPK[name]):
                add(c, name)
    for c in code_cands:
        add(c, "code")
    for c in glob_top:
        add(c, "global")
    return order, src


N_LVL = len(LEVELS)
FEAT_NAMES = []
for _n, _, _ in LEVELS:
    FEAT_NAMES += [f"{_n}_prob", f"{_n}_present", f"{_n}_cntlog", f"{_n}_rank"]
FEAT_NAMES += [
    "glob_freq_log", "char_len", "word_len", "in_code", "ovl_cnt", "ovl_frac",
    "src_code", "src_global", "n_levels_hit", "max_prob", "deepest_lvl",
    "best_rank", "masked_len", "n_left", "n_right", "gap_pos", "code_len_log",
    "n_code_ident", "len_minus_typ", "lm_mean_logp", "lm_first_logp",
]
N_FEAT = len(FEAT_NAMES)


def featurize(cands, src, rc, idx, glob, cache, code, idents, lm=None):
    n = len(cands)
    X = np.zeros((n, N_FEAT), dtype=np.float32)
    code_low = code.lower()
    lw2, rw1 = rc["left"][-2:], rc["right"][:1]
    code_len_log = np.log1p(len(code))
    n_ident = len(idents)
    for i, c in enumerate(cands):
        col = 0
        n_hit = 0
        max_prob = 0.0
        deepest = -1
        best_rank = RANK_CAP
        for li, (name, _, _) in enumerate(LEVELS):
            total, rankmap = level_stats(idx, name, rc["keys"][name], cache)
            rc_ = rankmap.get(c)
            if rc_ is not None and total > 0:
                rank, cnt = rc_
                prob = cnt / total
                X[i, col] = prob
                X[i, col + 1] = 1.0
                X[i, col + 2] = np.log1p(cnt)
                X[i, col + 3] = rank
                n_hit += 1
                if prob > max_prob:
                    max_prob = prob
                if deepest < 0:
                    deepest = li
                if rank < best_rank:
                    best_rank = rank
            else:
                X[i, col + 3] = RANK_CAP  # absent -> worst rank
            col += 4
        # candidate-global + lexical
        words = c.split()
        cl = c.lower()
        wl = set(w for w in re.split(r"[^a-z0-9]+", cl) if w)
        ovl = len(wl & idents)
        X[i, col + 0] = np.log1p(glob.get(c, 0))
        X[i, col + 1] = len(c)
        X[i, col + 2] = len(words)
        X[i, col + 3] = 1.0 if cl in code_low else 0.0
        X[i, col + 4] = ovl
        X[i, col + 5] = (ovl / len(wl)) if wl else 0.0
        X[i, col + 6] = 1.0 if "code" in src[c] else 0.0
        X[i, col + 7] = 1.0 if "global" in src[c] else 0.0
        X[i, col + 8] = n_hit
        X[i, col + 9] = max_prob
        X[i, col + 10] = deepest
        X[i, col + 11] = best_rank
        X[i, col + 12] = rc["masked_len"]
        X[i, col + 13] = rc["n_left"]
        X[i, col + 14] = rc["n_right"]
        X[i, col + 15] = rc["gap_pos"]
        X[i, col + 16] = code_len_log
        X[i, col + 17] = n_ident
        X[i, col + 18] = len(c) - 12.0
        if lm is not None:
            lm_mean, lm_first = lm_features(lm, lw2, rw1, c)
            X[i, col + 19] = lm_mean
            X[i, col + 20] = lm_first
    return X


# ------------------- driver helpers -------------------
def build_training_matrix(df_train_rows, idx_even, idx_odd, glob, gtop, cache_e, cache_o,
                          lm_even=None, lm_odd=None):
    """For each sampled train row use the OPPOSITE-parity index (no self/twin leak)."""
    Xs, ys, groups, keeprows = [], [], [], []
    for masked, tgt, bkt, code in zip(
        df_train_rows.masked_docstring.values, df_train_rows.target_span.values,
        df_train_rows._bkt.values, df_train_rows.code_context.values,
    ):
        tgt = str(tgt)
        use_odd = (bkt % 2 == 0)  # even-bucket row -> odd index
        idx = idx_odd if use_odd else idx_even
        cache = cache_o if use_odd else cache_e
        lm = lm_odd if use_odd else lm_even
        rc = row_ctx(masked)
        cc, idents = code_features(code)
        cands, src = gen_candidates(rc, idx, gtop, cc)
        if not cands:
            continue
        X = featurize(cands, src, rc, idx, glob, cache, code, idents, lm=lm)
        y = np.fromiter((f_pooled(c, tgt) for c in cands), dtype=np.float32, count=len(cands))
        Xs.append(X)
        ys.append(y)
        groups.append(len(cands))
        keeprows.append(1)
    return np.vstack(Xs), np.concatenate(ys), np.array(groups)


def predict_rows(df_rows, idx, glob, gtop, cache, booster, mbr_k=0, lm=None):
    """Return list of predicted strings for df_rows using given (full) index."""
    preds = []
    for masked, code in zip(df_rows.masked_docstring.values, df_rows.code_context.values):
        rc = row_ctx(masked)
        cc, idents = code_features(code)
        cands, src = gen_candidates(rc, idx, gtop, cc)
        if not cands:
            preds.append("value of the")
            continue
        X = featurize(cands, src, rc, idx, glob, cache, code, idents, lm=lm)
        s = booster.predict(X)
        if mbr_k and len(cands) > 1:
            k = min(mbr_k, len(cands))
            top = np.argsort(-s)[:k]
            w = s[top].astype(np.float64)
            w = np.exp((w - w.max()) / 0.15)
            w /= w.sum()
            tc = [cands[j] for j in top]
            best_j, best_v = top[0], -1.0
            for a in range(k):
                ev = 0.0
                for b in range(k):
                    ev += w[b] * f_pooled(tc[a], tc[b])
                if ev > best_v:
                    best_v, best_j = ev, top[a]
            preds.append(cands[best_j])
        else:
            preds.append(cands[int(np.argmax(s))])
    return preds

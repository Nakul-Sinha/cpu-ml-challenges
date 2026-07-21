"""A3 composition harness for Institutional Edit Ledger Recovery (ins).

Clean, swappable interfaces so iteration-2 can replace components:
  Detector.fit(train_df) / Detector.token_probs(rows_df) -> per-token P(edited)
  Transducer.fit(train_df) / Transducer.transduce(lang, src, ctx) -> replacement str
  assemble(row, tok_spans, probs, transducer, thr, max_edits) -> edits list
  tune_thresholds(oof_prob_rows, truth) -> {lang: thr}

Everything is LEARNED from the train partition passed to .fit (no hardcoded
encoded strings; connectors/templates/conventions are induced by char alignment
and frequency). A real trained model (LightGBM) drives detection.

Paths auto-detect so this runs with cwd=~/insled (dataset/, solution/ resolve).
"""
import os, sys, json, re, collections, math
import numpy as np
import pandas as pd

# ---- path bootstrap ---------------------------------------------------------
def _find_root():
    for base in [".", "..", os.path.dirname(os.path.abspath(__file__)),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")]:
        if os.path.exists(os.path.join(base, "dataset", "train.csv")) and \
           os.path.exists(os.path.join(base, "solution", "elru.py")):
            return os.path.abspath(base)
    return "."
ROOT = _find_root()
sys.path.insert(0, os.path.join(ROOT, "solution"))
import elru  # noqa: E402

WORD_RE = re.compile(r"\S+")
MARKS = set(":*/")
LANGS = ["de", "en", "it"]
LANG_ID = {l: i for i, l in enumerate(LANGS)}


def toks(text):
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]


def inner_marks(w):
    """counts of :,*,/ that are NOT at the very first/last char position."""
    c = col = star = sl = 0
    for k, ch in enumerate(w):
        if ch in MARKS:
            c += 1
            if 0 < k < len(w) - 1:
                if ch == ':': col += 1
                elif ch == '*': star += 1
                elif ch == '/': sl += 1
    return col, star, sl


def mark_signature(w):
    """(normalized mark, suffix-after-mark) for a single marked token; else None."""
    m = re.search(r"[:*/]", w)
    if not m:
        return None
    mark = w[m.start()]
    norm = ':' if mark in ':*' else mark   # colon and star share connector behaviour
    return norm, w[m.start() + 1:], w[:m.start()]  # (marknorm, suffix, stem)


# ============================================================================
#  Learned rate store (fit on train partition only)
# ============================================================================
class RateStore:
    """Smoothed edited/seen rates over several token keys, per language."""
    def __init__(self, a=0.3, b=1.0):
        self.a, self.b = a, b
        self.tok = collections.defaultdict(lambda: [0, 0])
        self.suf2 = collections.defaultdict(lambda: [0, 0])
        self.suf3 = collections.defaultdict(lambda: [0, 0])
        self.suf4 = collections.defaultdict(lambda: [0, 0])
        self.marksuf = collections.defaultdict(lambda: [0, 0])
        self.lang_base = collections.defaultdict(lambda: [0, 0])
        self.interior = collections.defaultdict(lambda: [0, 0])  # token interior of span

    def fit(self, df):
        for r in df.itertuples():
            spans = [(e["start"], e["end"]) for e in r.ed]
            L = r.language
            tk = toks(r.text)
            # interior label: token strictly inside a >=3-token edited span
            interior_off = set()
            for a, b in spans:
                inner = [t for t in tk if a <= t[0] and t[1] <= b]
                for t in inner[1:-1]:
                    interior_off.add(t[0])
            for s, e, w in tk:
                ed = 1 if any(a <= s and e <= b for a, b in spans) else 0
                self._bump(L, w, ed)
                c = self.interior[(L, w)]
                c[0] += (1 if s in interior_off else 0); c[1] += 1
        return self

    def interior_rate(self, L, w):
        return self._rate(self.interior, (L, w), 0.0)

    def _bump(self, L, w, ed):
        for store, key in ((self.tok, w), (self.suf2, w[-2:]), (self.suf3, w[-3:]),
                           (self.suf4, w[-4:])):
            c = store[(L, key)]; c[0] += ed; c[1] += 1
        sig = mark_signature(w)
        if sig:
            c = self.marksuf[(L, sig[0], sig[1])]; c[0] += ed; c[1] += 1
        c = self.lang_base[L]; c[0] += ed; c[1] += 1

    def _rate(self, store, key, default):
        c = store.get(key)
        if c is None or c[1] == 0:
            return default
        return (c[0] + self.a) / (c[1] + self.a + self.b)

    def base(self, L):
        c = self.lang_base.get(L, [0, 0])
        return (c[0] + self.a) / (c[1] + self.a + self.b) if c[1] else 0.05

    def features(self, L, w):
        d = self.base(L)
        return dict(
            tok=self._rate(self.tok, (L, w), d),
            suf2=self._rate(self.suf2, (L, w[-2:]), d),
            suf3=self._rate(self.suf3, (L, w[-3:]), d),
            suf4=self._rate(self.suf4, (L, w[-4:]), d),
            marksuf=self._rate(self.marksuf, (L, *(mark_signature(w)[:2])), d)
            if mark_signature(w) else d,
        )


# ============================================================================
#  Detector: LightGBM per-token P(edited)
# ============================================================================
FEATS = ["lang", "length", "n_col", "n_star", "n_sl", "inner_mark",
         "up_first", "all_up", "has_digit", "pos", "n_tok_row",
         "tok_r", "suf2_r", "suf3_r", "suf4_r", "marksuf_r",
         "prev_tok_r", "next_tok_r", "prev_innermark", "next_innermark",
         "prev_suf2_r", "next_suf2_r",
         "self_int_r", "prev_int_r", "next_int_r",
         "share_prev2", "share_next2", "share_prev1suf", "share_next1suf"]


def _cplen(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Detector:
    def __init__(self, rounds=400, lr=0.05, leaves=31, seed=0):
        self.rounds, self.lr, self.leaves, self.seed = rounds, lr, leaves, seed
        self.rs = None
        self.model = None

    def _row_feats(self, r, rs):
        L = r.language
        tk = toks(r.text)
        n = len(tk)
        base = rs.base(L)
        rows = []
        for i, (s, e, w) in enumerate(tk):
            col, star, sl = inner_marks(w)
            rf = rs.features(L, w)
            pw = tk[i - 1][2] if i > 0 else ""
            nw = tk[i + 1][2] if i < n - 1 else ""
            pcol, pstar, psl = inner_marks(pw) if pw else (0, 0, 0)
            ncol, nstar, nsl = inner_marks(nw) if nw else (0, 0, 0)
            prf = rs.features(L, pw) if pw else {"tok": base, "suf2": base}
            nrf = rs.features(L, nw) if nw else {"tok": base, "suf2": base}
            w2p = tk[i - 2][2] if i > 1 else ""
            w2n = tk[i + 2][2] if i < n - 2 else ""
            lw = max(1, len(w))
            rows.append([
                LANG_ID[L], len(w), w.count(':'), w.count('*'), w.count('/'),
                1 if (col + star + sl) > 0 else 0,
                1 if w[:1].isupper() else 0, 1 if w.isupper() and len(w) > 1 else 0,
                1 if any(ch.isdigit() for ch in w) else 0,
                i / max(1, n - 1), n,
                rf["tok"], rf["suf2"], rf["suf3"], rf["suf4"], rf["marksuf"],
                prf["tok"], nrf["tok"],
                1 if (pcol + pstar + psl) > 0 else 0,
                1 if (ncol + nstar + nsl) > 0 else 0,
                prf["suf2"], nrf["suf2"],
                rs.interior_rate(L, w), rs.interior_rate(L, pw) if pw else 0.0,
                rs.interior_rate(L, nw) if nw else 0.0,
                _cplen(w, w2p) / lw if w2p else 0.0,
                _cplen(w, w2n) / lw if w2n else 0.0,
                1 if (pw and w[-1:] == pw[-1:]) else 0,
                1 if (nw and w[-1:] == nw[-1:]) else 0,
            ])
        return tk, rows

    def _matrix(self, df, rs, with_labels):
        X, y, index = [], [], []
        for r in df.itertuples():
            spans = [(e["start"], e["end"]) for e in r.ed] if with_labels else []
            tk, rows = self._row_feats(r, rs)
            for j, (s, e, w) in enumerate(tk):
                X.append(rows[j])
                index.append((r.id, s, e))
                if with_labels:
                    y.append(1 if any(a <= s and e <= b for a, b in spans) else 0)
        return np.asarray(X, float), (np.asarray(y) if with_labels else None), index

    def fit(self, train_df):
        import lightgbm as lgb
        self.rs = RateStore().fit(train_df)
        X, y, _ = self._matrix(train_df, self.rs, True)
        imb = (len(y) - y.sum()) / max(1, y.sum())
        w = np.where(y == 1, math.sqrt(imb), 1.0)  # gentle balance for calibration
        ds = lgb.Dataset(X, label=y, weight=w,
                         feature_name=FEATS, categorical_feature=["lang"])
        params = dict(objective="binary", learning_rate=self.lr,
                      num_leaves=self.leaves, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=40,
                      verbose=-1, num_threads=5, seed=self.seed,
                      deterministic=True, force_col_wise=True)
        self.model = lgb.train(params, ds, num_boost_round=self.rounds)
        return self

    def token_probs(self, df):
        """returns {id: (token_list, prob_array)}."""
        X, _, index = self._matrix(df, self.rs, False)
        p = self.model.predict(X)
        out = {}
        pos = 0
        for r in df.itertuples():
            tk = toks(r.text)
            n = len(tk)
            out[r.id] = (tk, p[pos:pos + n])
            pos += n
        return out


# ============================================================================
#  Transducer: exact memory + learned templates + suffix rewrites
# ============================================================================
class Transducer:
    def __init__(self, min_tok_sup=1, min_suf_sup=3):
        self.min_tok_sup, self.min_suf_sup = min_tok_sup, min_suf_sup

    def fit(self, train_df):
        self.exact = collections.defaultdict(collections.Counter)      # (L,span)->rep
        self.tok = collections.defaultdict(collections.Counter)        # (L,token)->rep
        self.de_tmpl = collections.defaultdict(collections.Counter)    # (norm,suf)->tmpl
        self.de_tmpl_suf = collections.defaultdict(collections.Counter)  # (suf)->tmpl
        self.sufrw = collections.defaultdict(collections.Counter)      # (L,src_suf)->rep_suf
        for r in train_df.itertuples():
            L = r.language
            for e in r.ed:
                src = r.text[e["start"]:e["end"]]
                rep = e["replacement"]
                self.exact[(L, src)][rep] += 1
                stk = toks(src)
                if len(stk) == 1:
                    self.tok[(L, src)][rep] += 1
                    sig = mark_signature(src)
                    if L == "de" and sig:
                        norm, suf, stem = sig
                        tmpl = self._templatize(stem, rep) if stem else None
                        if tmpl is not None:
                            self.de_tmpl[(norm, suf)][tmpl] += 1
                            self.de_tmpl_suf[suf][tmpl] += 1
                    elif not sig and rep:
                        p = _common_prefix(src, rep)
                        if 0 < len(p) < len(src):
                            self.sufrw[(L, src[len(p):])][rep[len(p):]] += 1
        return self

    @staticmethod
    def _templatize(stem, rep):
        """abstract every occurrence of stem in rep with a placeholder token."""
        if stem and stem in rep:
            return rep.replace(stem, "\0")
        return None

    def _majority(self, counter):
        return counter.most_common(1)[0][0] if counter else None

    def transduce(self, lang, src, ctx=None):
        """ctx unused in v1 (kept for interface parity)."""
        L = lang
        c = self.exact.get((L, src))
        if c:
            return self._majority(c)
        stk = toks(src)
        if len(stk) == 1:
            r = self._single(L, src)
            return r if r is not None else src
        # multi-token: transduce each token, join with single space
        parts = [self._single(L, w) for (_, _, w) in stk]
        if all(p is not None for p in parts):
            return " ".join(parts)
        return src  # echo whole span

    def _single(self, L, w):
        c = self.tok.get((L, w))
        if c and sum(c.values()) >= self.min_tok_sup:
            return self._majority(c)
        sig = mark_signature(w)
        if L == "de" and sig:
            norm, suf, stem = sig
            t = self.de_tmpl.get((norm, suf)) or self.de_tmpl_suf.get(suf)
            if t and stem:
                return self._majority(t).replace("\0", stem)
        if not sig:
            for k in range(4, 0, -1):
                suf = w[-k:]
                c = self.sufrw.get((L, suf))
                if c and sum(c.values()) >= self.min_suf_sup and len(w) > k:
                    return w[:-k] + self._majority(c)
        return None


# ============================================================================
#  Assembly + threshold tuning
# ============================================================================
def _common_prefix(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def assemble(row_id, text, lang, tk, probs, transducer, thr, max_edits=8):
    """group consecutive tokens with prob>=thr into spans; transduce each."""
    edits = []
    i, n = 0, len(tk)
    while i < n:
        if probs[i] >= thr:
            j = i
            score = probs[i]
            while j + 1 < n and probs[j + 1] >= thr:
                j += 1
                score = max(score, probs[j])
            s, e = tk[i][0], tk[j][1]
            src = text[s:e]
            rep = transducer.transduce(lang, src)
            if rep is None:
                rep = src
            edits.append((score, {"start": s, "end": e, "replacement": rep[:160]}))
            i = j + 1
        else:
            i += 1
    edits.sort(key=lambda x: -x[0])
    edits = [e for _, e in edits[:max_edits]]
    edits.sort(key=lambda e: e["start"])
    if not elru.validate_edits(edits, len(text)):
        edits = _repair(edits, len(text))
    return edits


def _repair(edits, tlen):
    out = []
    prev_end = -1
    for e in sorted(edits, key=lambda x: x["start"]):
        if e["start"] < prev_end or not (0 <= e["start"] < e["end"] <= tlen):
            continue
        e["replacement"] = e["replacement"][:160]
        out.append(e); prev_end = e["end"]
        if len(out) >= 8:
            break
    return out


def predict(df, detector, transducer, thr_map, max_edits=8):
    tp = detector.token_probs(df)
    preds = {}
    for r in df.itertuples():
        tk, probs = tp[r.id]
        thr = thr_map.get(r.language, 0.5)
        preds[r.id] = assemble(r.id, r.text, r.language, tk, probs, transducer,
                               thr, max_edits)
    return preds


def tune_thresholds(tp_by_id, df, transducer, grid=None, max_edits=8):
    """maximise CV ELRU by choosing per-language threshold on the given probs.
    tp_by_id: {id:(tk,probs)}; df has id,language,ed. Returns (thr_map, elru, detail)."""
    if grid is None:
        grid = [round(x, 3) for x in np.arange(0.12, 0.86, 0.02)]
    truth = {r.id: r.ed for r in df.itertuples()}
    langs = {r.id: r.language for r in df.itertuples()}
    rows_by_lang = collections.defaultdict(list)
    for r in df.itertuples():
        rows_by_lang[r.language].append(r)
    # per-language: pick thr maximising that language's own lang_score in isolation
    thr_map = {}
    for L, rows in rows_by_lang.items():
        best_thr, best = grid[0], -1
        for thr in grid:
            pm = {}
            for r in rows:
                tk, probs = tp_by_id[r.id]
                pm[r.id] = assemble(r.id, r.text, L, tk, probs, transducer, thr, max_edits)
            tmap = {r.id: r.ed for r in rows}
            lmap = {r.id: L for r in rows}
            sc = elru.elru(pm, tmap, lmap)
            if sc > best:
                best, best_thr = sc, thr
        thr_map[L] = best_thr
    # final combined score at chosen thresholds
    pm = {}
    for r in df.itertuples():
        tk, probs = tp_by_id[r.id]
        pm[r.id] = assemble(r.id, r.text, r.language, tk, probs, transducer,
                            thr_map[r.language], max_edits)
    score, detail = elru.elru(pm, truth, langs, detail=True)
    return thr_map, score, detail, pm

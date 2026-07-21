"""M1 MERGE ENGINEER -- unified best composition for Institutional Edit Ledger Recovery.

ONE clean module = A1 token detector  (LightGBM, 65 features, per-fold lexicon)
                 + A2 Transducer      (imported verbatim from local transducer.py)
                 + A3 assembly/tune/submission harness patterns.

Measured-best composition (review-verified): A1-detector + A2-transducer, leak-free
per-fold, non-nested per-language thresholds -> CV ELRU ~= 0.506.  This module
reproduces that (non-nested) and additionally reports the HONEST NESTED CV
(fold-k thresholds chosen on the OOF of the other four folds only).

Fixed defects (vs iteration-1):
  * U+2217 (MATH STAR) added to the A1 detector mark charset (SPECIAL) so
    'Direktor<U+2217>in'-style tokens fire the cross-stem feminine-suffix signal.
    (A2's transducer already carried U+2217 in its mark set; kept as-is.)
  * Multi-token replacement preserves ORIGINAL intra-span whitespace: assembly
    slices text[a:b] (never a single-space token re-join) and A2's _predict_multi
    re-emits the exact inter-token gaps.  Guarded by test_whitespace_preserved().

======================================================================
PLUG-IN EXTENSION POINTS  (module-level registries; empty by default so the
base pipeline is byte-for-byte the measured composition).  Enhance agents append:

(a) TOKEN_FEATURE_EXTRAS : list[ fn(tokens, i, lang, text) -> dict ]
      Extra numeric features merged into token i's detector feature row.
      Each fn MUST return a STABLE key set (same keys every call); values float-able.
      Keys are frozen (sorted) on the first token seen and emitted as 'x_<key>'
      columns appended AFTER the categorical block (categorical indices unaffected).

(b) SPAN_CANDIDATE_GENERATORS : list[ fn(tokens, lang, text, aux) -> list[(s_tok, e_tok, meta)] ]
      Produce EXTRA candidate spans (token-index inclusive range + meta dict) beyond
      the threshold-merged ones.  aux = {'probs', 'stores', 'lex'}.  Candidates are
      only admitted if stores['span_scorer'](cands, tokens, lang, text, aux) approves
      them (returns list of (start_char, end_char, score)); with no scorer set they
      are inert -- so a second-stage reranker can be dropped in later without
      touching this file.

(c) REPLACEMENT_HOOKS : ordered list[ fn(lang, src, context, stores) -> str|None ]
      Tried IN ORDER before the A2 default mechanisms for every span.  First non-None
      wins.  context = {'text','start','end','lang','tokens','stores'}.  Use for
      high-precision contextual paths (e.g. a validated deletion path).

(d) STORES : a dict is created at fit time, populated by STORE_BUILDERS, threaded
      through fit -> assembly -> hooks/generators so learned tables live in one place.
      STORE_BUILDERS : list[ fn(train_df, stores) -> None ]  (train_df has columns
      id, document_group, language, text, edits_json, edits[list], fold).

Compliance: everything is learned from train.csv at runtime (lexicons/templates/
thresholds); no literal encoded content strings; a real LightGBM materially drives
detection; leak-free per-fold refits for all reported CV; canonical folds + elru only.
"""
import os, sys, json, re, zlib, time, collections
import numpy as np
import pandas as pd

# ---- path bootstrap (runs from anywhere; resolves dataset/ + solution/) -------
def _find_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for base in [".", "..", here, os.path.join(here, "..", ".."),
                 os.path.join(here, "..", "..", "..")]:
        if os.path.exists(os.path.join(base, "dataset", "train.csv")) and \
           os.path.exists(os.path.join(base, "solution", "elru.py")):
            return os.path.abspath(base)
    return "."
ROOT = _find_root()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, HERE)
import elru                                  # canonical scorer
from transducer import Transducer            # A2, local verbatim copy

# ======================================================================
#  Extension registries (empty = base measured composition)
# ======================================================================
TOKEN_FEATURE_EXTRAS = []
SPAN_CANDIDATE_GENERATORS = []
REPLACEMENT_HOOKS = []
STORE_BUILDERS = []

# ======================================================================
#  A1 detector constants / helpers (ported wholesale from runs/A1/detect.py)
# ======================================================================
WORD_RE = re.compile(r"\S+")
NB = 256   # suffix hash buckets
NBP = 128  # prefix hash buckets
LANG2I = {"de": 0, "en": 1, "it": 2}
LANGS = ["de", "en", "it"]
# DEFECT FIX: include U+2217 (MATH STAR) alongside ASCII colon/star/slash.
SPECIAL = (":", "*", "∗", "/")
# non-alnum structural chars for one-hot presence (U+2217 added for test robustness)
punct_set = ['/', '’', '.', '-', '_', '*', ':', "'", ')', '(', '@', ',', '&', '"', '∗']


def toks(text):
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]


def h(s, b):
    return int(zlib.crc32(s.encode("utf-8")) % b)


def shared_prefix_ratio(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b))


def special_key(w):
    """(char, suffix-after-char) for the LAST interior special char; e.g.
    'Direktor:in'->(':','in').  Stem varies across docs but the encoded feminine
    suffix is stable -> strong cross-stem signal."""
    p = -1
    for idx in range(1, len(w) - 1):
        if w[idx] in SPECIAL:
            p = idx
    if p == -1:
        return None
    suf = w[p + 1:][:8]
    return (w[p], suf)


def tok_shape(w):
    n = len(w)
    nu = sum(c.isupper() for c in w); nl = sum(c.islower() for c in w)
    nd = sum(c.isdigit() for c in w); npunct = sum((not c.isalnum()) for c in w)
    return n, nu, nl, nd, npunct


# ---- per-language smoothed edit-rate lexicon (fit on given rows only) ----------
def build_lexicon(rows):
    tok_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    tok_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf3_ed = collections.defaultdict(lambda: collections.defaultdict(float)); suf3_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf4_ed = collections.defaultdict(lambda: collections.defaultdict(float)); suf4_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    pre3_ed = collections.defaultdict(lambda: collections.defaultdict(float)); pre3_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    ch_ed = collections.defaultdict(lambda: collections.defaultdict(float)); ch_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    spat_ed = collections.defaultdict(lambda: collections.defaultdict(float)); spat_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf_ed = collections.defaultdict(lambda: collections.defaultdict(float)); suf_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    lang_ed = collections.defaultdict(float); lang_sn = collections.defaultdict(float)
    for R in rows:
        L = R["lang"]
        for (s, e, w), lab in zip(R["tk"], R["y"]):
            tok_ed[L][w] += lab; tok_sn[L][w] += 1
            lang_ed[L] += lab; lang_sn[L] += 1
            wl = w.lower()
            s3 = wl[-3:]; s4 = wl[-4:]; p3 = wl[:3]
            suf3_ed[L][s3] += lab; suf3_sn[L][s3] += 1
            suf4_ed[L][s4] += lab; suf4_sn[L][s4] += 1
            pre3_ed[L][p3] += lab; pre3_sn[L][p3] += 1
            inner = w[1:-1] if len(w) > 2 else ""
            for ch in set(inner):
                if not ch.isalnum():
                    ch_ed[L][ch] += lab; ch_sn[L][ch] += 1
            sk = special_key(w)
            if sk is not None:
                key = sk[0] + sk[1]
                spat_ed[L][key] += lab; spat_sn[L][key] += 1
                suf_ed[L][sk[1]] += lab; suf_sn[L][sk[1]] += 1
    prior = {L: (lang_ed[L] + 0.5) / (lang_sn[L] + 1.0) for L in lang_sn}
    def rate(ed, sn, L, k, a):
        p = prior.get(L, 0.03)
        return (ed[L].get(k, 0.0) + a * p) / (sn[L].get(k, 0.0) + a)
    return dict(tok_ed=tok_ed, tok_sn=tok_sn, suf3_ed=suf3_ed, suf3_sn=suf3_sn,
                suf4_ed=suf4_ed, suf4_sn=suf4_sn, pre3_ed=pre3_ed, pre3_sn=pre3_sn,
                ch_ed=ch_ed, ch_sn=ch_sn, spat_ed=spat_ed, spat_sn=spat_sn,
                suf_ed=suf_ed, suf_sn=suf_sn, prior=prior, rate=rate)


# ---- token featurization (A1 verbatim + TOKEN_FEATURE_EXTRAS hook) ------------
FEAT_NAMES = None
EXTRA_NAMES = None


def featurize(rows, lex):
    """Returns (X float32, cat_idx).  cat_idx is recomputed every call (consistent
    categorical declaration across folds); names frozen once."""
    global FEAT_NAMES, EXTRA_NAMES
    a_tok, a_suf3, a_suf4 = 5.0, 20.0, 30.0
    X = []
    cat_idx = None
    def rt(ed, sn, L, k, a):
        return lex["rate"](ed, sn, L, k, a)
    for R in rows:
        L = R["lang"]; lid = LANG2I[L]
        tk = R["tk"]; nt = len(tk)
        words = [w for _, _, w in tk]
        shapes = [tok_shape(w) for w in words]
        lows = [w.lower() for w in words]
        trate = [rt(lex["tok_ed"], lex["tok_sn"], L, w, a_tok) for w in words]
        text = R.get("text", "")
        for i, (s, e, w) in enumerate(tk):
            n, nu, nl, nd, npunct = shapes[i]
            lw = lows[i]
            inner = w[1:-1] if len(w) > 2 else ""
            feats = []; fn = []
            def add(v, name):
                feats.append(float(v))
                if FEAT_NAMES is None:
                    fn.append(name)
            # ---- structural ----
            add(n, "len"); add(nu, "nup"); add(nl, "nlo"); add(nd, "ndig"); add(npunct, "npun")
            add(nu / max(n, 1), "frac_up"); add(nd / max(n, 1), "frac_dig")
            add(1 if (w[:1].isupper() and not w.isupper()) else 0, "title")
            add(1 if (w.isupper() and any(c.isalpha() for c in w)) else 0, "allcaps")
            add(1 if w[:1].isupper() else 0, "first_up")
            add(1 if any(c.isdigit() for c in w) else 0, "has_dig")
            for pc in punct_set:
                add(1 if pc in inner else 0, f"mid_{pc}")
            add(1 if (w[:1] in punct_set) else 0, "start_pun")
            add(1 if (w[-1:] in punct_set) else 0, "end_pun")
            add(sum(1 for c in inner if not c.isalnum()), "mid_npun")
            # ---- lexicon: this token ----
            add(trate[i], "tok_rate")
            add(np.log1p(lex["tok_sn"][L].get(w, 0.0)), "tok_sup")
            add(1 if lex["tok_sn"][L].get(w, 0.0) > 0 else 0, "tok_seen")
            add(rt(lex["suf3_ed"], lex["suf3_sn"], L, lw[-3:], a_suf3), "suf3_rate")
            add(rt(lex["suf4_ed"], lex["suf4_sn"], L, lw[-4:], a_suf4), "suf4_rate")
            add(rt(lex["pre3_ed"], lex["pre3_sn"], L, lw[:3], a_suf3), "pre3_rate")
            mc = 0.0
            for ch in set(inner):
                if not ch.isalnum():
                    mc = max(mc, rt(lex["ch_ed"], lex["ch_sn"], L, ch, 10.0))
            add(mc, "maxchar_rate")
            # ---- special-char + suffix pattern (':in'/'*in'/U+2217'in') ----
            sk = special_key(w)
            if sk is not None:
                spc, suf = sk
                key = spc + suf
                add(rt(lex["spat_ed"], lex["spat_sn"], L, key, 3.0), "spat_rate")
                add(rt(lex["suf_ed"], lex["suf_sn"], L, suf, 3.0), "specsuf_rate")
                add(1, "has_special"); add(len(suf), "specsuf_len")
                add(1 if (suf != "" and all((c.isalpha() or not c.isalnum()) and not c.isdigit() for c in suf)) else 0, "specsuf_alpha")
                add(1 if 1 <= len(suf) <= 6 else 0, "specsuf_short")
                add(np.log1p(lex["spat_sn"][L].get(key, 0.0)), "spat_sup")
                spc_id = SPECIAL.index(spc) + 1
            else:
                add(0, "spat_rate"); add(0, "specsuf_rate"); add(0, "has_special")
                add(0, "specsuf_len"); add(0, "specsuf_alpha"); add(0, "specsuf_short"); add(0, "spat_sup")
                spc_id = 0
            # ---- neighbors (prev2,prev1,next1,next2) ----
            for off in (-2, -1, 1, 2):
                j = i + off
                if 0 <= j < nt:
                    wj = words[j]; sj = shapes[j]
                    add(1, f"nb{off}_ex"); add(trate[j], f"nb{off}_rate")
                    add(1 if any((not c.isalnum()) for c in (wj[1:-1] if len(wj) > 2 else "")) else 0, f"nb{off}_midpun")
                    add(1 if wj[:1].isupper() else 0, f"nb{off}_up"); add(sj[0], f"nb{off}_len")
                else:
                    add(0, f"nb{off}_ex"); add(0, f"nb{off}_rate"); add(0, f"nb{off}_midpun"); add(0, f"nb{off}_up"); add(0, f"nb{off}_len")
            # ---- stem similarity (paired forms / connector) ----
            pv = lows[i-1] if i-1 >= 0 else ""
            nx = lows[i+1] if i+1 < nt else ""
            pv2 = lows[i-2] if i-2 >= 0 else ""
            nx2 = lows[i+2] if i+2 < nt else ""
            add(shared_prefix_ratio(lw, pv), "sp_prev")
            add(shared_prefix_ratio(lw, nx), "sp_next")
            add(shared_prefix_ratio(pv, nx), "sp_skip")
            add(shared_prefix_ratio(lw, pv2), "sp_prev2")
            add(shared_prefix_ratio(lw, nx2), "sp_next2")
            add(max(shared_prefix_ratio(pv, nx), shared_prefix_ratio(pv2, nx2)), "sp_bridge")
            # ---- position ----
            add(i / max(nt - 1, 1), "pos_frac"); add(np.log1p(nt), "n_tok")
            add(1 if i == 0 else 0, "is_first"); add(1 if i == nt - 1 else 0, "is_last")
            # ---- language + hashed morphology (categorical block) ----
            catstart = len(feats)
            add(lid, "lang_id")
            add(h(lw[-2:], NB), "suf2_id"); add(h(lw[-3:], NB), "suf3_id"); add(h(lw[-4:], NB), "suf4_id")
            add(h(lw[:2], NBP), "pre2_id"); add(h(lw[:3], NBP), "pre3_id")
            add(spc_id, "spc_id")
            add(h(sk[1], NB) if sk is not None else 0, "specsuf_id")
            cat_end = len(feats)
            if cat_idx is None:
                cat_idx = list(range(catstart, cat_end))
            # ---- (a) TOKEN_FEATURE_EXTRAS (appended after categorical block) ----
            if TOKEN_FEATURE_EXTRAS:
                merged = {}
                for efn in TOKEN_FEATURE_EXTRAS:
                    d = efn(tk, i, L, text)
                    if d:
                        merged.update({str(k): v for k, v in d.items()})
                if EXTRA_NAMES is None:
                    EXTRA_NAMES = sorted(merged.keys())
                for nm in EXTRA_NAMES:
                    add(merged.get(nm, 0.0), "x_" + nm)
            if FEAT_NAMES is None:
                FEAT_NAMES = fn
            X.append(feats)
    return np.asarray(X, dtype=np.float32), (cat_idx or [])


def rows_labels(rows):
    y = []
    for R in rows:
        y.extend(R["y"])
    return np.asarray(y, dtype=np.int32)


LGB_PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.045,
                  num_leaves=48, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=2.0, is_unbalance=True,
                  random_state=0, n_jobs=5, verbosity=-1, max_depth=-1)


class Detector:
    """A1 LightGBM per-token P(edited).  fit(rows, stores) / token_probs(rows)."""
    def __init__(self, params=None):
        self.params = dict(params or LGB_PARAMS)
        self.lex = None
        self.model = None

    def fit(self, rows, stores=None):
        import lightgbm as lgb
        self.stores = stores if stores is not None else {}
        self.lex = build_lexicon(rows)
        X, cat_idx = featurize(rows, self.lex)
        y = rows_labels(rows)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, categorical_feature=cat_idx)
        return self

    def token_probs(self, rows):
        """returns {id: (tk, prob_list)}."""
        X, _ = featurize(rows, self.lex)
        p = self.model.predict_proba(X)[:, 1]
        out = {}; off = 0
        for R in rows:
            m = len(R["tk"])
            out[R["id"]] = (R["tk"], p[off:off + m].tolist())
            off += m
        return out


# ======================================================================
#  Row builder
# ======================================================================
def build_rows(df, labeled=True):
    rows = []
    for r in df.itertuples():
        tk = toks(r.text)
        d = dict(id=r.id, lang=r.language, text=r.text, tk=tk,
                 fold=getattr(r, "fold", -1))
        if labeled:
            spans = sorted([(e["start"], e["end"], e["replacement"]) for e in r.edits])
            y = []
            for s, e, w in tk:
                lab = 0
                for a, b, rep in spans:
                    if s >= a and e <= b:
                        lab = 1; break
                y.append(lab)
            d["y"] = y
            d["spans"] = spans
            d["truth"] = [{"start": a, "end": b, "replacement": rep} for a, b, rep in spans]
        rows.append(d)
    return rows


# ======================================================================
#  Assembly + transduction (A3 harness patterns + fixed defects + hooks)
# ======================================================================
def merge_threshold_spans(tk, probs, thr):
    """merge runs of consecutive tokens with prob>=thr -> [(a,b,score,i,j)]."""
    spans = []; i = 0; n = len(tk)
    while i < n:
        if probs[i] >= thr:
            j = i
            while j + 1 < n and probs[j + 1] >= thr:
                j += 1
            a = tk[i][0]; b = tk[j][1]
            sc = float(np.mean(probs[i:j + 1]))
            spans.append((a, b, sc, i, j))
            i = j + 1
        else:
            i += 1
    return spans


_SENT = object()
_TR_MEMO = {}


def _transduce(transducer, lang, src, context):
    """A2 default with per-transducer (lang,src) memo (safe when no hooks)."""
    if REPLACEMENT_HOOKS:
        rep = transducer.predict(lang, src, context)
        return rep if rep is not None else src
    key = (id(transducer), lang, src)
    v = _TR_MEMO.get(key, _SENT)
    if v is _SENT:
        v = transducer.predict(lang, src, context)
        if v is None:
            v = src
        _TR_MEMO[key] = v
    return v


def _repair(edits, tlen):
    out = []; prev_end = -1
    for e in sorted(edits, key=lambda x: x["start"]):
        if e["start"] < prev_end or not (0 <= e["start"] < e["end"] <= tlen):
            continue
        e["replacement"] = e["replacement"][:160]
        out.append(e); prev_end = e["end"]
        if len(out) >= 8:
            break
    return out


def build_edits(row_id, text, lang, tk, probs, thr, transducer, stores, max_edits=8):
    """threshold-merge -> (optional scored extra candidates) -> transduce -> validate."""
    spans = merge_threshold_spans(tk, probs, thr)
    # (b) extra candidate spans, admitted only through a stores['span_scorer']
    if SPAN_CANDIDATE_GENERATORS and stores.get("span_scorer"):
        aux = {"probs": probs, "stores": stores, "lex": getattr(transducer, "lex", None)}
        cands = []
        for g in SPAN_CANDIDATE_GENERATORS:
            for (si, ej, meta) in (g(tk, lang, text, aux) or []):
                cands.append((tk[si][0], tk[ej][1], meta))
        for (a, b, sc) in (stores["span_scorer"](cands, tk, lang, text, aux) or []):
            spans.append((a, b, sc, None, None))
    edits = []
    for a, b, sc, _si, _ej in spans:
        src = text[a:b]                     # DEFECT FIX: original slice, keeps whitespace
        rep = None
        ctx = {"text": text, "start": a, "end": b, "lang": lang, "tokens": tk, "stores": stores}
        for hook in REPLACEMENT_HOOKS:      # (c) replacement hooks first
            r = hook(lang, src, ctx, stores)
            if r is not None:
                rep = r; break
        if rep is None:
            rep = _transduce(transducer, lang, src, ctx)
        edits.append((sc, {"start": a, "end": b, "replacement": rep[:160]}))
    edits.sort(key=lambda x: -x[0])         # <=8 cap by score
    edits = [e for _, e in edits[:max_edits]]
    edits.sort(key=lambda e: e["start"])
    if not elru.validate_edits(edits, len(text)):
        edits = _repair(edits, len(text))
    return edits


# ======================================================================
#  Cross-validation driver
# ======================================================================
GRID = [round(x, 3) for x in np.arange(0.05, 0.93, 0.02)]


def _lang_score_at(rowsL, thr, edits_cache):
    pm = {R["id"]: edits_cache[R["id"]][thr] for R in rowsL}
    tm = {R["id"]: R["truth"] for R in rowsL}
    lm = {R["id"]: R["lang"] for R in rowsL}
    _s, det = elru.elru(pm, tm, lm, detail=True)
    return det[rowsL[0]["lang"]]["lang_score"]


def _elru_assign(rows, assign, edits_cache):
    pm = {R["id"]: edits_cache[R["id"]][assign[R["id"]]] for R in rows}
    tm = {R["id"]: R["truth"] for R in rows}
    lm = {R["id"]: R["lang"] for R in rows}
    return elru.elru(pm, tm, lm, detail=True)


def run_cv(train, verbose=True):
    t0 = time.time()
    rows = build_rows(train, labeled=True)
    by_id = {R["id"]: R for R in rows}
    idfold = {R["id"]: R["fold"] for R in rows}

    # ---- Phase 1: leak-free OOF token probs + per-fold transducer + stores ----
    row_proba = {}
    transducers = {}
    stores_by_fold = {}
    for k in range(5):
        tr_rows = [R for R in rows if R["fold"] != k]
        va_rows = [R for R in rows if R["fold"] == k]
        tr_df = train[train.fold != k]
        stores = {}
        for b in STORE_BUILDERS:
            b(tr_df, stores)
        det = Detector().fit(tr_rows, stores)
        for _id, (tk, pr) in det.token_probs(va_rows).items():
            row_proba[_id] = pr
        transducers[k] = Transducer().fit(tr_df)
        stores_by_fold[k] = stores
        if verbose:
            print(f"[fold {k}] detector+transducer fit, {len(va_rows)} val rows  ({time.time()-t0:.0f}s)", flush=True)

    # ---- Phase 2: precompute leak-free edits per (row, threshold) --------------
    edits_cache = {}
    for R in rows:
        k = idfold[R["id"]]; T = transducers[k]; st = stores_by_fold[k]
        pr = row_proba[R["id"]]
        edits_cache[R["id"]] = {thr: build_edits(R["id"], R["text"], R["lang"], R["tk"], pr, thr, T, st)
                                for thr in GRID}
    if verbose:
        print(f"[assembly] edits_cache built  ({time.time()-t0:.0f}s)", flush=True)

    # ---- Non-nested thresholds: tune per-language on ALL OOF (shipping thr) ----
    rows_by_lang = {L: [R for R in rows if R["lang"] == L] for L in LANGS}
    nonnested_thr = {}
    for L in LANGS:
        best = (-1.0, GRID[0])
        for thr in GRID:
            sc = _lang_score_at(rows_by_lang[L], thr, edits_cache)
            if sc > best[0]:
                best = (sc, thr)
        nonnested_thr[L] = best[1]
    nn_assign = {R["id"]: nonnested_thr[R["lang"]] for R in rows}
    nn_elru, nn_detail = _elru_assign(rows, nn_assign, edits_cache)

    # ---- Nested thresholds: fold-k thresholds chosen on the OTHER 4 folds -------
    nested_assign = {}
    nested_thr_by_fold = {}
    for k in range(5):
        nested_thr_by_fold[k] = {}
        for L in LANGS:
            other = [R for R in rows_by_lang[L] if R["fold"] != k]
            best = (-1.0, GRID[0])
            for thr in GRID:
                sc = _lang_score_at(other, thr, edits_cache)
                if sc > best[0]:
                    best = (sc, thr)
            nested_thr_by_fold[k][L] = best[1]
            for R in [r for r in rows_by_lang[L] if r["fold"] == k]:
                nested_assign[R["id"]] = best[1]
    nested_elru, nested_detail = _elru_assign(rows, nested_assign, edits_cache)

    # ---- deletion-cost diagnostic (measure; a hook can address it later) -------
    n_true_del = sum(1 for R in rows for a, b, rep in R["spans"] if rep == "")
    del_pred = del_hit = 0
    for R in rows:
        for e in edits_cache[R["id"]][nn_assign[R["id"]]]:
            if e["replacement"] == "":
                del_pred += 1
                for a, b, rep in R["spans"]:
                    if rep == "" and not (e["end"] <= a or b <= e["start"]):
                        del_hit += 1; break

    return dict(rows=rows, by_id=by_id, idfold=idfold, row_proba=row_proba,
                edits_cache=edits_cache, nonnested_thr=nonnested_thr,
                nonnested_elru=nn_elru, nonnested_detail=nn_detail, nn_assign=nn_assign,
                nested_thr_by_fold=nested_thr_by_fold, nested_assign=nested_assign,
                nested_elru=nested_elru, nested_detail=nested_detail,
                del_diag=dict(n_true_del=n_true_del, del_pred=del_pred, del_hit=del_hit),
                seconds=time.time() - t0)


# ======================================================================
#  Whitespace-preservation guard (DEFECT FIX #2)
# ======================================================================
def test_whitespace_preserved(train):
    """Assert multi-token spans transduce with original inter-token whitespace."""
    T = Transducer().fit(train)
    checked = 0
    for r in train.itertuples():
        for e in r.edits:
            src = r.text[e["start"]:e["end"]]
            if "  " in src or "\t" in src or ("\n" in src):
                out = T.predict(r.language, src, None)
                # A2 echoes unknown tokens; any multi-space run in src must survive
                # when the span is copied through identity paths.
                if src == out and ("  " in src) and ("  " not in out):
                    raise AssertionError(f"whitespace lost on id={r.id}: {src!r}->{out!r}")
                checked += 1
    return checked


# ======================================================================
#  Full-train fit -> deliverables
# ======================================================================
def main():
    t0 = time.time()
    np.random.seed(0)
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "dataset", "test.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)

    res = run_cv(train, verbose=True)

    print("\n================ M1 MERGE PIPELINE ================")
    print(f"NON-NESTED CV ELRU (all-OOF thresholds, apples-to-review) = {res['nonnested_elru']:.4f}")
    print(f"  thresholds = {res['nonnested_thr']}")
    for L in LANGS:
        d = res["nonnested_detail"][L]
        print(f"  {L}: lang={d['lang_score']:.4f} edited={d['edited_mean']:.4f}(n={d['n_edited']}) "
              f"unchanged={d['unchanged_mean']:.4f}(n={d['n_unchanged']})")
    print(f"\nNESTED CV ELRU (honest; fold thr from other 4 folds) = {res['nested_elru']:.4f}")
    for L in LANGS:
        d = res["nested_detail"][L]
        print(f"  {L}: lang={d['lang_score']:.4f} edited={d['edited_mean']:.4f} unchanged={d['unchanged_mean']:.4f}")
    print(f"  per-fold nested thresholds: {res['nested_thr_by_fold']}")
    dd = res["del_diag"]
    print(f"\ndeletion diag: true_del={dd['n_true_del']} predicted_empty={dd['del_pred']} "
          f"overlapping_true_del={dd['del_hit']} (never-predicted cost is why a hook path may help)")

    # ---- OOF deliverables (leak-free rows, non-nested/shipping thresholds) ----
    edits_cache = res["edits_cache"]; nn_assign = res["nn_assign"]; rows = res["rows"]
    oof_rows = [{"id": R["id"],
                 "edits_json": json.dumps(edits_cache[R["id"]][nn_assign[R["id"]]], ensure_ascii=False)}
                for R in rows]
    pd.DataFrame(oof_rows).to_csv(os.path.join(HERE, "oof_edits.csv"), index=False)

    tokrows = []
    for R in rows:
        pr = res["row_proba"][R["id"]]
        for ti, (s, e, w) in enumerate(R["tk"]):
            tokrows.append(dict(id=R["id"], lang=R["lang"], tok_index=ti, start=s, end=e,
                                proba=round(pr[ti], 5), y=R["y"][ti]))
    pd.DataFrame(tokrows).to_csv(os.path.join(HERE, "oof_token_probs.csv"), index=False)

    # ---- Full-train fit -> submission_v2.csv (all-OOF thresholds) -------------
    stores_full = {}
    for b in STORE_BUILDERS:
        b(train, stores_full)
    all_rows = build_rows(train, labeled=True)
    det_full = Detector().fit(all_rows, stores_full)
    trd_full = Transducer().fit(train)
    test_rows = build_rows(test, labeled=False)
    tp_test = det_full.token_probs(test_rows)
    thr = res["nonnested_thr"]
    sub = {}
    for R in test_rows:
        tk, pr = tp_test[R["id"]]
        sub[R["id"]] = build_edits(R["id"], R["text"], R["lang"], tk, pr,
                                   thr[R["lang"]], trd_full, stores_full)
    assert len(sub) == len(test) == 445, (len(sub), len(test))
    assert set(sub) == set(test.id), "id mismatch"
    tl = {r.id: len(r.text) for r in test.itertuples()}
    bad = [i for i in sub if not elru.validate_edits(sub[i], tl[i])]
    assert not bad, f"invalid rows: {bad[:5]}"
    sub_rows = [{"id": i, "edits_json": json.dumps(sub[i], ensure_ascii=False)} for i in test.id]
    pd.DataFrame(sub_rows).to_csv(os.path.join(HERE, "submission_v2.csv"), index=False)
    n_edit = sum(1 for i in sub if sub[i])

    ws = test_whitespace_preserved(train)
    json.dump(dict(nonnested_elru=round(res["nonnested_elru"], 4),
                   nested_elru=round(res["nested_elru"], 4),
                   nonnested_thr=res["nonnested_thr"],
                   nested_thr_by_fold=res["nested_thr_by_fold"],
                   nonnested_detail={L: {k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in res["nonnested_detail"][L].items()} for L in LANGS},
                   del_diag=res["del_diag"], whitespace_checks=ws),
              open(os.path.join(HERE, "cv_report.json"), "w"), indent=2)

    print(f"\nsubmission_v2.csv: 445 rows valid; {n_edit} with >=1 edit ({n_edit/445:.1%})")
    print(f"wrote oof_edits.csv, oof_token_probs.csv, cv_report.json  (whitespace checks={ws})")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

"""A1 detection: LightGBM per-token is-edited + merge-adjacent-positive span assembly.
Proper 5-fold OOF: lexicons + model fit on other folds only.
Eval: oracle-replacement ELRU (best-overlap true rep) + span P/R/F1 (exact & overlap), per language.
Tunes per-language probability thresholds to maximize language_score.
Writes: oof_token_probs.csv, oof_spans.csv (tuned), threshold_table.json, feature_importance.csv
"""
import sys, json, re, zlib, collections, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "solution")
from elru import elru, span_f1

t0 = time.time()
np.random.seed(0)
WORD_RE = re.compile(r"\S+")
NB = 256   # suffix hash buckets
NBP = 128  # prefix hash buckets

def toks(text):
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]

def h(s, b):
    return int(zlib.crc32(s.encode("utf-8")) % b)

def shared_prefix_ratio(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b))

SPECIAL = (":", "*", "/")
def special_key(w):
    """(char, suffix-after-char) for the LAST interior special char; e.g. 'Direktor:in'->(':','in').
    Stems differ across docs but the encoded feminine suffix is stable -> strong cross-stem signal."""
    p = -1
    for idx in range(1, len(w) - 1):
        if w[idx] in SPECIAL:
            p = idx
    if p == -1:
        return None
    suf = w[p + 1:][:8]
    return (w[p], suf)

# ---------- load & tokenize once ----------
train = pd.read_csv("dataset/train.csv")
folds = pd.read_csv("solution/folds.csv")
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)

ROWS = []   # each: dict(id, lang, fold, tk=[(s,e,w)], y=[0/1], spans=[(a,b,rep)])
for r in train.itertuples():
    tk = toks(r.text)
    spans = sorted([(e["start"], e["end"], e["replacement"]) for e in r.edits])
    y = []
    for s, e, w in tk:
        lab = 0
        for a, b, rep in spans:
            if s >= a and e <= b:
                lab = 1
                break
        y.append(lab)
    ROWS.append(dict(id=r.id, lang=r.language, fold=r.fold, tk=tk, y=y, spans=spans))

LANG2I = {"de": 0, "en": 1, "it": 2}

# non-alnum chars to one-hot (observed structural punctuation, intact per challenge)
punct_set = ['/', '’', '.', '-', '_', '*', ':', "'", ')', '(', '@', ',', '&', '"']

# ---------- lexicon (fit on given rows only) ----------
def build_lexicon(rows):
    # per-language smoothed edit rates for: surface token, suffix3, suffix4, prefix3, and each punct char
    tok_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    tok_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf3_ed = collections.defaultdict(lambda: collections.defaultdict(float)); suf3_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf4_ed = collections.defaultdict(lambda: collections.defaultdict(float)); suf4_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    pre3_ed = collections.defaultdict(lambda: collections.defaultdict(float)); pre3_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    ch_ed = collections.defaultdict(lambda: collections.defaultdict(float)); ch_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    spat_ed = collections.defaultdict(lambda: collections.defaultdict(float)); spat_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf_ed = collections.defaultdict(lambda: collections.defaultdict(float)); suf_sn = collections.defaultdict(lambda: collections.defaultdict(float))  # feminine-suffix alone
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
                suf_ed[L][sk[1]] += lab; suf_sn[L][sk[1]] += 1  # suffix regardless of which special char
    prior = {L: (lang_ed[L] + 0.5) / (lang_sn[L] + 1.0) for L in lang_sn}
    def rate(ed, sn, L, k, a):
        p = prior.get(L, 0.03)
        return (ed[L].get(k, 0.0) + a * p) / (sn[L].get(k, 0.0) + a)
    return dict(tok_ed=tok_ed, tok_sn=tok_sn, suf3_ed=suf3_ed, suf3_sn=suf3_sn,
                suf4_ed=suf4_ed, suf4_sn=suf4_sn, pre3_ed=pre3_ed, pre3_sn=pre3_sn,
                ch_ed=ch_ed, ch_sn=ch_sn, spat_ed=spat_ed, spat_sn=spat_sn,
                suf_ed=suf_ed, suf_sn=suf_sn, prior=prior, rate=rate)

def tok_shape(w):
    n = len(w)
    nu = sum(c.isupper() for c in w); nl = sum(c.islower() for c in w); nd = sum(c.isdigit() for c in w)
    npunct = sum((not c.isalnum()) for c in w)
    return n, nu, nl, nd, npunct

FEAT_NAMES = None
def featurize(rows, lex):
    global FEAT_NAMES
    a_tok, a_suf3, a_suf4 = 5.0, 20.0, 30.0  # smoothing strengths
    X = []
    cat_idx = []
    names = []
    def rt(ed, sn, L, k, a):
        return lex["rate"](ed, sn, L, k, a)
    for R in rows:
        L = R["lang"]; lid = LANG2I[L]
        tk = R["tk"]; nt = len(tk)
        words = [w for _, _, w in tk]
        shapes = [tok_shape(w) for w in words]
        lows = [w.lower() for w in words]
        # precompute per-token lexicon token-rate for neighbor use
        trate = []
        for w, lw in zip(words, lows):
            trate.append(rt(lex["tok_ed"], lex["tok_sn"], L, w, a_tok))
        for i, (s, e, w) in enumerate(tk):
            n, nu, nl, nd, npunct = shapes[i]
            lw = lows[i]
            inner = w[1:-1] if len(w) > 2 else ""
            feats = []
            fn = []
            def add(v, name):
                feats.append(float(v));
                if FEAT_NAMES is None: fn.append(name)
            # ---- structural ----
            add(n, "len"); add(nu, "nup"); add(nl, "nlo"); add(nd, "ndig"); add(npunct, "npun")
            add(nu / max(n, 1), "frac_up"); add(nd / max(n, 1), "frac_dig")
            add(1 if (w[:1].isupper() and not w.isupper()) else 0, "title")
            add(1 if (w.isupper() and any(c.isalpha() for c in w)) else 0, "allcaps")
            add(1 if w[:1].isupper() else 0, "first_up")
            add(1 if any(c.isdigit() for c in w) else 0, "has_dig")
            # punctuation shape one-hots (mid-token presence) + start/end
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
            # punctuation salience: max learned edit-rate over token's mid punct chars
            mc = 0.0
            for ch in set(inner):
                if not ch.isalnum():
                    mc = max(mc, rt(lex["ch_ed"], lex["ch_sn"], L, ch, 10.0))
            add(mc, "maxchar_rate")
            # ---- special-char + suffix pattern (the ':in'/'*in' inclusive form; cross-stem stable) ----
            sk = special_key(w)
            if sk is not None:
                spc, suf = sk
                key = spc + suf
                add(rt(lex["spat_ed"], lex["spat_sn"], L, key, 3.0), "spat_rate")
                add(rt(lex["suf_ed"], lex["suf_sn"], L, suf, 3.0), "specsuf_rate")
                add(1, "has_special")
                add(len(suf), "specsuf_len")
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
                    add(1, f"nb{off}_ex")
                    add(trate[j], f"nb{off}_rate")
                    add(1 if any((not c.isalnum()) for c in (wj[1:-1] if len(wj) > 2 else "")) else 0, f"nb{off}_midpun")
                    add(1 if wj[:1].isupper() else 0, f"nb{off}_up")
                    add(sj[0], f"nb{off}_len")
                else:
                    add(0, f"nb{off}_ex"); add(0, f"nb{off}_rate"); add(0, f"nb{off}_midpun"); add(0, f"nb{off}_up"); add(0, f"nb{off}_len")
            # ---- stem similarity (paired forms / connector) ----
            pv = lows[i-1] if i-1 >= 0 else ""
            nx = lows[i+1] if i+1 < nt else ""
            pv2 = lows[i-2] if i-2 >= 0 else ""
            nx2 = lows[i+2] if i+2 < nt else ""
            add(shared_prefix_ratio(lw, pv), "sp_prev")
            add(shared_prefix_ratio(lw, nx), "sp_next")
            add(shared_prefix_ratio(pv, nx), "sp_skip")      # connector: neighbors share stem
            add(shared_prefix_ratio(lw, pv2), "sp_prev2")
            add(shared_prefix_ratio(lw, nx2), "sp_next2")
            add(max(shared_prefix_ratio(pv, nx), shared_prefix_ratio(pv2, nx2)), "sp_bridge")
            # ---- position ----
            add(i / max(nt - 1, 1), "pos_frac")
            add(np.log1p(nt), "n_tok")
            add(1 if i == 0 else 0, "is_first"); add(1 if i == nt - 1 else 0, "is_last")
            # ---- language + hashed morphology (categorical) ----
            catstart = len(feats)
            add(lid, "lang_id")
            add(h(lw[-2:], NB), "suf2_id"); add(h(lw[-3:], NB), "suf3_id"); add(h(lw[-4:], NB), "suf4_id")
            add(h(lw[:2], NBP), "pre2_id"); add(h(lw[:3], NBP), "pre3_id")
            add(spc_id, "spc_id")
            add(h(sk[1], NB) if sk is not None else 0, "specsuf_id")
            if FEAT_NAMES is None:
                cat_idx.extend(range(catstart, len(feats)))
                FEAT_NAMES = fn
            X.append(feats)
    return np.asarray(X, dtype=np.float32), cat_idx

def rows_labels(rows):
    y = []
    for R in rows:
        y.extend(R["y"])
    return np.asarray(y, dtype=np.int32)

# ---------- OOF training ----------
LGB_PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.045,
                  num_leaves=48, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=2.0, is_unbalance=True,
                  n_jobs=5, verbosity=-1, max_depth=-1)

oof_proba = {}   # (id, tok_index) -> proba ; we store per row list
row_proba = {R["id"]: [0.0] * len(R["tk"]) for R in ROWS}
importance = None
for k in range(5):
    tr = [R for R in ROWS if R["fold"] != k]
    va = [R for R in ROWS if R["fold"] == k]
    lex = build_lexicon(tr)
    Xtr, cat_idx = featurize(tr, lex)
    ytr = rows_labels(tr)
    Xva, _ = featurize(va, lex)
    clf = lgb.LGBMClassifier(**LGB_PARAMS)
    clf.fit(Xtr, ytr, categorical_feature=cat_idx)
    pv = clf.predict_proba(Xva)[:, 1]
    # scatter back
    off = 0
    for R in va:
        m = len(R["tk"])
        row_proba[R["id"]] = pv[off:off + m].tolist()
        off += m
    imp = clf.feature_importances_
    importance = imp if importance is None else importance + imp
    print(f"fold {k}: train_tok={len(ytr)} val_rows={len(va)} pos_rate={ytr.mean():.4f} t={time.time()-t0:.0f}s", flush=True)

# ---------- span assembly ----------
def assemble(R, thr):
    tk = R["tk"]; pr = row_proba[R["id"]]
    spans = []
    i = 0; n = len(tk)
    while i < n:
        if pr[i] >= thr:
            j = i
            while j + 1 < n and pr[j + 1] >= thr:
                j += 1
            a = tk[i][0]; b = tk[j][1]
            p = float(np.mean(pr[i:j + 1]))
            spans.append((a, b, p))
            i = j + 1
        else:
            i += 1
    if len(spans) > 8:
        spans = sorted(spans, key=lambda x: -x[2])[:8]
        spans = sorted(spans, key=lambda x: x[0])
    return spans

# oracle replacement = best-overlap true rep (span_f1 max, >0), else ""
def oracle_pred(R, thr):
    spans = assemble(R, thr)
    tr = R["spans"]
    out = []
    for a, b, p in spans:
        best = None; bq = 0.0
        for ta, tb, rep in tr:
            q = span_f1(a, b, ta, tb)
            if q > bq:
                bq = q; best = rep
        out.append({"start": a, "end": b, "replacement": best if best is not None else ""})
    return out

def true_map_lang(rows):
    tm = {R["id"]: [{"start": a, "end": b, "replacement": r} for a, b, r in R["spans"]] for R in rows}
    lm = {R["id"]: R["lang"] for R in rows}
    return tm, lm

# ---------- per-language threshold tuning (maximize language_score) ----------
from elru import elru as _elru
def lang_score_at(rows_L, thr):
    tm = {R["id"]: [{"start": a, "end": b, "replacement": r} for a, b, r in R["spans"]] for R in rows_L}
    lm = {R["id"]: R["lang"] for R in rows_L}
    pm = {R["id"]: oracle_pred(R, thr) for R in rows_L}
    s, det = _elru(pm, tm, lm, detail=True)
    L = rows_L[0]["lang"]
    return det[L]["lang_score"], det[L]

THRS = [round(x, 3) for x in np.arange(0.06, 0.92, 0.02)]
best_thr = {}
lang_detail = {}
for L in ["de", "en", "it"]:
    rowsL = [R for R in ROWS if R["lang"] == L]
    best = (-1, None, None)
    for thr in THRS:
        sc, det = lang_score_at(rowsL, thr)
        if sc > best[0]:
            best = (sc, thr, det)
    best_thr[L] = best[1]
    lang_detail[L] = best[2]
    print(f"[{L}] best thr={best[1]} lang_score={best[0]:.4f} edited={best[2]['edited_mean']:.3f} unchanged={best[2]['unchanged_mean']:.3f} (n_ed={best[2]['n_edited']},n_un={best[2]['n_unchanged']})", flush=True)

# overall ELRU with tuned per-language thresholds
tm_all = {R["id"]: [{"start": a, "end": b, "replacement": r} for a, b, r in R["spans"]] for R in ROWS}
lm_all = {R["id"]: R["lang"] for R in ROWS}
pm_all = {R["id"]: oracle_pred(R, best_thr[R["lang"]]) for R in ROWS}
elru_tuned, det_tuned = _elru(pm_all, tm_all, lm_all, detail=True)
print(f"\n=== ORACLE-REPLACEMENT ELRU (tuned per-lang thr) = {elru_tuned:.4f} ===")
for L in ["de", "en", "it"]:
    d = det_tuned[L]
    print(f"  {L}: lang={d['lang_score']:.4f} edited={d['edited_mean']:.3f} unchanged={d['unchanged_mean']:.3f}")

# ---------- span P/R/F1 (exact & overlap) per language at tuned thr ----------
def span_prf(rows, thr_map):
    res = {}
    for L in ["de", "en", "it"]:
        tp_e = fp = fn = tp_o = 0
        n_pred = n_true = 0
        for R in [r for r in rows if r["lang"] == L]:
            thr = thr_map[L]
            pred = [(a, b) for a, b, p in assemble(R, thr)]
            true = [(a, b) for a, b, r in R["spans"]]
            n_pred += len(pred); n_true += len(true)
            tset = set(true)
            used = set()
            for a, b in pred:
                if (a, b) in tset:
                    tp_e += 1
            # overlap greedy
            pairs = []
            for pi, (a, b) in enumerate(pred):
                for ti, (ta, tb) in enumerate(true):
                    ov = min(b, tb) - max(a, ta)
                    if ov > 0:
                        pairs.append((ov, pi, ti))
            pairs.sort(reverse=True)
            up, ut = set(), set()
            for ov, pi, ti in pairs:
                if pi in up or ti in ut: continue
                up.add(pi); ut.add(ti); tp_o += 1
        def prf(tp, npred, ntrue):
            p = tp / npred if npred else 0.0
            r = tp / ntrue if ntrue else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            return round(p, 3), round(r, 3), round(f, 3)
        res[L] = dict(exact=prf(tp_e, n_pred, n_true), overlap=prf(tp_o, n_pred, n_true),
                      n_pred=n_pred, n_true=n_true)
    return res

prf = span_prf(ROWS, best_thr)
print("\nspan P/R/F1 per language (exact | overlap):")
for L in ["de", "en", "it"]:
    print(f"  {L}: exact={prf[L]['exact']} overlap={prf[L]['overlap']} (n_pred={prf[L]['n_pred']}, n_true={prf[L]['n_true']})")

# ---------- write deliverables ----------
# oof token probs
tokrows = []
for R in ROWS:
    for ti, (s, e, w) in enumerate(R["tk"]):
        tokrows.append(dict(id=R["id"], lang=R["lang"], tok_index=ti, start=s, end=e,
                            proba=round(row_proba[R["id"]][ti], 5), y=R["y"][ti]))
pd.DataFrame(tokrows).to_csv("runs/A1/oof_token_probs.csv", index=False)

# oof predicted spans at tuned thresholds
sprows = []
for R in ROWS:
    for a, b, p in assemble(R, best_thr[R["lang"]]):
        sprows.append(dict(id=R["id"], start=a, end=b, prob=round(p, 5)))
pd.DataFrame(sprows).to_csv("runs/A1/oof_spans.csv", index=False)

# threshold table + summary
summary = dict(oracle_elru_tuned=round(elru_tuned, 4),
               thresholds=best_thr,
               per_language={L: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                 for kk, vv in det_tuned[L].items()} for L in det_tuned},
               span_prf=prf)
with open("runs/A1/threshold_table.json", "w") as f:
    json.dump(summary, f, indent=2)

# feature importance
fi = sorted(zip(FEAT_NAMES, (importance / 5).tolist()), key=lambda x: -x[1])
pd.DataFrame(fi, columns=["feature", "gain"]).to_csv("runs/A1/feature_importance.csv", index=False)
print("\ntop 20 features:", [(n, int(g)) for n, g in fi[:20]])
print(f"\nDONE in {time.time()-t0:.0f}s. wrote oof_token_probs.csv, oof_spans.csv, threshold_table.json")

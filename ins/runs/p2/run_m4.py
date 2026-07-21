"""M4 COMPOSER + VERIFIER -- integrated pipeline with span-level reranker (GATE form).

Pipeline = M1 detector/transducer + M2 (de) + M3 (it/en) plug-ins (composed by
m4_ext) + a SECOND-STAGE SPAN RERANKER used as a PRECISION GATE over the base
threshold-merged spine spans (+ M2/M3 generator candidates), + an optional
group-consistency pass.

WHY GATE (not greedy-replace): a first cut that let the reranker choose boundaries
greedily REGRESSED (nested 0.499 vs base 0.519) -- it dropped multi-token NPs in
favour of high-score singletons.  The gate form keeps the base per-language merge
boundaries (recall preserved; at gate=0 it *is* the base merge) and uses the
reranker only to REMOVE low-quality spans -> attacks the de/it unchanged-row false
positives without sacrificing edited recall.  The final span decision is per-language
(spine threshold th_L, reranker gate g_L), selected NESTED (fold-k picks th/g from the
other 4 folds) and, for shipping, all-OOF.

Leak-free per fold: OOF detector probs (row scored by a detector that never saw its
fold) + per-fold transducer + per-fold stores; candidate features + labels use only
the row's own fold-out artifacts; the reranker is CROSS-FIT (reranker_k trained on
folds!=k scores fold==k).  Reported: overall + per-language NESTED and NON-NESTED
ELRU, per-type IoU>=.5 recall vs the iteration-1 loss map, unchanged-row FP counts.

Usage:  cd ~/insled && OMP_NUM_THREADS=5 nice -n 10 ~/venv/bin/python runs/M4/run_m4.py [mode]
"""
import os, sys, json, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pipeline
import m4_ext

elru = pipeline.elru
LANGS = pipeline.LANGS
HERE = os.path.dirname(os.path.abspath(__file__))
MARKS = set(":*∗/")
_STRIP = ".,;:()»«\"'“”’`-–—"

# candidate proposal ladder (dense so every spine-at-theta run is a scored candidate)
LADDER = [round(x, 3) for x in np.arange(0.03, 0.80, 0.02)]
# operating-point search grids (per language): spine threshold x reranker gate
THETA_GRID = [round(x, 3) for x in np.arange(0.03, 0.76, 0.04)]
GATE_GRID = [0.0, 0.04, 0.08, 0.12, 0.16, 0.22, 0.30, 0.40, 0.50, 0.62, 0.75]

RR_PARAMS = dict(objective="binary", n_estimators=350, learning_rate=0.04,
                 num_leaves=31, min_child_samples=25, subsample=0.85, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=3.0, is_unbalance=True,
                 random_state=0, n_jobs=5, verbosity=-1)

FEATS = ["mean_p", "max_p", "min_p", "first_p", "last_p", "left_p", "right_p", "gap_out",
         "n_tok", "n_char", "frac_hi", "n_hi",
         "has_mark", "n_mark_tok", "frac_mark", "lcp_first2", "multiword",
         "a2_exact", "a2_norm", "a2_mark", "a2_suffix", "a2_multi", "a2_identity", "a2_del",
         "hook_any", "hook_collapse", "hook_slash", "hook_ennorm", "mem_hit",
         "rep_changed", "rep_len_ratio", "src_short", "src_cap", "src_lower",
         "is_gen_de", "is_gen_np", "gen_sr", "gen_fem",
         "lang_de", "lang_en", "lang_it"]


def iou(a, b, c, d):
    ov = max(0, min(b, d) - max(a, c)); un = max(b, d) - min(a, c)
    return ov / un if un > 0 else 0.0


def _lcp_ratio(a, b):
    a, b = a.lower(), b.lower()
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b), 1)


def span_type(lang, src):
    nt = len(src.split())
    marked = any(c in MARKS for c in src)
    return ("single" if nt == 1 else "multi") + ("_marked" if marked else "_plain")


# ======================================================================
#  Phase 1: leak-free OOF detector probs + per-fold transducer/stores
# ======================================================================
def fit_folds(train, verbose=True):
    t0 = time.time()
    rows = pipeline.build_rows(train, labeled=True)
    for R in rows:
        R["fold"] = R["fold"]
    idfold = {R["id"]: R["fold"] for R in rows}
    row_proba, transducers, stores_by_fold = {}, {}, {}
    for k in range(5):
        tr_rows = [R for R in rows if R["fold"] != k]
        va_rows = [R for R in rows if R["fold"] == k]
        tr_df = train[train.fold != k]
        stores = {}
        for b in pipeline.STORE_BUILDERS:
            b(tr_df, stores)
        det = pipeline.Detector().fit(tr_rows, stores)
        for _id, (tk, pr) in det.token_probs(va_rows).items():
            row_proba[_id] = pr
        transducers[k] = pipeline.Transducer().fit(tr_df)
        stores_by_fold[k] = stores
        if verbose:
            print(f"[fold {k}] fit  ({time.time()-t0:.0f}s)", flush=True)
    return rows, idfold, row_proba, transducers, stores_by_fold


# ======================================================================
#  Phase 2: candidate proposal + transduction + featurization
# ======================================================================
def _transduce_full(transducer, lang, src, ctx, stores):
    rep, hookname = None, ""
    for hook in pipeline.REPLACEMENT_HOOKS:
        r = hook(lang, src, ctx, stores)
        if r is not None:
            rep, hookname = r, hook.__name__
            break
    a2_rep, a2_mech = transducer.predict_dbg(lang, src, ctx)
    if rep is None:
        rep = a2_rep
    return rep, hookname, a2_mech


def merge_at(tk, probs, thr):
    return [(a, b) for (a, b, sc, si, ej) in pipeline.merge_threshold_spans(tk, probs, thr)]


def propose_candidates(tk, probs, lang, text, stores, transducer):
    cmap = {}
    for thr in LADDER:
        for (a, b, sc, si, ej) in pipeline.merge_threshold_spans(tk, probs, thr):
            cmap.setdefault((a, b), dict(a=a, b=b, si=si, ej=ej, meta={}, is_gen=0.0))
    aux = {"probs": probs, "stores": stores, "lex": getattr(transducer, "lex", None)}
    for g in pipeline.SPAN_CANDIDATE_GENERATORS:
        gname = "de" if g.__name__ == "span_generator" else ("np" if g.__name__ == "np_generator" else g.__name__)
        for (si, ej, meta) in (g(tk, lang, text, aux) or []):
            a, b = tk[si][0], tk[ej][1]
            c = cmap.get((a, b))
            if c is None:
                c = dict(a=a, b=b, si=si, ej=ej, meta={}, is_gen=0.0)
                cmap[(a, b)] = c
            c["meta"] = {**c["meta"], **(meta or {})}
            c["is_gen"] = 1.0
            c["gen_" + gname] = 1.0
    return cmap


def featurize_candidate(c, tk, probs, lang, text, stores, transducer):
    n = len(tk); si, ej = c["si"], c["ej"]
    a, b = c["a"], c["b"]
    src = text[a:b]; c["src"] = src
    sp = probs[si:ej + 1]
    ctx = {"text": text, "start": a, "end": b, "lang": lang, "tokens": tk, "stores": stores}
    rep, hookname, a2_mech = _transduce_full(transducer, lang, src, ctx, stores)
    c["rep"] = rep
    words = [w for _, _, w in tk]
    n_mark_tok = sum(1 for j in range(si, ej + 1) if any(ch in MARKS for ch in words[j]))
    ntok = ej - si + 1
    left_p = probs[si - 1] if si - 1 >= 0 else 0.0
    right_p = probs[ej + 1] if ej + 1 < n else 0.0
    core = src.strip(_STRIP)
    d = {
        "mean_p": float(np.mean(sp)), "max_p": float(np.max(sp)), "min_p": float(np.min(sp)),
        "first_p": float(sp[0]), "last_p": float(sp[-1]), "left_p": float(left_p), "right_p": float(right_p),
        "gap_out": float(max(left_p, right_p) - np.min(sp)),
        "n_tok": float(ntok), "n_char": float(b - a),
        "frac_hi": float(np.mean([1.0 if x >= 0.5 else 0.0 for x in sp])),
        "n_hi": float(sum(1 for x in sp if x >= 0.5)),
        "has_mark": 1.0 if any(ch in MARKS for ch in src) else 0.0,
        "n_mark_tok": float(n_mark_tok), "frac_mark": float(n_mark_tok / max(ntok, 1)),
        "lcp_first2": _lcp_ratio(words[si], words[si + 1]) if ntok >= 2 else 0.0,
        "multiword": 1.0 if ntok >= 2 else 0.0,
        "a2_exact": 1.0 if a2_mech == "exact" else 0.0,
        "a2_norm": 1.0 if a2_mech == "norm" else 0.0,
        "a2_mark": 1.0 if a2_mech == "mark_tpl" else 0.0,
        "a2_suffix": 1.0 if a2_mech == "suffix" else 0.0,
        "a2_multi": 1.0 if a2_mech == "multi" else 0.0,
        "a2_identity": 1.0 if a2_mech == "identity" else 0.0,
        "a2_del": 1.0 if a2_mech == "del_ml" else 0.0,
        "hook_any": 1.0 if hookname else 0.0,
        "hook_collapse": 1.0 if hookname == "collapse_hook" else 0.0,
        "hook_slash": 1.0 if hookname == "it_slash_hook" else 0.0,
        "hook_ennorm": 1.0 if hookname == "en_norm_hook" else 0.0,
        "mem_hit": 1.0 if (a2_mech in ("exact", "norm") or hookname == "exact_first_hook") else 0.0,
        "rep_changed": 1.0 if rep != src else 0.0,
        "rep_len_ratio": float(len(rep) / max(len(src), 1)),
        "src_short": 1.0 if len(core) <= 6 else 0.0,
        "src_cap": 1.0 if core[:1].isupper() else 0.0,
        "src_lower": 1.0 if (core.isalpha() and core.islower()) else 0.0,
        "is_gen_de": float(c.get("gen_de", 0.0)), "is_gen_np": float(c.get("gen_np", 0.0)),
        "gen_sr": float(c["meta"].get("sr", 0.0)), "gen_fem": 1.0 if c["meta"].get("fem") else 0.0,
        "lang_de": 1.0 if lang == "de" else 0.0, "lang_en": 1.0 if lang == "en" else 0.0,
        "lang_it": 1.0 if lang == "it" else 0.0,
    }
    c["feat"] = [d[k] for k in FEATS]
    return c


def build_candidates(rows, idfold, row_proba, transducers, stores_by_fold, labeled=True):
    cand_map = {}
    for R in rows:
        k = idfold[R["id"]]; T = transducers[k]; st = stores_by_fold[k]
        pr = row_proba[R["id"]]; tk = R["tk"]; lang = R["lang"]; text = R["text"]
        cmap = propose_candidates(tk, pr, lang, text, st, T)
        for c in cmap.values():
            featurize_candidate(c, tk, pr, lang, text, st, T)
            if labeled:
                best = 0.0
                for (ts, te, rep) in R["spans"]:
                    if rep == "":
                        continue
                    best = max(best, iou(c["a"], c["b"], ts, te))
                c["label"] = 1 if best >= 0.5 else 0
        cand_map[R["id"]] = cmap
    return cand_map


# ======================================================================
#  Phase 3: cross-fit reranker
# ======================================================================
def crossfit_reranker(rows, idfold, cand_map):
    import lightgbm as lgb
    models = {}
    for k in range(5):
        Xtr, ytr = [], []
        for R in rows:
            if idfold[R["id"]] == k:
                continue
            for c in cand_map[R["id"]].values():
                Xtr.append(c["feat"]); ytr.append(c["label"])
        m = lgb.LGBMClassifier(**RR_PARAMS)
        m.fit(np.asarray(Xtr, dtype=np.float32), np.asarray(ytr, dtype=np.int32))
        models[k] = m
        va = [R for R in rows if idfold[R["id"]] == k]
        Xva, ref = [], []
        for R in va:
            for key, c in cand_map[R["id"]].items():
                Xva.append(c["feat"]); ref.append((R["id"], key))
        if Xva:
            pv = m.predict_proba(np.asarray(Xva, dtype=np.float32))[:, 1]
            for (rid, key), p in zip(ref, pv):
                cand_map[rid][key]["rr"] = float(p)
    return models


def train_full_reranker(rows, cand_map):
    import lightgbm as lgb
    X = [c["feat"] for R in rows for c in cand_map[R["id"]].values()]
    y = [c["label"] for R in rows for c in cand_map[R["id"]].values()]
    m = lgb.LGBMClassifier(**RR_PARAMS)
    m.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32))
    return m


# ======================================================================
#  Phase 4: GATE assembly (spine threshold th + reranker gate g)
# ======================================================================
def assemble_gate(R, probs, cmap, th, gate, drop_identity=False, max_edits=8):
    tk = R["tk"]; tlen = len(R["text"])
    picked = {}
    for (a, b) in merge_at(tk, probs, th):
        c = cmap.get((a, b))
        if c is None:
            continue
        if c.get("rr", 1.0) >= gate:
            picked[(a, b)] = c
    for key, c in cmap.items():
        if c.get("is_gen", 0.0) and c.get("rr", 0.0) >= gate:
            picked[key] = c
    cl = sorted(picked.values(), key=lambda c: -c.get("rr", 0.0))
    chosen, occ = [], []
    for c in cl:
        if any(not (c["b"] <= a or bb <= c["a"]) for (a, bb) in occ):
            continue
        chosen.append(c); occ.append((c["a"], c["b"]))
    edits = []
    for c in chosen:
        rep = c["rep"]
        if drop_identity and rep == c["src"]:
            continue
        edits.append({"start": c["a"], "end": c["b"], "replacement": rep[:160]})
    edits.sort(key=lambda e: e["start"])
    edits = edits[:max_edits]
    if not elru.validate_edits(edits, tlen):
        edits = pipeline._repair(edits, tlen)
    return edits


def lang_score(rowsL, edits_map):
    pm = {R["id"]: edits_map[R["id"]] for R in rowsL}
    tm = {R["id"]: R["truth"] for R in rowsL}
    lm = {R["id"]: R["lang"] for R in rowsL}
    _s, det = elru.elru(pm, tm, lm, detail=True)
    return det[rowsL[0]["lang"]]["lang_score"]


def assemble_all(rows, row_proba, cand_map, op, drop_identity=False):
    """op = {lang: (th, gate)} -> id->edits."""
    out = {}
    for R in rows:
        th, g = op[R["lang"]]
        out[R["id"]] = assemble_gate(R, row_proba[R["id"]], cand_map[R["id"]], th, g, drop_identity)
    return out


def select_ops(rows, idfold, row_proba, cand_map, drop_identity=False):
    """Grid (th,gate) per language.  Returns nonnested op, nn edits, nested op-by-fold,
    nested edits.  Caches per-(lang,th,gate) lang edits to avoid recompute."""
    rbl = {L: [R for R in rows if R["lang"] == L] for L in LANGS}
    # cache: (lang,th,gate) -> {id: edits} for that language's rows
    ecache = {}

    def edits_for(L, th, g):
        key = (L, th, g)
        e = ecache.get(key)
        if e is None:
            e = {R["id"]: assemble_gate(R, row_proba[R["id"]], cand_map[R["id"]], th, g, drop_identity)
                 for R in rbl[L]}
            ecache[key] = e
        return e

    def score_subset(L, th, g, ids):
        e = edits_for(L, th, g)
        sub = [R for R in rbl[L] if R["id"] in ids]
        pm = {R["id"]: e[R["id"]] for R in sub}
        tm = {R["id"]: R["truth"] for R in sub}
        lm = {R["id"]: R["lang"] for R in sub}
        _s, det = elru.elru(pm, tm, lm, detail=True)
        return det[L]["lang_score"]

    allids = {L: set(R["id"] for R in rbl[L]) for L in LANGS}
    nonnested = {}
    for L in LANGS:
        best = (-1.0, THETA_GRID[0], 0.0)
        for th in THETA_GRID:
            for g in GATE_GRID:
                sc = score_subset(L, th, g, allids[L])
                if sc > best[0]:
                    best = (sc, th, g)
        nonnested[L] = (best[1], best[2])
    nn_edits = {}
    for R in rows:
        th, g = nonnested[R["lang"]]
        nn_edits[R["id"]] = edits_for(R["lang"], th, g)[R["id"]]

    nested_by_fold = {}
    nest_edits = {}
    for k in range(5):
        nested_by_fold[k] = {}
        for L in LANGS:
            other = set(R["id"] for R in rbl[L] if R["fold"] != k)
            best = (-1.0, THETA_GRID[0], 0.0)
            for th in THETA_GRID:
                for g in GATE_GRID:
                    sc = score_subset(L, th, g, other)
                    if sc > best[0]:
                        best = (sc, th, g)
            nested_by_fold[k][L] = (best[1], best[2])
            e = edits_for(L, best[1], best[2])
            for R in [r for r in rbl[L] if r["fold"] == k]:
                nest_edits[R["id"]] = e[R["id"]]
    return nonnested, nn_edits, nested_by_fold, nest_edits


# ======================================================================
#  Phase 5: group consistency + replacement-convention majority
# ======================================================================
def _covered(tk, edits):
    cov = [False] * len(tk)
    for i, (s, e, w) in enumerate(tk):
        for ed in edits:
            if s >= ed["start"] and e <= ed["end"]:
                cov[i] = True; break
    return cov


def group_consistency(assign_edits, rows_by_id, group_by_id, transducers, stores_by_fold, idfold,
                      hi=0.60, lo=0.40, do_vote=True, do_conv=True, vote_langs=None, conv_langs=None,
                      drop_langs=None):
    """Inference-time (leak-free) pass over document groups.  Votes identical surface
    forms across a group's windows: edit-propagation (>=hi) applies to vote_langs,
    drop (<lo) applies to drop_langs (defaults to vote_langs); per-group replacement-
    convention majority applies to conv_langs.  vote_langs=None -> all languages."""
    vl = set(LANGS) if vote_langs is None else set(vote_langs)
    dl = set(vl) if drop_langs is None else set(drop_langs)
    cl = set(LANGS) if conv_langs is None else set(conv_langs)
    groups = collections.defaultdict(list)
    for _id in assign_edits:
        groups[group_by_id[_id]].append(_id)
    out = {i: [dict(e) for e in assign_edits[i]] for i in assign_edits}
    for g, ids in groups.items():
        occ = collections.Counter(); cov = collections.Counter()
        for i in ids:
            R = rows_by_id[i]; tk = R["tk"]; lang = R["lang"]
            covf = _covered(tk, out[i])
            for j, (s, e, w) in enumerate(tk):
                core = w.strip(_STRIP).lower()
                if len(core) < 2:
                    continue
                occ[(lang, core)] += 1
                if covf[j]:
                    cov[(lang, core)] += 1
        vote = {}
        for key, o in occ.items():
            if o < 2:
                continue
            r = cov[key] / o
            if r >= hi:
                vote[key] = "edit"
            elif r < lo:
                vote[key] = "drop"
        if do_vote:
            for i in ids:
                R = rows_by_id[i]; tk = R["tk"]; lang = R["lang"]; text = R["text"]
                if lang not in vl and lang not in dl:
                    continue
                k = idfold[i]; T = transducers[k]; st = stores_by_fold[k]
                if lang in dl:
                    new = []
                    for ed in out[i]:
                        inside = [w for (s, e, w) in tk if s >= ed["start"] and e <= ed["end"]]
                        if len(inside) == 1:
                            core = inside[0].strip(_STRIP).lower()
                            if vote.get((lang, core)) == "drop":
                                continue
                        new.append(ed)
                    out[i] = new
                if lang not in vl:
                    out[i].sort(key=lambda ed: ed["start"])
                    continue
                covf = _covered(tk, out[i])
                occupied = [(ed["start"], ed["end"]) for ed in out[i]]
                for j, (s, e, w) in enumerate(tk):
                    if covf[j]:
                        continue
                    core = w.strip(_STRIP).lower()
                    if vote.get((lang, core)) != "edit":
                        continue
                    if any(not (e <= a or bb <= s) for (a, bb) in occupied):
                        continue
                    src = text[s:e]
                    ctx = {"text": text, "start": s, "end": e, "lang": lang, "tokens": tk, "stores": st}
                    rep, hn, mech = _transduce_full(T, lang, src, ctx, st)
                    if rep == src:
                        continue
                    out[i].append({"start": s, "end": e, "replacement": rep[:160]})
                    occupied.append((s, e))
                out[i].sort(key=lambda ed: ed["start"])
                if len(out[i]) > 8 or not elru.validate_edits(out[i], len(text)):
                    out[i] = pipeline._repair(out[i], len(text))
        if do_conv:
            repmaj = collections.defaultdict(collections.Counter)
            for i in ids:
                R = rows_by_id[i]; text = R["text"]; lang = R["lang"]
                for ed in out[i]:
                    src = text[ed["start"]:ed["end"]]
                    key = (lang, " ".join(src.split()).lower())
                    repmaj[key][ed["replacement"]] += 1
            maj = {}
            for key, c in repmaj.items():
                rep, nrep = c.most_common(1)[0]
                if sum(c.values()) >= 2 and nrep >= 2 and len(c) > 1:
                    maj[key] = rep
            if maj:
                for i in ids:
                    R = rows_by_id[i]; text = R["text"]; lang = R["lang"]
                    if lang not in cl:
                        continue
                    for ed in out[i]:
                        src = text[ed["start"]:ed["end"]]
                        key = (lang, " ".join(src.split()).lower())
                        if key in maj:
                            ed["replacement"] = maj[key][:160]
    return out


# ======================================================================
#  Diagnostics
# ======================================================================
def score_edits(rows, edits_map):
    pm = edits_map
    tm = {R["id"]: R["truth"] for R in rows}
    lm = {R["id"]: R["lang"] for R in rows}
    return elru.elru(pm, tm, lm, detail=True)


def per_type_recall(rows, edits_map):
    rec = collections.defaultdict(lambda: [0, 0])
    for R in rows:
        preds = edits_map[R["id"]]
        for (ts, te, rep) in R["spans"]:
            if rep == "":
                continue
            key = (R["lang"], span_type(R["lang"], R["text"][ts:te]))
            rec[key][1] += 1
            best = max((iou(ed["start"], ed["end"], ts, te) for ed in preds), default=0.0)
            if best >= 0.5:
                rec[key][0] += 1
    return {k: (v[0] / v[1] if v[1] else 0.0, v[1]) for k, v in rec.items()}


def fp_counts(rows, edits_map):
    out = collections.Counter(); tot = collections.Counter()
    for R in rows:
        if len(R["spans"]) == 0:
            tot[R["lang"]] += 1
            if edits_map[R["id"]]:
                out[R["lang"]] += 1
    return {L: (out[L], tot[L]) for L in LANGS}


def print_detail(tag, s, det):
    print(f"{tag} ELRU = {s:.4f}")
    for L in LANGS:
        d = det[L]
        em = d["edited_mean"]; um = d["unchanged_mean"]
        print(f"    {L}: lang={d['lang_score']:.4f} edited={em:.4f}(n={d['n_edited']}) unchanged={um:.4f}(n={d['n_unchanged']})")


LOSSMAP = {("de", "single_marked"): .565, ("de", "multi_plain"): .299, ("de", "multi_marked"): .204,
           ("de", "single_plain"): .092, ("it", "single_plain"): .378, ("it", "multi_plain"): .508,
           ("en", "single_marked"): .385}


# ======================================================================
#  Driver
# ======================================================================
def load_train():
    ROOT = pipeline.ROOT
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    return train, ROOT


def prepare(verbose=True):
    m4_ext.register(pipeline)
    train, ROOT = load_train()
    group_by_id = {r.id: r.document_group for r in train.itertuples()}
    rows, idfold, row_proba, transducers, stores_by_fold = fit_folds(train, verbose)
    rows_by_id = {R["id"]: R for R in rows}
    if verbose:
        print("[building candidates + cross-fit reranker...]", flush=True)
    cand_map = build_candidates(rows, idfold, row_proba, transducers, stores_by_fold, labeled=True)
    ncand = sum(len(v) for v in cand_map.values())
    npos = sum(c["label"] for v in cand_map.values() for c in v.values())
    crossfit_reranker(rows, idfold, cand_map)
    if verbose:
        print(f"[{ncand} candidates, {npos} positive]", flush=True)
    return dict(train=train, rows=rows, rows_by_id=rows_by_id, idfold=idfold,
                row_proba=row_proba, transducers=transducers, stores_by_fold=stores_by_fold,
                cand_map=cand_map, group_by_id=group_by_id)


# ----- base (M2+M3 threshold-merge) reference, leak-free, using pipeline.build_edits -----
def base_cache(rows, idfold, row_proba, transducers, stores_by_fold):
    cache = {}
    for R in rows:
        k = idfold[R["id"]]; T = transducers[k]; st = stores_by_fold[k]
        pr = row_proba[R["id"]]
        cache[R["id"]] = {thr: pipeline.build_edits(R["id"], R["text"], R["lang"], R["tk"], pr, thr, T, st)
                          for thr in pipeline.GRID}
    return cache


def _score_ids(rows, cache, thr_by_id, ids):
    sub = [R for R in rows if R["id"] in ids]
    pm = {R["id"]: cache[R["id"]][thr_by_id[R["id"]]] for R in sub}
    tm = {R["id"]: R["truth"] for R in sub}
    lm = {R["id"]: R["lang"] for R in sub}
    _s, det = elru.elru(pm, tm, lm, detail=True)
    return det


def base_select(rows, cache):
    rbl = {L: [R for R in rows if R["lang"] == L] for L in LANGS}
    nn = {}
    for L in LANGS:
        best = (-1.0, pipeline.GRID[0])
        for thr in pipeline.GRID:
            e = {R["id"]: cache[R["id"]][thr] for R in rbl[L]}
            _s, det = elru.elru(e, {R["id"]: R["truth"] for R in rbl[L]},
                                {R["id"]: R["lang"] for R in rbl[L]}, detail=True)
            if det[L]["lang_score"] > best[0]:
                best = (det[L]["lang_score"], thr)
        nn[L] = best[1]
    nn_edits = {R["id"]: cache[R["id"]][nn[R["lang"]]] for R in rows}
    nby = {}; nest_edits = {}
    for k in range(5):
        nby[k] = {}
        for L in LANGS:
            other = [R for R in rbl[L] if R["fold"] != k]
            best = (-1.0, pipeline.GRID[0])
            for thr in pipeline.GRID:
                e = {R["id"]: cache[R["id"]][thr] for R in other}
                _s, det = elru.elru(e, {R["id"]: R["truth"] for R in other},
                                    {R["id"]: R["lang"] for R in other}, detail=True)
                if det[L]["lang_score"] > best[0]:
                    best = (det[L]["lang_score"], thr)
            nby[k][L] = best[1]
            for R in [r for r in rbl[L] if r["fold"] == k]:
                nest_edits[R["id"]] = cache[R["id"]][best[1]]
    return nn, nn_edits, nby, nest_edits


def select_ops_fixed_th(rows, row_proba, cand_map, th_map, drop_identity=False):
    """1D: fix spine threshold per language (th_map), grid only the reranker gate."""
    rbl = {L: [R for R in rows if R["lang"] == L] for L in LANGS}
    ecache = {}

    def edits_for(L, g):
        e = ecache.get((L, g))
        if e is None:
            th = th_map[L]
            e = {R["id"]: assemble_gate(R, row_proba[R["id"]], cand_map[R["id"]], th, g, drop_identity)
                 for R in rbl[L]}
            ecache[(L, g)] = e
        return e

    def sc(L, g, ids):
        e = edits_for(L, g); sub = [R for R in rbl[L] if R["id"] in ids]
        _s, det = elru.elru({R["id"]: e[R["id"]] for R in sub}, {R["id"]: R["truth"] for R in sub},
                            {R["id"]: R["lang"] for R in sub}, detail=True)
        return det[L]["lang_score"]

    allids = {L: set(R["id"] for R in rbl[L]) for L in LANGS}
    nn = {}
    for L in LANGS:
        best = (-1.0, 0.0)
        for g in GATE_GRID:
            s = sc(L, g, allids[L])
            if s > best[0]:
                best = (s, g)
        nn[L] = (th_map[L], best[1])
    nn_edits = {R["id"]: edits_for(R["lang"], nn[R["lang"]][1])[R["id"]] for R in rows}
    nby = {}; nest_edits = {}
    for k in range(5):
        nby[k] = {}
        for L in LANGS:
            other = set(R["id"] for R in rbl[L] if R["fold"] != k)
            best = (-1.0, 0.0)
            for g in GATE_GRID:
                s = sc(L, g, other)
                if s > best[0]:
                    best = (s, g)
            nby[k][L] = (th_map[L], best[1])
            e = edits_for(L, best[1])
            for R in [r for r in rbl[L] if r["fold"] == k]:
                nest_edits[R["id"]] = e[R["id"]]
    return nn, nn_edits, nby, nest_edits


def ablate():
    P = prepare()
    rows = P["rows"]; idfold = P["idfold"]; row_proba = P["row_proba"]; cand_map = P["cand_map"]
    rows_by_id = P["rows_by_id"]; gbi = P["group_by_id"]; trs = P["transducers"]; stf = P["stores_by_fold"]
    results = {}

    def report(tag, nn_edits, nest_edits, show_types=False):
        nn_s, nn_d = score_edits(rows, nn_edits)
        ne_s, ne_d = score_edits(rows, nest_edits)
        results[tag] = (ne_s, nn_s)
        print(f"\n---- {tag} ----")
        print_detail("  NONNEST", nn_s, nn_d)
        print_detail("  NESTED ", ne_s, ne_d)
        fp = fp_counts(rows, nn_edits)
        print("  FP(nonnest): " + ", ".join(f"{L}={fp[L][0]}/{fp[L][1]}" for L in LANGS))
        if show_types:
            ptr = per_type_recall(rows, nn_edits)
            for key in sorted(ptr):
                r, nsp = ptr[key]; b = LOSSMAP.get(key)
                print(f"      {key[0]} {key[1]:14s} rec={r:.3f}(n={nsp})" + (f" lm{b:.3f}" if b else ""))
        return nn_edits, nest_edits

    # 1) BASE M2+M3 threshold-merge (champion reference)
    t0 = time.time()
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    b_nn, b_nn_e, b_nby, b_ne_e = base_select(rows, bcache)
    print(f"[base cache {time.time()-t0:.0f}s] thr={b_nn}")
    report("BASE M2+M3", b_nn_e, b_ne_e, show_types=True)

    # 2) BASE + group vote, + conv, + both
    for dv, dc, nm in [(True, False, "BASE+group"), (False, True, "BASE+conv"), (True, True, "BASE+group+conv")]:
        gnn = group_consistency({i: b_nn_e[i] for i in b_nn_e}, rows_by_id, gbi, trs, stf, idfold, do_vote=dv, do_conv=dc)
        gne = group_consistency({i: b_ne_e[i] for i in b_ne_e}, rows_by_id, gbi, trs, stf, idfold, do_vote=dv, do_conv=dc)
        report(nm, gnn, gne)

    # 3) RERANKER gate 2D (th,gate)
    r_nn, r_nn_e, r_nby, r_ne_e = select_ops(rows, idfold, row_proba, cand_map)
    report("RERANK gate2D", r_nn_e, r_ne_e, show_types=True)

    # 4) RERANKER gate 1D (fix th at base nonnested optimum, tune gate only)
    f_nn, f_nn_e, f_nby, f_ne_e = select_ops_fixed_th(rows, row_proba, cand_map, b_nn)
    print(f"  [gate1D fixed th={b_nn}] ops={f_nn}")
    report("RERANK gate1D", f_nn_e, f_ne_e, show_types=True)

    # 5) RERANK gate1D + group
    gnn = group_consistency({i: f_nn_e[i] for i in f_nn_e}, rows_by_id, gbi, trs, stf, idfold, do_vote=True, do_conv=True)
    gne = group_consistency({i: f_ne_e[i] for i in f_ne_e}, rows_by_id, gbi, trs, stf, idfold, do_vote=True, do_conv=True)
    report("RERANK gate1D+group+conv", gnn, gne)

    print("\n================ SUMMARY (nested, nonnested) ================")
    for tag, (ne, nn) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {tag:28s} nested={ne:.4f}  nonnested={nn:.4f}")
    return results


def ablate2(P=None):
    if P is None:
        P = prepare()
    rows = P["rows"]; idfold = P["idfold"]; row_proba = P["row_proba"]; cand_map = P["cand_map"]
    rows_by_id = P["rows_by_id"]; gbi = P["group_by_id"]; trs = P["transducers"]; stf = P["stores_by_fold"]
    results = {}
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    b_nn, b_nn_e, b_nby, b_ne_e = base_select(rows, bcache)

    def rep(tag, nn_e, ne_e, verbose=False):
        nn_s, nn_d = score_edits(rows, nn_e); ne_s, ne_d = score_edits(rows, ne_e)
        results[tag] = (ne_s, nn_s)
        pl = " ".join(f"{L}={ne_d[L]['lang_score']:.4f}" for L in LANGS)
        print(f"  {tag:26s} nested={ne_s:.4f} nonnest={nn_s:.4f}   [{pl}]")
        if verbose:
            fp = fp_counts(rows, nn_e)
            print("      FP(nonnest): " + ", ".join(f"{L}={fp[L][0]}/{fp[L][1]}" for L in LANGS))

    print(f"\n================ M4 GROUP-LANGUAGE ABLATION (thr={b_nn}) ================")
    rep("BASE M2+M3", b_nn_e, b_ne_e, verbose=True)

    def grp(nn_e, ne_e, el, dl, conv=False, cl=None):
        gnn = group_consistency({i: nn_e[i] for i in nn_e}, rows_by_id, gbi, trs, stf, idfold,
                                vote_langs=el, drop_langs=dl, do_conv=conv, conv_langs=cl)
        gne = group_consistency({i: ne_e[i] for i in ne_e}, rows_by_id, gbi, trs, stf, idfold,
                                vote_langs=el, drop_langs=dl, do_conv=conv, conv_langs=cl)
        return gnn, gne

    variants = [
        ({"de", "en", "it"}, {"de", "en", "it"}, "grp[all]"),
        ({"de", "en"}, {"de", "en"}, "grp[de,en]"),
        ({"de", "en"}, {"de", "en", "it"}, "grp[de,en]+itDROP"),
        ({"en"}, {"en"}, "grp[en]"),
        ({"de"}, {"de"}, "grp[de]"),
        ({"de", "en"}, {"de", "en"}, "grp[de,en]+conv[it]"),  # conv on it slash-order
    ]
    winner = None
    for el, dl, tag in variants:
        conv = tag.endswith("conv[it]")
        gnn, gne = grp(b_nn_e, b_ne_e, el, dl, conv=conv, cl={"it"} if conv else None)
        rep(tag, gnn, gne, verbose=(tag == "grp[de,en]"))
        if tag == "grp[de,en]":
            winner = (gnn, gne)

    # reranker gate1D (de gated) + group[de,en]
    f_nn, f_nn_e, f_nby, f_ne_e = select_ops_fixed_th(rows, row_proba, cand_map, b_nn)
    rep("rerankGate1D", f_nn_e, f_ne_e)
    gnn, gne = grp(f_nn_e, f_ne_e, {"de", "en"}, {"de", "en"})
    rep("rerankGate1D+grp[de,en]", gnn, gne)

    print("\n---- SUMMARY (by nested) ----")
    for tag, (ne, nn) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {tag:28s} nested={ne:.4f}  nonnested={nn:.4f}")
    if winner:
        print("\n---- winner grp[de,en] nested per-language detail ----")
        _s, det = score_edits(rows, winner[1])
        print_detail("NESTED", _s, det)
        ptr = per_type_recall(rows, winner[0])
        for key in sorted(ptr):
            r, nsp = ptr[key]; b = LOSSMAP.get(key)
            print(f"    {key[0]} {key[1]:14s} rec={r:.3f}(n={nsp})" + (f" lm{b:.3f}" if b else ""))
    return results


SHIP_VOTE_LANGS = {"de", "en"}   # measured: it edit-propagation regresses (context-driven)
TRAIN_EDIT_RATE = {"de": 0.577, "en": 0.470, "it": 0.704}


def group_pass(edits_map, rows_by_id, group_by_id, transducer, stores):
    """Apply the shipped group vote (de+en, hi.60/lo.40) with a single transducer/stores
    (used for the full-train test submission)."""
    idf = {i: 0 for i in edits_map}
    return group_consistency(edits_map, rows_by_id, group_by_id, {0: transducer}, {0: stores}, idf,
                             vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)


def ship():
    t0 = time.time()
    P = prepare()
    rows = P["rows"]; idfold = P["idfold"]; row_proba = P["row_proba"]
    rows_by_id = P["rows_by_id"]; gbi = P["group_by_id"]; trs = P["transducers"]; stf = P["stores_by_fold"]
    train = P["train"]

    # ---- OOF operating points (base threshold merge, leak-free) ----
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    nn_thr, nn_e, nby, ne_e = base_select(rows, bcache)
    # ---- shipped group-consistency (de+en) on OOF ----
    nn_ship = group_consistency({i: nn_e[i] for i in nn_e}, rows_by_id, gbi, trs, stf, idfold,
                                vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    ne_ship = group_consistency({i: ne_e[i] for i in ne_e}, rows_by_id, gbi, trs, stf, idfold,
                                vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    nn_s, nn_d = score_edits(rows, nn_ship)
    ne_s, ne_d = score_edits(rows, ne_ship)

    print("\n================ M4 SHIP CONFIG: M2+M3 + group-vote[de,en] ================")
    print_detail("NON-NESTED (all-OOF op, = submission op)", nn_s, nn_d)
    print(f"  nonnested thresholds = {nn_thr}")
    print_detail("NESTED (honest headline)", ne_s, ne_d)
    fp = fp_counts(rows, nn_ship)
    print("unchanged-row FPs (nonnested): " + ", ".join(f"{L}={fp[L][0]}/{fp[L][1]}" for L in LANGS))
    ptr = per_type_recall(rows, nn_ship)
    print("per-type IoU>=.5 recall (nonnested) vs iter-1 loss map:")
    for key in sorted(ptr):
        r, nsp = ptr[key]; b = LOSSMAP.get(key)
        print(f"    {key[0]} {key[1]:14s} rec={r:.3f}(n={nsp})" + (f"  lm{b:.3f}" if b else ""))

    # ---- self-check with canonical scorer against train truth (OOF) ----
    oof_df = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(nn_ship[R["id"]], ensure_ascii=False)} for R in rows])
    true_df = train[["id", "language", "edits_json"]]
    chk, _ = elru.score_frames(oof_df, true_df)
    print(f"canonical elru.score_frames (OOF, nonnested+group) = {chk:.4f}")

    # ---- FULL-TRAIN fit -> test submission ----
    test = pd.read_csv(os.path.join(pipeline.ROOT, "dataset", "test.csv"))
    gbi_te = {r.id: r.document_group for r in test.itertuples()}
    stores_full = {}
    for b in pipeline.STORE_BUILDERS:
        b(train, stores_full)
    all_rows = pipeline.build_rows(train, labeled=True)
    det_full = pipeline.Detector().fit(all_rows, stores_full)
    trd_full = pipeline.Transducer().fit(train)
    test_rows = pipeline.build_rows(test, labeled=False)
    tp_test = det_full.token_probs(test_rows)
    sub = {}
    for R in test_rows:
        tk, pr = tp_test[R["id"]]
        sub[R["id"]] = pipeline.build_edits(R["id"], R["text"], R["lang"], tk, pr, nn_thr[R["lang"]], trd_full, stores_full)
    test_by_id = {R["id"]: R for R in test_rows}
    sub = group_pass(sub, test_by_id, gbi_te, trd_full, stores_full)

    # ---- strict validation ----
    assert len(sub) == len(test) == 445, (len(sub), len(test))
    assert set(sub) == set(test.id), "id mismatch"
    tl = {r.id: len(r.text) for r in test.itertuples()}
    bad = [i for i in sub if not elru.validate_edits(sub[i], tl[i])]
    assert not bad, f"invalid rows: {bad[:5]}"
    lang_te = {r.id: r.language for r in test.itertuples()}
    edn = collections.Counter(); totn = collections.Counter()
    for i in sub:
        L = lang_te[i]; totn[L] += 1
        if sub[i]:
            edn[L] += 1
    print("\nsubmission edited-row fractions vs train rates:")
    flags = []
    for L in LANGS:
        r_sub = edn[L] / max(totn[L], 1); r_tr = TRAIN_EDIT_RATE[L]; ratio = r_sub / r_tr
        flag = "" if 0.45 <= ratio <= 1.80 else "  <<< FLAG"
        if flag:
            flags.append(L)
        print(f"  {L}: sub={r_sub:.3f} ({edn[L]}/{totn[L]})  train={r_tr:.3f}  ratio={ratio:.2f}{flag}")

    # ---- deliverables ----
    pd.DataFrame([{"id": i, "edits_json": json.dumps(sub[i], ensure_ascii=False)} for i in test.id]
                 ).to_csv(os.path.join(HERE, "submission_v2.csv"), index=False)
    oof_df.to_csv(os.path.join(HERE, "oof_edits.csv"), index=False)
    tokrows = []
    for R in rows:
        pr = row_proba[R["id"]]
        for ti, (s, e, w) in enumerate(R["tk"]):
            tokrows.append(dict(id=R["id"], lang=R["lang"], tok_index=ti, start=s, end=e,
                                proba=round(pr[ti], 5), y=R["y"][ti]))
    pd.DataFrame(tokrows).to_csv(os.path.join(HERE, "oof_token_probs.csv"), index=False)
    report = dict(
        ship_config="M1 detector + A2 transducer + M2(de gen/collapse) + M3(it/en feats+hooks) + group-vote[de,en]",
        nested_elru=round(ne_s, 4), nonnested_elru=round(nn_s, 4),
        canonical_oof_check=round(chk, 4), nonnested_thr={L: float(nn_thr[L]) for L in LANGS},
        nested_detail={L: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ne_d[L].items()} for L in LANGS},
        nonnested_detail={L: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in nn_d[L].items()} for L in LANGS},
        unchanged_fp={L: list(fp[L]) for L in LANGS},
        submission_edit_rate={L: round(edn[L] / max(totn[L], 1), 3) for L in LANGS},
        train_edit_rate=TRAIN_EDIT_RATE, edit_rate_flags=flags,
        submission_rows=len(sub), submission_edited=sum(1 for i in sub if sub[i]),
        ai_baseline=0.56, beats_ai_baseline=bool(ne_s > 0.56))
    json.dump(report, open(os.path.join(HERE, "cv_report.json"), "w"), indent=2)
    print(f"\nwrote submission_v2.csv ({sum(1 for i in sub if sub[i])}/445 edited), oof_edits.csv, oof_token_probs.csv, cv_report.json")
    print(f"HEADLINE nested ELRU = {ne_s:.4f}   (AI baseline 0.56: {'BEAT' if ne_s>0.56 else 'below'})   [{time.time()-t0:.0f}s]")
    return report


def run_report(P=None, drop_identity=False, do_group=False, do_conv=False, label="rerank-gate"):
    t0 = time.time()
    if P is None:
        P = prepare()
    rows = P["rows"]; idfold = P["idfold"]; row_proba = P["row_proba"]; cand_map = P["cand_map"]
    nonnested, nn_edits, nested_by_fold, nest_edits = select_ops(rows, idfold, row_proba, cand_map, drop_identity)
    tag = label
    if do_group or do_conv:
        nn_edits = group_consistency(nn_edits, P["rows_by_id"], P["group_by_id"], P["transducers"],
                                     P["stores_by_fold"], idfold, do_vote=do_group, do_conv=do_conv)
        nest_edits = group_consistency(nest_edits, P["rows_by_id"], P["group_by_id"], P["transducers"],
                                       P["stores_by_fold"], idfold, do_vote=do_group, do_conv=do_conv)
        tag += ("+group" if do_group else "") + ("+conv" if do_conv else "")
    nn_s, nn_d = score_edits(rows, nn_edits)
    ne_s, ne_d = score_edits(rows, nest_edits)
    print(f"\n================ M4 [{tag}] drop_identity={drop_identity} ================")
    print_detail("NON-NESTED", nn_s, nn_d)
    print(f"  nonnested op (th,gate) = { {L: nonnested[L] for L in LANGS} }")
    print_detail("NESTED    ", ne_s, ne_d)
    print(f"  nested op by fold = {nested_by_fold}")
    fp = fp_counts(rows, nn_edits)
    print("unchanged-row FPs (nonnested): " + ", ".join(f"{L}={fp[L][0]}/{fp[L][1]}" for L in LANGS))
    ptr = per_type_recall(rows, nn_edits)
    print("per-type IoU>=.5 recall (nonnested) vs iter-1 loss map:")
    for key in sorted(ptr):
        r, nsp = ptr[key]
        base = LOSSMAP.get(key)
        bs = f"  (loss-map {base:.3f})" if base is not None else ""
        print(f"    {key[0]} {key[1]:14s} recall={r:.3f} (n={nsp}){bs}")
    print(f"[{time.time()-t0:.0f}s]  overall nested={ne_s:.4f} nonnested={nn_s:.4f}")
    return dict(nonnested_elru=nn_s, nested_elru=ne_s, nonnested=nonnested, nn_detail=nn_d,
                ne_detail=ne_d, nested_by_fold=nested_by_fold, nn_edits=nn_edits, nest_edits=nest_edits)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "report":
        run_report()
    elif mode == "ablate":
        ablate()
    elif mode == "ablate2":
        ablate2()
    elif mode == "ship":
        ship()

"""N1 exp5: strongest NP-capture. it-specific article-anchored NP candidate generator +
per-candidate learned gate (cross-fit, generator-local features) INDEPENDENT of token
threshold, union with base merge spans, whole-NP A2 transduction, joint (thr,gate) select.
Decisive test of the multi_plain-capture lever (oracle +.128). Reports net + precision.
"""
import os, sys, json, time, collections, re, pickle
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline, m4_ext
from transducer import Transducer
from run_m4 import base_cache, base_select, group_consistency, score_edits, fp_counts, SHIP_VOTE_LANGS
import elru

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")
WS = re.compile(r"\S+")


def rebuild(train):
    m4_ext.register(pipeline)
    trs, stf = {}, {}
    for k in range(5):
        trdf = train[train.fold != k]
        trs[k] = Transducer().fit(trdf)
        st = {}
        for b in pipeline.STORE_BUILDERS:
            b(trdf, st)
        stf[k] = st
    return trs, stf


def learn_np_tables(trdf):
    """per-fold it tables for NP anchoring/features."""
    occ = collections.Counter(); spaninit = collections.Counter(); spaninit_slash = collections.Counter()
    end2_ed = collections.Counter(); end2_tot = collections.Counter()
    tok_ed = collections.Counter()
    for r in trdf[trdf.language == "it"].itertuples():
        tk = [(m.start(), m.end(), m.group()) for m in WS.finditer(r.text)]
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in r.edits)
        startset = {a for a, _, _ in spans}
        rep_first_slash = {}
        for a, b, rep in spans:
            fw = rep.split()[0] if rep.split() else ""
            rep_first_slash[a] = ("/" in fw)

        def inside(s, e):
            return any(s >= a and e <= b for a, b, _ in spans)
        for i, (s, e, w) in enumerate(tk):
            core = w.strip(_STRIP).lower()
            if not core:
                continue
            occ[core] += 1
            isin = inside(s, e)
            if isin:
                tok_ed[core] += 1
            if s in startset:
                spaninit[core] += 1
                if rep_first_slash.get(s):
                    spaninit_slash[core] += 1
            if len(core) >= 2:
                end2_tot[core[-2:]] += 1
                if isin:
                    end2_ed[core[-2:]] += 1
    spaninit_rate = {w: spaninit[w] / occ[w] for w in occ}
    tok_edrate = {w: tok_ed[w] / occ[w] for w in occ}
    end2_rate = {k: end2_ed[k] / end2_tot[k] for k in end2_tot if end2_tot[k] >= 5}
    # anchors: tokens that start slash-NPs with decent frequency
    anchors = {w for w in occ if occ[w] >= 2 and spaninit.get(w, 0) >= 1
               and spaninit_slash.get(w, 0) / max(1, spaninit.get(w, 0)) >= 0.3}
    return dict(spaninit_rate=spaninit_rate, tok_edrate=tok_edrate, end2_rate=end2_rate,
                anchors=anchors, occ=occ)


NPFEATS = ["art_spaninit", "art_edr", "art_len", "L", "np_len_char",
           "mean_p", "max_p", "min_p", "art_p", "last_p",
           "foll_max_end2", "foll_mean_end2", "foll_max_edr", "n_masc_end",
           "has_slash_in", "n_cap"]


def np_candidates(R, probs, T, tab):
    """article-anchored NP spans (article + 1..3 following); featurize (generator-local)."""
    tk = R["tk"]; text = R["text"]; n = len(tk)
    cands = []
    for i in range(n):
        acore = tk[i][2].strip(_STRIP).lower()
        if acore not in tab["anchors"]:
            continue
        for L in (1, 2, 3):
            j = i + L
            if j >= n:
                break
            a, b = tk[i][0], tk[j][1]
            src = text[a:b]
            if any(c in MARKS for c in src):
                continue
            sp = probs[i:j + 1]
            foll_end2 = []; foll_edr = []; nmasc = 0; ncap = 0
            for kk in range(i + 1, j + 1):
                c = tk[kk][2].strip(_STRIP).lower()
                if len(c) >= 2:
                    foll_end2.append(tab["end2_rate"].get(c[-2:], 0.0))
                    if c[-1:] in ("g", "y", "o", "e", "i"):
                        nmasc += 1
                foll_edr.append(tab["tok_edrate"].get(c, 0.0))
                if tk[kk][2][:1].isupper():
                    ncap += 1
            f = [tab["spaninit_rate"].get(acore, 0.0), tab["tok_edrate"].get(acore, 0.0),
                 float(len(acore)), float(L), float(b - a),
                 float(np.mean(sp)), float(np.max(sp)), float(np.min(sp)), float(sp[0]), float(sp[-1]),
                 max(foll_end2) if foll_end2 else 0.0, float(np.mean(foll_end2)) if foll_end2 else 0.0,
                 max(foll_edr) if foll_edr else 0.0, float(nmasc),
                 1.0 if "/" in src else 0.0, float(ncap)]
            cands.append((i, j, a, b, f))
    return cands


def main():
    import lightgbm as lgb
    t0 = time.time()
    S = pickle.load(open(os.path.join(ROOT, "runs", "N1_state.pkl"), "rb"))
    rows = S["rows"]; idfold = S["idfold"]; row_proba = S["row_proba"]; gbi = S["group_by_id"]
    rows_by_id = {R["id"]: R for R in rows}
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    trs, stf = rebuild(train)
    tabs = {k: learn_np_tables(train[train.fold != k]) for k in range(5)}
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    print(f"[rebuild {time.time()-t0:.0f}s]")

    itrows = [R for R in rows if R["lang"] == "it"]

    # build NP candidates + labels for all it rows
    cand_by_id = {}
    for R in itrows:
        k = idfold[R["id"]]; T = trs[k]; tab = tabs[k]; pr = row_proba[R["id"]]
        cands = np_candidates(R, pr, T, tab)
        # label: IoU>=0.5 with any non-empty true edit
        lab = []
        for (i, j, a, b, f) in cands:
            best = 0.0
            for (ts, te, rep) in R["spans"]:
                if rep == "":
                    continue
                ov = max(0, min(b, te) - max(a, ts)); un = max(b, te) - min(a, ts)
                best = max(best, ov / un if un > 0 else 0.0)
            lab.append(1 if best >= 0.5 else 0)
        cand_by_id[R["id"]] = (cands, lab)
    ncand = sum(len(v[0]) for v in cand_by_id.values())
    npos = sum(sum(v[1]) for v in cand_by_id.values())
    print(f"NP candidates={ncand} positive(IoU>=.5)={npos} precision-ceiling={npos/max(1,ncand):.1%}")

    # cross-fit gate
    rr = {}
    for k in range(5):
        Xtr, ytr = [], []
        for R in itrows:
            if idfold[R["id"]] == k:
                continue
            rec = cand_by_id[R["id"]]; cs, lb = rec[0], rec[1]
            for (c, y) in zip(cs, lb):
                Xtr.append(c[4]); ytr.append(y)
        if not Xtr:
            continue
        m = lgb.LGBMClassifier(objective="binary", n_estimators=250, learning_rate=0.04,
                               num_leaves=24, min_child_samples=20, subsample=0.85,
                               colsample_bytree=0.8, reg_lambda=3.0, is_unbalance=True,
                               random_state=0, n_jobs=7, verbosity=-1)
        m.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, np.int32))
        for R in itrows:
            if idfold[R["id"]] != k:
                continue
            cs, lb = cand_by_id[R["id"]]
            if cs:
                pv = m.predict_proba(np.asarray([c[4] for c in cs], np.float32))[:, 1]
                cand_by_id[R["id"]] = (cs, lb, pv.tolist())
    # ensure 3-tuple
    for i in list(cand_by_id):
        rec = cand_by_id[i]
        if len(rec) == 2:
            cs, lb = rec
            cand_by_id[i] = (cs, lb, [0.0] * len(cs))

    # assemble: base merge spans (it) UNION NP cands with gate>=g, non-overlap prefer higher score
    def it_edits(R, thr, g):
        tk = R["tk"]; text = R["text"]; pr = row_proba[R["id"]]; n = len(tk)
        k = idfold[R["id"]]; T = trs[k]; st = stf[k]
        spans = []
        i = 0
        while i < n:
            if pr[i] >= thr:
                j = i
                while j + 1 < n and pr[j + 1] >= thr:
                    j += 1
                spans.append((tk[i][0], tk[j][1], float(np.mean(pr[i:j + 1])) + 1.0))
                i = j + 1
            else:
                i += 1
        cs, lb, pv = cand_by_id[R["id"]]
        for (c, p) in zip(cs, pv):
            if p >= g:
                spans.append((c[2], c[3], float(p)))
        spans.sort(key=lambda s: -s[2])
        chosen = []; occ = []
        for (a, b, sc) in spans:
            if any(not (b <= x or y <= a) for (x, y) in occ):
                continue
            chosen.append((a, b, sc)); occ.append((a, b))
        edits = []
        for (a, b, sc) in chosen:
            src = text[a:b]
            ctx = {"text": text, "start": a, "end": b, "lang": "it", "tokens": tk, "stores": st}
            rep = None
            for hook in pipeline.REPLACEMENT_HOOKS:
                r = hook("it", src, ctx, st)
                if r is not None:
                    rep = r; break
            if rep is None:
                rep = T.predict("it", src, ctx) or src
            edits.append((sc, {"start": a, "end": b, "replacement": rep[:160]}))
        edits.sort(key=lambda e: -e[0]); edits = [e for _, e in edits[:8]]
        edits.sort(key=lambda e: e["start"])
        if not elru.validate_edits(edits, len(text)):
            edits = pipeline._repair(edits, len(text))
        return edits

    THR = [round(x, 2) for x in np.arange(0.35, 0.62, 0.02)]
    GATES = [1.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]  # 1.01 = NP off
    truth = {R["id"]: R["truth"] for R in itrows}; lm = {R["id"]: "it" for R in itrows}
    # non-nested joint search
    cache = {}
    def edits_for(thr, g):
        key = (thr, g)
        if key not in cache:
            cache[key] = {R["id"]: it_edits(R, thr, g) for R in itrows}
        return cache[key]
    best = (-1, None, None)
    for thr in THR:
        for g in GATES:
            e = edits_for(thr, g)
            _s, d = elru.elru(e, truth, lm, detail=True)
            if d["it"]["lang_score"] > best[0]:
                best = (d["it"]["lang_score"], thr, g)
    bl, bthr, bg = best
    e = edits_for(bthr, bg)
    _s, d = elru.elru(e, truth, lm, detail=True)
    fp = sum(1 for R in itrows if len(R["truth"]) == 0 and e[R["id"]])
    print(f"BEST NN it lang={bl:.4f} (e{d['it']['edited_mean']:.3f}/u{d['it']['unchanged_mean']:.3f}) "
          f"thr={bthr} gate={bg} FP={fp}  (base it NN=0.4133 FP56)")
    # nested
    nby = {}; nest = {}
    for k in range(5):
        b2 = (-1, None, None)
        other = [R for R in itrows if R["fold"] != k]
        oids = set(R["id"] for R in other)
        for thr in THR:
            for g in GATES:
                e = edits_for(thr, g)
                sub = {i: e[i] for i in oids}
                _s, d = elru.elru(sub, {i: truth[i] for i in oids}, {i: "it" for i in oids}, detail=True)
                if d["it"]["lang_score"] > b2[0]:
                    b2 = (d["it"]["lang_score"], thr, g)
        nby[k] = (b2[1], b2[2])
        e = edits_for(b2[1], b2[2])
        for R in [r for r in itrows if r["fold"] == k]:
            nest[R["id"]] = e[R["id"]]
    _s, dn = elru.elru(nest, truth, lm, detail=True)
    print(f"NESTED it lang={dn['it']['lang_score']:.4f} (e{dn['it']['edited_mean']:.3f}/u{dn['it']['unchanged_mean']:.3f}) nby={nby}")
    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

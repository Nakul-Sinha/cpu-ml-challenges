"""N1 exp7: robust NP-only gated capture. Fix spine thr=0.45 (stable base optimum),
tune ONLY the gate nested (low variance). NP-only candidates + group-context features +
reorder. Test base-priority (safe) vs NP-override. Pick robust config.
"""
import os, sys, json, time, collections, re, pickle
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline, m4_ext
from transducer import Transducer
from run_m4 import base_cache, base_select, group_consistency, score_edits, fp_counts, per_type_recall, SHIP_VOTE_LANGS
import elru

_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")
WS = re.compile(r"\S+")
_SLASH = re.compile(r"[^\W\d_]/[^\W\d_]", re.UNICODE)


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


def learn_tab(trdf):
    occ = collections.Counter(); spaninit = collections.Counter(); spaninit_slash = collections.Counter()
    end2_ed = collections.Counter(); end2_tot = collections.Counter()
    end3_ed = collections.Counter(); end3_tot = collections.Counter(); tok_ed = collections.Counter()
    for r in trdf[trdf.language == "it"].itertuples():
        tk = [(m.start(), m.end(), m.group()) for m in WS.finditer(r.text)]
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in r.edits)
        startset = {a for a, _, _ in spans}
        rfs = {a: ("/" in (rep.split()[0] if rep.split() else "")) for a, b, rep in spans}

        def inside(s, e):
            return any(s >= a and e <= b for a, b, _ in spans)
        for i, (s, e, w) in enumerate(tk):
            core = w.strip(_STRIP).lower()
            if not core:
                continue
            occ[core] += 1; isin = inside(s, e)
            if isin:
                tok_ed[core] += 1
            if s in startset:
                spaninit[core] += 1
                if rfs.get(s):
                    spaninit_slash[core] += 1
            if len(core) >= 2:
                end2_tot[core[-2:]] += 1
                if isin:
                    end2_ed[core[-2:]] += 1
            if len(core) >= 3:
                end3_tot[core[-3:]] += 1
                if isin:
                    end3_ed[core[-3:]] += 1
    return dict(spaninit_rate={w: spaninit[w] / occ[w] for w in occ},
                tok_edrate={w: tok_ed[w] / occ[w] for w in occ},
                end2_rate={k: end2_ed[k] / end2_tot[k] for k in end2_tot if end2_tot[k] >= 5},
                end3_rate={k: end3_ed[k] / end3_tot[k] for k in end3_tot if end3_tot[k] >= 5},
                anchors={w for w in occ if occ[w] >= 2 and spaninit.get(w, 0) >= 1
                         and spaninit_slash.get(w, 0) / max(1, spaninit.get(w, 0)) >= 0.3})


def group_ctx(train):
    g = collections.defaultdict(lambda: [0, 0, 0])
    for r in train.itertuples():
        if r.language != "it":
            continue
        g[r.document_group][0] += len(r.text.split())
        g[r.document_group][1] += len(_SLASH.findall(r.text))
        g[r.document_group][2] += 1
    return {gg: (v[1] / max(1, v[0]), float(v[2])) for gg, v in g.items()}


def np_cands(R, pr, tab, gc):
    tk = R["tk"]; text = R["text"]; n = len(tk)
    gs, gsz = gc.get(R["document_group"], (0.0, 0.0))
    out = []
    for i in range(n):
        core = tk[i][2].strip(_STRIP).lower()
        if core not in tab["anchors"]:
            continue
        for L in (1, 2, 3):
            j = i + L
            if j >= n:
                break
            a, b = tk[i][0], tk[j][1]
            if any(c in MARKS for c in text[a:b]):
                continue
            sp = pr[i:j + 1]; fe2 = []; fedr = []; nmasc = 0
            for kk in range(i + 1, j + 1):
                c = tk[kk][2].strip(_STRIP).lower()
                if len(c) >= 2:
                    fe2.append(tab["end2_rate"].get(c[-2:], 0.0))
                    if c[-1:] in ("g", "y", "o", "e", "i"):
                        nmasc += 1
                fedr.append(tab["tok_edrate"].get(c, 0.0))
            f = [tab["spaninit_rate"].get(core, 0.0), tab["tok_edrate"].get(core, 0.0), float(len(core)),
                 float(L), float(b - a), float(np.mean(sp)), float(np.max(sp)), float(np.min(sp)),
                 float(sp[0]), float(sp[-1]), max(fe2) if fe2 else 0.0,
                 float(np.mean(fe2)) if fe2 else 0.0, max(fedr) if fedr else 0.0, float(nmasc),
                 1.0 if "/" in text[a:b] else 0.0, gs, gsz]
            out.append((a, b, f))
    return out


def reorder(src, rep):
    if rep.count("/") == 1 and " " not in rep and "/" in rep:
        core = src.strip(_STRIP); x, y = rep.split("/")
        if core == y and core != x:
            return core + "/" + x
    return rep


def main():
    import lightgbm as lgb
    t0 = time.time()
    S = pickle.load(open(os.path.join(ROOT, "runs", "N1_state.pkl"), "rb"))
    rows = S["rows"]; idfold = S["idfold"]; row_proba = S["row_proba"]; gbi = S["group_by_id"]
    for R in rows:
        R["document_group"] = gbi[R["id"]]
    rows_by_id = {R["id"]: R for R in rows}
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    trs, stf = rebuild(train)
    tabs = {k: learn_tab(train[train.fold != k]) for k in range(5)}
    gctxs = {k: group_ctx(train[train.fold != k]) for k in range(5)}
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    itrows = [R for R in rows if R["lang"] == "it"]

    cand = {}
    for R in itrows:
        k = idfold[R["id"]]; cs = np_cands(R, row_proba[R["id"]], tabs[k], gctxs[k])
        lab = []
        for (a, b, f) in cs:
            best = max((max(0, min(b, te) - max(a, ts)) / (max(b, te) - min(a, ts))
                        for (ts, te, rep) in R["spans"] if rep != "" and max(b, te) > min(a, ts)), default=0.0)
            lab.append(1 if best >= 0.5 else 0)
        cand[R["id"]] = [cs, lab, [0.0] * len(cs)]
    for k in range(5):
        Xtr, ytr = [], []
        for R in itrows:
            if idfold[R["id"]] == k:
                continue
            cs, lb, _ = cand[R["id"]]; Xtr += [c[2] for c in cs]; ytr += lb
        m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.04, num_leaves=20,
                               min_child_samples=25, subsample=0.85, colsample_bytree=0.8, reg_lambda=3.0,
                               is_unbalance=True, random_state=0, n_jobs=7, verbosity=-1)
        m.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, np.int32))
        for R in itrows:
            if idfold[R["id"]] != k:
                continue
            cs, lb, _ = cand[R["id"]]
            if cs:
                cand[R["id"]][2] = m.predict_proba(np.asarray([c[2] for c in cs], np.float32))[:, 1].tolist()

    def it_edits(R, g, override=False):
        tk = R["tk"]; text = R["text"]; pr = row_proba[R["id"]]; n = len(tk)
        k = idfold[R["id"]]; T = trs[k]; st = stf[k]; thr = 0.45
        spans = []; i = 0
        while i < n:
            if pr[i] >= thr:
                j = i
                while j + 1 < n and pr[j + 1] >= thr:
                    j += 1
                base_pri = (0.0 if override else 1.0)
                spans.append((tk[i][0], tk[j][1], float(np.mean(pr[i:j + 1])) + base_pri, "base"))
                i = j + 1
            else:
                i += 1
        cs, lb, pv = cand[R["id"]]
        for (c, p) in zip(cs, pv):
            if p >= g:
                spans.append((c[0], c[1], float(p) + (0.5 if override else 0.0), "np"))
        spans.sort(key=lambda s: -s[2]); chosen = []; occ = []
        for (a, b, sc, kind) in spans:
            if any(not (b <= x or y <= a) for (x, y) in occ):
                continue
            chosen.append((a, b)); occ.append((a, b))
        edits = []
        for (a, b) in chosen:
            src = text[a:b]
            ctx = {"text": text, "start": a, "end": b, "lang": "it", "tokens": tk, "stores": st}
            rep = None
            for hook in pipeline.REPLACEMENT_HOOKS:
                r = hook("it", src, ctx, st)
                if r is not None:
                    rep = r; break
            if rep is None:
                rep = T.predict("it", src, ctx) or src
            if len(src.split()) == 1:
                rep = reorder(src, rep)
            edits.append({"start": a, "end": b, "replacement": rep[:160]})
        edits.sort(key=lambda e: e["start"]); edits = edits[:8]
        if not elru.validate_edits(edits, len(text)):
            edits = pipeline._repair(edits, len(text))
        return edits

    truth = {R["id"]: R["truth"] for R in itrows}; lm = {R["id"]: "it" for R in itrows}
    GATES = [1.01, 0.8, 0.7, 0.6, 0.5, 0.4]
    for override in [False, True]:
        cache = {g: {R["id"]: it_edits(R, g, override) for R in itrows} for g in GATES}
        # nested: fix thr=0.45, select gate per fold on other folds
        nby = {}; nest = {}
        for k in range(5):
            b2 = (-1, None); oids = set(R["id"] for R in itrows if R["fold"] != k)
            for g in GATES:
                sub = {i: cache[g][i] for i in oids}
                _s, d = elru.elru(sub, {i: truth[i] for i in oids}, {i: "it" for i in oids}, detail=True)
                if d["it"]["lang_score"] > b2[0]:
                    b2 = (d["it"]["lang_score"], g)
            nby[k] = b2[1]
            for R in [r for r in itrows if r["fold"] == k]:
                nest[R["id"]] = cache[b2[1]][R["id"]]
        # non-nested best gate
        bg = max(GATES, key=lambda g: elru.elru(cache[g], truth, lm, detail=True)[1]["it"]["lang_score"])
        _sN, dN = elru.elru({i: cache[bg][i] for i in truth}, truth, lm, detail=True)
        _s, dn = elru.elru(nest, truth, lm, detail=True)
        fp = sum(1 for R in itrows if len(R["truth"]) == 0 and cache[bg][R["id"]])
        tag = "NP-override" if override else "NP-safe"
        print(f"{tag:12s} NN it={dN['it']['lang_score']:.4f}(e{dN['it']['edited_mean']:.3f}/u{dN['it']['unchanged_mean']:.3f},g={bg},FP{fp}) "
              f"NEST it={dn['it']['lang_score']:.4f}(e{dn['it']['edited_mean']:.3f}/u{dn['it']['unchanged_mean']:.3f}) nby={nby}")
    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

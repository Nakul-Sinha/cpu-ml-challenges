"""N1 exp8: does adding robustly-gated SINGLE-TOKEN candidates (below-threshold) to the
NP capture help, under the robust protocol (fixed spine thr=0.45, gate-only nested)?
Shared gate with a 'kind' feature. Compare vs NP-only (0.4217 nested).
"""
import os, sys, json, time, collections, re, pickle
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline, m4_ext
from transducer import Transducer
from run_m4 import base_cache, group_consistency, score_edits, SHIP_VOTE_LANGS
import elru
sys.path.insert(0, os.path.join(ROOT, "runs", "N1"))
from run_n1 import learn_tab, group_ctx, reorder

_STRIP = ".,;:()»«\"'“”’`-–—"; MARKS = set(":*∗/"); WS = re.compile(r"\S+")


def rebuild(train):
    m4_ext.register(pipeline)
    trs, stf = {}, {}
    for k in range(5):
        trdf = train[train.fold != k]
        trs[k] = Transducer().fit(trdf); st = {}
        for b in pipeline.STORE_BUILDERS:
            b(trdf, st)
        stf[k] = st
    return trs, stf


def cands(R, pr, tab, gc, add_single):
    tk = R["tk"]; text = R["text"]; n = len(tk); gs, gsz = gc.get(R["document_group"], (0.0, 0.0))
    out = []
    for i in range(n):
        core = tk[i][2].strip(_STRIP).lower()
        if not core:
            continue
        e2 = tab["end2_rate"].get(core[-2:], 0.0) if len(core) >= 2 else 0.0
        edr = tab["tok_edrate"].get(core, 0.0)
        # single-token candidate (below/near threshold, some edit signal)
        if add_single and (pr[i] >= 0.20 or edr >= 0.08 or e2 >= 0.15):
            a, b = tk[i][0], tk[i][1]
            f = [0.0, tab["spaninit_rate"].get(core, 0.0), edr, float(len(core)), 0.0, float(b - a),
                 float(pr[i]), float(pr[i]), float(pr[i]), float(pr[i]),
                 float(pr[i - 1]) if i > 0 else 0.0, e2, e2, edr, 0.0,
                 1.0 if "/" in tk[i][2] else 0.0, gs, gsz]
            out.append((a, b, f))
        # NP candidate
        if core in tab["anchors"]:
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
                f = [1.0, tab["spaninit_rate"].get(core, 0.0), tab["tok_edrate"].get(core, 0.0), float(len(core)),
                     float(L), float(b - a), float(np.mean(sp)), float(np.max(sp)), float(np.min(sp)),
                     float(sp[0]), float(sp[-1]), max(fe2) if fe2 else 0.0, float(np.mean(fe2)) if fe2 else 0.0,
                     max(fedr) if fedr else 0.0, float(nmasc), 1.0 if "/" in text[a:b] else 0.0, gs, gsz]
                out.append((a, b, f))
    return out


def main():
    import lightgbm as lgb
    t0 = time.time()
    S = pickle.load(open(os.path.join(ROOT, "runs", "N1_state.pkl"), "rb"))
    rows = S["rows"]; idfold = S["idfold"]; row_proba = S["row_proba"]; gbi = S["group_by_id"]
    for R in rows:
        R["document_group"] = gbi[R["id"]]
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    trs, stf = rebuild(train)
    tabs = {k: learn_tab(train[train.fold != k]) for k in range(5)}
    gctxs = {k: group_ctx(train[train.fold != k]) for k in range(5)}
    itrows = [R for R in rows if R["lang"] == "it"]
    truth = {R["id"]: R["truth"] for R in itrows}; lm = {R["id"]: "it" for R in itrows}
    GATES = [1.01, 0.8, 0.7, 0.6, 0.5, 0.4]

    def run(add_single):
        cand = {}
        for R in itrows:
            k = idfold[R["id"]]; cs = cands(R, row_proba[R["id"]], tabs[k], gctxs[k], add_single)
            lab = [1 if max((max(0, min(b, te) - max(a, ts)) / (max(b, te) - min(a, ts))
                             for (ts, te, rep) in R["spans"] if rep != "" and max(b, te) > min(a, ts)), default=0.0) >= 0.5 else 0
                   for (a, b, f) in cs]
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

        def it_edits(R, g):
            tk = R["tk"]; text = R["text"]; pr = row_proba[R["id"]]; n = len(tk)
            k = idfold[R["id"]]; T = trs[k]; st = stf[k]; spans = []; i = 0
            while i < n:
                if pr[i] >= 0.45:
                    j = i
                    while j + 1 < n and pr[j + 1] >= 0.45:
                        j += 1
                    spans.append((tk[i][0], tk[j][1], float(np.mean(pr[i:j + 1])) + 1.0)); i = j + 1
                else:
                    i += 1
            cs, lb, pv = cand[R["id"]]
            for (c, p) in zip(cs, pv):
                if p >= g:
                    spans.append((c[0], c[1], float(p)))
            spans.sort(key=lambda s: -s[2]); chosen = []; occ = []
            for (a, b, sc) in spans:
                if any(not (b <= x or y <= a) for (x, y) in occ):
                    continue
                chosen.append((a, b)); occ.append((a, b))
            edits = []
            for (a, b) in chosen:
                src = text[a:b]; ctx = {"text": text, "start": a, "end": b, "lang": "it", "tokens": tk, "stores": st}
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

        cache = {g: {R["id"]: it_edits(R, g) for R in itrows} for g in GATES}
        nby = {}; nest = {}
        for k in range(5):
            oids = set(R["id"] for R in itrows if R["fold"] != k)
            bg = max(GATES, key=lambda g: elru.elru({i: cache[g][i] for i in oids},
                     {i: truth[i] for i in oids}, {i: "it" for i in oids}, detail=True)[1]["it"]["lang_score"])
            nby[k] = bg
            for R in [r for r in itrows if r["fold"] == k]:
                nest[R["id"]] = cache[bg][R["id"]]
        _s, dn = elru.elru(nest, truth, lm, detail=True)
        fp = sum(1 for R in itrows if len(R["truth"]) == 0 and nest[R["id"]])
        return dn["it"], fp, nby

    for add_single, tag in [(False, "NP-only"), (True, "NP+single")]:
        d, fp, nby = run(add_single)
        print(f"{tag:12s} NEST it={d['lang_score']:.4f} (e{d['edited_mean']:.3f}/u{d['unchanged_mean']:.3f}) FP={fp} nby={nby}")
    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

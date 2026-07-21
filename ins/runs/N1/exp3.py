"""N1 exp3: MEMORY-DRIVEN candidate proposal for it (independent of token threshold).
Propose token-runs whose normalized form is in the per-fold transducer memory (known
editable surface forms) -> catches undetected-but-known edits. Measured net, leak-free.
Also test lexicon single-token proposal for comparison.
"""
import os, sys, json, time, collections, re, pickle
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline, m4_ext
from transducer import Transducer, _norm
from run_m4 import group_consistency, score_edits, fp_counts, SHIP_VOTE_LANGS
import elru

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"
GRID = pipeline.GRID


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


def mem_spans(tk, lang, T, max_len=4):
    """maximal token-runs whose _norm is in T.norm/T.exact with a NON-identity rep."""
    out = []
    n = len(tk)
    for i in range(n):
        best = None
        for L in range(min(max_len, n - i), 0, -1):
            j = i + L - 1
            a, b = tk[i][0], tk[j][1]
            src = None  # filled by caller via text slice; here use token join proxy
            out.append((i, j))
            break  # only the longest starting here (greedy); refine in build
    return out


def build_it_edits(R, probs, thr, T, st, use_mem=False, mem_min_change=True):
    """threshold-merge UNION memory-driven spans (it), transduce, non-overlap by len, cap 8."""
    tk = R["tk"]; text = R["text"]; n = len(tk)
    spans = []
    # base threshold merge
    i = 0
    while i < n:
        if probs[i] >= thr:
            j = i
            while j + 1 < n and probs[j + 1] >= thr:
                j += 1
            spans.append((i, j, float(np.mean(probs[i:j + 1])) + 1.0))  # +1 prio for base
            i = j + 1
        else:
            i += 1
    # memory-driven spans
    if use_mem:
        used = set()
        for i in range(n):
            for L in range(min(4, n - i), 0, -1):
                j = i + L - 1
                a, b = tk[i][0], tk[j][1]
                src = text[a:b]
                nk = (R["lang"], _norm(src))
                ek = (R["lang"], src)
                rep = None
                if ek in T.exact:
                    rep = T.exact[ek]
                elif nk in T.norm:
                    rep = T.norm[nk]
                if rep is not None and (not mem_min_change or (rep != src and rep != "")):
                    # score by mean prob (memory spans get lower prio than base)
                    sc = float(np.mean(probs[i:j + 1]))
                    spans.append((i, j, sc))
                    break  # longest memory match starting at i
    # non-overlap: sort by score desc, greedily keep
    spans.sort(key=lambda s: -s[2])
    chosen = []
    occ = []
    for (i, j, sc) in spans:
        a, b = tk[i][0], tk[j][1]
        if any(not (b <= x or y <= a) for (x, y) in occ):
            continue
        chosen.append((i, j, sc)); occ.append((a, b))
    # transduce
    edits = []
    for (i, j, sc) in chosen:
        a, b = tk[i][0], tk[j][1]
        src = text[a:b]
        ctx = {"text": text, "start": a, "end": b, "lang": R["lang"], "tokens": tk, "stores": st}
        rep = None
        for hook in pipeline.REPLACEMENT_HOOKS:
            r = hook(R["lang"], src, ctx, st)
            if r is not None:
                rep = r; break
        if rep is None:
            rep = T.predict(R["lang"], src, ctx)
            if rep is None:
                rep = src
        edits.append((sc, {"start": a, "end": b, "replacement": rep[:160]}))
    edits.sort(key=lambda e: -e[0])
    edits = [e for _, e in edits[:8]]
    edits.sort(key=lambda e: e["start"])
    if not elru.validate_edits(edits, len(text)):
        edits = pipeline._repair(edits, len(text))
    return edits


def main():
    t0 = time.time()
    S = pickle.load(open(os.path.join(ROOT, "runs", "N1_state.pkl"), "rb"))
    rows = S["rows"]; idfold = S["idfold"]; row_proba = S["row_proba"]; gbi = S["group_by_id"]
    rows_by_id = {R["id"]: R for R in rows}
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    trs, stf = rebuild(train)
    from run_m4 import base_cache, base_select
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    print(f"[rebuild {time.time()-t0:.0f}s]")

    itrows = [R for R in rows if R["lang"] == "it"]

    def it_cache(use_mem):
        cache = {}
        for R in itrows:
            k = idfold[R["id"]]; T = trs[k]; st = stf[k]; pr = row_proba[R["id"]]
            cache[R["id"]] = {thr: build_it_edits(R, pr, thr, T, st, use_mem=use_mem) for thr in GRID}
        return cache

    def select_it(cache):
        best = (-1, GRID[0])
        for thr in GRID:
            e = {R["id"]: cache[R["id"]][thr] for R in itrows}
            _s, d = elru.elru(e, {R["id"]: R["truth"] for R in itrows},
                              {R["id"]: R["lang"] for R in itrows}, detail=True)
            if d["it"]["lang_score"] > best[0]:
                best = (d["it"]["lang_score"], thr)
        # nested
        nby = {}
        nest = {}
        for k in range(5):
            b2 = (-1, GRID[0])
            other = [R for R in itrows if R["fold"] != k]
            for thr in GRID:
                e = {R["id"]: cache[R["id"]][thr] for R in other}
                _s, d = elru.elru(e, {R["id"]: R["truth"] for R in other},
                                  {R["id"]: R["lang"] for R in other}, detail=True)
                if d["it"]["lang_score"] > b2[0]:
                    b2 = (d["it"]["lang_score"], thr)
            nby[k] = b2[1]
            for R in [r for r in itrows if r["fold"] == k]:
                nest[R["id"]] = cache[R["id"]][b2[1]]
        nn = {R["id"]: cache[R["id"]][best[1]] for R in itrows}
        return best[1], nn, nby, nest

    # baseline it (rebuild via my builder, no mem) to verify parity
    for use_mem in [False, True]:
        c = it_cache(use_mem)
        thr, nn, nby, nest = select_it(c)
        _s, dnn = elru.elru(nn, {R["id"]: R["truth"] for R in itrows},
                            {R["id"]: R["lang"] for R in itrows}, detail=True)
        _s2, dne = elru.elru(nest, {R["id"]: R["truth"] for R in itrows},
                             {R["id"]: R["lang"] for R in itrows}, detail=True)
        fp = sum(1 for R in itrows if len(R["truth"]) == 0 and nn[R["id"]])
        tag = "MEM" if use_mem else "base(rebuilt)"
        print(f"{tag:16s} it_thr={thr} NN it={dnn['it']['lang_score']:.4f}"
              f"(e{dnn['it']['edited_mean']:.3f}/u{dnn['it']['unchanged_mean']:.3f}) "
              f"NEST it={dne['it']['lang_score']:.4f}(e{dne['it']['edited_mean']:.3f}/u{dne['it']['unchanged_mean']:.3f}) "
              f"FP={fp} nby={nby}")
    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

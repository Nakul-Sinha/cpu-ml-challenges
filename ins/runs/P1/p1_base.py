"""P1 base harness: fit the N3 folds ONCE (leak-free shared detector OOF probs +
per-fold transducers/stores + it NP-gate materials), cache to pickle, and reproduce
the baseline it nested (0.4205) / de / en so downstream detector-upgrade experiments
(Lever 1 it re-scorer, Lever 2 neural tagger) reuse the identical base without refit.

Run: cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/P1/p1_base.py
"""
import os, sys, json, time, pickle, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/insled")
for p in (os.path.join(ROOT, "runs", "M4"), os.path.join(ROOT, "runs", "N2"),
          os.path.join(ROOT, "runs", "N1"), os.path.join(ROOT, "solution"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
import pipeline, n2_ext
from transducer import Transducer
import elru
from run_m4 import base_cache, base_select, group_consistency, score_edits, fp_counts, SHIP_VOTE_LANGS
from run_n1 import (learn_tab, group_ctx, np_cands, it_gate_scores, assemble_it as n1_assemble_it,
                    build_it_cache, select_it, IT_SPINE_THR, GATE_GRID)

LANGS = pipeline.LANGS
CACHE = os.path.join(HERE, "p1_base_cache.pkl")


def load_train():
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    return train


def fit_folds(train, verbose=True):
    t0 = time.time()
    rows = pipeline.build_rows(train, labeled=True)
    idfold = {R["id"]: R["fold"] for R in rows}
    row_proba, trs, stf = {}, {}, {}
    for k in range(5):
        tr_rows = [R for R in rows if R["fold"] != k]
        va_rows = [R for R in rows if R["fold"] == k]
        trdf = train[train.fold != k]
        st = {}
        for b in pipeline.STORE_BUILDERS:
            b(trdf, st)
        det = pipeline.Detector().fit(tr_rows, st)
        for _id, (tk, pr) in det.token_probs(va_rows).items():
            row_proba[_id] = pr
        trs[k] = Transducer().fit(trdf)
        stf[k] = st
        if verbose:
            print(f"[fold {k}] fit ({time.time()-t0:.0f}s)", flush=True)
    return rows, idfold, row_proba, trs, stf


def prepare(verbose=True):
    n2_ext.register(pipeline)
    train = load_train()
    gbi = {r.id: r.document_group for r in train.itertuples()}
    rows, idfold, row_proba, trs, stf = fit_folds(train, verbose)
    rows_by_id = {R["id"]: R for R in rows}
    tabs = {k: learn_tab(train[train.fold != k]) for k in range(5)}
    gctxs = {k: group_ctx(train[train.fold != k]) for k in range(5)}
    gate_scores = it_gate_scores(rows, idfold, row_proba, tabs, gctxs, gbi)
    return dict(train=train, gbi=gbi, rows=rows, idfold=idfold, row_proba=row_proba,
                trs=trs, stf=stf, rows_by_id=rows_by_id, tabs=tabs, gctxs=gctxs,
                gate_scores=gate_scores)


def de_en_it_baseline(P):
    """Reproduce the N3 headline rung components (de/en base_select + it NP-gate fixed spine)."""
    rows = P["rows"]; idfold = P["idfold"]; gbi = P["gbi"]; trs = P["trs"]; stf = P["stf"]
    rbi = P["rows_by_id"]; row_proba = P["row_proba"]; gate_scores = P["gate_scores"]
    # de/en via N2 base cache
    cache_n2 = base_cache(rows, idfold, row_proba, trs, stf)
    b_nn_thr, b_nn_e, b_nby, b_ne_e = base_select(rows, cache_n2)
    # it via N1 fixed-spine NP-gate assembly
    itcache = build_it_cache(rows, idfold, row_proba, trs, stf, gate_scores)
    it_nn_g, it_nn_e, it_nby, it_ne_e = select_it(rows, itcache)

    def merge(base_map, it_map):
        return {i: (it_map[i] if rbi[i]["lang"] == "it" else base_map[i]) for i in base_map}

    nn_e = merge(b_nn_e, it_nn_e); ne_e = merge(b_ne_e, it_ne_e)
    nn = group_consistency(nn_e, rbi, gbi, trs, stf, idfold,
                           vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    ne = group_consistency(ne_e, rbi, gbi, trs, stf, idfold,
                           vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    nn_s, nn_d = score_edits(rows, nn); ne_s, ne_d = score_edits(rows, ne)
    return dict(nn_s=nn_s, nn_d=nn_d, ne_s=ne_s, ne_d=ne_d, b_nn_thr=b_nn_thr, b_nby=b_nby,
                b_nn_e=b_nn_e, b_ne_e=b_ne_e, it_nn_g=it_nn_g, it_nby=it_nby, cache_n2=cache_n2,
                itcache=itcache, nn=nn, ne=ne)


def main():
    t0 = time.time()
    P = prepare(verbose=True)
    print(f"[prepare done {time.time()-t0:.0f}s]", flush=True)
    B = de_en_it_baseline(P)
    print("\n================ P1 BASE REPRODUCTION (N3 headline rung) ================")
    print(f"NON-NESTED overall = {B['nn_s']:.4f}  " +
          " ".join(f"{L}={B['nn_d'][L]['lang_score']:.4f}" for L in LANGS))
    print(f"NESTED   overall = {B['ne_s']:.4f}  " +
          " ".join(f"{L}={B['ne_d'][L]['lang_score']:.4f}" for L in LANGS))
    print(f"  de/en thr(nonnested)={B['b_nn_thr']}  it NP gate(nonnested)={B['it_nn_g']} by-fold={B['it_nby']}")
    fp = fp_counts(P["rows"], B["nn"])
    print("unchanged-row FP (nonnested): " + ", ".join(f"{L}={fp[L][0]}/{fp[L][1]}" for L in LANGS))
    print(f"EXPECTED N3: nested 0.5503 (de .4237 en .8067 it .4205)")

    # ---- cache the reusable base artifacts (incl trs/stf/tabs/gctxs for zero-refit reuse) ----
    P["train"].to_pickle(os.path.join(HERE, "p1_train.pkl"))
    slim_rows = [{k: R[k] for k in ("id", "lang", "text", "tk", "y", "spans", "truth", "fold")}
                 for R in P["rows"]]
    payload = dict(rows=slim_rows, idfold=P["idfold"], row_proba=P["row_proba"],
                   gbi=P["gbi"], gate_scores=P["gate_scores"], trs=P["trs"], stf=P["stf"],
                   tabs=P["tabs"], gctxs=P["gctxs"],
                   b_nn_thr=B["b_nn_thr"], b_nby=B["b_nby"], it_nn_g=B["it_nn_g"], it_nby=B["it_nby"],
                   base_ne_lang={L: B["ne_d"][L]["lang_score"] for L in LANGS},
                   base_nn_lang={L: B["nn_d"][L]["lang_score"] for L in LANGS},
                   base_ne=B["ne_s"], base_nn=B["nn_s"])
    try:
        with open(CACHE, "wb") as f:
            pickle.dump(payload, f)
        print(f"[cached full -> {CACHE}]")
    except Exception as e:
        print(f"[full pickle failed: {e}; caching slim]")
        payload.pop("trs"); payload.pop("stf")
        with open(CACHE, "wb") as f:
            pickle.dump(payload, f)
    print(f"\n[base done]  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

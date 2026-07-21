"""Sanity: measure BASE (threshold-merge) pipeline with M2+M3 plug-ins integrated.
Confirms the composed registration is leak-free and roughly additive (de from M2,
it from M3) before the reranker is layered on."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pipeline
import m4_ext

def main():
    ap = sys.argv[1] if len(sys.argv) > 1 else "1"
    os.environ.setdefault("M4_EXACTFIRST", ap)
    m4_ext.register(pipeline)
    ROOT = pipeline.ROOT
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    t0 = time.time()
    res = pipeline.run_cv(train, verbose=False)
    print(f"M4 BASE+M2+M3 (exactfirst={os.environ['M4_EXACTFIRST']})  [{time.time()-t0:.0f}s]")
    print(f"  NON-NESTED ELRU = {res['nonnested_elru']:.4f}   thr={res['nonnested_thr']}")
    for L in pipeline.LANGS:
        d = res["nonnested_detail"][L]
        print(f"    {L}: lang={d['lang_score']:.4f} edited={d['edited_mean']:.4f}(n={d['n_edited']}) unchanged={d['unchanged_mean']:.4f}(n={d['n_unchanged']})")
    print(f"  NESTED ELRU     = {res['nested_elru']:.4f}")
    for L in pipeline.LANGS:
        d = res["nested_detail"][L]
        print(f"    {L}: lang={d['lang_score']:.4f} edited={d['edited_mean']:.4f} unchanged={d['unchanged_mean']:.4f}")
    print(f"  del_diag={res['del_diag']}")

if __name__ == "__main__":
    main()

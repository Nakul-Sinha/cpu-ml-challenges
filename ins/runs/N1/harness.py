"""N1 lean iteration harness: reproduces the M4 SHIP path (base threshold-merge +
group-vote[de,en]) WITHOUT the reranker (not in ship config), so it runs fast.
Confirms the 0.5423 nested baseline, then serves as the eval backbone for the it config.

Reuses pipeline + m4_ext exactly; only fit_folds (5 detector/transducer/stores) + the
base threshold-merge assembly are needed for the ship path.
"""
import os, sys, json, time, collections
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline
import m4_ext
import run_m4
from run_m4 import (fit_folds, base_cache, base_select, group_consistency, score_edits,
                    fp_counts, per_type_recall, print_detail, LOSSMAP, SHIP_VOTE_LANGS)

LANGS = pipeline.LANGS


def load():
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    return train


def baseline():
    t0 = time.time()
    m4_ext.register(pipeline)
    train = load()
    gbi = {r.id: r.document_group for r in train.itertuples()}
    rows, idfold, row_proba, trs, stf = fit_folds(train, verbose=True)
    rows_by_id = {R["id"]: R for R in rows}
    print(f"[fit_folds done {time.time()-t0:.0f}s]")
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    nn_thr, nn_e, nby, ne_e = base_select(rows, bcache)
    # ship group vote de+en
    nn_ship = group_consistency({i: nn_e[i] for i in nn_e}, rows_by_id, gbi, trs, stf, idfold,
                                vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    ne_ship = group_consistency({i: ne_e[i] for i in ne_e}, rows_by_id, gbi, trs, stf, idfold,
                                vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    nn_s, nn_d = score_edits(rows, nn_ship)
    ne_s, ne_d = score_edits(rows, ne_ship)
    print_detail("NON-NESTED", nn_s, nn_d)
    print(f"  thr={nn_thr}")
    print_detail("NESTED", ne_s, ne_d)
    fp = fp_counts(rows, nn_ship)
    print("FP(nonnest): " + ", ".join(f"{L}={fp[L][0]}/{fp[L][1]}" for L in LANGS))
    print(f"[total {time.time()-t0:.0f}s]")
    return dict(train=train, rows=rows, idfold=idfold, row_proba=row_proba,
               transducers=trs, stores_by_fold=stf, rows_by_id=rows_by_id, group_by_id=gbi,
               bcache=bcache, nn_thr=nn_thr, nn_e=nn_e, ne_e=ne_e, nby=nby)


if __name__ == "__main__":
    baseline()

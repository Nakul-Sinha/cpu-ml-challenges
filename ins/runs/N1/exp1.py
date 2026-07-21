"""N1 exp1: test SAFE it replacement-quality levers with honest nested selection.
fit_folds once; evaluate base + {group-conv[it], src-first slash reorder, combos}.
Post-fixes touch ONLY it replacement strings (no boundaries) -> de/en untouched, no new FPs.
"""
import os, sys, json, time, collections, re, copy
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline, m4_ext
from run_m4 import (fit_folds, base_cache, base_select, group_consistency, score_edits,
                    fp_counts, per_type_recall, print_detail, LOSSMAP, SHIP_VOTE_LANGS)
import elru

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"


def load():
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    return train


def reorder_srcfirst(src, rep):
    """it single-token: if rep = A/B (one slash, no space) and src-core in {A,B}, put src first."""
    if rep.count("/") != 1 or " " in rep or "/" not in rep:
        return rep
    core = src.strip(_STRIP)
    if not core:
        return rep
    a, b = rep.split("/")
    if core == b and core != a:      # src currently second -> swap to src-first
        return core + "/" + a
    return rep


def apply_postfix(cache, rows_by_id, do_reorder):
    """return a new cache with it replacement post-fixes applied (boundaries untouched)."""
    out = {}
    for rid, thrmap in cache.items():
        R = rows_by_id[rid]
        if R["lang"] != "it" or not do_reorder:
            out[rid] = thrmap
            continue
        text = R["text"]
        newmap = {}
        for thr, edits in thrmap.items():
            ne = []
            for e in edits:
                src = text[e["start"]:e["end"]]
                rep = e["replacement"]
                if len(src.split()) == 1:
                    rep = reorder_srcfirst(src, rep)
                ne.append({"start": e["start"], "end": e["end"], "replacement": rep})
            newmap[thr] = ne
        out[rid] = newmap
    return out


def main():
    t0 = time.time()
    m4_ext.register(pipeline)
    train = load()
    gbi = {r.id: r.document_group for r in train.itertuples()}
    rows, idfold, row_proba, trs, stf = fit_folds(train, verbose=False)
    rows_by_id = {R["id"]: R for R in rows}
    print(f"[fit_folds {time.time()-t0:.0f}s]")
    bcache = base_cache(rows, idfold, row_proba, trs, stf)

    def evaluate(cache, tag, conv_it=False):
        nn_thr, nn_e, nby, ne_e = base_select(rows, cache)
        # ship group vote de+en
        nn_ship = group_consistency({i: nn_e[i] for i in nn_e}, rows_by_id, gbi, trs, stf, idfold,
                                    vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS,
                                    do_conv=conv_it, conv_langs=({"it"} if conv_it else None))
        ne_ship = group_consistency({i: ne_e[i] for i in ne_e}, rows_by_id, gbi, trs, stf, idfold,
                                    vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS,
                                    do_conv=conv_it, conv_langs=({"it"} if conv_it else None))
        nn_s, nn_d = score_edits(rows, nn_ship)
        ne_s, ne_d = score_edits(rows, ne_ship)
        fp = fp_counts(rows, nn_ship)
        print(f"\n---- {tag} ----  it_thr={nn_thr['it']}")
        print(f"  NESTED  overall={ne_s:.4f}  it lang={ne_d['it']['lang_score']:.4f} "
              f"(ed={ne_d['it']['edited_mean']:.4f} un={ne_d['it']['unchanged_mean']:.4f})  "
              f"de={ne_d['de']['lang_score']:.4f} en={ne_d['en']['lang_score']:.4f}")
        print(f"  NONNEST overall={nn_s:.4f}  it lang={nn_d['it']['lang_score']:.4f} "
              f"(ed={nn_d['it']['edited_mean']:.4f} un={nn_d['it']['unchanged_mean']:.4f})  "
              f"FP it={fp['it'][0]}/{fp['it'][1]} de={fp['de'][0]} en={fp['en'][0]}")
        return ne_s, ne_d

    # E0 base
    evaluate(bcache, "E0 BASE (ship)")
    # E1 base + group-conv[it]
    evaluate(bcache, "E1 +group-conv[it]", conv_it=True)
    # E2 base + reorder src-first
    rc = apply_postfix(bcache, rows_by_id, do_reorder=True)
    evaluate(rc, "E2 +reorder-srcfirst")
    # E3 reorder + conv[it]
    evaluate(rc, "E3 +reorder+conv[it]", conv_it=True)

    print(f"\n[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

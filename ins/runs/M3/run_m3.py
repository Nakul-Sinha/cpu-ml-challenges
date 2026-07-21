"""M3 driver: run M1 pipeline with (subset of) M3 plug-ins; report nested/nonnested
ELRU + per-language + per-type recall so component deltas are visible.

usage: python run_m3.py <mode>
  mode in {base, feats, feats_repl, feats_repl_npgen, all}
Toggles are applied to m3_ext before registering.
"""
import os, sys, json, time, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P
import m3_ext as M

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
WS = P.WORD_RE
MARKS = set(":*∗/")


def classify(src, rep):
    ntok = len(src.split())
    marked = any(c in src for c in MARKS)
    if rep == "":
        return "deletion"
    if ntok == 1:
        return "single_marked" if marked else "single_plain"
    return "multi_marked" if marked else "multi_plain"


def configure(mode):
    # default: everything off, then enable per mode
    M.USE_FEATS = M.USE_IT_REPL = M.USE_EN_REPL = M.USE_DEL = M.USE_NPGEN = False
    if mode == "base":
        # register nothing -> pure M1
        P.STORE_BUILDERS = []; P.TOKEN_FEATURE_EXTRAS = []
        P.REPLACEMENT_HOOKS = []; P.SPAN_CANDIDATE_GENERATORS = []
        P.FEAT_NAMES = None; P.EXTRA_NAMES = None
        return
    if mode in ("feats", "feats_repl", "feats_repl_npgen", "all"):
        M.USE_FEATS = True
    if mode in ("feats_repl", "feats_repl_npgen", "all"):
        M.USE_IT_REPL = True; M.USE_EN_REPL = True
    if mode in ("feats_repl_npgen", "all"):
        M.USE_NPGEN = True
    if mode == "all":
        M.USE_DEL = True
    M.register(P)


def per_type_recall(res):
    """for each true span, hit if any predicted edit overlaps it; bucket by (lang,type)."""
    rows = res["rows"]; ec = res["edits_cache"]; assign = res["nn_assign"]
    tot = collections.Counter(); hit = collections.Counter()
    for R in rows:
        preds = ec[R["id"]][assign[R["id"]]]
        for a, b, rep in R["spans"]:
            src = R["text"][a:b]
            t = (R["lang"], classify(src, rep))
            tot[t] += 1
            ov = any(not (e["end"] <= a or b <= e["start"]) for e in preds)
            if ov:
                hit[t] += 1
    return tot, hit


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(P.ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(P.ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)

    configure(MODE)
    res = P.run_cv(train, verbose=False)

    print(f"==== MODE={MODE} ====  ({res['seconds']:.0f}s)")
    print(f"NONNESTED ELRU = {res['nonnested_elru']:.4f}   thr={ {k: float(v) for k,v in res['nonnested_thr'].items()} }")
    print(f"NESTED    ELRU = {res['nested_elru']:.4f}   (headline)")
    for L in P.LANGS:
        d = res["nonnested_detail"][L]
        print(f"  {L}: lang={d['lang_score']:.4f} edited={d['edited_mean']:.4f}(n={d['n_edited']}) "
              f"unch={d['unchanged_mean']:.4f}(n={d['n_unchanged']})")
    tot, hit = per_type_recall(res)
    print("  per-type recall (overlap):")
    for k in sorted(tot):
        print(f"     {k[0]} {k[1]:14} {hit[k]:4}/{tot[k]:<4} = {hit[k]/tot[k]:.3f}")
    dd = res["del_diag"]
    print(f"  deletion diag: true={dd['n_true_del']} pred_empty={dd['del_pred']} hit={dd['del_hit']}")
    # write a compact json for cross-mode comparison
    out = dict(mode=MODE, nonnested=round(res["nonnested_elru"], 4),
               nested=round(res["nested_elru"], 4),
               lang={L: round(res["nonnested_detail"][L]["lang_score"], 4) for L in P.LANGS},
               recall={f"{k[0]}_{k[1]}": [hit[k], tot[k]] for k in tot},
               del_diag=dd, seconds=round(res["seconds"], 1))
    json.dump(out, open(os.path.join(HERE, f"m3_{MODE}.json"), "w"), indent=2)
    print(f"  wrote m3_{MODE}.json   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

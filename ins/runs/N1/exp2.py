"""N1 exp2: FP-removal for it via DROP mechanisms (isolated from the edited-hurting
vote propagation) + training-rate gating. Fast: loads cached OOF state.
"""
import os, sys, json, time, collections, pickle
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline, m4_ext
from transducer import Transducer
from run_m4 import (base_cache, base_select, group_consistency, score_edits, fp_counts,
                    per_type_recall, SHIP_VOTE_LANGS)
import elru

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"


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


def learn_it_tokedr(train):
    """per-fold it token edit-rate (leak-free): core -> ed/occ on train[fold!=k]."""
    out = {}
    for k in range(5):
        occ = collections.Counter(); ed = collections.Counter()
        for r in train[train.fold != k].itertuples():
            if r.language != "it":
                continue
            spans = sorted((e["start"], e["end"]) for e in r.edits)
            import re
            for m in re.finditer(r"\S+", r.text):
                s, e = m.start(), m.end()
                core = m.group().strip(_STRIP).lower()
                if not core:
                    continue
                occ[core] += 1
                if any(s >= a and e <= b for a, b in spans):
                    ed[core] += 1
        out[k] = {w: ed[w] / occ[w] for w in occ}, occ
    return out


def drop_it(edits_map, rows_by_id, gbi, idfold, tokedr, lo=None, tau=None):
    """Remove single-token it edits by: group coverage<lo (inference-time) OR training
    tok_edrate<tau (leak-free per fold). Returns new edits_map."""
    out = {i: [dict(e) for e in edits_map[i]] for i in edits_map}
    # group coverage of cores
    groups = collections.defaultdict(list)
    for i in out:
        groups[gbi[i]].append(i)
    for g, ids in groups.items():
        occ = collections.Counter(); cov = collections.Counter()
        for i in ids:
            R = rows_by_id[i]
            if R["lang"] != "it":
                continue
            import re
            covspans = [(e["start"], e["end"]) for e in out[i]]
            for m in re.finditer(r"\S+", R["text"]):
                s, e = m.start(), m.end()
                core = m.group().strip(_STRIP).lower()
                if len(core) < 2:
                    continue
                occ[core] += 1
                if any(s >= a and e <= b for a, b in covspans):
                    cov[core] += 1
        for i in ids:
            R = rows_by_id[i]
            if R["lang"] != "it":
                continue
            k = idfold[i]; tr = tokedr[k][0]
            new = []
            for e in out[i]:
                toks_in = [w for w in [R["text"][e["start"]:e["end"]]] if len(w.split()) == 1]
                if len(e_src := R["text"][e["start"]:e["end"]].split()) == 1:
                    core = R["text"][e["start"]:e["end"]].strip(_STRIP).lower()
                    drop = False
                    if lo is not None and occ.get(core, 0) >= 2 and (cov.get(core, 0) / occ[core]) < lo:
                        drop = True
                    if tau is not None and tr.get(core, 1.0) < tau:
                        drop = True
                    if drop:
                        continue
                new.append(e)
            out[i] = new
    return out


def main():
    t0 = time.time()
    S = pickle.load(open(os.path.join(ROOT, "runs", "N1_state.pkl"), "rb"))
    rows = S["rows"]; idfold = S["idfold"]; row_proba = S["row_proba"]; gbi = S["group_by_id"]
    rows_by_id = {R["id"]: R for R in rows}
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    trs, stf = rebuild(train)
    tokedr = learn_it_tokedr(train)
    print(f"[rebuild {time.time()-t0:.0f}s]")
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    nn_thr, nn_e, nby, ne_e = base_select(rows, bcache)

    def ship_vote(em):
        return group_consistency({i: em[i] for i in em}, rows_by_id, gbi, trs, stf, idfold,
                                 vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)

    def rep(tag, nn_em, ne_em):
        nn_s, nn_d = score_edits(rows, nn_em); ne_s, ne_d = score_edits(rows, ne_em)
        fp = fp_counts(rows, nn_em)
        print(f"{tag:30s} NEST ov={ne_s:.4f} it={ne_d['it']['lang_score']:.4f}"
              f"(e{ne_d['it']['edited_mean']:.3f}/u{ne_d['it']['unchanged_mean']:.3f}) | "
              f"NN ov={nn_s:.4f} it={nn_d['it']['lang_score']:.4f} FPit={fp['it'][0]} "
              f"de={ne_d['de']['lang_score']:.3f} en={ne_d['en']['lang_score']:.3f}")

    # E0 base+ship vote
    rep("E0 base+shipvote", ship_vote(nn_e), ship_vote(ne_e))

    # drop sweeps (apply drop to it BEFORE de/en ship vote; drop is post-select)
    for lo in [0.30, 0.40, 0.50]:
        nn_d = drop_it(nn_e, rows_by_id, gbi, idfold, tokedr, lo=lo)
        ne_d = drop_it(ne_e, rows_by_id, gbi, idfold, tokedr, lo=lo)
        rep(f"drop lo={lo}", ship_vote(nn_d), ship_vote(ne_d))
    for tau in [0.05, 0.10, 0.15, 0.20]:
        nn_d = drop_it(nn_e, rows_by_id, gbi, idfold, tokedr, tau=tau)
        ne_d = drop_it(ne_e, rows_by_id, gbi, idfold, tokedr, tau=tau)
        rep(f"drop tau={tau}", ship_vote(nn_d), ship_vote(ne_d))
    # combined best-ish
    for lo, tau in [(0.40, 0.10), (0.50, 0.15), (0.40, 0.15)]:
        nn_d = drop_it(nn_e, rows_by_id, gbi, idfold, tokedr, lo=lo, tau=tau)
        ne_d = drop_it(ne_e, rows_by_id, gbi, idfold, tokedr, lo=lo, tau=tau)
        rep(f"drop lo={lo},tau={tau}", ship_vote(nn_d), ship_vote(ne_d))

    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

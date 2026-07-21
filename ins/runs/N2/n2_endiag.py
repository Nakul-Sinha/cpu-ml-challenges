"""Confirm the EN slash/mark masking actually changes en token probs, and reconcile
the en per-type recall (non-nested vs nested)."""
import os, sys, json, collections
M4 = os.path.expanduser("~/insled/runs/M4"); N2 = os.path.expanduser("~/insled/runs/N2")
sys.path.insert(0, M4); sys.path.insert(0, N2); sys.path.insert(0, os.path.expanduser("~/insled/solution"))
import numpy as np, pandas as pd
import pipeline, m4_ext, run_m4
import elru

SLASH_MARK_FEATS = {"has_special", "spat_rate", "specsuf_rate", "specsuf_len",
                    "specsuf_alpha", "specsuf_short", "spat_sup", "spc_id",
                    "specsuf_id", "mid_/", "mid_*", "mid_:", "mid_∗", "maxchar_rate"}


def main():
    m4_ext.register(pipeline)
    ROOT = pipeline.ROOT
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    rows = pipeline.build_rows(train, labeled=True)
    # fit on folds!=0, predict en fold==0 normal vs masked
    tr_rows = [R for R in rows if R["fold"] != 0]
    stores = {}
    for b in pipeline.STORE_BUILDERS:
        b(train[train.fold != 0], stores)
    det = pipeline.Detector().fit(tr_rows, stores)
    en_va = [R for R in rows if R["fold"] == 0 and R["lang"] == "en"]
    Xn, _ = pipeline.featurize(en_va, det.lex); Xn = np.array(Xn, dtype=np.float32)
    mask_idx = [i for i, nm in enumerate(pipeline.FEAT_NAMES) if nm in SLASH_MARK_FEATS]
    Xm = Xn.copy()
    for c in mask_idx:
        Xm[:, c] = 0.0
    pn = det.model.predict_proba(Xn)[:, 1]
    pm = det.model.predict_proba(Xm)[:, 1]
    changed = int(np.sum(np.abs(pn - pm) > 1e-6))
    # which tokens changed most: those with a slash/mark char
    MARKS = set(":*∗/")
    idx = 0; slash_tok_changed = 0; slash_tok_total = 0; maxdelta = 0.0
    for R in en_va:
        for (s, e, w) in R["tk"]:
            d = abs(pn[idx] - pm[idx])
            if any(ch in MARKS for ch in w):
                slash_tok_total += 1
                if d > 1e-6:
                    slash_tok_changed += 1
                maxdelta = max(maxdelta, d)
            idx += 1
    print(f"en fold0 tokens: {len(pn)}  prob-changed-by-masking: {changed}")
    print(f"  slash/mark tokens: {slash_tok_total}  of which prob-changed: {slash_tok_changed}  max|delta|={maxdelta:.4f}")
    print(f"  masked feature cols: {[pipeline.FEAT_NAMES[i] for i in mask_idx]}")
    # mean prob on slash tokens normal vs masked
    idx = 0; sn = []; sm = []
    for R in en_va:
        for (s, e, w) in R["tk"]:
            if any(ch in MARKS for ch in w):
                sn.append(pn[idx]); sm.append(pm[idx])
            idx += 1
    if sn:
        print(f"  slash-token mean prob: normal={np.mean(sn):.4f} masked={np.mean(sm):.4f}")

    # Reconcile per-type recall: non-nested en edits (thr .39) like the compare section
    row_proba = {}
    transducers = {}; stores_by_fold = {}
    for k in range(5):
        trk = [R for R in rows if R["fold"] != k]
        stk = {}
        for b in pipeline.STORE_BUILDERS:
            b(train[train.fold != k], stk)
        dk = pipeline.Detector().fit(trk, stk)
        for _id, (tk, pr) in dk.token_probs([R for R in rows if R["fold"] == k]).items():
            row_proba[_id] = pr
        transducers[k] = pipeline.Transducer().fit(train[train.fold != k])
        stores_by_fold[k] = stk
    idfold = {R["id"]: R["fold"] for R in rows}
    en_rows = [R for R in rows if R["lang"] == "en"]
    edits = {}
    for R in en_rows:
        k = idfold[R["id"]]
        edits[R["id"]] = pipeline.build_edits(R["id"], R["text"], R["lang"], R["tk"],
                                              row_proba[R["id"]], 0.39, transducers[k], stores_by_fold[k])
    rec = collections.defaultdict(lambda: [0, 0])
    for R in en_rows:
        preds = edits[R["id"]]
        for (ts, te, rep) in R["spans"]:
            if rep == "":
                continue
            key = run_m4.span_type("en", R["text"][ts:te])
            rec[key][1] += 1
            best = max((run_m4.iou(ed["start"], ed["end"], ts, te) for ed in preds), default=0.0)
            if best >= 0.5:
                rec[key][0] += 1
    print("\nen non-nested (thr .39) per-type recall (reconcile w/ compare 0.940 single_plain):")
    for key in sorted(rec):
        v = rec[key]
        print(f"    {key:14s} rec={v[0]/max(v[1],1):.3f} (n={v[1]})")


if __name__ == "__main__":
    main()

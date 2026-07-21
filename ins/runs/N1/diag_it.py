"""N1 DIAGNOSIS: explain the it multi_plain .508 -> .336 recall regression.

Pure span-boundary analysis on the M4 OOF artifacts (leak-free probs already baked).
For it, the ship path is threshold-merge only (no generators/reranker/group-vote), so
merge_threshold_spans(tk, probs, thr) reproduces the predicted edit-span boundaries
exactly.  Recall (IoU>=0.5) needs boundaries only -> no transducer required here.
"""
import os, sys, json, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "solution"))

MARKS = set(":*∗/")
_STRIP = ".,;:()»«\"'“”’`-–—"


def span_type(src):
    nt = len(src.split())
    marked = any(c in MARKS for c in src)
    return ("single" if nt == 1 else "multi") + ("_marked" if marked else "_plain")


def iou(a, b, c, d):
    ov = max(0, min(b, d) - max(a, c)); un = max(b, d) - min(a, c)
    return ov / un if un > 0 else 0.0


def merge_threshold_spans(tk, probs, thr):
    spans = []; i = 0; n = len(tk)
    while i < n:
        if probs[i] >= thr:
            j = i
            while j + 1 < n and probs[j + 1] >= thr:
                j += 1
            spans.append((tk[i][0], tk[j][1], i, j))
            i = j + 1
        else:
            i += 1
    return spans


def main():
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    tp = pd.read_csv(os.path.join(ROOT, "runs", "M4", "oof_token_probs.csv"))
    # token probs per id: list ordered by tok_index
    probs_by_id = {}
    toks_by_id = {}
    for _id, g in tp.groupby("id"):
        g = g.sort_values("tok_index")
        probs_by_id[_id] = g.proba.tolist()
        toks_by_id[_id] = list(zip(g.start.tolist(), g.end.tolist()))

    it = train[train.language == "it"].copy()
    print(f"IT rows: {len(it)}  (edited={(it.edits.apply(len)>0).sum()})")

    # collect true spans by type
    def classify_and_diagnose(thr):
        buckets = collections.defaultdict(lambda: [0, 0])  # type -> [hit, total]
        # multi_plain failure-mode tally
        fm = collections.Counter()
        mp_detail = []
        for r in it.itertuples():
            tk = toks_by_id.get(r.id)
            pr = probs_by_id.get(r.id)
            if tk is None:
                continue
            pred = merge_threshold_spans(tk, pr, thr)
            for e in r.edits:
                if e["replacement"] == "":
                    continue
                s, en = e["start"], e["end"]
                src = r.text[s:en]
                st = span_type(src)
                buckets[st][1] += 1
                best = max((iou(a, b, s, en) for (a, b, _i, _j) in pred), default=0.0)
                hit = best >= 0.5
                if hit:
                    buckets[st][0] += 1
                if st == "multi_plain":
                    # diagnose failure mode
                    # token indices spanned by true edit
                    tin = [idx for idx, (ts, te) in enumerate(tk) if ts >= s and te <= en]
                    # fraction of true-span tokens above threshold
                    above = [idx for idx in tin if pr[idx] >= thr]
                    # predicted spans overlapping the true region
                    ov_spans = [(a, b) for (a, b, _i, _j) in pred if not (b <= s or en <= a)]
                    frac_above = len(above) / max(1, len(tin))
                    if hit:
                        fm["HIT"] += 1
                    elif len(ov_spans) == 0:
                        fm["MISS_no_token_above_thr"] += 1
                    elif len(ov_spans) >= 2:
                        fm["FRAGMENTED_2plus_spans"] += 1
                    else:
                        # exactly one overlapping predicted span but IoU<0.5
                        (a, b) = ov_spans[0]
                        if (b - a) > (en - s) * 1.3:
                            fm["OVEREXTENDED"] += 1
                        elif (b - a) < (en - s) * 0.7:
                            fm["UNDERCOVERED_partial"] += 1
                        else:
                            fm["MISALIGNED"] += 1
                    if thr in (0.45,):
                        mp_detail.append(dict(id=r.id, src=src, ntok=len(tin),
                                              frac_above=round(frac_above, 2),
                                              nov=len(ov_spans), best_iou=round(best, 2)))
        return buckets, fm, mp_detail

    print("\n=== per-type IoU>=0.5 recall at several thresholds (it) ===")
    thrs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75]
    types = ["single_plain", "single_marked", "multi_plain", "multi_marked"]
    header = "thr    " + "  ".join(f"{t:>16s}" for t in types)
    print(header)
    for thr in thrs:
        b, fm, _ = classify_and_diagnose(thr)
        cells = []
        for t in types:
            hit, tot = b[t]
            cells.append(f"{(hit/tot if tot else 0):.3f}({tot:>3d})" if tot else "  -      ")
        print(f"{thr:.2f}  " + "  ".join(f"{c:>16s}" for c in cells))

    print("\n=== multi_plain failure-mode breakdown at thr=0.45 vs 0.68 ===")
    for thr in (0.45, 0.68):
        _, fm, det = classify_and_diagnose(thr)
        tot = sum(fm.values())
        print(f"\nthr={thr}: total multi_plain true spans={tot}")
        for k, v in fm.most_common():
            print(f"    {k:32s} {v:3d}  ({v/max(1,tot):.1%})")

    # detail listing at 0.45: show the non-hit multi_plain cases
    _, _, det = classify_and_diagnose(0.45)
    print("\n=== multi_plain cases at thr=0.45 (frac_above = frac of true-span tokens >=thr) ===")
    misses = [d for d in det if d["best_iou"] < 0.5]
    hits = [d for d in det if d["best_iou"] >= 0.5]
    print(f"HITS={len(hits)}  MISSES={len(misses)}")
    # distribution of frac_above among misses
    fa = collections.Counter()
    for d in misses:
        fa[d["frac_above"]] += 1
    print("frac_above distribution among MISSES:", dict(sorted(fa.items())))
    fa2 = collections.Counter()
    for d in hits:
        fa2[d["frac_above"]] += 1
    print("frac_above distribution among HITS:  ", dict(sorted(fa2.items())))
    # ntok distribution
    print("ntok among misses:", dict(sorted(collections.Counter(d["ntok"] for d in misses).items())))


if __name__ == "__main__":
    main()

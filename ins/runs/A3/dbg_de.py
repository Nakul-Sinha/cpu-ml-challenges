import sys, json; sys.path.insert(0, "runs/A3"); sys.path.insert(0, "solution")
import pandas as pd, numpy as np, compose as C, elru
train = pd.read_csv("dataset/train.csv"); folds = pd.read_csv("solution/folds.csv")
train = train.merge(folds, on="id"); train["ed"] = train.edits_json.apply(json.loads)
k = 0
tr = train[train.fold != k]; va = train[train.fold == k]
det = C.Detector(rounds=350).fit(tr); tp = det.token_probs(va)
trd = C.Transducer().fit(tr)
de = va[va.language == "de"]

# detection recall on marked vs unmarked EDITED tokens, at a few thresholds
for thr in [0.24, 0.4, 0.6]:
    tot_m = hit_m = tot_u = hit_u = 0
    for r in de.itertuples():
        tk, probs = tp[r.id]
        spans = [(e["start"], e["end"]) for e in r.ed]
        for j, (s, e, w) in enumerate(tk):
            if any(a <= s and e <= b for a, b in spans):
                marked = any(c in ":*/" for c in w)
                if marked: tot_m += 1; hit_m += probs[j] >= thr
                else:      tot_u += 1; hit_u += probs[j] >= thr
    print(f"thr {thr}: marked recall {hit_m}/{tot_m}  unmarked recall {hit_u}/{tot_u}")

print("\n=== sample German edited rows: pred vs truth @thr0.3 ===")
n = 0
for r in de[de.ed.apply(len) > 0].itertuples():
    tk, probs = tp[r.id]
    pred = C.assemble(r.id, r.text, "de", tk, probs, trd, 0.3)
    rs = elru.row_score(pred, r.ed)
    if n < 6:
        print(f"\nrow {r.id} row_score={rs:.3f}")
        for e in r.ed:
            print("  TRUE", repr(r.text[e['start']:e['end']]), "->", repr(e['replacement'][:60]))
        for e in pred:
            print("  PRED", repr(r.text[e['start']:e['end']]), "->", repr(e['replacement'][:60]))
        n += 1

# transducer accuracy on marked single tokens (exact-match rep) in this fold's val
print("\n=== transducer exact-rep accuracy on de single-token marked spans (val) ===")
ok = tot = 0
for r in de.itertuples():
    for e in r.ed:
        src = r.text[e["start"]:e["end"]]; stk = C.toks(src)
        if len(stk) == 1 and any(c in ":*/" for c in src):
            got = trd.transduce("de", src); tot += 1; ok += (got == e["replacement"])
print(f"  {ok}/{tot} exact rep match")

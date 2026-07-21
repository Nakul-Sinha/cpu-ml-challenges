"""Diagnostic baselines over the canonical folds (NOT a final model):
A) no-op: [] everywhere.
B) token-memorization: learn from in-fold train the surface tokens whose edit rate
   >= thr with support >= min_sup, emit majority replacement at each occurrence.
   Measures how much pure cross-group surface transfer exists.
"""
import sys, json, re, collections
import pandas as pd

sys.path.insert(0, "solution")
from elru import elru

WORD_RE = re.compile(r"\S+")
train = pd.read_csv("dataset/train.csv")
folds = pd.read_csv("solution/folds.csv")
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)

def toks(text):
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]

def fit_memo(df, thr=0.6, min_sup=2):
    seen = collections.Counter()
    edited = collections.Counter()
    reps = collections.defaultdict(collections.Counter)
    for r in df.itertuples():
        spans = [(e["start"], e["end"], e["replacement"]) for e in r.edits]
        for s, e, w in toks(r.text):
            key = (r.language, w)
            seen[key] += 1
            for a, b, rep in spans:
                if s == a and e == b:  # token IS the full span
                    edited[key] += 1
                    reps[key][rep] += 1
                    break
    memo = {}
    for key, ed in edited.items():
        if ed >= min_sup and ed / seen[key] >= thr:
            memo[key] = reps[key].most_common(1)[0][0]
    return memo

def predict_memo(df, memo):
    out = {}
    for r in df.itertuples():
        edits = []
        for s, e, w in toks(r.text):
            key = (r.language, w)
            if key in memo:
                edits.append({"start": s, "end": e, "replacement": memo[key]})
        out[r.id] = edits[:8]
    return out

for name, thr, sup in [("noop", None, None), ("memo t0.6 s2", 0.6, 2), ("memo t0.8 s2", 0.8, 2), ("memo t0.5 s1", 0.5, 1)]:
    scores = []
    for k in range(5):
        tr = train[train.fold != k]
        va = train[train.fold == k]
        tm = {r.id: r.edits for r in va.itertuples()}
        lm = {r.id: r.language for r in va.itertuples()}
        if name == "noop":
            pm = {r.id: [] for r in va.itertuples()}
        else:
            memo = fit_memo(tr, thr, sup)
            pm = predict_memo(va, memo)
        s, det = elru(pm, tm, lm, detail=True)
        scores.append(s)
    print(f"{name}: CV ELRU = {sum(scores)/5:.4f}  folds={[round(x,3) for x in scores]}")
    if name != "noop":
        print("   last-fold detail:", {L: {kk: (round(vv,3) if isinstance(vv,float) else vv) for kk,vv in d.items()} for L, d in det.items()})

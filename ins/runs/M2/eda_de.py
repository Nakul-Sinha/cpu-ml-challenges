"""German-focused EDA: characterize edit types, paired forms, connectors, collapse targets."""
import os, sys, json, re, collections
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P

train = pd.read_csv(os.path.join(P.ROOT, "dataset", "train.csv"))
folds = pd.read_csv(os.path.join(P.ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)

de = train[train.language == "de"]
print(f"de rows: {len(de)}  edited: {sum(len(e)>0 for e in de.edits)}  unchanged: {sum(len(e)==0 for e in de.edits)}")

MARKS = set(":*∗/")
def is_marked(s):
    return any(c in MARKS for c in s)

# --- classify every de edit span ---
buckets = collections.Counter()
examples = collections.defaultdict(list)
for r in de.itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        rep = e["replacement"]
        ntok = len(src.split())
        single = ntok == 1
        marked = is_marked(src)
        if rep == "":
            typ = "deletion"
        elif single and marked:
            typ = "single_marked"
        elif single and not marked:
            typ = "single_plain"
        elif not single and marked:
            typ = "multi_marked"
        else:
            typ = "multi_plain"
        buckets[typ] += 1
        if len(examples[typ]) < 25:
            examples[typ].append((src, rep, ntok))

print("\n=== de edit-span type buckets ===")
for t, c in buckets.most_common():
    print(f"  {t:16s} {c}")

for t in ["multi_plain", "multi_marked", "single_plain", "single_marked", "deletion"]:
    print(f"\n=== {t} examples (src -> rep) ===")
    for src, rep, ntok in examples[t]:
        print(f"  [{ntok}] {src!r:55s} -> {rep!r}")

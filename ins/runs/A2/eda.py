"""A2 EDA: understand (lang, src_span, rep) transduction structure."""
import sys, json, re, collections
import pandas as pd

sys.path.insert(0, "solution")
from elru import replacement_chrf

train = pd.read_csv("dataset/train.csv")
folds = pd.read_csv("solution/folds.csv")
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)

# collect all oracle edits as (lang, src, rep, ntoks)
rows = []
for r in train.itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        rep = e["replacement"]
        rows.append((r.language, src, rep, len(src.split()), r.fold, r.id))
edf = pd.DataFrame(rows, columns=["lang", "src", "rep", "ntok", "fold", "id"])
print("total oracle edits:", len(edf))
print("by lang:\n", edf.lang.value_counts())
print("\ndeletion (rep=='') fraction by lang:")
edf["is_del"] = edf.rep == ""
print(edf.groupby("lang").is_del.mean())
print("total deletions:", edf.is_del.sum())

print("\nntok distribution:")
print(edf.ntok.value_counts().sort_index())

# marker chars in src (single-token)
print("\n--- single-token src marker analysis ---")
one = edf[edf.ntok == 1]
def marker(s):
    for c in [":", "*", "/", "∗"]:
        if c in s:
            return c
    return "none"
one = one.copy()
one["mk"] = one.src.apply(marker)
print(one.groupby(["lang","mk"]).size())

# For German single-token colon/star: is rep = STEM + CONN + STEM + suffix?
print("\n--- sample de single-token edits ---")
for r in one[one.lang=="de"].head(20).itertuples():
    print(repr(r.src), "->", repr(r.rep))

print("\n--- sample it edits (all ntok) ---")
for r in edf[edf.lang=="it"].head(25).itertuples():
    print(r.ntok, repr(r.src), "->", repr(r.rep))

print("\n--- sample en edits ---")
for r in edf[edf.lang=="en"].head(25).itertuples():
    print(r.ntok, repr(r.src), "->", repr(r.rep))

# Exact-memory cross-fold coverage: for each val edit, is (lang,src) in train-fold memory?
print("\n--- exact-memory cross-fold coverage & quality ---")
tot_cov=0; tot=0; cov_chrf=0.0; all_chrf_if_memhit=[]
per_lang = collections.defaultdict(lambda: {"n":0,"cov":0,"chrf_cov":0.0})
for k in range(5):
    tr = edf[edf.fold!=k]; va = edf[edf.fold==k]
    memo = collections.defaultdict(collections.Counter)
    for r in tr.itertuples():
        memo[(r.lang, r.src)][r.rep]+=1
    for r in va.itertuples():
        tot+=1; per_lang[r.lang]["n"]+=1
        key=(r.lang,r.src)
        if key in memo:
            pred = memo[key].most_common(1)[0][0]
            c = replacement_chrf(pred, r.rep)
            tot_cov+=1; cov_chrf+=c
            per_lang[r.lang]["cov"]+=1; per_lang[r.lang]["chrf_cov"]+=c
print(f"exact-mem coverage: {tot_cov}/{tot} = {tot_cov/tot:.3f}; mean chrf on covered = {cov_chrf/max(tot_cov,1):.4f}")
for L,d in per_lang.items():
    print(f"  {L}: cov {d['cov']}/{d['n']}={d['cov']/d['n']:.3f}  chrf_on_cov={d['chrf_cov']/max(d['cov'],1):.4f}")

# How many distinct src have ambiguous rep (convention)?
amb = collections.defaultdict(set)
for r in edf.itertuples():
    amb[(r.lang,r.src)].add(r.rep)
nam = sum(1 for v in amb.values() if len(v)>1)
print(f"\ndistinct (lang,src): {len(amb)}, ambiguous (>1 rep): {nam}")

# baseline: predict rep=src (identity) chrf, and rep='' always
print("\n--- trivial baselines (mean chrf over ALL edits) ---")
id_c = edf.apply(lambda r: replacement_chrf(r.src, r.rep), axis=1).mean()
print("identity (pred=src):", round(id_c,4))

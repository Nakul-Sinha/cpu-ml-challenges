"""Docstring gap diagnostics on a dedup-aware holdout:
A) constant fillers ("the", "of the", ...) -> floor reference
B) anchored context retrieval: exact left-word + right-word around [GAP] from a
   train index (most common target for that context) -> transfer reference
C) oracle-ceiling probes: best train target by unigram-prior per row cap.
Split: hash of masked_docstring -> avoids twin leakage between train/val.
Reads CSVs with keep_default_na=False (spans like 'nan'/'null' are real text).
"""
import sys, re, collections, hashlib, time
import pandas as pd

sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists

t0 = time.time()
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
print("loaded", train.shape, f"{time.time()-t0:.1f}s")

def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20

b = train.masked_docstring.map(bucket)
val = train[b == 0].sample(8000, random_state=1)
trn = train[b != 0]
print("trn", len(trn), "val", len(val))

refs = val.target_span.astype(str).tolist()

# A) constant fillers
for c in ["the", "of the", "the given", "a", "and", "value of the"]:
    print(f"const {c!r}: {score_lists([c]*len(refs), refs):.4f}")

# B) anchored context retrieval, backoff left2+right2 -> left1+right1 -> left1 -> const
GAP = "[GAP]"
def ctx(s, nl, nr):
    i = s.find(GAP)
    L = s[:i].split()[-nl:] if nl else []
    R = s[i+len(GAP):].split()[:nr] if nr else []
    return (" ".join(L), " ".join(R))

idx2 = collections.defaultdict(collections.Counter)
idx1 = collections.defaultdict(collections.Counter)
idxL = collections.defaultdict(collections.Counter)
for r in trn.itertuples():
    s = r.masked_docstring
    tgt = str(r.target_span)
    idx2[ctx(s, 2, 2)][tgt] += 1
    idx1[ctx(s, 1, 1)][tgt] += 1
    idxL[ctx(s, 1, 0)][tgt] += 1

preds = []
src_counts = collections.Counter()
for r in val.itertuples():
    s = r.masked_docstring
    for name, idx, key in [("l2r2", idx2, ctx(s, 2, 2)), ("l1r1", idx1, ctx(s, 1, 1)), ("l1", idxL, ctx(s, 1, 0))]:
        if key in idx:
            preds.append(idx[key].most_common(1)[0][0])
            src_counts[name] += 1
            break
    else:
        preds.append("the")
        src_counts["const"] += 1
print("\nanchored retrieval:", f"{score_lists(preds, refs):.4f}", dict(src_counts))

# C) per-bucket quality: score by which index level was used
lvl = []
for r in val.itertuples():
    s = r.masked_docstring
    for name, idx, key in [("l2r2", idx2, ctx(s, 2, 2)), ("l1r1", idx1, ctx(s, 1, 1)), ("l1", idxL, ctx(s, 1, 0))]:
        if key in idx:
            lvl.append(name)
            break
    else:
        lvl.append("const")
val2 = val.copy()
val2["pred"] = preds
val2["lvl"] = lvl
val2["f"] = [f_pooled(p, r) for p, r in zip(preds, refs)]
print(val2.groupby("lvl").f.agg(["mean", "size"]))

print(f"\ntotal {time.time()-t0:.1f}s")

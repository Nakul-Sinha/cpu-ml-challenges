"""Probe candidate-pool ceilings + train/test coverage parity.
Pool per row: anchored-context candidates (top5 each level) + global top spans
+ code-derived tokens. Report oracle chrF of pool (upper bound for reranking)
and hit-rates on val vs test (drift check).
"""
import sys, re, collections, hashlib, time
import pandas as pd

sys.path.insert(0, "solution")
from chrf import f_pooled

t0 = time.time()
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
test = pd.read_csv("dataset/test.csv", keep_default_na=False)

def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20

b = train.masked_docstring.map(bucket)
val = train[b == 0].sample(4000, random_state=1)
trn = train[b != 0]

GAP = "[GAP]"
def ctx(s, nl, nr):
    i = s.find(GAP)
    L = s[:i].split()[-nl:] if nl else []
    R = s[i + len(GAP):].split()[:nr] if nr else []
    return (" ".join(L), " ".join(R))

idx2 = collections.defaultdict(collections.Counter)
idx1 = collections.defaultdict(collections.Counter)
idxL = collections.defaultdict(collections.Counter)
idxR = collections.defaultdict(collections.Counter)
for r in trn.itertuples():
    s = r.masked_docstring
    tgt = str(r.target_span)
    idx2[ctx(s, 2, 2)][tgt] += 1
    idx1[ctx(s, 1, 1)][tgt] += 1
    idxL[ctx(s, 1, 0)][tgt] += 1
    idxR[ctx(s, 0, 1)][tgt] += 1
top_global = [k for k, v in collections.Counter(train.target_span.astype(str)).most_common(20)]

ident_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
def code_cands(code):
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", code)
    out = []
    if m:
        name = m.group(1)
        words = re.split(r"_+", name)
        words = [w for w in words if w]
        if words:
            out.append(" ".join(words))
            out += words
        args = [a.split("=")[0].strip() for a in m.group(2).split(",")]
        for a in args[:4]:
            if a and a not in ("self", "cls"):
                out.append(a.replace("_", " "))
    return out[:8]

def pool_for(s, code):
    cands = []
    for idx, key, k in [(idx2, ctx(s, 2, 2), 5), (idx1, ctx(s, 1, 1), 8), (idxL, ctx(s, 1, 0), 5), (idxR, ctx(s, 0, 1), 5)]:
        if key in idx:
            cands += [c for c, _ in idx[key].most_common(k)]
    cands += code_cands(code)
    cands += top_global[:8]
    seen = set(); out = []
    for c in cands:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

# oracle over pool on val
oracle = []; pool_sizes = []; hit_exact = 0
t1 = time.time()
for r in val.itertuples():
    pool = pool_for(r.masked_docstring, r.code_context)
    pool_sizes.append(len(pool))
    tgt = str(r.target_span)
    best = max((f_pooled(c, tgt) for c in pool), default=0.0)
    oracle.append(best)
    hit_exact += tgt in pool
print(f"val pool oracle chrF: {sum(oracle)/len(oracle):.4f}  exact-hit: {hit_exact/len(val):.3f}  mean pool {sum(pool_sizes)/len(pool_sizes):.1f}  ({time.time()-t1:.1f}s)")

# coverage parity: what fraction of rows hit each index level, val vs test sample
def coverage(df):
    c = collections.Counter()
    for r in df.itertuples():
        s = r.masked_docstring
        c["l2r2"] += ctx(s, 2, 2) in idx2
        c["l1r1"] += ctx(s, 1, 1) in idx1
        c["l1"] += ctx(s, 1, 0) in idxL
        c["r1"] += ctx(s, 0, 1) in idxR
    return {k: v / len(df) for k, v in c.items()}

tsample = test.sample(4000, random_state=2)
print("coverage val :", {k: round(v, 3) for k, v in coverage(val).items()})
print("coverage test:", {k: round(v, 3) for k, v in coverage(tsample).items()})

# masked len + gap words parity
print("val masked len mean", val.masked_docstring.str.len().mean(), " test", tsample.masked_docstring.str.len().mean())
print(f"total {time.time()-t0:.1f}s")

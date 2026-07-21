"""Benchmark fuzzy full-sentence NN retrieval for the candidate pool.
HashingVectorizer char_wb 3-5, 2^18, l2, alternate_sign=False (NO idf).
Chunked sparse cosine top-k neighbors; measure timing + oracle contribution.
"""
import sys, os, time, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

os.environ.setdefault("OMP_NUM_THREADS", "5")
sys.path.insert(0, "solution")
from chrf import f_pooled

t0 = time.time()
train = pd.read_csv("dataset/train.csv", keep_default_na=False)

def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20

b = train.masked_docstring.map(bucket)
trn = train[b != 0].reset_index(drop=True)
val = train[b == 0].sample(4000, random_state=1).reset_index(drop=True)
print(f"trn {len(trn)} val {len(val)} loaded {time.time()-t0:.1f}s", flush=True)

GAP = "[GAP]"
def window(s, w=4):
    i = s.find(GAP)
    L = s[:i].split()[-w:]
    R = s[i+len(GAP):].split()[:w]
    return " ".join(L + R)

# ---- Full-sentence view ----
hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18,
                       lowercase=True, alternate_sign=False, norm="l2")
t1 = time.time()
Xtr = hv.transform(trn.masked_docstring.values)
print(f"transform train full: {time.time()-t1:.1f}s  nnz/row={Xtr.nnz/Xtr.shape[0]:.0f}  shape={Xtr.shape}", flush=True)
t1 = time.time()
Xq = hv.transform(val.masked_docstring.values)
print(f"transform val full: {time.time()-t1:.1f}s", flush=True)

XtrT = Xtr.T.tocsr()  # F x n
tgts = trn.target_span.astype(str).values

def knn_chunk(Xq, XtrT, k=15, chunk=256):
    n = Xq.shape[0]
    all_idx = np.empty((n, k), dtype=np.int32)
    all_sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        q = Xq[start:start+chunk]
        sims = (q @ XtrT)  # sparse c x N
        sims = sims.toarray()  # dense
        c = sims.shape[0]
        kk = min(k, sims.shape[1])
        part = np.argpartition(-sims, kk-1, axis=1)[:, :kk]
        row = np.arange(c)[:, None]
        pv = sims[row, part]
        order = np.argsort(-pv, axis=1)
        all_idx[start:start+c] = part[row, order]
        all_sim[start:start+c] = pv[row, order]
    return all_idx, all_sim

t1 = time.time()
idx, sim = knn_chunk(Xq, XtrT, k=15, chunk=256)
dt = time.time()-t1
print(f"knn full 4000 queries: {dt:.1f}s  -> 50k extrapolate {dt*50000/4000:.0f}s single-thread", flush=True)

# oracle of fuzzy neighbors alone (top-15 targets)
refs = val.target_span.astype(str).values
orc = []
for i in range(len(val)):
    cands = list(dict.fromkeys(tgts[idx[i]]))  # dedup keep order
    orc.append(max(f_pooled(c, refs[i]) for c in cands))
print(f"fuzzy-full top15 oracle: {np.mean(orc):.4f}  mean uniq cands {np.mean([len(set(tgts[idx[i]])) for i in range(len(val))]):.1f}", flush=True)

# neighbor sim stats
print(f"top1 sim mean {sim[:,0].mean():.3f}  top15 sim mean {sim[:,14].mean():.3f}", flush=True)
print(f"total {time.time()-t0:.1f}s", flush=True)

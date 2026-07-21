"""Optimize fuzzy KNN: test threading parallelism + profile matmul vs topk."""
import sys, os, time, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "solution")

train = pd.read_csv("dataset/train.csv", keep_default_na=False)
def bucket(s):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) % 20
b = train.masked_docstring.map(bucket)
trn = train[b != 0].reset_index(drop=True)
val = train[b == 0].sample(4000, random_state=1).reset_index(drop=True)

hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18,
                       lowercase=True, alternate_sign=False, norm="l2")
Xtr = hv.transform(trn.masked_docstring.values)
Xq = hv.transform(val.masked_docstring.values)
XtrT = Xtr.T.tocsr()
print("setup done", flush=True)

# profile one chunk: matmul vs toarray vs argpartition
q = Xq[:256]
t=time.time(); sims = q @ XtrT; t_mm=time.time()-t
t=time.time(); dense = sims.toarray(); t_ta=time.time()-t
k=15
t=time.time()
part = np.argpartition(-dense, k-1, axis=1)[:, :k]
t_ap=time.time()-t
print(f"chunk256: matmul {t_mm:.2f}s  toarray {t_ta:.2f}s  argpart {t_ap:.2f}s  result_nnz/row {sims.nnz/256:.0f}", flush=True)

def knn_range(args):
    s,e,k,chunk = args
    out_i=[]; out_s=[]
    for st in range(s,e,chunk):
        en=min(st+chunk,e)
        sims=(Xq[st:en] @ XtrT).toarray()
        kk=min(k,sims.shape[1])
        part=np.argpartition(-sims,kk-1,axis=1)[:,:kk]
        rr=np.arange(sims.shape[0])[:,None]
        pv=sims[rr,part]
        order=np.argsort(-pv,axis=1)
        out_i.append(part[rr,order]); out_s.append(pv[rr,order])
    return np.vstack(out_i), np.vstack(out_s)

# single thread baseline (1000 q)
t=time.time(); knn_range((0,1000,15,256)); t1=time.time()-t
print(f"1000q single-thread: {t1:.1f}s -> 50k {t1*50:.0f}s", flush=True)

# threaded over 5 workers on 5000 q (all val + repeat)
N=4000
bnds=np.linspace(0,N,6).astype(int)
tasks=[(bnds[i],bnds[i+1],15,256) for i in range(5)]
t=time.time()
with ThreadPoolExecutor(5) as ex:
    res=list(ex.map(knn_range,tasks))
tt=time.time()-t
print(f"{N}q 5-threads: {tt:.1f}s -> 50k {tt*50000/N:.0f}s (speedup {t1/ (tt*1000/N):.1f}x)", flush=True)

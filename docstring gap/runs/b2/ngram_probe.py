"""Test ngram ranges for fuzzy KNN: speed (result density) vs oracle."""
import sys, os, time, hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
os.environ.setdefault("OMP_NUM_THREADS","5")
sys.path.insert(0,"solution")
from chrf import f_pooled

train = pd.read_csv("dataset/train.csv", keep_default_na=False)
def bucket(s): return int(hashlib.md5(s.encode("utf-8","ignore")).hexdigest()[:8],16)%20
b=train.masked_docstring.map(bucket)
trn=train[b!=0].reset_index(drop=True)
val=train[b==0].sample(4000,random_state=1).reset_index(drop=True)
refs=val.target_span.astype(str).values

# dedup check
uniq = trn.masked_docstring.nunique()
print(f"train rows {len(trn)} unique masked {uniq} ({uniq/len(trn):.2%})", flush=True)

def knn(Xq,XtrT,k,chunk=256):
    n=Xq.shape[0]; oi=np.empty((n,k),np.int32)
    for st in range(0,n,chunk):
        sims=(Xq[st:st+chunk]@XtrT).toarray()
        kk=min(k,sims.shape[1])
        part=np.argpartition(-sims,kk-1,axis=1)[:,:kk]
        rr=np.arange(sims.shape[0])[:,None]
        order=np.argsort(-sims[rr,part],axis=1)
        oi[st:st+sims.shape[0]]=part[rr,order]
    return oi

tgts=trn.target_span.astype(str).values
for ng in [(3,5),(4,5),(4,6),(5,6),(3,4)]:
    hv=HashingVectorizer(analyzer="char_wb",ngram_range=ng,n_features=2**18,
                         lowercase=True,alternate_sign=False,norm="l2")
    Xtr=hv.transform(trn.masked_docstring.values); XtrT=Xtr.T.tocsr()
    Xq=hv.transform(val.masked_docstring.values)
    # measure matmul density on 256
    s0=(Xq[:256]@XtrT); dens=s0.nnz/256
    t=time.time(); idx=knn(Xq[:1000],XtrT,15); dt=time.time()-t
    orc=[max(f_pooled(c,refs[i]) for c in dict.fromkeys(tgts[idx[i]])) for i in range(1000)]
    print(f"ng{ng} nnz/row={Xtr.nnz/len(trn):.0f} resdens={dens:.0f} 1000q={dt:.1f}s(50k~{dt*50:.0f}s) oracle={np.mean(orc):.4f}", flush=True)

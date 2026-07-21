"""Fast gated-vs-ungated oracle + naive top1, optimized oracle (ref ngrams cached)."""
import sys, os, time, hashlib
from collections import Counter
import numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS","5")
sys.path.insert(0,"solution"); sys.path.insert(0,"runs/b2")
from chrf import f_pooled, score_lists
from pool_builder import PoolBuilder

train = pd.read_csv("dataset/train.csv", keep_default_na=False)
def bk(s): return int(hashlib.md5(s.encode("utf-8","ignore")).hexdigest()[:8],16)%20
b=train.masked_docstring.map(bk)
trn=train[b!=0].reset_index(drop=True)
val=train[b==0].sample(4000,random_state=7).reset_index(drop=True)
refs=val.target_span.astype(str).values

pb=PoolBuilder(cap=80); pb.fit(trn)

# cache ref ngram multisets
def grams(s,nmax=6):
    c=Counter()
    for n in range(1,nmax+1):
        for i in range(len(s)-n+1): c[s[i:i+n]]+=1
    return c
refg=[grams(r) for r in refs]; reflen=[sum(g.values()) for g in refg]
def fmax(cands,i):
    rg=refg[i]; tr=reflen[i]; best=0.0
    for c in cands:
        cg=grams(c); tp=sum(cg.values())
        if tp==0 or tr==0: continue
        m=sum(min(v,rg[k]) for k,v in cg.items())
        if m==0: continue
        p=m/tp; r=m/tr; f=2*p*r/(p+r)
        if f>best: best=f
    return best
def oracle(pools):
    return np.mean([fmax([t for t,_,_ in pools[i]],i) for i in range(len(pools))])

for gate,name in [(False,"ungated"),(True,"gated")]:
    pb.gate_fuzz=gate
    t=time.time(); P=pb.candidates_batch(val); dt=time.time()-t
    gf=np.mean([any(s in("fuzz","fuzzw") for _,s,_ in r) for r in P])
    o=oracle(P); top1=[r[0][0] if r else "" for r in P]
    print(f"{name:8s}: oracle {o:.4f}  fuzz-fired {gf:.0%}  time {dt:.1f}s(50k~{dt*12.5:.0f}s)  top1-chrF {score_lists(top1,refs.tolist()):.4f}", flush=True)

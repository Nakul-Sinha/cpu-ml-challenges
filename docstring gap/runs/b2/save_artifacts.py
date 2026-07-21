"""Build ungated pools on FULL bucket-0, compute oracle (cached-ref), save deliverables."""
import sys, os, time, hashlib, json, pickle
from collections import Counter
import numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS","5")
sys.path.insert(0,"solution"); sys.path.insert(0,"runs/b2")
from chrf import score_lists
from pool_builder import PoolBuilder

train = pd.read_csv("dataset/train.csv", keep_default_na=False)
def bk(s): return int(hashlib.md5(s.encode("utf-8","ignore")).hexdigest()[:8],16)%20
b=train.masked_docstring.map(bk)
trn=train[b!=0].reset_index(drop=True)
val=train[b==0].reset_index(drop=True)
refs=val.target_span.astype(str).values

pb=PoolBuilder(cap=80); pb.fit(trn)
t=time.time(); pools=pb.candidates_batch(val); dt=time.time()-t

def grams(s,nmax=6):
    c=Counter()
    for n in range(1,nmax+1):
        for i in range(len(s)-n+1): c[s[i:i+n]]+=1
    return c
def fmax(cands,rg,tr):
    best=0.0
    for c in cands:
        cg=grams(c); tp=sum(cg.values())
        if tp==0 or tr==0: continue
        m=sum(min(v,rg[k]) for k,v in cg.items())
        if m==0: continue
        p=m/tp; r=m/tr; f=2*p*r/(p+r)
        if f>best: best=f
    return best
orc=[]; hit=0; sizes=[]
for i in range(len(val)):
    cs=[t for t,_,_ in pools[i]]; sizes.append(len(cs))
    rg=grams(refs[i]); orc.append(fmax(cs,rg,sum(rg.values())))
    hit+=refs[i] in cs
orc=float(np.mean(orc)); hit=hit/len(val)
top1=[r[0][0] if r else "" for r in pools]

os.makedirs("runs/b2/out",exist_ok=True)
with open("runs/b2/out/val_pools.pkl","wb") as f:
    pickle.dump({"ids":val.id.tolist(),"refs":refs.tolist(),"pools":pools},f)
stats={"n_val_fullbucket0":len(val),"cap":80,"oracle_ungated":orc,
       "exact_hit":hit,"mean_pool":float(np.mean(sizes)),"max_pool":int(max(sizes)),
       "naive_top1_chrf":float(score_lists(top1,refs.tolist())),
       "fit_s":pb.t_fit,"build_fullbucket0_s":dt,
       "extrap50k_ungated_s":float(dt*50000/len(val))}
json.dump(stats,open("runs/b2/out/stats.json","w"),indent=2)
print(json.dumps(stats,indent=2),flush=True)
print(f"saved runs/b2/out/ ; build {dt:.1f}s for {len(val)} rows",flush=True)

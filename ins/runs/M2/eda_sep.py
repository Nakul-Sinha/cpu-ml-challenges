"""Feature separation TP vs FP for candidate reranker; test a reranker (LGBM) leak-free."""
import os, sys, json, re, collections
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import pipeline as P
train=pd.read_csv(os.path.join(P.ROOT,"dataset","train.csv"))
folds=pd.read_csv(os.path.join(P.ROOT,"solution","folds.csv"))
train=train.merge(folds,on="id"); train["edits"]=train.edits_json.apply(json.loads)
de=train[train.language=="de"]
MARKS=set(":*∗/")
def marked(s): return any(c in MARKS for c in s)
def lcp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i
def stem_ratio(a,b): return lcp(a.lower(),b.lower())/max(len(a),len(b),1)

# leak-free OOF token probs (from M1)
oof=pd.read_csv(os.path.join(P.ROOT,"runs","M1","oof_token_probs.csv"))
probmap={}
for _id,g in oof[oof.lang=="de"].groupby("id"):
    probmap[_id]={(row.start,row.end):row.proba for row in g.itertuples()}

def learn(df):
    interior=collections.Counter(); tot=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=src.split()
            if len(tks)>=2 and e["replacement"]!="":
                for i,t in enumerate(tks):
                    tc=t.strip(".,;:"); tot[tc]+=1
                    if 0<i<len(tks)-1: interior[tc]+=1
    conn=set(t for t,c in interior.items() if c>=3 and c/max(tot[t],1)>=0.7 and 2<=len(t)<=6 and t.islower())
    femsuf=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=[t.strip(".,;:") for t in src.split()]
            if len(tks)==3 and tks[1] in conn and not marked(src) and stem_ratio(tks[0],tks[2])>=0.5:
                cp=lcp(tks[0].lower(),tks[2].lower()); sa,sb=tks[0][cp:].lower(),tks[2][cp:].lower()
                femsuf[sa if len(sa)>=len(sb) else sb]+=1
    return conn,set(s for s,c in femsuf.items() if c>=2 and s)

def gen(tks,conn,femsuf):
    out=[]; n=len(tks); words=[w for _,_,w in tks]; cores=[w.strip(".,;:") for w in words]
    for i in range(1,n-1):
        if cores[i].lower() in conn:
            a,b=cores[i-1],cores[i+1]
            if not a or not b or not(a[:1].isupper() and b[:1].isupper()): continue
            cp=lcp(a.lower(),b.lower()); sa,sb=a[cp:].lower(),b[cp:].lower()
            fem=(sa in femsuf or sb in femsuf)
            if stem_ratio(a,b)>=0.5 or fem:
                si,ej=i-1,i+1; k=si-1
                while k-1>=0 and words[k].endswith(",") and stem_ratio(words[k].strip(".,;:"),a)>=0.4: si=k; k-=1
                out.append((si,ej,cores[i].lower(),fem,stem_ratio(a,b)))
    return out

# build candidate feature table with labels (leak-free)
def rowfeat(r,tks,si,ej,conn_type,fem,sr):
    a,b=tks[si][0],tks[ej][1]
    pm=probmap.get(r.id,{})
    sp=[pm.get((tks[t][0],tks[t][1]),0.0) for t in range(si,ej+1)]
    rowmax=max(pm.values()) if pm else 0.0
    return dict(sr=sr, fem=1.0 if fem else 0.0, ntok=ej-si+1,
                is_und=1.0 if conn_type=="äàw" else 0.0,
                spmax=max(sp) if sp else 0.0, spmean=float(np.mean(sp)) if sp else 0.0,
                rowmax=rowmax, lenA=tks[si][1]-tks[si][0], lenB=tks[ej][1]-tks[ej][0])
rows=[]
for r in de.itertuples():
    tks=P.toks(r.text)
    # need fold to learn leak-free; approximate: learn per fold below. store raw here.
    rows.append((r.id,r.fold,tks,[(e["start"],e["end"]) for e in r.edits if e["replacement"]!=""],len(r.edits)==0))

# per-fold: learn, gen, featurize, collect
import lightgbm as lgb
Xall=[]; yall=[]; meta=[]
FEATK=None
for k in range(5):
    conn,femsuf=learn(de[de.fold!=k])
    for (rid,fold,tks,truesp,isunch) in rows:
        if fold!=k: continue
        for (si,ej,ct,fem,sr) in gen(tks,conn,femsuf):
            a,b=tks[si][0],tks[ej][1]; best=0
            for (ts,te) in truesp:
                ov=max(0,min(b,te)-max(a,ts)); iou=ov/max(1,(max(b,te)-min(a,ts)))
                best=max(best,iou)
            rr=[x for x in rows if x[0]==rid][0]
            f=rowfeat(type("R",(),{"id":rid})(),tks,si,ej,ct,fem,sr)
            if FEATK is None: FEATK=sorted(f.keys())
            Xall.append([f[kk] for kk in FEATK]); yall.append(1 if best>=0.5 else 0)
            meta.append((rid,fold,isunch,best))
Xall=np.array(Xall); yall=np.array(yall)
print(f"candidates total={len(yall)} pos={yall.sum()} neg={(yall==0).sum()}")
print("feature means TP vs FP:")
for i,kk in enumerate(FEATK):
    print(f"  {kk:8s} TP={Xall[yall==1,i].mean():.3f}  FP={Xall[yall==0,i].mean():.3f}")

# leak-free reranker CV: train on 4 folds' candidates, predict on held fold
foldarr=np.array([m[1] for m in meta])
oof_score=np.zeros(len(yall))
for k in range(5):
    tr=foldarr!=k; va=foldarr==k
    if tr.sum()==0 or va.sum()==0: continue
    m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=10,reg_lambda=1.0,verbosity=-1,n_jobs=5)
    m.fit(Xall[tr],yall[tr]); oof_score[va]=m.predict_proba(Xall[va])[:,1]
# precision/recall at thresholds
for thr in (0.3,0.4,0.5,0.6,0.7):
    adm=oof_score>=thr
    tp=((adm)&(yall==1)).sum(); fp=((adm)&(yall==0)).sum()
    fp_unch=sum(1 for i in range(len(yall)) if adm[i] and yall[i]==0 and meta[i][2])
    print(f"  reranker thr={thr}: admit={adm.sum()} TP={tp} FP={fp} (FP_unch={fp_unch}) prec={tp/max(adm.sum(),1):.3f} recall_of_gen={tp/max(yall.sum(),1):.3f}")

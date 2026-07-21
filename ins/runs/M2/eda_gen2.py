"""Refine candidate generator: capitalization gate, FP placement (unchanged vs edited), connector fix."""
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

def learn(df):
    interior=collections.Counter(); tot=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=src.split()
            if len(tks)>=2 and e["replacement"]!="":
                for i,t in enumerate(tks):
                    tc=t.strip(".,;:"); tot[tc]+=1
                    if 0<i<len(tks)-1: interior[tc]+=1
    conn=set(t for t,c in interior.items() if c>=2 and c/max(tot[t],1)>=0.6 and 2<=len(t)<=6 and t.islower())
    femsuf=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=[t.strip(".,;:") for t in src.split()]
            if len(tks)==3 and tks[1] in conn and not marked(src) and stem_ratio(tks[0],tks[2])>=0.5:
                cp=lcp(tks[0].lower(),tks[2].lower())
                sa,sb=tks[0][cp:].lower(),tks[2][cp:].lower()
                femsuf[sa if len(sa)>=len(sb) else sb]+=1
    return conn,set(s for s,c in femsuf.items() if c>=2 and s)

def gen(tks,conn,femsuf,require_cap):
    cands=[]; n=len(tks); words=[w for _,_,w in tks]; cores=[w.strip(".,;:") for w in words]
    for i in range(1,n-1):
        if cores[i].lower() in conn:
            a,b=cores[i-1],cores[i+1]
            if not a or not b: continue
            if require_cap and not (a[:1].isupper() and b[:1].isupper()): continue
            cp=lcp(a.lower(),b.lower()); sa,sb=a[cp:].lower(),b[cp:].lower()
            if stem_ratio(a,b)>=0.5 or sa in femsuf or sb in femsuf:
                si,ej=i-1,i+1
                k=si-1
                while k-1>=0 and words[k].endswith(",") and stem_ratio(words[k].strip(".,;:"),a)>=0.4:
                    si=k; k-=1
                cands.append((si,ej))
    return cands

for require_cap in (False,True):
    tp=fp_unch=fp_ed=0; hit=0; tot_mp=0
    for k in range(5):
        tr=de[de.fold!=k]; va=de[de.fold==k]
        conn,femsuf=learn(tr)
        if k==0 and require_cap: print("conn:",conn,"| femsuf:",femsuf)
        for r in va.itertuples():
            tks=P.toks(r.text); cands=gen(tks,conn,femsuf,require_cap)
            truespans=[(e["start"],e["end"]) for e in r.edits if len(r.text[e["start"]:e["end"]].split())>=2 and not marked(r.text[e["start"]:e["end"]]) and e["replacement"]!=""]
            tot_mp+=len(truespans); isunch=len(r.edits)==0
            used=set()
            for (si,ej) in cands:
                a,b=tks[si][0],tks[ej][1]; best=0; bt=None
                for ti,(ts,te) in enumerate(truespans):
                    ov=max(0,min(b,te)-max(a,ts)); iou=ov/max(1,(max(b,te)-min(a,ts)))
                    if iou>best: best=iou; bt=ti
                if best>=0.5 and bt not in used: tp+=1; used.add(bt); hit+=1
                elif isunch: fp_unch+=1
                else: fp_ed+=1
    print(f"require_cap={require_cap}: TP={tp} recall={hit/max(tot_mp,1):.3f}({hit}/{tot_mp})  FP_unchanged={fp_unch}  FP_edited={fp_ed}  prec={tp/max(tp+fp_unch+fp_ed,1):.3f}")

"""Measure a paired-form candidate generator: precision/recall vs true de spans, leak-free per-fold."""
import os, sys, json, re, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P

train = pd.read_csv(os.path.join(P.ROOT, "dataset", "train.csv"))
folds = pd.read_csv(os.path.join(P.ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)
de = train[train.language == "de"]
MARKS = set(":*∗/")
def marked(s): return any(c in MARKS for c in s)
def lcp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i
def stem_ratio(a,b): return lcp(a.lower(),b.lower())/max(len(a),len(b),1)

def learn_connectors(df):
    interior=collections.Counter(); tot=collections.Counter()
    fem=collections.Counter(); base=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=src.split()
            if len(tks)>=2 and e["replacement"]!="":
                for i,t in enumerate(tks):
                    tc=t.strip(".,;:")
                    tot[tc]+=1
                    if 0<i<len(tks)-1: interior[tc]+=1
    conn=set(t for t,c in interior.items() if c>=3 and c/max(tot[t],1)>=0.75 and len(t)<=6 and t.islower())
    # learn feminine vs base suffixes from paired plain spans A conn B sharing stem
    femsuf=collections.Counter(); basesuf=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=[t.strip(".,;:") for t in src.split()]
            if len(tks)==3 and tks[1] in conn and not marked(src):
                a,_,b=tks
                if stem_ratio(a,b)>=0.5 and a and b:
                    cp=lcp(a.lower(),b.lower())
                    sa,sb=a[cp:].lower(),b[cp:].lower()
                    # feminine side = the longer/‑innen suffix
                    if len(sa)>=len(sb): femsuf[sa]+=1; basesuf[sb]+=1
                    else: femsuf[sb]+=1; basesuf[sa]+=1
    return conn, femsuf, basesuf

def gen_candidates(tokens, conn, femsuf, basesuf, text):
    """tokens: list of (s,e,w). emit (si,ej) inclusive token index candidate spans."""
    cands=[]
    n=len(tokens)
    words=[w for _,_,w in tokens]
    cores=[w.strip(".,;:") for w in words]
    for i in range(1,n-1):
        if cores[i].lower() in conn:
            a,b=cores[i-1],cores[i+1]
            if not a or not b: continue
            sr=stem_ratio(a,b)
            cp=lcp(a.lower(),b.lower()); sa,sb=a[cp:].lower(),b[cp:].lower()
            femcue=(sa in femsuf or sb in femsuf)
            if sr>=0.5 or femcue:
                si,ej=i-1,i+1
                # extend left over comma-listed same-stem tokens: "X, Y und Z"
                k=si-1
                while k-1>=0 and words[k].endswith(",") and stem_ratio(cores[k],a)>=0.4:
                    si=k; k-=1
                # candidate with and without trailing punctuation already in token
                cands.append((si,ej))
    return cands

# leak-free per-fold measurement
tp=fp=0; true_multi=0; matched_true=set()
per_bucket=collections.Counter(); hit_bucket=collections.Counter()
for k in range(5):
    tr=de[de.fold!=k]; va=de[de.fold==k]
    conn,femsuf,basesuf=learn_connectors(tr)
    if k==0: print("learned connectors:",conn); print("top femsuf:",femsuf.most_common(6),"basesuf:",basesuf.most_common(6))
    for r in va.itertuples():
        tks=P.toks(r.text)
        cands=gen_candidates(tks,conn,femsuf,basesuf,r.text)
        # true spans (multi-token, any)
        truespans=[]
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]
            if len(src.split())>=2 and e["replacement"]!="":
                b="multi_marked" if marked(src) else "multi_plain"
                truespans.append((e["start"],e["end"],b)); per_bucket[b]+=1
        used=set()
        for (si,ej) in cands:
            a,b=tks[si][0],tks[ej][1]
            best=0; bt=None
            for ti,(ts,te,tb) in enumerate(truespans):
                ov=max(0,min(b,te)-max(a,ts))
                iou=ov/max(1,(max(b,te)-min(a,ts)))
                if iou>best: best=iou; bt=ti
            if best>=0.5 and bt not in used:
                tp+=1; used.add(bt); hit_bucket[truespans[bt][2]]+=1
            else:
                fp+=1
print(f"\ncandidate gen (leak-free): TP={tp} FP={fp} precision={tp/max(tp+fp,1):.3f}")
print("true multi buckets:",dict(per_bucket))
print("hit (IoU>=.5) buckets:",dict(hit_bucket))
for b in per_bucket:
    print(f"  {b}: recall={hit_bucket[b]/max(per_bucket[b],1):.3f} ({hit_bucket[b]}/{per_bucket[b]})")

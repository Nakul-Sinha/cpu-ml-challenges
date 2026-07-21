"""Debug connector learning; measure missed multi_plain; test generator variants via inject-proxy."""
import os, sys, json, re, collections
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import pipeline as P, elru
train=pd.read_csv(os.path.join(P.ROOT,"dataset","train.csv"))
folds=pd.read_csv(os.path.join(P.ROOT,"solution","folds.csv"))
train=train.merge(folds,on="id"); train["edits"]=train.edits_json.apply(json.loads)
de=train[train.language=="de"]; by_id={r.id:r for r in train.itertuples()}
MARKS=set(":*∗/")
def marked(s): return any(c in MARKS for c in s)
def lcp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i
def sr(a,b): return lcp(a.lower(),b.lower())/max(len(a),len(b),1)

# --- why isn't "oder" learned? print interior counts per fold ---
def interior_counts(df):
    interior=collections.Counter(); tot=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=src.split()
            if len(tks)>=2 and e["replacement"]!="":
                for i,t in enumerate(tks):
                    tc=t.strip(".,;:"); tot[tc]+=1
                    if 0<i<len(tks)-1: interior[tc]+=1
    return interior,tot
inter,tot=interior_counts(de[de.fold!=0])
print("fold0-train interior top:",[(t,inter[t],tot[t]) for t in sorted(inter,key=lambda x:-inter[x])[:8]])
# candidate connectors under different rules
for lo in (2,3):
    conn=set(t for t,c in inter.items() if c>=lo and c/max(tot[t],1)>=0.6 and 2<=len(t)<=6 and t.islower())
    print(f"  c>={lo},frac>=.6:",conn)

# --- missed multi_plain: characterize the 55 misses (fold-agnostic full-corpus quick look) ---
inter,tot=interior_counts(de)
conn=set(t for t,c in inter.items() if c>=3 and c/max(tot[t],1)>=0.6 and 2<=len(t)<=6 and t.islower())
print("\nfull-corpus conn:",conn)
miss=[]
for r in de.itertuples():
    for e in r.edits:
        src=r.text[e["start"]:e["end"]]; tks=[t for t in src.split()]
        if len(tks)>=2 and not marked(src) and e["replacement"]:
            cores=[t.strip(".,;:") for t in tks]
            has_conn=any(c.lower() in conn for c in cores)
            cappair=any(cores[i].lower() in conn and i>0 and i<len(cores)-1 and cores[i-1][:1].isupper() and cores[i+1][:1].isupper() for i in range(len(cores)))
            if not cappair:
                miss.append((len(tks),has_conn,src[:50]))
print(f"multi_plain not-cap-pair-detectable: {len(miss)}")
c=collections.Counter((n,hc) for n,hc,_ in miss)
print("  by (ntok,has_conn):",dict(c))
for n,hc,s in miss[:18]: print(f"    ntok={n} conn={hc} {s!r}")

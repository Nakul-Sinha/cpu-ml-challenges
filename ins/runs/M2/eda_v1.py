"""Proxy-test generator variants: V0 adjacent, V1 skip intervening adj after connector.
Uses inject-into-M1-OOF proxy (tracked full pipeline within 0.001)."""
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

def learn(df):
    interior=collections.Counter(); tot=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=src.split()
            if len(tks)>=2 and e["replacement"]!="":
                for i,t in enumerate(tks):
                    tc=t.strip(".,;:"); tot[tc]+=1
                    if 0<i<len(tks)-1: interior[tc]+=1
    conn=set(t for t,c in interior.items() if c>=3 and c/max(tot[t],1)>=0.6 and 2<=len(t)<=6 and t.islower())
    femc=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=[t.strip(".,;:") for t in src.split()]
            if len(tks)==3 and tks[1] in conn and not marked(src) and sr(tks[0],tks[2])>=0.5:
                cp=lcp(tks[0].lower(),tks[2].lower()); a,b=tks[0][cp:].lower(),tks[2][cp:].lower()
                femc[a if len(a)>=len(b) else b]+=1
    femsuf=set(s for s,c in femc.items() if c>=2 and s)
    ex=collections.defaultdict(collections.Counter); stem=collections.defaultdict(collections.Counter)
    nc=lambda s: re.sub(r"\s+"," ",s.strip()).strip(".,;:")
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; rep=e["replacement"]; tks=src.split()
            if len(tks)>=2 and not marked(src) and rep:
                ex[nc(src)][nc(rep)]+=1
                base=tks[0].strip(".,;:").lower(); st=base[:max(4,int(len(base)*0.6))]
                if " " not in rep.strip(): stem[st][nc(rep)]+=1
    exM={k:c.most_common(1)[0][0] for k,c in ex.items()}
    stemM={k:c.most_common(1)[0][0] for k,c in stem.items() if sum(c.values())>=2 and c.most_common(1)[0][1]/sum(c.values())>=0.5}
    return conn,femsuf,exM,stemM

def gen(tks,conn,femsuf,skip):
    out=[]; n=len(tks); words=[w for _,_,w in tks]; cores=[w.strip(".,;:") for w in words]
    for i in range(1,n-1):
        if cores[i].lower() in conn:
            a=cores[i-1]
            # find right noun: adjacent, or skip up to `skip` intervening lowercase adj/articles
            for j in range(i+1,min(i+2+skip,n)):
                b=cores[j]
                if not b or not b[:1].isupper(): continue
                if not a[:1].isupper(): break
                cp=lcp(a.lower(),b.lower()); saa,sbb=a[cp:].lower(),b[cp:].lower()
                fem=(saa in femsuf or sbb in femsuf)
                if sr(a,b)>=0.5 or fem:
                    si,ej=i-1,j; k=si-1
                    while k-1>=0 and words[k].endswith(",") and sr(words[k].strip(".,;:"),a)>=0.4: si=k; k-=1
                    out.append((si,ej,fem));
                break
    return out

def collapse(src,exM,stemM):
    m=re.match(r"^(\s*)(.*?)([\s.,;:]*)$",src,re.S); lead,core,trail=m.group(1),m.group(2),m.group(3)
    key=re.sub(r"\s+"," ",core.strip()).strip(".,;:")
    if key in exM: return lead+exM[key]+trail
    base=core.split()[0].strip(".,;:").lower(); st=base[:max(4,int(len(base)*0.6))]
    if st in stemM: return lead+stemM[st]+trail
    return None

oof=pd.read_csv(os.path.join(P.ROOT,"runs","M1","oof_edits.csv"))
base_edits={r.id:json.loads(r.edits_json) for r in oof.itertuples()}
def score_de(pred):
    pm={r.id:pred[r.id] for r in de.itertuples()}; tm={r.id:by_id[r.id].edits for r in de.itertuples()}; lm={r.id:"de" for r in de.itertuples()}
    return elru.elru(pm,tm,lm,detail=True)[1]["de"]
d=score_de({r.id:base_edits.get(r.id,[]) for r in de.itertuples()}); print(f"BASE de lang={d['lang_score']:.4f}")

for skip in (0,1,2):
    pred={}
    for k in range(5):
        conn,femsuf,exM,stemM=learn(de[de.fold!=k])
        for r in de[de.fold==k].itertuples():
            tks=P.toks(r.text); ed=[dict(e) for e in base_edits.get(r.id,[])]
            for (si,ej,fem) in gen(tks,conn,femsuf,skip):
                if not fem: continue
                a,b=tks[si][0],tks[ej][1]
                rep=collapse(r.text[a:b],exM,stemM) or r.text[a:b]
                ed=[e for e in ed if (b<=e["start"] or e["end"]<=a)]
                ed.append({"start":a,"end":b,"replacement":rep[:160]})
            ed.sort(key=lambda e:e["start"])
            out=[]; pe=-1
            for e in sorted(ed,key=lambda x:x["start"]):
                if e["start"]>=pe and 0<=e["start"]<e["end"]<=len(r.text): out.append(e); pe=e["end"]
            pred[r.id]=out[:8]
    d=score_de(pred)
    print(f"skip={skip}: de lang={d['lang_score']:.4f} edited={d['edited_mean']:.4f} unchanged={d['unchanged_mean']:.4f} (delta={d['lang_score']-0.3528:+.4f})")

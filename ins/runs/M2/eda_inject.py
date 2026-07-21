"""Direct ELRU proxy: inject generated+collapsed paired-form candidates into M1 OOF de edits, re-score.
This uses the REAL elru scorer and REAL leak-free base predictions -> honest net effect."""
import os, sys, json, re, collections
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import pipeline as P
import elru
train=pd.read_csv(os.path.join(P.ROOT,"dataset","train.csv"))
folds=pd.read_csv(os.path.join(P.ROOT,"solution","folds.csv"))
train=train.merge(folds,on="id"); train["edits"]=train.edits_json.apply(json.loads)
de=train[train.language=="de"]
by_id={r.id:r for r in train.itertuples()}
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
    conn=set(t for t,c in interior.items() if c>=3 and c/max(tot[t],1)>=0.7 and 2<=len(t)<=6 and t.islower())
    femsuf=collections.Counter()
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; tks=[t.strip(".,;:") for t in src.split()]
            if len(tks)==3 and tks[1] in conn and not marked(src) and stem_ratio(tks[0],tks[2])>=0.5:
                cp=lcp(tks[0].lower(),tks[2].lower()); sa,sb=tks[0][cp:].lower(),tks[2][cp:].lower()
                femsuf[sa if len(sa)>=len(sb) else sb]+=1
    femsuf=set(s for s,c in femsuf.items() if c>=2 and s)
    # collapse memories
    ex=collections.defaultdict(collections.Counter); stem=collections.defaultdict(collections.Counter)
    def norm_core(s): return re.sub(r"\s+"," ",s.strip()).strip(".,;:")
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; rep=e["replacement"]
            tks=src.split()
            if len(tks)>=2 and not marked(src) and rep:
                ex[norm_core(src)][norm_core(rep)]+=1
                base=tks[0].strip(".,;:").lower(); st=base[:max(4,int(len(base)*0.6))]
                if " " not in rep.strip(): stem[st][norm_core(rep)]+=1
    exM={k:c.most_common(1)[0][0] for k,c in ex.items() if c.most_common(1)[0][1]>=1}
    stemM={k:c.most_common(1)[0][0] for k,c in stem.items() if sum(c.values())>=2 and c.most_common(1)[0][1]/sum(c.values())>=0.5}
    return conn,femsuf,exM,stemM

def gen(tks,conn,femsuf):
    out=[]; n=len(tks); words=[w for _,_,w in tks]; cores=[w.strip(".,;:") for w in words]
    for i in range(1,n-1):
        if cores[i].lower() in conn:
            a,b=cores[i-1],cores[i+1]
            if not a or not b or not(a[:1].isupper() and b[:1].isupper()): continue
            cp=lcp(a.lower(),b.lower()); sa,sb=a[cp:].lower(),b[cp:].lower()
            if stem_ratio(a,b)>=0.5 or sa in femsuf or sb in femsuf:
                si,ej=i-1,i+1; k=si-1
                while k-1>=0 and words[k].endswith(",") and stem_ratio(words[k].strip(".,;:"),a)>=0.4: si=k; k-=1
                out.append((si,ej))
    return out

def collapse(src,exM,stemM):
    m=re.match(r"^(\s*)(.*?)([\s.,;:]*)$",src,re.S)
    lead,core,trail=m.group(1),m.group(2),m.group(3)
    ncore=re.sub(r"\s+"," ",core.strip())
    key=ncore.strip(".,;:")
    if key in exM: return lead+exM[key]+trail
    base=core.split()[0].strip(".,;:").lower(); st=base[:max(4,int(len(base)*0.6))]
    if st in stemM: return lead+stemM[st]+trail
    return None

# load M1 leak-free OOF edits (base, at nonnested de thr 0.05)
oof=pd.read_csv(os.path.join(P.ROOT,"runs","M1","oof_edits.csv"))
base_edits={r.id:json.loads(r.edits_json) for r in oof.itertuples()}

def overlaps(a,b,edits):
    for e in edits:
        if not(b<=e["start"] or e["end"]<=a): return True
    return False

def score_de(pred):
    pm={r.id:pred[r.id] for r in de.itertuples()}
    tm={r.id:by_id[r.id].edits for r in de.itertuples()}
    lm={r.id:"de" for r in de.itertuples()}
    return elru.elru(pm,tm,lm,detail=True)

# baseline
base_de={r.id:base_edits.get(r.id,[]) for r in de.itertuples()}
s,d=score_de(base_de); print(f"BASE de: lang={d['de']['lang_score']:.4f} edited={d['de']['edited_mean']:.4f} unchanged={d['de']['unchanged_mean']:.4f}")

# inject (leak-free per fold), several policies
for policy in ["add_all","add_replace_overlap","fem_strict"]:
    pred={}
    for k in range(5):
        conn,femsuf,exM,stemM=learn(de[de.fold!=k])
        for r in de[de.fold==k].itertuples():
            tks=P.toks(r.text); ed=[dict(e) for e in base_edits.get(r.id,[])]
            for (si,ej) in gen(tks,conn,femsuf):
                a,b=tks[si][0],tks[ej][1]
                if policy=="fem_strict":
                    aa,bb=tks[si][2].strip(".,;:"),tks[ej][2].strip(".,;:")
                    cp=lcp(aa.lower(),bb.lower()); sa,sb=aa[cp:].lower(),bb[cp:].lower()
                    if not(sa in femsuf or sb in femsuf): continue
                rep=collapse(r.text[a:b],exM,stemM)
                if rep is None: rep=r.text[a:b]  # identity fallback (approx A2)
                if policy=="add_replace_overlap":
                    ed=[e for e in ed if (b<=e["start"] or e["end"]<=a)]
                    ed.append({"start":a,"end":b,"replacement":rep[:160]})
                else:
                    if not overlaps(a,b,ed): ed.append({"start":a,"end":b,"replacement":rep[:160]})
            ed.sort(key=lambda e:e["start"])
            if not elru.validate_edits(ed,len(r.text)):
                # drop overlaps
                out=[]; pe=-1
                for e in sorted(ed,key=lambda x:x["start"]):
                    if e["start"]>=pe and 0<=e["start"]<e["end"]<=len(r.text): out.append(e); pe=e["end"]
                ed=out[:8]
            pred[r.id]=ed
    s,d=score_de(pred)
    print(f"{policy:20s} de: lang={d['de']['lang_score']:.4f} edited={d['de']['edited_mean']:.4f} unchanged={d['de']['unchanged_mean']:.4f}  (delta_lang={d['de']['lang_score']-0.3528:+.4f})")

"""Proxy-test single_plain article slash-doubling: anchor on base-detected edits, add
preceding short-article token if it has a learned slash-double memory. Leak-free per fold."""
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

def learn_art(df):
    # short lowercase single-token edits -> majority rep (slash-double articles/pronouns)
    m=collections.defaultdict(collections.Counter)
    for r in df.itertuples():
        for e in r.edits:
            src=r.text[e["start"]:e["end"]]; rep=e["replacement"]
            if len(src.split())==1 and not marked(src) and rep:
                core=src.strip(".,;:()»«\"'")
                if core.isalpha() and core.islower() and len(core)<=5:
                    m[core][rep.strip()]+=1
    art={k:c.most_common(1)[0][0] for k,c in m.items() if sum(c.values())>=2 and c.most_common(1)[0][1]/sum(c.values())>=0.6}
    return art

oof=pd.read_csv(os.path.join(P.ROOT,"runs","M2","oof_edits.csv"))
base_edits={r.id:json.loads(r.edits_json) for r in oof.itertuples()}
def score_de(pred):
    pm={r.id:pred[r.id] for r in de.itertuples()}; tm={r.id:by_id[r.id].edits for r in de.itertuples()}; lm={r.id:"de" for r in de.itertuples()}
    return elru.elru(pm,tm,lm,detail=True)[1]["de"]
d0=score_de({r.id:base_edits.get(r.id,[]) for r in de.itertuples()})
print(f"M2-genonly de lang={d0['lang_score']:.4f} edited={d0['edited_mean']:.4f} unchanged={d0['unchanged_mean']:.4f}")

for require_mark_adj in (True,False):
    pred={}
    for k in range(5):
        art=learn_art(de[de.fold!=k])
        for r in de[de.fold==k].itertuples():
            tks=P.toks(r.text); ed=[dict(e) for e in base_edits.get(r.id,[])]
            # index tokens by start
            occupied=set()
            for e in ed:
                for (s,en,w) in tks:
                    if s>=e["start"] and en<=e["end"]: occupied.add(s)
            for ti,(s,en,w) in enumerate(tks):
                core=w.strip(".,;:()»«\"'")
                if s in occupied: continue
                if core in art:
                    # adjacency gate: next or prev token is marked OR is inside a base edit
                    nxt=tks[ti+1] if ti+1<len(tks) else None
                    prv=tks[ti-1] if ti-1>=0 else None
                    adj = (nxt and marked(nxt[2])) or (prv and marked(prv[2]))
                    adj = adj or (nxt and any(nxt[0]>=e["start"] and nxt[1]<=e["end"] for e in ed))
                    if require_mark_adj and not adj: continue
                    a,b=s,en
                    if any(not(b<=e["start"] or e["end"]<=a) for e in ed): continue
                    ed.append({"start":a,"end":b,"replacement":art[core][:160]})
            out=[]; pe=-1
            for e in sorted(ed,key=lambda x:x["start"]):
                if e["start"]>=pe and 0<=e["start"]<e["end"]<=len(r.text): out.append(e); pe=e["end"]
            pred[r.id]=out[:8]
    d=score_de(pred)
    print(f"require_mark_adj={require_mark_adj}: de lang={d['lang_score']:.4f} edited={d['edited_mean']:.4f} unchanged={d['unchanged_mean']:.4f} (delta={d['lang_score']-d0['lang_score']:+.4f})")

"""German EDA v2: connectors, paired-form structure, collapse consistency, detection gaps."""
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

# ---- 1) connector discovery: interior-token rate in multi-token edited spans ----
interior = collections.Counter(); first = collections.Counter(); last = collections.Counter()
alltok = collections.Counter()
for r in de.itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        tks = src.split()
        if len(tks) >= 2 and e["replacement"] != "":
            for i, t in enumerate(tks):
                alltok[t.strip(".,;:")] += 1
                if i == 0: first[t.strip(".,;:")] += 1
                elif i == len(tks)-1: last[t.strip(".,;:")] += 1
                else: interior[t.strip(".,;:")] += 1
print("=== top INTERIOR tokens of multi-token de spans (connector candidates) ===")
for t, c in interior.most_common(12):
    print(f"  {t!r:20s} interior={c}  total={alltok[t]}  interior_frac={c/max(alltok[t],1):.2f}")

# ---- 2) plain paired-form structure: tokA CONN tokB, stem overlap, collapse target ----
def lcp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i
print("\n=== plain paired forms (3-tok, A CONN B), collapse consistency ===")
collapse = collections.defaultdict(collections.Counter)
n3 = 0
for r in de.itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
        tks = src.split()
        if len(tks)==3 and not marked(src) and rep:
            a,conn,b = tks
            ac=a.strip(".,;:"); bc=b.strip(".,;:")
            sp = lcp(ac.lower(), bc.lower())/max(len(ac),len(bc),1)
            collapse[src.strip()][rep] += 1
            n3 += 1
print(f"  {n3} three-token plain spans")
# how consistent is the collapse for a repeated source form?
rep_src = [(s,c) for s,c in collapse.items() if sum(c.values())>=2]
print(f"  {len(rep_src)} distinct 3-tok src forms seen >=2x; consistency:")
consistent=0
for s,c in sorted(rep_src, key=lambda x:-sum(x[1].values()))[:15]:
    top,n = c.most_common(1)[0]; tot=sum(c.values())
    if n==tot: consistent+=1
    print(f"    n={tot} consistent={n==tot}  {s!r:45s} -> {top!r}")
print(f"  fully-consistent among repeated: {consistent}/{len(rep_src)}")

# ---- 3) stem-level neutral lexeme: map gendered stem -> collapse target ----
print("\n=== stem->collapse mapping (paired plain, learn neutral lexeme) ===")
stemmap = collections.defaultdict(collections.Counter)
for r in de.itertuples():
    for e in r.edits:
        src=r.text[e["start"]:e["end"]]; rep=e["replacement"]
        tks=src.split()
        if len(tks) in (2,3,4) and not marked(src) and rep and " " not in rep.strip():
            # single-word collapse target
            base = tks[0].strip(".,;:").lower()
            stem = base[:max(4,int(len(base)*0.6))]
            stemmap[stem][rep.strip()] += 1
nmap = sum(1 for s,c in stemmap.items())
print(f"  {nmap} stems map to single-word collapse; sample:")
for s,c in sorted(stemmap.items(), key=lambda x:-sum(x[1].values()))[:12]:
    top,n=c.most_common(1)[0]
    print(f"    stem={s!r:14s} -> {top!r} (n={n}/{sum(c.values())})")

# ---- 4) single_plain article/pronoun slash-doubling ----
print("\n=== single_plain short-token (<=5 char lowercase) slash-doubling ===")
artmap = collections.defaultdict(collections.Counter)
for r in de.itertuples():
    for e in r.edits:
        src=r.text[e["start"]:e["end"]]; rep=e["replacement"]
        if len(src.split())==1 and not marked(src) and rep:
            core = src.strip(".,;:()»«\"'")
            if core.islower() and len(core)<=5:
                artmap[core][rep.strip()] += 1
for s,c in sorted(artmap.items(), key=lambda x:-sum(x[1].values()))[:20]:
    top,n=c.most_common(1)[0]
    print(f"    {s!r:8s} -> {top!r:16s} (n={n}/{sum(c.values())})  slash={'/' in top}")

# ---- 5) DETECTION GAP: at thr 0.05, what fraction of multi_plain span tokens fire? ----
print("\n=== detection gap on de multi_plain (needs OOF token probs) ===")
oof = pd.read_csv(os.path.join(P.ROOT, "runs", "M1", "oof_token_probs.csv"))
oof_de = oof[oof.lang=="de"]
prob_by_id = {}
for _id, g in oof_de.groupby("id"):
    prob_by_id[_id] = {(row.start,row.end): row.proba for row in g.itertuples()}
def frac_fire(r, thr):
    pm = prob_by_id.get(r.id, {})
    hits=tot=0
    for e in r.edits:
        src=r.text[e["start"]:e["end"]];
        if len(src.split())>=2 and not marked(src) and e["replacement"]:
            for m in re.finditer(r"\S+", src):
                s,en=e["start"]+m.start(), e["start"]+m.end()
                tot+=1; hits += 1 if pm.get((s,en),0)>=thr else 0
    return hits,tot
for thr in (0.05,0.1,0.2):
    H=T=0
    for r in de.itertuples():
        h,t=frac_fire(r,thr); H+=h; T+=t
    print(f"  thr={thr}: multi_plain tokens firing {H}/{T} = {H/max(T,1):.2f}")

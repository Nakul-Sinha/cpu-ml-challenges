"""Measure edit-rate of precise structural inclusive-form pattern per language.
Pattern: interior special char [: * / _] followed by a short (<=6) run to end, OR trailing special.
Decides whether a high-precision structural rule is viable vs. genuinely ambiguous."""
import sys, json, re, collections
import numpy as np, pandas as pd
sys.path.insert(0, "solution")
from elru import elru

WORD_RE = re.compile(r"\S+")
train = pd.read_csv("dataset/train.csv")
train["edits"] = train.edits_json.apply(json.loads)

def toks(t): return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(t)]

SPEC = set(":*/_")
def spec_suffix(w):
    p = -1
    for i in range(1, len(w)-1):
        if w[i] in SPEC: p = i
    if p == -1: return None
    return w[p], w[p+1:]

# For each language, group structural tokens by (special_char, suffix) and report edit rate + support
for L in ["de","it","en"]:
    d = train[train.language==L]
    stat = collections.defaultdict(lambda: [0,0])  # key -> [edited, seen]
    for r in d.itertuples():
        spanset = {(e["start"], e["end"]) for e in r.edits}
        for s,e,w in toks(r.text):
            sk = spec_suffix(w)
            if sk is None: continue
            suf = sk[1]
            if not (1 <= len(suf) <= 6): continue
            key = sk[0] + "|" + suf
            stat[key][1] += 1
            if (s,e) in spanset: stat[key][0] += 1
    # report keys with support>=3 sorted by seen
    rows = sorted([(k,v[0],v[1],v[0]/v[1]) for k,v in stat.items() if v[1]>=3], key=lambda x:-x[2])
    hi = [r for r in rows if r[3]>=0.8]
    print(f"\n=== {L}: {len(rows)} (char,suf) patterns w/ support>=3; {len(hi)} have edit-rate>=0.8 ===")
    for k,ed,sn,rt in rows[:12]:
        print(f"   {k!r}: {ed}/{sn} = {rt:.2f}")
    tot_ed = sum(r[1] for r in rows); tot_sn = sum(r[2] for r in rows)
    hi_ed = sum(r[1] for r in hi); hi_sn = sum(r[2] for r in hi)
    print(f"   ALL struct-suffix tokens: {tot_ed}/{tot_sn} edited={tot_ed/max(tot_sn,1):.2f}")
    print(f"   in >=0.8 patterns: {hi_ed}/{hi_sn}  (precision if we fire all these: {hi_ed/max(hi_sn,1):.2f})")

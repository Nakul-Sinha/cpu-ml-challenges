# Analyze the deletion contexts: is the deleted span a gendered alt whose paired/neutral
# duplicate already exists ADJACENT in the window?
import json, re, collections
import pandas as pd
WS = re.compile(r"\S+")
_STRIP = ".,;:()»«\"'“”’`-–—"
tr = pd.read_csv("train.csv")
tr["edits"] = tr.edits_json.apply(json.loads)

def lcp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i

dels = []
for r in tr.itertuples():
    for e in r.edits:
        if e["replacement"] != "": continue
        s,en = e["start"], e["end"]
        span = r.text[s:en]
        # window: text around
        left = r.text[max(0,s-60):s]
        right = r.text[en:en+60]
        dels.append((r.language, span, left, right, r.id))

print(f"total deletions: {len(dels)}  (de={sum(1 for d in dels if d[0]=='de')} it={sum(1 for d in dels if d[0]=='it')})")

# characterize: does the deleted core share a long prefix with a neighboring token
# (left or right) -> "duplicate paired form adjacent"?
def toks_sp(t): return [(m.start(),m.end(),m.group()) for m in WS.finditer(t)]
n_dupleft=n_dupright=n_slash=n_nostem=0
for lang,span,left,right,rid in dels:
    core = span.strip(_STRIP).lower()
    ltoks=[w.strip(_STRIP).lower() for _,_,w in toks_sp(left)]
    rtoks=[w.strip(_STRIP).lower() for _,_,w in toks_sp(right)]
    first_core = span.split()[0].strip(_STRIP).lower() if span.split() else ""
    last_core = span.split()[-1].strip(_STRIP).lower() if span.split() else ""
    # nearest neighbor tokens
    ln = ltoks[-1] if ltoks else ""
    rn = rtoks[0] if rtoks else ""
    dl = lcp(first_core, ln) if ln else 0
    dr = lcp(last_core, rn) if rn else 0
    has_slash = "/" in span
    if dl>=3: n_dupleft+=1
    if dr>=3: n_dupright+=1
    if has_slash: n_slash+=1
    if dl<3 and dr<3: n_nostem+=1

print(f"deleted span shares >=3 prefix w/ LEFT neighbor: {n_dupleft}")
print(f"deleted span shares >=3 prefix w/ RIGHT neighbor: {n_dupright}")
print(f"deleted span contains slash: {n_slash}")
print(f"NO stem-dup on either side: {n_nostem}")

print("\n=== DELETION EXAMPLES (lang | ...left >>[DEL]<< right...) ===")
for lang,span,left,right,rid in dels[:40]:
    print(f"  {lang} ...{left[-40:]!r} >>[{span!r}]<< {right[:40]!r}")

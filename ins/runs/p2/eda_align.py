# Analyze aligned nsrc==nrep multi edits: per-token transform structure
import json, re, collections
import pandas as pd
WS = re.compile(r"\S+")
MARKS = set(":*∗/")
_STRIP = ".,;:()»«\"'“”’`-–—"
tr = pd.read_csv("train.csv")
tr["edits"] = tr.edits_json.apply(json.loads)
def toks(s): return [m.group() for m in WS.finditer(s)]
def lcp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i

# For aligned nsrc==nrep, categorize each token transform
cats = collections.Counter()   # (lang, transform_kind)
append_suffix = collections.Counter()  # (lang, src_last2 -> appended)  for append cases
n_aligned = collections.Counter()
lex_examples = collections.defaultdict(list)
for r in tr.itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
        if rep=="" : continue
        st=toks(src); rt=toks(rep)
        if len(st)<2 or len(st)!=len(rt): continue
        n_aligned[(r.language, len(st))]+=1
        for a,b in zip(st,rt):
            ca=a.strip(_STRIP); cb=b.strip(_STRIP)
            if ca==cb:
                cats[(r.language,"identical")]+=1
            elif cb.startswith(ca):  # append
                cats[(r.language,"append")]+=1
                app=cb[len(ca):]
                append_suffix[(r.language, ca[-2:], app)]+=1
            elif ca and cb and lcp(ca,cb)>=max(2,len(ca)-4):  # suffix-swap
                cats[(r.language,"suffix_swap")]+=1
            elif any(c in MARKS for c in a):
                cats[(r.language,"has_mark")]+=1
            else:
                cats[(r.language,"lexical")]+=1
                if len(lex_examples[r.language])<8:
                    lex_examples[r.language].append((a,b))

print("=== aligned nsrc==nrep counts (lang, ntok) ===")
for k in sorted(n_aligned): print(f"  {k[0]} n={k[1]}: {n_aligned[k]} edits")
print("\n=== per-token transform kind (aligned only) ===")
for k in sorted(cats): print(f"  {k[0]} {k[1]:12s} {cats[k]}")
print("\n=== IT append-suffix map (src_last2 -> appended) top ===")
for k,v in sorted([(k,v) for k,v in append_suffix.items() if k[0]=='it'], key=lambda x:-x[1])[:25]:
    print(f"  it '{k[1]}' -> '{k[2]}'  x{v}")
print("\n=== DE append-suffix map top ===")
for k,v in sorted([(k,v) for k,v in append_suffix.items() if k[0]=='de'], key=lambda x:-x[1])[:15]:
    print(f"  de '{k[1]}' -> '{k[2]}'  x{v}")

# EDA: multi-token edits, alignment structure, per-language
import json, re, collections
import pandas as pd
WS = re.compile(r"\S+")
MARKS = set(":*∗/")
_STRIP = ".,;:()»«\"'“”’`-–—"

tr = pd.read_csv("train.csv")
tr["edits"] = tr.edits_json.apply(json.loads)

def toks(s): return [m.group() for m in WS.finditer(s)]

# categorize edits
cat = collections.Counter()
multi_examples = collections.defaultdict(list)
n_tok_rep_ratio = collections.defaultdict(list)
for r in tr.itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
        st = toks(src); rt = toks(rep)
        has_mark = any(c in MARKS for c in src)
        if rep == "":
            cat[(r.language, "deletion")] += 1
            continue
        if len(st) == 1:
            cat[(r.language, "single_mark" if has_mark else "single_plain")] += 1
        else:
            key = "multi_mark" if has_mark else "multi_plain"
            cat[(r.language, key)] += 1
            n_tok_rep_ratio[(r.language, key)].append((len(st), len(rt)))
            if len(multi_examples[(r.language, key)]) < 12:
                multi_examples[(r.language, key)].append((src, rep, len(st), len(rt)))

print("=== EDIT CATEGORY COUNTS (lang, type) ===")
for k in sorted(cat): print(f"  {k[0]} {k[1]:14s} {cat[k]}")

print("\n=== src_ntok -> rep_ntok distribution (multi) ===")
for k in sorted(n_tok_rep_ratio):
    c = collections.Counter(n_tok_rep_ratio[k])
    print(f"  {k[0]} {k[1]}: {dict(sorted(c.items()))}")

for lang in ("it","de"):
    for typ in ("multi_plain","multi_mark"):
        exs = multi_examples[(lang,typ)]
        if not exs: continue
        print(f"\n=== {lang} {typ} EXAMPLES (src -> rep) [nsrc,nrep] ===")
        for src,rep,ns,nr in exs:
            print(f"  [{ns},{nr}] {src!r}")
            print(f"        -> {rep!r}")

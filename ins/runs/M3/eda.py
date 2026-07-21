"""M3 EDA: understand Italian NP-agreement, deletions, EN plain-token patterns.
Throwaway diagnostic -- print structure so we can design LEARNED extractors.
"""
import os, sys, json, re, collections
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for base in [".", "..", ROOT, os.path.join(HERE, "..", "..")]:
    if os.path.exists(os.path.join(base, "dataset", "train.csv")):
        ROOT = os.path.abspath(base); break
train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)

WS = re.compile(r"\S+")
MARKS = set(":*∗/")

def toks(text):
    return [(m.start(), m.end(), m.group()) for m in WS.finditer(text)]

# classify each edit span
def classify(src, rep):
    ntok = len(src.split())
    marked = any(c in src for c in MARKS)
    if rep == "":
        return "deletion"
    if ntok == 1:
        return "single_marked" if marked else "single_plain"
    return "multi_marked" if marked else "multi_plain"

by_lang_type = collections.Counter()
for r in train.itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        by_lang_type[(r.language, classify(src, e["replacement"]))] += 1
print("=== edit-type counts by lang ===")
for k in sorted(by_lang_type):
    print(f"  {k}: {by_lang_type[k]}")

# ---- Italian deep-dive ----
print("\n=== ITALIAN single_plain examples (src -> rep) ===")
it_sp = []
for r in train[train.language == "it"].itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        if classify(src, e["replacement"]) == "single_plain":
            it_sp.append((src, e["replacement"]))
for s, rp in it_sp[:25]:
    print(f"   {s!r:30} -> {rp!r}")
print(f"   ... total it single_plain = {len(it_sp)}")

print("\n=== ITALIAN multi_plain examples (src -> rep) ===")
it_mp = []
for r in train[train.language == "it"].itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        if classify(src, e["replacement"]) == "multi_plain":
            it_mp.append((src, e["replacement"]))
for s, rp in it_mp[:25]:
    print(f"   {s!r:35} -> {rp!r}")
print(f"   ... total it multi_plain = {len(it_mp)}")

print("\n=== ITALIAN marked examples ===")
it_mk = []
for r in train[train.language == "it"].itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        if any(c in src for c in MARKS) and e["replacement"] != "":
            it_mk.append((src, e["replacement"]))
for s, rp in it_mk[:20]:
    print(f"   {s!r:35} -> {rp!r}")
print(f"   ... total it marked-nonempty = {len(it_mk)}")

# ---- Italian: what token PRECEDES an it edit span (article hypothesis) ----
print("\n=== ITALIAN: token preceding edited spans (article candidates) ===")
prev_ct = collections.Counter()
first_tok_ct = collections.Counter()
for r in train[train.language == "it"].itertuples():
    tk = toks(r.text)
    starts = {e["start"] for e in r.edits}
    for i, (s, e, w) in enumerate(tk):
        if s in starts:
            first_tok_ct[w.lower()] += 1
            if i > 0:
                prev_ct[tk[i-1][2].lower()] += 1
print("  span-initial tokens (top 25):")
for w, c in first_tok_ct.most_common(25):
    print(f"     {w!r:20} {c}")

# ---- Italian endings: last char-class of edited single tokens ----
print("\n=== ITALIAN single-token edit: ending char (last 1-2) edit-rate ===")
end_ed = collections.Counter(); end_tot = collections.Counter()
for r in train[train.language == "it"].itertuples():
    tk = toks(r.text)
    edset = []
    for e in r.edits:
        if len(r.text[e["start"]:e["end"]].split()) == 1:
            edset.append((e["start"], e["end"]))
    edstarts = {s for s, _ in edset}
    for s, e, w in tk:
        core = w.strip(".,;:()»«\"'").lower()
        if len(core) < 2:
            continue
        end2 = core[-2:]
        end_tot[end2] += 1
        if s in edstarts:
            end_ed[end2] += 1
print("  ending -> edit_rate (only endings with >=15 occ, rate>0.15):")
rows = []
for end, tot in end_tot.items():
    if tot >= 15:
        rate = end_ed[end] / tot
        if rate > 0.15:
            rows.append((rate, end, end_ed[end], tot))
for rate, end, ed, tot in sorted(rows, reverse=True)[:30]:
    print(f"     {end!r:6} rate={rate:.3f}  ({ed}/{tot})")

# ---- DELETIONS deep-dive ----
print("\n=== DELETIONS (rep=='') by lang ===")
del_ex = collections.defaultdict(list)
for r in train.itertuples():
    for e in r.edits:
        if e["replacement"] == "":
            src = r.text[e["start"]:e["end"]]
            ctx_l = r.text[max(0, e["start"]-40):e["start"]]
            ctx_r = r.text[e["end"]:e["end"]+40]
            del_ex[r.language].append((src, ctx_l, ctx_r))
for lang, lst in del_ex.items():
    print(f"  --- {lang}: {len(lst)} deletions ---")
    for src, cl, cr in lst[:12]:
        print(f"     src={src!r}")
        print(f"        Lctx=...{cl!r}")
        print(f"        Rctx={cr!r}...")

# ---- EN deep-dive ----
print("\n=== EN all edits (src -> rep) ===")
for r in train[train.language == "en"].itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        t = classify(src, e["replacement"])
        print(f"   [{t}] {src!r:30} -> {e['replacement']!r}")

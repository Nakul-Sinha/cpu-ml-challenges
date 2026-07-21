"""M3 EDA2: EN full list, IT article lexicon, slash-order convention, deletion connector precision."""
import os, sys, json, re, collections
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for base in [".", "..", ROOT]:
    if os.path.exists(os.path.join(base, "dataset", "train.csv")):
        ROOT = os.path.abspath(base); break
train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
train = train.merge(folds, on="id")
train["edits"] = train.edits_json.apply(json.loads)
WS = re.compile(r"\S+")
MARKS = set(":*∗/")
def toks(t): return [(m.start(), m.end(), m.group()) for m in WS.finditer(t)]

# ---- FULL EN edit list ----
print("=== FULL EN edits ===")
for r in train[train.language == "en"].itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        print(f"   {src!r:28} -> {e['replacement']!r}")

# ---- IT article lexicon: tokens that appear as span-initial AND get slash-doubled ----
print("\n=== IT: does span-initial token get a slash-double replacement? (article detection) ===")
# an 'article' = a short token that starts many edited spans and whose rep is 'X/Y' (contains /)
art_slashrep = collections.Counter(); art_total = collections.Counter()
for r in train[train.language == "it"].itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        parts = src.split()
        rparts = e["replacement"].split()
        if not parts or not rparts: continue
        w0 = parts[0].lower().strip(".,;:()»«\"'")
        art_total[w0] += 1
        if "/" in rparts[0]:
            art_slashrep[w0] += 1
print("  token -> (slash-doubled-as-first / total-as-span-first), top by total:")
for w, c in art_total.most_common(30):
    print(f"     {w!r:16} {art_slashrep[w]}/{c}")

# ---- slash-order convention within groups: is order consistent per group? ----
print("\n=== IT slash-order convention: for tokens ending -o/-a, which order (masc/fem vs fem/masc)? ===")
# find single-token reps of the form STEM+X/Y where we can see order
order_by_group = collections.defaultdict(lambda: collections.Counter())
global_order = collections.Counter()
for r in train[train.language == "it"].itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]
        rep = e["replacement"]
        if len(src.split()) != 1 or "/" not in rep or " " in rep: continue
        # rep like 'xtèyaqèdètl/g' or 'xtèyaqèdètg/l' -> look at char right before '/'
        si = rep.index("/")
        left_end = rep[si-1] if si > 0 else ""
        right = rep[si+1:si+2]
        # masc marker in italian ~ 'g'(o) / 'è'(i); fem ~ 'l'(a) / 'y'(e). Use raw chars.
        order_by_group[r.document_group][(left_end, right)] += 1
        global_order[(left_end, right)] += 1
print("  global (left_char_before_slash, right_char_after) counts, top 15:")
for k, c in global_order.most_common(15):
    print(f"     {k}: {c}")
# per-group consistency: for groups with >=3 slash edits, is one order dominant?
consistent = 0; total_g = 0
for g, cnt in order_by_group.items():
    if sum(cnt.values()) >= 3:
        total_g += 1
        top = cnt.most_common(1)[0][1]
        if top / sum(cnt.values()) >= 0.7:
            consistent += 1
print(f"  groups w/ >=3 slash-edits: {total_g}, of which >=70% single-order: {consistent}")

# ---- deletion connector-based precision estimate ----
print("\n=== IT/DE deletion: do deletions start with a 'connector' token & duplicate nearby? ===")
for lang in ("it", "de"):
    del_srcs = []
    for r in train[train.language == lang].itertuples():
        for e in r.edits:
            if e["replacement"] == "":
                del_srcs.append(r.text[e["start"]:e["end"]])
    firsttok = collections.Counter(s.split()[0].lower() for s in del_srcs if s.split())
    print(f"  {lang}: {len(del_srcs)} deletions; first-token of deletion span:")
    for w, c in firsttok.most_common(8):
        print(f"     {w!r:12} {c}")

# ---- how many it edited rows have the whole NP as ONE span vs multiple edits? ----
print("\n=== IT edited-row edit-count distribution ===")
ec = collections.Counter()
for r in train[train.language == "it"].itertuples():
    if r.edits: ec[len(r.edits)] += 1
for k in sorted(ec): print(f"   {k} edits: {ec[k]} rows")

# ---- IT: fraction of edited single tokens whose rep = src + '/' + suffix-variant (slash-append) ----
print("\n=== IT single_plain rep shapes ===")
shape = collections.Counter()
for r in train[train.language == "it"].itertuples():
    for e in r.edits:
        src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
        if len(src.split()) != 1 or any(c in src for c in MARKS) or rep == "": continue
        core = src.strip(".,;:()»«\"'")
        if rep.startswith(core) and "/" in rep[len(core):]:
            shape["src+/suffix (slash-append after full src)"] += 1
        elif "/" in rep:
            lcp = 0
            while lcp < min(len(core), len(rep)) and core[lcp]==rep[lcp]: lcp+=1
            shape[f"slash-with-stem-change"] += 1
        elif rep.endswith(core) or core.endswith(rep):
            shape["contained (no slash)"] += 1
        else:
            shape["other"] += 1
for k, c in shape.most_common(): print(f"   {k}: {c}")

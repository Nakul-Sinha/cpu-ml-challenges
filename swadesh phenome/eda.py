import pandas as pd, numpy as np, sys, io
from collections import Counter, defaultdict

out = io.open("eda_out.txt", "w", encoding="utf-8")
def p(*a):
    print(*a)
    print(*a, file=out)

tr = pd.read_csv("dataset/public/train.csv")
te = pd.read_csv("dataset/public/test.csv")

p("=== TRAIN ===")
p("rows:", len(tr), "langs:", tr.language.nunique(), "families:", tr.family.nunique(),
  "subfamilies:", tr.subfamily.nunique(), "concepts:", tr.concept.nunique())

ura = tr[tr.family == "Uralic"]
p("\n=== URALIC subfamilies/langs ===")
p(ura.groupby(["subfamily","language"]).size().to_string())

# inventory size per language
p("\n=== inventory size per language (distinct segments) ===")
inv = {}
for lang, g in tr.groupby("language"):
    segs = Counter()
    for s in g.ipa: segs.update(str(s).split())
    inv[lang] = len(segs)
invs = pd.Series(inv).sort_values(ascending=False)
p("global inv range:", invs.min(), "-", invs.max())
p("\nUralic inventory sizes (sorted):")
ura_langs = sorted(ura.language.unique())
for l in sorted(ura_langs, key=lambda x: -inv[x]):
    sf = ura[ura.language==l].subfamily.iloc[0]
    p(f"  {l} ({sf}): {inv[l]}")

# test token count = 70. Which uralic langs have inventory near 70?
p("\n=== TEST cipher stats ===")
toks = Counter()
for c in te.cipher: toks.update(c.split())
p("distinct tokens:", len(toks), "total occ:", sum(toks.values()))
freq = sorted(toks.values(), reverse=True)
p("token freq sorted:", freq)
cum = np.cumsum(freq)/sum(freq)
p("cum coverage by rank: top10=%.3f top20=%.3f top30=%.3f top40=%.3f top50=%.3f"%(
    cum[9],cum[19],cum[29],cum[39],cum[49]))

# uralic union inventory
useg = Counter()
for s in ura.ipa: useg.update(str(s).split())
p("\n=== URALIC union inventory ===")
p("distinct segments:", len(useg))
p("segments by freq:")
line=[]
for s,c in useg.most_common():
    line.append(f"{s}:{c}")
p("  " + "  ".join(line))

# how many uralic segments appear in >=2 langs (robust inventory)
seg_langs = defaultdict(set)
for lang,g in ura.groupby("language"):
    for s in g.ipa:
        for seg in str(s).split(): seg_langs[seg].add(lang)
robust = {s:len(ls) for s,ls in seg_langs.items()}
p("\nUralic segments in >=3 langs:", sum(1 for v in robust.values() if v>=3))
p("Uralic segments in >=5 langs:", sum(1 for v in robust.values() if v>=5))

# global union inventory size
gseg = Counter()
for s in tr.ipa: gseg.update(str(s).split())
p("\nGLOBAL distinct segments:", len(gseg))

# multi-char segments
multi = [s for s in useg if len(s)>1]
p("\nUralic multi-char segments (%d):"%len(multi), "  ".join(sorted(multi)))

# test word length distribution
wl = te.cipher.str.split().apply(len)
p("\ntest word lengths: min %d max %d mean %.2f"%(wl.min(),wl.max(),wl.mean()))
p("length hist:", dict(sorted(Counter(wl).items())))

# concept synonyms in test
csyn = te.groupby("concept").size()
p("\ntest concepts with multiple rows (synonyms):", (csyn>1).sum())
p("max rows for one concept:", csyn.max())

out.close()
print("\n[written to eda_out.txt]")

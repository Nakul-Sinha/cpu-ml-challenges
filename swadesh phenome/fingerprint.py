"""Which Uralic subfamily is the real target closest to? Fingerprint by per-concept
word-length agreement between the enciphered target and each support language (honest:
uses only lengths, no decoding). The closest relatives should top the ranking."""
import numpy as np, pandas as pd
from collections import defaultdict

tr = pd.read_csv("dataset/public/train.csv"); tr.ipa = tr.ipa.fillna("")
te = pd.read_csv("dataset/public/test.csv")

# target: concept -> list of cipher lengths
tgt = defaultdict(list)
for c, cip in zip(te.concept, te.cipher):
    tgt[c].append(len(str(cip).split()))

# per language: concept -> mean form length
rows = []
for lang, g in tr.groupby("language"):
    fam = g.family.iloc[0]; sf = g.subfamily.iloc[0]
    llen = defaultdict(list)
    for c, ipa in zip(g.concept, g.ipa):
        if str(ipa).split():
            llen[c].append(len(str(ipa).split()))
    # compare on shared concepts
    xs, ys = [], []
    for c, tl in tgt.items():
        if c in llen:
            xs.append(np.mean(tl)); ys.append(np.mean(llen[c]))
    xs, ys = np.array(xs), np.array(ys)
    corr = np.corrcoef(xs, ys)[0, 1] if len(xs) > 5 else np.nan
    mae = np.mean(np.abs(xs - ys))
    rows.append((lang, fam, sf, len(xs), corr, mae))

df = pd.DataFrame(rows, columns=["lang", "family", "subfam", "nshared", "lencorr", "lenMAE"])
print("=== Top 15 languages by length-correlation with the target cipher ===")
print(df.sort_values("lencorr", ascending=False).head(15).to_string(index=False))
print("\n=== Bottom 5 ===")
print(df.sort_values("lencorr", ascending=False).tail(5).to_string(index=False))
print("\n=== Uralic subfamily means (corr desc) ===")
ur = df[df.family == "Uralic"]
print(ur.groupby("subfam")[["lencorr", "lenMAE"]].mean().sort_values("lencorr", ascending=False).to_string())
print("\n=== best Uralic langs by low MAE ===")
print(ur.sort_values("lenMAE").head(10)[["lang","subfam","lencorr","lenMAE"]].to_string(index=False))

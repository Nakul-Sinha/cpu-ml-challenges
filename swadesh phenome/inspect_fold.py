"""Inspect a single CV fold: compare recovered sigma to the true bijection,
broken down by token frequency, to see where recovery fails."""
import sys
import numpy as np
import pandas as pd
from collections import Counter
from decipher import Decipherer, forms_by_lang_concept, similarity, load_train

tr = load_train("dataset/public/train.csv")
uralic = sorted(tr[tr.family == "Uralic"].language.unique())
L = sys.argv[1] if len(sys.argv) > 1 else "sme"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

df_lang = tr[tr.language == L]
rng = np.random.default_rng(seed)
segs = sorted({s for ipa in df_lang.ipa for s in str(ipa).split()})
perm = rng.permutation(len(segs))
seg2tok = {s: f"x{perm[i]}" for i, s in enumerate(segs)}
tok2seg_true = {v: k for k, v in seg2tok.items()}

target_words, truth = [], []
for concept, ipa in zip(df_lang.concept.values, df_lang.ipa.values):
    ts = str(ipa).split()
    if not ts:
        continue
    target_words.append((concept, [seg2tok[s] for s in ts]))
    truth.append((concept, ts))

crib = forms_by_lang_concept(tr[(tr.language != L) & (tr.language.isin(uralic))])
fp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
hp = dict(gap=-6.0, pmi_k=0.5, beta=1.5, tau=8.0, n_iter=12, lensim_pow=2.0,
          rel_pow=1.0, align_scale=0.5, seg_min_langs=2, aff_keep=0.15, damp=0.5,
          freq_prior=fp)
dec = Decipherer(**hp).fit(target_words, crib)

# token frequencies
tokf = Counter(t for _, w in target_words for t in w)
score = np.mean([similarity(dec.decode(w), ts) for (_, w), (_, ts) in zip(target_words, truth)])
print(f"lang={L} nseg={len(segs)} score={score:.4f}")

# accuracy by frequency rank
toks_sorted = [t for t, _ in tokf.most_common()]
correct = {t: (dec.sigma.get(t) == tok2seg_true[t]) for t in toks_sorted}
in_cand = {t: (tok2seg_true[t] in dec.seg_id) for t in toks_sorted}
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 55), (55, len(toks_sorted))]:
    grp = toks_sorted[lo:hi]
    acc = np.mean([correct[t] for t in grp]) if grp else 0
    cov = np.mean([in_cand[t] for t in grp]) if grp else 0
    occ = sum(tokf[t] for t in grp)
    print(f"  rank {lo:2d}-{hi:2d}: acc={acc:.2f}  true-seg-in-candidates={cov:.2f}  occ={occ}")

print("\nTop-20 tokens: predicted vs true (freq):")
for t in toks_sorted[:20]:
    ok = "OK " if correct[t] else "XX "
    print(f"  {ok} {t:5} freq={tokf[t]:4d}  pred={dec.sigma.get(t):5}  true={tok2seg_true[t]:5}")

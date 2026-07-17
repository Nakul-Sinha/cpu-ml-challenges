"""Sanity check: is the cognate signal recoverable with a trivial positional method?
For each enciphered target word, align position-by-position to every equal-length
crib form, accumulate co-occurrence, score with PMI, commit with Hungarian."""
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment
from decipher import forms_by_lang_concept, similarity, load_train
from cv import encipher_language

tr = load_train("dataset/public/train.csv")
uralic = sorted(tr[tr.family == "Uralic"].language.unique())

def run(target_lang, seed=0, min_langs=2, lenslack=0):
    df_lang = tr[tr.language == target_lang]
    target_words, truth, nseg = encipher_language(df_lang, seed)
    crib_df = tr[(tr.language != target_lang) & (tr.language.isin(uralic))]
    crib = forms_by_lang_concept(crib_df)

    # token & segment universes
    toks = sorted({t for _, w in target_words for t in w})
    tid = {t: i for i, t in enumerate(toks)}
    seg_langs = defaultdict(set)
    for lang, cd in crib.items():
        for c, forms in cd.items():
            for f in forms:
                for s in f: seg_langs[s].add(lang)
    segs = sorted([s for s, ls in seg_langs.items() if len(ls) >= min_langs])
    sid = {s: i for i, s in enumerate(segs)}
    T, Sn = len(toks), len(segs)

    C = np.zeros((T, Sn))
    tok_occ = np.zeros(T); seg_occ = np.zeros(Sn)
    # concept -> crib forms
    ccribs = defaultdict(list)
    for lang, cd in crib.items():
        for c, forms in cd.items():
            for f in forms:
                ids = [sid[s] for s in f if s in sid]
                ccribs[c].append(ids)
    for (concept, w) in target_words:
        xi = [tid[t] for t in w]
        for f in ccribs.get(concept, []):
            if abs(len(f) - len(xi)) <= lenslack and len(f) == len(xi):
                for a, b in zip(xi, f):
                    C[a, b] += 1
    # PMI-style score
    tot = C.sum() + 1e-9
    rt = C.sum(axis=1, keepdims=True) + 1e-9
    rs = C.sum(axis=0, keepdims=True) + 1e-9
    pmi = np.log((C + 0.5) * tot / (rt * rs))
    ri, ci = linear_sum_assignment(-pmi)
    sigma = {toks[t]: segs[c] for t, c in zip(ri, ci)}
    # score
    sims = []
    for (concept, w), (_, true_segs) in zip(target_words, truth):
        pred = [sigma.get(t, "") for t in w]
        sims.append(similarity(pred, true_segs))
    return float(np.mean(sims)), nseg

for l in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["fin", "sme", "krl", "myv"]):
    sf = tr[tr.language == l].subfamily.iloc[0]
    s, nseg = run(l)
    print(f"{l:6} {sf:12} nseg={nseg:3d}  positional-PMI score={s:.4f}", flush=True)

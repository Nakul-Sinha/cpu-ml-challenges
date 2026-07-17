"""
Leave-one-Uralic-language-out validation for the decipherment engine.

Mirror of the real task: take a known Uralic language, replace each of its IPA
segments by an opaque token under a random bijection, hide that language, recover
the map from the remaining relatives, and score the decoded words against truth.
The mean similarity here is an honest estimate of the leaderboard metric.
"""
import sys, time, argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from decipher import Decipherer, forms_by_lang_concept, similarity, load_train


def encipher_language(df_lang, seed):
    """Return list of (concept, [token,...]) enciphering every segment via a random
    bijection, plus the list of (concept, true_segment_list) for scoring."""
    rng = np.random.default_rng(seed)
    segs = sorted({s for ipa in df_lang.ipa for s in str(ipa).split()})
    perm = rng.permutation(len(segs))
    seg2tok = {s: f"x{perm[i]}" for i, s in enumerate(segs)}
    target_words, truth = [], []
    for concept, ipa in zip(df_lang.concept.values, df_lang.ipa.values):
        true_segs = str(ipa).split()
        if not true_segs:
            continue
        toks = [seg2tok[s] for s in true_segs]
        target_words.append((concept, toks))
        truth.append((concept, true_segs))
    return target_words, truth, len(segs)


def run_fold(tr, target_lang, family_langs, hp, seed=0, trace=False):
    df_lang = tr[tr.language == target_lang]
    target_words, truth, nseg = encipher_language(df_lang, seed)

    crib_df = tr[(tr.language != target_lang) & (tr.language.isin(family_langs))]
    crib = forms_by_lang_concept(crib_df)

    def eval_fn(sigma):
        return float(np.mean([similarity([sigma.get(t, "") for t in toks], ts)
                              for (_, toks), (_, ts) in zip(target_words, truth)]))

    dec = Decipherer(**hp).fit(target_words, crib, eval_fn=eval_fn if trace else None)

    sims = []
    for (concept, toks), (_, true_segs) in zip(target_words, truth):
        pred = dec.decode(toks)
        sims.append(similarity(pred, true_segs))
    return float(np.mean(sims)), nseg, len(target_words), dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/public/train.csv")
    ap.add_argument("--langs", default="")  # comma list; default = all Uralic
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_iter", type=int, default=14)
    ap.add_argument("--gap", type=float, default=-6.0)
    ap.add_argument("--pmi_k", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--lensim_pow", type=float, default=2.0)
    ap.add_argument("--rel_pow", type=float, default=1.0)
    ap.add_argument("--align_scale", type=float, default=0.5)
    ap.add_argument("--seg_min_langs", type=int, default=2)
    ap.add_argument("--aff_keep", type=float, default=0.15)
    ap.add_argument("--damp", type=float, default=0.5)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    tr = load_train(args.data)
    uralic = sorted(tr[tr.family == "Uralic"].language.unique())
    if args.langs:
        folds = args.langs.split(",")
    else:
        folds = uralic

    hp = dict(gap=args.gap, pmi_k=args.pmi_k, beta=args.beta, tau=args.tau,
              n_iter=args.n_iter, lensim_pow=args.lensim_pow, rel_pow=args.rel_pow,
              align_scale=args.align_scale, seg_min_langs=args.seg_min_langs,
              aff_keep=args.aff_keep, damp=args.damp)

    print("HP:", hp)
    print(f"{'lang':6} {'subfam':12} {'nseg':>4} {'nword':>5} {'score':>7}  {'time':>6}")
    results = []
    for l in folds:
        sf = tr[tr.language == l].subfamily.iloc[0]
        t0 = time.time()
        score, nseg, nword, dec = run_fold(tr, l, uralic, hp, seed=args.seed, trace=args.trace)
        dt = time.time() - t0
        results.append((l, sf, nseg, score))
        print(f"{l:6} {sf:12} {nseg:4d} {nword:5d} {score:7.4f}  {dt:6.1f}s", flush=True)

    results = pd.DataFrame(results, columns=["lang", "subfam", "nseg", "score"])
    print("\nMean score (all folds): %.4f" % results.score.mean())
    saami = results[results.subfam == "Saami"]
    if len(saami):
        print("Mean score (Saami folds, ~70-seg like target): %.4f" % saami.score.mean())
    print("By subfamily:")
    print(results.groupby("subfam").score.mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()

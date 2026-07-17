"""Parallel leave-one-Uralic-out CV across all folds, for hyperparameter sweeps.
Run on the 16-core box.  Prints per-fold scores plus the all-fold and Saami-only means
(Saami folds, ~70 segments, best mirror the real ~70-token target)."""
import argparse, time, os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from decipher import Decipherer, forms_by_lang_concept, similarity, load_train

_TR = None
_URALIC = None


def _init(path):
    global _TR, _URALIC
    _TR = load_train(path)
    _URALIC = sorted(_TR[_TR.family == "Uralic"].language.unique())


def _fold(args):
    lang, seed, hp = args
    tr, uralic = _TR, _URALIC
    df_lang = tr[tr.language == lang]
    rng = np.random.default_rng(seed)
    segs = sorted({s for ipa in df_lang.ipa for s in str(ipa).split()})
    perm = rng.permutation(len(segs))
    seg2tok = {s: f"x{perm[i]}" for i, s in enumerate(segs)}
    target_words, truth = [], []
    for concept, ipa in zip(df_lang.concept.values, df_lang.ipa.values):
        ts = str(ipa).split()
        if not ts:
            continue
        target_words.append((concept, [seg2tok[s] for s in ts]))
        truth.append(ts)
    crib = forms_by_lang_concept(tr[(tr.language != lang) & (tr.language.isin(uralic))])
    dec = Decipherer(**hp).fit(target_words, crib)
    sims = [similarity(dec.decode(w), ts) for (_, w), ts in zip(target_words, truth)]
    sf = df_lang.subfamily.iloc[0]
    return lang, sf, len(segs), float(np.mean(sims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/public/train.csv")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--langs", default="")
    ap.add_argument("--tag", default="")
    for name, typ, dflt in [("n_iter", int, 14), ("gap", float, -6.0), ("pmi_k", float, 0.5),
                            ("beta", float, 1.5), ("tau", float, 8.0), ("lensim_pow", float, 2.0),
                            ("rel_pow", float, 1.0), ("align_scale", float, 0.5),
                            ("seg_min_langs", int, 2), ("aff_keep", float, 0.15),
                            ("damp", float, 0.5), ("freq_prior", float, 0.5)]:
        ap.add_argument(f"--{name}", type=typ, default=dflt)
    args = ap.parse_args()

    hp = dict(n_iter=args.n_iter, gap=args.gap, pmi_k=args.pmi_k, beta=args.beta,
              tau=args.tau, lensim_pow=args.lensim_pow, rel_pow=args.rel_pow,
              align_scale=args.align_scale, seg_min_langs=args.seg_min_langs,
              aff_keep=args.aff_keep, damp=args.damp, freq_prior=args.freq_prior)

    _init(args.data)
    folds = args.langs.split(",") if args.langs else _URALIC
    tasks = [(l, args.seed, hp) for l in folds]
    t0 = time.time()
    with Pool(args.procs, initializer=_init, initargs=(args.data,)) as pool:
        res = pool.map(_fold, tasks)
    dt = time.time() - t0

    df = pd.DataFrame(res, columns=["lang", "subfam", "nseg", "score"]).sort_values("score")
    print(f"\n[{args.tag}] HP: {hp}")
    print(df.to_string(index=False))
    saami = df[df.subfam == "Saami"]
    print(f"\n[{args.tag}] mean_all={df.score.mean():.4f}  mean_saami={saami.score.mean():.4f}  "
          f"median={df.score.median():.4f}  n={len(df)}  time={dt:.0f}s")


if __name__ == "__main__":
    main()

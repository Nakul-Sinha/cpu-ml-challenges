"""Leave-one-Uralic-out CV for the CONFIG ENSEMBLE (ensemble_decode). Compares against
the single best config. seg_min_langs is fixed across members (shared segment universe)."""
import argparse, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from decipher import Decipherer, ensemble_decode, forms_by_lang_concept, similarity, load_train

# Ensemble members: variations around the best config (all seg_min_langs=2, n_iter=18).
BASE = dict(n_iter=18, gap=-6.0, pmi_k=0.5, beta=1.5, tau=8.0, lensim_pow=2.0,
            align_scale=0.5, seg_min_langs=2, aff_keep=0.15, damp=0.5)
ENSEMBLE = [
    {**BASE, "rel_pow": 3.0, "freq_prior": 0.7, "cog_floor": 0.5},
    {**BASE, "rel_pow": 2.0, "freq_prior": 1.0, "cog_floor": 0.5},
    {**BASE, "rel_pow": 3.0, "freq_prior": 0.7, "cog_floor": 0.3, "gap": -9.0},
    {**BASE, "rel_pow": 2.0, "freq_prior": 0.85, "cog_floor": 0.5, "align_scale": 0.65},
    {**BASE, "rel_pow": 4.0, "freq_prior": 0.7, "cog_floor": 0.5},
]
SINGLE = {**BASE, "rel_pow": 3.0, "freq_prior": 0.7, "cog_floor": 0.5}

_TR = None; _URALIC = None
def _init(path):
    global _TR, _URALIC
    _TR = load_train(path); _URALIC = sorted(_TR[_TR.family == "Uralic"].language.unique())

def _fold(lang):
    tr, uralic = _TR, _URALIC
    dl = tr[tr.language == lang]
    rng = np.random.default_rng(0)
    segs = sorted({s for i in dl.ipa for s in str(i).split()})
    perm = rng.permutation(len(segs))
    s2t = {s: f"x{perm[i]}" for i, s in enumerate(segs)}
    tw, truth = [], []
    for c, i in zip(dl.concept, dl.ipa):
        ts = str(i).split()
        if ts:
            tw.append((c, [s2t[s] for s in ts])); truth.append(ts)
    crib = forms_by_lang_concept(tr[(tr.language != lang) & (tr.language.isin(uralic))])
    # single
    d = Decipherer(**SINGLE).fit(tw, crib)
    s_single = np.mean([similarity(d.decode(w), ts) for (_, w), ts in zip(tw, truth)])
    # ensemble
    sigma, _ = ensemble_decode(tw, crib, ENSEMBLE)
    s_ens = np.mean([similarity([sigma.get(t, "") for t in w], ts) for (_, w), ts in zip(tw, truth)])
    return lang, dl.subfamily.iloc[0], len(segs), float(s_single), float(s_ens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/public/train.csv")
    ap.add_argument("--procs", type=int, default=13)
    ap.add_argument("--langs", default="")
    args = ap.parse_args()
    _init(args.data)
    folds = args.langs.split(",") if args.langs else _URALIC
    t0 = time.time()
    with Pool(args.procs, initializer=_init, initargs=(args.data,)) as pool:
        res = pool.map(_fold, folds)
    df = pd.DataFrame(res, columns=["lang", "subfam", "nseg", "single", "ens"]).sort_values("single")
    df["delta"] = df.ens - df.single
    print(df.to_string(index=False))
    fin = df[df.subfam == "Finnic"]
    print(f"\nSINGLE mean_all={df.single.mean():.4f} finnic={fin.single.mean():.4f}")
    print(f"ENSEMB mean_all={df.ens.mean():.4f} finnic={fin.ens.mean():.4f}  "
          f"delta_all={df.delta.mean():+.4f} time={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

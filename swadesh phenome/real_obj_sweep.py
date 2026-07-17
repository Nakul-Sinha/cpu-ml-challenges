"""Select the config for the REAL target by maximising its decoding self-consistency (obj),
an unsupervised criterion that correlates with the true metric at r=0.90 in CV. Runs a grid in
parallel and reports the top configs by real obj (with the closest relatives as a sanity check)."""
import itertools, time
import pandas as pd
from multiprocessing import Pool
from decipher import Decipherer, forms_by_lang_concept, load_train

_CRIB = None; _TW = None
def _init():
    global _CRIB, _TW
    tr = load_train("dataset/public/train.csv"); te = pd.read_csv("dataset/public/test.csv")
    _CRIB = forms_by_lang_concept(tr[tr.family == "Uralic"])
    _TW = [(c, str(x).split()) for c, x in zip(te.concept, te.cipher)]

def _run(cfg):
    dec = Decipherer(**cfg).fit(_TW, _CRIB)
    return cfg, float(dec.best_obj), dec.top_relatives(3)

def main():
    _init()
    base = dict(n_iter=20, gap=-6.0, pmi_k=0.5, beta=1.5, tau=8.0, lensim_pow=2.0,
                align_scale=0.5, aff_keep=0.15, damp=0.5)
    grid = []
    for rp, fp, cf, sml, gp in itertools.product(
            [2.0, 3.0, 4.0], [0.5, 0.7, 1.0], [0.3, 0.5, 1.0], [2, 3], [-6.0, -9.0]):
        grid.append({**base, "rel_pow": rp, "freq_prior": fp, "cog_floor": cf,
                     "seg_min_langs": sml, "gap": gp})
    t0 = time.time()
    with Pool(16, initializer=_init) as pool:
        res = pool.map(_run, grid)
    res.sort(key=lambda x: -x[1])
    print(f"ran {len(grid)} configs in {time.time()-t0:.0f}s\n")
    print("TOP 15 by real obj:")
    for cfg, obj, top in res[:15]:
        k = {x: cfg[x] for x in ["rel_pow", "freq_prior", "cog_floor", "seg_min_langs", "gap"]}
        print(f"  obj={obj:.4f}  {k}  top={top}")
    print("\nBOTTOM 3:")
    for cfg, obj, top in res[-3:]:
        k = {x: cfg[x] for x in ["rel_pow", "freq_prior", "cog_floor", "seg_min_langs", "gap"]}
        print(f"  obj={obj:.4f}  {k}")

if __name__ == "__main__":
    main()

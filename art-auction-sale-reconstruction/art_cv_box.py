"""Test-faithful CV for Art-Auction. The test set has lower field coverage than train
(seller 0.61 vs 0.69, price 0.35 vs 0.50, nat 0.64 vs 0.72), so we mask validation pools to
those rates and optionally augment training with masked copies for robustness. Group CV,
parallel over folds; reports mean ARI at several merge thresholds."""
import os
# cap threads per worker: folds run in a Pool, each sklearn fit would otherwise grab all cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"
import argparse, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score
from art_common import pool_lots, pair_features, pool_context, group_pool, mask_pool

# test/train coverage ratios -> keep-probabilities that map train coverage down to test
KEEP = dict(seller_keep=0.605 / 0.688, price_keep=0.352 / 0.497, nat_keep=0.640 / 0.719)

_DF = None; _FOLD = None
def _init(path, folds, seed):
    global _DF, _FOLD
    _DF = pd.read_csv(path).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    _FOLD = rng.permutation(len(_DF)) % folds


def _pairs_from_pool(lots, g):
    X, yy = [], []
    n = len(lots)
    ctx = pool_context(lots)
    for a in range(n):
        if lots[a] is None:
            continue
        for b in range(a + 1, n):
            if lots[b] is None:
                continue
            X.append(pair_features(lots[a], lots[b], ctx))
            yy.append(1 if g[a] == g[b] else 0)
    return X, yy


def _fold(args):
    f, thresholds, augment, seed, seed_eff, refine = args
    df, fold = _DF, _FOLD
    # ---- train pairs from non-fold pools (+ optional masked augmentation)
    X, y = [], []
    rng = np.random.default_rng(1000 + f)
    for i in range(len(df)):
        if fold[i] == f:
            continue
        lots = pool_lots(df.iloc[i]); g = str(df.iloc[i]["grouping"]).split()
        xx, yy = _pairs_from_pool(lots, g); X += xx; y += yy
        if augment:
            for _ in range(augment):
                ml = mask_pool(lots, rng, **KEEP)
                xx, yy = _pairs_from_pool(ml, g); X += xx; y += yy
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07, max_leaf_nodes=31,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=0)
    model.fit(np.array(X, dtype=np.float32), np.array(y))
    # ---- evaluate on masked validation pools
    out = {t: [] for t in thresholds}
    rng = np.random.default_rng(5000 + f)
    for i in range(len(df)):
        if fold[i] != f:
            continue
        row = df.iloc[i]; n = int(row["n_lots"]); tl = np.array([int(x) for x in str(row["grouping"]).split()])
        ml = mask_pool(pool_lots(row), rng, **KEEP)
        for t in thresholds:
            lab = group_pool(ml, model, threshold=t, seed_eff=seed_eff, refine_iters=refine)[:n]
            out[t].append(adjusted_rand_score(tl, lab))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/public/train.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", type=int, default=0, help="masked copies per training pool")
    ap.add_argument("--seed_eff", action="store_true", help="seed clusters by artist-imputed consignor")
    ap.add_argument("--refine", type=int, default=0, help="second-stage reassignment iterations")
    ap.add_argument("--thresholds", default="0.2,0.25,0.3,0.35")
    args = ap.parse_args()
    _init(args.data, args.folds, args.seed)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    t0 = time.time()
    with Pool(min(args.folds, 8), initializer=_init, initargs=(args.data, args.folds, args.seed)) as pool:
        res = pool.map(_fold, [(f, thresholds, args.augment, args.seed, args.seed_eff, args.refine)
                               for f in range(args.folds)])
    agg = {t: [] for t in thresholds}
    for r in res:
        for t in thresholds:
            agg[t] += r[t]
    print(f"[augment={args.augment}] test-faithful CV (masked val):")
    for t in thresholds:
        print(f"  threshold={t:.2f}  mean ARI = {np.mean(agg[t]):.4f}  (n={len(agg[t])})")
    print(f"  time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

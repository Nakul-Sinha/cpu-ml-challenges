"""Is the best per-pool merge threshold predictable from pool features? If so, an adaptive
threshold captures some of the oracle-per-pool-threshold headroom (0.62 -> 0.648 ceiling).
Test-faithful masking; reports correlations and a learned-threshold CV ARI."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score
from art_common import pool_lots, pair_features, pool_context, group_pool, mask_pool, prob_matrix

KEEP = dict(seller_keep=0.605 / 0.688, price_keep=0.352 / 0.497, nat_keep=0.640 / 0.719)
THRS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
_DF = None; _FOLD = None
def _init(path, folds, seed):
    global _DF, _FOLD
    _DF = pd.read_csv(path).reset_index(drop=True)
    _FOLD = np.random.default_rng(seed).permutation(len(_DF)) % folds

def _pairs(lots, g):
    X, y = [], []; ctx = pool_context(lots); n = len(lots)
    for a in range(n):
        if lots[a] is None: continue
        for b in range(a + 1, n):
            if lots[b] is None: continue
            X.append(pair_features(lots[a], lots[b], ctx)); y.append(1 if g[a] == g[b] else 0)
    return X, y

def _poolfeat(lots):
    ctx = pool_context(lots); n = ctx["n"]
    sellers = set(l["seller"] for l in lots if l and l["seller"])
    return [ctx["seller_cov"], n / 48.0, len(sellers) / n,
            sum(1 for l in lots if l and not np.isnan(l["price"])) / n,
            sum(1 for l in lots if l and l["nat_ok"]) / n]

def _fold(f):
    df, fold = _DF, _FOLD
    X, y = [], []
    rng = np.random.default_rng(1000 + f)
    for i in range(len(df)):
        if fold[i] == f: continue
        lots = pool_lots(df.iloc[i]); g = str(df.iloc[i]["grouping"]).split()
        xx, yy = _pairs(lots, g); X += xx; y += yy
        for _ in range(2):
            xx, yy = _pairs(mask_pool(lots, rng, **KEEP), g); X += xx; y += yy
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07, max_leaf_nodes=31,
                                           min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    model.fit(np.array(X, dtype=np.float32), np.array(y))
    rng = np.random.default_rng(5000 + f)
    rows = []
    for i in range(len(df)):
        if fold[i] != f: continue
        row = df.iloc[i]; n = int(row["n_lots"]); tl = np.array([int(x) for x in str(row["grouping"]).split()])
        ml = mask_pool(pool_lots(row), rng, **KEEP)
        aris = [adjusted_rand_score(tl, group_pool(ml, model, threshold=t)[:n]) for t in THRS]
        rows.append((_poolfeat(ml), aris))
    return rows

def main():
    _init("dataset/public/train.csv", 5, 0)
    with Pool(5, initializer=_init, initargs=("dataset/public/train.csv", 5, 0)) as pool:
        res = pool.map(_fold, range(5))
    feats, aris = [], []
    for r in res:
        for pf, ar in r:
            feats.append(pf); aris.append(ar)
    feats = np.array(feats); aris = np.array(aris)          # (P, 5), (P, T)
    best_t = np.array([THRS[i] for i in aris.argmax(axis=1)])
    fixed = aris[:, THRS.index(0.3)].mean()
    oracle = aris.max(axis=1).mean()
    print(f"fixed-0.30 ARI={fixed:.4f}  oracle-per-pool ARI={oracle:.4f}")
    names = ["seller_cov", "n_lots", "n_sellers/n", "price_cov", "nat_cov"]
    for j, nm in enumerate(names):
        print(f"  corr(best_threshold, {nm:12}) = {np.corrcoef(feats[:, j], best_t)[0,1]:+.3f}")
    # learned adaptive threshold via 5-fold (predict best_t from features), snap to grid
    P = len(feats); fold = np.arange(P) % 5; pred_ari = []
    for f in range(5):
        tr = fold != f; va = fold == f
        reg = LinearRegression().fit(feats[tr], best_t[tr])
        pt = np.clip(reg.predict(feats[va]), min(THRS), max(THRS))
        idx = [min(range(len(THRS)), key=lambda k: abs(THRS[k] - p)) for p in pt]
        pred_ari += [aris[va][r, idx[r]] for r in range(va.sum())]
    print(f"learned adaptive-threshold ARI = {np.mean(pred_ari):.4f}")

if __name__ == "__main__":
    main()

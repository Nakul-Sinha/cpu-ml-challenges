"""Art-Auction CV: pairwise same-sale model + constrained agglomerative grouping (seed by
consignor, merge only confident pairs). Group CV over pools, mean ARI. Threshold sweep."""
import argparse, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score
from art_common import pool_lots, pair_features, group_pool, pool_context

N_GROUPS = 6


def build_pairs(df):
    X, y, pid = [], [], []
    for k, (_, row) in enumerate(df.iterrows()):
        lots = pool_lots(row)
        ctx = pool_context(lots)
        g = str(row["grouping"]).split()
        n = len(lots)
        for a in range(n):
            if lots[a] is None:
                continue
            for b in range(a + 1, n):
                if lots[b] is None:
                    continue
                X.append(pair_features(lots[a], lots[b], ctx))
                y.append(1 if g[a] == g[b] else 0)
                pid.append(k)
    return np.array(X, dtype=np.float32), np.array(y), np.array(pid)


def true_labels(row):
    return np.array([int(x) for x in str(row["grouping"]).split()])


def seller_singleton(lots):
    n = len(lots); labels = np.full(n, -1); nxt = 0; seller_to = {}
    for i in range(n):
        if lots[i] is None:
            continue
        s = lots[i]["seller"]
        if s:
            if s not in seller_to:
                seller_to[s] = nxt; nxt += 1
            labels[i] = seller_to[s]
        else:
            labels[i] = nxt; nxt += 1
    return labels


def seller_oneblob(lots):
    """Seller groups; all no-seller lots share one group (the problem's ~0.43 baseline)."""
    n = len(lots); labels = np.full(n, -1); nxt = 0; seller_to = {}; blob = None
    for i in range(n):
        if lots[i] is None:
            continue
        s = lots[i]["seller"]
        if s:
            if s not in seller_to:
                seller_to[s] = nxt; nxt += 1
            labels[i] = seller_to[s]
        else:
            if blob is None:
                blob = nxt; nxt += 1
            labels[i] = blob
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/public/train.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thresholds", default="0.4,0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    df = pd.read_csv(args.data).reset_index(drop=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    fold_of = {i: perm[i] % args.folds for i in range(len(df))}

    X, y, pid = build_pairs(df)
    print(f"pairs: {len(y)}  pos-rate {y.mean():.3f}")

    lots_by_pool = {i: pool_lots(df.iloc[i]) for i in range(len(df))}
    truth_by_pool = {i: true_labels(df.iloc[i]) for i in range(len(df))}
    nlots = {i: int(df.iloc[i]["n_lots"]) for i in range(len(df))}

    def ari_all(fn):
        return np.mean([adjusted_rand_score(truth_by_pool[i], fn(lots_by_pool[i])[:nlots[i]])
                        for i in range(len(df))])
    print(f"[baseline] all-one-group   ARI = {ari_all(lambda L: np.zeros(len(L), int)):.4f}")
    print(f"[baseline] all-singletons  ARI = {ari_all(lambda L: np.arange(len(L))):.4f}")
    print(f"[baseline] seller+oneblob  ARI = {ari_all(seller_oneblob):.4f}")
    print(f"[baseline] seller+singleton ARI = {ari_all(seller_singleton):.4f}")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    # OOF predictions per pool with each threshold
    oof = {t: {} for t in thresholds}
    t0 = time.time()
    for f in range(args.folds):
        tr_mask = np.array([fold_of[p] != f for p in pid])
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                max_leaf_nodes=31, min_samples_leaf=50,
                                                l2_regularization=1.0, random_state=0)
        model.fit(X[tr_mask], y[tr_mask])
        val_idx = [i for i in range(len(df)) if fold_of[i] == f]
        for i in val_idx:
            for t in thresholds:
                oof[t][i] = group_pool(lots_by_pool[i], model, threshold=t)
    for t in thresholds:
        aris = [adjusted_rand_score(truth_by_pool[i], oof[t][i][:nlots[i]]) for i in range(len(df))]
        print(f"[model] threshold={t:.2f}  mean ARI = {np.mean(aris):.4f}")
    print(f"(train+cluster {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

"""
Art-Auction Sale Reconstruction -- official solution.

Reads ./dataset/public/{train,test}.csv, learns a pairwise same-sale model from the training
pools, and groups each test pool's lots into sales by seeding clusters on the shared consignor
and agglomeratively merging only confident pairs. Writes ./working/submission.csv.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from art_common import pool_lots, pair_features, pool_context, group_pool

THRESHOLD = 0.25   # merge clusters while average same-sale prob exceeds this (CV-tuned)


def build_pairs(df):
    X, y = [], []
    for _, row in df.iterrows():
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
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    public_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/public"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "working/submission.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tr = pd.read_csv(os.path.join(public_dir, "train.csv"))
    te = pd.read_csv(os.path.join(public_dir, "test.csv"))

    X, y = build_pairs(tr)
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07, max_leaf_nodes=31,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=0)
    model.fit(X, y)

    ids, preds = [], []
    for _, row in te.iterrows():
        lots = pool_lots(row)
        n = int(row["n_lots"])
        lab = group_pool(lots, model, threshold=THRESHOLD)[:n]
        ids.append(row["id"])
        preds.append(" ".join(str(int(x)) for x in lab))

    out = pd.DataFrame({"id": ids, "prediction": preds})
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows -> {out_path}")
    ng = [len(set(p.split())) for p in preds]
    print(f"groups/pool: mean {np.mean(ng):.1f}  min {min(ng)}  max {max(ng)}")


if __name__ == "__main__":
    main()

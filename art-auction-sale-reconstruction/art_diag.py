"""Diagnostics: verify ARI by hand on one pool, find the clustering ceiling with oracle
same-sale probabilities, and measure the pairwise model's accuracy on no-seller pairs."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from art_common import pool_lots, pair_features, group_pool
from art_cv import build_pairs, true_labels, seller_singleton

df = pd.read_csv("dataset/public/train.csv").reset_index(drop=True)

# --- 1. manual check on pool 0
row = df.iloc[0]; lots = pool_lots(row); tl = true_labels(row)
sl = seller_singleton(lots)[:int(row.n_lots)]
print("pool0 n_lots:", int(row.n_lots), "true groups:", len(set(tl)), "seller groups:", len(set(sl)))
print("true  :", " ".join(map(str, tl[:30])))
print("seller:", " ".join(map(str, sl[:30])))
print("pool0 seller-singleton ARI:", round(adjusted_rand_score(tl, sl), 4))
sellers = [l["seller"] if l else "" for l in lots]
print("pool0 lots with seller:", sum(1 for s in sellers if s), "/", len(lots))

# --- 2. oracle ceiling: cluster using true same-sale probs
def oracle_group(i, min_clusters=6):
    row = df.iloc[i]; lots = pool_lots(row); tl = true_labels(row)
    n = len(lots)
    idx = [j for j in range(n) if lots[j] is not None]

    class Oracle:  # returns P(same sale)=1/0 from truth
        def predict_proba(self, X):  # unused; we bypass prob_matrix
            pass
    # build P from truth directly
    P = np.zeros((n, n))
    for a in idx:
        for b in idx:
            if a != b:
                P[a, b] = 1.0 if tl[a] == tl[b] else 0.0
    # reuse group_pool merging logic manually with this P
    clusters, seller_to = [], {}
    for j in idx:
        s = lots[j]["seller"]
        if s:
            seller_to.setdefault(s, len(clusters))
            if len(clusters) <= seller_to[s]:
                clusters.append([])
            clusters[seller_to[s]].append(j)
        else:
            clusters.append([j])
    while len(clusters) > min_clusters:
        best, bi, bj = -1, -1, -1
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                aff = np.mean([P[p, q] for p in clusters[a] for q in clusters[b]])
                if aff > best:
                    best, bi, bj = aff, a, b
        if best < 0.5:
            break
        clusters[bi] += clusters[bj]; del clusters[bj]
    lab = np.zeros(n, int)
    for ci, c in enumerate(clusters):
        for j in c:
            lab[j] = ci
    return adjusted_rand_score(tl, lab[:int(row.n_lots)])

oracle_aris = [oracle_group(i) for i in range(len(df))]
print("\nORACLE-prob clustering mean ARI (ceiling of this clustering):", round(np.mean(oracle_aris), 4))

# --- 3. model AUC on no-seller pairs vs seller pairs
X, y, pid = build_pairs(df)
# split
rng = np.random.default_rng(0); perm = rng.permutation(len(df)); fold_of = perm % 5
trm = np.array([fold_of[p] != 0 for p in pid]); vam = ~trm
model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                        min_samples_leaf=50, l2_regularization=1.0, random_state=0)
model.fit(X[trm], y[trm])
pv = model.predict_proba(X[vam])[:, 1]; yv = y[vam]
# seller_match is feature 0, seller_unknown is feature 2
Xv = X[vam]
seller_pair = Xv[:, 0] == 1
noseller_pair = Xv[:, 2] == 1
both_seller_diff = Xv[:, 1] == 1
print("\nvalidation pairs:", len(yv), "pos-rate", round(yv.mean(), 3))
print("AUC all:", round(roc_auc_score(yv, pv), 4))
for name, mask in [("same-seller", seller_pair), ("diff-seller", both_seller_diff),
                   ("no-seller(>=1 missing)", noseller_pair)]:
    if mask.sum() > 10 and len(set(yv[mask])) > 1:
        print(f"  {name:24}: n={mask.sum():6d} pos-rate={yv[mask].mean():.3f} AUC={roc_auc_score(yv[mask], pv[mask]):.4f}")

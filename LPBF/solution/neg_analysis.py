"""Detect whether the TEST set contains negative (no-alert) images by comparing
the per-image max candidate confidence of test vs train. Train is 100% positive,
so a low-confidence subpopulation in test that train lacks indicates negatives
(and tells us where to set the empty-prediction threshold)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C
import detect as D
from features import FeatureExtractor

PUB = C.find_public_dir(None)
POS_R = {"gray": 4.0, "color": 7.0}
FAMS = ["gray", "color"]
_ANCH = _PRIOR = _BG = None


def _odd(s):
    s = int(round(s)); return max(15, min(41, s + (s % 2 == 0)))


def _feats(row, with_label):
    fam = C.family(row["height"])
    bgr = C.load_bgr(PUB, row["image_path"])
    if bgr is None:
        return None
    fe = FeatureExtractor(bgr, fam, prior_hm=_PRIOR[fam], bg=_BG[fam])
    cands = D.gen_candidates(fe, _ANCH[fam], fam)
    X = np.stack([D.featurize(fe, c) for c in cands]).astype(np.float32) if cands else np.zeros((0, 1), np.float32)
    y = None
    if with_label:
        gcen = [((x0 + x1) / 2, (y0 + y1) / 2) for (x0, y0, x1, y1) in C.parse_boxes(row["boxes"])]
        y = np.array([1 if any(abs(gx - c["cx"]) <= POS_R[fam] and abs(gy - c["cy"]) <= POS_R[fam]
                               for gx, gy in gcen) else 0 for c in cands])
    return dict(iid=row["image_id"], fam=fam, X=X, y=y)


def pre(df, with_label):
    rows = [dict(image_id=r["image_id"], image_path=r["image_path"], height=r["height"],
                 boxes=r.get("boxes", "")) for _, r in df.iterrows()]
    res = Parallel(n_jobs=14, backend="multiprocessing")(delayed(_feats)(r, with_label) for r in rows)
    return [x for x in res if x is not None]


def main():
    global _ANCH, _PRIOR, _BG
    tr = pd.read_csv(os.path.join(PUB, "train.csv")).reset_index(drop=True)
    te = pd.read_csv(os.path.join(PUB, "test.csv")).reset_index(drop=True)
    _ANCH = {f: C.build_anchors(tr, f) for f in FAMS}
    _PRIOR = {f: C.prior_heatmap(tr, f, 448 if f == "color" else 358, 448) for f in FAMS}
    _BG = {f: C.build_background(tr, f, PUB) for f in FAMS}
    trd = pre(tr, True)
    ted = pre(te, False)

    # OOF train max-score via 5-fold
    ids = [d["iid"] for d in trd]
    rng = np.random.RandomState(0); fold = np.array([i % 5 for i in range(len(trd))]); rng.shuffle(fold)
    train_max = {}
    for k in range(5):
        Xtr = np.concatenate([trd[i]["X"] for i in range(len(trd)) if fold[i] != k and len(trd[i]["y"])])
        ytr = np.concatenate([trd[i]["y"] for i in range(len(trd)) if fold[i] != k and len(trd[i]["y"])])
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
        for i in range(len(trd)):
            if fold[i] == k and len(trd[i]["X"]):
                train_max[trd[i]["iid"]] = float(m.predict_proba(trd[i]["X"])[:, 1].max())

    # full model for test
    Xall = np.concatenate([d["X"] for d in trd if len(d["y"])])
    yall = np.concatenate([d["y"] for d in trd if len(d["y"])])
    M = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=20, random_state=0).fit(Xall, yall)
    test_max = {d["iid"]: (float(M.predict_proba(d["X"])[:, 1].max()) if len(d["X"]) else 0.0) for d in ted}

    trv = np.array(list(train_max.values()))
    tev = np.array(list(test_max.values()))
    ps = [1, 5, 10, 15, 20, 25, 50]
    print("per-image MAX candidate score percentiles:")
    print("  train(OOF): " + " ".join("p%d=%.3f" % (p, np.percentile(trv, p)) for p in ps))
    print("  test:       " + " ".join("p%d=%.3f" % (p, np.percentile(tev, p)) for p in ps))
    print("\nimages with max-score below threshold (candidate negatives):")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        print("  th=%.2f  train=%.1f%% (%d)  test=%.1f%% (%d)" %
              (t, 100 * np.mean(trv < t), int(np.sum(trv < t)),
               100 * np.mean(tev < t), int(np.sum(tev < t))))
    # by family
    for fam in FAMS:
        tv = np.array([test_max[d["iid"]] for d in ted if d["fam"] == fam])
        print("  test[%s] max-score: p10=%.3f p25=%.3f med=%.3f  (<0.2: %d/%d)" %
              (fam, np.percentile(tv, 10), np.percentile(tv, 25), np.median(tv),
               int(np.sum(tv < 0.2)), len(tv)))


if __name__ == "__main__":
    main()

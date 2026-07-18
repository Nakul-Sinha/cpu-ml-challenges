"""Pinpoint where the score is lost: presence AUC/AP, size-prediction MAE
(regressor vs per-anchor median vs global median), and coverage-miss geometry."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import common as C
import detect as D
from features import FeatureExtractor, SIZE_GRID
import model as M

PUB = C.find_public_dir(sys.argv[1] if len(sys.argv) > 1 else None)


def labeled(df, anchors, prior):
    rows = []
    for _, r in df.iterrows():
        fam = C.family(r["height"])
        bgr = C.load_bgr(PUB, r["image_path"])
        if bgr is None:
            continue
        fe = FeatureExtractor(bgr, fam, prior_hm=prior[fam])
        gcen = [((x0 + x1) / 2.0, (y0 + y1) / 2.0, max(x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in C.parse_boxes(r["boxes"])]
        cands = D.gen_candidates(fe, anchors[fam], fam)
        for c in cands:
            m = next((gs for (gx, gy, gs) in gcen if abs(gx - c["cx"]) <= M.POS_R[fam] and abs(gy - c["cy"]) <= M.POS_R[fam]), None)
            rows.append(dict(fam=fam, fc=D.featurize(fe, c), sf=M.size_feat(fe, c),
                             lab=0 if m is None else 1, truesize=m, anchsize=c["s"]))
    return rows


def main():
    df = pd.read_csv(os.path.join(PUB, "train.csv"))
    tr, va = C.make_split(df)
    anchors = {f: C.build_anchors(tr, f) for f in M.FAMS}
    prior = {f: C.prior_heatmap(tr, f, 448 if f == "color" else 358, 448) for f in M.FAMS}
    Xc, yc, Xs, ys = M.build_rows(tr, anchors, prior)
    cls = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=20, random_state=0).fit(Xc, yc)
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=15, random_state=0).fit(Xs, ys)
    gmed = float(np.median(ys))

    vr = labeled(va, anchors, prior)
    for fam in M.FAMS + ["all"]:
        sub = [r for r in vr if fam == "all" or r["fam"] == fam]
        y = np.array([r["lab"] for r in sub])
        p = cls.predict_proba(np.stack([r["fc"] for r in sub]))[:, 1]
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
        print("presence[%-5s] AUC=%.3f AP=%.3f  n=%d pos=%d" % (fam, auc, ap, len(y), y.sum()))

    # size MAE on matched candidates
    pos = [r for r in vr if r["lab"] == 1]
    ts = np.array([r["truesize"] for r in pos])
    sp = reg.predict(np.stack([r["sf"] for r in pos]))
    am = np.array([r["anchsize"] for r in pos])
    print("\nsize MAE: regressor=%.2f  per-anchor-median=%.2f  global-median=%.2f (n=%d)"
          % (np.abs(sp - ts).mean(), np.abs(am - ts).mean(), np.abs(gmed - ts).mean(), len(ts)))
    print("true size std=%.2f  per-anchor-med std of resid=%.2f" % (ts.std(), (am - ts).std()))

    # coverage-miss geometry
    for fam in M.FAMS:
        miss_a, miss_p = [], []
        for _, r in va.iterrows():
            if C.family(r["height"]) != fam:
                continue
            bgr = C.load_bgr(PUB, r["image_path"]);  fe = FeatureExtractor(bgr, fam, prior_hm=prior[fam])
            cands = D.gen_candidates(fe, anchors[fam], fam)
            acen = [(a["cx"], a["cy"]) for a in anchors[fam]]
            for (x0, y0, x1, y1) in C.parse_boxes(r["boxes"]):
                gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
                covered = any(abs(gx - c["cx"]) <= M.POS_R[fam] and abs(gy - c["cy"]) <= M.POS_R[fam] for c in cands)
                if not covered:
                    da = min((abs(gx - ax) + abs(gy - ay) for ax, ay in acen), default=99)
                    dp = min((abs(gx - c["cx"]) + abs(gy - c["cy"]) for c in cands), default=99)
                    miss_a.append(da); miss_p.append(dp)
        print("miss[%s]: n=%d  nearest-anchor L1 med=%.0f  nearest-cand L1 med=%.0f"
              % (fam, len(miss_a), np.median(miss_a) if miss_a else 0, np.median(miss_p) if miss_p else 0))


if __name__ == "__main__":
    main()

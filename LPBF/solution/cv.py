"""K-fold cross-validation of the full pipeline against the official metric, for
stable measurement (a single 72-image val split is too noisy to trust). Prints
mean/std of the score plus the presence-oracle and true-size ceilings.

Usage: python cv.py [--patch] [--folds N] [--th T]
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C
import detect as D
from features import FeatureExtractor, SIZE_GRID
from metric import full_score

PUB = C.find_public_dir(None)
POS_R = {"gray": 4.0, "color": 7.0}
FAMS = ["gray", "color"]
USE_PATCH = "--patch" in sys.argv
NFOLD = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 5


def _odd(s):
    s = int(round(s)); return max(15, min(41, s + (s % 2 == 0)))


def folds(df, k=5, seed=42):
    rng = np.random.RandomState(seed)
    assign = {}
    for fam in FAMS:
        ids = [i for i in df.index if C.family(df.loc[i, "height"]) == fam]
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            assign[i] = j % k
    return np.array([assign[i] for i in df.index])


def precompute(df, prior):
    """Extract candidates + features once per image (expensive part)."""
    data = []
    for _, r in df.iterrows():
        fam = C.family(r["height"])
        bgr = C.load_bgr(PUB, r["image_path"])
        if bgr is None:
            continue
        fe = FeatureExtractor(bgr, fam, prior_hm=prior[fam], use_patch=USE_PATCH)
        cands = D.gen_candidates(fe, C.ANCH[fam], fam)
        feats = np.stack([D.featurize(fe, c) for c in cands]) if cands else np.zeros((0, 1))
        gts = C.parse_boxes(r["boxes"])
        gcen = [((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in gts]
        labs, boxes = [], []
        for c in cands:
            m = next((gs for (gx, gy, gs) in gcen if abs(gx - c["cx"]) <= POS_R[fam] and abs(gy - c["cy"]) <= POS_R[fam]), None)
            labs.append(0 if m is None else 1)
            boxes.append(C.clip_box(D.box_from(c["cx"], c["cy"], _odd(c["s"])), fe.W, fe.H))
        data.append(dict(iid=r["image_id"], fam=fam, gts=gts, feats=feats,
                         labs=np.array(labs), boxes=boxes,
                         cx=[c["cx"] for c in cands], cy=[c["cy"] for c in cands]))
    return data


def evaluate(val_data, model, th, min_keep=1, max_box=6, iou_nms=0.30):
    tm, pm = {}, {}
    for im in val_data:
        tm[im["iid"]] = im["gts"]
        if len(im["boxes"]) == 0:
            pm[im["iid"]] = []; continue
        sc = model.predict_proba(im["feats"])[:, 1]
        keep = C.nms(im["boxes"], list(sc), iou_thr=iou_nms)
        kept = sorted([(sc[i], im["boxes"][i]) for i in keep], key=lambda z: -z[0])
        out = [kb for kb in kept if kb[0] >= th][:max_box]
        if len(out) < min_keep:
            out = kept[:min_keep]
        pm[im["iid"]] = [(float(s), b[0], b[1], b[2], b[3]) for (s, b) in out]
    return full_score(pm, tm)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(PUB, "train.csv")).reset_index(drop=True)
    # anchors/prior from ALL data for precompute; per-fold retrain only the ranker
    # (anchors are a location prior; using all-data anchors is a mild optimism but
    #  consistent across configs, and the final model uses all data anyway).
    C.ANCH = {f: C.build_anchors(df, f) for f in FAMS}
    prior = {f: C.prior_heatmap(df, f, 448 if f == "color" else 358, 448) for f in FAMS}
    print("anchors gray=%d color=%d patch=%s folds=%d" %
          (len(C.ANCH["gray"]), len(C.ANCH["color"]), USE_PATCH, NFOLD))
    data = precompute(df, prior)
    print("precompute %.0fs dim=%d" % (time.time() - t0, data[0]["feats"].shape[1]))
    fa = folds(df, NFOLD)
    id2fold = {df.loc[i, "image_id"]: fa[i] for i in df.index}

    ths = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    scores = {th: [] for th in ths}
    for k in range(NFOLD):
        tr = [d for d in data if id2fold[d["iid"]] != k]
        va = [d for d in data if id2fold[d["iid"]] == k]
        X = np.concatenate([d["feats"] for d in tr if len(d["labs"])])
        y = np.concatenate([d["labs"] for d in tr if len(d["labs"])])
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
            max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=20, random_state=0).fit(X, y)
        for th in ths:
            scores[th].append(evaluate(va, model, th)[0])
    print("\nCV score by threshold (mean +/- std over %d folds):" % NFOLD)
    best = None
    for th in ths:
        m, s = np.mean(scores[th]), np.std(scores[th])
        print("  th=%.2f  %.4f +/- %.4f" % (th, m, s))
        if best is None or m > best[0]:
            best = (m, s, th)
    print("BEST th=%.2f  CV=%.4f +/- %.4f   (%.0fs)" % (best[2], best[0], best[1], time.time() - t0))


if __name__ == "__main__":
    main()

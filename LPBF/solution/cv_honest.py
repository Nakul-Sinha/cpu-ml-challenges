"""Fold-HONEST cross-validation: the spatial anchors, prior heatmap and median
background are rebuilt from each fold's TRAINING portion only (no leakage of the
validation box locations), so the estimate reflects how the prior transfers to
unseen images. This is the realistic proxy for the held-out test score.

Also sweeps emission (threshold, max boxes) and a negative-image gate.
Usage: python cv_honest.py [--folds N] [--jobs J]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import common as C
import detect as D
from features import FeatureExtractor
from metric import full_score

PUB = C.find_public_dir(None)
POS_R = {"gray": 4.0, "color": 7.0}
FAMS = ["gray", "color"]
NFOLD = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 5
JOBS = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 14
REGP = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=25, random_state=0)
CLSP = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=20, random_state=0)

_ANCH = _PRIOR = _BG = None


def _odd(s):
    s = int(round(s)); return max(15, min(41, s + (s % 2 == 0)))


def _one(row):
    fam = C.family(row["height"])
    bgr = C.load_bgr(PUB, row["image_path"])
    if bgr is None:
        return None
    fe = FeatureExtractor(bgr, fam, prior_hm=_PRIOR[fam], bg=_BG[fam])
    cands = D.gen_candidates(fe, _ANCH[fam], fam)
    if not cands:
        return dict(iid=row["image_id"], fam=fam, gts=C.parse_boxes(row["boxes"]),
                    feats=np.zeros((0, 1), np.float32), labs=np.zeros(0),
                    cx=np.zeros(0), cy=np.zeros(0), sz=np.zeros(0), rtgt=np.zeros((0, 3), np.float32))
    feats = np.stack([D.featurize(fe, c) for c in cands]).astype(np.float32)
    gc = [((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in C.parse_boxes(row["boxes"])]
    labs, cx, cy, sz, rtgt = [], [], [], [], []
    for c in cands:
        cx.append(c["cx"]); cy.append(c["cy"]); sz.append(_odd(c["s"]))
        m = next(((gx, gy, gs) for (gx, gy, gs) in gc
                  if abs(gx - c["cx"]) <= POS_R[fam] and abs(gy - c["cy"]) <= POS_R[fam]), None)
        labs.append(0 if m is None else 1)
        rtgt.append((m[0] - c["cx"], m[1] - c["cy"], m[2] - _odd(c["s"])) if m else (np.nan, np.nan, np.nan))
    return dict(iid=row["image_id"], fam=fam, gts=C.parse_boxes(row["boxes"]),
                feats=feats, labs=np.array(labs), cx=np.array(cx), cy=np.array(cy),
                sz=np.array(sz), rtgt=np.array(rtgt, np.float32))


def extract(df):
    rows = [dict(image_id=r["image_id"], image_path=r["image_path"], height=r["height"], boxes=r["boxes"])
            for _, r in df.iterrows()]
    return [x for x in Parallel(n_jobs=JOBS, backend="multiprocessing")(delayed(_one)(r) for r in rows) if x]


def folds(df, k, seed=42):
    rng = np.random.RandomState(seed); assign = {}
    for fam in FAMS:
        ids = [i for i in df.index if C.family(df.loc[i, "height"]) == fam]
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            assign[i] = j % k
    return assign


def score_val(va, cls, regs, refine=True):
    out = []
    for d in va:
        if not len(d["feats"]):
            out.append((d["iid"], d["gts"], [])); continue
        p = cls.predict_proba(d["feats"])[:, 1]
        if refine:
            dcx = regs[0].predict(d["feats"]); dcy = regs[1].predict(d["feats"]); ds = regs[2].predict(d["feats"])
            boxes = [C.clip_box(D.box_from(d["cx"][i] + np.clip(dcx[i], -10, 10),
                     d["cy"][i] + np.clip(dcy[i], -10, 10), _odd(d["sz"][i] + np.clip(ds[i], -12, 12))), 448, 448)
                     for i in range(len(p))]
        else:
            boxes = [C.clip_box(D.box_from(d["cx"][i], d["cy"][i], int(d["sz"][i])), 448, 448) for i in range(len(p))]
        keep = C.nms(boxes, list(p), 0.30)
        kept = sorted([(float(p[i]), boxes[i]) for i in keep], key=lambda z: -z[0])
        out.append((d["iid"], d["gts"], kept))
    return out


def select(scored, th, max_box, gate=0.0):
    tm, pm = {}, {}
    for iid, gts, kept in scored:
        tm[iid] = gts
        top = kept[0][0] if kept else 0.0
        if top < gate:
            pm[iid] = []; continue
        o = [kb for kb in kept if kb[0] >= th][:max_box]
        if not o and kept:
            o = kept[:1]
        pm[iid] = [(s, b[0], b[1], b[2], b[3]) for (s, b) in o]
    return full_score(pm, tm)


def main():
    global _ANCH, _PRIOR, _BG
    t0 = time.time()
    df = pd.read_csv(os.path.join(PUB, "train.csv")).reset_index(drop=True)
    assign = folds(df, NFOLD)
    from collections import defaultdict
    agg = defaultdict(list)
    cov = []
    for k in range(NFOLD):
        trdf = df[df.index.map(lambda i: assign[i] != k)]
        vadf = df[df.index.map(lambda i: assign[i] == k)]
        _ANCH = {f: C.build_anchors(trdf, f) for f in FAMS}
        _PRIOR = {f: C.prior_heatmap(trdf, f, 448 if f == "color" else 358, 448) for f in FAMS}
        _BG = {f: C.build_background(trdf, f, PUB) for f in FAMS}
        tr = extract(trdf); va = extract(vadf)
        X = np.concatenate([d["feats"] for d in tr if len(d["labs"])])
        y = np.concatenate([d["labs"] for d in tr if len(d["labs"])])
        cls = HistGradientBoostingClassifier(**CLSP).fit(X, y)
        Xp = np.concatenate([d["feats"][d["labs"] == 1] for d in tr if (d["labs"] == 1).any()])
        T = np.concatenate([d["rtgt"][d["labs"] == 1] for d in tr if (d["labs"] == 1).any()])
        regs = [HistGradientBoostingRegressor(**REGP).fit(Xp, T[:, j]) for j in range(3)]
        # honest coverage
        ngt = sum(len(d["gts"]) for d in va)
        hit = sum(1 for d in va for g in d["gts"]
                  if any(d["labs"][i] == 1 and abs((g[0]+g[2])/2 - d["cx"][i]) <= POS_R[d["fam"]]
                         and abs((g[1]+g[3])/2 - d["cy"][i]) <= POS_R[d["fam"]] for i in range(len(d["labs"]))))
        cov.append(hit / max(1, ngt))
        scored = score_val(va, cls, regs, refine=True)
        for th in [0.1, 0.15, 0.2, 0.25, 0.3]:
            for mb in [2, 3, 4, 6]:
                for gate in [0.0, 0.3, 0.4, 0.5]:
                    agg[(th, mb, gate)].append(select(scored, th, mb, gate=gate)[0])
    print("HONEST anchors/prior/bg from train-fold only. coverage=%.1f%% (positives-only CV)" % (100 * np.mean(cov)))
    print("NOTE: gate>0 only *costs* on this all-positive CV; it *protects* the")
    print("negative_image_penalty on the real test, which this CV cannot measure.")
    res = sorted(((np.mean(v), np.std(v), cfg) for cfg, v in agg.items()), reverse=True)
    print("\nPositive-only score by config (th, maxbox, gate):")
    for m, s, cfg in res[:8]:
        print("  %.4f +/- %.4f  th=%.2f maxb=%d gate=%.1f" % (m, s, cfg[0], cfg[1], cfg[2]))
    print("\nGate cost (th=0.15 maxb=3):")
    for gate in [0.0, 0.3, 0.4, 0.5]:
        v = agg[(0.15, 3, gate)]
        print("  gate=%.1f -> %.4f (positives-only)" % (gate, np.mean(v)))
    print("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

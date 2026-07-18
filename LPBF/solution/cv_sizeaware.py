"""Size-aware detection: score each (location, size) pair with the ranker and
pick the size with max confidence per location, instead of a fixed per-anchor
median size. Parallelised precompute. 5-fold CV against the official metric."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C
import detect as D
from features import FeatureExtractor
from metric import full_score

PUB = C.find_public_dir(None)
POS_R = {"gray": 4.0, "color": 7.0}
SIZE_TOL = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 2.0
FAMS = ["gray", "color"]
GRID = list(range(17, 38, 2))
JOBS = 14
_ANCH = None
_PRIOR = None


def _odd(s):
    s = int(round(s)); return max(15, min(41, s + (s % 2 == 0)))


def _locations(fe, anchors, fam):
    locs = [dict(cx=float(a["cx"]), cy=float(a["cy"]), acount=float(a["count"]), adist=0.0) for a in anchors]
    peaks = C.peak_candidates(fe.sal, 55, 8, 0.30) if fam == "color" else C.peak_candidates(fe.sal, 40, 9, 0.35)
    for (px, py) in peaks:
        d = min((abs(a["cx"] - px) + abs(a["cy"] - py) for a in anchors), default=99.0)
        if d > 4:
            locs.append(dict(cx=float(px), cy=float(py), acount=0.0, adist=float(d)))
    return locs


def _one(row):
    fam = C.family(row["height"])
    bgr = C.load_bgr(PUB, row["image_path"])
    if bgr is None:
        return None
    fe = FeatureExtractor(bgr, fam, prior_hm=_PRIOR[fam])
    gcen = [((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in C.parse_boxes(row["boxes"])]
    L = []
    for lo in _locations(fe, _ANCH[fam], fam):
        feats = np.stack([fe.features(lo["cx"], lo["cy"], s, lo["acount"], lo["adist"]) for s in GRID]).astype(np.float32)
        labs = np.zeros(len(GRID), np.int8)
        match = next(((gx, gy, gs) for (gx, gy, gs) in gcen
                      if abs(gx - lo["cx"]) <= POS_R[fam] and abs(gy - lo["cy"]) <= POS_R[fam]), None)
        if match is not None:
            for i, s in enumerate(GRID):
                if abs(s - match[2]) <= SIZE_TOL:
                    labs[i] = 1
        boxes = [C.clip_box(D.box_from(lo["cx"], lo["cy"], _odd(s)), fe.W, fe.H) for s in GRID]
        L.append(dict(feats=feats, labs=labs, boxes=boxes))
    return dict(iid=row["image_id"], fam=fam, gts=C.parse_boxes(row["boxes"]), locs=L)


def precompute(df):
    rows = [dict(image_id=r["image_id"], image_path=r["image_path"], height=r["height"], boxes=r["boxes"])
            for _, r in df.iterrows()]
    res = Parallel(n_jobs=JOBS, backend="multiprocessing")(delayed(_one)(r) for r in rows)
    return [x for x in res if x is not None]


def folds(df, k, seed=42):
    rng = np.random.RandomState(seed); assign = {}
    for fam in FAMS:
        ids = [i for i in df.index if C.family(df.loc[i, "height"]) == fam]
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            assign[i] = j % k
    return {df.loc[i, "image_id"]: assign[i] for i in df.index}


def score_val(va, model):
    """Score each image once: per location pick the max-confidence size."""
    out = []
    for im in va:
        if not im["locs"]:
            out.append((im["iid"], im["gts"], [])); continue
        # batch all locations x sizes into one predict call
        big = np.concatenate([lo["feats"] for lo in im["locs"]])
        p = model.predict_proba(big)[:, 1]
        off = 0; lb = []
        for lo in im["locs"]:
            n = len(lo["boxes"]); pp = p[off:off + n]; off += n
            j = int(np.argmax(pp)); lb.append((float(pp[j]), lo["boxes"][j]))
        out.append((im["iid"], im["gts"], lb))
    return out


def select(scored, th, min_keep=1, max_box=6, nms=0.30):
    tm, pm = {}, {}
    for iid, gts, lb in scored:
        tm[iid] = gts
        boxes = [b for _, b in lb]; scores = [s for s, _ in lb]
        keep = C.nms(boxes, scores, iou_thr=nms)
        kept = sorted([(scores[i], boxes[i]) for i in keep], key=lambda z: -z[0])
        o = [kb for kb in kept if kb[0] >= th][:max_box]
        if len(o) < min_keep and kept:
            o = kept[:min_keep]
        pm[iid] = [(s, b[0], b[1], b[2], b[3]) for (s, b) in o]
    return full_score(pm, tm)


def main():
    global _ANCH, _PRIOR
    t0 = time.time()
    df = pd.read_csv(os.path.join(PUB, "train.csv")).reset_index(drop=True)
    _ANCH = {f: C.build_anchors(df, f) for f in FAMS}
    _PRIOR = {f: C.prior_heatmap(df, f, 448 if f == "color" else 358, 448) for f in FAMS}
    data = precompute(df)
    print("precompute %.1fs sizes=%d tol=%.0f" % (time.time() - t0, len(GRID), SIZE_TOL))
    id2f = folds(df, 5)
    ths = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    sc = {t: [] for t in ths}; bd = {t: [] for t in ths}
    for k in range(5):
        tr = [d for d in data if id2f[d["iid"]] != k]; va = [d for d in data if id2f[d["iid"]] == k]
        X = np.concatenate([lo["feats"] for d in tr for lo in d["locs"]])
        y = np.concatenate([lo["labs"] for d in tr for lo in d["locs"]])
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=20, random_state=0).fit(X, y)
        scored = score_val(va, model)
        for t in ths:
            s, d = select(scored, t); sc[t].append(s); bd[t].append(d)
    best = None
    print("size-aware CV (mean +/- std):")
    for t in ths:
        m, s = np.mean(sc[t]), np.std(sc[t]); d = bd[t]
        print("  th=%.2f  %.4f +/- %.4f  @.50=%.3f @.75=%.3f @.85=%.3f" %
              (t, m, s, np.mean([x["m50"] for x in d]), np.mean([x["m75"] for x in d]), np.mean([x["m85"] for x in d])))
        if best is None or m > best[0]:
            best = (m, s, t)
    print("BEST th=%.2f CV=%.4f +/- %.4f  (total %.1fs)" % (best[2], best[0], best[1], time.time() - t0))


if __name__ == "__main__":
    main()

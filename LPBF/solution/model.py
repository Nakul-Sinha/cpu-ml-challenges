"""Train a single presence ranker + size regressor (family is a feature),
evaluate on the val split with the official metric. Caches per-image candidate
scores so selection strategies sweep cheaply. Research/iteration script."""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import common as C
import detect as D
from features import FeatureExtractor, SIZE_GRID
from metric import full_score

PUB = C.find_public_dir(sys.argv[1] if len(sys.argv) > 1 else None)
POS_R = {"gray": 4.0, "color": 7.0}
FAMS = ["gray", "color"]


def _odd(s):
    s = int(round(s))
    return max(15, min(41, s + (s % 2 == 0)))


def size_feat(fe, c):
    prof = fe.size_profile(c["cx"], c["cy"], SIZE_GRID)
    extra = np.array([c["s"] / 30.0, c["cx"] / fe.W, c["cy"] / fe.H,
                      1.0 if fe.fam == "color" else 0.0], np.float32)
    return np.concatenate([prof, extra])


def build_rows(df, anchors, prior):
    Xc, yc, Xs, ys = [], [], [], []
    cov_hit = {f: 0 for f in FAMS}; cov_tot = {f: 0 for f in FAMS}
    for _, r in df.iterrows():
        fam = C.family(r["height"])
        bgr = C.load_bgr(PUB, r["image_path"])
        if bgr is None:
            continue
        fe = FeatureExtractor(bgr, fam, prior_hm=prior[fam])
        gts = C.parse_boxes(r["boxes"])
        gcen = [((x0 + x1) / 2.0, (y0 + y1) / 2.0, max(x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in gts]
        cands = D.gen_candidates(fe, anchors[fam], fam)
        for c in cands:
            Xc.append(D.featurize(fe, c))
            match = None
            for (gx, gy, gs) in gcen:
                if abs(gx - c["cx"]) <= POS_R[fam] and abs(gy - c["cy"]) <= POS_R[fam]:
                    match = gs; break
            yc.append(0 if match is None else 1)
            if match is not None:
                Xs.append(size_feat(fe, c)); ys.append(match)
    return (np.stack(Xc), np.array(yc), np.stack(Xs), np.array(ys))


def coverage(df, anchors, prior):
    hit = {f: 0.0 for f in FAMS}; tot = {f: 0.0 for f in FAMS}
    for _, r in df.iterrows():
        fam = C.family(r["height"])
        bgr = C.load_bgr(PUB, r["image_path"])
        if bgr is None:
            continue
        fe = FeatureExtractor(bgr, fam, prior_hm=prior[fam])
        cands = D.gen_candidates(fe, anchors[fam], fam)
        for (x0, y0, x1, y1) in C.parse_boxes(r["boxes"]):
            gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
            tot[fam] += 1
            if any(abs(gx - c["cx"]) <= POS_R[fam] and abs(gy - c["cy"]) <= POS_R[fam] for c in cands):
                hit[fam] += 1
    for f in FAMS:
        print("  coverage[%s] = %.1f%% (%d gt)" % (f, 100 * hit[f] / max(1, tot[f]), tot[f]))


def cache_val(df, anchors, prior, cls, reg):
    cache = []
    for _, r in df.iterrows():
        fam = C.family(r["height"])
        bgr = C.load_bgr(PUB, r["image_path"])
        if bgr is None:
            continue
        fe = FeatureExtractor(bgr, fam, prior_hm=prior[fam])
        cands = D.gen_candidates(fe, anchors[fam], fam)
        rec = []
        if cands:
            p = cls.predict_proba(np.stack([D.featurize(fe, c) for c in cands]))[:, 1]
            spred = reg.predict(np.stack([size_feat(fe, c) for c in cands]))
            for c, pr, sp in zip(cands, p, spred):
                rec.append(dict(score=float(pr), cx=c["cx"], cy=c["cy"],
                                box_def=C.clip_box(D.box_from(c["cx"], c["cy"], _odd(c["s"])), fe.W, fe.H),
                                box_reg=C.clip_box(D.box_from(c["cx"], c["cy"], _odd(sp)), fe.W, fe.H)))
        cache.append(dict(image_id=r["image_id"], fam=fam, gts=C.parse_boxes(r["boxes"]), cands=rec))
    return cache


def select(cache, boxkey="box_def", thresh=0.5, iou_nms=0.30, min_keep=1, max_box=25, fam=None):
    tm, pm = {}, {}
    for im in cache:
        if fam and im["fam"] != fam:
            continue
        tm[im["image_id"]] = im["gts"]
        boxes = [c[boxkey] for c in im["cands"]]; scores = [c["score"] for c in im["cands"]]
        keep = C.nms(boxes, scores, iou_thr=iou_nms)
        kept = sorted([(scores[i], boxes[i]) for i in keep], key=lambda z: -z[0])
        out = [kb for kb in kept if kb[0] >= thresh][:max_box]
        if len(out) < min_keep:
            out = kept[:min_keep]
        pm[im["image_id"]] = [(s, b[0], b[1], b[2], b[3]) for (s, b) in out]
    return full_score(pm, tm)


def oracle(cache, mode="def"):
    tm, pm = {}, {}
    for im in cache:
        tm[im["image_id"]] = im["gts"]
        gcen = [((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in im["gts"]]
        R = POS_R[im["fam"]]; boxes = []
        for c in im["cands"]:
            m = next((gs for (gx, gy, gs) in gcen if abs(gx - c["cx"]) <= R and abs(gy - c["cy"]) <= R), None)
            if m is None:
                continue
            boxes.append(c["box_def"] if mode == "def" else c["box_reg"] if mode == "reg"
                         else C.clip_box(D.box_from(c["cx"], c["cy"], _odd(m)), 448, 448))
        keep = C.nms(boxes, [1.0] * len(boxes), iou_thr=0.30)
        pm[im["image_id"]] = [(1.0, *boxes[i]) for i in keep]
    return full_score(pm, tm)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(PUB, "train.csv"))
    tr, va = C.make_split(df)
    anchors = {f: C.build_anchors(tr, f) for f in FAMS}
    prior = {f: C.prior_heatmap(tr, f, 448 if f == "color" else 358, 448) for f in FAMS}
    print("anchors gray=%d color=%d" % (len(anchors["gray"]), len(anchors["color"])))
    coverage(va, anchors, prior)
    Xc, yc, Xs, ys = build_rows(tr, anchors, prior)
    print("cls rows=%d pos=%.1f%% dim=%d | reg rows=%d dim=%d (%.0fs)"
          % (len(yc), 100 * yc.mean(), Xc.shape[1], len(ys), Xs.shape[1], time.time() - t0))
    cls = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=20, random_state=0).fit(Xc, yc)
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=15, random_state=0).fit(Xs, ys)
    cache = cache_val(va, anchors, prior, cls, reg)
    print("built (%.0fs)\n" % (time.time() - t0))

    for boxkey in ["box_def", "box_reg"]:
        best = max((select(cache, boxkey=boxkey, thresh=th, min_keep=mk, max_box=mb, iou_nms=nm)
                    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                    for mb in [3, 4, 5, 6, 8, 25] for mk in [1] for nm in [0.2, 0.3]),
                   key=lambda z: z[0])
        sc, d = best
        print("%-8s SCORE=%.4f @.50=%.3f @.75=%.3f @.85=%.3f rec=%.3f"
              % (boxkey, sc, d["m50"], d["m75"], d["m85"], d["rec"]))
    for fam in FAMS:
        b1 = max((select(cache, boxkey="box_def", thresh=th, min_keep=1, max_box=mb, fam=fam)
                  for th in [0.1, 0.15, 0.2, 0.3] for mb in [4, 6, 25]), key=lambda z: z[0])
        print("  [%s] SCORE=%.4f @.50=%.3f @.75=%.3f @.85=%.3f"
              % (fam, b1[0], b1[1]["m50"], b1[1]["m75"], b1[1]["m85"]))
    for mode in ["def", "reg", "true"]:
        sc, d = oracle(cache, mode=mode)
        print("ORACLE %-4s SCORE=%.4f @.50=%.3f @.75=%.3f @.85=%.3f" % (mode, sc, d["m50"], d["m75"], d["m85"]))
    print("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

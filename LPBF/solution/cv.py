"""K-fold cross-validation of the full pipeline against the official metric.
Per-image feature extraction is parallelised across cores (fork backend shares
the read-only anchors/prior), so a full 5-fold sweep runs in a few seconds on the
16-core box. Usage: python cv.py [--patch] [--folds N] [--jobs J]"""
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
USE_PATCH = "--patch" in sys.argv
NFOLD = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 5
JOBS = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 14

_ANCH = None
_PRIOR = None
_BG = None


def _odd(s):
    s = int(round(s)); return max(15, min(41, s + (s % 2 == 0)))


def _one(row):
    fam = C.family(row["height"])
    bgr = C.load_bgr(PUB, row["image_path"])
    if bgr is None:
        return None
    fe = FeatureExtractor(bgr, fam, prior_hm=_PRIOR[fam], use_patch=USE_PATCH, bg=_BG[fam])
    cands = D.gen_candidates(fe, _ANCH[fam], fam)
    if not cands:
        return dict(iid=row["image_id"], fam=fam, gts=C.parse_boxes(row["boxes"]),
                    feats=np.zeros((0, 1), np.float32), labs=np.zeros(0), boxes=[])
    feats = np.stack([D.featurize(fe, c) for c in cands]).astype(np.float32)
    gts = C.parse_boxes(row["boxes"])
    gc = [((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in gts]
    labs, boxes, cx, cy, sz, rtgt = [], [], [], [], [], []
    for c in cands:
        cx.append(c["cx"]); cy.append(c["cy"]); sz.append(_odd(c["s"]))
        boxes.append(C.clip_box(D.box_from(c["cx"], c["cy"], _odd(c["s"])), fe.W, fe.H))
        m = None
        for (gx, gy, gs) in gc:
            if abs(gx - c["cx"]) <= POS_R[fam] and abs(gy - c["cy"]) <= POS_R[fam]:
                m = (gx, gy, gs); break
        labs.append(0 if m is None else 1)
        rtgt.append((m[0] - c["cx"], m[1] - c["cy"], m[2] - _odd(c["s"])) if m else (np.nan, np.nan, np.nan))
    sprof = np.stack([fe.size_profile(c["cx"], c["cy"], list(range(15, 40, 2))) for c in cands]).astype(np.float32)
    return dict(iid=row["image_id"], fam=fam, gts=gts, feats=feats, labs=np.array(labs),
                boxes=boxes, cx=np.array(cx), cy=np.array(cy), sz=np.array(sz),
                rtgt=np.array(rtgt, np.float32), sprof=sprof)


def precompute(df):
    rows = [dict(image_id=r["image_id"], image_path=r["image_path"],
                 height=r["height"], boxes=r["boxes"]) for _, r in df.iterrows()]
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


def score_val(va, models, iou_nms):
    """Score each val image once; average proba over a list of models (ensemble);
    return per-image nms-kept (score,box) list."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    out = []
    for im in va:
        if len(im["boxes"]) == 0:
            out.append((im["iid"], im["gts"], [])); continue
        sc = np.mean([m.predict_proba(im["feats"])[:, 1] for m in models], axis=0)
        keep = C.nms(im["boxes"], list(sc), iou_thr=iou_nms)
        kept = sorted([(float(sc[i]), im["boxes"][i]) for i in keep], key=lambda z: -z[0])
        out.append((im["iid"], im["gts"], kept))
    return out


def select(scored, th, min_keep=1, max_box=6):
    tm, pm = {}, {}
    for iid, gts, kept in scored:
        tm[iid] = gts
        o = [kb for kb in kept if kb[0] >= th][:max_box]
        if len(o) < min_keep and kept:
            o = kept[:min_keep]
        pm[iid] = [(s, b[0], b[1], b[2], b[3]) for (s, b) in o]
    return full_score(pm, tm)


def main():
    global _ANCH, _PRIOR, _BG
    t0 = time.time()
    df = pd.read_csv(os.path.join(PUB, "train.csv")).reset_index(drop=True)
    _ANCH = {f: C.build_anchors(df, f) for f in FAMS}
    _PRIOR = {f: C.prior_heatmap(df, f, 448 if f == "color" else 358, 448) for f in FAMS}
    _BG = {f: C.build_background(df, f, PUB) for f in FAMS}
    print("anchors gray=%d color=%d patch=%s folds=%d jobs=%d" %
          (len(_ANCH["gray"]), len(_ANCH["color"]), USE_PATCH, NFOLD, JOBS))
    data = precompute(df)
    print("precompute %.1fs dim=%d n=%d" % (time.time() - t0, data[0]["feats"].shape[1], len(data)))
    for fam in FAMS:
        sub = [d for d in data if d["fam"] == fam]
        ngt = sum(len(d["gts"]) for d in sub)
        avgc = np.mean([len(d["labs"]) for d in sub])
        pos = sum(int(d["labs"].sum()) for d in sub)
        print("  [%s] gt=%d avg_cands=%.1f pos_cands=%d" % (fam, ngt, avgc, pos))
    id2fold = folds(df, NFOLD)

    from collections import defaultdict
    MODELS = {
        "base": dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=20),
        "deep": dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=1.0, min_samples_leaf=15),
        "reg":  dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=3.0, min_samples_leaf=40),
    }
    ths = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2]
    nmss = [0.3]
    maxbs = [4, 6, 8]
    agg = defaultdict(list)
    for k in range(NFOLD):
        tr = [d for d in data if id2fold[d["iid"]] != k]
        va = [d for d in data if id2fold[d["iid"]] == k]
        X = np.concatenate([d["feats"] for d in tr if len(d["labs"])])
        y = np.concatenate([d["labs"] for d in tr if len(d["labs"])])
        fitted = {mn: HistGradientBoostingClassifier(random_state=0, **mp).fit(X, y)
                  for mn, mp in MODELS.items()}
        # extra seeds of the regularised model for a bagged ensemble
        seeds = [HistGradientBoostingClassifier(random_state=s, **MODELS["reg"]).fit(X, y)
                 for s in (1, 2, 3)]
        eval_sets = {"base": [fitted["base"]], "reg": [fitted["reg"]],
                     "ens3": [fitted["base"], fitted["reg"], fitted["deep"]],
                     "ensReg4": [fitted["reg"]] + seeds}
        for mname, mods in eval_sets.items():
            for nm in nmss:
                scored = score_val(va, mods, nm)
                for th in ths:
                    for mb in maxbs:
                        sc, _ = select(scored, th, max_box=mb)
                        agg[(mname, nm, th, mb)].append(sc)
    # presence-perfect oracle (emit positive-labelled candidate boxes) = ceiling
    tm, pm = {}, {}
    for d in data:
        tm[d["iid"]] = d["gts"]
        bx = [d["boxes"][i] for i in range(len(d["boxes"])) if d["labs"][i] == 1]
        keep = C.nms(bx, [1.0] * len(bx), iou_thr=0.30)
        pm[d["iid"]] = [(1.0, *bx[i]) for i in keep]
    osc, od = full_score(pm, tm)
    print("ORACLE (presence-perfect, prior size) = %.4f  @.50=%.3f @.75=%.3f @.85=%.3f"
          % (osc, od["m50"], od["m75"], od["m85"]))

    # recall vs precision diagnostic (OOF, base model, th=0.10, maxb=8)
    def _iou(a, b):
        ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1]); ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
        iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0); inter = iw * ih
        if inter <= 0: return 0.0
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    covered = matched = ngt = emitted = tp = 0
    for k in range(NFOLD):
        tr = [d for d in data if id2fold[d["iid"]] != k]; va = [d for d in data if id2fold[d["iid"]] == k]
        X = np.concatenate([d["feats"] for d in tr if len(d["labs"])]); y = np.concatenate([d["labs"] for d in tr if len(d["labs"])])
        mdl = HistGradientBoostingClassifier(random_state=0, **MODELS["base"]).fit(X, y)
        sc = score_val(va, mdl, 0.30)
        by = {d["iid"]: d for d in va}
        for iid, gts, kept in sc:
            d = by[iid]
            emit = [b for s, b in kept if s >= 0.10][:8]
            emitted += len(emit)
            for g in gts:
                ngt += 1
                if any(d["labs"][i] == 1 and _iou(d["boxes"][i], g) >= 0.5 for i in range(len(d["boxes"]))):
                    covered += 1
                if any(_iou(e, g) >= 0.5 for e in emit):
                    matched += 1
            for e in emit:
                if any(_iou(e, g) >= 0.5 for g in gts):
                    tp += 1
    print("DIAG: coverage(recall ceiling)=%.1f%%  model recall@.5=%.1f%%  precision@.5=%.1f%%  (emit/img=%.1f)"
          % (100*covered/ngt, 100*matched/ngt, 100*tp/max(1, emitted), emitted/len(data)))

    # box-offset regression: predict (dcx,dcy,ds) to the true box. Paired
    # comparison (same folds) of unrefined vs refined at fixed configs.
    REGP = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=25, random_state=0)
    def augf(d):
        return np.concatenate([d["feats"], d["sprof"]], axis=1)
    def refined_boxes(d, cls, regs):
        p = cls.predict_proba(d["feats"])[:, 1]
        A = augf(d)
        dcx = regs[0].predict(A); dcy = regs[1].predict(A); ds = regs[2].predict(A)
        rb = [C.clip_box(D.box_from(d["cx"][i] + dcx[i], d["cy"][i] + dcy[i], _odd(d["sz"][i] + ds[i])), 448, 448)
              for i in range(len(d["feats"]))]
        return p, rb
    cfgs = [(0.03, 10), (0.04, 8), (0.05, 8), (0.06, 8), (0.08, 8)]
    unref = {c: [] for c in cfgs}; refd = {c: [] for c in cfgs}; b85 = {c: [] for c in cfgs}; b75 = {c: [] for c in cfgs}
    for k in range(NFOLD):
        tr = [d for d in data if id2fold[d["iid"]] != k]; va = [d for d in data if id2fold[d["iid"]] == k]
        X = np.concatenate([d["feats"] for d in tr if len(d["labs"])]); y = np.concatenate([d["labs"] for d in tr if len(d["labs"])])
        cls = HistGradientBoostingClassifier(random_state=0, **MODELS["base"]).fit(X, y)
        Xp = np.concatenate([augf(d)[d["labs"] == 1] for d in tr if (d["labs"] == 1).any()])
        T = np.concatenate([d["rtgt"][d["labs"] == 1] for d in tr if (d["labs"] == 1).any()])
        regs = [HistGradientBoostingRegressor(**REGP).fit(Xp, T[:, j]) for j in range(3)]
        su, sr = [], []
        for d in va:
            if not len(d["feats"]):
                su.append((d["iid"], d["gts"], [])); sr.append((d["iid"], d["gts"], [])); continue
            p, rb = refined_boxes(d, cls, regs)
            ku = C.nms(d["boxes"], list(p), 0.30)
            su.append((d["iid"], d["gts"], sorted([(float(p[i]), d["boxes"][i]) for i in ku], key=lambda z: -z[0])))
            kr = C.nms(rb, list(p), 0.30)
            sr.append((d["iid"], d["gts"], sorted([(float(p[i]), rb[i]) for i in kr], key=lambda z: -z[0])))
        for c in cfgs:
            unref[c].append(select(su, c[0], max_box=c[1])[0])
            s, dd = select(sr, c[0], max_box=c[1]); refd[c].append(s); b85[c].append(dd["m85"]); b75[c].append(dd["m75"])
    print("PAIRED unrefined vs refined (base model, size-profile-augmented regressor):")
    for c in cfgs:
        print("  th=%.2f mb=%d : unref=%.4f  refined=%.4f (+%.4f) @.75ref=%.3f @.85ref=%.3f" %
              (c[0], c[1], np.mean(unref[c]), np.mean(refd[c]), np.mean(refd[c]) - np.mean(unref[c]),
               np.mean(b75[c]), np.mean(b85[c])))

    results = sorted(((np.mean(v), np.std(v), cfg) for cfg, v in agg.items()), reverse=True)
    print("\nTop 12 configs (CV mean +/- std):")
    for m, s, cfg in results[:12]:
        print("  %.4f +/- %.4f  model=%s nms=%.1f th=%.2f maxb=%d" % (m, s, cfg[0], cfg[1], cfg[2], cfg[3]))
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

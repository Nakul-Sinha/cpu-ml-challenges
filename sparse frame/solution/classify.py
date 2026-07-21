"""Dedicated category classifier on detected-box geometry + appearance features.
Compares CNN class head vs a GBM on features (from GT boxes = ceiling, from detected = realistic).
Usage: python classify.py <root> <ckpt> [--res 256x144]"""
import sys, os, argparse, collections
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CAT2I, FRAME_W, FRAME_H, read_train_csv, stratified_split
import proto3

def feats_from_track(cen, siz):
    """cen,siz: (>=4,2) normalized. Geometry + trajectory features (category-discriminative)."""
    w = siz[:4, 0]*FRAME_W; h = siz[:4, 1]*FRAME_H
    aspect = w/np.maximum(h, 1e-3); area = w*h
    cx = cen[:4, 0]*FRAME_W; cy = cen[:4, 1]*FRAME_H
    dx = np.diff(cx); dy = np.diff(cy); speed = np.sqrt(dx**2+dy**2)
    f = [
        w.mean(), h.mean(), np.median(w), np.median(h),
        aspect.mean(), np.median(aspect), aspect.std(),
        area.mean(), np.median(area), np.log(np.median(area)+1),
        w[-1], h[-1], aspect[-1], area[-1],
        (w.std()+h.std()), speed.mean(), speed.max(),
        cy.mean()/FRAME_H, cx.std(),
    ]
    return np.array(f, np.float32)

def brier_macro(probs, y, cats):
    per = collections.defaultdict(list)
    for p, g in zip(probs, y):
        oh = np.zeros(4); oh[g] = 1; per[g].append(1-0.5*np.sum((p-oh)**2))
    return float(np.mean([np.mean(per[c]) for c in range(4) if per[c]]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("ckpt")
    ap.add_argument("--res", default="256x144"); ap.add_argument("--cache", default="/mnt/work/data/cache")
    args = ap.parse_args()
    outW, outH = map(int, args.res.split("x")); gh, gw = outH//4, outW//4; nf = 4
    clips, X, cen, siz, cls, cat = proto3.build_cache(args.root, outW, outH, args.cache)
    net = proto3.Net(); ck = torch.load(args.ckpt, map_location="cpu"); net.load_state_dict(ck["state"]); net.eval()
    tr_clips, va_clips = stratified_split(clips, cat, 0.15, 0)
    idx = {c: i for i, c in enumerate(clips)}
    tr = [idx[c] for c in tr_clips]; va = [idx[c] for c in va_clips]

    # decode detected tracks + cnn probs
    detcen = {}; detsiz = {}; cnnprob = {}
    for i in range(len(clips)):
        c, s, p = proto3.decode(net, X, i, outW, outH, gh, gw, nf)
        detcen[i] = c; detsiz[i] = s; cnnprob[i] = p

    def build(idxs, use_gt):
        Xf = []; y = []
        for i in idxs:
            if use_gt: f = feats_from_track(cen[i], siz[i])
            else: f = feats_from_track(detcen[i], detsiz[i])
            Xf.append(f); y.append(int(cls[i]))
        return np.array(Xf), np.array(y)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    for use_gt in [True, False]:
        Xtr, ytr = build(tr, use_gt); Xva, yva = build(va, use_gt)
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.08,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(Xtr, ytr)
        pva = clf.predict_proba(Xva)
        acc = accuracy_score(yva, pva.argmax(1)); bm = brier_macro(pva, yva, cat)
        tag = "GT-box (ceiling)" if use_gt else "DETECTED-box (realistic)"
        print(f"GBM on {tag:26s}: clsAcc={acc:.3f}  macroBrier={bm:.4f}")
    # CNN head baseline on val
    cnn_pva = np.array([cnnprob[i] for i in va]); yva = np.array([int(cls[i]) for i in va])
    print(f"CNN class head              : clsAcc={accuracy_score(yva, cnn_pva.argmax(1)):.3f}  "
          f"macroBrier={brier_macro(cnn_pva, yva, cat):.4f}")

if __name__ == "__main__":
    main()

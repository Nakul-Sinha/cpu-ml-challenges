"""Classification ceiling with geometry + APPEARANCE features (blob shape/polarity/fill
inside the box from raw masks). Usage: python cls_appearance.py <root>"""
import sys, os, collections
import numpy as np
from PIL import Image
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CAT2I, FRAME_W, FRAME_H, read_train_csv, stratified_split, masks
from cls_ceiling import feats_from_track, brier_macro
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

root = sys.argv[1]
boxes, cat = read_train_csv(os.path.join(root, "train.csv"))
clips = sorted(boxes.keys())
IMG = os.path.join(root, "images", "train")

def appearance(red, blue, box):
    x, y, w, h = box
    H, W = red.shape
    x0 = int(max(0, x)); y0 = int(max(0, y)); x1 = int(min(W, x+w)); y1 = int(min(H, y+h))
    if x1-x0 < 2 or y1-y0 < 2:
        return np.zeros(9, np.float32)
    r = red[y0:y1, x0:x1]; b = blue[y0:y1, x0:x1]; lit = (r+b) > 0
    area = lit.size; nlit = lit.sum()
    fill = nlit/area
    pol = r.sum()/(r.sum()+b.sum()+1e-3)
    # top/bottom and left/right fill (structure)
    hh = lit.shape[0]; ww = lit.shape[1]
    topf = lit[:hh//2].mean(); botf = lit[hh//2:].mean()
    leftf = lit[:, :ww//2].mean(); rightf = lit[:, ww//2:].mean()
    # largest connected component solidity + count
    m = lit.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]; big = areas.max()
        solidity = big/max(1, nlit); ncc = int((areas > 3).sum())
    else:
        solidity = 0.0; ncc = 0
    # vertical/horizontal spread of lit pixels (normalized std)
    ys, xs = np.where(lit)
    vspread = ys.std()/max(1, hh) if len(ys) else 0
    hspread = xs.std()/max(1, ww) if len(xs) else 0
    return np.array([fill, pol, topf-botf, leftf-rightf, solidity, ncc, vspread, hspread, nlit/ (area+1)], np.float32)

def clip_feats(clip):
    cen = np.array([[(boxes[clip][t][0]+boxes[clip][t][2]/2)/FRAME_W,
                     (boxes[clip][t][1]+boxes[clip][t][3]/2)/FRAME_H] for t in range(4)])
    siz = np.array([[boxes[clip][t][2]/FRAME_W, boxes[clip][t][3]/FRAME_H] for t in range(4)])
    geo = feats_from_track(cen, siz)
    apps = []
    for t in range(4):
        img = np.asarray(Image.open(os.path.join(IMG, clip, f"t{t}.png")).convert("RGB"))
        red, blue = masks(img)
        apps.append(appearance(red, blue, boxes[clip][t]))
    apps = np.array(apps)
    app = np.concatenate([apps.mean(0), apps.std(0), apps[-1]])  # aggregate
    return np.concatenate([geo, app])

def main():
    tr_clips, va_clips = stratified_split(clips, cat, 0.15, 0)
    import time; t0 = time.time()
    feat = {}
    for i, c in enumerate(clips):
        feat[c] = clip_feats(c)
        if (i+1) % 200 == 0: print(f"  feats {i+1}/{len(clips)} ({time.time()-t0:.0f}s)", flush=True)
    Xtr = np.array([feat[c] for c in tr_clips]); ytr = np.array([CAT2I[cat[c]] for c in tr_clips])
    Xva = np.array([feat[c] for c in va_clips]); yva = np.array([CAT2I[cat[c]] for c in va_clips])
    clf = HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.06,
                                         l2_regularization=2.0, random_state=0)
    clf.fit(Xtr, ytr); pva = clf.predict_proba(Xva)
    print(f"GT-box GEO+APPEARANCE -> clsAcc={accuracy_score(yva, pva.argmax(1)):.3f}  macroBrier={brier_macro(pva, yva, cat):.4f}")
    cm = confusion_matrix(yva, pva.argmax(1)); print("confusion:\n", cm)
    for i, c in enumerate(CATS):
        print(f"  {c:7s} recall={cm[i, i]/max(1, cm[i].sum()):.3f}")

if __name__ == "__main__":
    main()

"""Classification ceiling: predict category from GT-box geometry/trajectory (no CNN).
Also reports per-frame single-box accuracy. Usage: python cls_ceiling.py <root>"""
import sys, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CAT2I, FRAME_W, FRAME_H, read_train_csv, stratified_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def feats_from_track(cen, siz):
    w = siz[:4, 0]*FRAME_W; h = siz[:4, 1]*FRAME_H
    aspect = w/np.maximum(h, 1e-3); area = w*h
    cx = cen[:4, 0]*FRAME_W; cy = cen[:4, 1]*FRAME_H
    dx = np.diff(cx); dy = np.diff(cy); speed = np.sqrt(dx**2+dy**2)
    return np.array([w.mean(), h.mean(), np.median(w), np.median(h), aspect.mean(), np.median(aspect),
                     aspect.std(), area.mean(), np.median(area), np.log(np.median(area)+1), w[-1], h[-1],
                     aspect[-1], area[-1], (w.std()+h.std()), speed.mean(), speed.max(),
                     cy.mean()/FRAME_H, cx.std()], np.float32)

def brier_macro(probs, y, cats):
    per = collections.defaultdict(list)
    for p, g in zip(probs, y):
        oh = np.zeros(4); oh[g] = 1; per[g].append(1-0.5*np.sum((p-oh)**2))
    return float(np.mean([np.mean(per[c]) for c in range(4) if per[c]]))

root = sys.argv[1]
boxes, cat = read_train_csv(os.path.join(root, "train.csv"))
clips = sorted(boxes.keys())
tr_clips, va_clips = stratified_split(clips, cat, 0.15, 0)

def track(clip):
    cen = np.array([[(boxes[clip][t][0]+boxes[clip][t][2]/2)/FRAME_W,
                     (boxes[clip][t][1]+boxes[clip][t][3]/2)/FRAME_H] for t in range(4)])
    siz = np.array([[boxes[clip][t][2]/FRAME_W, boxes[clip][t][3]/FRAME_H] for t in range(4)])
    return cen, siz

def build(cl):
    X = []; y = []
    for c in cl:
        cen, siz = track(c); X.append(feats_from_track(cen, siz)); y.append(CAT2I[cat[c]])
    return np.array(X), np.array(y)

Xtr, ytr = build(tr_clips); Xva, yva = build(va_clips)
clf = HistGradientBoostingClassifier(max_iter=400, max_depth=4, learning_rate=0.07,
                                     l2_regularization=1.0, random_state=0)
clf.fit(Xtr, ytr)
pva = clf.predict_proba(Xva)
print(f"GT-box track features -> clsAcc={accuracy_score(yva, pva.argmax(1)):.3f}  macroBrier={brier_macro(pva, yva, cat):.4f}")
print("confusion (rows=true people/car/cat/uav):")
print(confusion_matrix(yva, pva.argmax(1)))
# per-category recall
cm = confusion_matrix(yva, pva.argmax(1))
for i, c in enumerate(CATS):
    print(f"  {c:7s} recall={cm[i, i]/max(1, cm[i].sum()):.3f} (n={cm[i].sum()})")

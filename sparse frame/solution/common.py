"""Shared utilities: data loading, channel extraction, scoring (macro-MCFS)."""
import os, csv, collections
import numpy as np
from PIL import Image
import cv2

CATS = ["people", "car", "cat", "uav"]
CAT2I = {c: i for i, c in enumerate(CATS)}
FRAME_W, FRAME_H = 640, 360

def read_train_csv(path):
    boxes = collections.defaultdict(dict); cat = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            boxes[r["clip_id"]][int(r["frame_index"])] = (
                float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            cat[r["clip_id"]] = r["category"]
    return boxes, cat

def read_test_ids(path):
    ids = []
    with open(path) as f:
        for r in csv.DictReader(f):
            ids.append(r["clip_id"])
    return ids

def masks(img):
    R, G, B = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    red = ((R > 150) & (G < 110) & (B < 110)).astype(np.float32)
    blue = ((B > 150) & (R < 110) & (G < 110)).astype(np.float32)
    return red, blue

def frame_channels(img, outW, outH):
    """Return (2,outH,outW) float32 density (red,blue) via area-average downsample."""
    red, blue = masks(img)
    red_d = cv2.resize(red, (outW, outH), interpolation=cv2.INTER_AREA)
    blue_d = cv2.resize(blue, (outW, outH), interpolation=cv2.INTER_AREA)
    return np.stack([red_d, blue_d], 0)

def load_clip_input(img_dir, clip, outW, outH, nframes=4):
    """Stack t0..t{nframes-1} -> (2*nframes, outH, outW)."""
    chans = []
    for t in range(nframes):
        img = np.asarray(Image.open(os.path.join(img_dir, clip, f"t{t}.png")).convert("RGB"))
        chans.append(frame_channels(img, outW, outH))
    return np.concatenate(chans, 0)

def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by); x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1); inter = iw*ih
    u = aw*ah + bw*bh - inter
    return inter/u if u > 0 else 0.0

def clip_quality(pred_box, gt_box, prob, gt_cat_idx):
    """Per-clip MCFS quality = loc * cls."""
    iou = iou_xywh(pred_box, gt_box)
    loc = iou if iou >= 0.5 else 0.0
    onehot = np.zeros(4); onehot[gt_cat_idx] = 1.0
    prob = np.asarray(prob, float)
    cls = 1.0 - 0.5 * np.sum((prob - onehot) ** 2)
    return loc * cls, iou, cls

def macro_mcfs(pred_boxes, probs, gt_boxes_t4, gt_cats_idx):
    """pred_boxes: list of (x,y,w,h); probs: (N,4); gt_boxes_t4: list; gt_cats_idx: list."""
    per_cat = collections.defaultdict(list)
    ious = []; clss = []
    for pb, pr, gb, gc in zip(pred_boxes, probs, gt_boxes_t4, gt_cats_idx):
        q, iou, cls = clip_quality(pb, gb, pr, gc)
        per_cat[gc].append(q); ious.append(iou); clss.append(cls)
    cat_scores = {}
    for ci in range(4):
        if per_cat[ci]:
            cat_scores[ci] = float(np.mean(per_cat[ci]))
    macro = float(np.mean([cat_scores[ci] for ci in range(4) if ci in cat_scores]))
    return macro, cat_scores, float(np.mean(np.array(ious) >= 0.5)), float(np.mean(ious)), float(np.mean(clss))

def stratified_split(clips, cat, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    bycat = collections.defaultdict(list)
    for c in clips: bycat[cat[c]].append(c)
    train, val = [], []
    for c, cl in bycat.items():
        cl = sorted(cl); rng.shuffle(cl)
        n_val = max(1, int(round(len(cl) * val_frac)))
        val += cl[:n_val]; train += cl[n_val:]
    return sorted(train), sorted(val)

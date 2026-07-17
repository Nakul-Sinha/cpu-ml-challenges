"""Compare classical detectors' ability to recover GT boxes on train frames.
Methods: density-peak, erosion+largestCC, polarity-coherence, coherence+density.
Usage: python detect_compare.py <root> [n_sample]"""
import sys, os, csv, collections, time
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/work/eris/dataset/public"
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 250
TRAIN_IMG = os.path.join(ROOT, "images", "train")
rng = np.random.default_rng(0)

def load(clip, t):
    return np.asarray(Image.open(os.path.join(TRAIN_IMG, clip, f"t{t}.png")).convert("RGB"))

def gt_boxes(csv_path):
    d = collections.defaultdict(dict); cat = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            d[r["clip_id"]][int(r["frame_index"])] = (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            cat[r["clip_id"]] = r["category"]
    return d, cat

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by); x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    iw = max(0, x2-x1); ih = max(0, y2-y1); inter = iw*ih
    u = aw*ah + bw*bh - inter
    return inter/u if u > 0 else 0.0

def masks(img):
    R, G, B = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    red = ((R > 150) & (G < 110) & (B < 110)).astype(np.uint8)
    blue = ((B > 150) & (R < 110) & (G < 110)).astype(np.uint8)
    return red, blue

def box_from_points(xs, ys, lo=5, hi=95):
    if len(xs) < 4:
        return None
    x0, x1 = np.percentile(xs, [lo, hi]); y0, y1 = np.percentile(ys, [lo, hi])
    return (float(x0), float(y0), float(max(4, x1-x0)), float(max(4, y1-y0)))

# ---- detectors: each returns a box (x,y,w,h) ----
def det_density(img, sigma=15):
    red, blue = masks(img); m = (red | blue)
    d = gaussian_filter(m.astype(np.float32), sigma)
    if d.max() < 1e-6: return (300, 160, 40, 40)
    cy, cx = np.unravel_index(np.argmax(d), d.shape)
    ys, xs = np.where(m)
    sel = (np.abs(xs-cx) < 60) & (np.abs(ys-cy) < 60)
    b = box_from_points(xs[sel], ys[sel]) if sel.sum() >= 5 else None
    return b or (cx-20, cy-20, 40, 40)

def det_erode(img, r=2):
    red, blue = masks(img); m = (red | blue).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    e = cv2.erode(m, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(e, connectivity=8)
    if n <= 1: return det_density(img)
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    x, y, w, h, a = stats[i]
    # dilate box back a bit to compensate erosion
    return (float(x-r), float(y-r), float(w+2*r), float(h+2*r))

def det_coherence(img, sigma=12):
    """Object = region where one polarity dominates (|red_density - blue_density| high).
    Background camera-motion edges have mixed +/- so the diff cancels."""
    red, blue = masks(img)
    rd = gaussian_filter(red.astype(np.float32), sigma)
    bd = gaussian_filter(blue.astype(np.float32), sigma)
    coh = np.abs(rd - bd)
    if coh.max() < 1e-6: return det_density(img)
    cy, cx = np.unravel_index(np.argmax(coh), coh.shape)
    # dominant polarity at peak
    dom = red if rd[cy, cx] >= bd[cy, cx] else blue
    ys, xs = np.where(dom)
    sel = (np.abs(xs-cx) < 70) & (np.abs(ys-cy) < 70)
    b = box_from_points(xs[sel], ys[sel]) if sel.sum() >= 5 else None
    return b or (cx-20, cy-20, 40, 40)

def det_coh_erode(img, sigma=12, r=2):
    """Coherence to pick the region, then erode+CC within a window for a tight box."""
    red, blue = masks(img)
    rd = gaussian_filter(red.astype(np.float32), sigma)
    bd = gaussian_filter(blue.astype(np.float32), sigma)
    coh = np.abs(rd - bd)
    if coh.max() < 1e-6: return det_density(img)
    cy, cx = np.unravel_index(np.argmax(coh), coh.shape)
    dom = (red if rd[cy, cx] >= bd[cy, cx] else blue).astype(np.uint8)
    H, W = dom.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    e = cv2.erode(dom, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(e, connectivity=8)
    if n <= 1:
        ys, xs = np.where(dom)
        sel = (np.abs(xs-cx) < 70) & (np.abs(ys-cy) < 70)
        return box_from_points(xs[sel], ys[sel]) or (cx-20, cy-20, 40, 40)
    # pick component whose centroid is nearest the coherence peak, weighted by area
    best, bscore = None, -1
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        ccx, ccy = cent[i]
        dist = ((ccx-cx)**2 + (ccy-cy)**2)**0.5
        score = a / (1 + dist)
        if score > bscore: bscore, best = score, (x-r, y-r, w+2*r, h+2*r)
    return tuple(float(v) for v in best)

DETECTORS = {
    "density":   det_density,
    "erode_r2":  lambda im: det_erode(im, 2),
    "erode_r3":  lambda im: det_erode(im, 3),
    "coherence": det_coherence,
    "coh_erode": det_coh_erode,
}

def main():
    boxes, cats = gt_boxes(os.path.join(ROOT, "train.csv"))
    clips = sorted(boxes.keys())
    # stratified sample
    bycat = collections.defaultdict(list)
    for c in clips: bycat[cats[c]].append(c)
    sample = []
    for c, cl in bycat.items():
        cl = list(cl); rng.shuffle(cl)
        sample += cl[:max(1, NSAMP*len(cl)//len(clips))]
    print(f"sampled {len(sample)} clips")
    res = {name: collections.defaultdict(list) for name in DETECTORS}
    t0 = time.time()
    for i, clip in enumerate(sample):
        c = cats[clip]
        for t in range(4):
            if t not in boxes[clip]: continue
            img = load(clip, t); g = boxes[clip][t]
            for name, fn in DETECTORS.items():
                try: pred = fn(img)
                except Exception: pred = (300, 160, 40, 40)
                res[name][c].append(iou(pred, g))
        if (i+1) % 100 == 0: print(f"  ...{i+1}/{len(sample)} ({time.time()-t0:.0f}s)")
    print(f"\n{'method':11s} " + " ".join(f"{c:>16s}" for c in ["people","car","cat","uav"]) + f"  {'OVERALL':>16s}")
    for name in DETECTORS:
        cells = []
        allio = []
        for c in ["people","car","cat","uav"]:
            a = np.array(res[name][c]); allio.append(a)
            cells.append(f"{a.mean():.2f}/{np.mean(a>=0.5):.2f}")
        A = np.concatenate(allio)
        cells.append(f"{A.mean():.2f}/{np.mean(A>=0.5):.2f}")
        print(f"{name:11s} " + " ".join(f"{x:>16s}" for x in cells))
    print("(cells = meanIoU / frac IoU>=0.5)")

if __name__ == "__main__":
    main()

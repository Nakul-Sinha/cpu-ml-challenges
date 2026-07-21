"""Exploration: verify pixel palette + measure simple detector recovery vs GT boxes.
Reads dataset root (containing images/train, train.csv). Usage: python explore.py <root>"""
import sys, os, csv, collections, time
import numpy as np
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/work/data/dataset/public"
TRAIN_IMG = os.path.join(ROOT, "images", "train")
np.random.seed(0)

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

def motion_channels(img):
    R, G, B = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    red = (R > 150) & (G < 110) & (B < 110)
    blue = (B > 150) & (R < 110) & (G < 110)
    return red, blue

def box_from_density(mask, sigma=15):
    """Peak-density detector: gaussian-blur the mask, take peak, estimate box from local mass."""
    from scipy.ndimage import gaussian_filter
    d = gaussian_filter(mask.astype(np.float32), sigma)
    if d.max() < 1e-6:
        H, W = mask.shape
        return (W/2-20, H/2-20, 40, 40)
    cy, cx = np.unravel_index(np.argmax(d), d.shape)
    # estimate extent: pixels within window around peak weighted by mask
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (cx-20, cy-20, 40, 40)
    # keep mask pixels near the peak (within 2.5 sigma-ish adaptive radius)
    r = 60
    sel = (np.abs(xs-cx) < r) & (np.abs(ys-cy) < r)
    if sel.sum() < 5:
        return (cx-20, cy-20, 40, 40)
    xs2, ys2 = xs[sel], ys[sel]
    x0, x1 = np.percentile(xs2, [5, 95]); y0, y1 = np.percentile(ys2, [5, 95])
    return (x0, y0, max(4, x1-x0), max(4, y1-y0))

def main():
    boxes, cats = gt_boxes(os.path.join(ROOT, "train.csv"))
    clips = sorted(boxes.keys())
    # --- palette check on a few clips ---
    print("=== palette check ===")
    for clip in clips[:2]:
        img = load(clip, 0)
        cols, cnts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(-cnts)[:6]
        print(clip, "top colors:", [(tuple(cols[i]), int(cnts[i])) for i in order])
        red, blue = motion_channels(img)
        print("   red px:", int(red.sum()), "blue px:", int(blue.sum()), "of", img.shape[0]*img.shape[1])

    # --- detector recovery on a sample, per category ---
    print("\n=== density-peak detector recovery vs GT (t0-t3), by category ===")
    sample = clips  # all
    per_cat = collections.defaultdict(lambda: {"iou": [], "cerr": [], "n": 0})
    t0 = time.time()
    for i, clip in enumerate(sample):
        c = cats[clip]
        for t in range(4):
            if t not in boxes[clip]:
                continue
            img = load(clip, t)
            red, blue = motion_channels(img)
            mask = red | blue
            pred = box_from_density(mask)
            g = boxes[clip][t]
            per_cat[c]["iou"].append(iou(pred, g))
            pc = (pred[0]+pred[2]/2, pred[1]+pred[3]/2)
            gc = (g[0]+g[2]/2, g[1]+g[3]/2)
            per_cat[c]["cerr"].append(((pc[0]-gc[0])**2 + (pc[1]-gc[1])**2)**0.5)
        per_cat[c]["n"] += 1
        if (i+1) % 200 == 0:
            print(f"  ...{i+1}/{len(sample)} clips  ({time.time()-t0:.0f}s)")
    for c in ["people", "car", "cat", "uav"]:
        d = per_cat[c]
        iou_arr = np.array(d["iou"]); cerr = np.array(d["cerr"])
        print(f"{c:7s} n_clips={d['n']:3d}  frames={len(iou_arr):4d}  "
              f"meanIoU={iou_arr.mean():.3f}  IoU>=0.5={np.mean(iou_arr>=0.5):.3f}  "
              f"medianCenterErr={np.median(cerr):.1f}px")
    allio = np.concatenate([np.array(per_cat[c]["iou"]) for c in per_cat])
    print(f"\nOVERALL meanIoU={allio.mean():.3f}  IoU>=0.5={np.mean(allio>=0.5):.3f}  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()

"""Visualize LPBF alert boxes: overlays + box crops vs random crops.

Writes PNGs into working/ (gitignored) so we can eyeball what an alert region
looks like before designing detection cues.
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2

PUB = sys.argv[1] if len(sys.argv) > 1 else "dataset/public"
OUT = "working"
os.makedirs(OUT, exist_ok=True)


def parse_boxes(s):
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for tok in s.split():
        p = tok.split(",")
        if len(p) == 4:
            out.append([int(round(float(x))) for x in p])
    return out


def load(pub, path):
    img = cv2.imread(os.path.join(pub, path), cv2.IMREAD_COLOR)
    return img


def main():
    rng = np.random.RandomState(0)
    tr = pd.read_csv(os.path.join(PUB, "train.csv"))

    # 1) overlays: pick images with varied alert counts
    picks = []
    for c in [1, 2, 3, 4]:
        sub = tr[tr["alert_count"] == c]
        picks += list(sub.index[:2])
    tiles = []
    for idx in picks:
        r = tr.loc[idx]
        img = load(PUB, r["image_path"])
        if img is None:
            continue
        vis = img.copy()
        for (x0, y0, x1, y1) in parse_boxes(r["boxes"]):
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # pad to 448x448 for tiling
        canvas = np.zeros((448, 448, 3), np.uint8)
        canvas[:vis.shape[0], :vis.shape[1]] = vis
        tiles.append(canvas)
    # grid 4x2
    rows = []
    for i in range(0, len(tiles), 4):
        rows.append(np.hstack(tiles[i:i + 4]))
    grid = np.vstack(rows)
    cv2.imwrite(os.path.join(OUT, "overlays.png"), grid)
    print("wrote overlays.png", grid.shape)

    # 2) box crops vs random crops (context 2x), upsampled
    box_crops = []
    rand_crops = []
    ctx = 1.6
    up = 4
    for _, r in tr.iterrows():
        img = load(PUB, r["image_path"])
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bxs = parse_boxes(r["boxes"])
        for (x0, y0, x1, y1) in bxs:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            s = max(x1 - x0, y1 - y0)
            hw = int(s * ctx / 2)
            a = int(cx - hw); b = int(cy - hw); a2 = int(cx + hw); b2 = int(cy + hw)
            a = max(0, a); b = max(0, b); a2 = min(W, a2); b2 = min(H, b2)
            crop = gray[b:b2, a:a2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_NEAREST)
            box_crops.append(crop)
        if len(box_crops) > 64:
            break
    # random non-box crops from same images
    for _, r in tr.sample(40, random_state=1).iterrows():
        img = load(PUB, r["image_path"])
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bxs = parse_boxes(r["boxes"])
        for _ in range(3):
            s = 27
            cx = rng.randint(s, W - s); cy = rng.randint(s, H - s)
            # avoid overlapping a real box center
            ok = True
            for (x0, y0, x1, y1) in bxs:
                if abs((x0 + x1) / 2 - cx) < 30 and abs((y0 + y1) / 2 - cy) < 30:
                    ok = False
            if not ok:
                continue
            hw = int(s * ctx / 2)
            crop = gray[cy - hw:cy + hw, cx - hw:cx + hw]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_NEAREST)
            rand_crops.append(crop)
        if len(rand_crops) > 64:
            break

    def montage(crops, n_cols=8):
        crops = crops[:64]
        while len(crops) % n_cols:
            crops.append(np.zeros((48, 48), np.uint8))
        rows = []
        for i in range(0, len(crops), n_cols):
            rows.append(np.hstack([np.pad(c, 1, constant_values=255) for c in crops[i:i + n_cols]]))
        return np.vstack(rows)

    cv2.imwrite(os.path.join(OUT, "box_crops.png"), montage(box_crops))
    cv2.imwrite(os.path.join(OUT, "rand_crops.png"), montage(rand_crops))
    print("wrote box_crops.png (n=%d) and rand_crops.png (n=%d)" % (len(box_crops), len(rand_crops)))


if __name__ == "__main__":
    main()

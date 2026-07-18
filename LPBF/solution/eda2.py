"""EDA round 2: condition on image type (height), color signal, spatial-prior
transfer reliability, and per-type center heatmaps."""
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


def main():
    tr = pd.read_csv(os.path.join(PUB, "train.csv"))
    te = pd.read_csv(os.path.join(PUB, "test.csv"))

    print("=== boxes/img by height ===")
    for h in sorted(tr["height"].unique()):
        sub = tr[tr["height"] == h]
        ac = sub["alert_count"]
        print(f" H={h}: {len(sub)} imgs, alert_count mean={ac.mean():.2f} dist={dict(ac.value_counts().sort_index())}")

    # box side by height
    for h in sorted(tr["height"].unique()):
        sub = tr[tr["height"] == h]
        sides = []
        for _, r in sub.iterrows():
            for (x0, y0, x1, y1) in parse_boxes(r["boxes"]):
                sides.append(x1 - x0)
        sides = np.array(sides)
        print(f" H={h}: side p10/50/90 = {np.percentile(sides,[10,50,90])}, mean {sides.mean():.1f}")

    # color check: are images actually colored?
    print("\n=== COLOR CHECK (per-channel mean, and B-R diff) ===")
    for h in sorted(tr["height"].unique()):
        sub = tr[tr["height"] == h].head(20)
        bs, gs, rs = [], [], []
        for _, r in sub.iterrows():
            img = cv2.imread(os.path.join(PUB, r["image_path"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            b, g, rr = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
            bs.append(b); gs.append(g); rs.append(rr)
        print(f" H={h}: B={np.mean(bs):.1f} G={np.mean(gs):.1f} R={np.mean(rs):.1f} "
              f"(B-R={np.mean(bs)-np.mean(rs):.1f})  -> {'COLOR' if abs(np.mean(bs)-np.mean(rs))>8 else 'gray-ish'}")

    # spatial prior transfer: split train in half by index parity, measure if
    # a box center in half A has a near neighbor center in half B.
    print("\n=== SPATIAL PRIOR TRANSFER (per height) ===")
    for h in sorted(tr["height"].unique()):
        sub = tr[tr["height"] == h].reset_index(drop=True)
        A = sub[sub.index % 2 == 0]
        B = sub[sub.index % 2 == 1]

        def centers(df):
            c = []
            for _, r in df.iterrows():
                for (x0, y0, x1, y1) in parse_boxes(r["boxes"]):
                    c.append(((x0 + x1) / 2, (y0 + y1) / 2))
            return np.array(c)
        ca, cb = centers(A), centers(B)
        # for each center in B, distance to nearest center in A
        near = []
        for (x, y) in cb:
            d = np.sqrt((ca[:, 0] - x) ** 2 + (ca[:, 1] - y) ** 2).min()
            near.append(d)
        near = np.array(near)
        print(f" H={h}: B-centers with a train-A center within "
              f"3px={np.mean(near<=3):.1%}, 6px={np.mean(near<=6):.1%}, 12px={np.mean(near<=12):.1%} "
              f"(n_B={len(cb)}, n_A={len(ca)})")

    # heatmaps of centers per height
    for h in sorted(tr["height"].unique()):
        sub = tr[tr["height"] == h]
        hm = np.zeros((h, 448), np.float32)
        for _, r in sub.iterrows():
            for (x0, y0, x1, y1) in parse_boxes(r["boxes"]):
                cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
                if 0 <= cy < h and 0 <= cx < 448:
                    hm[cy, cx] += 1
        hm = cv2.GaussianBlur(hm, (0, 0), 4)
        hm = (hm / (hm.max() + 1e-9) * 255).astype(np.uint8)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(OUT, f"heatmap_H{h}.png"), hm)
        print(f" wrote heatmap_H{h}.png")

    # unique center modes per height at 6px quantization -> how many "hotspots"
    print("\n=== hotspot count (6px cells with >=3 boxes) per height ===")
    for h in sorted(tr["height"].unique()):
        sub = tr[tr["height"] == h]
        cells = {}
        n = 0
        for _, r in sub.iterrows():
            for (x0, y0, x1, y1) in parse_boxes(r["boxes"]):
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                k = (round(cx / 6), round(cy / 6))
                cells[k] = cells.get(k, 0) + 1
                n += 1
        hot = sum(1 for v in cells.values() if v >= 3)
        cov = sum(v for v in cells.values() if v >= 3)
        print(f" H={h}: {len(cells)} cells, {hot} hotspots(>=3), covering {cov}/{n} boxes ({cov/n:.0%})")


if __name__ == "__main__":
    main()

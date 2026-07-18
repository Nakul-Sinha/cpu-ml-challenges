"""Exploratory data analysis for LPBF Visual Alert Box Localization.

Goal: characterize the target boxes (count, size, aspect, position) and the
negative-image fraction so proposal generation and the ranker can be designed
against real structure rather than guesses.
"""
import os
import sys
import numpy as np
import pandas as pd

PUB = sys.argv[1] if len(sys.argv) > 1 else "dataset/public"


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
    print("=== SHAPES ===")
    print("train rows:", len(tr), "test rows:", len(te))
    print("train cols:", list(tr.columns))
    print("test  cols:", list(te.columns))

    print("\n=== IMAGE SIZES ===")
    print("train (w,h) value counts:")
    print(tr.groupby(["width", "height"]).size())
    print("test (w,h) value counts:")
    print(te.groupby(["width", "height"]).size())

    # alert_count distribution
    print("\n=== ALERT COUNT (train) ===")
    print(tr["alert_count"].value_counts().sort_index())
    n_neg = int((tr["alert_count"] == 0).sum())
    print("negative images (alert_count==0):", n_neg, f"({n_neg/len(tr):.1%})")

    # parse all boxes
    all_boxes = []
    ws, hs, aspects, areas, cxs, cys = [], [], [], [], [], []
    per_img_counts = []
    box_count_matches = 0
    for _, r in tr.iterrows():
        bxs = parse_boxes(r["boxes"])
        per_img_counts.append(len(bxs))
        if len(bxs) == int(r["alert_count"]):
            box_count_matches += 1
        for (x0, y0, x1, y1) in bxs:
            w = x1 - x0
            h = y1 - y0
            all_boxes.append((x0, y0, x1, y1))
            ws.append(w)
            hs.append(h)
            aspects.append(w / max(1, h))
            areas.append(w * h)
            cxs.append((x0 + x1) / 2.0)
            cys.append((y0 + y1) / 2.0)

    ws = np.array(ws); hs = np.array(hs); aspects = np.array(aspects)
    areas = np.array(areas); cxs = np.array(cxs); cys = np.array(cys)
    print("\n=== BOX GEOMETRY (n=%d boxes) ===" % len(ws))
    print("parsed box count == alert_count for all rows:", box_count_matches == len(tr))

    def stats(name, a):
        qs = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(f"{name:8s} min={qs[0]:.1f} p1={qs[1]:.1f} p5={qs[2]:.1f} "
              f"p25={qs[3]:.1f} med={qs[4]:.1f} p75={qs[5]:.1f} "
              f"p95={qs[6]:.1f} p99={qs[7]:.1f} max={qs[8]:.1f} mean={a.mean():.2f}")

    stats("width", ws)
    stats("height", hs)
    stats("aspect", aspects)
    stats("area", areas)
    stats("side", np.sqrt(areas))

    # squareness
    sq = np.abs(ws - hs)
    print("\n|w-h| distribution: ==0:", int((sq == 0).sum()),
          " <=1:", int((sq <= 1).sum()), " <=2:", int((sq <= 2).sum()),
          " of", len(ws))
    print("aspect in [0.9,1.1]:", int(((aspects >= 0.9) & (aspects <= 1.1)).sum()), "of", len(ws))

    # side length histogram (integer)
    print("\n=== SIDE LENGTH (width) histogram ===")
    vals, counts = np.unique(ws, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  w={v:3d}: {c:4d}  {'#'*min(60, c)}")

    # position: are centers on a grid / recurring?
    print("\n=== CENTER POSITIONS ===")
    stats("cx", cxs)
    stats("cy", cys)
    # quantize centers to 8px cells and see how concentrated
    cell = 8
    keys = (np.round(cxs / cell).astype(int) * 1000 + np.round(cys / cell).astype(int))
    uk, uc = np.unique(keys, return_counts=True)
    order = np.argsort(-uc)
    print(f"unique 8px center cells: {len(uk)} for {len(cxs)} boxes")
    print("top 15 recurring center cells (cx,cy)~ count:")
    for i in order[:15]:
        k = uk[i]
        cyq = k % 1000
        cxq = k // 1000
        print(f"  (~{cxq*cell},{cyq*cell})  count={uc[i]}")

    # coordinate ranges
    xs0 = np.array([b[0] for b in all_boxes]); ys0 = np.array([b[1] for b in all_boxes])
    xs1 = np.array([b[2] for b in all_boxes]); ys1 = np.array([b[3] for b in all_boxes])
    print("\nxmin range", xs0.min(), xs0.max(), " xmax range", xs1.min(), xs1.max())
    print("ymin range", ys0.min(), ys0.max(), " ymax range", ys1.min(), ys1.max())

    # do boxes ever exceed image bounds?
    print("\nany xmax>width?", "n/a (need per-row)")


if __name__ == "__main__":
    main()

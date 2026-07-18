"""Geometry-only prior ceiling. For each val box, find the nearest training box
center (same family) and measure how well center+size transfer -> IoU ceiling
achievable from the spatial prior alone (no images loaded)."""
import os
import sys
import numpy as np
import pandas as pd
import common as C

PUB = C.find_public_dir(sys.argv[1] if len(sys.argv) > 1 else None)


def boxes_of(df):
    out = {"gray": [], "color": []}
    for _, r in df.iterrows():
        fam = C.family(r["height"])
        for (x0, y0, x1, y1) in C.parse_boxes(r["boxes"]):
            out[fam].append([(x0 + x1) / 2.0, (y0 + y1) / 2.0, float(max(x1 - x0, y1 - y0))])
    return {k: np.array(v) for k, v in out.items()}


def sq_iou(c1, s1, c2, s2):
    # IoU of two axis-aligned squares given centers and sides
    x0a, y0a = c1[0] - s1 / 2, c1[1] - s1 / 2
    x0b, y0b = c2[0] - s2 / 2, c2[1] - s2 / 2
    return C.iou([x0a, y0a, x0a + s1, y0a + s1], [x0b, y0b, x0b + s2, y0b + s2])


def main():
    df = pd.read_csv(os.path.join(PUB, "train.csv"))
    tr, va = C.make_split(df)
    TB = boxes_of(tr)
    VB = boxes_of(va)
    for fam in ["gray", "color"]:
        T = TB[fam]; V = VB[fam]
        if len(V) == 0:
            continue
        dists, iou_pred, iou_center_only, iou_size_only = [], [], [], []
        for v in V:
            cv = v[:2]; sv = v[2]
            d = np.sqrt(((T[:, :2] - cv) ** 2).sum(1))
            j = int(np.argmin(d))
            ct = T[j, :2]; st = T[j, 2]
            dists.append(d[j])
            iou_pred.append(sq_iou(ct, st, cv, sv))            # anchor center+size
            iou_center_only.append(sq_iou(ct, sv, cv, sv))     # anchor center, true size
            iou_size_only.append(sq_iou(cv, st, cv, sv))       # true center, anchor size
        dists = np.array(dists)
        print(f"\n=== {fam} (n_val_boxes={len(V)}, n_train_boxes={len(T)}) ===")
        print(f" nearest-train-center dist: p50={np.percentile(dists,50):.1f} "
              f"p90={np.percentile(dists,90):.1f} max={dists.max():.1f}")
        for name, arr in [("anchor center+size", iou_pred),
                          ("anchor center, true size", iou_center_only),
                          ("true center, anchor size", iou_size_only)]:
            arr = np.array(arr)
            print(f"  IoU[{name}]: mean={arr.mean():.3f} "
                  f">=.5={np.mean(arr>=.5):.0%} >=.75={np.mean(arr>=.75):.0%} "
                  f">=.85={np.mean(arr>=.85):.0%}")


if __name__ == "__main__":
    main()

"""Overlay the submission's predicted boxes on test images to inspect whether we
are predicting boxes on blank (likely-negative) images. Renders the images with
the FEWEST-confidence top box (candidate negatives) and a normal sample."""
import os
import sys
import numpy as np
import pandas as pd
import cv2

PUB = sys.argv[1] if len(sys.argv) > 1 else "dataset/public"
SUB = sys.argv[2] if len(sys.argv) > 2 else "working/submission.csv"
OUT = "working"


def parse_pred(s):
    if not isinstance(s, str) or not s.strip():
        return []
    t = s.split()
    return [(float(t[i]), int(float(t[i+1])), int(float(t[i+2])), int(float(t[i+3])), int(float(t[i+4])))
            for i in range(0, len(t), 5)]


def main():
    te = pd.read_csv(os.path.join(PUB, "test.csv"))
    sub = pd.read_csv(SUB).fillna("")
    m = dict(zip(sub["image_id"], sub["prediction_string"]))
    info = []
    for _, r in te.iterrows():
        preds = parse_pred(m.get(r["image_id"], ""))
        top = max([p[0] for p in preds], default=0.0)
        info.append((r["image_id"], r["image_path"], len(preds), top, preds))
    counts = np.array([x[2] for x in info]); tops = np.array([x[3] for x in info])
    print("boxes/img: mean=%.2f min=%d max=%d" % (counts.mean(), counts.min(), counts.max()))
    print("per-image TOP score: min=%.3f p10=%.3f p25=%.3f med=%.3f" %
          (tops.min(), np.percentile(tops, 10), np.percentile(tops, 25), np.median(tops)))
    print("images with top<0.3: %d, top<0.4: %d, top<0.5: %d of %d" %
          ((tops < 0.3).sum(), (tops < 0.4).sum(), (tops < 0.5).sum(), len(info)))

    order = sorted(info, key=lambda x: x[3])  # ascending top score
    picks = order[:8] + order[len(order)//2:len(order)//2+4]  # weakest 8 + median 4
    tiles = []
    for (iid, path, n, top, preds) in picks:
        img = cv2.imread(os.path.join(PUB, path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        vis = img.copy()
        for (sc, x0, y0, x1, y1) in preds:
            col = (0, 0, 255) if sc >= 0.3 else (0, 200, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 1)
        cv2.putText(vis, "top%.2f n%d" % (top, n), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        canvas = np.zeros((448, 448, 3), np.uint8); canvas[:vis.shape[0], :vis.shape[1]] = vis[:448, :448]
        tiles.append(canvas)
    rows = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(os.path.join(OUT, "test_preds.png"), np.vstack(rows))
    print("wrote test_preds.png (weakest 8 + median 4)")


if __name__ == "__main__":
    main()

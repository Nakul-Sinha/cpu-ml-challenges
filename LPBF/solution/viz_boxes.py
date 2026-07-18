"""Render individual GT boxes at high zoom with the exact GT square drawn, to
understand what determines the (odd, 19-35) square size. Separate montages per
family, sampling across the size range."""
import os
import sys
import numpy as np
import pandas as pd
import cv2
import common as C

PUB = C.find_public_dir(sys.argv[1] if len(sys.argv) > 1 else None)
OUT = "working"


def collect(df, fam, per_size=3):
    rows = []
    for _, r in df.iterrows():
        if C.family(r["height"]) != fam:
            continue
        for b in C.parse_boxes(r["boxes"]):
            rows.append((r["image_path"], b, max(b[2] - b[0], b[3] - b[1])))
    # sample across sizes
    bysize = {}
    for item in rows:
        bysize.setdefault(item[2], []).append(item)
    picks = []
    for s in sorted(bysize):
        picks += bysize[s][:per_size]
    return picks


def render(df, fam):
    picks = collect(df, fam)
    tiles = []
    zoom = 6
    ctx = 24  # px of context each side (pre-zoom)
    for (path, b, s) in picks:
        bgr = C.load_bgr(PUB, path)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        x0, y0, x1, y1 = b
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        a = max(0, cx - ctx); bb = max(0, cy - ctx)
        a2 = min(W, cx + ctx); b2 = min(H, cy + ctx)
        crop = bgr[bb:b2, a:a2].copy()
        crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        # draw GT box in crop coords
        gx0 = (x0 - a) * zoom; gy0 = (y0 - bb) * zoom
        gx1 = (x1 - a) * zoom; gy1 = (y1 - bb) * zoom
        cv2.rectangle(crop, (gx0, gy0), (gx1, gy1), (0, 0, 255), 1)
        cv2.putText(crop, str(s), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # pad to fixed tile
        th = (2 * ctx) * zoom
        canvas = np.zeros((th, th, 3), np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop[:th, :th]
        tiles.append(canvas)
    ncol = 8
    while len(tiles) % ncol:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack([np.pad(t, ((1, 1), (1, 1), (0, 0)), constant_values=255)
                       for t in tiles[i:i + ncol]]) for i in range(0, len(tiles), ncol)]
    grid = np.vstack(rows)
    p = os.path.join(OUT, f"boxes_{fam}.png")
    cv2.imwrite(p, grid)
    print("wrote", p, grid.shape, "n=", len(picks))


def main():
    df = pd.read_csv(os.path.join(PUB, "train.csv"))
    render(df, "gray")
    render(df, "color")


if __name__ == "__main__":
    main()

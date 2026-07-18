"""Diagnostic: candidate/refinement recall ceiling.

For the val split, generate candidates (anchors UNION saliency peaks), refine
each to a box, and measure what fraction of GT boxes have a candidate box with
IoU >= {0.5,0.75,0.85}. This is the ceiling the ranker can reach.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import common as C

PUB = C.find_public_dir(sys.argv[1] if len(sys.argv) > 1 else None)


def main():
    df = pd.read_csv(os.path.join(PUB, "train.csv"))
    tr, va = C.make_split(df)
    anchors = {f: C.build_anchors(tr, f) for f in ["gray", "color"]}
    print("anchors: gray=%d color=%d" % (len(anchors["gray"]), len(anchors["color"])))

    thr = [0.5, 0.75, 0.85]
    hit = {t: 0 for t in thr}
    ncand_tot = 0
    n_gt = 0
    n_img = 0
    t0 = time.time()
    for _, r in va.iterrows():
        fam = C.family(r["height"])
        bgr = C.load_bgr(PUB, r["image_path"])
        if bgr is None:
            continue
        gray = C.to_gray(bgr)
        H, W = gray.shape
        maps = C.cue_maps(gray)
        integ = C.Integrals(maps["sal"])
        cands = C.gen_candidates(maps["sal"], anchors[fam], H, W)
        boxes = []
        for (cx, cy) in cands:
            x0, y0, x1, y1, resp, s, ncx, ncy = C.refine_box(integ, cx, cy)
            boxes.append(C.clip_box([x0, y0, x1, y1], W, H))
        ncand_tot += len(boxes)
        n_img += 1
        gts = C.parse_boxes(r["boxes"])
        for g in gts:
            n_gt += 1
            best = max((C.iou(b, g) for b in boxes), default=0.0)
            for t in thr:
                if best >= t:
                    hit[t] += 1
    dt = time.time() - t0
    print("val images=%d  gt boxes=%d  avg cand/img=%.1f  time=%.1fs (%.2fs/img)"
          % (n_img, n_gt, ncand_tot / max(1, n_img), dt, dt / max(1, n_img)))
    for t in thr:
        print("  candidate recall@%.2f = %.1f%%" % (t, 100.0 * hit[t] / max(1, n_gt)))


if __name__ == "__main__":
    main()

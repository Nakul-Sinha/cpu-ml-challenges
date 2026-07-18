"""Isolate the size-estimation problem. Using the TRUE center of each val box,
search odd sizes with several visual objectives and report IoU@{.75,.85} of the
resulting (true-center, predicted-size) square. Whichever objective recovers
size best is the core of the localizer."""
import os
import sys
import numpy as np
import pandas as pd
import cv2
import common as C

PUB = C.find_public_dir(sys.argv[1] if len(sys.argv) > 1 else None)
SIZES = list(range(15, 42, 2))


def energy_maps(gray):
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    blur = cv2.GaussianBlur(g, (0, 0), 2)
    hf = np.abs(g - blur)
    k = 5
    mean = cv2.boxFilter(g, cv2.CV_32F, (k, k))
    mean2 = cv2.boxFilter(g * g, cv2.CV_32F, (k, k))
    lstd = np.sqrt(np.clip(mean2 - mean * mean, 0, None))
    return dict(grad=grad, hf=hf, lstd=lstd, energy=grad + 2 * hf)


def obj_cs(integ, cx, cy, s, r=4):
    h = s / 2.0
    m_in = integ.mean(cx - h, cy - h, cx + h, cy + h)
    big = integ.mean(cx - h - r, cy - h - r, cx + h + r, cy + h + r)
    Ab = (s + 2 * r) ** 2; Ai = s * s
    ring = (big * Ab - m_in * Ai) / max(1.0, Ab - Ai)
    return m_in - ring


def obj_encl(integ, cx, cy, s, r=4):
    h = s / 2.0
    m_in = integ.mean(cx - h, cy - h, cx + h, cy + h)
    m_big = integ.mean(cx - h - r, cy - h - r, cx + h + r, cy + h + r)
    return m_in - m_big


def best_size(integ, cx, cy, fn):
    best = None
    for s in SIZES:
        v = fn(integ, cx, cy, s)
        if best is None or v > best[1]:
            best = (s, v)
    return best[0]


def main():
    df = pd.read_csv(os.path.join(PUB, "train.csv"))
    tr, va = C.make_split(df)
    objs = ["grad_cs", "hf_cs", "lstd_cs", "energy_cs", "energy_encl", "grad_encl"]
    res = {fam: {o: [] for o in objs} for fam in ["gray", "color"]}
    truth_side = {fam: [] for fam in ["gray", "color"]}
    for _, r in va.iterrows():
        fam = C.family(r["height"])
        bgr = C.load_bgr(PUB, r["image_path"])
        if bgr is None:
            continue
        gray = C.to_gray(bgr)
        H, W = gray.shape
        em = energy_maps(gray)
        integ = {k: C.Integrals(v) for k, v in em.items()}
        for (x0, y0, x1, y1) in C.parse_boxes(r["boxes"]):
            cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
            true_s = max(x1 - x0, y1 - y0)
            truth_side[fam].append(true_s)
            preds = {
                "grad_cs": best_size(integ["grad"], cx, cy, obj_cs),
                "hf_cs": best_size(integ["hf"], cx, cy, obj_cs),
                "lstd_cs": best_size(integ["lstd"], cx, cy, obj_cs),
                "energy_cs": best_size(integ["energy"], cx, cy, obj_cs),
                "energy_encl": best_size(integ["energy"], cx, cy, obj_encl),
                "grad_encl": best_size(integ["grad"], cx, cy, obj_encl),
            }
            for o, ps in preds.items():
                # IoU of concentric squares side ps vs true_s
                inter = min(ps, true_s) ** 2
                union = ps * ps + true_s * true_s - inter
                res[fam][o].append(inter / union)
    for fam in ["gray", "color"]:
        ts = np.array(truth_side[fam])
        print(f"\n=== {fam} (n={len(ts)}) true side mean={ts.mean():.1f} ===")
        for o in objs:
            a = np.array(res[fam][o])
            if len(a) == 0:
                continue
            print(f"  {o:12s} IoU mean={a.mean():.3f} >=.75={np.mean(a>=.75):.0%} "
                  f">=.85={np.mean(a>=.85):.0%}")


if __name__ == "__main__":
    main()

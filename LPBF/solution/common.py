"""Shared utilities for LPBF alert-box localization:
- data loading + box parsing
- image family handling (gray 358 vs color 448)
- multi-cue saliency maps + integral-image box features
- spatial-prior anchors, candidate generation, box refinement, NMS

Kept dependency-light (numpy, cv2, pandas) so it runs on the constrained CPU
runtime.
"""
import os
import numpy as np
import pandas as pd
import cv2

ODD_SIZES = list(range(19, 36, 2))  # 19,21,...,35 -> ~99% of boxes


def find_public_dir(explicit=None):
    if explicit and os.path.isdir(explicit):
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        "dataset/public", "public", "dataset",
        os.path.join(here, "..", "dataset", "public"),
        os.path.join(here, "dataset", "public"),
    ]
    for c in cands:
        if os.path.isfile(os.path.join(c, "test.csv")):
            return c
    return "dataset/public"


def make_split(df, val_frac=0.2, seed=42):
    """Deterministic train/val split stratified by image family."""
    rng = np.random.RandomState(seed)
    val_idx = []
    for fam in ["gray", "color"]:
        ids = [i for i in df.index if family(df.loc[i, "height"]) == fam]
        ids = list(ids)
        rng.shuffle(ids)
        k = int(round(len(ids) * val_frac))
        val_idx += ids[:k]
    val_mask = df.index.isin(val_idx)
    return df[~val_mask].copy(), df[val_mask].copy()


def parse_boxes(s):
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for tok in s.split():
        p = tok.split(",")
        if len(p) == 4:
            out.append([int(round(float(x))) for x in p])
    return out


def family(h):
    """Two image families keyed by height."""
    return "gray" if int(h) == 358 else "color"


def load_bgr(pub, path):
    return cv2.imread(os.path.join(pub, path), cv2.IMREAD_COLOR)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


# ---------------- integral images ----------------
class Integrals:
    """Integral images for O(1) rectangle mean / std over a gray map."""
    def __init__(self, g):
        g = g.astype(np.float64)
        self.H, self.W = g.shape
        self.S = cv2.integral(g)          # (H+1,W+1)
        self.S2 = cv2.integral(g * g)

    def _rectsum(self, S, x0, y0, x1, y1):
        x0 = max(0, min(self.W, x0)); x1 = max(0, min(self.W, x1))
        y0 = max(0, min(self.H, y0)); y1 = max(0, min(self.H, y1))
        return S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]

    def mean_std(self, x0, y0, x1, y1):
        x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
        n = max(1, (x1i - x0i) * (y1i - y0i))
        s = self._rectsum(self.S, x0i, y0i, x1i, y1i)
        s2 = self._rectsum(self.S2, x0i, y0i, x1i, y1i)
        m = s / n
        v = max(0.0, s2 / n - m * m)
        return m, np.sqrt(v)

    def mean(self, x0, y0, x1, y1):
        x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
        n = max(1, (x1i - x0i) * (y1i - y0i))
        return self._rectsum(self.S, x0i, y0i, x1i, y1i) / n


# ---------------- cue maps ----------------
def robust_norm(m):
    md = np.median(m)
    mad = np.median(np.abs(m - md)) + 1e-6
    z = (m - md) / (1.4826 * mad)
    return np.clip(z, 0, None)


def cue_maps(gray):
    """Return dict of cue maps and a combined saliency map. All float32, same
    HxW as gray. Cues: local std (texture/contrast), gradient magnitude (edges),
    top-hat (bright compact), black-hat (dark compact), high-frequency."""
    g = gray.astype(np.float32)
    H, W = g.shape
    # gradient magnitude
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    # local std over 7x7 (texture / contrast)
    k = 7
    mean = cv2.boxFilter(g, cv2.CV_32F, (k, k))
    mean2 = cv2.boxFilter(g * g, cv2.CV_32F, (k, k))
    var = np.clip(mean2 - mean * mean, 0, None)
    lstd = np.sqrt(var)
    # morphological top-hat / black-hat (compact bright / dark structures)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tophat = cv2.morphologyEx(g8, cv2.MORPH_TOPHAT, se).astype(np.float32)
    blackhat = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    # high frequency residual
    blur = cv2.GaussianBlur(g, (0, 0), 3)
    hf = np.abs(g - blur)

    maps = dict(grad=grad, lstd=lstd, tophat=tophat, blackhat=blackhat, hf=hf)
    # combined saliency: emphasise edge density + texture + compact residual
    sal = (robust_norm(grad) + robust_norm(lstd) + robust_norm(hf)
           + 0.5 * robust_norm(tophat) + 0.5 * robust_norm(blackhat))
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2)
    maps["sal"] = sal
    return maps


# ---------------- spatial prior anchors ----------------
def build_anchors(train_df, fam, cell=2, min_count=2, merge=3):
    """Cluster training box centers (for a family) into anchor locations.
    Returns list of dicts {cx,cy,count,med_size,sizes}. Honest spatial prior:
    only provides candidate locations; per-image presence decided by the visual
    ranker. Kept fine (small cell/merge) since centers recur near-exactly."""
    pts = []
    sizes = []
    for _, r in train_df.iterrows():
        if family(r["height"]) != fam:
            continue
        for (x0, y0, x1, y1) in parse_boxes(r.get("boxes", "")):
            pts.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
            sizes.append(max(x1 - x0, y1 - y0))
    if not pts:
        return []
    pts = np.array(pts); sizes = np.array(sizes)
    keys = {}
    for i, (x, y) in enumerate(pts):
        k = (int(round(x / cell)), int(round(y / cell)))
        keys.setdefault(k, []).append(i)
    cellinfo = []
    for k, idxs in keys.items():
        cx = pts[idxs, 0].mean(); cy = pts[idxs, 1].mean()
        cellinfo.append([len(idxs), cx, cy, list(sizes[idxs])])
    cellinfo.sort(key=lambda z: -z[0])
    anchors = []
    for cnt, cx, cy, szs in cellinfo:
        if cnt < min_count:
            continue
        merged = False
        for a in anchors:
            if abs(a["cx"] - cx) <= merge and abs(a["cy"] - cy) <= merge:
                tot = a["count"] + cnt
                a["cx"] = (a["cx"] * a["count"] + cx * cnt) / tot
                a["cy"] = (a["cy"] * a["count"] + cy * cnt) / tot
                a["count"] = tot
                a["sizes"].extend(szs)
                a["med_size"] = float(np.median(a["sizes"]))
                merged = True
                break
        if not merged:
            anchors.append(dict(cx=cx, cy=cy, count=int(cnt),
                                med_size=float(np.median(szs)), sizes=list(szs)))
    return anchors


def prior_heatmap(train_df, fam, H, W, sigma=3.0):
    hm = np.zeros((H, W), np.float32)
    for _, r in train_df.iterrows():
        if family(r["height"]) != fam:
            continue
        for (x0, y0, x1, y1) in parse_boxes(r.get("boxes", "")):
            cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
            if 0 <= cy < H and 0 <= cx < W:
                hm[cy, cx] += 1
    n_imgs = max(1, int((train_df["height"].map(family) == fam).sum()))
    hm = cv2.GaussianBlur(hm, (0, 0), sigma) / n_imgs
    return hm


# ---------------- candidate generation ----------------
def peak_candidates(sal, max_peaks=60, min_dist=10, rel_thresh=0.35):
    """Local maxima of the saliency map -> candidate centers."""
    H, W = sal.shape
    d = int(min_dist)
    dil = cv2.dilate(sal, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1)))
    peaks = (sal >= dil - 1e-6) & (sal > sal.max() * rel_thresh)
    ys, xs = np.where(peaks)
    vals = sal[ys, xs]
    order = np.argsort(-vals)[:max_peaks]
    return [(int(xs[i]), int(ys[i])) for i in order]


def gen_candidates(sal, anchors, H, W, snap=6):
    """Union of anchor locations (snapped to nearest local saliency peak) and
    pure saliency peaks. Returns list of (cx,cy)."""
    cands = []
    # anchors, snapped to local saliency max within +-snap
    for a in anchors:
        cx0, cy0 = int(a["cx"]), int(a["cy"])
        x0 = max(0, cx0 - snap); x1 = min(W, cx0 + snap + 1)
        y0 = max(0, cy0 - snap); y1 = min(H, cy0 + snap + 1)
        patch = sal[y0:y1, x0:x1]
        if patch.size:
            j = np.argmax(patch)
            py, px = np.unravel_index(j, patch.shape)
            cands.append((x0 + px, y0 + py))
        else:
            cands.append((cx0, cy0))
    # visual peaks
    cands += peak_candidates(sal, max_peaks=50, min_dist=9)
    # dedupe close points
    out = []
    for (x, y) in cands:
        if all(abs(x - ox) + abs(y - oy) > 5 for (ox, oy) in out):
            out.append((x, y))
    return out


# ---------------- box refinement ----------------
def center_surround(integ_sal, cx, cy, s):
    """Center-surround saliency response for a square of side s at (cx,cy):
    mean saliency inside minus mean saliency in a surrounding ring."""
    h = s / 2.0
    x0, y0, x1, y1 = cx - h, cy - h, cx + h, cy + h
    m_in = integ_sal.mean(x0, y0, x1, y1)
    o = s * 0.6
    m_out_big = integ_sal.mean(cx - h - o, cy - h - o, cx + h + o, cy + h + o)
    # ring mean = (big area mean*big - inner*inner)/(ring area)
    A_big = (2 * (h + o)) ** 2
    A_in = (2 * h) ** 2
    ring = (m_out_big * A_big - m_in * A_in) / max(1.0, (A_big - A_in))
    return m_in - ring


def refine_box(integ_sal, cx, cy, sizes=ODD_SIZES, off=3):
    """Search small center offsets and odd sizes to maximise center-surround
    response. Returns (x0,y0,x1,y1,resp,size,ncx,ncy)."""
    best = None
    for dy in range(-off, off + 1):
        for dx in range(-off, off + 1):
            ncx, ncy = cx + dx, cy + dy
            for s in sizes:
                r = center_surround(integ_sal, ncx, ncy, s)
                if best is None or r > best[0]:
                    best = (r, s, ncx, ncy)
    r, s, ncx, ncy = best
    h = s // 2
    x0 = int(round(ncx - h)); y0 = int(round(ncy - h))
    x1 = x0 + s; y1 = y0 + s
    return x0, y0, x1, y1, r, s, ncx, ncy


# ---------------- NMS ----------------
def nms(boxes, scores, iou_thr=0.30):
    if not boxes:
        return []
    idx = np.argsort(-np.asarray(scores))
    keep = []
    taken = []
    for i in idx:
        b = boxes[i]
        if all(iou(b, boxes[j]) < iou_thr for j in taken):
            keep.append(i)
            taken.append(i)
    return keep


def clip_box(b, W, H):
    x0, y0, x1, y1 = b
    x0 = int(max(0, min(W - 1, x0)))
    y0 = int(max(0, min(H - 1, y0)))
    x1 = int(max(x0 + 1, min(W, x1)))
    y1 = int(max(y0 + 1, min(H, y1)))
    return [x0, y0, x1, y1]

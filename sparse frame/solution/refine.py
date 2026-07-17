"""Local mask-based box refinement: given a coarse box (from the CNN center), find the
object's tight bbox via a local full-res DoG + morphological close + connected components,
using the coarse box as a center/size prior with sanity gating."""
import numpy as np
import cv2
from common import masks

def refine_box(red, blue, box, expand=2.6, dog_sigma=8.0, thr=0.15, close_k=7,
               max_shift_frac=0.8, area_lo=0.15, area_hi=6.0):
    """red,blue: full-res float masks (H,W). box=(x,y,w,h). Returns refined (x,y,w,h) or box."""
    H, W = red.shape
    x, y, w, h = box; cx, cy = x+w/2, y+h/2
    sw = min(max(w*expand, 26), 460); sh = min(max(h*expand, 22), 340)
    x0 = int(max(0, cx-sw/2)); x1 = int(min(W, cx+sw/2)); y0 = int(max(0, cy-sh/2)); y1 = int(min(H, cy+sh/2))
    if x1-x0 < 4 or y1-y0 < 4: return box
    comb = red[y0:y1, x0:x1]+blue[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(comb, (0, 0), dog_sigma)
    dog = np.clip(comb-blur, 0, None)
    m = (dog > thr).astype(np.uint8)
    if m.sum() < 6:
        m = (comb > 0).astype(np.uint8)
    if close_k > 1:  # merge fragmented object parts
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return box
    lcx, lcy = cx-x0, cy-y0
    refscale = 0.5*max(w, h)+10
    coarse_area = max(1.0, w*h)
    best = None; bs = -1
    for i in range(1, n):
        bx, by, bw, bh, ar = stats[i]
        if ar < 4: continue
        ccx, ccy = cent[i]; dist = np.hypot(ccx-lcx, ccy-lcy)
        score = ar / (1.0 + dist/refscale)
        if score > bs: bs = score; best = (float(x0+bx), float(y0+by), float(bw), float(bh))
    if best is None: return box
    # sanity gate: reject absurd jumps
    rx, ry, rw, rh = best; rcx, rcy = rx+rw/2, ry+rh/2
    if np.hypot(rcx-cx, rcy-cy) > max_shift_frac*max(w, h)+30: return box
    if not (area_lo*coarse_area <= rw*rh <= area_hi*coarse_area): return box
    return best

def refine_track(img_dir, clip, coarse_cen, coarse_siz, FRAME_W, FRAME_H, load_img, **kw):
    """Refine t0-3 boxes for a clip. coarse_cen/siz normalized (>=4,2). Returns refined cen,siz + masks list."""
    rc = np.array(coarse_cen[:4], float).copy(); rs = np.array(coarse_siz[:4], float).copy()
    mlist = []
    for t in range(4):
        img = load_img(img_dir, clip, t)
        red, blue = masks(img); mlist.append((red, blue))
        cb = (coarse_cen[t][0]*FRAME_W-coarse_siz[t][0]*FRAME_W/2, coarse_cen[t][1]*FRAME_H-coarse_siz[t][1]*FRAME_H/2,
              max(4, coarse_siz[t][0]*FRAME_W), max(4, coarse_siz[t][1]*FRAME_H))
        rb = refine_box(red, blue, cb, **kw)
        rc[t] = [(rb[0]+rb[2]/2)/FRAME_W, (rb[1]+rb[3]/2)/FRAME_H]; rs[t] = [rb[2]/FRAME_W, rb[3]/FRAME_H]
    return rc, rs, mlist

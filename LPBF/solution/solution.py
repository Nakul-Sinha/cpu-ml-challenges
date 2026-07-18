"""LPBF Visual Alert Box Localization - official, self-contained CPU solution.

Runs end-to-end: reads dataset/public/{train,test}.csv + images, trains a
spatial-prior + visual-cue presence ranker plus a box-offset regressor, localizes
square alert boxes, writes working/submission.csv. No GPU, no internet, no
precomputed predictions. CPU-only, deterministic.

Usage: python solution.py [dataset_public_dir] [out_csv]
Defaults: dataset/public  and  working/submission.csv

Approach (see approach.md). Target boxes are axis-aligned squares (odd side
~19-35 px) that recur at a small set of registered locations across two image
families (gray 448x358 powder bed, colour 448x448 grid of parts). We build
per-family spatial anchor locations from the training boxes, union them with
multi cue saliency peaks (contrast, edges, texture, colour, deviation from a
per-family median background), describe each candidate with center-surround
features plus the learned spatial prior, and train a gradient boosted classifier
to decide which candidates are real alert boxes. A gradient boosted box-offset
regressor then refines each box toward the true location and size. NMS removes
duplicates and a calibrated, negative-safe threshold keeps predictions sparse.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import cv2

FAMS = ["gray", "color"]
POS_R = {"gray": 4.0, "color": 7.0}       # px: candidate<->GT match radius (labels)
FAM_MED_SIZE = {"gray": 25, "color": 29}
CUES = ["gray", "grad", "lstd", "hf", "tophat", "blackhat"]
TH_MAIN = 0.05                             # emit candidates scoring >= this
FLOOR = 0.08                               # else force top-1 if its score >= this
MAX_BOX = 8
NMS_IOU = 0.30
SEED = 0
CLS_PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                  l2_regularization=1.0, min_samples_leaf=20)
REG_PARAMS = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=25)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ io / boxes
def find_public_dir(explicit=None):
    if explicit and os.path.isdir(explicit):
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for c in ["dataset/public", "public", "dataset",
              os.path.join(here, "..", "dataset", "public"),
              os.path.join(here, "dataset", "public")]:
        if os.path.isfile(os.path.join(c, "test.csv")):
            return c
    return "dataset/public"


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
    return "gray" if int(h) == 358 else "color"


def load_bgr(pub, path):
    return cv2.imread(os.path.join(pub, path), cv2.IMREAD_COLOR)


def iou(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def clip_box(b, W, H):
    x0 = int(max(0, min(W - 1, b[0]))); y0 = int(max(0, min(H - 1, b[1])))
    x1 = int(max(x0 + 1, min(W, b[2]))); y1 = int(max(y0 + 1, min(H, b[3])))
    return [x0, y0, x1, y1]


def box_from(cx, cy, s):
    h = s // 2
    x0 = int(round(cx - h)); y0 = int(round(cy - h))
    return [x0, y0, x0 + s, y0 + s]


def odd(s):
    s = int(round(s))
    return max(15, min(41, s + (s % 2 == 0)))


def nms(boxes, scores, iou_thr=0.30):
    if not boxes:
        return []
    order = np.argsort(-np.asarray(scores))
    keep, taken = [], []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < iou_thr for j in taken):
            keep.append(int(i)); taken.append(int(i))
    return keep


# ------------------------------------------------------------------ integral imgs
class Integrals:
    def __init__(self, g):
        g = g.astype(np.float64)
        self.H, self.W = g.shape
        self.S = cv2.integral(g)
        self.S2 = cv2.integral(g * g)

    def _rs(self, S, x0, y0, x1, y1):
        x0 = max(0, min(self.W, x0)); x1 = max(0, min(self.W, x1))
        y0 = max(0, min(self.H, y0)); y1 = max(0, min(self.H, y1))
        return S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]

    def mean(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        n = max(1, (x1 - x0) * (y1 - y0))
        return self._rs(self.S, x0, y0, x1, y1) / n

    def mean_std(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        n = max(1, (x1 - x0) * (y1 - y0))
        mm = self._rs(self.S, x0, y0, x1, y1) / n
        v = max(0.0, self._rs(self.S2, x0, y0, x1, y1) / n - mm * mm)
        return mm, np.sqrt(v)


def robust_norm(m):
    md = np.median(m)
    mad = np.median(np.abs(m - md)) + 1e-6
    return np.clip((m - md) / (1.4826 * mad), 0, None)


def cue_maps(gray):
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    k = 7
    mean = cv2.boxFilter(g, cv2.CV_32F, (k, k))
    mean2 = cv2.boxFilter(g * g, cv2.CV_32F, (k, k))
    lstd = np.sqrt(np.clip(mean2 - mean * mean, 0, None))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tophat = cv2.morphologyEx(g8, cv2.MORPH_TOPHAT, se).astype(np.float32)
    blackhat = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    hf = np.abs(g - cv2.GaussianBlur(g, (0, 0), 3))
    sal = (robust_norm(grad) + robust_norm(lstd) + robust_norm(hf)
           + 0.5 * robust_norm(tophat) + 0.5 * robust_norm(blackhat))
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2)
    return dict(grad=grad, lstd=lstd, tophat=tophat, blackhat=blackhat, hf=hf, sal=sal)


# ------------------------------------------------------------------ prior / bg
def build_anchors(df, fam, cell=2, min_count=2, merge=3):
    pts, sizes = [], []
    for _, r in df.iterrows():
        if family(r["height"]) != fam:
            continue
        for (x0, y0, x1, y1) in parse_boxes(r.get("boxes", "")):
            pts.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0)); sizes.append(max(x1 - x0, y1 - y0))
    if not pts:
        return []
    pts = np.array(pts); sizes = np.array(sizes)
    keys = {}
    for i, (x, y) in enumerate(pts):
        keys.setdefault((int(round(x / cell)), int(round(y / cell))), []).append(i)
    cellinfo = [[len(ix), pts[ix, 0].mean(), pts[ix, 1].mean(), list(sizes[ix])] for ix in keys.values()]
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
                a["count"] = tot; a["sizes"].extend(szs)
                a["med_size"] = float(np.median(a["sizes"])); merged = True; break
        if not merged:
            anchors.append(dict(cx=cx, cy=cy, count=int(cnt),
                                med_size=float(np.median(szs)), sizes=list(szs)))
    return anchors


def prior_heatmap(df, fam, H, W, sigma=3.0):
    hm = np.zeros((H, W), np.float32); n = 0
    for _, r in df.iterrows():
        if family(r["height"]) != fam:
            continue
        n += 1
        for (x0, y0, x1, y1) in parse_boxes(r.get("boxes", "")):
            cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
            if 0 <= cy < H and 0 <= cx < W:
                hm[cy, cx] += 1
    return cv2.GaussianBlur(hm, (0, 0), sigma) / max(1, n)


def build_background(df, fam, pub):
    imgs = []
    for _, r in df.iterrows():
        if family(r["height"]) != fam:
            continue
        bgr = cv2.imread(os.path.join(pub, r["image_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if not imgs:
        return None
    return np.median(np.stack(imgs), axis=0).astype(np.float32)


def peak_candidates(sal, max_peaks, min_dist, rel_thresh):
    d = int(min_dist)
    dil = cv2.dilate(sal, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1)))
    peaks = (sal >= dil - 1e-6) & (sal > sal.max() * rel_thresh)
    ys, xs = np.where(peaks)
    order = np.argsort(-sal[ys, xs])[:max_peaks]
    return [(int(xs[i]), int(ys[i])) for i in order]


# ------------------------------------------------------------------ features
class FeatureExtractor:
    def __init__(self, bgr, fam, prior_hm, bg):
        self.fam = fam
        self.H, self.W = bgr.shape[:2]
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.gray = g
        m = cue_maps(g)
        self.sal = m["sal"]
        ch = {"gray": g, "grad": m["grad"], "lstd": m["lstd"], "hf": m["hf"],
              "tophat": m["tophat"], "blackhat": m["blackhat"],
              "B": bgr[..., 0].astype(np.float32), "G": bgr[..., 1].astype(np.float32),
              "R": bgr[..., 2].astype(np.float32)}
        if fam == "color":
            rb = np.abs(ch["R"] - ch["B"]); k = 7
            m1 = cv2.boxFilter(rb, cv2.CV_32F, (k, k)); m2 = cv2.boxFilter(rb * rb, cv2.CV_32F, (k, k))
            rb_tex = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
            col = robust_norm(cv2.GaussianBlur(rb, (0, 0), 2)) + robust_norm(rb_tex)
            self.sal = self.sal + 0.8 * cv2.GaussianBlur(col.astype(np.float32), (0, 0), 2)
        self.has_bg = bg is not None and bg.shape == g.shape
        if self.has_bg:
            resid = (g - float(g.mean())) - (bg - float(bg.mean()))
            ch["resid"] = np.abs(resid); ch["sresid"] = resid
            self.sal = self.sal + 1.0 * robust_norm(
                cv2.GaussianBlur(np.abs(resid).astype(np.float32), (0, 0), 2))
        self.integ = {k: Integrals(v) for k, v in ch.items()}
        self.integ["sal"] = Integrals(self.sal)
        self.g_mean = float(g.mean()); self.g_std = float(g.std() + 1e-6)
        self.prior_hm = prior_hm

    def _ring(self, integ, cx, cy, s, m):
        h = s / 2.0
        A_in = s * s; A_out = (s + 2 * m) ** 2
        m_in = integ.mean(cx - h, cy - h, cx + h, cy + h)
        big = integ.mean(cx - h - m, cy - h - m, cx + h + m, cy + h + m)
        ring = (big * A_out - m_in * A_in) / max(1.0, A_out - A_in)
        return m_in, ring

    def features(self, cx, cy, s, acount, adist):
        f = [cx / self.W, cy / self.H, s / 30.0, np.log1p(acount), min(adist, 30.0) / 30.0]
        xi = int(np.clip(cx, 0, self.W - 1)); yi = int(np.clip(cy, 0, self.H - 1))
        f.append(float(self.prior_hm[yi, xi]))
        m = max(4, int(round(s * 0.5)))
        for key in CUES:
            mi, rg = self._ring(self.integ[key], cx, cy, s, m)
            f.append(mi / 255.0); f.append((mi - rg) / 255.0)
        _, gstd = self.integ["gray"].mean_std(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
        f.append(gstd / 64.0)
        mi, rg = self._ring(self.integ["sal"], cx, cy, s, m)
        f.append(mi); f.append(mi - rg)
        if self.has_bg:
            ri, rr = self._ring(self.integ["resid"], cx, cy, s, m)
            f.append(ri / 64.0); f.append((ri - rr) / 64.0)
            si = self.integ["sresid"].mean(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
            f.append(si / 64.0)
            _, rstd = self.integ["resid"].mean_std(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
            f.append(rstd / 64.0)
        else:
            f.extend([0.0, 0.0, 0.0, 0.0])
        for key in ["R", "B"]:
            mi, rg = self._ring(self.integ[key], cx, cy, s, m)
            f.append((mi - rg) / 255.0)
        rin, rr = self._ring(self.integ["R"], cx, cy, s, m)
        bi, br = self._ring(self.integ["B"], cx, cy, s, m)
        f.append(((rin - bi) - (rr - br)) / 255.0)
        for sc in (0.6, 1.4):
            ss = max(9, s * sc)
            mi, rg = self._ring(self.integ["sal"], cx, cy, ss, max(4, int(ss * 0.5)))
            f.append(mi - rg)
        gm = self.integ["gray"].mean(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
        f.append((gm - self.g_mean) / self.g_std)
        f.append(1.0 if self.fam == "color" else 0.0)
        return np.asarray(f, dtype=np.float32)


def gen_candidates(fe, anchors, fam):
    cands = []
    for a in anchors:
        cands.append(dict(cx=float(a["cx"]), cy=float(a["cy"]), s=float(a["med_size"]),
                          acount=float(a["count"]), adist=0.0))
    peaks = (peak_candidates(fe.sal, 55, 8, 0.30) if fam == "color"
             else peak_candidates(fe.sal, 40, 9, 0.35))
    for (px, py) in peaks:
        d = min((abs(a["cx"] - px) + abs(a["cy"] - py) for a in anchors), default=99.0)
        if d <= 4:
            continue
        cands.append(dict(cx=float(px), cy=float(py), s=float(FAM_MED_SIZE[fam]),
                          acount=0.0, adist=float(d)))
    return cands


# ------------------------------------------------------------------ pipeline
def train(df, anchors, prior, bg):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    X, y, Xp, T = [], [], [], []
    for _, r in df.iterrows():
        fam = family(r["height"])
        bgr = load_bgr(PUB, r["image_path"])
        if bgr is None:
            continue
        fe = FeatureExtractor(bgr, fam, prior[fam], bg[fam])
        gc = [((x0 + x1) / 2.0, (y0 + y1) / 2.0, max(x1 - x0, y1 - y0))
              for (x0, y0, x1, y1) in parse_boxes(r["boxes"])]
        for c in gen_candidates(fe, anchors[fam], fam):
            fv = fe.features(c["cx"], c["cy"], c["s"], c["acount"], c["adist"])
            X.append(fv)
            mm = None
            for (gx, gy, gs) in gc:
                if abs(gx - c["cx"]) <= POS_R[fam] and abs(gy - c["cy"]) <= POS_R[fam]:
                    mm = (gx, gy, gs); break
            if mm is None:
                y.append(0)
            else:
                y.append(1)
                Xp.append(fv)
                T.append((mm[0] - c["cx"], mm[1] - c["cy"], mm[2] - odd(c["s"])))
    X = np.stack(X); y = np.array(y); Xp = np.stack(Xp); T = np.array(T, np.float32)
    cls = HistGradientBoostingClassifier(random_state=SEED, **CLS_PARAMS).fit(X, y)
    regs = [HistGradientBoostingRegressor(random_state=SEED, **REG_PARAMS).fit(Xp, T[:, j]) for j in range(3)]
    return cls, regs, X.shape


def predict_image(fe, anchors, cls, regs, fam):
    cands = gen_candidates(fe, anchors[fam], fam)
    if not cands:
        return []
    Xp = np.stack([fe.features(c["cx"], c["cy"], c["s"], c["acount"], c["adist"]) for c in cands])
    p = cls.predict_proba(Xp)[:, 1]
    dcx = regs[0].predict(Xp); dcy = regs[1].predict(Xp); ds = regs[2].predict(Xp)
    boxes = []
    for i, c in enumerate(cands):
        ncx = c["cx"] + float(np.clip(dcx[i], -10, 10))
        ncy = c["cy"] + float(np.clip(dcy[i], -10, 10))
        ns = odd(c["s"] + float(np.clip(ds[i], -12, 12)))
        boxes.append(clip_box(box_from(ncx, ncy, ns), fe.W, fe.H))
    keep = nms(boxes, list(p), iou_thr=NMS_IOU)
    kept = sorted([(float(p[i]), boxes[i]) for i in keep], key=lambda z: -z[0])
    out = [kb for kb in kept if kb[0] >= TH_MAIN]
    if not out and kept and kept[0][0] >= FLOOR:
        out = [kept[0]]
    return out[:MAX_BOX]


def main():
    global PUB
    t0 = time.time()
    PUB = find_public_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join("working", "submission.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    log("public dir:", PUB)

    train_df = pd.read_csv(os.path.join(PUB, "train.csv"))
    test_df = pd.read_csv(os.path.join(PUB, "test.csv"))
    anchors = {f: build_anchors(train_df, f) for f in FAMS}
    prior = {f: prior_heatmap(train_df, f, 448 if f == "color" else 358, 448) for f in FAMS}
    bg = {f: build_background(train_df, f, PUB) for f in FAMS}
    log("anchors gray=%d color=%d" % (len(anchors["gray"]), len(anchors["color"])))

    cls, regs, shp = train(train_df, anchors, prior, bg)
    log("trained ranker+regressor rows=%d dim=%d (%.0fs)" % (shp[0], shp[1], time.time() - t0))

    rows = []; nb = 0
    for _, r in test_df.iterrows():
        fam = family(r["height"])
        bgr = load_bgr(PUB, r["image_path"])
        if bgr is None:
            rows.append((r["image_id"], "")); continue
        fe = FeatureExtractor(bgr, fam, prior[fam], bg[fam])
        out = predict_image(fe, anchors, cls, regs, fam)
        nb += len(out)
        pred = " ".join("%.4f %d %d %d %d" % (s, b[0], b[1], b[2], b[3]) for (s, b) in out)
        rows.append((r["image_id"], pred))
    sub = pd.DataFrame(rows, columns=["image_id", "prediction_string"]).fillna("")
    sub.to_csv(out_csv, index=False)
    log("wrote %s rows=%d boxes=%d avg=%.2f (%.0fs)"
        % (out_csv, len(sub), nb, nb / max(1, len(sub)), time.time() - t0))


if __name__ == "__main__":
    main()

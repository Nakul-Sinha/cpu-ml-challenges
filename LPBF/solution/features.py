"""Feature extraction for a candidate square box.

A FeatureExtractor precomputes per-image cue maps and integral images so that
inside/surround statistics for any candidate (cx,cy,size) are O(1). Features
combine multiple visual cues (contrast, edge density, texture, brightness and
color asymmetry, multi-scale center-surround) plus a learned spatial prior.
"""
import numpy as np
import cv2
import common as C  # skimage imported lazily inside patch_desc (optional feature)


CUES = ["gray", "grad", "lstd", "hf", "tophat", "blackhat"]
_LBP_P = 8
_LBP_BINS = _LBP_P + 2  # 'uniform' -> P+2 bins


class FeatureExtractor:
    def __init__(self, bgr, fam, prior_hm=None, use_patch=False, bg=None):
        self.fam = fam
        self.use_patch = use_patch
        self.bgr = bgr
        self.H, self.W = bgr.shape[:2]
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.gray = g
        maps = C.cue_maps(g)
        self.sal = maps["sal"]
        chan = {
            "gray": g,
            "grad": maps["grad"],
            "lstd": maps["lstd"],
            "hf": maps["hf"],
            "tophat": maps["tophat"],
            "blackhat": maps["blackhat"],
        }
        # deviation from the registered-family background = anomaly evidence
        self.has_bg = bg is not None and bg.shape == g.shape
        if self.has_bg:
            # remove global brightness offset so residual isolates local anomalies
            resid = (g - float(g.mean())) - (bg - float(bg.mean()))
            chan["resid"] = np.abs(resid)
            chan["sresid"] = resid
            self.sal = self.sal + 1.0 * C.robust_norm(
                cv2.GaussianBlur(np.abs(resid).astype(np.float32), (0, 0), 2))
        # colour channels (BGR)
        chan["B"] = bgr[..., 0].astype(np.float32)
        chan["G"] = bgr[..., 1].astype(np.float32)
        chan["R"] = bgr[..., 2].astype(np.float32)
        # colour-aware saliency: the colour family's alert features are red/white
        # lattices on a blue background, invisible to luminance texture alone.
        if fam == "color":
            rb = np.abs(chan["R"] - chan["B"])
            k = 7
            m1 = cv2.boxFilter(rb, cv2.CV_32F, (k, k))
            m2 = cv2.boxFilter(rb * rb, cv2.CV_32F, (k, k))
            rb_tex = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
            col = C.robust_norm(cv2.GaussianBlur(rb, (0, 0), 2)) + C.robust_norm(rb_tex)
            self.sal = self.sal + 0.8 * cv2.GaussianBlur(col.astype(np.float32), (0, 0), 2)
        self.integ = {k: C.Integrals(v) for k, v in chan.items()}
        self.integ["sal"] = C.Integrals(self.sal)
        self.resid_map = chan["resid"] if self.has_bg else None
        self.grad_map = maps["grad"]
        self.sal_hi = float(np.percentile(self.sal, 90))
        # global stats for normalisation
        self.g_mean = float(g.mean())
        self.g_std = float(g.std() + 1e-6)
        self.prior_hm = prior_hm

    def _peakfeats(self, cx, cy, s):
        """Box-max / peak features (integral images give means only). A corner
        anomaly is often a small peak the mean misses."""
        h = int(s / 2)
        x0 = int(np.clip(cx - h, 0, self.W - 1)); x1 = int(np.clip(cx + h, x0 + 1, self.W))
        y0 = int(np.clip(cy - h, 0, self.H - 1)); y1 = int(np.clip(cy + h, y0 + 1, self.H))
        sp = self.sal[y0:y1, x0:x1]
        gp = self.grad_map[y0:y1, x0:x1]
        out = [float(sp.max()), float((sp > self.sal_hi).mean()), float(gp.max()) / 255.0]
        if self.resid_map is not None:
            rp = self.resid_map[y0:y1, x0:x1]
            out.append(float(rp.max()) / 64.0)
        else:
            out.append(0.0)
        return out

    def patch_desc(self, cx, cy, s):
        """HOG + LBP + intensity-percentile descriptor of the candidate patch
        (challenge recommends HOG/LBP-style features). Captures corner/edge
        structure and local texture that separate active from inactive spots."""
        r = int(round(s * 0.7))
        x0 = int(np.clip(cx - r, 0, self.W - 2)); x1 = int(np.clip(cx + r, x0 + 2, self.W))
        y0 = int(np.clip(cy - r, 0, self.H - 2)); y1 = int(np.clip(cy + r, y0 + 2, self.H))
        patch = self.gray[y0:y1, x0:x1]
        p = cv2.resize(patch, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        pct = np.percentile(p, [5, 25, 50, 75, 95])
        std = p.std() + 1e-6
        pn = (p - p.mean()) / std
        from skimage.feature import hog, local_binary_pattern
        h = hog(pn, orientations=6, pixels_per_cell=(12, 12), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True)
        u8 = np.clip(p, 0, 255).astype(np.uint8)
        lbp = local_binary_pattern(u8, _LBP_P, 1.0, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=_LBP_BINS, range=(0, _LBP_BINS), density=True)
        return np.concatenate([
            h.astype(np.float32),
            lbp_hist.astype(np.float32),
            ((pct - self.g_mean) / self.g_std).astype(np.float32),
            np.array([std / 64.0, (p.max() - p.min()) / 255.0], np.float32),
        ])

    def _ring_mean(self, integ, cx, cy, s, m):
        h = s / 2.0
        A_in = s * s
        A_out = (s + 2 * m) ** 2
        m_in = integ.mean(cx - h, cy - h, cx + h, cy + h)
        m_big = integ.mean(cx - h - m, cy - h - m, cx + h + m, cy + h + m)
        ring = (m_big * A_out - m_in * A_in) / max(1.0, A_out - A_in)
        return m_in, ring

    def features(self, cx, cy, s, anchor_count=0.0, anchor_dist=99.0):
        f = []
        # geometry / prior
        f.append(cx / self.W)
        f.append(cy / self.H)
        f.append(s / 30.0)
        f.append(np.log1p(anchor_count))
        f.append(min(anchor_dist, 30.0) / 30.0)
        if self.prior_hm is not None:
            xi = int(np.clip(cx, 0, self.W - 1)); yi = int(np.clip(cy, 0, self.H - 1))
            f.append(float(self.prior_hm[yi, xi]))
        else:
            f.append(0.0)

        m = max(4, int(round(s * 0.5)))
        # per-cue inside mean, inside std (gray), center-surround contrast
        for key in CUES:
            integ = self.integ[key]
            m_in, ring = self._ring_mean(integ, cx, cy, s, m)
            f.append(m_in / 255.0)
            f.append((m_in - ring) / 255.0)
        # gray inside std (texture strength)
        _, gstd = self.integ["gray"].mean_std(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
        f.append(gstd / 64.0)
        # saliency inside + contrast
        m_in, ring = self._ring_mean(self.integ["sal"], cx, cy, s, m)
        f.append(m_in)
        f.append(m_in - ring)

        # deviation from the registered background (anomaly evidence)
        if self.has_bg:
            ri, rr = self._ring_mean(self.integ["resid"], cx, cy, s, m)
            f.append(ri / 64.0)             # mean |deviation| inside
            f.append((ri - rr) / 64.0)      # inside vs surround deviation
            si = self.integ["sresid"].mean(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
            f.append(si / 64.0)             # signed deviation (brighter/darker than typical)
            _, rstd = self.integ["resid"].mean_std(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
            f.append(rstd / 64.0)
        else:
            f.extend([0.0, 0.0, 0.0, 0.0])

        # colour asymmetry (helps the colour family: red/white features on blue)
        for key in ["R", "B"]:
            m_in, ring = self._ring_mean(self.integ[key], cx, cy, s, m)
            f.append((m_in - ring) / 255.0)
        # redness: R - B inside vs ring
        rin, rring = self._ring_mean(self.integ["R"], cx, cy, s, m)
        bin_, bring = self._ring_mean(self.integ["B"], cx, cy, s, m)
        f.append(((rin - bin_) - (rring - bring)) / 255.0)

        # multi-scale saliency (how the response changes with scale -> presence/size)
        for sc in (0.6, 1.4):
            ss = max(9, s * sc)
            mi, rg = self._ring_mean(self.integ["sal"], cx, cy, ss, max(4, int(ss * 0.5)))
            f.append(mi - rg)

        # brightness extremes inside (bright/dark asymmetry vs global)
        gm_in = self.integ["gray"].mean(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
        f.append((gm_in - self.g_mean) / self.g_std)
        # (peak/max box features were tried here and regressed; reverted)
        # family indicator (one shared model can still specialise per family)
        f.append(1.0 if self.fam == "color" else 0.0)
        base = np.asarray(f, dtype=np.float32)
        if self.use_patch:
            return np.concatenate([base, self.patch_desc(cx, cy, s)])
        return base


    def size_profile(self, cx, cy, sizes):
        """Multi-scale center-surround profile used by the size regressor: for
        each candidate size, how strongly the box stands out from its ring on
        several cues. The shape of this profile encodes the box extent."""
        prof = []
        for s in sizes:
            m = max(3, int(s * 0.4))
            for key in ("gray", "grad", "lstd", "hf"):
                integ = self.integ[key]
                m_in, ring = self._ring_mean(integ, cx, cy, s, m)
                prof.append((m_in - ring) / 255.0)
        return np.asarray(prof, dtype=np.float32)


SIZE_GRID = list(range(15, 42, 2))
FEATURE_DIM = None

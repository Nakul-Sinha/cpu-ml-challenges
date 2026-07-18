"""Candidate generation, scoring, size refinement, NMS and per-image selection.
Shared by model.py (training/eval) and solution.py (final pipeline) so train and
inference stay identical.
"""
import numpy as np
import common as C
from features import FeatureExtractor

FAM_MED_SIZE = {"gray": 25, "color": 29}


def _nearest_anchor_dist(cx, cy, anchors):
    if not anchors:
        return 99.0
    return min(abs(a["cx"] - cx) + abs(a["cy"] - cy) for a in anchors)


def gen_candidates(fe, anchors, fam):
    """Anchor locations UNION saliency peaks. Each candidate carries an approx
    center, a prior size, and prior stats. Returns list of dicts."""
    cands = []
    for a in anchors:
        cands.append(dict(cx=float(a["cx"]), cy=float(a["cy"]),
                          s=float(a["med_size"]), acount=float(a["count"]),
                          adist=0.0, src="anchor"))
    # saliency peaks for coverage of novel locations + hard negatives.
    if fam == "color":
        peaks = C.peak_candidates(fe.sal, max_peaks=55, min_dist=8, rel_thresh=0.30)
    else:
        peaks = C.peak_candidates(fe.sal, max_peaks=40, min_dist=9, rel_thresh=0.35)
    for (px, py) in peaks:
        d = _nearest_anchor_dist(px, py, anchors)
        if d <= 4:
            continue  # already covered by an anchor
        cands.append(dict(cx=float(px), cy=float(py),
                          s=float(FAM_MED_SIZE[fam]), acount=0.0,
                          adist=float(d), src="peak"))
    return cands


def featurize(fe, cand):
    return fe.features(cand["cx"], cand["cy"], cand["s"],
                       anchor_count=cand["acount"], anchor_dist=cand["adist"])


def _odd(s):
    s = int(round(s))
    if s % 2 == 0:
        s += 1
    return max(15, min(41, s))


def refine_size(fe, cx, cy, s0, sizes=None):
    """Pick the odd size maximising saliency center-surround at the (fixed)
    center; constrained near the prior size s0 to avoid drifting."""
    if sizes is None:
        lo = max(15, int(s0) - 8); hi = min(41, int(s0) + 8)
        sizes = list(range(lo | 1, hi + 1, 2))
    integ = fe.integ_sal if hasattr(fe, "integ_sal") else fe.integ["sal"]
    best = None
    for s in sizes:
        m = max(4, int(s * 0.5))
        h = s / 2.0
        A_in = s * s; A_out = (s + 2 * m) ** 2
        m_in = integ.mean(cx - h, cy - h, cx + h, cy + h)
        big = integ.mean(cx - h - m, cy - h - m, cx + h + m, cy + h + m)
        ring = (big * A_out - m_in * A_in) / max(1.0, A_out - A_in)
        v = m_in - ring
        if best is None or v > best[1]:
            best = (s, v)
    return best[0]


def box_from(cx, cy, s):
    h = s // 2
    x0 = int(round(cx - h)); y0 = int(round(cy - h))
    return [x0, y0, x0 + s, y0 + s]


def predict(fe, anchors, model, fam, thresh=0.5, iou_nms=0.30, max_box=25,
            use_size_refine=False, min_keep=0):
    cands = gen_candidates(fe, anchors, fam)
    if not cands:
        return []
    X = np.stack([featurize(fe, c) for c in cands])
    p = model.predict_proba(X)[:, 1]
    boxes, scores = [], []
    for c, pr in zip(cands, p):
        s = c["s"]
        if use_size_refine:
            s = refine_size(fe, c["cx"], c["cy"], s)
        b = C.clip_box(box_from(c["cx"], c["cy"], _odd(s)), fe.W, fe.H)
        boxes.append(b); scores.append(float(pr))
    keep = C.nms(boxes, scores, iou_thr=iou_nms)
    kept = [(scores[i], boxes[i]) for i in keep]
    kept.sort(key=lambda z: -z[0])
    out = [kb for kb in kept if kb[0] >= thresh][:max_box]
    if len(out) < min_keep:
        out = kept[:min_keep]
    return out

"""Faithful reimplementation of the official LPBF weighted object-localization
metric, used for local validation.

score = (0.25*wmAP@.50 + 0.40*wmAP@.75 + 0.25*wmAP@.85 + 0.10*wrecall@.85)
        * negative_image_penalty * duplicate_penalty   (clipped to [0,1])

Ground truth per image: list of [x0,y0,x1,y1] boxes.
Predictions per image: list of (score, x0, y0, x1, y1).
Both keyed by image_id.
"""
import numpy as np


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def ap_image_at_T(preds, truths, T):
    """preds: list of (score,x0,y0,x1,y1) ; truths: list of [x0,y0,x1,y1]."""
    n_t = len(truths)
    if n_t == 0:
        return 1.0 if len(preds) == 0 else 0.0
    if len(preds) == 0:
        return 0.0
    order = sorted(range(len(preds)), key=lambda i: -preds[i][0])
    matched = [False] * n_t
    tp = 0
    ap = 0.0
    seen = 0
    for i in order:
        pbox = preds[i][1:]
        best_j = -1
        best_iou = T
        for j in range(n_t):
            if matched[j]:
                continue
            v = iou(pbox, truths[j])
            if v >= best_iou:
                best_iou = v
                best_j = j
        seen += 1
        if best_j >= 0:
            matched[best_j] = True
            tp += 1
            # recall increased by 1/n_t; precision = tp/seen
            ap += (1.0 / n_t) * (tp / seen)
    return ap


def weighted_mAP_at_T(pred_map, truth_map, T):
    num = 0.0
    den = 0.0
    for img, truths in truth_map.items():
        w = 1.0 + min(3.0, 0.35 * len(truths))
        ap = ap_image_at_T(pred_map.get(img, []), truths, T)
        num += w * ap
        den += w
    return num / den if den > 0 else 0.0


def weighted_recall_at_T(pred_map, truth_map, T=0.85, box_w=1.35):
    total = 0.0
    matched_w = 0.0
    for img, truths in truth_map.items():
        preds = pred_map.get(img, [])
        n_t = len(truths)
        total += box_w * n_t
        if n_t == 0 or len(preds) == 0:
            continue
        # greedy max-IoU assignment truth<->pred, each pred used once
        pairs = []
        for ti, t in enumerate(truths):
            for pi, p in enumerate(preds):
                v = iou(p[1:], t)
                if v >= T:
                    pairs.append((v, ti, pi))
        pairs.sort(reverse=True)
        used_t = set(); used_p = set()
        for v, ti, pi in pairs:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti); used_p.add(pi)
            matched_w += box_w
    return matched_w / total if total > 0 else 0.0


def negative_image_penalty(pred_map, truth_map):
    neg = [img for img, t in truth_map.items() if len(t) == 0]
    if len(neg) == 0:
        return 1.0, 0.0
    fa = sum(1 for img in neg if len(pred_map.get(img, [])) > 0)
    rate = fa / len(neg)
    return max(0.55, 1.0 - 0.75 * rate), rate


def duplicate_penalty(pred_map, truth_map, T=0.50):
    matched_truth = 0
    dup = 0
    for img, truths in truth_map.items():
        preds = pred_map.get(img, [])
        for t in truths:
            cnt = sum(1 for p in preds if iou(p[1:], t) >= T)
            if cnt >= 1:
                matched_truth += 1
                dup += max(0, cnt - 1)
    rate = dup / max(1, matched_truth)
    return max(0.60, 1.0 - 0.40 * rate), rate


def full_score(pred_map, truth_map, verbose=False):
    m50 = weighted_mAP_at_T(pred_map, truth_map, 0.50)
    m75 = weighted_mAP_at_T(pred_map, truth_map, 0.75)
    m85 = weighted_mAP_at_T(pred_map, truth_map, 0.85)
    rec = weighted_recall_at_T(pred_map, truth_map, 0.85)
    base = 0.25 * m50 + 0.40 * m75 + 0.25 * m85 + 0.10 * rec
    negp, nrate = negative_image_penalty(pred_map, truth_map)
    dupp, drate = duplicate_penalty(pred_map, truth_map)
    score = float(np.clip(base * negp * dupp, 0.0, 1.0))
    if verbose:
        print(f"  wmAP@.50={m50:.4f} @.75={m75:.4f} @.85={m85:.4f} "
              f"wrecall@.85={rec:.4f}")
        print(f"  base={base:.4f} negp={negp:.3f}(rate {nrate:.2f}) "
              f"dupp={dupp:.3f}(rate {drate:.2f}) -> SCORE={score:.4f}")
    return score, dict(m50=m50, m75=m75, m85=m85, rec=rec, base=base,
                       negp=negp, dupp=dupp, score=score)


def _selftest():
    truth = {
        "a": [[10, 10, 30, 30], [100, 100, 120, 120]],
        "b": [[50, 50, 76, 76]],
        "c": [],  # negative
    }
    # perfect predictions
    perfect = {
        "a": [(0.9, 10, 10, 30, 30), (0.8, 100, 100, 120, 120)],
        "b": [(0.95, 50, 50, 76, 76)],
        "c": [],
    }
    s, d = full_score(perfect, truth, verbose=True)
    assert abs(s - 1.0) < 1e-6, s
    print("perfect -> 1.0 OK")

    empty = {"a": [], "b": [], "c": []}
    s, d = full_score(empty, truth, verbose=True)
    # positives AP 0; negative c correct (no pred) -> but base uses wmAP over all
    # images incl c AP=1. base = (.25+.40+.25)*wmAP + .10*rec.
    print("empty -> score", round(s, 4))

    # duplicate test: two boxes on same truth
    dup = {"a": [(0.9, 10, 10, 30, 30), (0.7, 11, 11, 31, 31)],
           "b": [(0.9, 50, 50, 76, 76)], "c": []}
    s, d = full_score(dup, truth, verbose=True)
    print("dup dupp should be <1:", round(d["dupp"], 3))

    # false alert on negative
    fa = {"a": [(0.9, 10, 10, 30, 30)], "b": [(0.9, 50, 50, 76, 76)],
          "c": [(0.5, 5, 5, 25, 25)]}
    s, d = full_score(fa, truth, verbose=True)
    print("false-alert negp should be <1:", round(d["negp"], 3))


if __name__ == "__main__":
    _selftest()

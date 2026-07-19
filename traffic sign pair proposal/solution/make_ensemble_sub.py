"""Build an offset-tuned ensemble submission from saved arm OOF/test logits.

Averages softmax(logits) across the given arm dirs, fits per-class additive
log-offsets on the FULL ensemble OOF (coordinate ascent maximizing comp_score),
applies them to the ensemble test logits, and writes a validated submission.

Usage: python make_ensemble_sub.py <out_submission.csv> <arm_dir> [<arm_dir> ...]
Prints the (in-sample) tuned OOF comp_score, the offsets, and the test dist.
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

ROUTES = ["JOINT_ALIGNED", "JOINT_OFFSET", "SEPARATION_ALIGNED",
          "SEPARATION_OFFSET", "INDEPENDENT_PROPOSALS"]
IDX = list(range(5))


def comp(y, p):
    mf1 = f1_score(y, p, labels=IDX, average="macro", zero_division=0)
    rec = recall_score(y, p, labels=IDX, average=None, zero_division=0)
    return 0.60 * mf1 + 0.25 * rec.mean() + 0.15 * rec.min()


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def tune_offsets(logits, labels, iters=5, grid=None):
    if grid is None:
        grid = np.linspace(-4, 4, 41)
    off = np.zeros(5)
    best = comp(labels, logits.argmax(1))
    for _ in range(iters):
        for c in range(5):
            bestv = off[c]
            for v in grid:
                off[c] = v
                s = comp(labels, (logits + off).argmax(1))
                if s > best:
                    best, bestv = s, v
            off[c] = bestv
    return off, best


def main():
    out = sys.argv[1]
    dirs = sys.argv[2:]
    assert dirs, "pass arm dirs"

    # OOF ensemble (align by ids)
    ref_ids = ref_labels = None
    oof_probs = []
    for d in dirs:
        z = np.load(os.path.join(d, "oof.npz"), allow_pickle=True)
        ids, lg, lb = z["ids"].astype(str), z["logits"].astype(float), z["labels"].astype(int)
        o = np.argsort(ids)
        ids, lg, lb = ids[o], lg[o], lb[o]
        if ref_ids is None:
            ref_ids, ref_labels = ids, lb
        else:
            assert np.array_equal(ids, ref_ids) and np.array_equal(lb, ref_labels)
        oof_probs.append(softmax(lg))
    ens_oof = np.mean(oof_probs, axis=0)
    oof_logit = np.log(np.clip(ens_oof, 1e-9, 1))

    off, tuned = tune_offsets(oof_logit, ref_labels)
    raw = comp(ref_labels, oof_logit.argmax(1))
    print(f"ensemble arms: {dirs}")
    print(f"OOF comp raw={raw:.4f}  in-sample-tuned={tuned:.4f}")
    print(f"offsets={np.round(off,3).tolist()}")

    # TEST ensemble (align by ids)
    ref_tids = None
    test_probs = []
    for d in dirs:
        z = np.load(os.path.join(d, "test.npz"), allow_pickle=True)
        ids, lg = z["ids"].astype(str), z["logits"].astype(float)
        o = np.argsort(ids)
        ids, lg = ids[o], lg[o]
        if ref_tids is None:
            ref_tids = ids
        else:
            assert np.array_equal(ids, ref_tids)
        test_probs.append(softmax(lg))
    ens_test = np.mean(test_probs, axis=0)
    test_logit = np.log(np.clip(ens_test, 1e-9, 1))
    pred = (test_logit + off).argmax(1)

    sub = pd.DataFrame({"id": ref_tids, "screening_route": [ROUTES[i] for i in pred]})
    sub.to_csv(out, index=False)
    dist = {r: int((sub["screening_route"] == r).sum()) for r in ROUTES}
    print(f"wrote {out}  n={len(sub)}  test dist={dist}")


if __name__ == "__main__":
    main()

"""Central OOF scorer for Traffic Sign Proposal Coupling experiment arms.

Each arm dir must contain oof.npz with:
  ids    : (N,) str      test/train ids in a FIXED order (train.csv order)
  logits : (N,5) float   RAW pre-softmax logits, columns in ROUTES order
  labels : (N,) int      gold class index in ROUTES order

Usage: python score_oof.py <arm_dir> [<arm_dir> ...]
Prints raw-argmax and honestly nested per-class logit-offset-tuned comp_score
for every single arm and every arm combination (softmax-mean ensemble).
"""
import sys, os, itertools
import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROUTES = ["JOINT_ALIGNED", "JOINT_OFFSET", "SEPARATION_ALIGNED",
          "SEPARATION_OFFSET", "INDEPENDENT_PROPOSALS"]
IDX = list(range(5))


def comp(y, p):
    mf1 = f1_score(y, p, labels=IDX, average="macro", zero_division=0)
    rec = recall_score(y, p, labels=IDX, average=None, zero_division=0)
    return 0.60 * mf1 + 0.25 * rec.mean() + 0.15 * rec.min(), mf1, rec.mean(), rec.min(), rec


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def tune_offsets(logits, labels, iters=4, grid=None):
    """Coordinate ascent on 5 additive logit offsets maximizing comp_score."""
    if grid is None:
        grid = np.linspace(-4, 4, 41)
    off = np.zeros(5)
    best = comp(labels, logits.argmax(1))[0]
    for _ in range(iters):
        for c in range(5):
            keep = off[c]
            bestv = keep
            for v in grid:
                off[c] = v
                s = comp(labels, (logits + off).argmax(1))[0]
                if s > best:
                    best, bestv = s, v
            off[c] = bestv
    return off


def nested_tuned(logits, labels, seed=0):
    """Honest: fit offsets on 4/5 of OOF, apply to held-out 1/5, aggregate."""
    n = len(labels)
    pred = np.zeros(n, int)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(logits, labels):
        off = tune_offsets(logits[tr], labels[tr])
        pred[va] = (logits[va] + off).argmax(1)
    return comp(labels, pred)


def load(d):
    z = np.load(os.path.join(d, "oof.npz"), allow_pickle=True)
    return z["ids"].astype(str), z["logits"].astype(float), z["labels"].astype(int)


def main():
    dirs = sys.argv[1:]
    assert dirs, "pass arm dirs"
    arms = {}
    ref_ids = ref_labels = None
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        ids, logits, labels = load(d)
        order = np.argsort(ids)
        ids, logits, labels = ids[order], logits[order], labels[order]
        if ref_ids is None:
            ref_ids, ref_labels = ids, labels
        else:
            assert np.array_equal(ids, ref_ids), f"{name} ids differ"
            assert np.array_equal(labels, ref_labels), f"{name} labels differ"
        arms[name] = softmax(logits)

    def evaluate(prob):
        logit = np.log(np.clip(prob, 1e-9, 1))
        raw = comp(ref_labels, logit.argmax(1))
        tun = nested_tuned(logit, ref_labels)
        return raw, tun

    rows = []
    for r in range(1, len(arms) + 1):
        for combo in itertools.combinations(arms.keys(), r):
            prob = np.mean([arms[c] for c in combo], axis=0)
            raw, tun = evaluate(prob)
            rows.append(("+".join(combo), raw, tun))

    rows.sort(key=lambda x: x[2][0], reverse=True)
    print(f"{'combo':40s} {'raw':>7s} {'tuned':>7s}  {'mf1':>5s} {'bal':>5s} {'min':>5s}")
    print("-" * 78)
    for name, raw, tun in rows:
        print(f"{name:40s} {raw[0]:7.4f} {tun[0]:7.4f}  {tun[1]:.3f} {tun[2]:.3f} {tun[3]:.3f}")
    print("\nPer-class recall (tuned) for the top combo:")
    top = rows[0]
    print("  ", top[0], "->", dict(zip(ROUTES, np.round(top[1][4], 3))))


if __name__ == "__main__":
    main()

"""Decoder diagnostics on full bucket-0 (index = buckets 1-19).

Isolates where the oracle gap lives and tests retrieval-posterior MBR, which
should exploit chrF partial credit far better than reranker-softmax argmax.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "5")
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists
sys.path.insert(0, "runs/B1")
import pipeline_v1 as P

t0 = time.time()
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
train["_bkt"] = train.masked_docstring.map(P.bucket)
val = train[train._bkt == 0].copy()
fold = train[train._bkt != 0].copy()
idx, glob = P.build_index(fold)
gtop = P.global_top(glob, 12)
print(f"[setup] val {len(val)} idx built {time.time()-t0:.1f}s", flush=True)

# level weights for retrieval posterior (deeper == more specific == higher)
LVLW = {"l2r2": 3.0, "l2r1": 2.2, "l1r2": 2.2, "l1r1": 1.6,
        "l1": 1.0, "r1": 1.0, "l2": 0.8, "r2": 0.8}


def row_data(masked, code):
    rc = P.row_ctx(masked)
    cc, idents = P.code_features(code)
    cands, src = P.gen_candidates(rc, idx, gtop, cc)
    if not cands:
        return ["value of the"], np.array([1.0]), np.array([1.0]), np.array([1.0])
    # weights
    w_back = np.zeros(len(cands))   # deepest-level prob (backoff)
    w_sum = np.zeros(len(cands))    # level-weighted sum of probs
    w_glob = np.zeros(len(cands))   # global frequency
    cache = {}
    for i, c in enumerate(cands):
        deepest = None
        for name, _, _ in P.LEVELS:
            total, rankmap = P.level_stats(idx, name, rc["keys"][name], cache)
            rcinfo = rankmap.get(c)
            if rcinfo is not None and total > 0:
                prob = rcinfo[1] / total
                w_sum[i] += LVLW[name] * prob
                if deepest is None:
                    deepest = prob
        w_back[i] = deepest if deepest is not None else 0.0
        w_glob[i] = glob.get(c, 0)
    return cands, w_back, w_sum, w_glob


refs = val.target_span.astype(str).tolist()
rows = [row_data(m, c) for m, c in zip(val.masked_docstring.values, val.code_context.values)]
print(f"[rowdata] {time.time()-t0:.1f}s", flush=True)

# ---- oracle: full pool vs retrieval-only (w_back>0) ----
orc_full, orc_ret = [], []
for (cands, wb, ws, wg), tgt in zip(rows, refs):
    orc_full.append(max(f_pooled(c, tgt) for c in cands))
    ret = [c for c, b in zip(cands, wb) if b > 0]
    orc_ret.append(max((f_pooled(c, tgt) for c in ret), default=0.0))
print(f"[oracle] full-pool {np.mean(orc_full):.4f}  retrieval-only {np.mean(orc_ret):.4f}")


def decode(fn):
    return score_lists([fn(*r) for r in rows], refs)


# D0 backoff top1 (first pool candidate)
print(f"[D0 backoff-top1]      {decode(lambda c,wb,ws,wg: c[0]):.4f}")
# D1 argmax level-weighted retrieval
print(f"[D1 argmax w_sum]      {decode(lambda c,wb,ws,wg: c[int(np.argmax(ws))]):.4f}")
# D2 argmax global freq
print(f"[D2 argmax w_glob]     {decode(lambda c,wb,ws,wg: c[int(np.argmax(wg))]):.4f}")


def mbr(cands, weights, temp, floor=1e-3, topm=25):
    w = np.asarray(weights, dtype=np.float64) + floor
    if temp != 1.0:
        w = w ** (1.0 / temp)
    w = w / w.sum()
    m = min(topm, len(cands))
    # restrict expectation to top-m by weight for speed
    order = np.argsort(-w)[:m]
    cw = w[order]; cw = cw / cw.sum()
    cc = [cands[j] for j in order]
    best_j, best_v = order[0], -1.0
    for a in range(m):
        ev = 0.0
        for b in range(m):
            ev += cw[b] * f_pooled(cc[a], cc[b])
        if ev > best_v:
            best_v, best_j = ev, order[a]
    return cands[best_j]


for temp in [1.0, 0.5, 0.35, 0.25]:
    s = decode(lambda c, wb, ws, wg: mbr(c, ws, temp))
    print(f"[MBR w_sum temp={temp}]  {s:.4f}  ({time.time()-t0:.1f}s)", flush=True)

# MBR under backoff weights
for temp in [0.5, 0.35]:
    s = decode(lambda c, wb, ws, wg: mbr(c, wb, temp))
    print(f"[MBR w_back temp={temp}] {s:.4f}")

print(f"total {time.time()-t0:.1f}s")

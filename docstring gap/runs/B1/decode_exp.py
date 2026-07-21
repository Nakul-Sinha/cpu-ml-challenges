"""Sweep decoders over cached bucket-0 reranker scores + retrieval weights.

Precomputes a pairwise chrF matrix per row over the MBR candidate subset ONCE,
then every decoder config is a cheap weighted argmax. The trained reranker
drives everything: it defines the candidate subset and the primary weight;
MBR under the reranker posterior exploits chrF partial credit.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "5")
import sys, time, pickle
import numpy as np

sys.path.insert(0, "solution")
from chrf import f_pooled, score_lists
sys.path.insert(0, "runs/B1")

M = 12  # MBR subset size (union of top reranker + top retrieval)
t0 = time.time()
with open("runs/B1/valcache.pkl", "rb") as f:
    C = pickle.load(f)
rows, refs = C["rows"], C["refs"]
print(f"[load] {len(rows)} rows {time.time()-t0:.1f}s", flush=True)


def softmax(x, temp):
    x = np.asarray(x, dtype=np.float64) / temp
    x = np.exp(x - x.max())
    return x / x.sum()


# build per-row MBR subset (indices into cands) + pairwise chrF matrix, ONCE
subsets = []
for r in rows:
    cands, rr, wsum = r["cands"], r["rr"], r["wsum"]
    n = len(cands)
    if n == 1:
        subsets.append((cands, np.array([[1.0]], dtype=np.float32), rr, wsum, r["wglob"]))
        continue
    top_rr = list(np.argsort(-rr)[:M])
    top_ws = list(np.argsort(-wsum)[:M])
    sel = list(dict.fromkeys(top_rr + top_ws))[:M]
    cs = [cands[j] for j in sel]
    k = len(cs)
    Kmat = np.empty((k, k), dtype=np.float32)
    for a in range(k):
        Kmat[a, a] = 1.0
        for b in range(a + 1, k):
            v = f_pooled(cs[a], cs[b])
            Kmat[a, b] = v
            Kmat[b, a] = v
    subsets.append((cs, Kmat, rr[sel], wsum[sel], r["wglob"][sel]))
print(f"[pairwise built] {time.time()-t0:.1f}s", flush=True)


def eval_argmax(weight_key):
    preds = []
    for r in rows:
        w = r[weight_key]
        preds.append(r["cands"][int(np.argmax(w))])
    return score_lists(preds, refs)


def eval_mbr(weight_fn):
    """weight_fn(rr_sub, wsum_sub, wglob_sub) -> weight over subset."""
    preds = []
    for cs, K, rr, ws, wg in subsets:
        if len(cs) == 1:
            preds.append(cs[0]); continue
        w = weight_fn(rr, ws, wg)
        w = w / w.sum()
        ev = K @ w
        preds.append(cs[int(np.argmax(ev))])
    return score_lists(preds, refs)


print(f"[argmax rr]        {eval_argmax('rr'):.4f}", flush=True)
print(f"[argmax wsum]      {eval_argmax('wsum'):.4f}", flush=True)

# MBR under reranker-softmax posterior, temperature sweep
for temp in [2.0, 1.0, 0.6, 0.4, 0.25]:
    s = eval_mbr(lambda rr, ws, wg, t=temp: softmax(rr, t))
    print(f"[MBR rr-softmax T={temp}]   {s:.4f}", flush=True)

# MBR under retrieval posterior (wsum), power sweep
for pw in [1.0, 1.5, 2.0]:
    s = eval_mbr(lambda rr, ws, wg, p=pw: (ws + 1e-3) ** p)
    print(f"[MBR wsum^{pw}]          {s:.4f}", flush=True)

# MBR under blend: reranker-softmax * retrieval
for temp in [1.0, 0.6, 0.4]:
    for a in [0.5, 1.0]:
        s = eval_mbr(lambda rr, ws, wg, t=temp, aa=a: softmax(rr, t) * ((ws + 1e-3) ** aa))
        print(f"[MBR blend T={temp} ws^{a}]  {s:.4f}", flush=True)

print(f"total {time.time()-t0:.1f}s", flush=True)

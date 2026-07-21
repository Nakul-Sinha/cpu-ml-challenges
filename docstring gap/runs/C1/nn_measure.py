"""C1 pre-step: quick-measure NN top-10 union-oracle delta.

Base pool = B2 PoolBuilder pool (val_pools.pkl) UNION B3 LMBridge top-10 (lm_val.csv).
Candidate NN pool = B3 neural top-10 (nn_val.csv).
Include NN candidates in the integration ONLY if it lifts union oracle by >= +0.01.
Measured on a 3k-row sample (recipe) and on the full bucket-0 for reliability.
Gram-cached f_pooled for speed. Fit-on-test never occurs (these are val artifacts).
"""
import sys, os, time, pickle
from collections import Counter
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/docgap"))
sys.path.insert(0, "solution")
from chrf import f_pooled

t0 = time.time()
# ---- load the three candidate sources on bucket-0 ----
with open("runs/b2/out/val_pools.pkl", "rb") as f:
    P2 = pickle.load(f)
ids2, refs2, pools2 = P2["ids"], P2["refs"], P2["pools"]
b2 = {rid: [t for t, _, _ in pool] for rid, pool in zip(ids2, pools2)}
id2ref = {rid: str(r) for rid, r in zip(ids2, refs2)}

lm = pd.read_csv("runs/B3/lm_val.csv", keep_default_na=False)
lm_c = {i: [c for c in s.split("\t") if c] for i, s in zip(lm.id.values, lm.cands.astype(str).values)}

nn = pd.read_csv("runs/B3/nn_val.csv", keep_default_na=False)
nn_c = {i: [c for c in s.split("\t") if c] for i, s in zip(nn.id.values, nn.nn_top10.astype(str).values)}

ids = list(id2ref.keys())
assert set(ids) == set(lm_c) == set(nn_c), "id mismatch across sources"
print(f"[load] {len(ids)} bucket-0 rows  {time.time()-t0:.0f}s", flush=True)

# ---- gram-cached oracle helper ----
_gcache = {}
def grams(s):
    g = _gcache.get(s)
    if g is None:
        c = Counter()
        for n in range(1, 7):
            for i in range(len(s) - n + 1):
                c[s[i:i+n]] += 1
        g = (c, sum(c.values()))
        _gcache[s] = g
    return g

def fmax(cands, ref):
    rg, tr = grams(ref)
    if tr == 0:
        return 0.0
    best = 0.0
    for c in cands:
        cg, tp = grams(c)
        if tp == 0:
            continue
        # iterate smaller
        if len(cg) <= len(rg):
            m = sum(min(v, rg.get(k, 0)) for k, v in cg.items())
        else:
            m = sum(min(v, cg.get(k, 0)) for k, v in rg.items())
        if m == 0:
            continue
        p = m / tp; r = m / tr
        f = 2 * p * r / (p + r)
        if f > best:
            best = f
    return best

def eval_subset(subset_ids):
    base_o, nn_o = [], []
    for rid in subset_ids:
        ref = id2ref[rid]
        base = set(b2[rid]) | set(lm_c[rid])
        withnn = base | set(nn_c[rid])
        base_o.append(fmax(base, ref))
        nn_o.append(fmax(withnn, ref))
    return float(np.mean(base_o)), float(np.mean(nn_o))

# 3k sample (recipe) + full
rng = np.random.RandomState(0)
samp = list(rng.choice(ids, size=3000, replace=False))
b3k, n3k = eval_subset(samp)
print(f"[3k sample]  base(B2+LM) oracle={b3k:.4f}  +NN oracle={n3k:.4f}  delta={n3k-b3k:+.4f}", flush=True)
bfull, nfull = eval_subset(ids)
print(f"[full 11462] base(B2+LM) oracle={bfull:.4f}  +NN oracle={nfull:.4f}  delta={nfull-bfull:+.4f}", flush=True)
print(f"DECISION: include NN candidates = {(n3k-b3k) >= 0.01}  (threshold +0.01 on 3k)", flush=True)
print(f"[done] {time.time()-t0:.0f}s", flush=True)

"""D2 integration analysis: does fine-tuned CodeT5 add coverage over retrieval,
and how does it compare head-to-head with C2's fine-tuned t5-small on the SAME
500 bucket-0 rows (seed=7)? Reports standalone chrF, union-oracle lift over the
B2 pool, two-way oracle vs C1 reranker PICK, corr(seq_logprob, chrF), and a
direct codet5-vs-t5 side-by-side.
"""
import sys, os, pickle, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, "solution")
from chrf import f_pooled

def bucket(s): return int(hashlib.md5(s.encode("utf-8","ignore")).hexdigest()[:8],16)%20

HERE = os.path.join("runs","D2")
dump = pd.read_csv(os.path.join(HERE, "ct5_dump_ft_500.csv"), keep_default_na=False)
ids = dump.id.tolist(); idset = set(ids)
print(f"CodeT5 FT dump rows: {len(dump)}")

# canonical refs
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
b = train.masked_docstring.map(bucket)
val = train[b==0]
id2tgt = dict(zip(val.id.values, val.target_span.astype(str).values))
refs = [id2tgt[i] for i in ids]

# CodeT5 preds
ct5pred = dict(zip(dump.id, dump.ct5_pred.astype(str)))
ct5f = {i: f_pooled(str(p), id2tgt[i]) for i,p in zip(dump.id, dump.ct5_pred.astype(str))}

# B1 retrieval PICK + C1 reranker PICK
b1 = pd.read_csv("runs/B1/val_pred.csv", keep_default_na=False)
b1pred = dict(zip(b1.id, b1.iloc[:,1].astype(str)))
rpick_f = {i: f_pooled(b1pred.get(i,""), id2tgt[i]) for i in ids}
c1 = pd.read_csv("runs/C1/val_pred.csv", keep_default_na=False)
c1pred = dict(zip(c1.id, c1.iloc[:,1].astype(str)))
c1pick_f = {i: f_pooled(c1pred.get(i,""), id2tgt[i]) for i in ids}

# B2 pool ORACLE + union
P = pickle.load(open("runs/b2/out/val_pools.pkl","rb"))
pool_map = {rid: [t for t,_,_ in pool] for rid,pool in zip(P['ids'], P['pools'])}
rorc_f, union_f = {}, {}
ct5_in_pool = 0
for i in ids:
    r = id2tgt[i]
    cl = pool_map.get(i, [])
    ro = max((f_pooled(c, r) for c in cl), default=0.0)
    rorc_f[i] = ro
    union_f[i] = max(ro, ct5f[i])
    if ct5pred[i] in set(cl): ct5_in_pool += 1

def m(d): return float(np.mean([d[i] for i in ids]))
print(f"\n=== MEANS over {len(ids)} rows (bucket-0 sample, seed=7) ===")
print(f"  retrieval PICK  (B1)         : {m(rpick_f):.4f}")
print(f"  C1 reranker PICK             : {m(c1pick_f):.4f}")
print(f"  CodeT5 FT standalone         : {m(ct5f):.4f}")
print(f"  retrieval ORACLE (B2 pool)   : {m(rorc_f):.4f}")
print(f"  UNION oracle (B2 pool U CT5) : {m(union_f):.4f}  (lift over pool {m(union_f)-m(rorc_f):+.4f})")
twoway = {i: max(c1pick_f[i], ct5f[i]) for i in ids}
print(f"  TWO-WAY oracle max(C1pick,CT5): {m(twoway):.4f}  (headroom over C1 pick {m(twoway)-m(c1pick_f):+.4f})")
print(f"  CodeT5 pred already in B2 pool: {ct5_in_pool}/{len(ids)} ({100*ct5_in_pool/len(ids):.1f}%)")
agree = sum(1 for i in ids if ct5pred[i] == c1pred.get(i,""))
print(f"  CodeT5 pred == C1 pick exactly: {agree}/{len(ids)} ({100*agree/len(ids):.1f}%)")

# agreement / rescue
hi = [i for i in ids if c1pick_f[i] > 0.8]
lo = [i for i in ids if c1pick_f[i] < 0.3]
print(f"\n=== AGREEMENT (vs C1 pick) ===")
print(f"  C1 pick good (f>0.8): {len(hi)} rows -> mean CT5 f = {np.mean([ct5f[i] for i in hi]) if hi else 0:.4f}")
print(f"  C1 pick bad  (f<0.3): {len(lo)} rows -> CT5 rescues (f>0.5): {sum(ct5f[i]>0.5 for i in lo)}/{len(lo)}"
      f"  mean CT5 f = {np.mean([ct5f[i] for i in lo]) if lo else 0:.4f}")
better = [i for i in ids if ct5f[i] > rorc_f[i] + 1e-9]
print(f"  CodeT5 f > retrieval ORACLE  : {len(better)}/{len(ids)} rows (adds coverage the pool lacks)")
sc = dict(zip(dump.id, dump.ct5_seq_logprob))
print(f"  corr(C1 pick f, CodeT5 f)    = {np.corrcoef([c1pick_f[i] for i in ids],[ct5f[i] for i in ids])[0,1]:.3f}")
print(f"  corr(CodeT5 seq_logprob, f)  = {np.corrcoef([sc[i] for i in ids],[ct5f[i] for i in ids])[0,1]:.3f}")

# gate realization: seq_logprob threshold sweep (test-only gate like C2's -0.35)
print(f"\n=== seq_logprob GATE (what CT5 realizes over C1 pick if we accept CT5 when conf>=thr) ===")
base = m(c1pick_f)
for thr in [-0.15, -0.25, -0.35, -0.5]:
    acc = {i: (ct5f[i] if sc[i] >= thr else c1pick_f[i]) for i in ids}
    cov = np.mean([sc[i] >= thr for i in ids])
    print(f"  thr={thr:+.2f}: coverage={cov*100:4.1f}%  blended chrF={m(acc):.4f}  (realized {m(acc)-base:+.4f})")

# ---- HEAD-TO-HEAD vs C2 t5 dump (same seed=7 500 ids) ----
t5path = "runs/C2/t5_dump_ft_500.csv"
if os.path.exists(t5path):
    t5 = pd.read_csv(t5path, keep_default_na=False)
    t5f = {i: f_pooled(str(p), id2tgt[i]) for i,p in zip(t5.id, t5.t5_pred.astype(str))}
    common = [i for i in ids if i in t5f]
    print(f"\n=== HEAD-TO-HEAD codet5 vs t5 (FT, {len(common)} common rows) ===")
    print(f"  t5-small FT standalone   : {np.mean([t5f[i] for i in common]):.4f}")
    print(f"  codet5   FT standalone   : {np.mean([ct5f[i] for i in common]):.4f}")
    ct5_wins = sum(ct5f[i] > t5f[i] + 1e-9 for i in common)
    t5_wins  = sum(t5f[i] > ct5f[i] + 1e-9 for i in common)
    tie      = sum(abs(t5f[i]-ct5f[i]) <= 1e-9 for i in common)
    print(f"  per-row: codet5 wins {ct5_wins}, t5 wins {t5_wins}, tie {tie}")
    best2 = {i: max(t5f[i], ct5f[i]) for i in common}
    print(f"  oracle max(t5,codet5)    : {np.mean([best2[i] for i in common]):.4f}  (both-generators ceiling)")
    # union of BOTH neural + pool
    union3 = {i: max(rorc_f[i], t5f[i], ct5f[i]) for i in common}
    print(f"  UNION(pool,t5,codet5)    : {np.mean([union3[i] for i in common]):.4f}  (vs pool-only {np.mean([rorc_f[i] for i in common]):.4f})")
else:
    print("\n(no C2 t5 dump found for head-to-head)")

"""C2 integration analysis: does fine-tuned T5 add coverage over retrieval?
Joins the 500-row T5 FT dump against B1 (retrieval PICK), B2 (retrieval pool ORACLE),
and canonical bucket-0 refs. Reports union oracle + agreement/rescue stats.
"""
import sys, os, pickle, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, "solution")
from chrf import f_pooled

def bucket(s): return int(hashlib.md5(s.encode("utf-8","ignore")).hexdigest()[:8],16)%20

HERE = os.path.join("runs","C2")
dump = pd.read_csv(os.path.join(HERE, "t5_dump_ft_500.csv"), keep_default_na=False)
ids = dump.id.tolist()
idset = set(ids)
print(f"T5 FT dump rows: {len(dump)}")

# canonical refs
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
b = train.masked_docstring.map(bucket)
val = train[b==0]
id2tgt = dict(zip(val.id.values, val.target_span.astype(str).values))
refs = [id2tgt[i] for i in ids]

# T5 preds
t5pred = dict(zip(dump.id, dump.t5_pred.astype(str)))
t5f = {i: f_pooled(str(p), id2tgt[i]) for i,p in zip(dump.id, dump.t5_pred.astype(str))}

# B1 retrieval PICK
b1 = pd.read_csv("runs/B1/val_pred.csv", keep_default_na=False)
b1pred = dict(zip(b1.id, b1.iloc[:,1].astype(str)))
rpick_f = {i: f_pooled(b1pred.get(i,""), id2tgt[i]) for i in ids}

# B2 pool ORACLE + union
P = pickle.load(open("runs/b2/out/val_pools.pkl","rb"))
pool_map = {rid: [t for t,_,_ in pool] for rid,pool in zip(P['ids'], P['pools'])}
rorc_f, union_f = {}, {}
t5_in_pool = 0
for i in ids:
    r = id2tgt[i]
    cl = pool_map.get(i, [])
    ro = max((f_pooled(c, r) for c in cl), default=0.0)
    rorc_f[i] = ro
    union_f[i] = max(ro, t5f[i])
    if t5pred[i] in set(cl): t5_in_pool += 1

def m(d): return float(np.mean([d[i] for i in ids]))
print(f"\n=== MEANS over {len(ids)} rows (bucket-0 sample) ===")
print(f"  retrieval PICK  (B1)      : {m(rpick_f):.4f}")
print(f"  T5 FT standalone          : {m(t5f):.4f}")
print(f"  retrieval ORACLE (B2 pool): {m(rorc_f):.4f}")
print(f"  UNION oracle (B2 ∪ T5)    : {m(union_f):.4f}  (lift over pool {m(union_f)-m(rorc_f):+.4f})")
print(f"  T5 pred already in B2 pool: {t5_in_pool}/{len(ids)} ({100*t5_in_pool/len(ids):.1f}%)")

# agreement: when retrieval PICK is good, is T5 good?
hi = [i for i in ids if rpick_f[i] > 0.8]
lo = [i for i in ids if rpick_f[i] < 0.3]
print(f"\n=== AGREEMENT ===")
print(f"  retrieval PICK good (f>0.8): {len(hi)} rows -> mean T5 f = {np.mean([t5f[i] for i in hi]) if hi else 0:.4f}")
print(f"  retrieval PICK bad  (f<0.3): {len(lo)} rows -> T5 rescues (f>0.5): {sum(t5f[i]>0.5 for i in lo)}/{len(lo)}"
      f"  mean T5 f = {np.mean([t5f[i] for i in lo]) if lo else 0:.4f}")
# T5 strictly beats retrieval oracle (adds new coverage)
better = [i for i in ids if t5f[i] > rorc_f[i] + 1e-9]
print(f"  T5 f > retrieval ORACLE   : {len(better)}/{len(ids)} rows (T5 adds coverage the pool lacks)")
corr = np.corrcoef([rpick_f[i] for i in ids],[t5f[i] for i in ids])[0,1]
print(f"  corr(retrieval PICK f, T5 f) = {corr:.3f}")
# seq_logprob as confidence: does high T5 confidence => high T5 f?
sc = dict(zip(dump.id, dump.t5_seq_logprob))
print(f"  corr(T5 seq_logprob, T5 f)   = {np.corrcoef([sc[i] for i in ids],[t5f[i] for i in ids])[0,1]:.3f}")

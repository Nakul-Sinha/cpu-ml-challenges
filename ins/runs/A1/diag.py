"""Diagnose token-level recall by structural signal, using saved OOF probs."""
import sys, json, re
import numpy as np, pandas as pd
sys.path.insert(0, "solution")

train = pd.read_csv("dataset/train.csv").set_index("id")
op = pd.read_csv("runs/A1/oof_token_probs.csv")
op["w"] = [train.loc[i, "text"][s:e] for i, s, e in zip(op.id, op.start, op.end)]

def has_struct(w):
    inner = w[1:-1] if len(w) > 2 else ""
    return any(c in inner for c in [":", "*", "/", "_"])
op["struct"] = op.w.apply(has_struct)

print("=== token counts ===")
print(op.groupby(["lang"]).agg(n=("y","size"), pos=("y","sum")))

for L in ["de","en","it"]:
    d = op[op.lang==L]
    ed = d[d.y==1]
    print(f"\n--- {L}: {len(ed)} edited tokens ---")
    for name, sub in [("struct(:/*_) edited", ed[ed.struct]), ("plain edited", ed[~ed.struct])]:
        if len(sub)==0:
            print(f"  {name}: 0"); continue
        pr = sub.proba.values
        print(f"  {name}: n={len(sub)} proba mean={pr.mean():.3f} median={np.median(pr):.3f} "
              f"| recall@0.2={np.mean(pr>=0.2):.2f} @0.35={np.mean(pr>=0.35):.2f} @0.5={np.mean(pr>=0.5):.2f}")
    # precision context: how many NON-edited tokens are high-proba (false positives)
    neg = d[d.y==0]
    print(f"  neg tokens n={len(neg)}: frac>=0.2={np.mean(neg.proba>=0.2):.3f} >=0.5={np.mean(neg.proba>=0.5):.3f}")
    # structural non-edited: are they truly rare / do they get predicted?
    ns = neg[neg.struct]
    print(f"  neg struct tokens n={len(ns)}: frac>=0.5={np.mean(ns.proba>=0.5) if len(ns) else 0:.3f} (these hurt precision if edit-rate<1)")

# how many edited tokens are 'plain' (the hard contextual case) per language
print("\n=== edited-token composition ===")
for L in ["de","en","it"]:
    ed = op[(op.lang==L)&(op.y==1)]
    print(f"  {L}: struct={ed.struct.sum()} plain={(~ed.struct).sum()} ({(~ed.struct).mean()*100:.0f}% plain)")

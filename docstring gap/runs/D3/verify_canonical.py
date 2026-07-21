"""D3 step-2 independent verification: re-score the shipped config's bucket-0 predictions
with the OFFICIAL canonical scorer module solution/chrf.py (NOT solution_v3's internal
copy), for each policy, so the reported bucket-0 chrF is verified under the grader's scorer.
Also reports the chrF-macro variant for sensitivity.
Usage: python verify_canonical.py [--prefix runs/D3/d3]"""
import sys, os, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/docgap"))            # for `solution` package
from solution import chrf as CH                                 # official scorer

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default=os.path.expanduser("~/docgap/runs/D3/d3"))
a = ap.parse_args()
v0 = pd.read_csv(a.prefix + "_val0_rows.csv", keep_default_na=False)
refs0 = v0.ref.astype(str).tolist()


def pol(df, name, thr=-1.0):
    out = []
    for r in df.itertuples():
        c1 = str(r.c1_pick); ct5 = str(r.codet5); lp = float(r.codet5_logp); anc = int(r.anchor_strength)
        if name == "standalone":
            out.append(ct5 if ct5 else c1)
        elif name == "hybrid":
            out.append(str(r.hybrid_pick))
        elif name == "c1":
            out.append(c1)
        elif name == "logp_rescue":
            out.append(c1 if (ct5 == "" or lp < thr) else ct5)
    return out


print("[verify] official solution/chrf.py, bucket-0", flush=True)
for name in ("c1", "standalone", "hybrid"):
    p = pol(v0, name)
    pooled = CH.score_lists(p, refs0, mode="pooled")
    macro = CH.score_lists(p, refs0, mode="macro")
    print(f"  {name:11s}  pooled={pooled:.4f}  macro={macro:.4f}", flush=True)
# spot-check: pooled scorer parity on 200 rows vs a hand loop
sub = pol(v0, "hybrid")[:200]
manual = np.mean([CH.f_pooled(sub[i], refs0[i]) for i in range(200)])
print(f"  [parity] hybrid first-200 mean f_pooled = {manual:.4f}", flush=True)

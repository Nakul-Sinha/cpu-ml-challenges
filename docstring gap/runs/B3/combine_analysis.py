"""Complementarity of PROBE1 (LM bridge) and PROBE2 (neural) candidate pools on the
SAME bucket-0 holdout: combined oracle, exact-hit union, and rows each rescues."""
import sys
import pandas as pd
sys.path.insert(0, "solution")
from chrf import f_pooled

lm = pd.read_csv("runs/B3/lm_val.csv", keep_default_na=False)
nn = pd.read_csv("runs/B3/nn_val.csv", keep_default_na=False)
d = lm.merge(nn[["id", "nn_pred", "nn_conf", "nn_is_other", "nn_top10"]], on="id")
print("merged rows:", len(d))

def cands_lm(r):
    return [c for c in r["cands"].split("\t") if c]

def cands_nn(r):
    return [c for c in r["nn_top10"].split("\t") if c]

lm_argmax, nn_top1_real, lm_orc, nn_orc, comb_orc = [], [], [], [], []
lm_exact, nn_exact, comb_exact = 0, 0, 0
nn_rescue = 0   # NN commit beats LM argmax by >0.2 chrF
for _, r in d.iterrows():
    tgt = r["target_span"]
    lc = cands_lm(r); nc = cands_nn(r)
    fa = f_pooled(r["prediction"], tgt)
    lm_argmax.append(fa)
    lo = max((f_pooled(c, tgt) for c in lc), default=0.0)
    no = max((f_pooled(c, tgt) for c in nc), default=0.0)
    co = max(lo, no)
    lm_orc.append(lo); nn_orc.append(no); comb_orc.append(co)
    lm_exact += tgt in lc
    nn_exact += tgt in nc
    comb_exact += (tgt in lc) or (tgt in nc)
    # NN commit rescue: high-conf real-class prediction where LM argmax is weak
    if r["nn_is_other"] == 0 and r["nn_conf"] >= 0.5:
        fn = f_pooled(r["nn_pred"], tgt)
        if fn - fa > 0.2:
            nn_rescue += 1

N = len(d)
mean = lambda x: sum(x) / len(x)
print(f"LM argmax chrF        : {mean(lm_argmax):.4f}")
print(f"LM oracle@10          : {mean(lm_orc):.4f}   exact-hit {lm_exact/N:.3f}")
print(f"NN oracle@10          : {mean(nn_orc):.4f}   exact-hit {nn_exact/N:.3f}")
print(f"COMBINED oracle@20    : {mean(comb_orc):.4f}   exact-hit {comb_exact/N:.3f}")
print(f"  -> combined lifts LM oracle by {mean(comb_orc)-mean(lm_orc):+.4f}")
print(f"NN commit rescues (conf>=0.5 & +0.2 chrF over LM argmax): {nn_rescue} rows ({nn_rescue/N:.3f})")

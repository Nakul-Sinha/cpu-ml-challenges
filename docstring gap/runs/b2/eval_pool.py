"""Evaluate PoolBuilder: oracle chrF, exact-hit, per-source marginal lift,
pool size, phase timing extrapolated to 50k rows. Fit on buckets 1-19, eval
on bucket-0 sample."""
import sys, os, time, hashlib, argparse
import numpy as np
import pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "5")
sys.path.insert(0, "solution")
sys.path.insert(0, "runs/b2")
from chrf import f_pooled, score_lists
from pool_builder import PoolBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--nval", type=int, default=5000)
ap.add_argument("--cap", type=int, default=80)
args = ap.parse_args()

t0 = time.time()
train = pd.read_csv("dataset/train.csv", keep_default_na=False)
def bucket(s): return int(hashlib.md5(s.encode("utf-8","ignore")).hexdigest()[:8],16)%20
b = train.masked_docstring.map(bucket)
trn = train[b != 0].reset_index(drop=True)
val_all = train[b == 0].reset_index(drop=True)
val = val_all.sample(min(args.nval, len(val_all)), random_state=1).reset_index(drop=True)
print(f"trn {len(trn)} val {len(val)} (bucket0 total {len(val_all)}) load {time.time()-t0:.1f}s", flush=True)

pb = PoolBuilder(cap=1000)  # uncapped; cap applied in eval per-ablation
tf = time.time()
pb.fit(trn)
print(f"FIT {time.time()-tf:.1f}s (anchor {pb.t_anchor_fit:.1f} global {pb.t_global_fit:.1f} fuzz {pb.t_fuzz_fit:.1f})", flush=True)

refs = val.target_span.astype(str).values

# ---- build production pools (gated fuzz), uncapped, timed by phase ----
t = time.time()
pools_nf = pb.candidates_batch(val, use_fuzz=False)  # anchored+code+global only
t_af = time.time() - t
t = time.time()
pools = pb.candidates_batch(val, use_fuzz=True)       # + gated fuzz
t_full = time.time() - t
t_fuzz = t_full - t_af
# gate fraction
gate_frac = np.mean([any(s == "fuzz" for _, s, _ in row) for row in pools])
print(f"\nTIMING on {len(val)} rows: anchored+code+global {t_af:.1f}s | +fuzz(gated {gate_frac:.0%}) {t_fuzz:.1f}s", flush=True)
sc = 50000 / len(val)
print(f"  -> extrap 50k rows single-thread: anc+code+glob {t_af*sc:.0f}s | fuzz {t_fuzz*sc:.0f}s | TOTAL {t_full*sc:.0f}s", flush=True)

def oracle_of(pools_cap):
    orc = np.empty(len(pools_cap)); hit = 0
    for i, row in enumerate(pools_cap):
        cs = [t for t, _, _ in row]
        best = 0.0
        for c in cs:
            f = f_pooled(c, refs[i])
            if f > best: best = f
        orc[i] = best
        hit += refs[i] in cs
    return orc.mean(), hit / len(pools_cap)

def cap_pools(pools, cap, drop_src=None):
    out = []
    for row in pools:
        r = [(t, s, sc) for t, s, sc in row if (drop_src is None or s not in drop_src)]
        r.sort(key=lambda x: -x[2])
        out.append(r[:cap])
    return out

# ---- headline: full pool oracle at cap ----
capped = cap_pools(pools, args.cap)
orc, hit = oracle_of(capped)
sizes = [len(r) for r in capped]
print(f"\n=== HEADLINE (cap {args.cap}) ===", flush=True)
print(f"pool oracle chrF: {orc:.4f}   exact-hit: {hit:.3f}   mean pool {np.mean(sizes):.1f} (max {max(sizes)})", flush=True)

# ---- per-source marginal: oracle if we DROP each source family ----
fams = {
    "anchor(all)": {"l2r2","l1r1","skipR","skipL","r1","l1"},
    "fuzz(full)": {"fuzz"}, "fuzz(window)": {"fuzzw"},
    "fuzz(both)": {"fuzz","fuzzw"}, "code": {"code","codeP"}, "global": {"global"},
    " l2r2": {"l2r2"}, " l1r1": {"l1r1"}, " skipR/L": {"skipR","skipL"},
    " r1": {"r1"}, " l1": {"l1"},
}
print("\n=== PER-SOURCE MARGINAL (oracle drop if removed) ===", flush=True)
for name, drop in fams.items():
    o2, _ = oracle_of(cap_pools(pools, args.cap, drop_src=drop))
    print(f"  drop {name:14s}: oracle {o2:.4f}   marginal {orc-o2:+.4f}", flush=True)

# ---- per-source STANDALONE oracle (that source only) ----
print("\n=== STANDALONE (this source only, capped) ===", flush=True)
allsrc = {"l2r2","l1r1","skipR","skipL","r1","l1","fuzz","fuzzw","code","codeP","global"}
for name, keep in [("anchor", {"l2r2","l1r1","skipR","skipL","r1","l1"}),
                   ("fuzz", {"fuzz","fuzzw"}), ("code", {"code","codeP"}), ("global", {"global"})]:
    o2, h2 = oracle_of(cap_pools(pools, args.cap, drop_src=allsrc - keep))
    print(f"  {name:10s} only: oracle {o2:.4f}  exact-hit {h2:.3f}", flush=True)

# ---- GATED variant (runtime-safe fallback: fuzz only on weak-anchor rows) ----
pb.gate_fuzz = True
tg = time.time()
pools_g = pb.candidates_batch(val, use_fuzz=True)
t_g = time.time() - tg
gfrac = np.mean([any(s in ("fuzz","fuzzw") for _, s, _ in row) for row in pools_g])
o_g, h_g = oracle_of(cap_pools(pools_g, args.cap))
print(f"\n=== GATED FUZZ (weak-anchor rows only, {gfrac:.0%}) ===", flush=True)
print(f"oracle {o_g:.4f} exact-hit {h_g:.3f}  (vs ungated {orc:.4f})  time {t_g:.1f}s -> 50k {t_g*sc:.0f}s", flush=True)

# ---- save deliverable: val pools (ungated) + stats ----
import pickle, json
os.makedirs("runs/b2/out", exist_ok=True)
with open("runs/b2/out/val_pools.pkl", "wb") as f:
    pickle.dump({"ids": val.id.tolist(), "refs": refs.tolist(),
                 "pools": capped}, f)
stats = {"n_val": len(val), "cap": args.cap,
         "oracle_ungated": float(orc), "exact_hit": float(hit),
         "oracle_gated": float(o_g), "mean_pool": float(np.mean(sizes)),
         "fit_s": pb.t_fit, "extrap50k_ungated_s": float(t_full*sc),
         "extrap50k_gated_s": float(t_g*sc)}
with open("runs/b2/out/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
# naive top-1 prediction from pool (recall artifact, not the reranker's job)
top1 = [row[0][0] if row else "" for row in capped]
print(f"naive pool-top1 chrF: {score_lists(top1, refs.tolist()):.4f}", flush=True)
print("saved runs/b2/out/{val_pools.pkl,stats.json}", flush=True)
print(f"\nTOTAL {time.time()-t0:.1f}s", flush=True)

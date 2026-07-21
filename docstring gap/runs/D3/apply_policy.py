"""D3: apply a chosen selection policy to the per-row dumps from d3_eval and score with the
canonical scorer. Reports bucket-0 chrF, bucket-1 chrF (LOCKED, scored ONCE here), and the
bucket0-minus-bucket1 optimism gap. Optionally writes the bucket-1 prediction CSV.

Policies:
  standalone           -> codet5
  hybrid               -> hybrid_pick (learned reranker argmax over pool + codet5)
  logp_rescue   --thr T-> codet5 unless codet5_logp < T -> c1_pick
  anchor_rescue --thr T-> codet5 unless (codet5_logp < T AND anchor_strength>=1) -> c1_pick
Empty codet5 always falls back to c1_pick.
"""
import sys, os, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "D1"))
from solution_v3 import f_pooled  # canonical scorer


def apply_policy(df, policy, thr):
    preds = []
    for r in df.itertuples():
        c1 = str(r.c1_pick); ct5 = str(r.codet5); lp = float(r.codet5_logp)
        anc = int(r.anchor_strength)
        if ct5 == "":
            preds.append(c1); continue
        if policy == "standalone":
            preds.append(ct5)
        elif policy == "hybrid":
            preds.append(str(r.hybrid_pick))
        elif policy == "logp_rescue":
            preds.append(c1 if lp < thr else ct5)
        elif policy == "anchor_rescue":
            preds.append(c1 if (lp < thr and anc >= 1) else ct5)
        else:
            raise ValueError(policy)
    return preds


def chrf(preds, refs):
    return float(np.mean([f_pooled(p, r) for p, r in zip(preds, refs)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "d3"))
    ap.add_argument("--policy", required=True)
    ap.add_argument("--thr", type=float, default=-0.7)
    ap.add_argument("--write_bucket1", default="")
    a = ap.parse_args()
    v0 = pd.read_csv(a.prefix + "_val0_rows.csv", keep_default_na=False)
    v1 = pd.read_csv(a.prefix + "_val1_rows.csv", keep_default_na=False)
    refs0 = v0.ref.astype(str).tolist(); refs1 = v1.ref.astype(str).tolist()

    p0 = apply_policy(v0, a.policy, a.thr); c0 = chrf(p0, refs0)
    p1 = apply_policy(v1, a.policy, a.thr); c1 = chrf(p1, refs1)
    print(f"[policy={a.policy} thr={a.thr}]", flush=True)
    print(f"  bucket-0 chrF = {c0:.4f}  (n={len(v0)})", flush=True)
    print(f"  bucket-1 chrF = {c1:.4f}  (n={len(v1)})  [LOCKED - scored once]", flush=True)
    print(f"  optimism gap (b0 - b1) = {c0 - c1:+.4f}", flush=True)
    # sanity references
    for pol, th in [("standalone", 0), ("hybrid", 0)]:
        q0 = chrf(apply_policy(v0, pol, th), refs0)
        q1 = chrf(apply_policy(v1, pol, th), refs1)
        print(f"  [ref {pol:11s}] b0={q0:.4f} b1={q1:.4f} gap={q0-q1:+.4f}", flush=True)
    if a.write_bucket1:
        pd.DataFrame({"id": v1.id.values, "prediction": p1}).to_csv(a.write_bucket1, index=False)
        empty = int((pd.Series(p1).astype(str).str.len() == 0).sum())
        print(f"  [save] {a.write_bucket1} ({len(p1)} rows, {empty} empty)", flush=True)


if __name__ == "__main__":
    main()

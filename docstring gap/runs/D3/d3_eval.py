"""D3 codet5 policy eval on bucket-0 (fit 2-19). Measures, in ONE run, every ship-candidate
policy so we can pick the best, using solution_v3's EXACT building blocks (imported).

Policies measured on bucket-0 (val0):
  P0  C1-only reranker argmax                       (retrieval chassis, ~0.319)
  P1  codet5 zero-shot standalone                   (generator alone, ~0.41)
  P2  learned hybrid reranker (codet5 injected cov=1.0, 5 T5 feats), argmax
  P3  learned hybrid reranker + confident-codet5 override (logp>=thr)
  P4  codet5-primary + retrieval rescue: codet5 unless logp<thr -> C1 pick (sweep thr)
  P5  codet5-primary + rescue only where retrieval strong: codet5 unless (logp<thr AND
      anchor_strength>=1) -> C1 pick (sweep thr)
  ORA two-way oracle max(C1 pick, codet5)           (selection ceiling)

Caches codet5 + C1-pick + hybrid-pick on val1 (LOCKED) and dumps both val0/val1 per-row so the
final bucket-1 prediction for the CHOSEN policy is produced offline (scored once, separately).
"""
import sys, os, time, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "D1"))
import solution_v3 as S  # noqa: E402


def pool_oracle(texts, refs):
    return float(np.mean([max((S.f_pooled(c, r) for c in cs), default=0.0)
                          for cs, r in zip(texts, refs)]))


def tier_of(masked, pb):
    a = S.anchors(masked)
    if a["l2r2"] in pb.idx["l2r2"]:
        return "l2r2"
    if a["l1r1"] in pb.idx["l1r1"]:
        return "l1r1"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=40000)
    ap.add_argument("--nval", type=int, default=0, help=">0 subsamples bucket-0 (smoke)")
    ap.add_argument("--nval1", type=int, default=0, help=">0 subsamples bucket-1")
    ap.add_argument("--threads", type=int, default=S.NUM_THREADS)
    ap.add_argument("--train_t5_cap", type=int, default=16000)
    ap.add_argument("--out_prefix", default=os.path.join(HERE, "d3"))
    a = ap.parse_args()
    t0 = time.time()

    def el():
        return (time.time() - t0) / 60.0

    d = pd.read_csv("dataset/train.csv", keep_default_na=False)
    d["_bkt"] = d.masked_docstring.map(S.bucket)
    train_fit = d[d._bkt >= 2].reset_index(drop=True)
    val0 = d[d._bkt == 0].reset_index(drop=True)
    val1 = d[d._bkt == 1].reset_index(drop=True)
    if a.nval > 0:
        val0 = val0.sample(a.nval, random_state=3).reset_index(drop=True)
    if a.nval1 > 0:
        val1 = val1.sample(a.nval1, random_state=3).reset_index(drop=True)
    refs0 = val0.target_span.astype(str).values.tolist()
    print(f"[data] train_fit(2-19)={len(train_fit)} val0={len(val0)} val1={len(val1)} "
          f"threads={a.threads} model={S.T5_MODEL_NAME} zeroshot={S.T5_ZEROSHOT}", flush=True)

    # ---------------- C1 base (all fork stages first, before torch) ----------------
    even = train_fit[train_fit._bkt % 2 == 0]; odd = train_fit[train_fit._bkt % 2 == 1]
    fits_even = S.build_fits(even, a.threads, "even")
    fits_odd = S.build_fits(odd, a.threads, "odd")
    samp = train_fit.sample(n=min(a.ntrain, len(train_fit)), random_state=7).reset_index(drop=True)
    (Xtr_b, ytr_b, gtr, txtr, masked_tr, code_tr,
     unions_tr, astr_tr, tag_tr, tgt_tr) = S.build_training_base(samp, fits_even, fits_odd, a.threads)
    fits_full = S.build_fits(train_fit, a.threads, "full")
    astr0, unions0 = S.build_union(val0, fits_full, a.threads)
    Xv_b, _, gv, txv = S.featurize_rows(val0, fits_full, astr0, unions0, a.threads, want_labels=False)
    masked0 = val0.masked_docstring.values.tolist(); code0 = val0.code_context.values.tolist()
    astr1, unions1 = S.build_union(val1, fits_full, a.threads)
    Xv1_b, _, gv1, txv1 = S.featurize_rows(val1, fits_full, astr1, unions1, a.threads, want_labels=False)
    masked1 = val1.masked_docstring.values.tolist(); code1 = val1.code_context.values.tolist()
    print(f"[c1-base] Xtr{Xtr_b.shape} Xv{Xv_b.shape} Xv1{Xv1_b.shape} all-fork-done @ {el():.1f}min", flush=True)

    # C1-only reranker (no codet5) -> C1 pick on val0 and val1
    base_boost = S.train_reranker(Xtr_b, ytr_b, gtr, a.threads, S.FEAT_NAMES)
    c1_preds0 = S.decode(base_boost, Xv_b, gv, txv)
    c1_preds1 = S.decode(base_boost, Xv1_b, gv1, txv1)
    c1_chrf = S.score_lists(c1_preds0, refs0)
    c1_f0 = np.array([S.f_pooled(p, r) for p, r in zip(c1_preds0, refs0)])
    print(f"[P0 c1-only] bucket-0 chrF={c1_chrf:.4f}  pool_oracle={pool_oracle(txv, refs0):.4f}", flush=True)

    # ---------------- codet5 (main process, after fork) ZERO-SHOT ----------------
    t5 = S.T5Gen(a.threads).load_zeroshot()
    if S.T5_USE_QUANT:
        t5.quantize()
    tgen = time.time()
    t5p0, t5l0 = t5.generate(masked0, code0)
    t5p1, t5l1 = t5.generate(masked1, code1)
    print(f"[codet5-gen] val0+val1 done @ {el():.1f}min ({time.time()-tgen:.0f}s)", flush=True)
    # train codet5 (capped) for the learned hybrid reranker
    gidx_tr = np.arange(len(masked_tr))
    if len(gidx_tr) > a.train_t5_cap:
        rng = np.random.RandomState(5)
        gidx_tr = np.sort(rng.choice(gidx_tr, a.train_t5_cap, replace=False))
    t5p_tr = [None] * len(masked_tr); t5l_tr = [0.0] * len(masked_tr)
    pr, lp = t5.generate([masked_tr[i] for i in gidx_tr], [code_tr[i] for i in gidx_tr])
    for k, i in enumerate(gidx_tr):
        t5p_tr[i] = pr[k]; t5l_tr[i] = lp[k]
    print(f"[codet5-train] gated={len(gidx_tr)} done @ {el():.1f}min", flush=True)

    p1_f0 = np.array([S.f_pooled(t5p0[i], refs0[i]) for i in range(len(val0))])
    print(f"[P1 codet5-standalone] bucket-0 chrF={p1_f0.mean():.4f} "
          f"exact={np.mean([t5p0[i]==refs0[i] for i in range(len(val0))]):.4f} "
          f"empty={np.mean([str(t5p0[i])=='' for i in range(len(val0))]):.4f}", flush=True)

    tiers0 = np.array([tier_of(m, fits_full.pb) for m in masked0])
    astr0_arr = np.array(astr0)
    l0 = np.array(t5l0)

    # ---------------- P2/P3 learned hybrid reranker (coverage 1.0) ----------------
    gated_tr = np.zeros(len(masked_tr), dtype=bool); gated_tr[gidx_tr] = True
    fits_list_tr = [(fits_odd if tag_tr[i] == 0 else fits_even) for i in range(len(masked_tr))]
    Xtr_a, gtr_a, txtr_a, ytr_a, _ = S.augment_with_t5(
        Xtr_b, gtr, txtr, unions_tr, masked_tr, code_tr, astr_tr, gated_tr,
        t5p_tr, t5l_tr, fits_list_tr, ybase=ytr_b, tgts=tgt_tr)
    hyb_boost = S.train_reranker(Xtr_a, ytr_a, gtr_a, a.threads, S.FEAT_NAMES3)
    imp = sorted(zip(S.FEAT_NAMES3, hyb_boost.feature_importance("gain")), key=lambda x: -x[1])
    print(f"[hybrid reranker] iters {hyb_boost.best_iteration} top12 {[(n,int(g)) for n,g in imp[:12]]}", flush=True)

    gated0 = np.ones(len(val0), dtype=bool)
    fits_list_te = [fits_full] * len(val0)
    Xv_a, gv_a, txv_a, _, row_t5_0 = S.augment_with_t5(
        Xv_b, gv, txv, unions0, masked0, code0, astr0, gated0, t5p0, t5l0, fits_list_te)
    assert Xtr_a.shape[1] == Xv_a.shape[1] == S.N_FEAT3
    hyb_preds0 = S.decode(hyb_boost, Xv_a, gv_a, txv_a)
    p2 = S.score_lists(hyb_preds0, refs0)
    hyb_f0 = np.array([S.f_pooled(p, r) for p, r in zip(hyb_preds0, refs0)])
    print(f"[P2 learned-hybrid argmax] bucket-0 chrF={p2:.4f}  ({p2-c1_chrf:+.4f} vs C1, "
          f"{p2-p1_f0.mean():+.4f} vs codet5)  oracle={pool_oracle(txv_a, refs0):.4f}", flush=True)
    for thr in (-0.35, -0.5, -0.7):
        hp = S.decode(hyb_boost, Xv_a, gv_a, txv_a, row_t5=row_t5_0, override_thr=thr)
        print(f"[P3 hybrid+override thr={thr}] chrF={S.score_lists(hp, refs0):.4f}", flush=True)

    # ---------------- P4/P5 codet5-primary + retrieval rescue ----------------
    def blend(thr, require_anchor):
        out = []
        for i in range(len(val0)):
            use_c1 = (l0[i] < thr) and ((astr0_arr[i] >= 1) if require_anchor else True)
            out.append(c1_preds0[i] if use_c1 else (t5p0[i] if str(t5p0[i]) != "" else c1_preds0[i]))
        return out
    print("[P4 codet5-primary + logp rescue -> C1]", flush=True)
    best_p4 = (-1, None)
    for thr in (-1.5, -1.2, -1.0, -0.85, -0.7, -0.55, -0.4):
        pv = blend(thr, False); sc = S.score_lists(pv, refs0)
        cov_c1 = np.mean([l0[i] < thr for i in range(len(val0))])
        print(f"    thr={thr:+.2f}  chrF={sc:.4f}  c1_used={cov_c1:.3f}", flush=True)
        if sc > best_p4[0]:
            best_p4 = (sc, thr)
    print("[P5 codet5-primary + (logp AND anchor>=1) rescue -> C1]", flush=True)
    best_p5 = (-1, None)
    for thr in (-1.2, -1.0, -0.85, -0.7, -0.55, -0.4, -0.25):
        pv = blend(thr, True); sc = S.score_lists(pv, refs0)
        cov_c1 = np.mean([(l0[i] < thr and astr0_arr[i] >= 1) for i in range(len(val0))])
        print(f"    thr={thr:+.2f}  chrF={sc:.4f}  c1_used={cov_c1:.3f}", flush=True)
        if sc > best_p5[0]:
            best_p5 = (sc, thr)

    # ---------------- oracle ceiling ----------------
    ora = np.mean([max(c1_f0[i], p1_f0[i]) for i in range(len(val0))])
    print(f"[ORA two-way max(C1,codet5)] {ora:.4f}", flush=True)

    # per-tier: codet5 vs C1
    print("[tiers] tier  n     C1      codet5  (codet5-C1)", flush=True)
    for t in ("l2r2", "l1r1", "none"):
        m = tiers0 == t
        if m.sum():
            print(f"    {t:5s} {int(m.sum()):5d}  {c1_f0[m].mean():.4f}  {p1_f0[m].mean():.4f}  "
                  f"({p1_f0[m].mean()-c1_f0[m].mean():+.4f})", flush=True)

    # ---------------- dump per-row val0 + val1 for offline policy application ----------------
    hyb_preds1 = None
    gated1 = np.ones(len(val1), dtype=bool)
    Xv1_a, gv1_a, txv1_a, _, row_t5_1 = S.augment_with_t5(
        Xv1_b, gv1, txv1, unions1, masked1, code1, astr1, gated1, t5p1, t5l1, [fits_full] * len(val1))
    assert Xv1_a.shape[1] == S.N_FEAT3
    hyb_preds1 = S.decode(hyb_boost, Xv1_a, gv1_a, txv1_a)
    tiers1 = [tier_of(m, fits_full.pb) for m in masked1]

    pd.DataFrame({"id": val0.id.values, "ref": refs0, "c1_pick": c1_preds0, "codet5": t5p0,
                  "codet5_logp": t5l0, "hybrid_pick": hyb_preds0, "anchor_strength": astr0_arr,
                  "tier": tiers0}).to_csv(a.out_prefix + "_val0_rows.csv", index=False)
    pd.DataFrame({"id": val1.id.values, "ref": val1.target_span.astype(str).values,
                  "c1_pick": c1_preds1, "codet5": t5p1, "codet5_logp": t5l1,
                  "hybrid_pick": hyb_preds1, "anchor_strength": np.array(astr1),
                  "tier": tiers1}).to_csv(a.out_prefix + "_val1_rows.csv", index=False)
    print(f"[save] {a.out_prefix}_val0_rows.csv ({len(val0)}) + _val1_rows.csv ({len(val1)})", flush=True)

    print(f"\n#### SUMMARY @ {el():.1f}min ####", flush=True)
    print(f"  P0 C1-only            {c1_chrf:.4f}", flush=True)
    print(f"  P1 codet5-standalone  {p1_f0.mean():.4f}", flush=True)
    print(f"  P2 learned-hybrid     {p2:.4f}", flush=True)
    print(f"  P4 codet5+logp-rescue {best_p4[0]:.4f} (thr={best_p4[1]})", flush=True)
    print(f"  P5 codet5+anchor-resc {best_p5[0]:.4f} (thr={best_p5[1]})", flush=True)
    print(f"  ORA max(C1,codet5)    {ora:.4f}", flush=True)
    print(f"[done] total {el():.1f}min", flush=True)


if __name__ == "__main__":
    try:
        import multiprocessing
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()

"""D1 hybrid bucket-0 parity eval.

Uses the EXACT building blocks from solution_v3 (imported, not reimplemented) so the
eval measures the shipped artifact. Fits everything on buckets 2-19 (bucket 0 = held-out
validation, bucket 1 = LOCKED and never scored here). T5 is fine-tuned on buckets 2-19.

Coverage is the ONLY thing swept: {0.30, 0.45, 0.60}. Everything else (FT budget, decode
rule, seq_logprob override threshold) is fixed per the review spec.

Reports per coverage: bucket-0 chrF (argmax / argmax+override), realized-vs-oracle,
T5-candidate pick-rate + win-rate, per-gate-tier gains, exact-hit. Picks the best coverage,
writes runs/D1/val_pred_v3.csv (bucket-0) and, with the SAME artifacts at that coverage,
runs/D1/val_pred_v3_bucket1.csv (bucket-1, unscored).

Usage:
  python runs/D1/hybrid_eval.py --ft_s 960                # 16-min in-script FT (faithful)
  python runs/D1/hybrid_eval.py --ckpt runs/C2/ft_hint    # DEV: reuse a 2-19 checkpoint
  python runs/D1/hybrid_eval.py --ft_s 120 --nval 1500 --ntrain 8000   # quick plumbing smoke
"""
import sys, os, time, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solution_v3 as S  # noqa: E402


def eval_pool_oracle(texts, refs):
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
    ap.add_argument("--ft_s", type=int, default=960, help="in-script FT budget seconds")
    ap.add_argument("--ckpt", default=None, help="DEV: reuse a fine-tuned 2-19 checkpoint")
    ap.add_argument("--ntrain", type=int, default=40000)
    ap.add_argument("--nval", type=int, default=0, help=">0 subsamples bucket-0 (smoke)")
    ap.add_argument("--threads", type=int, default=S.NUM_THREADS)
    ap.add_argument("--coverages", default="0.30,0.45,0.60")
    ap.add_argument("--out_prefix", default=os.path.join(HERE, "val_pred_v3"))
    a = ap.parse_args()
    covs = [float(x) for x in a.coverages.split(",")]
    thr_ov = S.T5_OVERRIDE_THR
    t0 = time.time()

    def el():
        return (time.time() - t0) / 60.0

    d = pd.read_csv("dataset/train.csv", keep_default_na=False)
    d["_bkt"] = d.masked_docstring.map(S.bucket)
    train_fit = d[d._bkt >= 2].reset_index(drop=True)          # buckets 2-19
    val0 = d[d._bkt == 0].reset_index(drop=True)               # held-out validation
    val1 = d[d._bkt == 1].reset_index(drop=True)               # LOCKED (unscored)
    if a.nval > 0:
        val0 = val0.sample(a.nval, random_state=3).reset_index(drop=True)
    refs0 = val0.target_span.astype(str).values.tolist()
    print(f"[data] train_fit(2-19)={len(train_fit)} val0={len(val0)} val1={len(val1)} "
          f"threads={a.threads} ft_s={a.ft_s} ckpt={a.ckpt}", flush=True)

    # ---------------- C1 base (all fork stages first) ----------------
    even = train_fit[train_fit._bkt % 2 == 0]; odd = train_fit[train_fit._bkt % 2 == 1]
    fits_even = S.build_fits(even, a.threads, "even")
    fits_odd = S.build_fits(odd, a.threads, "odd")
    samp = train_fit.sample(n=min(a.ntrain, len(train_fit)), random_state=7).reset_index(drop=True)
    (Xtr_b, ytr_b, gtr, txtr, masked_tr, code_tr,
     unions_tr, astr_tr, tag_tr, tgt_tr) = S.build_training_base(samp, fits_even, fits_odd, a.threads)
    fits_full = S.build_fits(train_fit, a.threads, "full")
    astr0, unions0 = S.build_union(val0, fits_full, a.threads)
    Xv_b, _, gv, txv = S.featurize_rows(val0, fits_full, astr0, unions0, a.threads, want_labels=False)
    masked0 = val0.masked_docstring.values.tolist()
    code0 = val0.code_context.values.tolist()
    # bucket-1 base features MUST be built here, before torch loads (fork stages cannot
    # run after torch is initialised in-process).
    astr1, unions1 = S.build_union(val1, fits_full, a.threads)
    Xv1_b, _, gv1, txv1 = S.featurize_rows(val1, fits_full, astr1, unions1, a.threads, want_labels=False)
    masked1 = val1.masked_docstring.values.tolist(); code1 = val1.code_context.values.tolist()
    print(f"[c1-base] Xtr{Xtr_b.shape} Xv{Xv_b.shape} Xv1{Xv1_b.shape} all-fork-done @ {el():.1f}min", flush=True)

    # C1-only reference reranker + decode (coverage-independent)
    base_boost = S.train_reranker(Xtr_b, ytr_b, gtr, a.threads, S.FEAT_NAMES)
    c1_preds = S.decode(base_boost, Xv_b, gv, txv)
    c1_chrf = S.score_lists(c1_preds, refs0)
    c1_f = np.array([S.f_pooled(p, r) for p, r in zip(c1_preds, refs0)])
    print(f"[c1-only] bucket-0 chrF={c1_chrf:.4f}  pool_oracle={eval_pool_oracle(txv, refs0):.4f}", flush=True)

    # ---------------- T5 (main process, after fork) ----------------
    t5 = S.T5Gen(a.threads)
    if a.ckpt:
        t5.load_checkpoint(a.ckpt); print(f"[t5] loaded checkpoint {a.ckpt} @ {el():.1f}min", flush=True)
    else:
        tft = time.time()
        t5.finetune(train_fit, a.ft_s)
        print(f"[t5] FT {(time.time()-tft)/60:.1f}min @ {el():.1f}min", flush=True)
    if S.T5_USE_QUANT:
        t5.quantize()

    # parity weakness of the reranker-train rows: a train row scored against its opposite-parity
    # fit does not see itself, mirroring the unseen condition of val rows. Thresholds are
    # calibrated on THIS distribution (not full-fit train, which is self-leaked and over-gates).
    tw = np.array([S.t5_weakness_score(masked_tr[i], (fits_odd.pb if tag_tr[i] == 0 else fits_even.pb))
                   for i in range(len(masked_tr))])
    cov_max = max(covs)
    # cache T5 on val0: run on rows gated at the LARGEST coverage (superset of all sweeps)
    thr_max = float(np.quantile(tw, 1.0 - cov_max))
    gated0_max = S.gate_mask(masked0, fits_full.pb, thr_max)
    idx0 = np.where(gated0_max)[0]
    t5p0 = [None] * len(val0); t5l0 = [0.0] * len(val0)
    if len(idx0):
        pr, lp = t5.generate([masked0[i] for i in idx0], [code0[i] for i in idx0])
        for k, i in enumerate(idx0):
            t5p0[i] = pr[k]; t5l0[i] = lp[k]
    print(f"[t5-cache] val0 gated@{cov_max}={len(idx0)} done @ {el():.1f}min", flush=True)

    # cache T5 on train sample: rows gated at largest coverage (parity weakness), capped
    gm_tr_max = tw >= thr_max
    gidx_tr = np.where(gm_tr_max)[0]
    if len(gidx_tr) > S.TRAIN_T5_CAP:
        rng = np.random.RandomState(5)
        gidx_tr = np.sort(rng.choice(gidx_tr, S.TRAIN_T5_CAP, replace=False))
    t5p_tr = [None] * len(masked_tr); t5l_tr = [0.0] * len(masked_tr)
    if len(gidx_tr):
        pr, lp = t5.generate([masked_tr[i] for i in gidx_tr], [code_tr[i] for i in gidx_tr])
        for k, i in enumerate(gidx_tr):
            t5p_tr[i] = pr[k]; t5l_tr[i] = lp[k]
    print(f"[t5-cache] train gated@{cov_max}(cap)={len(gidx_tr)} done @ {el():.1f}min", flush=True)

    tiers0 = np.array([tier_of(m, fits_full.pb) for m in masked0])
    fits_list_te = [fits_full] * len(val0)
    fits_list_tr = [(fits_odd if tag_tr[i] == 0 else fits_even) for i in range(len(masked_tr))]

    results = []
    for cov in covs:
        thr = float(np.quantile(tw, 1.0 - cov))  # train-parity calibrated (unseen-like)
        # gate subsets of the cached superset
        gated0 = S.gate_mask(masked0, fits_full.pb, thr)
        gated_tr = (tw >= thr) & gm_tr_max  # only rows we have cached T5 for
        # train augment + reranker
        Xtr_a, gtr_a, txtr_a, ytr_a, _ = S.augment_with_t5(
            Xtr_b, gtr, txtr, unions_tr, masked_tr, code_tr, astr_tr, gated_tr,
            t5p_tr, t5l_tr, fits_list_tr, ybase=ytr_b, tgts=tgt_tr)
        boost = S.train_reranker(Xtr_a, ytr_a, gtr_a, a.threads, S.FEAT_NAMES3)
        # val0 augment + decode
        Xv_a, gv_a, txv_a, _, row_t5_0 = S.augment_with_t5(
            Xv_b, gv, txv, unions0, masked0, code0, astr0, gated0,
            t5p0, t5l0, fits_list_te)
        assert Xtr_a.shape[1] == Xv_a.shape[1] == S.N_FEAT3
        preds_arg = S.decode(boost, Xv_a, gv_a, txv_a)
        preds_ovr = S.decode(boost, Xv_a, gv_a, txv_a, row_t5=row_t5_0, override_thr=thr_ov)
        chrf_arg = S.score_lists(preds_arg, refs0)
        chrf_ovr = S.score_lists(preds_ovr, refs0)
        oracle = eval_pool_oracle(txv_a, refs0)
        # diagnostics on gated rows
        f_arg = np.array([S.f_pooled(p, r) for p, r in zip(preds_arg, refs0)])
        f_ovr = np.array([S.f_pooled(p, r) for p, r in zip(preds_ovr, refs0)])
        gm = gated0
        n_g = int(gm.sum())
        t5_pick = np.array([(row_t5_0[i] is not None and preds_arg[i] == row_t5_0[i][0]) for i in range(len(val0))])
        t5_f = np.array([(S.f_pooled(row_t5_0[i][0], refs0[i]) if row_t5_0[i] is not None else 0.0)
                         for i in range(len(val0))])
        pickrate = float(t5_pick[gm].mean()) if n_g else 0.0
        winrate = float((t5_f[gm] > c1_f[gm] + 1e-9).mean()) if n_g else 0.0
        exact = float(np.mean([p == r for p, r in zip(preds_arg, refs0)]))
        ship = preds_arg; ship_f = f_arg  # shipped = pure argmax (override measured to hurt; see below)
        # per-tier gain (shipped - c1)
        tier_rows = []
        for t in ("l2r2", "l1r1", "none"):
            mtier = tiers0 == t
            if mtier.sum():
                tier_rows.append((t, int(mtier.sum()), float(c1_f[mtier].mean()),
                                  float(ship_f[mtier].mean()), float(gated0[mtier].mean())))
        results.append(dict(cov=cov, thr=thr, realized_cov=float(gated0.mean()), n_gated=n_g,
                            chrf_arg=chrf_arg, chrf_ovr=chrf_ovr, oracle=oracle,
                            pickrate=pickrate, winrate=winrate, exact=exact,
                            gain_arg=chrf_arg - c1_chrf, gain_ovr=chrf_ovr - c1_chrf,
                            tiers=tier_rows,
                            _preds=ship, _boost=boost, _thr=thr))
        print(f"\n==== coverage {cov} (thr={thr:.3f} realized_cov={gated0.mean():.3f} n_gated={n_g}) ====", flush=True)
        print(f"   C1-only            chrF={c1_chrf:.4f}", flush=True)
        print(f"   hybrid argmax      chrF={chrf_arg:.4f}  ({chrf_arg-c1_chrf:+.4f})   [SHIPPED]", flush=True)
        print(f"   hybrid +override   chrF={chrf_ovr:.4f}  ({chrf_ovr-c1_chrf:+.4f})   (override rejected)", flush=True)
        print(f"   pool oracle        {oracle:.4f}   realized/oracle={chrf_arg/oracle:.3f}", flush=True)
        print(f"   T5 pick-rate(gated)={pickrate:.3f}  win-rate(gated)={winrate:.3f}  exact-hit={exact:.4f}", flush=True)
        for t, n, cf, hf, gc in tier_rows:
            print(f"     tier {t:5s} n={n:5d} gated_frac={gc:.2f}  C1={cf:.4f} -> hybrid={hf:.4f} ({hf-cf:+.4f})", flush=True)

    # pick best coverage by shipped (pure-argmax) chrF
    best = max(results, key=lambda r: r["chrf_arg"])
    print(f"\n#### BEST coverage={best['cov']} shipped(argmax) chrF={best['chrf_arg']:.4f} "
          f"(C1 {c1_chrf:.4f}, +{best['chrf_arg']-c1_chrf:.4f}) @ {el():.1f}min ####", flush=True)

    # save bucket-0 predictions at best coverage
    pd.DataFrame({"id": val0.id.values, "prediction": best["_preds"]}).to_csv(a.out_prefix + ".csv", index=False)
    print(f"[save] {a.out_prefix}.csv ({len(val0)} rows)", flush=True)

    # ---------------- bucket-1 (LOCKED) prediction with SAME artifacts at best coverage ----------------
    # (Xv1_b was built up-front, before torch; only T5 inference + augment + decode remain.)
    thrB = best["_thr"]; boostB = best["_boost"]
    gated1 = S.gate_mask(masked1, fits_full.pb, thrB)
    i1 = np.where(gated1)[0]
    t5p1 = [None] * len(val1); t5l1 = [0.0] * len(val1)
    if len(i1):
        pr, lp = t5.generate([masked1[i] for i in i1], [code1[i] for i in i1])
        for k, i in enumerate(i1):
            t5p1[i] = pr[k]; t5l1[i] = lp[k]
    Xv1_a, gv1_a, txv1_a, _, row_t5_1 = S.augment_with_t5(
        Xv1_b, gv1, txv1, unions1, masked1, code1, astr1, gated1,
        t5p1, t5l1, [fits_full] * len(val1))
    assert Xv1_a.shape[1] == S.N_FEAT3
    preds1 = S.decode(boostB, Xv1_a, gv1_a, txv1_a)  # pure argmax (shipped decode)
    pd.DataFrame({"id": val1.id.values, "prediction": preds1}).to_csv(a.out_prefix + "_bucket1.csv", index=False)
    print(f"[save] {a.out_prefix}_bucket1.csv ({len(val1)} rows) -- UNSCORED (reviewer scores once)", flush=True)
    print(f"[done] total {el():.1f}min", flush=True)


if __name__ == "__main__":
    try:
        import multiprocessing
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()

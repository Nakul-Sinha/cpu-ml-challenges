"""P2 end-to-end nested eval: baseline transducer vs P2-enhanced transducer, on the
N3 CV-optimal config (M4 base + N2 de + N1 it gated-NP + group-vote de/en), leak-free.

Detection (LGBM probs) is transducer-independent and identical across variants (same
seed), so this isolates the TRANSDUCTION gain.  Prints overall nested + per-lang for the
"+N1+N2 (fixed it spine)" rung (the shipped CV-optimal) and the full ladder.

Run on box:  cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/N3/eval_p2.py
"""
import os, sys, json, time, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_final as PF          # sets up sys.path (M4, N2, N1, solution)
import pipeline
import transducer_p2
import elru
from run_m4 import score_edits, fp_counts, SHIP_VOTE_LANGS


def cv_optimal_rung(P):
    """Reproduce the '+N1+N2 (fixed it spine)' CV-optimal rung -> (nested, per-lang, fp)."""
    rows = P["rows"]; rbi = P["rows_by_id"]; gbi = P["gbi"]
    trs = P["trs"]; stf = P["stf"]; idfold = P["idfold"]
    cache_n2 = P["cache_n2"]; it_cache_n1 = P["it_cache_n1"]
    # de/en base_select on n2 cache; it fixed-spine N1 select
    n2_nn_thr, n2_nn_e, n2_nby, n2_ne_e = PF.de_en_edits(rows, cache_n2)
    it_n1_op, it_n1_nn, it_n1_nby, it_n1_ne = PF.select_it(rows, it_cache_n1, [PF.N1_SPINE], PF.N1_GATE_GRID)
    # combine (it overrides) + group-vote de+en (hi .60 lo .40)
    nn_e, ne_e = PF.combine(rows, rbi, n2_nn_e, n2_ne_e, it_n1_nn, it_n1_ne)
    nn, ne = PF.apply_vote(nn_e, ne_e, rbi, gbi, trs, stf, idfold, SHIP_VOTE_LANGS, SHIP_VOTE_LANGS, 0.60, 0.40)
    nn_s, nn_d = score_edits(rows, nn)
    ne_s, ne_d = score_edits(rows, ne)
    fp = fp_counts(rows, nn)
    return dict(nested=ne_s, nonnested=nn_s,
                nested_lang={L: ne_d[L]["lang_score"] for L in ("de", "en", "it")},
                nonnested_lang={L: nn_d[L]["lang_score"] for L in ("de", "en", "it")},
                nested_detail=ne_d, fp={L: list(fp[L]) for L in ("de", "en", "it")},
                it_op=it_n1_op, de_thr=n2_nn_thr["de"], en_thr=n2_nn_thr["en"])


def run_variant(label, patch):
    if patch:
        PF.Transducer = transducer_p2.Transducer
        pipeline.Transducer = transducer_p2.Transducer
    else:
        import transducer as T_base
        PF.Transducer = T_base.Transducer
        pipeline.Transducer = T_base.Transducer
    t0 = time.time()
    P = PF.prepare(verbose=False)
    r = cv_optimal_rung(P)
    r["seconds"] = round(time.time() - t0, 1)
    print(f"\n===== {label} (CV-optimal rung: M4+N2+N1 fixed-spine, vote de/en) =====")
    print(f"  NESTED   overall={r['nested']:.4f}   de={r['nested_lang']['de']:.4f} "
          f"en={r['nested_lang']['en']:.4f} it={r['nested_lang']['it']:.4f}")
    print(f"  NONNEST  overall={r['nonnested']:.4f}   de={r['nonnested_lang']['de']:.4f} "
          f"en={r['nonnested_lang']['en']:.4f} it={r['nonnested_lang']['it']:.4f}")
    for L in ("de", "en", "it"):
        d = r["nested_detail"][L]
        print(f"    {L}: edited={d['edited_mean']:.4f}(n={d['n_edited']}) unchanged={d['unchanged_mean']:.4f}(n={d['n_unchanged']})")
    print(f"  unchanged FP: " + ", ".join(f"{L}={r['fp'][L][0]}/{r['fp'][L][1]}" for L in ("de", "en", "it")))
    print(f"  ops: de_thr={r['de_thr']} en_thr={r['en_thr']} it_op={r['it_op']}  [{r['seconds']}s]")
    return r


if __name__ == "__main__":
    rb = run_variant("BASELINE transducer", patch=False)
    re_ = run_variant("P2-ENHANCED transducer", patch=True)
    print("\n================ P2 TRANSDUCTION DELTA (end-to-end nested) ================")
    print(f"  overall nested: {rb['nested']:.4f} -> {re_['nested']:.4f}   delta {re_['nested']-rb['nested']:+.4f}")
    print(f"  overall nonnest:{rb['nonnested']:.4f} -> {re_['nonnested']:.4f}   delta {re_['nonnested']-rb['nonnested']:+.4f}")
    for L in ("de", "en", "it"):
        print(f"  {L} nested: {rb['nested_lang'][L]:.4f} -> {re_['nested_lang'][L]:.4f}   "
              f"delta {re_['nested_lang'][L]-rb['nested_lang'][L]:+.4f}")
    json.dump(dict(baseline=rb, enhanced=re_,
                   delta_nested=round(re_['nested']-rb['nested'], 4)),
              open(os.path.join(HERE, "p2_eval.json"), "w"), indent=2, default=float)
    print("wrote p2_eval.json")

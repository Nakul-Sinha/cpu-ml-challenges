"""P2 productionize: full-train fit with the IT-only enhanced transducer, emit
P2-namespaced submissions (CV-optimal de@.07 + robust de@.11) WITHOUT clobbering N3's
baseline artifacts.  Also records OOF nested (CV-optimal + robust) for the enhanced config.

Run on box: cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/N3/ship_p2.py
"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_final as PF
import pipeline, transducer_p2, elru, pandas as pd
from run_m4 import score_edits, SHIP_VOTE_LANGS, group_consistency

# ---- patch enhanced transducer everywhere the pipeline builds one ----
PF.Transducer = transducer_p2.Transducer
pipeline.Transducer = transducer_p2.Transducer


def main():
    P = PF.prepare(verbose=False)
    rows = P["rows"]; rbi = P["rows_by_id"]; gbi = P["gbi"]
    trs = P["trs"]; stf = P["stf"]; idfold = P["idfold"]
    cache_n2 = P["cache_n2"]; it_cache_n1 = P["it_cache_n1"]; train = P["train"]

    # CV-optimal rung edits (nn=all-OOF op, ne=nested)
    n2_nn_thr, n2_nn_e, n2_nby, n2_ne_e = PF.de_en_edits(rows, cache_n2)
    it_op, it_nn, it_nby, it_ne = PF.select_it(rows, it_cache_n1, [PF.N1_SPINE], PF.N1_GATE_GRID)
    nn_e, ne_e = PF.combine(rows, rbi, n2_nn_e, n2_ne_e, it_nn, it_ne)
    nn, ne = PF.apply_vote(nn_e, ne_e, rbi, gbi, trs, stf, idfold, SHIP_VOTE_LANGS, SHIP_VOTE_LANGS, 0.60, 0.40)
    ne_s, ne_d = score_edits(rows, ne); nn_s, nn_d = score_edits(rows, nn)

    # robust: de fixed @0.11, en nested, it fixed op
    de_robust = 0.11
    rob_ne = {}
    for R in rows:
        i = R["id"]; L = R["lang"]; k = idfold[i]
        if L == "de":
            rob_ne[i] = cache_n2[i][de_robust]
        elif L == "en":
            rob_ne[i] = cache_n2[i][n2_nby[k]["en"]]
        else:
            rob_ne[i] = ne[i]
    rob_ne = group_consistency({i: rob_ne[i] for i in rob_ne}, rbi, gbi, trs, stf, idfold,
                               vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    rob_s, rob_d = score_edits(rows, rob_ne)

    # canonical scorer self-check
    oof = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(ne[R["id"]], ensure_ascii=False)} for R in rows])
    chk, _ = elru.score_frames(oof, train[["id", "language", "edits_json"]])
    oof.to_csv(os.path.join(HERE, "oof_edits_p2.csv"), index=False)

    print(f"CV-optimal nested={ne_s:.4f} (de {ne_d['de']['lang_score']:.4f} en {ne_d['en']['lang_score']:.4f} it {ne_d['it']['lang_score']:.4f})")
    print(f"           nonnested={nn_s:.4f}   canonical OOF check={chk:.4f}")
    print(f"robust(de@.11) nested={rob_s:.4f} (de {rob_d['de']['lang_score']:.4f} en {rob_d['en']['lang_score']:.4f} it {rob_d['it']['lang_score']:.4f})")

    # ---- full-train fit + submissions ----
    test = pd.read_csv(os.path.join(PF.ROOT, "dataset", "test.csv"))
    gbi_tr = {r.id: r.document_group for r in train.itertuples()}
    stores_full, all_rows, det_full, trd_full = PF._full_train_artifacts(train)
    gate_model, tab_full, gc_full = PF._fit_full_it_gate(train, all_rows, det_full, gbi_tr)
    de_thr = float(n2_nn_thr["de"]); en_thr = float(n2_nn_thr["en"])
    it_spine, it_gate = float(it_op[0]), float(it_op[1])

    sub, _ = PF._build_submission(test, det_full, trd_full, stores_full, gate_model, tab_full, gc_full,
                                  de_thr, en_thr, it_spine, it_gate)
    rates = PF._validate_and_rates(sub, test)
    pd.DataFrame([{"id": i, "edits_json": json.dumps(sub[i], ensure_ascii=False)} for i in test.id]
                 ).to_csv(os.path.join(HERE, "submission_v3_p2.csv"), index=False)
    subR, _ = PF._build_submission(test, det_full, trd_full, stores_full, gate_model, tab_full, gc_full,
                                   de_robust, en_thr, it_spine, it_gate)
    ratesR = PF._validate_and_rates(subR, test)
    pd.DataFrame([{"id": i, "edits_json": json.dumps(subR[i], ensure_ascii=False)} for i in test.id]
                 ).to_csv(os.path.join(HERE, "submission_v3_p2_robust.csv"), index=False)
    ne_full = sum(1 for i in sub if sub[i]); ne_fullR = sum(1 for i in subR if subR[i])
    print(f"submission_v3_p2.csv edited {ne_full}/445  rates " + ", ".join(f"{L}={rates[L]['frac']}(r{rates[L]['ratio']})" for L in ('de','en','it')))
    print(f"submission_v3_p2_robust.csv edited {ne_fullR}/445  rates " + ", ".join(f"{L}={ratesR[L]['frac']}(r{ratesR[L]['ratio']})" for L in ('de','en','it')))

    rep = dict(config="N3 CV-optimal + IT-only P2-enhanced transducer (multi-token append compose)",
               headline_nested=round(ne_s, 4), nonnested=round(nn_s, 4), canonical_oof_check=round(chk, 4),
               robust_nested=round(rob_s, 4),
               nested_lang={L: round(ne_d[L]['lang_score'], 4) for L in ('de', 'en', 'it')},
               baseline_nested=0.5503, delta_nested=round(ne_s - 0.5503, 4),
               submission_edit_frac={L: rates[L]['frac'] for L in ('de', 'en', 'it')})
    json.dump(rep, open(os.path.join(HERE, "cv_report_p2.json"), "w"), indent=2, ensure_ascii=False)
    print("wrote submission_v3_p2.csv, submission_v3_p2_robust.csv, oof_edits_p2.csv, cv_report_p2.json")


if __name__ == "__main__":
    main()

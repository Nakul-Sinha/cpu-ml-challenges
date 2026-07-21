"""P3 pipeline_v4 -- FINAL COMPOSER.  Integrate P1 (de BiGRU ensemble + it LGBM
re-scorer) AND P2 (IT-only multi-token enhanced transducer) onto the N3 base and
report the honest ladder  N3 -> +P1 -> +P2 -> joint(+P1+P2)  (nested + non-nested,
per-language, FP), with an independent canonical elru.score_frames self-check.

Detection levers (P1):
  * de : ensembled per-token prob (1-a)*shared_LGBM + a*BiGRU ; a fixed 0.6 (pre-
         committed) + de spine threshold selected nested.  en FROZEN (a=0).
  * it : IT-only LGBM re-scorer additive-boosted into the NP-gate spine (w nested).
Transduction lever (P2):
  * it : the enhanced transducer (multi-token decomp + append rules, ENHANCE_LANGS=it)
         replaces the A2 fallback in the it NP-gate assembly.  de/en byte-identical.

Selection is honest: every operating point nested (fold-k op from the other 4) + the
all-OOF non-nested reference.  group-vote de+en (hi.60/lo.40) as N3.  Reports whether
each component survives integration; drops any that does not.

Run: cd ~/insled && OMP_NUM_THREADS=7 P1_GRU_SEEDS=5 nice -n 10 ~/venv/bin/python runs/P3/pipeline_v4.py
"""
import os, sys, json, time, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/insled")
for p in (os.path.join(ROOT, "runs", "M4"), os.path.join(ROOT, "runs", "N2"),
          os.path.join(ROOT, "runs", "N1"), os.path.join(ROOT, "runs", "N3"),
          os.path.join(ROOT, "runs", "P1"), os.path.join(ROOT, "solution"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import pipeline, elru
from run_n1 import assemble_it as n1_assemble_it, IT_SPINE_THR
from run_m4 import (base_cache, base_select, group_consistency, score_edits, fp_counts,
                    per_type_recall, print_detail, LOSSMAP, SHIP_VOTE_LANGS)
from transducer import Transducer as TransducerBase
from transducer_p2 import Transducer as TransducerP2
import p1_base, p1_lever1, p1_lever2

LANGS = pipeline.LANGS
A_GRID = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
IT_W = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]


def ens_probs(ids, shared, seq, a):
    return {i: ((1 - a) * np.asarray(shared[i]) + a * np.asarray(seq[i])).tolist() for i in ids}


def it_boost_probs(itrows, shared, p_it, w):
    out = {}
    for R in itrows:
        sh = np.asarray(shared[R["id"]]); pit = np.asarray(p_it[R["id"]]) if p_it[R["id"]] else sh
        out[R["id"]] = np.clip(sh + w * np.clip(pit - 0.3, 0, None), 0, 1).tolist()
    return out


def fit_p2_transducers(train):
    """P2 (it-enhanced) transducer per fold, fit leak-free on the fold-train frame."""
    trs_p2 = {}
    for k in range(5):
        trs_p2[k] = TransducerP2().fit(train[train.fold != k])
    return trs_p2


def main():
    t0 = time.time()
    P = p1_base.prepare(verbose=True)
    train = P["train"]; rows = P["rows"]; idfold = P["idfold"]; gbi = P["gbi"]
    shared = P["row_proba"]; trs = P["trs"]; stf = P["stf"]; gate_scores = P["gate_scores"]
    rbi = P["rows_by_id"]; tabs = P["tabs"]; gctxs = P["gctxs"]
    rbl = {L: [R for R in rows if R["lang"] == L] for L in LANGS}
    truth = {R["id"]: R["truth"] for R in rows}
    print(f"[prepare {time.time()-t0:.0f}s]", flush=True)

    # P2 transducers (it-enhanced) per fold
    trs_p2 = fit_p2_transducers(train)
    print(f"[P2 transducers fit {time.time()-t0:.0f}s]", flush=True)

    # ---- Lever 2: BiGRU OOF probs ----
    n_seeds = int(os.environ.get("P1_GRU_SEEDS", "5"))
    seq = p1_lever2.gru_oof(rows, idfold, verbose=True, t0=t0, n_seeds=n_seeds)
    # ---- Lever 1: it re-scorer OOF probs ----
    itrows = rbl["it"]
    p_it = p1_lever1.rescorer_oof(itrows, idfold, train, tabs, gctxs, gbi, shared=None)
    print(f"[both levers trained {time.time()-t0:.0f}s]", flush=True)

    # ================= de/en operating-point selection (P1 Lever 2) =================
    deen_rows = rbl["de"] + rbl["en"]
    cache_by_a = {}
    for a in A_GRID:
        rp = ens_probs([R["id"] for R in deen_rows], shared, seq, a)
        cache_by_a[a] = base_cache(deen_rows, idfold, rp, trs, stf)
    print(f"[de/en caches {time.time()-t0:.0f}s]", flush=True)

    def lang_sub(L, a, thr, ids):
        e = {i: cache_by_a[a][i][thr] for i in ids}
        _s, d = elru.elru(e, {i: truth[i] for i in ids}, {i: L for i in ids}, detail=True)
        return d[L]["lang_score"]

    GRID = pipeline.GRID

    def select_deen(L, a_choices, thr_choices):
        allids = set(R["id"] for R in rbl[L])
        ops = [(a, thr) for a in a_choices for thr in thr_choices]
        nn_op = max(ops, key=lambda op: lang_sub(L, op[0], op[1], allids))
        nn_e = {R["id"]: cache_by_a[nn_op[0]][R["id"]][nn_op[1]] for R in rbl[L]}
        nby = {}; ne_e = {}
        for k in range(5):
            other = set(R["id"] for R in rbl[L] if R["fold"] != k)
            b = max(ops, key=lambda op: lang_sub(L, op[0], op[1], other))
            nby[k] = b
            for R in rbl[L]:
                if R["fold"] == k:
                    ne_e[R["id"]] = cache_by_a[b[0]][R["id"]][b[1]]
        return nn_e, ne_e, nn_op, nby

    de_base_nn, de_base_ne, de_base_op, de_base_nby = select_deen("de", [0.0], GRID)      # N3 (a=0)
    de_gA_nn, de_gA_ne, de_gA_op, de_gA_nby = select_deen("de", [0.6], GRID)              # SHIP (a=0.6 fixed)
    de_gF_nn, de_gF_ne, de_gF_op, de_gF_nby = select_deen("de", A_GRID, GRID)             # ref (a,thr nested)
    en_nn, en_ne, en_op, en_nby = select_deen("en", [0.0], GRID)                          # en frozen

    # ================= it: (boost source, w) x transducer selection =================
    def it_assemble(probs, trset):
        out = {}
        for R in itrows:
            k = idfold[R["id"]]
            gs = [(ab[0], ab[1], p) for (ab, p) in gate_scores[R["id"]]]
            out[R["id"]] = n1_assemble_it(R["tk"], R["text"], probs[R["id"]], 0.8, gs, trset[k], stf[k])
        return out

    it_src = {"rescorer": p_it,
              "gru": {R["id"]: seq[R["id"]] for R in itrows}}
    truth_it = {R["id"]: R["truth"] for R in itrows}

    # build it caches for both transducers over (source, w)
    def build_it_cache(trset):
        c = {}
        for sname, src in it_src.items():
            for w in IT_W:
                c[(sname, w)] = it_assemble(it_boost_probs(itrows, shared, src, w), trset)
        return c
    it_cache_base = build_it_cache(trs)
    it_cache_p2 = build_it_cache(trs_p2)
    print(f"[it caches (base+P2) {time.time()-t0:.0f}s]", flush=True)

    it_ops = [(s, w) for s in it_src for w in IT_W]

    def it_select(it_cache):
        def it_sub(op, ids):
            e = {i: it_cache[op][i] for i in ids}
            _s, d = elru.elru(e, {i: truth_it[i] for i in ids}, {i: "it" for i in ids}, detail=True)
            return d["it"]["lang_score"]
        allids = set(truth_it)
        nn_op = max(it_ops, key=lambda op: it_sub(op, allids))
        nn_e = {i: it_cache[nn_op][i] for i in truth_it}
        nby = {}; ne_e = {}
        for k in range(5):
            other = set(R["id"] for R in itrows if idfold[R["id"]] != k)
            bop = max(it_ops, key=lambda op: it_sub(op, other))
            nby[k] = bop
            for R in itrows:
                if idfold[R["id"]] == k:
                    ne_e[R["id"]] = it_cache[bop][R["id"]]
        return nn_op, nn_e, nby, ne_e

    # it variants
    it_base0 = {i: it_cache_base[("rescorer", 0.0)][i] for i in truth_it}   # N3 it (baseline trd, w=0)
    it_p2_0 = {i: it_cache_p2[("rescorer", 0.0)][i] for i in truth_it}       # +P2 it (P2 trd, w=0)
    it_b_nn_op, it_b_nn, it_b_nby, it_b_ne = it_select(it_cache_base)        # +P1 it (baseline trd, boost)
    it_j_nn_op, it_j_nn, it_j_nby, it_j_ne = it_select(it_cache_p2)          # joint it (P2 trd, boost)

    # ================= ladder (group-vote de+en) =================
    def assemble_full(de_nn, de_ne, it_nn, it_ne):
        nn = {}; ne = {}
        for R in rows:
            L = R["lang"]; i = R["id"]
            if L == "de":
                nn[i] = de_nn[i]; ne[i] = de_ne[i]
            elif L == "en":
                nn[i] = en_nn[i]; ne[i] = en_ne[i]
            else:
                nn[i] = it_nn[i]; ne[i] = it_ne[i]
        nnv = group_consistency(nn, rbi, gbi, trs, stf, idfold, vote_langs=SHIP_VOTE_LANGS,
                                drop_langs=SHIP_VOTE_LANGS, do_conv=False)
        nev = group_consistency(ne, rbi, gbi, trs, stf, idfold, vote_langs=SHIP_VOTE_LANGS,
                                drop_langs=SHIP_VOTE_LANGS, do_conv=False)
        return nnv, nev

    ladder = collections.OrderedDict()

    def rung(tag, de_nn, de_ne, it_nn, it_ne, show=False):
        nnv, nev = assemble_full(de_nn, de_ne, it_nn, it_ne)
        nn_s, nn_d = score_edits(rows, nnv); ne_s, ne_d = score_edits(rows, nev)
        fp = fp_counts(rows, nnv)
        ladder[tag] = dict(nested=round(ne_s, 4), nonnested=round(nn_s, 4),
                           ne_lang={L: round(ne_d[L]["lang_score"], 4) for L in LANGS},
                           nn_lang={L: round(nn_d[L]["lang_score"], 4) for L in LANGS},
                           fp={L: list(fp[L]) for L in LANGS}, nnv=nnv, nev=nev)
        print(f"  {tag:30s} nested={ne_s:.4f} nonnest={nn_s:.4f}  "
              f"[de {ne_d['de']['lang_score']:.4f} en {ne_d['en']['lang_score']:.4f} it {ne_d['it']['lang_score']:.4f}]  "
              f"FP de={fp['de'][0]}/{fp['de'][1]} it={fp['it'][0]}/{fp['it'][1]}")
        if show:
            ptr = per_type_recall(rows, nnv)
            for key in sorted(ptr):
                r, nsp = ptr[key]; b = LOSSMAP.get(key)
                if key[0] in ("de", "it"):
                    print(f"        {key[0]} {key[1]:14s} rec={r:.3f}(n={nsp})" + (f" lm{b:.3f}" if b else ""))
        return ladder[tag]

    print("\n================ P3 v4 LADDER (group-vote de+en, hi.60/lo.40) ================")
    rung("N3 base (reproduce)", de_base_nn, de_base_ne, it_base0, it_base0)
    rung("+P1 (de BiGRU + it rescorer)", de_gA_nn, de_gA_ne, it_b_nn, it_b_ne)
    rung("+P2 (it enhanced transducer)", de_base_nn, de_base_ne, it_p2_0, it_p2_0)
    ship = rung("+P1+P2 JOINT (de a=.6)", de_gA_nn, de_gA_ne, it_j_nn, it_j_ne, show=True)
    ship_ref = rung("+P1+P2 ref (de a,thr nest)", de_gF_nn, de_gF_ne, it_j_nn, it_j_ne)

    # ---- token PR-AUC summary ----
    print("\n---- token PR-AUC (shared vs GRU vs it-rescorer) ----")
    for L in LANGS:
        rL = rbl[L]; y = np.concatenate([np.asarray(R["y"]) for R in rL])
        sh = np.concatenate([np.asarray(shared[R["id"]]) for R in rL])
        gr = np.concatenate([np.asarray(seq[R["id"]]) for R in rL])
        line = f"  {L}: shared={average_precision_score(y,sh):.4f} GRU={average_precision_score(y,gr):.4f}"
        if L == "it":
            pit = np.concatenate([np.asarray(p_it[R["id"]] if p_it[R["id"]] else shared[R["id"]]) for R in rL])
            line += f" it-rescorer={average_precision_score(y,pit):.4f}"
        print(line)

    # ---- selected ops (for ship baking) ----
    print("\n================ SELECTED OPERATING POINTS (for solution.py) ================")
    de_a_ship = 0.6
    de_thr_ship_nn = float(de_gA_op[1])                     # de non-nested thr at a=0.6
    en_thr_ship_nn = float(en_op[1])
    it_ship_op = it_j_nn_op                                 # (source, w) non-nested with P2 trd
    print(f"  de: a={de_a_ship} thr_nonnested={de_thr_ship_nn}  thr_by_fold={ {k: round(de_gA_nby[k][1],3) for k in range(5)} }")
    print(f"  en: a=0 thr_nonnested={en_thr_ship_nn}  thr_by_fold={ {k: round(en_nby[k][1],3) for k in range(5)} }")
    print(f"  it: spine={IT_SPINE_THR} gate=0.8 boost=({it_ship_op[0]},{it_ship_op[1]}) transducer=P2  by_fold={ {k:(it_j_nby[k][0],float(it_j_nby[k][1])) for k in range(5)} }")
    print(f"  de (a,thr) ref by-fold: {[ (round(de_gF_nby[k][0],2),round(de_gF_nby[k][1],3)) for k in range(5)]}")

    # ---- SHIP-FIXED honest nested: de FIXED thr @ a=0.6 (pre-committed), en nested,
    #      it FIXED (rescorer,0.6)+P2, group-vote -- the direct honest number for the
    #      fixed-threshold submission (analog of N3 robust_variant). ----
    it_fixed = {i: it_cache_p2[("rescorer", 0.6)][i] for i in truth_it}
    print("\n================ SHIP-FIXED honest nested (de fixed @a=.6, en nested, it fixed rescorer0.6+P2) ================")
    ship_fixed_scores = {}
    for de_thr_fixed in [0.15, 0.19, 0.25, 0.29, 0.31, 0.35]:
        de_fx = {R["id"]: cache_by_a[0.6][R["id"]][de_thr_fixed] for R in rbl["de"]}
        nn = {}; ne = {}
        for R in rows:
            L = R["lang"]; i = R["id"]
            if L == "de":
                nn[i] = de_fx[i]; ne[i] = de_fx[i]
            elif L == "en":
                nn[i] = en_nn[i]; ne[i] = en_ne[i]     # en nested per fold
            else:
                nn[i] = it_fixed[i]; ne[i] = it_fixed[i]
        nev = group_consistency(ne, rbi, gbi, trs, stf, idfold, vote_langs=SHIP_VOTE_LANGS,
                                drop_langs=SHIP_VOTE_LANGS, do_conv=False)
        nnv = group_consistency(nn, rbi, gbi, trs, stf, idfold, vote_langs=SHIP_VOTE_LANGS,
                                drop_langs=SHIP_VOTE_LANGS, do_conv=False)
        s_ne, d_ne = score_edits(rows, nev); s_nn, _dn = score_edits(rows, nnv)
        ship_fixed_scores[de_thr_fixed] = dict(nested=round(s_ne, 4), nonnested=round(s_nn, 4),
                                               de=round(d_ne["de"]["lang_score"], 4),
                                               it=round(d_ne["it"]["lang_score"], 4))
        print(f"  de_thr={de_thr_fixed:.2f}  nested={s_ne:.4f} nonnest={s_nn:.4f}  "
              f"[de {d_ne['de']['lang_score']:.4f} en {d_ne['en']['lang_score']:.4f} it {d_ne['it']['lang_score']:.4f}]")

    # ---- canonical self-check on JOINT ship edits (nested + non-nested) ----
    oof_nn = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(ship["nnv"][R["id"]], ensure_ascii=False)} for R in rows])
    oof_ne = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(ship["nev"][R["id"]], ensure_ascii=False)} for R in rows])
    chk_nn, _ = elru.score_frames(oof_nn, train[["id", "language", "edits_json"]])
    chk_ne, _ = elru.score_frames(oof_ne, train[["id", "language", "edits_json"]])
    print(f"\ncanonical elru.score_frames  JOINT non-nested={chk_nn:.4f}  nested={chk_ne:.4f}")
    n_valid = sum(1 for R in rows if elru.validate_edits(ship["nev"][R["id"]], len(R["text"])))
    print(f"OOF rows valid: {n_valid}/{len(rows)}")
    oof_ne.to_csv(os.path.join(HERE, "oof_edits_v4.csv"), index=False)

    report = dict(
        config="P3 v4: N3 base + P1(de BiGRU a=.6 + it rescorer) + P2(it enhanced transducer) + group-vote[de,en]",
        headline_nested=round(ship["nested"], 4), nonnested=round(ship["nonnested"], 4),
        canonical_check_nested=round(chk_ne, 4), canonical_check_nonnested=round(chk_nn, 4),
        oof_rows_valid=f"{n_valid}/{len(rows)}",
        n3_base_nested=0.5503, delta_vs_n3=round(ship["nested"] - 0.5503, 4),
        ai_baseline=0.56, beats_ai_baseline=bool(ship["nested"] > 0.56),
        ladder={t: {k: v for k, v in r.items() if k not in ("nnv", "nev")} for t, r in ladder.items()},
        ref_de_a_thr={k: v for k, v in ship_ref.items() if k not in ("nnv", "nev")},
        ops=dict(de_a=de_a_ship, de_thr_nonnested=de_thr_ship_nn,
                 de_thr_by_fold={str(k): float(de_gA_nby[k][1]) for k in range(5)},
                 en_thr_nonnested=en_thr_ship_nn,
                 en_thr_by_fold={str(k): float(en_nby[k][1]) for k in range(5)},
                 it_spine=IT_SPINE_THR, it_gate=0.8,
                 it_boost_source=it_ship_op[0], it_boost_w=float(it_ship_op[1]),
                 it_boost_by_fold={str(k): [it_j_nby[k][0], float(it_j_nby[k][1])] for k in range(5)},
                 it_transducer="P2", en_frozen=True,
                 group_vote=list(SHIP_VOTE_LANGS), group_hi=0.6, group_lo=0.4),
        ship_fixed_de_thr_scan=ship_fixed_scores,
        it_baseline_boost_op=[it_b_nn_op[0], float(it_b_nn_op[1])],
        pr_auc={L: dict(shared=round(float(average_precision_score(
                    np.concatenate([np.asarray(R["y"]) for R in rbl[L]]),
                    np.concatenate([np.asarray(shared[R["id"]]) for R in rbl[L]]))), 4),
                 gru=round(float(average_precision_score(
                    np.concatenate([np.asarray(R["y"]) for R in rbl[L]]),
                    np.concatenate([np.asarray(seq[R["id"]]) for R in rbl[L]]))), 4)) for L in LANGS})
    json.dump(report, open(os.path.join(HERE, "cv_report_v4.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nHEADLINE v4 nested = {ship['nested']:.4f}  (N3 0.5503, AI baseline 0.56: "
          f"{'BEAT' if ship['nested']>0.56 else 'below'})   [{time.time()-t0:.0f}s]")
    print("wrote cv_report_v4.json, oof_edits_v4.csv")


if __name__ == "__main__":
    main()

"""P1 COMBINED -- detector upgrades on the N3 base.

Composes the two honest detector levers and reports the full ablation ladder
(nested + non-nested, per-language, FP, per-type recall vs the gap table), plus a
canonical-scorer self-check.

  * de : BiGRU sequence-tagger ensemble  (Lever 2).  Per-token prob =
         (1-a)*shared_LGBM + a*BiGRU ; a (and the de spine threshold) selected NESTED.
         Sequence context sharpens de plain / paired-form tokens the per-token LGBM
         caps -> big de gain.
  * en : FROZEN at the N3 base (GRU raises en PR-AUC but does NOT transfer downstream --
         en ELRU is replacement/budget-bound, not detection-bound; measure-and-drop).
  * it : IT-ONLY LGBM re-scorer  (Lever 1) additive-boosted into the fixed NP-gate spine
         (independent view beats stacked/convex; w selected NESTED).  Small honest gain.

de/en group-vote (hi.60/lo.40) shipped as in N3.  Everything leak-free per fold; de/en
ops nested; it op nested; reports BOTH nested (honest headline) and non-nested.

Run: cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/P1/p1_combined.py
"""
import os, sys, json, time, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/insled")
for p in (os.path.join(ROOT, "runs", "M4"), os.path.join(ROOT, "runs", "N2"),
          os.path.join(ROOT, "runs", "N1"), os.path.join(ROOT, "solution"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import pipeline, elru
from run_n1 import assemble_it as n1_assemble_it, IT_SPINE_THR
from run_m4 import (base_cache, base_select, group_consistency, score_edits, fp_counts,
                    per_type_recall, print_detail, LOSSMAP, SHIP_VOTE_LANGS)
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


def main():
    t0 = time.time()
    P = p1_base.prepare(verbose=True)
    train = P["train"]; rows = P["rows"]; idfold = P["idfold"]; gbi = P["gbi"]
    shared = P["row_proba"]; trs = P["trs"]; stf = P["stf"]; gate_scores = P["gate_scores"]
    rbi = P["rows_by_id"]; tabs = P["tabs"]; gctxs = P["gctxs"]
    rbl = {L: [R for R in rows if R["lang"] == L] for L in LANGS}
    truth = {R["id"]: R["truth"] for R in rows}
    print(f"[prepare {time.time()-t0:.0f}s]", flush=True)

    # ---- Lever 2: BiGRU OOF probs (all langs) ----
    n_seeds = int(os.environ.get("P1_GRU_SEEDS", "1"))
    seq = p1_lever2.gru_oof(rows, idfold, verbose=True, t0=t0, n_seeds=n_seeds)
    # ---- Lever 1: it re-scorer OOF probs (independent) ----
    itrows = rbl["it"]
    p_it = p1_lever1.rescorer_oof(itrows, idfold, train, tabs, gctxs, gbi, shared=None)
    print(f"[both levers trained {time.time()-t0:.0f}s]", flush=True)

    # ================= per-language operating-point selection =================
    # de/en base caches per ensemble weight a (a=0 == shared == N3 base)
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

    def select_deen(L, a_choices, thr_choices):
        """returns (nn_edits, ne_edits, nn_op, nby) over a_choices x thr_choices, honest nested."""
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

    GRID = pipeline.GRID
    # de variants: N3 (a=0); (a,thr) both nested (upper ref); a FIXED 0.6 + thr nested (robust, 1-DOF like N3)
    de_base_nn, de_base_ne, de_base_op, de_base_nby = select_deen("de", [0.0], GRID)
    de_gA_nn, de_gA_ne, de_gA_op, de_gA_nby = select_deen("de", [0.6], GRID)   # robust: a pre-committed
    de_gF_nn, de_gF_ne, de_gF_op, de_gF_nby = select_deen("de", A_GRID, GRID)
    # en frozen at N3 (a=0, thr nested)
    en_nn, en_ne, en_op, en_nby = select_deen("en", [0.0], GRID)

    def de_lang(ne_e):
        _s, d = elru.elru({R["id"]: ne_e[R["id"]] for R in rbl["de"]},
                          {R["id"]: R["truth"] for R in rbl["de"]},
                          {R["id"]: "de" for R in rbl["de"]}, detail=True)
        return d["de"]["lang_score"]
    print(f"\nde nested variants (pre-vote de lang): N3(a0)={de_lang(de_base_ne):.4f}  "
          f"a=.6-fixed,thr-nested={de_lang(de_gA_ne):.4f} (thr-by-fold {[round(de_gA_nby[k][1],3) for k in range(5)]})  "
          f"(a,thr)-nested={de_lang(de_gF_ne):.4f} (ops {[ (round(de_gF_nby[k][0],2),round(de_gF_nby[k][1],3)) for k in range(5)]})")

    # ---- it: additive boost, source in {re-scorer, GRU-ensemble, their avg}; (src,w) nested ----
    def it_assemble(probs):
        out = {}
        for R in itrows:
            k = idfold[R["id"]]
            gs = [(ab[0], ab[1], p) for (ab, p) in gate_scores[R["id"]]]
            out[R["id"]] = n1_assemble_it(R["tk"], R["text"], probs[R["id"]], 0.8, gs, trs[k], stf[k])
        return out
    it_src = {"rescorer": p_it,
              "gru": {R["id"]: seq[R["id"]] for R in itrows},
              "avg": {R["id"]: (0.5 * (np.asarray(p_it[R["id"]]) if p_it[R["id"]] else np.asarray(shared[R["id"]]))
                                + 0.5 * np.asarray(seq[R["id"]])).tolist() for R in itrows}}
    it_cache = {}
    for sname, src in it_src.items():
        for w in IT_W:
            it_cache[(sname, w)] = it_assemble(it_boost_probs(itrows, shared, src, w))
    truth_it = {R["id"]: R["truth"] for R in itrows}
    it_ops = [(s, w) for s in it_src for w in IT_W]
    def it_sub(op, ids):
        e = {i: it_cache[op][i] for i in ids}
        _s, d = elru.elru(e, {i: truth_it[i] for i in ids}, {i: "it" for i in ids}, detail=True)
        return d["it"]["lang_score"]
    it_allids = set(truth_it)
    it_nn_op = max(it_ops, key=lambda op: it_sub(op, it_allids))
    it_nn_w = it_nn_op                                          # (source, w)
    it_base_edits = {i: it_cache[("rescorer", 0.0)][i] for i in truth_it}   # w=0 == shared spine (N3 it)
    it_nn_edits = {i: it_cache[it_nn_op][i] for i in truth_it}
    it_nby = {}; it_ne_edits = {}
    for k in range(5):
        other = set(R["id"] for R in itrows if idfold[R["id"]] != k)
        bop = max(it_ops, key=lambda op: it_sub(op, other))
        it_nby[k] = bop
        for R in itrows:
            if idfold[R["id"]] == k:
                it_ne_edits[R["id"]] = it_cache[bop][R["id"]]

    # ================= ablation ladder (group-vote de+en) =================
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
        print(f"  {tag:26s} nested={ne_s:.4f} nonnest={nn_s:.4f}  "
              f"[de {ne_d['de']['lang_score']:.4f} en {ne_d['en']['lang_score']:.4f} it {ne_d['it']['lang_score']:.4f}]  "
              f"FP de={fp['de'][0]}/{fp['de'][1]} it={fp['it'][0]}/{fp['it'][1]}")
        if show:
            ptr = per_type_recall(rows, nnv)
            for key in sorted(ptr):
                r, nsp = ptr[key]; b = LOSSMAP.get(key)
                if key[0] in ("de", "it"):
                    print(f"        {key[0]} {key[1]:14s} rec={r:.3f}(n={nsp})" + (f" lm{b:.3f}" if b else ""))
        return ladder[tag]

    print("\n================ P1 DETECTOR-UPGRADE ABLATION LADDER (group-vote de+en) ================")
    rung("N3 base (reproduce)", de_base_nn, de_base_ne, it_base_edits, it_base_edits)
    rung("+de BiGRU (a=.6 robust)", de_gA_nn, de_gA_ne, it_base_edits, it_base_edits)
    rung("+de BiGRU (a,thr nest)", de_gF_nn, de_gF_ne, it_base_edits, it_base_edits)
    rung("+it re-scorer", de_base_nn, de_base_ne, it_nn_edits, it_ne_edits)
    ship = rung("+both SHIP (de a=.6)", de_gA_nn, de_gA_ne, it_nn_edits, it_ne_edits, show=True)
    ship_ref = rung("+both ref (de a,thr nest)", de_gF_nn, de_gF_ne, it_nn_edits, it_ne_edits)

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

    print(f"\n  it boost (source,w) nonnested={it_nn_w} by-fold={dict((k,(v[0],float(v[1]))) for k,v in it_nby.items())}")
    print(f"  de (a,thr) nested by-fold: {[ (round(de_gF_nby[k][0],2),round(de_gF_nby[k][1],3)) for k in range(5)]}")

    # ---- canonical scorer self-check on SHIP nested + non-nested edits ----
    oof_nn = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(ship["nnv"][R["id"]], ensure_ascii=False)} for R in rows])
    oof_ne = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(ship["nev"][R["id"]], ensure_ascii=False)} for R in rows])
    chk_nn, _ = elru.score_frames(oof_nn, train[["id", "language", "edits_json"]])
    chk_ne, _ = elru.score_frames(oof_ne, train[["id", "language", "edits_json"]])
    print(f"\ncanonical elru.score_frames  SHIP non-nested={chk_nn:.4f}  nested={chk_ne:.4f}")
    oof_ne.to_csv(os.path.join(HERE, "oof_edits_p1.csv"), index=False)

    report = dict(
        config="P1 detector upgrades on N3 base: de=BiGRU-ensemble, en=frozen, it=LGBM re-scorer boost + group-vote[de,en]",
        headline_nested=round(ship["nested"], 4), nonnested=round(ship["nonnested"], 4),
        canonical_check_nested=round(chk_ne, 4), canonical_check_nonnested=round(chk_nn, 4),
        n3_base_nested=0.5503, delta_vs_n3=round(ship["nested"] - 0.5503, 4),
        ai_baseline=0.56, beats_ai_baseline=bool(ship["nested"] > 0.56),
        ladder={t: {k: v for k, v in r.items() if k not in ("nnv", "nev")} for t, r in ladder.items()},
        upper_ref_de_a_thr_nested={k: v for k, v in ship_ref.items() if k not in ("nnv", "nev")},
        ops=dict(ship_de_a_fixed=0.6, ship_de_thr_by_fold={str(k): float(de_gA_nby[k][1]) for k in range(5)},
                 ref_de_a_thr_by_fold={str(k): [float(de_gF_nby[k][0]), float(de_gF_nby[k][1])] for k in range(5)},
                 en_frozen=True, it_boost_by_fold={str(k): [v[0], float(v[1])] for k, v in it_nby.items()},
                 it_boost_nonnested=[it_nn_w[0], float(it_nn_w[1])], it_spine=0.45, it_gate=0.8),
        pr_auc={L: dict(shared=round(float(average_precision_score(
                    np.concatenate([np.asarray(R["y"]) for R in rbl[L]]),
                    np.concatenate([np.asarray(shared[R["id"]]) for R in rbl[L]]))), 4),
                 gru=round(float(average_precision_score(
                    np.concatenate([np.asarray(R["y"]) for R in rbl[L]]),
                    np.concatenate([np.asarray(seq[R["id"]]) for R in rbl[L]]))), 4)) for L in LANGS})
    json.dump(report, open(os.path.join(HERE, "cv_report_p1.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nHEADLINE P1 nested = {ship['nested']:.4f}  (N3 0.5503, AI baseline 0.56: "
          f"{'BEAT' if ship['nested']>0.56 else 'below'})   [{time.time()-t0:.0f}s]")
    print("wrote cv_report_p1.json, oof_edits_p1.csv")


if __name__ == "__main__":
    main()

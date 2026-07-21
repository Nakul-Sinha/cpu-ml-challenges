"""P2 eval v3: baseline vs IT-only enhanced transducer, with MATCHED-SPAN chrF
(isolates realized transduction quality on the spans detection actually catches) and a
USE_DEL (duplicate-adjacency deletion) precision + nested probe.

Run on box: cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/N3/eval_p2c.py
"""
import os, sys, json, time, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_final as PF
import pipeline, m3_ext
import transducer_p2
import elru
from run_m4 import score_edits, fp_counts, SHIP_VOTE_LANGS


def matched_chrf(pred_map, truth_map, lang_map):
    """Greedy elru matching; return per-lang (mean matched chrf, n_matched, n_true_nonempty)."""
    agg = collections.defaultdict(lambda: [0.0, 0, 0])
    for _id, true in truth_map.items():
        L = lang_map[_id]; pred = pred_map.get(_id, [])
        agg[L][2] += sum(1 for e in true if e["replacement"] != "")
        if not pred or not true:
            continue
        pairs = []
        for i, pe in enumerate(pred):
            for j, te in enumerate(true):
                pairs.append((elru.pair_quality(pe, te), i, j))
        pairs.sort(key=lambda x: -x[0])
        up, ut = set(), set()
        for q, i, j in pairs:
            if i in up or j in ut:
                continue
            up.add(i); ut.add(j)
            te = true[j]; pe = pred[i]
            if te["replacement"] != "" or pe["replacement"] != "":
                agg[L][0] += elru.replacement_chrf(pe["replacement"], te["replacement"])
                agg[L][1] += 1
    return {L: (v[0] / v[1] if v[1] else 0.0, v[1], v[2]) for L, v in agg.items()}


def del_precision(pred_map, truth_map, lang_map):
    """precision of predicted deletions (rep=='') vs true deletions, per lang + overall."""
    out = collections.defaultdict(lambda: [0, 0])  # [hits, pred]
    for _id, true in truth_map.items():
        L = lang_map[_id]; pred = pred_map.get(_id, [])
        tdels = [(e["start"], e["end"]) for e in true if e["replacement"] == ""]
        for pe in pred:
            if pe["replacement"] != "":
                continue
            out[L][1] += 1
            if any(not (pe["end"] <= a or b <= pe["start"]) for a, b in tdels):
                out[L][0] += 1
    tot = [sum(v[0] for v in out.values()), sum(v[1] for v in out.values())]
    return {L: (v[0], v[1]) for L, v in out.items()}, tot


def rung_edits(P):
    rows = P["rows"]; rbi = P["rows_by_id"]; gbi = P["gbi"]
    trs = P["trs"]; stf = P["stf"]; idfold = P["idfold"]
    cache_n2 = P["cache_n2"]; it_cache_n1 = P["it_cache_n1"]
    _t, n2_nn_e, _b, n2_ne_e = PF.de_en_edits(rows, cache_n2)
    _o, it_n1_nn, _n, it_n1_ne = PF.select_it(rows, it_cache_n1, [PF.N1_SPINE], PF.N1_GATE_GRID)
    nn_e, ne_e = PF.combine(rows, rbi, n2_nn_e, n2_ne_e, it_n1_nn, it_n1_ne)
    nn, ne = PF.apply_vote(nn_e, ne_e, rbi, gbi, trs, stf, idfold, SHIP_VOTE_LANGS, SHIP_VOTE_LANGS, 0.60, 0.40)
    return nn, ne


def run(label, patch_enh, use_del):
    if patch_enh:
        PF.Transducer = transducer_p2.Transducer
        pipeline.Transducer = transducer_p2.Transducer
    else:
        import transducer as Tb
        PF.Transducer = Tb.Transducer
        pipeline.Transducer = Tb.Transducer
    m3_ext.USE_DEL = use_del
    t0 = time.time()
    P = PF.prepare(verbose=False)
    rows = P["rows"]
    truth = {R["id"]: R["truth"] for R in rows}
    lang = {R["id"]: R["lang"] for R in rows}
    nn, ne = rung_edits(P)
    ne_s, ne_d = score_edits(rows, ne)
    nn_s, nn_d = score_edits(rows, nn)
    mc = matched_chrf(nn, truth, lang)      # matched chrf on non-nested (shipping) edits
    dp, dtot = del_precision(nn, truth, lang)
    dt = time.time() - t0
    print(f"\n===== {label}  (USE_DEL={use_del}) [{dt:.0f}s] =====")
    print(f"  NESTED overall={ne_s:.4f}  de={ne_d['de']['lang_score']:.4f} en={ne_d['en']['lang_score']:.4f} it={ne_d['it']['lang_score']:.4f}")
    print(f"  it edited={ne_d['it']['edited_mean']:.4f}  de edited={ne_d['de']['edited_mean']:.4f}")
    print(f"  matched-span chrf: " + ", ".join(f"{L}={mc[L][0]:.4f}(n={mc[L][1]})" for L in ('de','en','it')))
    if dtot[1] > 0:
        print(f"  predicted deletions: {dtot[1]} precision={dtot[0]}/{dtot[1]}={dtot[0]/max(1,dtot[1]):.3f}  per-lang " +
              ", ".join(f"{L}={dp[L][0]}/{dp[L][1]}" for L in dp))
    return dict(nested=ne_s, ne_d={L: {k: v for k, v in ne_d[L].items()} for L in ('de','en','it')},
                matched_chrf={L: mc[L] for L in mc}, del_prec=[dtot[0], dtot[1]])


if __name__ == "__main__":
    rb = run("BASELINE", patch_enh=False, use_del=False)
    re_ = run("IT-ONLY ENHANCED", patch_enh=True, use_del=False)
    rd = run("IT-ONLY ENHANCED + USE_DEL", patch_enh=True, use_del=True)
    print("\n================ SUMMARY ================")
    print(f"  nested:            base={rb['nested']:.4f}  enh={re_['nested']:.4f}  enh+del={rd['nested']:.4f}")
    print(f"  it matched chrf:   base={rb['matched_chrf']['it'][0]:.4f}  enh={re_['matched_chrf']['it'][0]:.4f}")
    print(f"  de matched chrf:   base={rb['matched_chrf']['de'][0]:.4f}  enh={re_['matched_chrf']['de'][0]:.4f}")
    print(f"  del precision (enh+del): {rd['del_prec'][0]}/{rd['del_prec'][1]}")
    json.dump(dict(baseline=rb, enhanced=re_, enhanced_del=rd), open(os.path.join(HERE, "p2_eval_c.json"), "w"), indent=2, default=float)
    print("wrote p2_eval_c.json")

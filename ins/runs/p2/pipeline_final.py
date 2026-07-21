"""N3 FINAL COMPOSER -- integrate N1 (Italian gated-NP) + N2 (German markrun + masc_only)
onto the M4 base, joint-retune all operating points, report the full ablation ladder, and
emit two full-train submissions (CV-optimal + robust).

Composition
-----------
  * Detector/transducer/stores are IDENTICAL across M4/N1/N2 (the enhancement plug-ins only
    touch ASSEMBLY, never the LGBM detector features or the STORE_BUILDERS), so folds are fit
    ONCE (leak-free OOF probs + per-fold transducer + per-fold stores).
  * de/en  : base threshold-merge via pipeline.build_edits.  With n2_ext registered the de
             markrun candidate generator (admitted through M2's fem-gate span_scorer) and the
             masc_only replacement hook fire automatically inside build_edits -> the N2 gain.
             de/en per-language thresholds selected nested (fold-k from other 4) + non-nested.
  * it     : N1's article-anchored NP-span assembly -- base contiguous merge at a spine
             threshold UNION a cross-fit LGBM-gated NP generator (base spans keep priority),
             whole-NP transduction + src-first slash reorder.  (spine, gate) selected nested.
  * group  : inference-time group-consistency vote (de+en, hi.60/lo.40).  group-vote-it is
             RE-TESTED here (plain + doc-prior-gated) and shipped only if it nets positive.

Everything learned from train.csv at runtime; leak-free per-fold; canonical elru + folds only;
reports BOTH nested (honest headline) and non-nested.  Modes: ladder | ship | all.

Usage: cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/N3/pipeline_final.py [mode]
"""
import os, sys, json, time, collections, re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if not os.path.exists(os.path.join(ROOT, "dataset", "train.csv")):
    ROOT = os.path.expanduser("~/insled")
for p in (os.path.join(ROOT, "runs", "M4"), os.path.join(ROOT, "runs", "N2"),
          os.path.join(ROOT, "runs", "N1"), os.path.join(ROOT, "solution"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
import pipeline, m4_ext, m2_ext, m3_ext, n2_ext
from transducer import Transducer
import elru
from run_m4 import (base_cache, base_select, group_consistency, score_edits, fp_counts,
                    per_type_recall, print_detail, LOSSMAP, SHIP_VOTE_LANGS, TRAIN_EDIT_RATE,
                    iou, span_type)
from run_n1 import learn_tab, group_ctx, np_cands, reorder, it_gate_scores, GATE_PARAMS

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")

# ---- it operating-point grids (joint retune DOF) ----
IT_SPINE_GRID = [0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.51]
IT_GATE_GRID = [1.01, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4]
N1_SPINE = 0.45           # N1's fixed spine (for the reproduction rungs)
N1_GATE_GRID = [1.01, 0.8, 0.7, 0.6, 0.5, 0.4]


# ======================================================================
#  Phase 1: leak-free fold fitting (register n2_ext so stores carry everything)
# ======================================================================
def fit_folds(train, verbose=True):
    t0 = time.time()
    rows = pipeline.build_rows(train, labeled=True)
    idfold = {R["id"]: R["fold"] for R in rows}
    row_proba, trs, stf = {}, {}, {}
    for k in range(5):
        tr_rows = [R for R in rows if R["fold"] != k]
        va_rows = [R for R in rows if R["fold"] == k]
        trdf = train[train.fold != k]
        st = {}
        for b in pipeline.STORE_BUILDERS:
            b(trdf, st)
        det = pipeline.Detector().fit(tr_rows, st)
        for _id, (tk, pr) in det.token_probs(va_rows).items():
            row_proba[_id] = pr
        trs[k] = Transducer().fit(trdf)
        stf[k] = st
        if verbose:
            print(f"[fold {k}] fit ({time.time()-t0:.0f}s)", flush=True)
    return rows, idfold, row_proba, trs, stf


# ======================================================================
#  Italian NP assembly (N1), spine-parametric for the joint retune
# ======================================================================
def assemble_it(tk, text, pr, spine, gate, gate_scores, T, st):
    """base contiguous-merge(spine) UNION gated NP (base priority), transduce, reorder."""
    n = len(tk); spans = []; i = 0
    while i < n:
        if pr[i] >= spine:
            j = i
            while j + 1 < n and pr[j + 1] >= spine:
                j += 1
            spans.append((tk[i][0], tk[j][1], float(np.mean(pr[i:j + 1])) + 1.0))  # +1 => base priority
            i = j + 1
        else:
            i += 1
    for (a, b, p) in gate_scores:
        if p >= gate:
            spans.append((a, b, float(p)))
    spans.sort(key=lambda s: -s[2]); chosen = []; occ = []
    for (a, b, sc) in spans:
        if any(not (b <= x or y <= a) for (x, y) in occ):
            continue
        chosen.append((a, b)); occ.append((a, b))
    edits = []
    for (a, b) in chosen:
        src = text[a:b]
        ctx = {"text": text, "start": a, "end": b, "lang": "it", "tokens": tk, "stores": st}
        rep = None
        for hook in pipeline.REPLACEMENT_HOOKS:
            r = hook("it", src, ctx, st)
            if r is not None:
                rep = r; break
        if rep is None:
            rep = T.predict("it", src, ctx) or src
        if len(src.split()) == 1:
            rep = reorder(src, rep)
        edits.append({"start": a, "end": b, "replacement": rep[:160]})
    edits.sort(key=lambda e: e["start"]); edits = edits[:8]
    if not elru.validate_edits(edits, len(text)):
        edits = pipeline._repair(edits, len(text))
    return edits


def build_it_cache(rows, idfold, row_proba, trs, stf, gate_scores, spine_grid, gate_grid):
    """{id: {(spine,gate): edits}} for it rows over the joint grid."""
    itrows = [R for R in rows if R["lang"] == "it"]
    cache = {}
    for R in itrows:
        k = idfold[R["id"]]; T = trs[k]; st = stf[k]; tk = R["tk"]; text = R["text"]; pr = row_proba[R["id"]]
        gs = [(ab[0], ab[1], p) for (ab, p) in gate_scores[R["id"]]]
        cache[R["id"]] = {(sp, g): assemble_it(tk, text, pr, sp, g, gs, T, st)
                          for sp in spine_grid for g in gate_grid}
    return cache


def select_it(rows, itcache, spine_grid, gate_grid):
    """nested (fold-k op from other 4) + non-nested (all-OOF) over the (spine,gate) grid."""
    itrows = [R for R in rows if R["lang"] == "it"]
    truth = {R["id"]: R["truth"] for R in itrows}
    ops = [(sp, g) for sp in spine_grid for g in gate_grid]

    def sc(op, ids):
        sub = {i: itcache[i][op] for i in ids}
        _s, d = elru.elru(sub, {i: truth[i] for i in ids}, {i: "it" for i in ids}, detail=True)
        return d["it"]["lang_score"]

    allids = set(truth)
    nn_op = max(ops, key=lambda op: sc(op, allids))
    nn_edits = {i: itcache[i][nn_op] for i in truth}
    nby = {}; nest = {}
    for k in range(5):
        other = set(R["id"] for R in itrows if R["fold"] != k)
        bop = max(ops, key=lambda op: sc(op, other))
        nby[k] = bop
        for R in [r for r in itrows if r["fold"] == k]:
            nest[R["id"]] = itcache[R["id"]][bop]
    return nn_op, nn_edits, nby, nest


# ======================================================================
#  de/en base-cache selection + language combine
# ======================================================================
def de_en_edits(rows, cache):
    """base_select on `cache`; returns (nn_thr, nn_edits, nby, ne_edits) for ALL langs."""
    return base_select(rows, cache)


def combine(rows, rows_by_id, base_nn_e, base_ne_e, it_nn_e=None, it_ne_e=None):
    """Take de/en (and default it) from base edits; optionally override it rows."""
    nn = {i: base_nn_e[i] for i in base_nn_e}
    ne = {i: base_ne_e[i] for i in base_ne_e}
    if it_nn_e is not None:
        for i in it_nn_e:
            nn[i] = it_nn_e[i]
    if it_ne_e is not None:
        for i in it_ne_e:
            ne[i] = it_ne_e[i]
    return nn, ne


def apply_vote(nn_e, ne_e, rows_by_id, gbi, trs, stf, idfold, vote_langs, drop_langs,
               hi=0.60, lo=0.40, do_conv=False, conv_langs=None):
    nn = group_consistency({i: nn_e[i] for i in nn_e}, rows_by_id, gbi, trs, stf, idfold,
                           hi=hi, lo=lo, vote_langs=vote_langs, drop_langs=drop_langs,
                           do_conv=do_conv, conv_langs=conv_langs)
    ne = group_consistency({i: ne_e[i] for i in ne_e}, rows_by_id, gbi, trs, stf, idfold,
                           hi=hi, lo=lo, vote_langs=vote_langs, drop_langs=drop_langs,
                           do_conv=do_conv, conv_langs=conv_langs)
    return nn, ne


# ======================================================================
#  doc-prior (leak-free per fold): P(edited) at group and row level
# ======================================================================
def learn_doc_prior(trdf, lang="it"):
    """group-level and row-length edited-rate priors from a fold-train frame (leak-free)."""
    g_ed = collections.Counter(); g_tot = collections.Counter()
    for r in trdf.itertuples():
        if r.language != lang:
            continue
        edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
        g_ed[r.document_group] += 1 if len(edits) > 0 else 0
        g_tot[r.document_group] += 1
    glob = (sum(g_ed.values()) + 1.0) / (sum(g_tot.values()) + 2.0)
    grate = {g: (g_ed[g] + 2.0 * glob) / (g_tot[g] + 2.0) for g in g_tot}
    return dict(grate=grate, glob=glob)


# ======================================================================
#  reporting helpers
# ======================================================================
def summarize(rows, nn, ne):
    nn_s, nn_d = score_edits(rows, nn)
    ne_s, ne_d = score_edits(rows, ne)
    return nn_s, nn_d, ne_s, ne_d


def per_lang(det):
    return {L: round(det[L]["lang_score"], 4) for L in LANGS}


# ======================================================================
#  prepare (fit once + build all caches)
# ======================================================================
def load_train():
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    return train


def prepare(verbose=True):
    # register the FULL N2 stack (composes m4_ext + N2 de plug-ins) BEFORE fitting folds,
    # so per-fold stores carry span_scorer + collapse memories + the transducer stash used
    # by the markrun generator's learned mark-suffix set.
    n2_ext.register(pipeline)
    train = load_train()
    gbi = {r.id: r.document_group for r in train.itertuples()}
    rows, idfold, row_proba, trs, stf = fit_folds(train, verbose)
    rows_by_id = {R["id"]: R for R in rows}

    # it learned tables + group context + cross-fit NP gate (leak-free per fold)
    tabs = {k: learn_tab(train[train.fold != k]) for k in range(5)}
    gctxs = {k: group_ctx(train[train.fold != k]) for k in range(5)}
    gate_scores = it_gate_scores(rows, idfold, row_proba, tabs, gctxs, gbi)
    doc_priors = {k: learn_doc_prior(train[train.fold != k], "it") for k in range(5)}

    # de/en base cache WITH n2 plug-ins active (de markrun + masc_only fire in build_edits)
    cache_n2 = base_cache(rows, idfold, row_proba, trs, stf)
    # de base cache WITHOUT n2 plug-ins (M4 de) -- re-register m4_ext, rebuild, then restore n2
    m4_ext.register(pipeline)
    cache_m4 = base_cache(rows, idfold, row_proba, trs, stf)
    n2_ext.register(pipeline)   # restore full stack for all subsequent assembly

    # it caches: joint (spine x gate) and N1 (fixed spine)
    it_cache_joint = build_it_cache(rows, idfold, row_proba, trs, stf, gate_scores,
                                    IT_SPINE_GRID, IT_GATE_GRID)
    it_cache_n1 = build_it_cache(rows, idfold, row_proba, trs, stf, gate_scores,
                                 [N1_SPINE], N1_GATE_GRID)
    if verbose:
        print("[prepare] caches built", flush=True)
    return dict(train=train, gbi=gbi, rows=rows, idfold=idfold, row_proba=row_proba,
                trs=trs, stf=stf, rows_by_id=rows_by_id, tabs=tabs, gctxs=gctxs,
                gate_scores=gate_scores, doc_priors=doc_priors,
                cache_n2=cache_n2, cache_m4=cache_m4,
                it_cache_joint=it_cache_joint, it_cache_n1=it_cache_n1)


# ======================================================================
#  ABLATION LADDER + JOINT RETUNE
# ======================================================================
def ladder(P=None):
    t0 = time.time()
    if P is None:
        P = prepare()
    rows = P["rows"]; idfold = P["idfold"]; gbi = P["gbi"]; trs = P["trs"]; stf = P["stf"]
    rbi = P["rows_by_id"]; row_proba = P["row_proba"]
    cache_n2 = P["cache_n2"]; cache_m4 = P["cache_m4"]
    it_cache_joint = P["it_cache_joint"]; it_cache_n1 = P["it_cache_n1"]

    # base_selects on both caches
    m4_nn_thr, m4_nn_e, m4_nby, m4_ne_e = de_en_edits(rows, cache_m4)
    n2_nn_thr, n2_nn_e, n2_nby, n2_ne_e = de_en_edits(rows, cache_n2)

    # it selections
    it_n1_op, it_n1_nn, it_n1_nby, it_n1_ne = select_it(rows, it_cache_n1, [N1_SPINE], N1_GATE_GRID)
    it_j_op, it_j_nn, it_j_nby, it_j_ne = select_it(rows, it_cache_joint, IT_SPINE_GRID, IT_GATE_GRID)

    VL = SHIP_VOTE_LANGS   # {de, en}
    results = collections.OrderedDict()

    def rung(tag, base_nn, base_ne, it_nn=None, it_ne=None, vote_langs=VL, drop_langs=VL,
             hi=0.60, lo=0.40, show=False):
        nn_e, ne_e = combine(rows, rbi, base_nn, base_ne, it_nn, it_ne)
        nn, ne = apply_vote(nn_e, ne_e, rbi, gbi, trs, stf, idfold, vote_langs, drop_langs, hi, lo)
        nn_s, nn_d, ne_s, ne_d = summarize(rows, nn, ne)
        results[tag] = dict(nested=round(ne_s, 4), nonnested=round(nn_s, 4),
                            nested_lang=per_lang(ne_d), nonnested_lang=per_lang(nn_d),
                            nn=nn, ne=ne, nn_d=nn_d, ne_d=ne_d)
        fp = fp_counts(rows, nn)
        print(f"  {tag:26s} nested={ne_s:.4f} nonnest={nn_s:.4f}  "
              f"[de {ne_d['de']['lang_score']:.4f} en {ne_d['en']['lang_score']:.4f} it {ne_d['it']['lang_score']:.4f}]  "
              f"FP de={fp['de'][0]}/{fp['de'][1]} en={fp['en'][0]}/{fp['en'][1]} it={fp['it'][0]}/{fp['it'][1]}")
        if show:
            ptr = per_type_recall(rows, nn)
            for key in sorted(ptr):
                r, nsp = ptr[key]; b = LOSSMAP.get(key)
                print(f"        {key[0]} {key[1]:14s} rec={r:.3f}(n={nsp})" + (f" lm{b:.3f}" if b else ""))
        return results[tag]

    print("\n================ N3 ABLATION LADDER (group-vote de+en, hi.60/lo.40) ================")
    rung("M4 ship", m4_nn_e, m4_ne_e)                                   # de M4, en, it base
    rung("+N1 (it gated-NP)", m4_nn_e, m4_ne_e, it_n1_nn, it_n1_ne)     # de M4, it N1
    rung("+N2 (de markrun+masc)", n2_nn_e, n2_ne_e)                     # de N2, it base
    joint = rung("+N1+N2 (fixed it spine)", n2_nn_e, n2_ne_e, it_n1_nn, it_n1_ne, show=True)
    jointR = rung("joint retune (it spine grid)", n2_nn_e, n2_ne_e, it_j_nn, it_j_ne, show=True)

    print(f"\n  it N1 op(nonnested)={it_n1_op} nby={it_n1_nby}")
    print(f"  it joint op(nonnested)={it_j_op} nby={it_j_nby}")
    print(f"  de/en thr(n2, nonnested)={{'de':{n2_nn_thr['de']},'en':{n2_nn_thr['en']}}}  nby={ { k:{L:n2_nby[k][L] for L in ('de','en')} for k in n2_nby} }")

    # -------- group-vote-it re-test (task item): plain + doc-prior-gated --------
    print("\n---- group-vote-it re-test on the joint base (it NP-gated) ----")
    base_nn, base_ne = combine(rows, rbi, n2_nn_e, n2_ne_e, it_j_nn, it_j_ne)
    # (a) no it vote (reference = joint retune)
    print(f"  no-it-vote (ref)           nested={jointR['nested']:.4f} it={jointR['nested_lang']['it']:.4f}")
    # (b) plain it vote (de+en+it)
    nn_v, ne_v = apply_vote(base_nn, base_ne, rbi, gbi, trs, stf, idfold,
                            {"de", "en", "it"}, {"de", "en", "it"}, 0.60, 0.40)
    s = score_edits(rows, ne_v); n = score_edits(rows, nn_v)
    it_vote_nested = round(s[0], 4)
    print(f"  +it vote (de+en+it)        nested={s[0]:.4f} it={s[1]['it']['lang_score']:.4f}  nonnest={n[0]:.4f}")
    # (c) it vote gated by doc prior (only high-prior it groups get propagation)
    for pth in (0.60, 0.75, 0.90):
        nn_g = _it_vote_docprior(base_nn, rbi, gbi, trs, stf, idfold, P["doc_priors"], pth)
        ne_g = _it_vote_docprior(base_ne, rbi, gbi, trs, stf, idfold, P["doc_priors"], pth)
        sg = score_edits(rows, ne_g)
        print(f"  +it vote docprior>={pth:.2f}    nested={sg[0]:.4f} it={sg[1]['it']['lang_score']:.4f}")

    # -------- group hi/lo: NESTED-validated selection on the FIXED-spine base --------
    # (use the fixed-spine +N1+N2 base = CV-optimal it; hi/lo affects only de+en)
    fs_nn, fs_ne = combine(rows, rbi, n2_nn_e, n2_ne_e, it_n1_nn, it_n1_ne)
    HILO = [(0.60, 0.40), (0.55, 0.45), (0.50, 0.40), (0.60, 0.50), (0.70, 0.40), (0.65, 0.35)]
    print("\n---- de+en group-vote hi/lo sensitivity (fixed-spine base; ref (.60,.40)=0.5503) ----")
    hilo_nonnested = {}
    for hi, lo in HILO:
        nn_h, ne_h = apply_vote(fs_nn, fs_ne, rbi, gbi, trs, stf, idfold, VL, VL, hi, lo)
        sh = score_edits(rows, ne_h)[0]; nh = score_edits(rows, nn_h)[0]
        hilo_nonnested[(hi, lo)] = (nh, sh)
        tag = "  <-- current ship" if (hi, lo) == (0.60, 0.40) else ""
        print(f"  hi={hi:.2f} lo={lo:.2f}   nonnest(all-OOF)={nh:.4f} [non-nested-tuned nested]={sh:.4f}{tag}")
    # HONEST nested hi/lo: fold-k selects (hi,lo) maximizing de+en on the other 4 folds, post-vote
    hilo_by_fold, nested_hilo_edits = _nested_hilo(rows, rbi, gbi, trs, stf, idfold, fs_nn, fs_ne, HILO, VL)
    nh_s = score_edits(rows, nested_hilo_edits)[0]
    print(f"  NESTED-selected hi/lo -> nested={nh_s:.4f}   (per-fold {hilo_by_fold})")
    print(f"  => honest nested hi/lo gain over fixed(.60,.40)={nh_s - results['+N1+N2 (fixed it spine)']['nested']:+.4f} "
          f"(non-nested-tuned optimism was illusory)")

    print("\n================ LADDER SUMMARY (nested) ================")
    for tag, r in results.items():
        print(f"  {tag:28s} nested={r['nested']:.4f}  nonnested={r['nonnested']:.4f}")
    print(f"\n[ladder {time.time()-t0:.0f}s]  CV-optimal nested=0.5503 (fixed-spine additive); "
          f"nested-selected hi/lo={nh_s:.4f} (no honest gain)")

    return dict(P=P, results=results, m4_nn_thr=m4_nn_thr, n2_nn_thr=n2_nn_thr, n2_nby=n2_nby,
                it_j_op=it_j_op, it_j_nby=it_j_nby, it_n1_op=it_n1_op, it_n1_nby=it_n1_nby,
                it_j_nn=it_j_nn, it_j_ne=it_j_ne, it_n1_nn=it_n1_nn, it_n1_ne=it_n1_ne,
                joint=joint, jointR=jointR,
                retune=dict(fixed_spine_nested=joint["nested"], spine_grid_nested=jointR["nested"],
                            it_vote_nested=it_vote_nested, nested_hilo=round(nh_s, 4),
                            nested_hilo_by_fold={str(k): list(v) for k, v in hilo_by_fold.items()}))


def _nested_hilo(rows, rbi, gbi, trs, stf, idfold, fs_nn, fs_ne, HILO, VL):
    """Honest nested selection of the de+en group-vote (hi,lo): for each fold k, pick the
    (hi,lo) that maximizes the post-vote de+en mean lang_score on the OTHER 4 folds, then
    apply it to fold k.  it rows carry through unchanged (it not voted)."""
    # precompute post-vote edits for every (hi,lo) once (non-nested edits base = fs_ne)
    voted = {}
    for hi, lo in HILO:
        _nn, ne = apply_vote(fs_nn, fs_ne, rbi, gbi, trs, stf, idfold, VL, VL, hi, lo)
        voted[(hi, lo)] = ne

    def deen_score(ne_map, ids):
        sub = [rbi[i] for i in ids if rbi[i]["lang"] in ("de", "en")]
        if not sub:
            return 0.0
        pm = {R["id"]: ne_map[R["id"]] for R in sub}
        tm = {R["id"]: R["truth"] for R in sub}
        lm = {R["id"]: R["lang"] for R in sub}
        _s, det = elru.elru(pm, tm, lm, detail=True)
        return (det["de"]["lang_score"] + det["en"]["lang_score"]) / 2.0

    hilo_by_fold = {}
    out = {i: fs_ne[i] for i in fs_ne}   # start from unvoted nested edits; fill de/en per fold
    for k in range(5):
        other = [i for i in fs_ne if idfold[i] != k]
        best = max(HILO, key=lambda hl: deen_score(voted[hl], other))
        hilo_by_fold[k] = best
        for i in fs_ne:
            if idfold[i] == k and rbi[i]["lang"] in ("de", "en"):
                out[i] = voted[best][i]
    return hilo_by_fold, out


def _it_vote_docprior(edits_map, rbi, gbi, trs, stf, idfold, doc_priors, prior_thr):
    """apply de+en vote as usual, PLUS it propagation only for it groups whose leak-free
    edited-row prior (from the row's own fold-out frame) exceeds prior_thr."""
    # de+en normal vote
    out = group_consistency({i: edits_map[i] for i in edits_map}, rbi, gbi, trs, stf, idfold,
                            vote_langs={"de", "en"}, drop_langs={"de", "en"}, do_conv=False)
    # build set of it groups allowed to vote (per-fold prior)
    allow = set()
    for i in edits_map:
        R = rbi[i]
        if R["lang"] != "it":
            continue
        k = idfold[i]; grate = doc_priors[k]["grate"]; glob = doc_priors[k]["glob"]
        if grate.get(gbi[i], glob) >= prior_thr:
            allow.add(gbi[i])
    # restrict it vote to allowed groups by temporarily filtering group membership
    it_ids = [i for i in edits_map if rbi[i]["lang"] == "it" and gbi[i] in allow]
    if it_ids:
        sub = group_consistency({i: out[i] for i in it_ids}, rbi, gbi, trs, stf, idfold,
                                vote_langs={"it"}, drop_langs={"it"}, do_conv=False)
        for i in it_ids:
            out[i] = sub[i]
    return out


# ======================================================================
#  SHIP: full-train fit -> two submissions + OOF deliverables + report
# ======================================================================
def _full_train_artifacts(train):
    stores_full = {}
    for b in pipeline.STORE_BUILDERS:
        b(train, stores_full)
    all_rows = pipeline.build_rows(train, labeled=True)
    det_full = pipeline.Detector().fit(all_rows, stores_full)
    trd_full = Transducer().fit(train)
    return stores_full, all_rows, det_full, trd_full


def _fit_full_it_gate(train, all_rows, det_full, gbi_tr):
    import lightgbm as lgb
    tab_full = learn_tab(train); gc_full = group_ctx(train)
    itrows_tr = [R for R in all_rows if R["lang"] == "it"]
    tp_tr = det_full.token_probs(itrows_tr)
    Xtr, ytr = [], []
    for R in itrows_tr:
        pr = tp_tr[R["id"]][1]
        cs = np_cands(R["tk"], R["text"], gbi_tr[R["id"]], pr, tab_full, gc_full)
        for (a, b, f) in cs:
            best = max((max(0, min(b, te) - max(a, ts)) / (max(b, te) - min(a, ts))
                        for (ts, te, rep) in R["spans"] if rep != "" and max(b, te) > min(a, ts)), default=0.0)
            Xtr.append(f); ytr.append(1 if best >= 0.5 else 0)
    gate_model = lgb.LGBMClassifier(**GATE_PARAMS)
    gate_model.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, np.int32))
    return gate_model, tab_full, gc_full


def _build_submission(test, det_full, trd_full, stores_full, gate_model, tab_full, gc_full,
                      de_thr, en_thr, it_spine, it_gate):
    gbi_te = {r.id: r.document_group for r in test.itertuples()}
    thr_by_lang = {"de": de_thr, "en": en_thr}
    test_rows = pipeline.build_rows(test, labeled=False)
    tp_test = det_full.token_probs(test_rows)
    sub = {}
    for R in test_rows:
        tk, pr = tp_test[R["id"]]; lang = R["lang"]; text = R["text"]
        if lang == "it":
            cs = np_cands(tk, text, gbi_te[R["id"]], pr, tab_full, gc_full)
            gs = []
            if cs:
                pv = gate_model.predict_proba(np.asarray([c[2] for c in cs], np.float32))[:, 1]
                gs = [(c[0], c[1], float(p)) for c, p in zip(cs, pv)]
            sub[R["id"]] = assemble_it(tk, text, pr, it_spine, it_gate, gs, trd_full, stores_full)
        else:
            sub[R["id"]] = pipeline.build_edits(R["id"], text, lang, tk, pr, thr_by_lang[lang],
                                                trd_full, stores_full)
    # group-vote de+en
    test_by_id = {R["id"]: R for R in test_rows}
    idf = {i: 0 for i in sub}
    sub = group_consistency(sub, test_by_id, gbi_te, {0: trd_full}, {0: stores_full}, idf,
                            vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    return sub, gbi_te


def _validate_and_rates(sub, test):
    assert len(sub) == len(test) == 445, (len(sub), len(test))
    assert set(sub) == set(test.id), "id mismatch"
    tl = {r.id: len(r.text) for r in test.itertuples()}
    bad = [i for i in sub if not elru.validate_edits(sub[i], tl[i])]
    assert not bad, f"invalid rows: {bad[:5]}"
    lang_te = {r.id: r.language for r in test.itertuples()}
    edn = collections.Counter(); totn = collections.Counter()
    for i in sub:
        L = lang_te[i]; totn[L] += 1
        if sub[i]:
            edn[L] += 1
    rates = {}
    for L in LANGS:
        r_sub = edn[L] / max(totn[L], 1); ratio = r_sub / TRAIN_EDIT_RATE[L]
        rates[L] = dict(edited=edn[L], total=totn[L], frac=round(r_sub, 3),
                        train=TRAIN_EDIT_RATE[L], ratio=round(ratio, 2),
                        flag=not (0.45 <= ratio <= 1.80))
    return rates


def ship(P=None, L=None):
    t0 = time.time()
    if L is None:
        L = ladder(P)
    P = L["P"]; rows = P["rows"]; train = P["train"]
    rbi = P["rows_by_id"]; gbi = P["gbi"]; trs = P["trs"]; stf = P["stf"]; idfold = P["idfold"]

    # ---- chosen CV-optimal operating points ----
    de_thr = float(L["n2_nn_thr"]["de"])
    en_thr = float(L["n2_nn_thr"]["en"])
    # CV-optimal it = FIXED spine (beats spine-grid on nested); non-nested all-OOF op = (0.45, 0.8)
    it_spine, it_gate = L["it_n1_op"]
    it_spine = float(it_spine); it_gate = float(it_gate)
    # robust de threshold (N2 recommendation: ~0.10-0.11 for test edited-ratio ~1.29)
    de_thr_robust = 0.11

    # ---- OOF headline (nested + non-nested) for the shipped FIXED-spine joint config ----
    jr = L["joint"]
    nn_ship, ne_ship = jr["nn"], jr["ne"]
    nn_s, nn_d = score_edits(rows, nn_ship)
    ne_s, ne_d = score_edits(rows, ne_ship)
    fp = fp_counts(rows, nn_ship)

    print("\n================ N3 SHIP CONFIG (joint retune) ================")
    print_detail("NON-NESTED (all-OOF op)", nn_s, nn_d)
    print(f"  ops: de_thr={de_thr} en_thr={en_thr} it_spine={it_spine} it_gate={it_gate}")
    print_detail("NESTED (honest headline)", ne_s, ne_d)
    print("unchanged-row FPs (nonnested): " + ", ".join(f"{Lg}={fp[Lg][0]}/{fp[Lg][1]}" for Lg in LANGS))

    # ---- canonical scorer self-check on OOF (both non-nested and nested shipped edits) ----
    oof_df = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(nn_ship[R["id"]], ensure_ascii=False)} for R in rows])
    chk, _ = elru.score_frames(oof_df, train[["id", "language", "edits_json"]])
    oof_nested_df = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(ne_ship[R["id"]], ensure_ascii=False)} for R in rows])
    chk_ne, _ = elru.score_frames(oof_nested_df, train[["id", "language", "edits_json"]])
    print(f"canonical elru.score_frames (OOF, non-nested) = {chk:.4f}   (nested) = {chk_ne:.4f}")
    oof_df.to_csv(os.path.join(HERE, "oof_edits_final.csv"), index=False)

    # ---- ROBUST variant OOF ELRU (de FIXED at de_thr_robust; en/it at their ops; vote) ----
    cache_n2 = P["cache_n2"]
    rob_nn = {}
    for R in rows:
        i = R["id"]; Lg = R["lang"]
        if Lg == "de":
            rob_nn[i] = cache_n2[i][de_thr_robust]
        elif Lg == "en":
            rob_nn[i] = cache_n2[i][en_thr]
        else:
            rob_nn[i] = nn_ship[i]   # it already = non-nested (0.45,0.8), pre-vote is same (it not voted)
    rob_voted = group_consistency({i: rob_nn[i] for i in rob_nn}, rbi, gbi, trs, stf, idfold,
                                  vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    rob_s, rob_d = score_edits(rows, rob_voted)
    # robust NESTED: de FIXED a-priori at de_thr_robust (no selection), en nested per-fold, it fixed op
    n2_nby = L["n2_nby"]
    rob_ne = {}
    for R in rows:
        i = R["id"]; Lg = R["lang"]; k = idfold[i]
        if Lg == "de":
            rob_ne[i] = cache_n2[i][de_thr_robust]
        elif Lg == "en":
            rob_ne[i] = cache_n2[i][n2_nby[k]["en"]]
        else:
            rob_ne[i] = ne_ship[i]
    rob_ne_voted = group_consistency({i: rob_ne[i] for i in rob_ne}, rbi, gbi, trs, stf, idfold,
                                     vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    rob_ne_s, rob_ne_d = score_edits(rows, rob_ne_voted)
    print(f"ROBUST variant OOF ELRU (de@{de_thr_robust} fixed): non-nested-op={rob_s:.4f} nested={rob_ne_s:.4f}  "
          f"[de {rob_ne_d['de']['lang_score']:.4f} en {rob_ne_d['en']['lang_score']:.4f} it {rob_ne_d['it']['lang_score']:.4f}]")

    # ---- FULL-TRAIN fit ----
    test = pd.read_csv(os.path.join(ROOT, "dataset", "test.csv"))
    gbi_tr = {r.id: r.document_group for r in train.itertuples()}
    stores_full, all_rows, det_full, trd_full = _full_train_artifacts(train)
    gate_model, tab_full, gc_full = _fit_full_it_gate(train, all_rows, det_full, gbi_tr)

    # ---- submission v3 (CV-optimal) ----
    sub, _ = _build_submission(test, det_full, trd_full, stores_full, gate_model, tab_full, gc_full,
                               de_thr, en_thr, it_spine, it_gate)
    rates = _validate_and_rates(sub, test)
    pd.DataFrame([{"id": i, "edits_json": json.dumps(sub[i], ensure_ascii=False)} for i in test.id]
                 ).to_csv(os.path.join(HERE, "submission_v3.csv"), index=False)

    # ---- submission v3 robust (higher de threshold) ----
    subR, _ = _build_submission(test, det_full, trd_full, stores_full, gate_model, tab_full, gc_full,
                                de_thr_robust, en_thr, it_spine, it_gate)
    ratesR = _validate_and_rates(subR, test)
    pd.DataFrame([{"id": i, "edits_json": json.dumps(subR[i], ensure_ascii=False)} for i in test.id]
                 ).to_csv(os.path.join(HERE, "submission_v3_robust.csv"), index=False)

    print("\nsubmission_v3.csv edited-row fractions vs train:")
    for Lg in LANGS:
        r = rates[Lg]
        print(f"  {Lg}: sub={r['frac']:.3f} ({r['edited']}/{r['total']}) train={r['train']:.3f} ratio={r['ratio']:.2f}"
              + ("  <<FLAG" if r["flag"] else ""))
    print("submission_v3_robust.csv (de_thr={}) edited-row fractions:".format(de_thr_robust))
    for Lg in LANGS:
        r = ratesR[Lg]
        print(f"  {Lg}: sub={r['frac']:.3f} ({r['edited']}/{r['total']}) ratio={r['ratio']:.2f}"
              + ("  <<FLAG" if r["flag"] else ""))

    n_edit = sum(1 for i in sub if sub[i]); n_editR = sum(1 for i in subR if subR[i])

    report = dict(
        config="N3 joint: M4 base + N2(de markrun+masc_only) + N1(it gated-NP) + group-vote[de,en]",
        headline_nested_elru=round(ne_s, 4), nonnested_elru=round(nn_s, 4),
        canonical_oof_check_nonnested=round(chk, 4), canonical_oof_check_nested=round(chk_ne, 4),
        beats_ai_baseline_nested=bool(ne_s > 0.56), ai_baseline=0.56,
        base_m4_nested=0.5423, base_m4_nonnested=0.5517,
        delta_vs_m4_nested=round(ne_s - 0.5423, 4), delta_vs_m4_nonnested=round(nn_s - 0.5517, 4),
        ladder={tag: {"nested": r["nested"], "nonnested": r["nonnested"],
                      "nested_lang": r["nested_lang"], "nonnested_lang": r["nonnested_lang"]}
                for tag, r in L["results"].items()},
        ops=dict(de_thr=de_thr, en_thr=en_thr, it_spine=it_spine, it_gate=it_gate,
                 de_thr_robust=de_thr_robust, group_vote=list(SHIP_VOTE_LANGS), group_hi=0.60, group_lo=0.40),
        it_op_shipped=[it_spine, it_gate], it_op_by_fold_fixedspine={str(k): list(v) for k, v in L["it_n1_nby"].items()},
        retune_findings=dict(
            fixed_spine_nested=L["retune"]["fixed_spine_nested"],
            it_spine_grid_nested=L["retune"]["spine_grid_nested"],
            group_vote_it_nested=L["retune"]["it_vote_nested"],
            nested_selected_hilo_nested=L["retune"]["nested_hilo"],
            nested_hilo_by_fold=L["retune"]["nested_hilo_by_fold"],
            conclusion=("Every added tuning DOF fails to lift the HONEST nested number: it-spine-grid "
                        "overfits (0.5490<0.5503); group-vote-it net-negative (0.5388); nested-selected "
                        "hi/lo ties fixed .60/.40 (0.5503) -- the non-nested-tuned hi/lo gain was selection "
                        "optimism. Minimal-DOF additive N1+N2 is the honest optimum.")),
        cv_optimal_variant=dict(nested=round(ne_s, 4), nonnested=round(nn_s, 4),
                                nested_lang={Lg: round(ne_d[Lg]["lang_score"], 4) for Lg in LANGS}),
        robust_variant=dict(de_thr=de_thr_robust, oof_elru_nonnested_op=round(rob_s, 4),
                            oof_elru_nested=round(rob_ne_s, 4),
                            nested_lang={Lg: round(rob_ne_d[Lg]["lang_score"], 4) for Lg in LANGS},
                            note=("de fixed a-priori at 0.11 for transfer safety (chosen from N2 edited-ratio "
                                  "analysis, NOT by maximizing CV), so de generalization carries no selection "
                                  "optimism; nested here fixes de, nests en, fixes it op")),
        nested_detail={Lg: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ne_d[Lg].items()} for Lg in LANGS},
        nonnested_detail={Lg: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in nn_d[Lg].items()} for Lg in LANGS},
        unchanged_fp={Lg: list(fp[Lg]) for Lg in LANGS},
        submission_v3=dict(rows=len(sub), edited=n_edit, edit_rates={Lg: rates[Lg] for Lg in LANGS}),
        submission_v3_robust=dict(rows=len(subR), edited=n_editR, de_thr=de_thr_robust,
                                  edit_rates={Lg: ratesR[Lg] for Lg in LANGS}))
    json.dump(report, open(os.path.join(HERE, "cv_report_final.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote submission_v3.csv ({n_edit}/445), submission_v3_robust.csv ({n_editR}/445), "
          f"oof_edits_final.csv, cv_report_final.json")
    print(f"HEADLINE nested ELRU = {ne_s:.4f}  (AI baseline 0.56: {'BEAT' if ne_s > 0.56 else 'below'})  [{time.time()-t0:.0f}s]")
    return report


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ladder"
    if mode == "ladder":
        ladder()
    elif mode == "ship":
        ship()
    elif mode == "all":
        L = ladder()
        ship(L=L)

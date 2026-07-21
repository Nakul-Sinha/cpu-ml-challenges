"""N2 leak-free measurement harness on the M4 ship base.

fit_all() runs the per-fold detector/transducer/stores loop ONCE (base m4 registration;
my gen/hook do NOT change detector features), and additionally produces en OOF probs
with slash/mark features masked (for the EN shift stress test).  All configs then reuse
those OOF artifacts and only re-run the cheap assembly (build_edits) under different
SPAN_CANDIDATE_GENERATORS / REPLACEMENT_HOOKS globals.

Sections (argv): compare | robust | enshift | all  (default all)
"""
import os, sys, json, time, collections
M4 = os.path.expanduser("~/insled/runs/M4")
N2 = os.path.expanduser("~/insled/runs/N2")
sys.path.insert(0, M4)
sys.path.insert(0, N2)
sys.path.insert(0, os.path.expanduser("~/insled/solution"))
import numpy as np, pandas as pd
import pipeline, m2_ext, m3_ext, m4_ext, run_m4, n2_ext
import elru
from transducer import Transducer

LANGS = pipeline.LANGS
MARKS = set(":*∗/")
SHIP_VOTE = {"de", "en"}
TRAIN_EDIT_RATE = {"de": 0.577, "en": 0.470, "it": 0.704}
LOSSMAP = run_m4.LOSSMAP

# slash/mark-derived detector feature names to zero for the EN shift stress test
SLASH_MARK_FEATS = {"has_special", "spat_rate", "specsuf_rate", "specsuf_len",
                    "specsuf_alpha", "specsuf_short", "spat_sup", "spc_id",
                    "specsuf_id", "mid_/", "mid_*", "mid_:", "mid_∗", "maxchar_rate"}


def load_train():
    ROOT = pipeline.ROOT
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    return train, ROOT


def fit_all(train, verbose=True):
    """One fold loop: OOF probs (normal) + en masked probs + per-fold transducer/stores."""
    t0 = time.time()
    rows = pipeline.build_rows(train, labeled=True)
    idfold = {R["id"]: R["fold"] for R in rows}
    row_proba, row_proba_en_masked = {}, {}
    transducers, stores_by_fold = {}, {}
    mask_idx = None
    for k in range(5):
        tr_rows = [R for R in rows if R["fold"] != k]
        va_rows = [R for R in rows if R["fold"] == k]
        tr_df = train[train.fold != k]
        stores = {}
        for b in pipeline.STORE_BUILDERS:
            b(tr_df, stores)
        det = pipeline.Detector().fit(tr_rows, stores)
        tp = det.token_probs(va_rows)
        for _id, (tk, pr) in tp.items():
            row_proba[_id] = pr
        # en masked probs: refeaturize en val rows, zero slash/mark cols, re-predict
        if mask_idx is None:
            mask_idx = [i for i, nm in enumerate(pipeline.FEAT_NAMES) if nm in SLASH_MARK_FEATS]
        en_va = [R for R in va_rows if R["lang"] == "en"]
        if en_va:
            X, _ = pipeline.featurize(en_va, det.lex)
            X = np.array(X, dtype=np.float32)
            for c in mask_idx:
                X[:, c] = 0.0
            p = det.model.predict_proba(X)[:, 1]
            off = 0
            for R in en_va:
                m = len(R["tk"])
                row_proba_en_masked[R["id"]] = p[off:off + m].tolist()
                off += m
        transducers[k] = pipeline.Transducer().fit(tr_df)
        stores_by_fold[k] = stores
        if verbose:
            print(f"[fold {k}] fit ({time.time()-t0:.0f}s)", flush=True)
    return dict(rows=rows, idfold=idfold, row_proba=row_proba,
                row_proba_en_masked=row_proba_en_masked, transducers=transducers,
                stores_by_fold=stores_by_fold, mask_feats=[pipeline.FEAT_NAMES[i] for i in (mask_idx or [])])


# ---------------------------------------------------------------------------
def base_hooks_with_masc(base_hooks):
    hooks = list(base_hooks)
    try:
        idx = hooks.index(m2_ext.collapse_hook) + 1
    except ValueError:
        idx = len(hooks)
    hooks.insert(idx, n2_ext.masc_only_hook)
    return hooks


def set_config(base_gens, base_hooks, markrun, mascfb):
    pipeline.SPAN_CANDIDATE_GENERATORS = (list(base_gens) + [n2_ext.de_markrun_generator]) if markrun else list(base_gens)
    pipeline.REPLACEMENT_HOOKS = base_hooks_with_masc(base_hooks) if mascfb else list(base_hooks)


def ship_eval(A, group_by_id, rows_by_id):
    """base threshold-merge + per-lang thresholds + group vote[de,en]; nested + nonnested."""
    rows = A["rows"]; idfold = A["idfold"]; row_proba = A["row_proba"]
    trs = A["transducers"]; stf = A["stores_by_fold"]
    bcache = run_m4.base_cache(rows, idfold, row_proba, trs, stf)
    nn_thr, nn_e, nby, ne_e = run_m4.base_select(rows, bcache)
    nn = run_m4.group_consistency({i: nn_e[i] for i in nn_e}, rows_by_id, group_by_id, trs, stf, idfold,
                                  vote_langs=SHIP_VOTE, drop_langs=SHIP_VOTE, do_conv=False)
    ne = run_m4.group_consistency({i: ne_e[i] for i in ne_e}, rows_by_id, group_by_id, trs, stf, idfold,
                                  vote_langs=SHIP_VOTE, drop_langs=SHIP_VOTE, do_conv=False)
    nn_s, nn_d = run_m4.score_edits(rows, nn)
    ne_s, ne_d = run_m4.score_edits(rows, ne)
    fp = run_m4.fp_counts(rows, nn)
    ptr = run_m4.per_type_recall(rows, nn)
    return dict(nn_thr=nn_thr, bcache=bcache, nn_e=nn_e, ne_e=ne_e, nn=nn, ne=ne,
                nn_s=nn_s, ne_s=ne_s, nn_d=nn_d, ne_d=ne_d, fp=fp, ptr=ptr)


def print_ship(tag, r):
    print(f"\n---- {tag} ----")
    print(f"  NESTED  ELRU={r['ne_s']:.4f}   " + " ".join(f"{L}={r['ne_d'][L]['lang_score']:.4f}" for L in LANGS))
    print(f"  NONNEST ELRU={r['nn_s']:.4f}   thr={r['nn_thr']}  " + " ".join(f"{L}={r['nn_d'][L]['lang_score']:.4f}" for L in LANGS))
    print("  FP(nonnest): " + ", ".join(f"{L}={r['fp'][L][0]}/{r['fp'][L][1]}" for L in LANGS))


def print_types(r, only=("de", "en")):
    for key in sorted(r["ptr"]):
        if key[0] not in only:
            continue
        rec, nsp = r["ptr"][key]; b = LOSSMAP.get(key)
        print(f"      {key[0]} {key[1]:14s} rec={rec:.3f}(n={nsp})" + (f"  lm{b:.3f}" if b else ""))


# ======================================================================
def section_compare(A, group_by_id, rows_by_id, base_gens, base_hooks):
    print("\n" + "=" * 72)
    print("SECTION 1: BASE vs +markrun vs +mascfb vs +both  (leak-free, ship path)")
    print("=" * 72)
    configs = [("BASE (m4)", False, False), ("+markrun", True, False),
               ("+mascfb", False, True), ("+both (N2)", True, True)]
    out = {}
    for tag, mr, mf in configs:
        set_config(base_gens, base_hooks, mr, mf)
        r = ship_eval(A, group_by_id, rows_by_id)
        out[tag] = r
        print_ship(tag, r)
        if mr or mf:
            print("    de/en per-type recall:")
            print_types(r)
    b = out["BASE (m4)"]; n = out["+both (N2)"]
    print("\n  DELTA (+both - BASE): "
          f"nested {n['ne_s']-b['ne_s']:+.4f}  nonnested {n['nn_s']-b['nn_s']:+.4f}  "
          f"de-nested {n['ne_d']['de']['lang_score']-b['ne_d']['de']['lang_score']:+.4f}  "
          f"en-nested {n['ne_d']['en']['lang_score']-b['ne_d']['en']['lang_score']:+.4f}")
    return out


# ======================================================================
def section_robust(A, group_by_id, rows_by_id, base_gens, base_hooks, train, markrun, mascfb):
    print("\n" + "=" * 72)
    print(f"SECTION 2: DE THRESHOLD ROBUSTNESS (config markrun={markrun} mascfb={mascfb})")
    print("=" * 72)
    set_config(base_gens, base_hooks, markrun, mascfb)
    rows = A["rows"]; idfold = A["idfold"]; row_proba = A["row_proba"]
    trs = A["transducers"]; stf = A["stores_by_fold"]
    rbl = {L: [R for R in rows if R["lang"] == L] for L in LANGS}

    # en/it non-nested optimum from a base_cache
    bcache = run_m4.base_cache(rows, idfold, row_proba, trs, stf)
    nn_thr, _, _, _ = run_m4.base_select(rows, bcache)
    en_opt, it_opt = nn_thr["en"], nn_thr["it"]

    de_thrs = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30]

    def de_edits_at(thr):
        out = {}
        for R in rbl["de"]:
            k = idfold[R["id"]]; T = trs[k]; st = stf[k]
            out[R["id"]] = pipeline.build_edits(R["id"], R["text"], R["lang"], R["tk"],
                                                row_proba[R["id"]], thr, T, st)
        return out

    en_edits = {R["id"]: bcache[R["id"]][en_opt] for R in rbl["en"]}
    it_edits = {R["id"]: bcache[R["id"]][it_opt] for R in rbl["it"]}

    # full-train fit for test edited-row ratio
    stores_full = {}
    for b in pipeline.STORE_BUILDERS:
        b(train, stores_full)
    all_rows = pipeline.build_rows(train, labeled=True)
    det_full = pipeline.Detector().fit(all_rows, stores_full)
    trd_full = pipeline.Transducer().fit(train)
    test = pd.read_csv(os.path.join(pipeline.ROOT, "dataset", "test.csv"))
    test_rows = pipeline.build_rows(test, labeled=False)
    tp_test = det_full.token_probs(test_rows)
    de_test = [R for R in test_rows if R["lang"] == "de"]
    n_de_test = len(de_test)

    print(f"  en_opt={en_opt} it_opt={it_opt}   de_test_rows={n_de_test}")
    print(f"  {'de_thr':>7} {'ELRU':>7} {'de_lang':>8} {'de_edOOF':>9} {'test_edFrac':>11} {'ratio':>6}")
    curve = []
    for thr in de_thrs:
        de_e = de_edits_at(thr)
        allmap = {**de_e, **en_edits, **it_edits}
        voted = run_m4.group_consistency(allmap, rows_by_id, group_by_id, trs, stf, idfold,
                                         vote_langs=SHIP_VOTE, drop_langs=SHIP_VOTE, do_conv=False)
        s, det = run_m4.score_edits(rows, voted)
        de_lang = det["de"]["lang_score"]
        # de edited OOF fraction (voted)
        de_ed_oof = sum(1 for R in rbl["de"] if voted[R["id"]]) / len(rbl["de"])
        # test de edited fraction at this thr
        sub_de = {}
        for R in de_test:
            tk, pr = tp_test[R["id"]]
            sub_de[R["id"]] = pipeline.build_edits(R["id"], R["text"], R["lang"], tk, pr, thr, trd_full, stores_full)
        # apply group vote on test de (single transducer/stores)
        test_by_id = {R["id"]: R for R in de_test}
        gbi_te = {r.id: r.document_group for r in test.itertuples() if r.language == "de"}
        sub_de_v = run_m4.group_consistency(sub_de, test_by_id, gbi_te, {0: trd_full}, {0: stores_full},
                                            {i: 0 for i in sub_de}, vote_langs=SHIP_VOTE, drop_langs=SHIP_VOTE, do_conv=False)
        test_frac = sum(1 for i in sub_de_v if sub_de_v[i]) / max(n_de_test, 1)
        ratio = test_frac / TRAIN_EDIT_RATE["de"]
        curve.append((thr, s, de_lang, de_ed_oof, test_frac, ratio))
        print(f"  {thr:>7.2f} {s:>7.4f} {de_lang:>8.4f} {de_ed_oof:>9.3f} {test_frac:>11.3f} {ratio:>6.2f}")
    # recommendation
    robust = [c for c in curve if c[5] <= 1.25]
    maxcv = max(curve, key=lambda c: c[1])
    print(f"\n  max-CV de_thr={maxcv[0]} ELRU={maxcv[1]:.4f} ratio={maxcv[5]:.2f}")
    if robust:
        rb = max(robust, key=lambda c: c[1])
        print(f"  robust (ratio<=1.25) de_thr={rb[0]} ELRU={rb[1]:.4f} ratio={rb[5]:.2f}  CV cost vs max = {rb[1]-maxcv[1]:+.4f}")
    else:
        print("  no de_thr in set reaches ratio<=1.25")
    return curve


# ======================================================================
def section_enshift(A, group_by_id, rows_by_id, base_gens, base_hooks, train, markrun, mascfb):
    print("\n" + "=" * 72)
    print("SECTION 3: EN SHIFT STRESS TEST (slash/mark detector features zeroed for en)")
    print("=" * 72)
    print(f"  masked features: {A['mask_feats']}")
    set_config(base_gens, base_hooks, markrun, mascfb)
    rows = A["rows"]; idfold = A["idfold"]
    trs = A["transducers"]; stf = A["stores_by_fold"]
    rbl_en = [R for R in rows if R["lang"] == "en"]

    def en_nested(proba):
        # per-fold nested threshold for en, using given proba source
        cache = {}
        for R in rbl_en:
            k = idfold[R["id"]]; T = trs[k]; st = stf[k]
            pr = proba[R["id"]]
            cache[R["id"]] = {thr: pipeline.build_edits(R["id"], R["text"], R["lang"], R["tk"], pr, thr, T, st)
                              for thr in pipeline.GRID}
        # nested: fold-k thr chosen on other folds
        nest = {}
        for k in range(5):
            other = [R for R in rbl_en if R["fold"] != k]
            best = (-1.0, pipeline.GRID[0])
            for thr in pipeline.GRID:
                e = {R["id"]: cache[R["id"]][thr] for R in other}
                _s, det = elru.elru(e, {R["id"]: R["truth"] for R in other},
                                    {R["id"]: R["lang"] for R in other}, detail=True)
                if det["en"]["lang_score"] > best[0]:
                    best = (det["en"]["lang_score"], thr)
            for R in [r for r in rbl_en if r["fold"] == k]:
                nest[R["id"]] = cache[R["id"]][best[1]]
        # group vote (en)
        voted = run_m4.group_consistency({i: nest[i] for i in nest}, rows_by_id, group_by_id, trs, stf, idfold,
                                         vote_langs={"en"}, drop_langs={"en"}, do_conv=False)
        _s, det = elru.elru(voted, {R["id"]: R["truth"] for R in rbl_en},
                            {R["id"]: R["lang"] for R in rbl_en}, detail=True)
        # per-type recall (en)
        rec = collections.defaultdict(lambda: [0, 0])
        for R in rbl_en:
            preds = voted[R["id"]]
            for (ts, te, rep) in R["spans"]:
                if rep == "":
                    continue
                key = run_m4.span_type("en", R["text"][ts:te])
                rec[key][1] += 1
                best = max((run_m4.iou(ed["start"], ed["end"], ts, te) for ed in preds), default=0.0)
                if best >= 0.5:
                    rec[key][0] += 1
        return det["en"]["lang_score"], det["en"], {k: (v[0] / v[1] if v[1] else 0, v[1]) for k, v in rec.items()}

    normal_proba = A["row_proba"]
    masked_proba = dict(A["row_proba"])
    masked_proba.update(A["row_proba_en_masked"])  # en rows -> masked probs

    ns, nd, nrec = en_nested(normal_proba)
    ms, md, mrec = en_nested(masked_proba)
    print(f"  en nested (normal)  = {ns:.4f}  edited={nd['edited_mean']:.4f} unchanged={nd['unchanged_mean']:.4f}")
    print(f"  en nested (masked)  = {ms:.4f}  edited={md['edited_mean']:.4f} unchanged={md['unchanged_mean']:.4f}")
    print(f"  delta = {ms-ns:+.4f}   holds>0.75: {'YES' if ms>0.75 else 'NO'}")
    print("  en per-type recall  normal -> masked:")
    for key in sorted(set(nrec) | set(mrec)):
        nr = nrec.get(key, (0, 0)); mr = mrec.get(key, (0, 0))
        print(f"      {key:14s} {nr[0]/max(nr[1],1):.3f} -> {mr[0]/max(mr[1],1):.3f} (n={nr[1]})")

    # verify en norm_mem punctuation-attached coverage
    print("\n  en replacement norm-memory coverage check (punct-attached variants):")
    en_tr = train[train.language == "en"]
    from m3_ext import _normkey
    nm = m3_ext._learn_en(train)["norm_mem"]
    n_punct_keys = sum(1 for k in nm if k != k.strip(".,;:()»«\"'“”’`-"))
    print(f"    norm_mem entries={len(nm)}  (keys are punct-normalized via _normkey; punct-attached src map to same key)")
    # test how many en test tokens' normkey are covered
    test = pd.read_csv(os.path.join(pipeline.ROOT, "dataset", "test.csv"))
    en_test = test[test.language == "en"]
    covered = 0; total = 0
    for r in en_test.itertuples():
        for m in __import__("re").finditer(r"\S+", r.text):
            total += 1
            if _normkey(m.group()) in nm:
                covered += 1
    print(f"    en TEST tokens whose normkey is in norm_mem: {covered}/{total}")
    return ns, ms


# ======================================================================
def main():
    section = sys.argv[1] if len(sys.argv) > 1 else "all"
    train, ROOT = load_train()
    group_by_id = {r.id: r.document_group for r in train.itertuples()}
    # base m4 registration -> capture base gens/hooks
    m4_ext.register(pipeline)
    base_gens = list(pipeline.SPAN_CANDIDATE_GENERATORS)
    base_hooks = list(pipeline.REPLACEMENT_HOOKS)
    A = fit_all(train)
    rows_by_id = {R["id"]: R for R in A["rows"]}

    comp = None
    if section in ("compare", "all"):
        comp = section_compare(A, group_by_id, rows_by_id, base_gens, base_hooks)
    # decide de config for downstream: use +both unless it regresses de-nested
    markrun, mascfb = True, True
    if comp is not None:
        b = comp["BASE (m4)"]; n = comp["+both (N2)"]
        if n["ne_s"] < b["ne_s"]:
            # fall back to whichever single helped most
            cand = sorted(comp.items(), key=lambda kv: -kv[1]["ne_s"])[0]
            markrun = "markrun" in cand[0] or cand[0] == "+both (N2)"
            mascfb = "mascfb" in cand[0] or cand[0] == "+both (N2)"
            print(f"\n  [downstream config -> best={cand[0]}]")
    if section in ("robust", "all"):
        section_robust(A, group_by_id, rows_by_id, base_gens, base_hooks, train, markrun, mascfb)
    if section in ("enshift", "all"):
        section_enshift(A, group_by_id, rows_by_id, base_gens, base_hooks, train, markrun, mascfb)


if __name__ == "__main__":
    main()

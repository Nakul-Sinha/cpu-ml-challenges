"""N1 ITALIAN-RECOVERY ship module (on the M4 base).

it config = base threshold-merge(0.45) UNION a gated article-anchored NP-span generator
(learned per-candidate gate, cross-fit, generator-local + group-raw-text features,
INDEPENDENT of the token threshold), safe priority (base spans always kept; NP fills
zero-coverage regions), whole-NP A2 transduction, + src-first slash reorder.  de/en are
the UNTOUCHED M4 base (threshold-merge + group-vote[de,en]).

Everything learned from train at runtime, leak-free per fold (gate cross-fit; NP tables +
group-context per fold); nested = fixed it spine thr 0.45, NP gate selected per fold on
the other 4 folds; de/en thresholds selected as in M4.  Reports BOTH nested and non-nested.

Usage: cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/N1/run_n1.py [report|ship]
"""
import os, sys, json, time, collections, re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if not os.path.exists(os.path.join(ROOT, "dataset", "train.csv")):
    ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import pipeline, m4_ext
from transducer import Transducer
from run_m4 import (base_cache, base_select, group_consistency, score_edits, fp_counts,
                    per_type_recall, print_detail, LOSSMAP, SHIP_VOTE_LANGS, TRAIN_EDIT_RATE)
import elru

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")
WS = re.compile(r"\S+")
_SLASH = re.compile(r"[^\W\d_]/[^\W\d_]", re.UNICODE)

IT_SPINE_THR = 0.45                       # fixed stable base optimum (low-variance)
GATE_GRID = [1.01, 0.8, 0.7, 0.6, 0.5, 0.4]
ANCHOR_MIN_SLASHFRAC = 0.30               # anchor = token that reliably begins a slash-NP
GATE_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.04, num_leaves=20,
                   min_child_samples=25, subsample=0.85, colsample_bytree=0.8, reg_lambda=3.0,
                   is_unbalance=True, random_state=0, n_jobs=7, verbosity=-1)


# ---------------------------------------------------------------- learned it tables
def learn_tab(trdf):
    occ = collections.Counter(); spaninit = collections.Counter(); spaninit_slash = collections.Counter()
    end2_ed = collections.Counter(); end2_tot = collections.Counter()
    end3_ed = collections.Counter(); end3_tot = collections.Counter(); tok_ed = collections.Counter()
    for r in trdf[trdf.language == "it"].itertuples():
        edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
        tk = [(m.start(), m.end(), m.group()) for m in WS.finditer(r.text)]
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in edits)
        startset = {a for a, _, _ in spans}
        rfs = {a: ("/" in (rep.split()[0] if rep.split() else "")) for a, b, rep in spans}

        def inside(s, e):
            return any(s >= a and e <= b for a, b, _ in spans)
        for s, e, w in tk:
            core = w.strip(_STRIP).lower()
            if not core:
                continue
            occ[core] += 1; isin = inside(s, e)
            if isin:
                tok_ed[core] += 1
            if s in startset:
                spaninit[core] += 1
                if rfs.get(s):
                    spaninit_slash[core] += 1
            if len(core) >= 2:
                end2_tot[core[-2:]] += 1
                if isin:
                    end2_ed[core[-2:]] += 1
            if len(core) >= 3:
                end3_tot[core[-3:]] += 1
                if isin:
                    end3_ed[core[-3:]] += 1
    return dict(spaninit_rate={w: spaninit[w] / occ[w] for w in occ},
                tok_edrate={w: tok_ed[w] / occ[w] for w in occ},
                end2_rate={k: end2_ed[k] / end2_tot[k] for k in end2_tot if end2_tot[k] >= 5},
                anchors={w for w in occ if occ[w] >= 2 and spaninit.get(w, 0) >= 1
                         and spaninit_slash.get(w, 0) / max(1, spaninit.get(w, 0)) >= ANCHOR_MIN_SLASHFRAC})


def group_ctx(df):
    g = collections.defaultdict(lambda: [0, 0, 0])
    for r in df.itertuples():
        if r.language != "it":
            continue
        g[r.document_group][0] += len(r.text.split())
        g[r.document_group][1] += len(_SLASH.findall(r.text))
        g[r.document_group][2] += 1
    return {gg: (v[1] / max(1, v[0]), float(v[2])) for gg, v in g.items()}


def np_cands(tk, text, group, pr, tab, gc):
    """article-anchored NP spans (anchor + 1..3 following); generator-local + group feats."""
    n = len(tk); gs, gsz = gc.get(group, (0.0, 0.0)); out = []
    for i in range(n):
        core = tk[i][2].strip(_STRIP).lower()
        if core not in tab["anchors"]:
            continue
        for L in (1, 2, 3):
            j = i + L
            if j >= n:
                break
            a, b = tk[i][0], tk[j][1]
            if any(c in MARKS for c in text[a:b]):
                continue
            sp = pr[i:j + 1]; fe2 = []; fedr = []; nmasc = 0
            for kk in range(i + 1, j + 1):
                c = tk[kk][2].strip(_STRIP).lower()
                if len(c) >= 2:
                    e2c = tab["end2_rate"].get(c[-2:], 0.0)
                    fe2.append(e2c)
                    if e2c >= 0.15:          # data-driven "agreeing gendered ending" (learned, no literal cipher chars)
                        nmasc += 1
                fedr.append(tab["tok_edrate"].get(c, 0.0))
            f = [tab["spaninit_rate"].get(core, 0.0), tab["tok_edrate"].get(core, 0.0), float(len(core)),
                 float(L), float(b - a), float(np.mean(sp)), float(np.max(sp)), float(np.min(sp)),
                 float(sp[0]), float(sp[-1]), max(fe2) if fe2 else 0.0,
                 float(np.mean(fe2)) if fe2 else 0.0, max(fedr) if fedr else 0.0, float(nmasc),
                 1.0 if "/" in text[a:b] else 0.0, gs, gsz]
            out.append((a, b, f))
    return out


def reorder(src, rep):
    if rep.count("/") == 1 and " " not in rep and "/" in rep:
        core = src.strip(_STRIP); x, y = rep.split("/")
        if core == y and core != x:
            return core + "/" + x
    return rep


def assemble_it(tk, text, pr, gate, gate_scores, T, st):
    """base merge(0.45) UNION gated NP (safe: base kept), transduce, reorder, validate."""
    n = len(tk); spans = []; i = 0
    while i < n:
        if pr[i] >= IT_SPINE_THR:
            j = i
            while j + 1 < n and pr[j + 1] >= IT_SPINE_THR:
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


# ---------------------------------------------------------------- fold fitting
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
        trs[k] = Transducer().fit(trdf); stf[k] = st
        if verbose:
            print(f"[fold {k}] fit ({time.time()-t0:.0f}s)", flush=True)
    return rows, idfold, row_proba, trs, stf


def it_gate_scores(rows, idfold, row_proba, tabs, gctxs, gbi):
    """cross-fit gate; return {id: [(a,b,gate_prob)]} for it rows (leak-free)."""
    import lightgbm as lgb
    itrows = [R for R in rows if R["lang"] == "it"]
    cand = {}
    for R in itrows:
        k = idfold[R["id"]]
        cs = np_cands(R["tk"], R["text"], gbi[R["id"]], row_proba[R["id"]], tabs[k], gctxs[k])
        lab = []
        for (a, b, f) in cs:
            best = max((max(0, min(b, te) - max(a, ts)) / (max(b, te) - min(a, ts))
                        for (ts, te, rep) in R["spans"] if rep != "" and max(b, te) > min(a, ts)), default=0.0)
            lab.append(1 if best >= 0.5 else 0)
        cand[R["id"]] = [cs, lab, [0.0] * len(cs)]
    for k in range(5):
        Xtr, ytr = [], []
        for R in itrows:
            if idfold[R["id"]] == k:
                continue
            cs, lb, _ = cand[R["id"]]; Xtr += [c[2] for c in cs]; ytr += lb
        m = lgb.LGBMClassifier(**GATE_PARAMS)
        m.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, np.int32))
        for R in itrows:
            if idfold[R["id"]] != k:
                continue
            cs, lb, _ = cand[R["id"]]
            if cs:
                cand[R["id"]][2] = m.predict_proba(np.asarray([c[2] for c in cs], np.float32))[:, 1].tolist()
    return {i: list(zip([ (c[0], c[1]) for c in cand[i][0]], cand[i][2])) for i in cand}


def build_it_cache(rows, idfold, row_proba, trs, stf, gate_scores):
    """{id: {gate: edits}} for it rows over the gate grid."""
    itrows = [R for R in rows if R["lang"] == "it"]
    cache = {}
    for R in itrows:
        k = idfold[R["id"]]; T = trs[k]; st = stf[k]; tk = R["tk"]; text = R["text"]; pr = row_proba[R["id"]]
        gs = [(ab[0], ab[1], p) for (ab, p) in gate_scores[R["id"]]]
        cache[R["id"]] = {g: assemble_it(tk, text, pr, g, gs, T, st) for g in GATE_GRID}
    return cache


def select_it(rows, itcache):
    itrows = [R for R in rows if R["lang"] == "it"]
    truth = {R["id"]: R["truth"] for R in itrows}; lm = {R["id"]: "it" for R in itrows}
    def sc(g, ids):
        sub = {i: itcache[i][g] for i in ids}
        _s, d = elru.elru(sub, {i: truth[i] for i in ids}, {i: "it" for i in ids}, detail=True)
        return d["it"]["lang_score"]
    allids = set(truth)
    nn_g = max(GATE_GRID, key=lambda g: sc(g, allids))
    nn_edits = {i: itcache[i][nn_g] for i in truth}
    nby = {}; nest = {}
    for k in range(5):
        other = set(R["id"] for R in itrows if R["fold"] != k)
        bg = max(GATE_GRID, key=lambda g: sc(g, other))
        nby[k] = bg
        for R in [r for r in itrows if r["fold"] == k]:
            nest[R["id"]] = itcache[R["id"]][bg]
    return nn_g, nn_edits, nby, nest


# ---------------------------------------------------------------- driver
def load_train():
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    return train


def prepare(verbose=True):
    m4_ext.register(pipeline)
    train = load_train()
    gbi = {r.id: r.document_group for r in train.itertuples()}
    rows, idfold, row_proba, trs, stf = fit_folds(train, verbose)
    rows_by_id = {R["id"]: R for R in rows}
    tabs = {k: learn_tab(train[train.fold != k]) for k in range(5)}
    gctxs = {k: group_ctx(train[train.fold != k]) for k in range(5)}
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    gate_scores = it_gate_scores(rows, idfold, row_proba, tabs, gctxs, gbi)
    itcache = build_it_cache(rows, idfold, row_proba, trs, stf, gate_scores)
    return dict(train=train, gbi=gbi, rows=rows, idfold=idfold, row_proba=row_proba, trs=trs,
                stf=stf, rows_by_id=rows_by_id, bcache=bcache, itcache=itcache,
                tabs=tabs, gctxs=gctxs)


def run(mode="report"):
    t0 = time.time()
    P = prepare(verbose=True)
    rows = P["rows"]; idfold = P["idfold"]; gbi = P["gbi"]; trs = P["trs"]; stf = P["stf"]
    rows_by_id = P["rows_by_id"]; bcache = P["bcache"]; itcache = P["itcache"]; train = P["train"]

    # de/en from M4 base_select; it overridden by our NP-gated assembly
    b_nn_thr, b_nn_e, b_nby, b_ne_e = base_select(rows, bcache)
    it_nn_g, it_nn_e, it_nby, it_ne_e = select_it(rows, itcache)

    def merge_langs(base_map, it_map):
        return {i: (it_map[i] if rows_by_id[i]["lang"] == "it" else base_map[i]) for i in base_map}

    nn_e = merge_langs(b_nn_e, it_nn_e)
    ne_e = merge_langs(b_ne_e, it_ne_e)
    # ship group-vote de+en (it untouched by vote)
    nn = group_consistency(nn_e, rows_by_id, gbi, trs, stf, idfold,
                           vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    ne = group_consistency(ne_e, rows_by_id, gbi, trs, stf, idfold,
                           vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    nn_s, nn_d = score_edits(rows, nn); ne_s, ne_d = score_edits(rows, ne)

    print("\n================ N1 IT-RECOVERY CONFIG (M4 base; it = base+gated-NP) ================")
    print_detail("NON-NESTED (all-OOF op)", nn_s, nn_d)
    print(f"  it: spine_thr={IT_SPINE_THR} NP_gate(nonnested)={it_nn_g}  de/en thr={{'de':{b_nn_thr['de']},'en':{b_nn_thr['en']}}}")
    print_detail("NESTED (honest headline)", ne_s, ne_d)
    print(f"  it NP gate by fold = {it_nby}")
    fp = fp_counts(rows, nn)
    print("unchanged-row FPs (nonnested): " + ", ".join(f"{L}={fp[L][0]}/{fp[L][1]}" for L in LANGS))
    itR = [R for R in rows if R["lang"] == "it"]
    ptr = per_type_recall(itR, {i: nn[i] for i in nn if rows_by_id[i]["lang"] == "it"})
    print("it per-type IoU>=.5 recall (nonnested) vs iter-1 loss map:")
    for key in sorted(ptr):
        r, nsp = ptr[key]; b = LOSSMAP.get(key)
        print(f"    it {key[1]:14s} rec={r:.3f}(n={nsp})" + (f"  lm{b:.3f}" if b else ""))

    # canonical scorer self-check (OOF, nonnested)
    oof_df = pd.DataFrame([{"id": R["id"], "edits_json": json.dumps(nn[R["id"]], ensure_ascii=False)} for R in rows])
    chk, _ = elru.score_frames(oof_df, train[["id", "language", "edits_json"]])
    print(f"canonical elru.score_frames (OOF, nonnested) = {chk:.4f}")
    print(f"\nHEADLINE overall NESTED = {ne_s:.4f}  (base M4 0.5423; delta {ne_s-0.5423:+.4f})")
    print(f"it lang: nested {ne_d['it']['lang_score']:.4f} (base .4109), non-nested {nn_d['it']['lang_score']:.4f} (base .4133)")

    report = dict(
        config="M4 base (de/en untouched) + it: base-merge(0.45) UNION gated article-NP generator + src-first reorder",
        headline_nested_elru=round(ne_s, 4), nonnested_elru=round(nn_s, 4),
        canonical_oof_check=round(chk, 4),
        base_m4_nested=0.5423, base_m4_nonnested=0.5517, delta_nested=round(ne_s - 0.5423, 4),
        it_spine_thr=IT_SPINE_THR, it_np_gate_nonnested=float(it_nn_g), it_np_gate_by_fold={k: float(v) for k, v in it_nby.items()},
        nested_detail={L: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ne_d[L].items()} for L in LANGS},
        nonnested_detail={L: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in nn_d[L].items()} for L in LANGS},
        unchanged_fp={L: list(fp[L]) for L in LANGS},
        it_per_type_recall={f"{k[0]}_{k[1]}": [round(v[0], 3), v[1]] for k, v in ptr.items()})

    if mode == "ship":
        _ship_submission(P, b_nn_thr, it_nn_g, report)
    json.dump(report, open(os.path.join(HERE, "cv_report_n1.json"), "w"), indent=2)
    oof_df.to_csv(os.path.join(HERE, "oof_edits_n1.csv"), index=False)
    print(f"\nwrote cv_report_n1.json, oof_edits_n1.csv  [{time.time()-t0:.0f}s]")
    return report


def _ship_submission(P, b_nn_thr, it_nn_g, report):
    """full-train fit -> test submission (de/en base + group-vote; it base+gated-NP)."""
    import lightgbm as lgb
    train = P["train"]
    test = pd.read_csv(os.path.join(ROOT, "dataset", "test.csv"))
    gbi_te = {r.id: r.document_group for r in test.itertuples()}
    # full-train artifacts
    stores_full = {}
    for b in pipeline.STORE_BUILDERS:
        b(train, stores_full)
    all_rows = pipeline.build_rows(train, labeled=True)
    det_full = pipeline.Detector().fit(all_rows, stores_full)
    trd_full = Transducer().fit(train)
    tab_full = learn_tab(train); gc_full = group_ctx(train)
    gbi_tr = {r.id: r.document_group for r in train.itertuples()}
    # full-train gate (fit on all train NP candidates); batch token_probs (in-sample, test-only path)
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
    # test inference
    test_rows = pipeline.build_rows(test, labeled=False)
    tp_test = det_full.token_probs(test_rows)
    sub = {}
    for R in test_rows:
        tk, pr = tp_test[R["id"]]; lang = R["lang"]; text = R["text"]
        if lang == "it":
            cs = np_cands(tk, text, gbi_te[R["id"]], pr, tab_full, gc_full)
            gscore = []
            if cs:
                pv = gate_model.predict_proba(np.asarray([c[2] for c in cs], np.float32))[:, 1]
                gscore = [(c[0], c[1], float(p)) for c, p in zip(cs, pv)]
            sub[R["id"]] = assemble_it(tk, text, pr, it_nn_g, gscore, trd_full, stores_full)
        else:
            sub[R["id"]] = pipeline.build_edits(R["id"], text, lang, tk, pr, b_nn_thr[lang], trd_full, stores_full)
    # group-vote de/en
    test_by_id = {R["id"]: R for R in test_rows}
    idf = {i: 0 for i in sub}
    sub = group_consistency(sub, test_by_id, gbi_te, {0: trd_full}, {0: stores_full}, idf,
                            vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    assert len(sub) == len(test) == 445 and set(sub) == set(test.id)
    tl = {r.id: len(r.text) for r in test.itertuples()}
    bad = [i for i in sub if not elru.validate_edits(sub[i], tl[i])]
    assert not bad, f"invalid: {bad[:5]}"
    lang_te = {r.id: r.language for r in test.itertuples()}
    edn = collections.Counter(); totn = collections.Counter()
    for i in sub:
        totn[lang_te[i]] += 1
        if sub[i]:
            edn[lang_te[i]] += 1
    pd.DataFrame([{"id": i, "edits_json": json.dumps(sub[i], ensure_ascii=False)} for i in test.id]
                 ).to_csv(os.path.join(HERE, "submission_v2_n1.csv"), index=False)
    report["submission_edit_rate"] = {L: round(edn[L] / max(1, totn[L]), 3) for L in LANGS}
    report["submission_rows"] = len(sub)
    print("submission edited-row fractions: " +
          ", ".join(f"{L}={edn[L]}/{totn[L]}" for L in LANGS))
    print("wrote submission_v2_n1.csv")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "report")

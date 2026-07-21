"""N1 exp6: UNIFIED gated candidate generator for it (single-token below-threshold +
article-anchored NP), rich generator-local + group raw-text features, cross-fit learned
gate, union with base merge, whole-span transduction, joint (thr,gate) select nested.
Also applies src-first reorder. Reports net nested/non-nested + FP + per-type recall.
"""
import os, sys, json, time, collections, re, pickle
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline, m4_ext
from transducer import Transducer
from run_m4 import base_cache, base_select, group_consistency, score_edits, fp_counts, per_type_recall, SHIP_VOTE_LANGS
import elru

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")
WS = re.compile(r"\S+")
_SLASH = re.compile(r"[^\W\d_]/[^\W\d_]", re.UNICODE)


def rebuild(train):
    m4_ext.register(pipeline)
    trs, stf = {}, {}
    for k in range(5):
        trdf = train[train.fold != k]
        trs[k] = Transducer().fit(trdf)
        st = {}
        for b in pipeline.STORE_BUILDERS:
            b(trdf, st)
        stf[k] = st
    return trs, stf


def learn_tab(trdf):
    occ = collections.Counter(); spaninit = collections.Counter(); spaninit_slash = collections.Counter()
    end2_ed = collections.Counter(); end2_tot = collections.Counter()
    end3_ed = collections.Counter(); end3_tot = collections.Counter()
    tok_ed = collections.Counter()
    for r in trdf[trdf.language == "it"].itertuples():
        tk = [(m.start(), m.end(), m.group()) for m in WS.finditer(r.text)]
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in r.edits)
        startset = {a for a, _, _ in spans}
        rfs = {}
        for a, b, rep in spans:
            fw = rep.split()[0] if rep.split() else ""
            rfs[a] = ("/" in fw)

        def inside(s, e):
            return any(s >= a and e <= b for a, b, _ in spans)
        for i, (s, e, w) in enumerate(tk):
            core = w.strip(_STRIP).lower()
            if not core:
                continue
            occ[core] += 1
            isin = inside(s, e)
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
    spaninit_rate = {w: spaninit[w] / occ[w] for w in occ}
    tok_edrate = {w: tok_ed[w] / occ[w] for w in occ}
    end2_rate = {k: end2_ed[k] / end2_tot[k] for k in end2_tot if end2_tot[k] >= 5}
    end3_rate = {k: end3_ed[k] / end3_tot[k] for k in end3_tot if end3_tot[k] >= 5}
    anchors = {w for w in occ if occ[w] >= 2 and spaninit_slash.get(w, 0) / max(1, spaninit.get(w, 0)) >= 0.3
               and spaninit.get(w, 0) >= 1}
    return dict(spaninit_rate=spaninit_rate, tok_edrate=tok_edrate, end2_rate=end2_rate,
                end3_rate=end3_rate, anchors=anchors, occ=occ)


def group_ctx(train):
    """raw-text group features (compliant, generalizes to test): slash-density, size."""
    g = collections.defaultdict(lambda: [0, 0, 0])  # ntok, nslash, nrow
    for r in train.itertuples():
        if r.language != "it":
            continue
        toks = r.text.split()
        g[r.document_group][0] += len(toks)
        g[r.document_group][1] += len(_SLASH.findall(r.text))
        g[r.document_group][2] += 1
    return {gg: (v[1] / max(1, v[0]), float(v[2])) for gg, v in g.items()}


FEATN = ["kind", "det_p", "left_p", "right_p", "tok_edr", "spaninit", "end2", "end3",
         "len", "L", "n_char", "is_slash_in", "cap", "art_spaninit", "foll_max_end2",
         "foll_max_edr", "n_masc", "g_slashden", "g_size"]


def candidates(R, pr, tab, gctx):
    """single-token (edit-signal) + NP (anchor+1..3). Returns list of (a,b,feat)."""
    tk = R["tk"]; text = R["text"]; n = len(tk)
    gs, gsz = gctx.get(R["document_group"], (0.0, 0.0))
    out = []
    for i in range(n):
        core = tk[i][2].strip(_STRIP).lower()
        if not core:
            continue
        edr = tab["tok_edrate"].get(core, 0.0)
        e2 = tab["end2_rate"].get(core[-2:], 0.0) if len(core) >= 2 else 0.0
        e3 = tab["end3_rate"].get(core[-3:], 0.0) if len(core) >= 3 else 0.0
        # single-token candidate if any edit signal (loose net; gate decides)
        if pr[i] >= 0.20 or edr >= 0.05 or e2 >= 0.12:
            a, b = tk[i][0], tk[i][1]
            f = [0.0, float(pr[i]), float(pr[i - 1]) if i > 0 else 0.0,
                 float(pr[i + 1]) if i + 1 < n else 0.0, edr, tab["spaninit_rate"].get(core, 0.0),
                 e2, e3, float(len(core)), 1.0, float(b - a),
                 1.0 if "/" in tk[i][2] else 0.0, 1.0 if tk[i][2][:1].isupper() else 0.0,
                 0.0, 0.0, 0.0, 0.0, gs, gsz]
            out.append((a, b, f))
        # NP candidate
        if core in tab["anchors"]:
            for L in (1, 2, 3):
                j = i + L
                if j >= n:
                    break
                a, b = tk[i][0], tk[j][1]
                if any(c in MARKS for c in text[a:b]):
                    continue
                sp = pr[i:j + 1]
                fe2 = []; fedr = []; nmasc = 0
                for kk in range(i + 1, j + 1):
                    c = tk[kk][2].strip(_STRIP).lower()
                    if len(c) >= 2:
                        fe2.append(tab["end2_rate"].get(c[-2:], 0.0))
                        if c[-1:] in ("g", "y", "o", "e", "i"):
                            nmasc += 1
                    fedr.append(tab["tok_edrate"].get(c, 0.0))
                f = [1.0, float(np.mean(sp)), float(pr[i - 1]) if i > 0 else 0.0,
                     float(pr[j + 1]) if j + 1 < n else 0.0, tab["tok_edrate"].get(core, 0.0),
                     tab["spaninit_rate"].get(core, 0.0), 0.0, 0.0, float(len(core)), float(L),
                     float(b - a), 1.0 if "/" in text[a:b] else 0.0,
                     1.0 if tk[i][2][:1].isupper() else 0.0, tab["spaninit_rate"].get(core, 0.0),
                     max(fe2) if fe2 else 0.0, max(fedr) if fedr else 0.0, float(nmasc), gs, gsz]
                out.append((a, b, f))
    return out


def gctx_row(R, gctx):
    return gctx.get(R.get("document_group", None), (0.0, 0.0)) if isinstance(R, dict) else (0.0, 0.0)


def reorder(src, rep):
    if rep.count("/") == 1 and " " not in rep and "/" in rep:
        core = src.strip(_STRIP); a, b = rep.split("/")
        if core == b and core != a:
            return core + "/" + a
    return rep


def main():
    import lightgbm as lgb
    t0 = time.time()
    S = pickle.load(open(os.path.join(ROOT, "runs", "N1_state.pkl"), "rb"))
    rows = S["rows"]; idfold = S["idfold"]; row_proba = S["row_proba"]; gbi = S["group_by_id"]
    for R in rows:
        R["document_group"] = gbi[R["id"]]
    rows_by_id = {R["id"]: R for R in rows}
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    trs, stf = rebuild(train)
    tabs = {k: learn_tab(train[train.fold != k]) for k in range(5)}
    gctxs = {k: group_ctx(train[train.fold != k]) for k in range(5)}
    bcache = base_cache(rows, idfold, row_proba, trs, stf)
    print(f"[rebuild {time.time()-t0:.0f}s]")
    itrows = [R for R in rows if R["lang"] == "it"]

    cand_by_id = {}
    for R in itrows:
        k = idfold[R["id"]]; tab = tabs[k]; pr = row_proba[R["id"]]; gc = gctxs[k]
        cs = candidates(R, pr, tab, gc)
        lab = []
        for (a, b, f) in cs:
            best = 0.0
            for (ts, te, rep) in R["spans"]:
                if rep == "":
                    continue
                ov = max(0, min(b, te) - max(a, ts)); un = max(b, te) - min(a, ts)
                best = max(best, ov / un if un > 0 else 0.0)
            lab.append(1 if best >= 0.5 else 0)
        cand_by_id[R["id"]] = [cs, lab, [0.0] * len(cs)]
    ncand = sum(len(v[0]) for v in cand_by_id.values()); npos = sum(sum(v[1]) for v in cand_by_id.values())
    print(f"candidates={ncand} pos={npos} ceiling={npos/max(1,ncand):.1%}")

    for k in range(5):
        Xtr, ytr = [], []
        for R in itrows:
            if idfold[R["id"]] == k:
                continue
            cs, lb, _ = cand_by_id[R["id"]]
            Xtr += [c[2] for c in cs]; ytr += lb
        m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.04,
                               num_leaves=24, min_child_samples=25, subsample=0.85,
                               colsample_bytree=0.8, reg_lambda=3.0, is_unbalance=True,
                               random_state=0, n_jobs=7, verbosity=-1)
        m.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, np.int32))
        for R in itrows:
            if idfold[R["id"]] != k:
                continue
            cs, lb, _ = cand_by_id[R["id"]]
            if cs:
                pv = m.predict_proba(np.asarray([c[2] for c in cs], np.float32))[:, 1]
                cand_by_id[R["id"]][2] = pv.tolist()

    def it_edits(R, thr, g, do_reorder=True):
        tk = R["tk"]; text = R["text"]; pr = row_proba[R["id"]]; n = len(tk)
        k = idfold[R["id"]]; T = trs[k]; st = stf[k]
        spans = []
        i = 0
        while i < n:
            if pr[i] >= thr:
                j = i
                while j + 1 < n and pr[j + 1] >= thr:
                    j += 1
                spans.append((tk[i][0], tk[j][1], float(np.mean(pr[i:j + 1])) + 1.0))
                i = j + 1
            else:
                i += 1
        cs, lb, pv = cand_by_id[R["id"]]
        for (c, p) in zip(cs, pv):
            if p >= g:
                spans.append((c[0], c[1], float(p)))
        spans.sort(key=lambda s: -s[2])
        chosen = []; occ = []
        for (a, b, sc) in spans:
            if any(not (b <= x or y <= a) for (x, y) in occ):
                continue
            chosen.append((a, b, sc)); occ.append((a, b))
        edits = []
        for (a, b, sc) in chosen:
            src = text[a:b]
            ctx = {"text": text, "start": a, "end": b, "lang": "it", "tokens": tk, "stores": st}
            rep = None
            for hook in pipeline.REPLACEMENT_HOOKS:
                r = hook("it", src, ctx, st)
                if r is not None:
                    rep = r; break
            if rep is None:
                rep = T.predict("it", src, ctx) or src
            if do_reorder and len(src.split()) == 1:
                rep = reorder(src, rep)
            edits.append((sc, {"start": a, "end": b, "replacement": rep[:160]}))
        edits.sort(key=lambda e: -e[0]); edits = [e for _, e in edits[:8]]
        edits.sort(key=lambda e: e["start"])
        if not elru.validate_edits(edits, len(text)):
            edits = pipeline._repair(edits, len(text))
        return edits

    THR = [round(x, 2) for x in np.arange(0.35, 0.60, 0.02)]
    GATES = [1.01, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
    truth = {R["id"]: R["truth"] for R in itrows}; lm = {R["id"]: "it" for R in itrows}
    cache = {}
    def ef(thr, g):
        if (thr, g) not in cache:
            cache[(thr, g)] = {R["id"]: it_edits(R, thr, g) for R in itrows}
        return cache[(thr, g)]
    # non-nested
    best = (-1, None, None)
    for thr in THR:
        for g in GATES:
            _s, d = elru.elru(ef(thr, g), truth, lm, detail=True)
            if d["it"]["lang_score"] > best[0]:
                best = (d["it"]["lang_score"], thr, g)
    bl, bthr, bg = best
    e = ef(bthr, bg); _s, d = elru.elru(e, truth, lm, detail=True)
    fp = sum(1 for R in itrows if len(R["truth"]) == 0 and e[R["id"]])
    print(f"BEST NN it={bl:.4f} (e{d['it']['edited_mean']:.3f}/u{d['it']['unchanged_mean']:.3f}) "
          f"thr={bthr} g={bg} FP={fp}/182  (base 0.4133 FP56)")
    # nested
    nby = {}; nest = {}
    for k in range(5):
        b2 = (-1, None, None); oids = set(R["id"] for R in itrows if R["fold"] != k)
        for thr in THR:
            for g in GATES:
                e = ef(thr, g); sub = {i: e[i] for i in oids}
                _s, d = elru.elru(sub, {i: truth[i] for i in oids}, {i: "it" for i in oids}, detail=True)
                if d["it"]["lang_score"] > b2[0]:
                    b2 = (d["it"]["lang_score"], thr, g)
        nby[k] = (b2[1], b2[2]); e = ef(b2[1], b2[2])
        for R in [r for r in itrows if r["fold"] == k]:
            nest[R["id"]] = e[R["id"]]
    _s, dn = elru.elru(nest, truth, lm, detail=True)
    print(f"NESTED it={dn['it']['lang_score']:.4f} (e{dn['it']['edited_mean']:.3f}/u{dn['it']['unchanged_mean']:.3f}) nby={nby}")
    # overall nested with it config swapped into full ship (de/en base + group vote)
    nn_thr, nn_e, nbyD, ne_e = base_select(rows, bcache)
    # replace it rows in nested with our nest, de/en stay base-nested; then de/en group vote
    ne_full = {i: (nest[i] if rows_by_id[i]["lang"] == "it" else ne_e[i]) for i in ne_e}
    nn_full = {i: (ef(bthr, bg)[i] if rows_by_id[i]["lang"] == "it" else nn_e[i]) for i in nn_e}
    ne_full = group_consistency(ne_full, rows_by_id, gbi, trs, stf, idfold,
                                vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    nn_full = group_consistency(nn_full, rows_by_id, gbi, trs, stf, idfold,
                                vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    ne_s, ne_d = score_edits(rows, ne_full); nn_s, nn_d = score_edits(rows, nn_full)
    print(f"\nOVERALL NESTED={ne_s:.4f} (it={ne_d['it']['lang_score']:.4f} de={ne_d['de']['lang_score']:.4f} en={ne_d['en']['lang_score']:.4f})")
    print(f"OVERALL NONNEST={nn_s:.4f} (it={nn_d['it']['lang_score']:.4f})")
    ptr = per_type_recall([R for R in rows if R["lang"] == "it"], {i: nn_full[i] for i in nn_full if rows_by_id[i]["lang"] == "it"})
    for key in sorted(ptr):
        r, nsp = ptr[key]
        print(f"    it {key[1]:14s} rec={r:.3f}(n={nsp})")
    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

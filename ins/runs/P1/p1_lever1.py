"""P1 LEVER 1 -- IT-ONLY token re-scorer.

A second LightGBM trained ONLY on Italian tokens with it-specific features
(morphological ending char-classes 1-3 with learned edit-rates, article-anchor
distances both directions, agreement-chain features, hashed char-ngrams of token +
neighbors, token position, NP-context flags, group slash-density).  Cross-fit per fold
(leak-free).  Its per-token P(edited) is BLENDED with the shared detector prob:
    blended = (1-w)*shared + w*p_it        (convex; w tuned NESTED per fold)
and an alternative additive "boost" variant is also measured.  Blended probs feed the
FIXED it NP-gate assembly (spine 0.45, gate 0.8; NP gate_scores unchanged) so the
measurement isolates the detector-prob change.

Reports: it token PR-AUC shared vs p_it vs blended; zero-coverage recovery on
it multi_plain spans; per-type IoU>=.5 recall; it nested (blend tuned nested) and
resulting overall nested on the N3 base (de/en frozen).

Run: cd ~/insled && OMP_NUM_THREADS=7 nice -n 10 ~/venv/bin/python runs/P1/p1_lever1.py
"""
import os, sys, json, time, collections, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/insled")
for p in (os.path.join(ROOT, "runs", "M4"), os.path.join(ROOT, "runs", "N2"),
          os.path.join(ROOT, "runs", "N1"), os.path.join(ROOT, "solution"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import pipeline, elru
from run_n1 import assemble_it as n1_assemble_it, IT_SPINE_THR
import p1_base

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")
NBH = 512

RESC_PARAMS = dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=24,
                   min_child_samples=25, subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=3.0, is_unbalance=True, random_state=0, n_jobs=7, verbosity=-1)
W_GRID = [round(x, 2) for x in np.arange(0.0, 0.92, 0.1)]
BOOST_GRID = [round(x, 2) for x in np.arange(0.0, 0.81, 0.1)]   # additive boost coeff


def h(s, b=NBH):
    return int(zlib.crc32(s.encode("utf-8")) % b)


# ---------------------------------------------------------------- learned it morphology
def learn_it_morph(trdf):
    """Per fold-train it edit-rates for suffix classes (1/2/3), prefix(2), token core."""
    suf = [collections.Counter() for _ in range(4)]      # ed counts by suffix length 1..3
    suft = [collections.Counter() for _ in range(4)]
    pre_ed = collections.Counter(); pre_tot = collections.Counter()
    tok_ed = collections.Counter(); tok_tot = collections.Counter()
    lang_ed = 0; lang_tot = 0
    for r in trdf[trdf.language == "it"].itertuples():
        edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
        tk = [(m.start(), m.end(), m.group()) for m in pipeline.WORD_RE.finditer(r.text)]
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in edits)

        def inside(s, e):
            return any(s >= a and e <= b for a, b, _ in spans)
        for s, e, w in tk:
            core = w.strip(_STRIP).lower()
            if not core:
                continue
            isin = 1 if inside(s, e) else 0
            lang_ed += isin; lang_tot += 1
            tok_ed[core] += isin; tok_tot[core] += 1
            for L in (1, 2, 3):
                if len(core) >= L:
                    suf[L][core[-L:]] += isin; suft[L][core[-L:]] += 1
            if len(core) >= 2:
                pre_ed[core[:2]] += isin; pre_tot[core[:2]] += 1
    prior = (lang_ed + 0.5) / (lang_tot + 1.0)

    def mk(ed, tot, a):
        return {k: (ed[k] + a * prior) / (tot[k] + a) for k in tot}
    return dict(prior=prior,
                suf1=mk(suf[1], suft[1], 8.0), suf2=mk(suf[2], suft[2], 12.0),
                suf3=mk(suf[3], suft[3], 20.0), pre2=mk(pre_ed, pre_tot, 12.0),
                tok=mk(tok_ed, tok_tot, 5.0), tok_tot=tok_tot)


IT_FEAT_NAMES = None
IT_CAT_NAMES = ["suf2_id", "suf3_id", "pre2_id", "tok_id", "prev_id", "next_id"]


def it_feats(R, i, tab, gc, gbi, morph, shp=None):
    """it-specific feature row for token i.  Returns list (schema frozen once).
    shp = shared detector prob list for the row (stacking context) or None."""
    global IT_FEAT_NAMES
    tk = R["tk"]; n = len(tk); text = R["text"]
    group = gbi[R["id"]]; gs, gsz = gc.get(group, (0.0, 0.0))
    w = tk[i][2]; core = w.strip(_STRIP).lower()
    cl = len(core)
    feats = []; names = []
    def add(v, nm):
        feats.append(float(v))
        if IT_FEAT_NAMES is None:
            names.append(nm)
    # ---- morphological ending edit-rates (1/2/3) + prefix + token ----
    add(morph["suf1"].get(core[-1:], morph["prior"]) if cl >= 1 else morph["prior"], "suf1_r")
    add(morph["suf2"].get(core[-2:], morph["prior"]) if cl >= 2 else morph["prior"], "suf2_r")
    add(morph["suf3"].get(core[-3:], morph["prior"]) if cl >= 3 else morph["prior"], "suf3_r")
    add(morph["pre2"].get(core[:2], morph["prior"]) if cl >= 2 else morph["prior"], "pre2_r")
    add(morph["tok"].get(core, morph["prior"]), "tok_r")
    add(np.log1p(morph["tok_tot"].get(core, 0.0)), "tok_sup")
    add(tab["tok_edrate"].get(core, 0.0), "n1_tok_edr")
    add(tab["end2_rate"].get(core[-2:], 0.0) if cl >= 2 else 0.0, "n1_end2")
    add(tab["spaninit_rate"].get(core, 0.0), "spaninit")
    # ---- shape ----
    add(cl, "corelen"); add(1.0 if w[:1].isupper() else 0.0, "cap")
    add(1.0 if (w.isupper() and any(c.isalpha() for c in w)) else 0.0, "allcaps")
    add(1.0 if any(c in MARKS for c in w) else 0.0, "has_mark")
    add(1.0 if any((not c.isalnum()) for c in w[1:-1]) else 0.0, "mid_pun")
    # ---- article-anchor distances (both directions) ----
    dprev = 99; dnext = 99
    for d in range(1, 6):
        if i - d >= 0 and dprev == 99:
            pc = tk[i - d][2].strip(_STRIP).lower()
            if pc in tab["anchors"] or tab["spaninit_rate"].get(pc, 0.0) >= 0.30:
                dprev = d
        if i + d < n and dnext == 99:
            nc = tk[i + d][2].strip(_STRIP).lower()
            if nc in tab["anchors"] or tab["spaninit_rate"].get(nc, 0.0) >= 0.30:
                dnext = d
    add(min(dprev, 6), "dist_prev_anchor"); add(min(dnext, 6), "dist_next_anchor")
    add(1.0 if dprev <= 3 else 0.0, "prev_anchor3")
    # ---- agreement chain: high-suf2-rate neighbours (window +/-2) + run length ----
    def hi2(j):
        if 0 <= j < n:
            cj = tk[j][2].strip(_STRIP).lower()
            return len(cj) >= 2 and morph["suf2"].get(cj[-2:], 0.0) >= 0.20
        return False
    chain = sum(1 for j in range(i - 2, i + 3) if hi2(j))
    add(float(chain), "agree_chain")
    rl = 0
    if hi2(i):
        rl = 1; j = i - 1
        while hi2(j):
            rl += 1; j -= 1
        j = i + 1
        while hi2(j):
            rl += 1; j += 1
    add(float(rl), "agree_run")
    # ---- NP-context flags ----
    pc = tk[i - 1][2].strip(_STRIP).lower() if i - 1 >= 0 else ""
    nc = tk[i + 1][2].strip(_STRIP).lower() if i + 1 < n else ""
    add(1.0 if (pc in tab["anchors"] or tab["spaninit_rate"].get(pc, 0.0) >= 0.30) else 0.0, "prev_is_art")
    add(morph["suf2"].get(nc[-2:], 0.0) if len(nc) >= 2 else 0.0, "next_suf2_r")
    add(morph["suf2"].get(pc[-2:], 0.0) if len(pc) >= 2 else 0.0, "prev_suf2_r")
    s0, e0 = tk[i][0], tk[i][1]
    win = text[max(0, s0 - 90):e0 + 90]
    import re as _re
    add(1.0 if _re.search(r"[^\W\d_]/[^\W\d_]", win) else 0.0, "slashwin")
    # ---- position ----
    add(i / max(n - 1, 1), "pos"); add(1.0 if i == 0 else 0.0, "is_first")
    add(1.0 if i == n - 1 else 0.0, "is_last"); add(np.log1p(n), "n_tok")
    # ---- group slash-density ----
    add(gs, "grp_slash"); add(np.log1p(gsz), "grp_sz")
    # ---- shared-detector stacking context (probs around i) ----
    if shp is not None:
        add(float(shp[i]), "sh_p")
        add(float(shp[i - 1]) if i - 1 >= 0 else 0.0, "sh_prev")
        add(float(shp[i + 1]) if i + 1 < n else 0.0, "sh_next")
        add(float(np.mean(shp[max(0, i - 2):i + 3])), "sh_win")
        add(float(np.max(shp[max(0, i - 2):i + 3])), "sh_winmax")
    # ---- categorical hashed morphology (token + neighbours) ----
    catstart = len(feats)
    add(h(core[-2:]) if cl >= 2 else 0, "suf2_id")
    add(h(core[-3:]) if cl >= 3 else 0, "suf3_id")
    add(h(core[:2]) if cl >= 2 else 0, "pre2_id")
    add(h(core), "tok_id")
    add(h(pc) if pc else 0, "prev_id")
    add(h(nc) if nc else 0, "next_id")
    if IT_FEAT_NAMES is None:
        IT_FEAT_NAMES = names
    cat_idx = list(range(catstart, len(feats)))
    return feats, cat_idx


def build_it_matrix(itrows, idfold, tabs, gctxs, gbi, morphs, shared=None):
    """Return {id: (X[ntok,F], y[ntok])}, cat_idx using each row's OWN-fold-out tables."""
    out = {}
    cat_idx = None
    for R in itrows:
        k = idfold[R["id"]]
        shp = shared[R["id"]] if shared is not None else None
        X = []; ci = None
        for i in range(len(R["tk"])):
            f, ci = it_feats(R, i, tabs[k], gctxs[k], gbi, morphs[k], shp)
            X.append(f)
        out[R["id"]] = (np.asarray(X, np.float32), np.asarray(R["y"], np.int32))
        cat_idx = ci
    return out, cat_idx


def crossfit_rescorer(itrows, idfold, mats, cat_idx):
    """Train re-scorer on it tokens folds!=k, predict fold k -> {id: p_it list}."""
    p_it = {}
    for k in range(5):
        Xtr = np.concatenate([mats[R["id"]][0] for R in itrows if idfold[R["id"]] != k])
        ytr = np.concatenate([mats[R["id"]][1] for R in itrows if idfold[R["id"]] != k])
        m = lgb.LGBMClassifier(**RESC_PARAMS)
        m.fit(Xtr, ytr, categorical_feature=cat_idx)
        for R in itrows:
            if idfold[R["id"]] == k:
                X = mats[R["id"]][0]
                p_it[R["id"]] = (m.predict_proba(X)[:, 1].tolist() if len(X) else [])
    return p_it


def it_lang_score(itrows, edits):
    truth = {R["id"]: R["truth"] for R in itrows}
    _s, d = elru.elru(edits, truth, {R["id"]: "it" for R in itrows}, detail=True)
    return d["it"]["lang_score"], d["it"]


def assemble_all_it(itrows, idfold, probs, gate_scores, trs, stf, gate=0.8):
    out = {}
    for R in itrows:
        k = idfold[R["id"]]
        gs = [(ab[0], ab[1], p) for (ab, p) in gate_scores[R["id"]]]
        out[R["id"]] = n1_assemble_it(R["tk"], R["text"], probs[R["id"]], gate, gs, trs[k], stf[k])
    return out


def blend_probs(itrows, shared, p_it, w, mode="convex"):
    out = {}
    for R in itrows:
        i = R["id"]; sh = np.asarray(shared[i]); pit = np.asarray(p_it[i]) if p_it[i] else sh
        if mode == "convex":
            b = (1 - w) * sh + w * pit
        else:  # boost: additive lift where re-scorer confident, capped at 1
            b = np.clip(sh + w * np.clip(pit - 0.3, 0, None), 0, 1)
        out[i] = b.tolist()
    return out


def zero_coverage_recovery(itrows, shared, blended, thr=IT_SPINE_THR):
    """Fraction of it multi_plain true spans with >=1 token above `thr` (shared vs blended)."""
    from run_m4 import span_type
    def cov(probs):
        hit = tot = 0
        for R in itrows:
            for (ts, te, rep) in R["spans"]:
                if rep == "" or span_type("it", R["text"][ts:te]) != "multi_plain":
                    continue
                tot += 1
                any_hi = any(probs[R["id"]][ix] >= thr for ix, (s, e, w) in enumerate(R["tk"])
                             if s >= ts and e <= te)
                hit += 1 if any_hi else 0
        return hit, tot
    return cov(shared), cov(blended)


def rescorer_oof(itrows, idfold, train, tabs, gctxs, gbi, shared=None):
    """Leak-free OOF it re-scorer probs (independent view: no shared-prob features)."""
    global IT_FEAT_NAMES
    IT_FEAT_NAMES = None
    morphs = {k: learn_it_morph(train[train.fold != k]) for k in range(5)}
    mats, cat_idx = build_it_matrix(itrows, idfold, tabs, gctxs, gbi, morphs, shared)
    return crossfit_rescorer(itrows, idfold, mats, cat_idx)


def main():
    global IT_FEAT_NAMES
    t0 = time.time()
    P = p1_base.prepare(verbose=True)
    train = P["train"]; rows = P["rows"]; idfold = P["idfold"]; gbi = P["gbi"]
    shared = P["row_proba"]; trs = P["trs"]; stf = P["stf"]; gate_scores = P["gate_scores"]
    tabs = P["tabs"]; gctxs = P["gctxs"]
    itrows = [R for R in rows if R["lang"] == "it"]
    print(f"[prepare {time.time()-t0:.0f}s]  {len(itrows)} it rows", flush=True)

    # learned morphology per fold
    morphs = {k: learn_it_morph(train[train.fold != k]) for k in range(5)}
    y_all = np.concatenate([np.asarray(R["y"]) for R in itrows])
    sh_all = np.concatenate([np.asarray(shared[R["id"]]) for R in itrows])
    ap_sh = average_precision_score(y_all, sh_all)

    # cross-fit re-scorer in two flavours: independent (it-context only) + stacked (+shared ctx)
    rescorers = {}
    print("\n================ IT TOKEN PR-AUC ================")
    print(f"  shared detector      : {ap_sh:.4f}")
    for flav, use_sh in (("independent", None), ("stacked", shared)):
        IT_FEAT_NAMES = None
        mats, cat_idx = build_it_matrix(itrows, idfold, tabs, gctxs, gbi, morphs, use_sh)
        p_it = crossfit_rescorer(itrows, idfold, mats, cat_idx)
        pit_all = np.concatenate([np.asarray(p_it[R["id"]] if p_it[R["id"]] else shared[R["id"]]) for R in itrows])
        ap_pit = average_precision_score(y_all, pit_all)
        blends = {w: average_precision_score(y_all, (1 - w) * sh_all + w * pit_all) for w in (0.3, 0.5, 0.7)}
        print(f"  re-scorer [{flav:11s}]: {ap_pit:.4f}   blend " +
              " ".join(f"w{w}:{blends[w]:.4f}" for w in (0.3, 0.5, 0.7)) +
              f"   [{len(IT_FEAT_NAMES)+len(IT_CAT_NAMES)} feats]")
        rescorers[flav] = p_it
    print(f"[re-scorers cross-fit {time.time()-t0:.0f}s]", flush=True)
    p_it = rescorers["stacked"]   # default downstream; independent kept for reference

    # ---- baseline it edits (shared, no blend) ----
    base_edits = assemble_all_it(itrows, idfold, shared, gate_scores, trs, stf, gate=0.8)
    base_it, base_det = it_lang_score(itrows, base_edits)
    print(f"\nbaseline it lang_score (shared, spine {IT_SPINE_THR} gate 0.8) = {base_it:.4f}")
    from run_m4 import per_type_recall
    truth_it = {R["id"]: R["truth"] for R in itrows}

    def subset_score(edits_by_w, w, ids):
        sub = {i: edits_by_w[w][i] for i in ids}
        _s, d = elru.elru(sub, {i: truth_it[i] for i in ids}, {i: "it" for i in ids}, detail=True)
        return d["it"]["lang_score"]

    best_overall = {"it": base_it, "flav": "base", "mode": "base", "w": None, "edits": base_edits}
    for flav in ("independent", "stacked"):
        pit = rescorers[flav]
        for mode, grid in (("convex", W_GRID), ("boost", BOOST_GRID)):
            edits_by_w = {}
            for w in grid:
                pr = blend_probs(itrows, shared, pit, w, mode)
                edits_by_w[w] = assemble_all_it(itrows, idfold, pr, gate_scores, trs, stf, gate=0.8)
            allids = set(truth_it)
            scores = {w: subset_score(edits_by_w, w, allids) for w in grid}
            nn_w = max(grid, key=lambda w: scores[w])
            nby = {}; nest_edits = {}
            for k in range(5):
                other = set(R["id"] for R in itrows if idfold[R["id"]] != k)
                bw = max(grid, key=lambda w: subset_score(edits_by_w, w, other))
                nby[k] = bw
                for R in itrows:
                    if idfold[R["id"]] == k:
                        nest_edits[R["id"]] = edits_by_w[bw][R["id"]]
            nest_it, nest_det = it_lang_score(itrows, nest_edits)
            nn_it = scores[nn_w]
            print(f"\n---- [{flav}] {mode} ----")
            print("  w it: " + " ".join(f"{w}:{scores[w]:.4f}" for w in grid))
            print(f"  NON-NESTED w={nn_w} it={nn_it:.4f} | NESTED it={nest_it:.4f} (delta {nest_it-base_it:+.4f}) "
                  f"ed={nest_det['edited_mean']:.4f} un={nest_det['unchanged_mean']:.4f}  by-fold={dict((k,float(v)) for k,v in nby.items())}")
            if nest_it > best_overall["it"]:
                best_overall = {"it": nest_it, "flav": flav, "mode": mode, "w": {k: float(v) for k, v in nby.items()}, "edits": dict(nest_edits)}
            if flav == "stacked" and mode == "boost":
                nn_pr = blend_probs(itrows, shared, pit, nn_w, mode)
                (sh_hit, sh_tot), (bl_hit, bl_tot) = zero_coverage_recovery(itrows, shared, nn_pr)
                print(f"  it multi_plain spine-coverage: shared {sh_hit}/{sh_tot}={sh_hit/max(sh_tot,1):.3f}"
                      f" -> boosted {bl_hit}/{bl_tot}={bl_hit/max(bl_tot,1):.3f}")
                ptr_b = per_type_recall(itrows, base_edits); ptr_n = per_type_recall(itrows, edits_by_w[nn_w])
                print("  per-type IoU>=.5 recall (base -> boost nn):")
                for key in sorted(ptr_b):
                    rb, nb = ptr_b[key]; rn, _ = ptr_n.get(key, (0, 0))
                    print(f"    {key[1]:14s} n={nb}  {rb:.3f} -> {rn:.3f}")

    # ---- overall nested on N3 base (de/en frozen at .4237/.8067) ----
    de_ne, en_ne = 0.4237, 0.8067
    overall = (de_ne + en_ne + best_overall["it"]) / 3
    print(f"\n================ LEVER 1 SUMMARY ================")
    print(f"  best honest it nested = {best_overall['it']:.4f} ({best_overall['mode']}, w={best_overall['w']})  base {base_it:.4f}")
    print(f"  => overall nested (de .4237 en .8067 it {best_overall['it']:.4f}) = {overall:.4f}  (N3 base 0.5503)")
    print(f"[lever1 {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()

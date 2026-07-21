"""N2 EDA: de multi_marked structure, de unseen-stem paired collapse fallbacks, en surfaces.
Leak-appropriate per-fold learning (m2/m3 build_stores fit on fold!=k, applied to fold==k).
No literal encoded content strings are hardcoded; everything printed is derived from data.
"""
import os, sys, json, collections
M4 = os.path.expanduser("~/insled/runs/M4")
sys.path.insert(0, M4)
sys.path.insert(0, os.path.expanduser("~/insled/solution"))
import numpy as np, pandas as pd
import pipeline, m2_ext, m3_ext
import elru
from transducer import Transducer, _norm as _tnorm

ROOT = pipeline.ROOT
MARKS = set(":*∗/")


def load():
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id")
    train["edits"] = train.edits_json.apply(json.loads)
    test = pd.read_csv(os.path.join(ROOT, "dataset", "test.csv"))
    return train, test


def span_type(lang, src):
    nt = len(src.split()); marked = any(c in MARKS for c in src)
    return ("single" if nt == 1 else "multi") + ("_marked" if marked else "_plain")


def toks(t):
    import re
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", t)]


def main():
    train, test = load()
    oof_tok = pd.read_csv(os.path.join(M4, "oof_token_probs.csv"))
    # map id -> list of (start,end,proba,y)
    probmap = collections.defaultdict(list)
    for r in oof_tok.itertuples():
        probmap[r.id].append((r.start, r.end, r.proba, r.y))

    print("=" * 70)
    print("PART 1: de span-type census + multi_marked detection status (thr=.07)")
    print("=" * 70)
    de = train[train.language == "de"]
    typecount = collections.Counter()
    for r in de.itertuples():
        for e in r.edits:
            if e["replacement"] == "":
                continue
            src = r.text[e["start"]:e["end"]]
            typecount[span_type("de", src)] += 1
    print("de edited span types:", dict(typecount))

    # multi_marked spans: analyze token structure + detection
    mm = []
    for r in de.itertuples():
        for e in r.edits:
            src = r.text[e["start"]:e["end"]]
            if e["replacement"] == "" or span_type("de", src) != "multi_marked":
                continue
            mm.append((r.id, e["start"], e["end"], src, e["replacement"]))
    print(f"\nde multi_marked spans: {len(mm)}")
    # for each, count tokens, marked tokens, whether all tokens marked, detection @.07
    allmarked = 0; connector_present = 0
    det_full = det_partial = det_none = 0
    thr = 0.07
    ntok_hist = collections.Counter()
    for (rid, s, e, src, rep) in mm:
        tks = src.split()
        ntok_hist[len(tks)] += 1
        nmarked = sum(1 for t in tks if any(c in MARKS for c in t))
        if nmarked == len(tks):
            allmarked += 1
        # any interior lowercase short token (connector-like)?
        if any(t.islower() and 2 <= len(t.strip(".,;:")) <= 4 for t in tks[1:-1]) if len(tks) > 2 else False:
            connector_present += 1
        # detection: fraction of span tokens with prob>=thr
        pr = probmap.get(rid, [])
        span_toks = [(a, b, p, y) for (a, b, p, y) in pr if a >= s and b <= e]
        if span_toks:
            frac = sum(1 for (a, b, p, y) in span_toks if p >= thr) / len(span_toks)
            if frac >= 0.99:
                det_full += 1
            elif frac > 0:
                det_partial += 1
            else:
                det_none += 1
    print(f"  ntok histogram: {dict(sorted(ntok_hist.items()))}")
    print(f"  all-tokens-marked: {allmarked}/{len(mm)}   interior-connector-present: {connector_present}/{len(mm)}")
    print(f"  detection@.07: full={det_full} partial={det_partial} none={det_none} (of {len(mm)})")

    # Adjacent-marked RUN analysis: how many mm spans are pure runs of 2+ marked tokens
    pure_run = 0
    for (rid, s, e, src, rep) in mm:
        tks = src.split()
        marks = [any(c in MARKS for c in t) for t in tks]
        if len(tks) >= 2 and all(marks):
            pure_run += 1
    print(f"  pure adjacent-marked runs (all tokens marked, 2+): {pure_run}/{len(mm)}")

    print("\n" + "=" * 70)
    print("PART 2: de unseen-stem paired-collapse fallback chrF (leak-free per fold)")
    print("=" * 70)
    # Build per-fold M2 stores; for each de multi-token unmarked paired edit in fold k,
    # check collapse memory (fold!=k). If MISS, compute 4 fallbacks' chrF vs truth.
    stores_by_fold = {}
    for k in range(5):
        st = {}
        m2_ext.build_stores(train[train.fold != k], st)
        stores_by_fold[k] = st

    def lcp(a, b):
        n = min(len(a), len(b)); i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def norm_core(s):
        import re
        return re.sub(r"\s+", " ", s.strip()).strip(".,;:()»«\"'")

    fbk = {"identity": [], "masc_only": [], "fem_only": [], "sufrw": []}
    n_paired = 0; n_miss = 0; n_hit_exact = 0; n_hit_stem = 0
    miss_examples = []
    for r in de.itertuples():
        k = r.fold; st = stores_by_fold[k]
        conn = st["connectors"]; femsuf = st["femsuf"]
        ce = st["collapse_exact"]; cs = st["collapse_stem"]; csr = st["collapse_sufrw"]
        for e in r.edits:
            src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
            if rep == "" or any(c in MARKS for c in src):
                continue
            tks = [t.strip(".,;:") for t in src.split()]
            if len(tks) < 2:
                continue
            # paired? connector present OR fem-adjacent
            paired = any(t.lower() in conn for t in tks)
            if not paired:
                for i in range(len(tks) - 1):
                    cp = lcp(tks[i].lower(), tks[i + 1].lower())
                    if cp >= 3 and (tks[i][cp:].lower() in femsuf or tks[i + 1][cp:].lower() in femsuf):
                        paired = True; break
            if not paired:
                continue
            n_paired += 1
            ncore = norm_core(src)
            if ncore in ce:
                n_hit_exact += 1; continue
            first = src.split()[0].strip(".,;:").lower()
            stkey = first[:max(4, int(len(first) * 0.6))]
            if stkey in cs:
                n_hit_stem += 1; continue
            # MISS -> compute fallbacks
            n_miss += 1
            # identify fem vs masc token
            words = [t.strip(".,;:") for t in src.split()]
            # pick the two content tokens (first and last non-connector caps)
            content = [w for w in words if w and w.lower() not in conn]
            if len(content) >= 2:
                a, b = content[0], content[-1]
            else:
                a, b = words[0], words[-1]
            cp = lcp(a.lower(), b.lower())
            sa, sb = a[cp:].lower(), b[cp:].lower()
            fem_tok, masc_tok = (a, b) if len(sa) >= len(sb) else (b, a)
            # sufrw fallback (participle) from stores
            sufrw_rep = None
            base = first
            for L in range(min(6, len(base)), 2, -1):
                if len(base) > L:
                    key = (base[len(base) - L - 1], base[-L:])
                    if key in csr:
                        rawbase = src.split()[0].strip(".,;:")
                        sufrw_rep = rawbase[:len(rawbase) - L] + csr[key]
                        break
            cand = {
                "identity": src,
                "masc_only": masc_tok,
                "fem_only": fem_tok,
                "sufrw": sufrw_rep if sufrw_rep is not None else src,
            }
            for name, c in cand.items():
                fbk[name].append(elru.replacement_chrf(c, rep))
            if len(miss_examples) < 12:
                miss_examples.append((len(src.split()), src[:40], rep[:40], masc_tok[:24], fem_tok[:24]))
    print(f"de paired edits: {n_paired}  hit_exact={n_hit_exact} hit_stem={n_hit_stem}  MISS(unseen-stem)={n_miss}")
    print("fallback mean replacement_chrF on MISS set:")
    for name in ("identity", "masc_only", "fem_only", "sufrw"):
        v = fbk[name]
        print(f"   {name:12s} chrF={np.mean(v):.4f}  (n={len(v)})")
    print("miss examples (ntok, src, truth_rep, masc_tok, fem_tok):")
    for ex in miss_examples:
        print("   ", ex)

    print("\n" + "=" * 70)
    print("PART 3: en edited-surface structure (slash vs plain closed-class)")
    print("=" * 70)
    en = train[train.language == "en"]
    en_edit_types = collections.Counter()
    slash_src = plain_src = 0
    edited_surfaces = collections.Counter()
    cap_surfaces = 0
    punct_attached = 0
    for r in en.itertuples():
        for e in r.edits:
            src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
            if rep == "":
                continue
            en_edit_types[span_type("en", src)] += 1
            if "/" in src:
                slash_src += 1
            else:
                plain_src += 1
            core = src.strip(".,;:()»«\"'“”’`-")
            edited_surfaces[core.lower()] += 1
            if core[:1].isupper():
                cap_surfaces += 1
            if src != core:
                punct_attached += 1
    print("en edited span types:", dict(en_edit_types))
    print(f"en edited spans: slash_src={slash_src} plain_src={plain_src} (slash frac={slash_src/max(slash_src+plain_src,1):.3f})")
    print(f"en edited: capitalized-surface={cap_surfaces} punct-attached-src={punct_attached}")
    print(f"en distinct edited surfaces (lowered): {len(edited_surfaces)}  most common: {edited_surfaces.most_common(10)}")

    # en test: slash rate in test text
    en_test = test[test.language == "en"]
    import re
    slashform = re.compile(r"[^\W\d_]/[^\W\d_]", re.UNICODE)
    rows_with_slash = sum(1 for r in en_test.itertuples() if slashform.search(r.text))
    print(f"en TEST rows with any slash-form: {rows_with_slash}/{len(en_test)} ({rows_with_slash/len(en_test):.3f})")
    en_tr = train[train.language == "en"]
    rows_with_slash_tr = sum(1 for r in en_tr.itertuples() if slashform.search(r.text))
    print(f"en TRAIN rows with any slash-form: {rows_with_slash_tr}/{len(en_tr)} ({rows_with_slash_tr/len(en_tr):.3f})")


if __name__ == "__main__":
    main()

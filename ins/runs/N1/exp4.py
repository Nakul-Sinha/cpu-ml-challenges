"""N1 exp4: dedicated it SLASH-TRANSDUCER. Learn per-token slash transforms from BOTH
single and multi-token edits (multi internal nouns get no A2 suffix rule today), apply
src-first slash order. Replacement-only (boundaries untouched) -> de/en safe, no new FPs.
Measured net, leak-free per fold.
"""
import os, sys, json, time, collections, re, pickle
ROOT = os.path.expanduser("~/insled")
sys.path.insert(0, os.path.join(ROOT, "runs", "M4"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import numpy as np
import pandas as pd
import pipeline, m4_ext
from transducer import Transducer, _norm
from run_m4 import base_cache, base_select, group_consistency, score_edits, fp_counts, SHIP_VOTE_LANGS
import elru

LANGS = pipeline.LANGS
_STRIP = ".,;:()»«\"'“”’`-–—"
MARKS = set(":*∗/")
WS = re.compile(r"\S+")


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


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def learn_slash(trdf):
    """per-fold it slash rules. For each aligned (src_tok, rep_tok) pair (single-token
    edits AND token-aligned multi-token edits with equal token count), learn:
      key = last-2 chars of src core -> Counter of (ssuf, rsuf) where rep = base + rsuf,
      restricted to rsuf containing '/' (slash forms) and src-first order.
    Returns rules[key] = (ssuf, rsuf, support, srcfirst_frac)."""
    tail = collections.defaultdict(collections.Counter)
    order = collections.Counter()
    for r in trdf[trdf.language == "it"].itertuples():
        for e in r.edits:
            src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
            if rep == "" or any(c in MARKS for c in src if c != "/"):
                continue
            st = src.split(); rt = rep.split()
            pairs = []
            if len(st) == 1 and len(rt) == 1:
                pairs = [(st[0], rt[0])]
            elif len(st) == len(rt) and 2 <= len(st) <= 4:
                pairs = list(zip(st, rt))
            for s_tok, r_tok in pairs:
                sc = s_tok.strip(_STRIP); rc = r_tok.strip(_STRIP)
                if not sc or "/" not in rc:
                    continue
                # order: does src-core appear first in the slash form?
                if rc.count("/") == 1:
                    a, b = rc.split("/")
                    if sc == a:
                        order["first"] += 1
                    elif sc == b:
                        order["second"] += 1
                cp = lcp(sc, rc)
                ssuf = sc[cp:]; rsuf = rc[cp:]
                if len(rsuf) > 12:
                    continue
                key = sc[-2:].lower() if len(sc) >= 2 else sc.lower()
                tail[key][(ssuf, rsuf)] += 1
    rules = {}
    for key, c in tail.items():
        (ssuf, rsuf), n = c.most_common(1)[0]
        tot = sum(c.values())
        if tot >= 3 and n / tot >= 0.5 and "/" in rsuf:
            rules[key] = (ssuf, rsuf, tot)
    srcfirst = order["first"] / max(1, order["first"] + order["second"])
    return dict(rules=rules, srcfirst=srcfirst)


def slash_one(core, rules):
    """apply learned slash rule to a single core (returns None if no rule)."""
    key = core[-2:].lower() if len(core) >= 2 else core.lower()
    rule = rules["rules"].get(key)
    if not rule:
        return None
    ssuf, rsuf, _ = rule
    if ssuf and not core.lower().endswith(ssuf.lower()):
        return None
    base = core[:len(core) - len(ssuf)] if ssuf else core
    return base + rsuf


def postfix_it(edits, text, rules, T, mode):
    """rewrite it replacements. mode: 'reorder' | 'slash' | 'slashmulti'."""
    out = []
    for e in edits:
        s, en = e["start"], e["end"]; src = text[s:en]; rep = e["replacement"]
        toks = src.split()
        if mode in ("slash", "slashmulti"):
            if len(toks) == 1 and not any(c in MARKS for c in src if c != "/"):
                core = src.strip(_STRIP)
                pre = src[:src.index(core)] if core and core in src else ""
                post = src[src.index(core) + len(core):] if core and core in src else ""
                nv = slash_one(core, rules)
                if nv is not None:
                    rep = pre + nv + post
            elif mode == "slashmulti" and len(toks) >= 2 and "/" not in src and not any(c in MARKS for c in src):
                # per-token compose using slash rules; fall back to A2 token if no rule
                parts = [(m.start(), m.end(), m.group()) for m in WS.finditer(src)]
                buf = []; prev = 0; any_rule = False
                for (a, b, tok) in parts:
                    buf.append(src[prev:a])
                    core = tok.strip(_STRIP)
                    pre = tok[:tok.index(core)] if core and core in tok else ""
                    post = tok[tok.index(core) + len(core):] if core and core in tok else ""
                    nv = slash_one(core, rules) if core else None
                    if nv is not None:
                        buf.append(pre + nv + post); any_rule = True
                    else:
                        buf.append(tok)
                    prev = b
                buf.append(src[prev:])
                if any_rule:
                    rep = "".join(buf)
        # reorder src-first for single slash forms
        if rep.count("/") == 1 and " " not in rep and "/" in rep:
            core = src.strip(_STRIP)
            a, b = rep.split("/")
            if core == b and core != a:
                rep = core + "/" + a
        out.append({"start": s, "end": en, "replacement": rep[:160]})
    return out


def main():
    t0 = time.time()
    S = pickle.load(open(os.path.join(ROOT, "runs", "N1_state.pkl"), "rb"))
    rows = S["rows"]; idfold = S["idfold"]; row_proba = S["row_proba"]; gbi = S["group_by_id"]
    rows_by_id = {R["id"]: R for R in rows}
    train = pd.read_csv(os.path.join(ROOT, "dataset", "train.csv"))
    folds = pd.read_csv(os.path.join(ROOT, "solution", "folds.csv"))
    train = train.merge(folds, on="id"); train["edits"] = train.edits_json.apply(json.loads)
    trs, stf = rebuild(train)
    slash = {k: learn_slash(train[train.fold != k]) for k in range(5)}
    print(f"[rebuild {time.time()-t0:.0f}s] srcfirst_frac fold0={slash[0]['srcfirst']:.2f} "
          f"nrules fold0={len(slash[0]['rules'])}")
    bcache = base_cache(rows, idfold, row_proba, trs, stf)

    def fixed_cache(mode):
        out = {}
        for rid, thrmap in bcache.items():
            R = rows_by_id[rid]
            if R["lang"] != "it" or mode is None:
                out[rid] = thrmap; continue
            k = idfold[rid]; ru = slash[k]; T = trs[k]
            out[rid] = {thr: postfix_it(ed, R["text"], ru, T, mode) for thr, ed in thrmap.items()}
        return out

    def evaluate(cache, tag):
        nn_thr, nn_e, nby, ne_e = base_select(rows, cache)
        nn = group_consistency({i: nn_e[i] for i in nn_e}, rows_by_id, gbi, trs, stf, idfold,
                               vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
        ne = group_consistency({i: ne_e[i] for i in ne_e}, rows_by_id, gbi, trs, stf, idfold,
                               vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
        nn_s, nn_d = score_edits(rows, nn); ne_s, ne_d = score_edits(rows, ne)
        fp = fp_counts(rows, nn)
        print(f"{tag:20s} NEST ov={ne_s:.4f} it={ne_d['it']['lang_score']:.4f}"
              f"(e{ne_d['it']['edited_mean']:.3f}/u{ne_d['it']['unchanged_mean']:.3f}) | "
              f"NN ov={nn_s:.4f} it={nn_d['it']['lang_score']:.4f} FPit={fp['it'][0]} "
              f"de={ne_d['de']['lang_score']:.3f} en={ne_d['en']['lang_score']:.3f} it_thr={nn_thr['it']}")

    evaluate(fixed_cache(None), "E0 base")
    evaluate(fixed_cache("reorder"), "E1 reorder")
    evaluate(fixed_cache("slash"), "E2 slash-single")
    evaluate(fixed_cache("slashmulti"), "E3 slash+multi")
    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

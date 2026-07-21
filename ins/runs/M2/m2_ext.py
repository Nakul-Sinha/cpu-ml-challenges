"""M2 GERMAN SPECIALIST plug-ins for the Institutional Edit Ledger Recovery pipeline.

Registers into pipeline.py's four extension registries (NO core-logic fork):
  (d) STORE_BUILDERS       -> learn connectors / feminine+base suffix classes /
                              paired-form collapse memories / candidate reranker;
                              install stores['span_scorer'].
  (b) SPAN_CANDIDATE_GENERATORS -> paired-form span candidates (tokA CONN tokB with
                              shared stem or feminine/base suffix pair; 3-4 tok list
                              variants with trailing commas).
  (c) REPLACEMENT_HOOKS    -> collapse detected German paired forms to the learned
                              neutral lexeme (stem memory -> participle rewrite ->
                              defer to A2), preserving whitespace + trailing punct.
  (a) TOKEN_FEATURE_EXTRAS -> de-only contextual features for single_plain
                              (article slash-doubling near gendered / marked nouns).

Everything is learned from fold-train at runtime (connectors DISCOVERED via interior
rate, suffix classes induced, memories counted). No literal encoded content strings.
A real LightGBM reranker scores candidates. Leak-free: every table is rebuilt per
fold inside STORE_BUILDERS from that fold's train_df.

Toggles (env): M2_GEN=1 paired-form gen+collapse; M2_FEAT=1 token extras;
M2_RERANK=1 use LGBM reranker gate; M2_ADMIT=<float> reranker admission threshold.
"""
import os, re, json, collections
import numpy as np

MARKS = set(":*∗/")
_WS = re.compile(r"\S+")

# module-global learned tables, refreshed per fold by build_stores (for token extras,
# which do not receive `stores`); span paths use stores[...] directly.
M2G = {"connectors": set(), "femsuf": set(), "basesuf": set(), "gendered_suf": set()}


def _lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _stem_ratio(a, b):
    return _lcp(a.lower(), b.lower()) / max(len(a), len(b), 1)


def _marked(s):
    return any(c in MARKS for c in s)


def _norm_core(s):
    return re.sub(r"\s+", " ", s.strip()).strip(".,;:()»«\"'")


def _strip_affix(s):
    """-> (lead_ws_punct, core, trail_ws_punct) preserving originals for re-attach."""
    m = re.match(r"^(\s*[.,;:(«»\"']*\s*)(.*?)(\s*[.,;:)«»\"']*\s*)$", s, re.S)
    if not m or m.group(2) == "":
        return "", s, ""
    return m.group(1), m.group(2), m.group(3)


# ======================================================================
# (d) STORE BUILDER  (leak-free: called per fold with that fold's train_df)
# ======================================================================
def build_stores(train_df, stores):
    de = train_df[train_df.language == "de"]

    def edits_of(r):
        return r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)

    # ---- connector discovery: high interior-rate tokens of multi-token de spans ----
    interior = collections.Counter(); tot = collections.Counter()
    for r in de.itertuples():
        for e in edits_of(r):
            src = r.text[e["start"]:e["end"]]; tks = src.split()
            if len(tks) >= 2 and e["replacement"] != "":
                for i, t in enumerate(tks):
                    tc = t.strip(".,;:"); tot[tc] += 1
                    if 0 < i < len(tks) - 1:
                        interior[tc] += 1
    connectors = set(t for t, c in interior.items()
                     if c >= 3 and c / max(tot[t], 1) >= 0.6 and 2 <= len(t) <= 6 and t.islower())

    # ---- feminine / base suffix classes from plain 3-tok paired forms A CONN B ----
    femc = collections.Counter(); basec = collections.Counter(); gendc = collections.Counter()
    for r in de.itertuples():
        for e in edits_of(r):
            src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
            tks = [t.strip(".,;:") for t in src.split()]
            if len(tks) == 3 and tks[1] in connectors and not _marked(src) and rep:
                a, b = tks[0], tks[2]
                if _stem_ratio(a, b) >= 0.5 and a and b:
                    cp = _lcp(a.lower(), b.lower()); sa, sb = a[cp:].lower(), b[cp:].lower()
                    if len(sa) >= len(sb):
                        femc[sa] += 1; basec[sb] += 1
                    else:
                        femc[sb] += 1; basec[sa] += 1
    femsuf = set(s for s, c in femc.items() if c >= 2 and s)
    basesuf = set(s for s, c in basec.items() if c >= 2)
    # also single-token feminine markers (…innen/…in surface, learned from marked singles)
    for r in de.itertuples():
        for e in edits_of(r):
            src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
            if len(src.split()) == 1 and rep and _stem_ratio(src, rep) < 0.95:
                core = src.strip(".,;:")
                if len(core) >= 4:
                    gendc[core[-4:].lower()] += 1
    gendered_suf = set(s for s, c in gendc.items() if c >= 3)

    # ---- collapse memories: exact paired-span core -> rep core; first-stem -> rep ----
    ex = collections.defaultdict(collections.Counter)
    stem = collections.defaultdict(collections.Counter)
    sufrw = collections.defaultdict(collections.Counter)   # participle backoff
    for r in de.itertuples():
        for e in edits_of(r):
            src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
            tks = src.split()
            if len(tks) >= 2 and not _marked(src) and rep:
                ex[_norm_core(src)][_norm_core(rep)] += 1
                base = tks[0].strip(".,;:").lower()
                st = base[:max(4, int(len(base) * 0.6))]
                nr = _norm_core(rep)
                if " " not in nr:
                    stem[st][nr] += 1
                    # suffix rewrite first-token -> single-word rep (participle)
                    cp = _lcp(base, nr.lower())
                    if cp >= 3:
                        ssuf, rsuf = base[cp:], nr[cp:]
                        if len(ssuf) <= 6 and len(rsuf) <= 8:
                            sufrw[(base[cp - 1], ssuf)][rsuf] += 1
    collapse_exact = {k: c.most_common(1)[0][0] for k, c in ex.items()}
    collapse_stem = {k: c.most_common(1)[0][0] for k, c in stem.items()
                     if sum(c.values()) >= 2 and c.most_common(1)[0][1] / sum(c.values()) >= 0.5}
    collapse_sufrw = {k: c.most_common(1)[0][0] for k, c in sufrw.items()
                      if sum(c.values()) >= 3 and c.most_common(1)[0][1] / sum(c.values()) >= 0.5}

    stores.update(connectors=connectors, femsuf=femsuf, basesuf=basesuf,
                  gendered_suf=gendered_suf, collapse_exact=collapse_exact,
                  collapse_stem=collapse_stem, collapse_sufrw=collapse_sufrw)
    M2G["connectors"] = connectors; M2G["femsuf"] = femsuf
    M2G["basesuf"] = basesuf; M2G["gendered_suf"] = gendered_suf

    # ---- (optional) LightGBM candidate reranker, trained on fold-train candidates ----
    if os.environ.get("M2_RERANK", "0") == "1":
        _train_reranker(de, stores)

    # ---- install span scorer (admission gate) ----
    admit_thr = float(os.environ.get("M2_ADMIT", "0.0"))
    fem_strict = os.environ.get("M2_FEMSTRICT", "1") == "1"

    def span_scorer(cands, tokens, lang, text, aux):
        if lang != "de":
            return []
        st = aux["stores"]; rr = st.get("reranker")
        out = []
        for (a, b, meta) in cands:
            if fem_strict and not meta.get("fem"):
                continue
            score = 0.5 + 0.3 * meta.get("sr", 0.0) + 0.2 * (1.0 if meta.get("fem") else 0.0)
            if rr is not None:
                pr = rr["model"].predict_proba(np.array([[meta.get(k, 0.0) for k in rr["feats"]]]))[0, 1]
                if pr < admit_thr:
                    continue
                score = float(pr)
            out.append((a, b, score))
        return out

    stores["span_scorer"] = span_scorer


def _cand_features(meta):
    return dict(sr=meta.get("sr", 0.0), fem=1.0 if meta.get("fem") else 0.0,
                ntok=meta.get("ntok", 0), is_und=meta.get("is_und", 0.0),
                lenA=meta.get("lenA", 0), lenB=meta.get("lenB", 0))


def _train_reranker(de, stores):
    import lightgbm as lgb
    X = []; y = []
    for r in de.itertuples():
        tks = [(m.start(), m.end(), m.group()) for m in _WS.finditer(r.text)]
        truesp = [(e["start"], e["end"]) for e in (r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)) if e["replacement"] != ""]
        for (si, ej, meta) in _generate(tks, stores):
            a, b = tks[si][0], tks[ej][1]; best = 0.0
            for (ts, te) in truesp:
                ov = max(0, min(b, te) - max(a, ts)); best = max(best, ov / max(1, (max(b, te) - min(a, ts))))
            f = _cand_features(meta); feats = sorted(f.keys())
            X.append([f[k] for k in feats]); y.append(1 if best >= 0.5 else 0)
    if len(set(y)) < 2:
        return
    feats = sorted(_cand_features({}).keys())
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15,
                           min_child_samples=8, reg_lambda=1.0, verbosity=-1, n_jobs=5)
    m.fit(np.array(X), np.array(y))
    stores["reranker"] = {"model": m, "feats": feats}


# ======================================================================
# (b) SPAN CANDIDATE GENERATOR
# ======================================================================
def _generate(tokens, stores):
    """emit (start_tok_idx, end_tok_idx, meta) paired-form candidates (inclusive)."""
    conn = stores.get("connectors", set()); femsuf = stores.get("femsuf", set())
    n = len(tokens); words = [w for _, _, w in tokens]
    cores = [w.strip(".,;:") for w in words]
    out = []
    for i in range(1, n - 1):
        if cores[i].lower() in conn:
            a, b = cores[i - 1], cores[i + 1]
            if not a or not b:
                continue
            if not (a[:1].isupper() and b[:1].isupper()):
                continue
            cp = _lcp(a.lower(), b.lower()); sa, sb = a[cp:].lower(), b[cp:].lower()
            fem = (sa in femsuf or sb in femsuf)
            sr = _stem_ratio(a, b)
            if sr < 0.5 and not fem:
                continue
            si, ej = i - 1, i + 1
            # extend left across comma-listed same-stem tokens: "X, Y und Z"
            k = si - 1
            while k - 1 >= 0 and words[k].endswith(",") and _stem_ratio(cores[k], a) >= 0.4:
                si = k; k -= 1
            meta = dict(sr=sr, fem=fem, ntok=ej - si + 1,
                        is_und=1.0 if cores[i].lower() == min(conn, key=len, default="") else 0.0,
                        lenA=tokens[si][1] - tokens[si][0], lenB=tokens[ej][1] - tokens[ej][0])
            out.append((si, ej, meta))
    return out


def span_generator(tokens, lang, text, aux):
    if lang != "de":
        return []
    return _generate(tokens, aux["stores"])


# ======================================================================
# (c) REPLACEMENT HOOK -- collapse German paired forms
# ======================================================================
def _looks_paired(core, stores):
    conn = stores.get("connectors", set()); femsuf = stores.get("femsuf", set())
    tks = [t.strip(".,;:") for t in core.split()]
    if len(tks) < 2:
        return False
    if any(t.lower() in conn for t in tks):
        return True
    # innen/base adjacency without explicit connector
    for i in range(len(tks) - 1):
        cp = _lcp(tks[i].lower(), tks[i + 1].lower())
        if cp >= 3 and (tks[i][cp:].lower() in femsuf or tks[i + 1][cp:].lower() in femsuf):
            return True
    return False


def collapse_hook(lang, src, context, stores):
    if lang != "de":
        return None
    lead, core, trail = _strip_affix(src)
    if not _looks_paired(core, stores):
        return None
    ncore = _norm_core(core)
    ce = stores.get("collapse_exact", {})
    if ncore in ce:
        return lead + ce[ncore] + trail
    cs = stores.get("collapse_stem", {})
    first = core.split()[0].strip(".,;:").lower()
    st = first[:max(4, int(len(first) * 0.6))]
    if st in cs:
        return lead + cs[st] + trail
    # participle suffix-rewrite backoff
    sr = stores.get("collapse_sufrw", {})
    for L in range(min(6, len(first)), 2, -1):
        key = (first[len(first) - L - 1], first[-L:]) if len(first) > L else None
        if key and key in sr:
            base = core.split()[0].strip(".,;:")
            neut = base[:len(base) - L] + sr[key]
            return lead + neut + trail
    return None  # defer to A2 default


# ======================================================================
# (a) TOKEN FEATURE EXTRAS -- de single_plain context (article slash-doubling)
# ======================================================================
def token_extras(tokens, i, lang, text):
    keys = ("de_short_lower", "de_next_marked", "de_next_fem", "de_next_cap",
            "de_prev_marked", "de_dist_mark", "de_gendered_self")
    if lang != "de":
        return {k: 0.0 for k in keys}
    words = [w for _, _, w in tokens]; n = len(words)
    w = words[i]; core = w.strip(".,;:()»«\"'")
    femsuf = M2G.get("femsuf", set()); gend = M2G.get("gendered_suf", set())

    def is_marked(t):
        return 1.0 if any(c in MARKS for c in t) else 0.0

    def is_fem(t):
        c = t.strip(".,;:");
        return 1.0 if (len(c) >= 3 and (c[-5:].lower() in femsuf or c[-4:].lower() in femsuf or c[-4:].lower() in gend)) else 0.0

    nxt = words[i + 1] if i + 1 < n else ""
    prv = words[i - 1] if i - 1 >= 0 else ""
    dist = 9
    for d in range(1, 6):
        if (i + d < n and is_marked(words[i + d])) or (i - d >= 0 and is_marked(words[i - d])):
            dist = d; break
    return {
        "de_short_lower": 1.0 if (core.isalpha() and core.islower() and len(core) <= 5) else 0.0,
        "de_next_marked": is_marked(nxt),
        "de_next_fem": is_fem(nxt),
        "de_next_cap": 1.0 if nxt[:1].isupper() else 0.0,
        "de_prev_marked": is_marked(prv),
        "de_dist_mark": float(dist),
        "de_gendered_self": 1.0 if (len(core) >= 4 and core[-4:].lower() in gend) else 0.0,
    }


# ======================================================================
# registration
# ======================================================================
def register(P):
    # gen-only is the measured-best M2 config; token extras (M2_FEAT) regress de
    # detection (-0.005 overall, no single_plain gain) so they default OFF.
    gen_on = os.environ.get("M2_GEN", "1") == "1"
    feat_on = os.environ.get("M2_FEAT", "0") == "1"
    P.STORE_BUILDERS = list(P.STORE_BUILDERS)
    P.SPAN_CANDIDATE_GENERATORS = list(P.SPAN_CANDIDATE_GENERATORS)
    P.REPLACEMENT_HOOKS = list(P.REPLACEMENT_HOOKS)
    P.TOKEN_FEATURE_EXTRAS = list(P.TOKEN_FEATURE_EXTRAS)
    # store builder always on (installs span_scorer + memories used by hooks/gen)
    P.STORE_BUILDERS.append(build_stores)
    if gen_on:
        P.SPAN_CANDIDATE_GENERATORS.append(span_generator)
        P.REPLACEMENT_HOOKS.append(collapse_hook)
    if feat_on:
        P.TOKEN_FEATURE_EXTRAS.append(token_extras)
    return P

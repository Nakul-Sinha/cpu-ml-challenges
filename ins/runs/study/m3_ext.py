"""M3 plug-ins for the M1 pipeline -- Italian + EN + Deletions specialist.

Registered onto pipeline.py's extension registries (never forks core logic):
  STORE_BUILDERS[0]        = build_stores           (learns all tables per fold)
  TOKEN_FEATURE_EXTRAS[0]  = it_en_feats            (NP-agreement + en closed-class)
  REPLACEMENT_HOOKS        = [it_slash_hook, en_norm_hook, del_hook]  (order matters)
  SPAN_CANDIDATE_GENERATORS= [np_generator]  (+ stores['span_scorer']=np_scorer)

Everything is LEARNED from the fold-train frame at runtime -- no literal encoded
content strings (only punctuation-class literals).  Feature fns read the ACTIVE
per-fold tables through a module global that build_stores refreshes right before
each Detector.fit (leak-free: tables are fold!=k, applied to featurize fold==k).
"""
import re, collections

WS = re.compile(r"\S+")
MARKS = set(":*∗/")
_STRIP = ".,;:()»«\"'“”’`-–—"          # punctuation-class only (compliant)
# a slash flanked by letters = an inclusive slash-doubled surface form (style cue)
_SLASHFORM = re.compile(r"[^\W\d_]/[^\W\d_]", re.UNICODE)

_ACTIVE = None   # per-fold tables; refreshed by build_stores before each fit

# component toggles (run_m3 flips these; defaults = everything on)
USE_FEATS = True
USE_IT_REPL = True
USE_EN_REPL = True
USE_DEL = False           # forfeited: measured per-fold precision 0/4 << 0.6 gate
USE_NPGEN = False         # inert: gate admits only spans the detector already catches


def toks(t):
    return [(m.start(), m.end(), m.group()) for m in WS.finditer(t)]


def _strip(w):
    return w.strip(_STRIP)


def _lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ======================================================================
#  LEARNING (per fold-train frame)
# ======================================================================
def _learn_it(df):
    occ = collections.Counter(); ed = collections.Counter()
    spaninit = collections.Counter(); spaninit_slash = collections.Counter()
    end2_ed = collections.Counter(); end2_tot = collections.Counter()
    end3_ed = collections.Counter(); end3_tot = collections.Counter()
    del_first = collections.Counter()          # first token of deletion spans
    for r in df[df.language == "it"].itertuples():
        tk = toks(r.text)
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in r.edits)
        startset = {a for a, _, _ in spans}
        rep_first_slash = {}
        for a, b, rep in spans:
            fw = rep.split()[0] if rep.split() else ""
            rep_first_slash[a] = ("/" in fw)
            if rep == "":
                w0 = r.text[a:b].split()
                if w0:
                    del_first[_strip(w0[0]).lower()] += 1

        def inside(s, e):
            for a, b, _ in spans:
                if s >= a and e <= b:
                    return True
            return False
        for i, (s, e, w) in enumerate(tk):
            core = _strip(w).lower()
            if not core:
                continue
            occ[core] += 1
            isin = inside(s, e)
            if isin:
                ed[core] += 1
            if s in startset:
                spaninit[core] += 1
                if rep_first_slash.get(s):
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
    tok_edrate = {w: ed[w] / occ[w] for w in occ}
    # articles: short-ish function words that reliably begin a slash-doubled NP
    article_set = {w for w in occ
                   if occ[w] >= 3 and spaninit_slash[w] / occ[w] >= 0.40}
    # connectors: very short tokens that begin deletion coordinations (o / e)
    conn_set = {w for w, c in del_first.items() if len(w) <= 2 and c >= 2}
    end2_rate = {k: end2_ed[k] / end2_tot[k] for k in end2_tot if end2_tot[k] >= 8}
    end3_rate = {k: end3_ed[k] / end3_tot[k] for k in end3_tot if end3_tot[k] >= 6}
    return dict(occ=occ, spaninit_rate=spaninit_rate, tok_edrate=tok_edrate,
                article_set=article_set, conn_set=conn_set,
                end2_rate=end2_rate, end3_rate=end3_rate)


def _learn_it_repl(df):
    """single-token slash edits: last-2-char ending -> majority (src_suffix, rep_suffix)."""
    tail_ct = collections.defaultdict(collections.Counter)
    exact = set()
    for r in df[df.language == "it"].itertuples():
        for e in r.edits:
            src = r.text[e["start"]:e["end"]]; rep = e["replacement"]
            if len(src.split()) != 1 or rep == "":
                continue
            core = _strip(src)
            exact.add(core.lower())
            if "/" not in rep or " " in rep:
                continue
            cp = _lcp(core, rep)
            ssuf = core[cp:]; rsuf = rep[cp:]
            if len(rsuf) > 10:
                continue
            key = core[-2:].lower() if len(core) >= 2 else core.lower()
            tail_ct[key][(ssuf, rsuf)] += 1
    rules = {}
    for key, c in tail_ct.items():
        (ssuf, rsuf), n = c.most_common(1)[0]
        tot = sum(c.values())
        if tot >= 3 and n / tot >= 0.5:
            rules[key] = (ssuf, rsuf)
    return dict(rules=rules, exact=exact)


def _learn_en(df):
    occ = collections.Counter(); ed = collections.Counter()
    exact_ct = collections.defaultdict(collections.Counter)
    for r in df[df.language == "en"].itertuples():
        tk = toks(r.text)
        spans = sorted((e["start"], e["end"], e["replacement"]) for e in r.edits)
        startset = {a for a, _, _ in spans}

        def inside(s, e):
            for a, b, _ in spans:
                if s >= a and e <= b:
                    return True
            return False
        for s, e, w in tk:
            core = _strip(w).lower()
            if not core:
                continue
            occ[core] += 1
            if inside(s, e):
                ed[core] += 1
        for a, b, rep in spans:
            src = r.text[a:b]
            exact_ct[_normkey(src)][rep] += 1
    tok_rate = {w: ed[w] / occ[w] for w in occ}
    cc_set = {w for w in occ if occ[w] >= 2 and tok_rate[w] >= 0.5}
    norm_mem = {}
    for k, c in exact_ct.items():
        rep, n = c.most_common(1)[0]
        if n / sum(c.values()) >= 0.6:
            norm_mem[k] = rep
    return dict(occ=occ, tok_rate=tok_rate, cc_set=cc_set, norm_mem=norm_mem)


def _normkey(s):
    t = s.lower().strip()
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(_STRIP)


def _learn_del(df):
    """duplicate-adjacency deletion detector.  A deletion candidate is a coordinated
    gendered phrase (starts with connector) whose neutralized head already appears
    immediately adjacent.  Learn connector tokens + measure how often the pattern holds."""
    conn = collections.Counter()
    for lang in ("it", "de"):
        for r in df[df.language == lang].itertuples():
            for e in r.edits:
                if e["replacement"] == "":
                    w = r.text[e["start"]:e["end"]].split()
                    if w:
                        conn[(lang, _strip(w[0]).lower())] += 1
    conn_set = {k for k, c in conn.items() if c >= 2 and len(k[1]) <= 3}
    return dict(conn_set=conn_set)


def build_stores(train_df, stores):
    global _ACTIVE
    tables = dict(it=_learn_it(train_df), it_repl=_learn_it_repl(train_df),
                  en=_learn_en(train_df), dele=_learn_del(train_df))
    stores["m3"] = tables
    _ACTIVE = tables
    if USE_NPGEN:
        stores["span_scorer"] = np_scorer


# ======================================================================
#  (a) TOKEN_FEATURE_EXTRAS  -- one fn, STABLE key set
# ======================================================================
_KEYS = ["it_art", "it_art_score", "it_tok_edr", "it_end2", "it_end3",
         "it_next_end2", "it_prev_art", "it_prev_conn", "it_is_conn",
         "it_chain", "it_slashwin", "en_cc", "en_cc_rate"]
# NOTE: row-level slash-density + distance-to-article were MEASURED and REGRESSED
# (nested 0.5064 -> 0.5016): slash-density correlates with edited rows and raised
# FPs on unchanged rows inside inclusive-style docs.  Dropped by measurement.


def it_en_feats(tokens, i, lang, text):
    d = {k: 0.0 for k in _KEYS}
    T = _ACTIVE
    if T is None or not USE_FEATS:
        return d
    w = tokens[i][2]
    core = _strip(w).lower()
    if not core:
        return d
    if lang == "it":
        it = T["it"]
        d["it_art"] = 1.0 if core in it["article_set"] else 0.0
        d["it_art_score"] = it["spaninit_rate"].get(core, 0.0)
        d["it_tok_edr"] = it["tok_edrate"].get(core, 0.0)
        if len(core) >= 2:
            d["it_end2"] = it["end2_rate"].get(core[-2:], 0.0)
        if len(core) >= 3:
            d["it_end3"] = it["end3_rate"].get(core[-3:], 0.0)
        d["it_is_conn"] = 1.0 if core in it["conn_set"] else 0.0
        if i + 1 < len(tokens):
            nc = _strip(tokens[i + 1][2]).lower()
            if len(nc) >= 2:
                d["it_next_end2"] = it["end2_rate"].get(nc[-2:], 0.0)
        if i - 1 >= 0:
            pc = _strip(tokens[i - 1][2]).lower()
            d["it_prev_art"] = 1.0 if pc in it["article_set"] else 0.0
            d["it_prev_conn"] = 1.0 if pc in it["conn_set"] else 0.0
        # agreement chain: # of tokens in [i-1..i+2] with end2 edit-rate >= 0.2
        ch = 0
        for j in range(max(0, i - 1), min(len(tokens), i + 3)):
            cj = _strip(tokens[j][2]).lower()
            if len(cj) >= 2 and it["end2_rate"].get(cj[-2:], 0.0) >= 0.2:
                ch += 1
        d["it_chain"] = float(ch)
        # slash-doubled form present in a local window -> doc uses inclusive style
        s0 = tokens[i][0]; e0 = tokens[i][1]
        win = text[max(0, s0 - 90):e0 + 90]
        d["it_slashwin"] = 1.0 if _SLASHFORM.search(win) else 0.0
    elif lang == "en":
        en = T["en"]
        d["en_cc"] = 1.0 if core in en["cc_set"] else 0.0
        d["en_cc_rate"] = en["tok_rate"].get(core, 0.0)
    return d


# ======================================================================
#  (c) REPLACEMENT_HOOKS
# ======================================================================
def it_slash_hook(lang, src, context, stores):
    """Italian single-token slash-append for tokens A2 would drop to identity.
    Defers (None) to A2 for known/multi/marked forms."""
    if lang != "it" or not USE_IT_REPL:
        return None
    T = stores.get("m3")
    if not T:
        return None
    if len(src.split()) != 1:
        return None
    if any(c in src for c in MARKS):        # marked -> A2 mark-template
        return None
    core = _strip(src)
    low = core.lower()
    rep = T["it_repl"]
    if low in rep["exact"]:                 # A2 exact-memory will handle it better
        return None
    key = low[-2:] if len(core) >= 2 else low
    rule = rep["rules"].get(key)
    if not rule:
        return None
    ssuf, rsuf = rule
    if ssuf and not core.lower().endswith(ssuf.lower()):
        # ending mismatch under case-fold; only apply pure-append rules safely
        if ssuf != "":
            return None
    pre = src[:src.index(core)] if core and core in src else ""
    post = src[src.index(core) + len(core):] if core and core in src else ""
    base = core[:len(core) - len(ssuf)] if ssuf else core
    return pre + base + rsuf + post


def en_norm_hook(lang, src, context, stores):
    """EN case/punct-normalized nearest-memory for the identity-fallback bucket."""
    if lang != "en" or not USE_EN_REPL:
        return None
    T = stores.get("m3")
    if not T:
        return None
    nm = T["en"]["norm_mem"]
    v = nm.get(_normkey(src))
    return v if v is not None else None


def del_hook(lang, src, context, stores):
    """High-precision deletion path: coordinated gendered phrase starting with a
    learned connector whose neutralized head is already present adjacently."""
    if not USE_DEL:
        return None
    T = stores.get("m3")
    if not T:
        return None
    parts = src.split()
    if not parts:
        return None
    first = _strip(parts[0]).lower()
    if (lang, first) not in T["dele"]["conn_set"]:
        return None
    # require the coordinated content to duplicate immediately-preceding text
    text = context["text"]; a = context["start"]
    left = text[max(0, a - len(src) - 6):a]
    body = " ".join(_strip(p).lower() for p in parts[1:])
    if len(body) >= 4 and any(_strip(p).lower() in left.lower() for p in parts[1:] if len(_strip(p)) >= 4):
        return ""            # delete
    return None


# ======================================================================
#  (b) SPAN_CANDIDATE_GENERATORS + scorer (conservative NP capture)
# ======================================================================
def np_generator(tokens, lang, text, aux):
    """Whole-NP candidates: learned article + 1..3 following agreeing tokens."""
    if lang != "it" or not USE_NPGEN:
        return []
    T = _ACTIVE
    if not T:
        return []
    it = T["it"]
    out = []
    n = len(tokens)
    for i in range(n):
        core = _strip(tokens[i][2]).lower()
        if core in it["article_set"]:
            for L in (1, 2, 3):
                j = i + L
                if j < n:
                    out.append((i, j, {"art": core, "si": i, "ej": j}))
    return out


def np_scorer(cands, tokens, lang, text, aux):
    """Admit an NP candidate only if article is high-confidence AND at least one
    following token has a high gendered-ending edit-rate (precision gate).
    Harness passes cands as (start_char, end_char, meta); token indices live in meta."""
    if not USE_NPGEN:
        return []
    T = _ACTIVE
    if not T:
        return []
    it = T["it"]; probs = aux["probs"]
    keep = []
    for (a_char, b_char, meta) in cands:
        art = meta["art"]; si = meta["si"]; ej = meta["ej"]
        art_sc = it["spaninit_rate"].get(art, 0.0)
        best_end = 0.0
        for k in range(si + 1, ej + 1):
            cj = _strip(tokens[k][2]).lower()
            if len(cj) >= 2:
                best_end = max(best_end, it["end2_rate"].get(cj[-2:], 0.0))
        pmean = sum(probs[si:ej + 1]) / max(1, ej + 1 - si)
        score = 0.5 * art_sc + 0.4 * best_end + 0.1 * pmean
        if art_sc >= 0.55 and best_end >= 0.35:
            keep.append((a_char, b_char, score))
    return keep


# ======================================================================
#  Registration helper
# ======================================================================
def register(P):
    P.STORE_BUILDERS = [build_stores]
    P.TOKEN_FEATURE_EXTRAS = [it_en_feats]
    P.REPLACEMENT_HOOKS = [it_slash_hook, en_norm_hook, del_hook]
    P.SPAN_CANDIDATE_GENERATORS = [np_generator]
    # reset frozen feature schema so the extra columns re-freeze
    P.FEAT_NAMES = None; P.EXTRA_NAMES = None

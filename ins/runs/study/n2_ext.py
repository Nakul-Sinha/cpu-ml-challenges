"""N2 plug-ins composed onto the M4 base (m4_ext):

  (1) de_markrun_generator  -- SPAN_CANDIDATE_GENERATOR: runs of 2+ adjacent marked
      tokens whose post-mark suffix is a LEARNED feminine-suffix class (from the
      per-fold transducer's mark templates + M2 fem/gendered suffix sets).  Admitted
      through M2's existing span_scorer (meta fem=True) INDEPENDENT of the detector
      threshold -> fixes detection of multi_marked runs the LGBM misses; A2's
      _predict_multi composes the per-token mark-template replacement.

  (2) masc_only_hook        -- REPLACEMENT_HOOK (inserted right after M2 collapse_hook):
      for de paired forms whose collapse target is NOT in memory (unseen stem), emit
      the masculine token only (drop the feminine conjunct).  MEASURED winner over
      identity / feminine-only / participle-suffix-rewrite (OOF replacement chrF
      0.699 vs 0.474 / 0.666 / 0.474).

Everything learned from the fold-train frame at runtime; no literal encoded content
strings.  Toggles: N2_MARKRUN, N2_MASCFB, N2_MARKRUN_CAP, N2_MARKRUN_BRIDGE.
"""
import os
import pipeline
import m2_ext
import m4_ext

MARKS = set(":*∗/")
_EDGE = ".,;:()»«\"'“”’`-–—"

USE_MARKRUN = os.environ.get("N2_MARKRUN", "1") == "1"
USE_MASCFB = os.environ.get("N2_MASCFB", "1") == "1"
MARKRUN_CAP = os.environ.get("N2_MARKRUN_CAP", "1") == "1"       # require capitalized core
MARKRUN_BRIDGE = os.environ.get("N2_MARKRUN_BRIDGE", "1") == "1"  # bridge one connector
# emit run only when under-detected (min token prob < MINP); 1.0 = always emit.
# Gating avoids displacing well-detected adjacent single_marked spans.
MARKRUN_MINP = float(os.environ.get("N2_MARKRUN_MINP", "1.0"))


def _lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _last_mark_suffix(core):
    """suffix (lowered, <=8) after the LAST interior mark char; None if no interior mark."""
    p = -1
    for idx in range(len(core)):
        if core[idx] in MARKS:
            p = idx
    if p == -1 or p >= len(core) - 1:
        return None
    return core[p + 1:][:8].lower()


def _marksuf_set(stores):
    """Learned post-mark feminine suffixes for de: transducer mark-template suffixes
    (these are exactly the suffixes A2 can neutralize) plus M2 fem/gendered sets."""
    s = set()
    T = stores.get("_transducer")
    if T is not None:
        for key in T.mark_tpl:
            lang, mk, suf = key
            if lang == "de" and suf:
                s.add(suf.lower())
        for key in T.mark_tpl_bo:
            pass  # backoff has no suffix; skip
    for x in stores.get("femsuf", set()):
        if x:
            s.add(x)
    for x in stores.get("gendered_suf", set()):
        if x:
            s.add(x)
    return s


# ======================================================================
# (1) de marked-run candidate generator
# ======================================================================
def de_markrun_generator(tokens, lang, text, aux):
    if lang != "de" or not USE_MARKRUN:
        return []
    stores = aux["stores"]
    marksuf = stores.get("_n2_marksuf")
    if marksuf is None:
        marksuf = _marksuf_set(stores)
        stores["_n2_marksuf"] = marksuf
    if not marksuf:
        return []
    conn = stores.get("connectors", set())
    probs = aux.get("probs")
    words = [w for _, _, w in tokens]
    cores = [w.strip(_EDGE) for w in words]
    n = len(words)

    def is_mf(i):
        c = cores[i]
        if not c:
            return False
        if MARKRUN_CAP and not c[:1].isupper():
            return False
        suf = _last_mark_suffix(c)
        return suf is not None and suf in marksuf

    out = []
    i = 0
    while i < n:
        if is_mf(i):
            j = i
            while True:
                if j + 1 < n and is_mf(j + 1):
                    j += 1
                    continue
                if (MARKRUN_BRIDGE and j + 2 < n
                        and cores[j + 1].lower() in conn and is_mf(j + 2)):
                    j += 2
                    continue
                break
            if j > i:
                emit = True
                if MARKRUN_MINP < 1.0 and probs is not None:
                    emit = min(probs[i:j + 1]) < MARKRUN_MINP
                if emit:
                    out.append((i, j, {"fem": True, "markrun": 1.0, "sr": 1.0,
                                       "ntok": j - i + 1}))
            i = j + 1
        else:
            i += 1
    return out


# ======================================================================
# (2) masc-only unseen-stem paired-collapse replacement fallback
# ======================================================================
def masc_only_hook(lang, src, context, stores):
    if lang != "de" or not USE_MASCFB:
        return None
    lead, core, trail = m2_ext._strip_affix(src)
    if not m2_ext._looks_paired(core, stores):
        return None
    # defer to M2 collapse memory when it can answer (exact / stem)
    ce = stores.get("collapse_exact", {})
    if m2_ext._norm_core(core) in ce:
        return None
    cs = stores.get("collapse_stem", {})
    first = core.split()[0].strip(".,;:").lower()
    st = first[:max(4, int(len(first) * 0.6))]
    if st in cs:
        return None
    # unseen stem -> masculine token only (shorter LCP suffix)
    conn = stores.get("connectors", set())
    words = [t.strip(".,;:") for t in core.split()]
    content = [w for w in words if w and w.lower() not in conn]
    if len(content) < 2:
        return None
    a, b = content[0], content[-1]
    cp = _lcp(a.lower(), b.lower())
    sa, sb = a[cp:].lower(), b[cp:].lower()
    masc = b if len(sa) >= len(sb) else a
    if not masc:
        return None
    return lead + masc + trail


# ======================================================================
# registration: compose onto the M4 base
# ======================================================================
def register(P=pipeline):
    m4_ext.register(P)
    if USE_MARKRUN:
        P.SPAN_CANDIDATE_GENERATORS = list(P.SPAN_CANDIDATE_GENERATORS) + [de_markrun_generator]
    if USE_MASCFB:
        hooks = list(P.REPLACEMENT_HOOKS)
        try:
            idx = hooks.index(m2_ext.collapse_hook) + 1
        except ValueError:
            idx = len(hooks)
        hooks.insert(idx, masc_only_hook)
        P.REPLACEMENT_HOOKS = hooks
    P.FEAT_NAMES = None
    P.EXTRA_NAMES = None
    return P

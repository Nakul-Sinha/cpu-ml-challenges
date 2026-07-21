"""A2 Transduction hierarchy: given oracle span text -> predicted replacement.

Everything is LEARNED from train pairs at fit() time (no literal encoded strings).
Hierarchy (most specific first), per single token / composed for multi:
  0. deletion model      trained LogisticRegression P(rep=="") -> ""  (high precision)
  1. exact memory        (lang, src) -> majority rep
  1b.normalized memory   (lang, norm(src)) -> majority rep   (case/space/pad variants)
  2. mark-template        STEM<mark>SUFFIX -> L+STEM+MID+STEM+R  (colon/star expansion)
  3. suffix-transform     replace learned src-ending with learned rep-ending (guarded)
  4. multi-token compose  whole-span memory, else per-token transduce + rejoin
  5. fallback             identity (copy src)

Importable: fit(train_df) -> Transducer with .predict(lang, src_text, context)->str
"""
import json, re, collections

WS = re.compile(r"\S+")
# characters that are morphological markers (kept as span-core), not punctuation
_MARKS = set(":*∗/")


def _lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _norm(s):
    # lowercase, collapse whitespace, drop spaces around slash, strip edge punct
    t = s.lower().strip()
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(".,;:()»«\"'")
    return t


def _split_punct(tok):
    """leading/trailing punctuation (not morphological marks) split from a core."""
    i = 0
    while i < len(tok) and not (tok[i].isalnum() or tok[i] in _MARKS):
        i += 1
    j = len(tok)
    while j > i and not (tok[j - 1].isalnum() or tok[j - 1] in _MARKS):
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


class Transducer:
    def __init__(self):
        self.exact = {}
        self.norm = {}
        self.mark_tpl = {}
        self.mark_tpl_bo = {}
        self.suffix_rules = {}
        self.del_keys = set()
        self.langs = set()
        self.del_clf = None       # trained ML deletion classifier (materially drives del outputs)
        self.model = None         # optional seq2seq scorer/generator

    # ---------- fit ----------
    def fit(self, df):
        exact_ct = collections.defaultdict(collections.Counter)
        norm_ct = collections.defaultdict(collections.Counter)
        mark_ct = collections.defaultdict(collections.Counter)
        markbo_ct = collections.defaultdict(collections.Counter)
        suf_ct = collections.defaultdict(collections.Counter)
        for r in df.itertuples():
            lang = r.language
            self.langs.add(lang)
            edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
            for e in edits:
                src = r.text[e["start"]:e["end"]]
                rep = e["replacement"]
                exact_ct[(lang, src)][rep] += 1
                norm_ct[(lang, _norm(src))][rep] += 1
                if len(src.split()) == 1:
                    self._learn_single(lang, src, rep, mark_ct, markbo_ct, suf_ct)
        self.exact = {k: c.most_common(1)[0][0] for k, c in exact_ct.items()}
        # normalized memory: only keep confident (single dominant rep) keys
        self.norm = {}
        for k, c in norm_ct.items():
            rep, n = c.most_common(1)[0]
            if n / sum(c.values()) >= 0.7:
                self.norm[k] = rep
        self.del_keys = {k for k, v in self.exact.items() if v == ""}
        self.mark_tpl = {k: c.most_common(1)[0][0] for k, c in mark_ct.items()}
        self.mark_tpl_bo = {k: c.most_common(1)[0][0] for k, c in markbo_ct.items()}
        # suffix rules: guarded -- no whitespace in replacement ending, support & consistency
        self.suffix_rules = {}
        for k, c in suf_ct.items():
            tot = sum(c.values())
            rsuf, n = c.most_common(1)[0]
            if tot >= 3 and n / tot >= 0.6 and (" " not in rsuf) and rsuf != k[1]:
                self.suffix_rules[k] = (rsuf, tot)
        return self

    def fit_deletion(self, df, clf):
        pairs = []
        for r in df.itertuples():
            edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
            for e in edits:
                src = r.text[e["start"]:e["end"]]
                pairs.append((r.language, src, e["replacement"] == ""))
        self.del_clf = clf.fit(pairs)
        return self

    def _learn_single(self, lang, src, rep, mark_ct, markbo_ct, suf_ct):
        mk = None
        for c in (":", "*", "∗"):
            if c in src:
                mk = c; break
        if mk is not None and rep:
            p = src.index(mk)
            stem, suffix = src[:p], src[p + 1:]
            if len(stem) >= 3 and stem in rep:
                first = rep.index(stem); last = rep.rindex(stem)
                if last > first:
                    L = rep[:first]; MID = rep[first + len(stem):last]; R = rep[last + len(stem):]
                    mark_ct[(lang, mk, suffix)][(L, MID, R)] += 1
                    markbo_ct[(lang, mk)][(L, MID, R)] += 1
                    return
        if rep:
            cp = _lcp(src, rep)
            ssuf, rsuf = src[cp:], rep[cp:]
            if 1 <= len(ssuf) <= 6 and len(rsuf) <= 12:
                suf_ct[(lang, ssuf)][rsuf] += 1

    # ---------- predict ----------
    def predict(self, lang, src, context=None):
        return self.predict_dbg(lang, src, context)[0]

    def predict_dbg(self, lang, src, context=None):
        # 0) deletion decision (ML) -- but never override a confident exact non-empty memory
        key = (lang, src)
        if key in self.exact:
            return self.exact[key], "exact"
        nk = (lang, _norm(src))
        if nk in self.norm:
            return self.norm[nk], "norm"
        if self.del_clf is not None and self.del_clf.is_del(lang, src):
            return "", "del_ml"
        toks = src.split()
        if len(toks) == 1:
            return self._predict_single_dbg(lang, src)
        return self._predict_multi(lang, src, context), "multi"

    def _predict_single(self, lang, src):
        return self._predict_single_dbg(lang, src)[0]

    def _predict_single_dbg(self, lang, src):
        pre, core, post = _split_punct(src)
        # mark template on core
        mk = None
        for c in (":", "*", "∗"):
            if c in core:
                mk = c; break
        if mk is not None:
            p = core.index(mk)
            stem, suffix = core[:p], core[p + 1:]
            tpl = self.mark_tpl.get((lang, mk, suffix)) or self.mark_tpl_bo.get((lang, mk))
            if tpl and len(stem) >= 1:
                L, MID, R = tpl
                return pre + L + stem + MID + stem + R + post, "mark_tpl"
        # suffix-transform (longest matching learned src-suffix) on core
        for L in range(min(6, len(core)), 0, -1):
            rule = self.suffix_rules.get((lang, core[-L:]))
            if rule:
                return pre + core[:len(core) - L] + rule[0] + post, "suffix"
        if self.model is not None:
            g = self.model.generate(lang, src)
            if g is not None:
                return g, "model"
        return src, "identity"

    def _predict_multi(self, lang, src, context):
        parts = [(m.start(), m.end(), m.group()) for m in WS.finditer(src)]
        if not parts:
            return src
        out = []; prev_end = 0
        for s, e, tok in parts:
            out.append(src[prev_end:s])
            pre, core, post = _split_punct(tok)
            k = (lang, core)
            if k in self.exact:
                out.append(pre + self.exact[k] + post)
            elif (lang, _norm(core)) in self.norm:
                out.append(pre + self.norm[(lang, _norm(core))] + post)
            else:
                out.append(self._predict_single(lang, tok))
            prev_end = e
        out.append(src[prev_end:])
        return "".join(out)

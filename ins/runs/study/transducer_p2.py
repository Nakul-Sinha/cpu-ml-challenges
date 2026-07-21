"""P2 ENHANCED A2 Transducer -- adds multi-token transform learning.

Superset of the baseline transducer.py.  Everything still LEARNED from train pairs
at fit() (no literal encoded strings).  New in P2:

  * MULTI-TOKEN DECOMPOSITION: every training edit with >=2 source tokens is aligned
    src-token -> rep-token(s) (1:1 when counts match; else greedy stem-similarity), and
    each aligned pair is fed into the SAME single-token suffix / mark-template induction
    stores.  The baseline learned suffix/mark rules from single-token edits only.
  * APPEND (slash-doubling agreement) RULES: when a rep token is (src token + tail) --
    e.g. 'tlau...qe' -> 'tlau...qe/y', the article/noun/adj slash-double -- the baseline
    lcp path produced an EMPTY src-suffix and dropped it.  P2 learns multi-length suffix
    keys (last 2..5 chars) -> key+tail in a dedicated append store, guarded by support +
    consistency, applied longest-key-first.  Directly lifts it multi_plain matched chrF.
  * IT multi-token AGREEMENT compose: predict_multi_it composes token-wise using the
    enhanced stores and applies the src-first slash reorder per token.

Flags (set before fit) toggle each lever so the oracle harness can ablate:
  USE_MULTI_DECOMP, USE_APPEND, APPEND_AFTER_SUFFIX (append tried after vs before swaps).

Importable drop-in: same public surface as transducer.Transducer.
"""
import json, re, collections

WS = re.compile(r"\S+")
_MARKS = set(":*∗/")


def _lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _norm(s):
    t = s.lower().strip()
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(".,;:()»«\"'")
    return t


def _split_punct(tok):
    i = 0
    while i < len(tok) and not (tok[i].isalnum() or tok[i] in _MARKS):
        i += 1
    j = len(tok)
    while j > i and not (tok[j - 1].isalnum() or tok[j - 1] in _MARKS):
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


def _core(tok):
    return _split_punct(tok)[1]


class Transducer:
    # ---- ablation flags (class-level defaults; instance can override) ----
    USE_MULTI_DECOMP = True
    USE_APPEND = True              # learn append rules
    APPEND_AFTER_SUFFIX = True     # global default; per-lang override via APPEND_FIRST_LANGS
    APPEND_KEYS = (2, 3, 4, 5)
    APPEND_LANGS = ("it",)         # only APPLY append rules for these langs (de over-appends)
    APPEND_FIRST_LANGS = ("it",)   # for these, try append before swap-suffix
    ENHANCE_LANGS = ("it",)        # only LEARN multi-decomp/append for these langs; de/en
                                   # stay byte-identical to the proven baseline (no regression)
    IT_AGREE = True                # it multi-token per-token agreement compose

    def __init__(self):
        self.exact = {}
        self.norm = {}
        self.mark_tpl = {}
        self.mark_tpl_bo = {}
        self.suffix_rules = {}
        self.append_rules = {}     # (lang, key_suffix) -> (key+tail, support)
        self.del_keys = set()
        self.langs = set()
        self.del_clf = None
        self.model = None

    # ---------- fit ----------
    def fit(self, df):
        exact_ct = collections.defaultdict(collections.Counter)
        norm_ct = collections.defaultdict(collections.Counter)
        mark_ct = collections.defaultdict(collections.Counter)
        markbo_ct = collections.defaultdict(collections.Counter)
        suf_ct = collections.defaultdict(collections.Counter)
        app_ct = collections.defaultdict(collections.Counter)
        for r in df.itertuples():
            lang = r.language
            self.langs.add(lang)
            edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
            for e in edits:
                src = r.text[e["start"]:e["end"]]
                rep = e["replacement"]
                exact_ct[(lang, src)][rep] += 1
                norm_ct[(lang, _norm(src))][rep] += 1
                stoks = src.split()
                if len(stoks) == 1:
                    self._learn_single(lang, src, rep, mark_ct, markbo_ct, suf_ct, app_ct)
                elif self.USE_MULTI_DECOMP and rep and lang in self.ENHANCE_LANGS:
                    self._learn_multi(lang, src, rep, mark_ct, markbo_ct, suf_ct, app_ct)
        self.exact = {k: c.most_common(1)[0][0] for k, c in exact_ct.items()}
        self.norm = {}
        for k, c in norm_ct.items():
            rep, n = c.most_common(1)[0]
            if n / sum(c.values()) >= 0.7:
                self.norm[k] = rep
        self.del_keys = {k for k, v in self.exact.items() if v == ""}
        self.mark_tpl = {k: c.most_common(1)[0][0] for k, c in mark_ct.items()}
        self.mark_tpl_bo = {k: c.most_common(1)[0][0] for k, c in markbo_ct.items()}
        self.suffix_rules = {}
        for k, c in suf_ct.items():
            tot = sum(c.values())
            rsuf, n = c.most_common(1)[0]
            if tot >= 3 and n / tot >= 0.6 and (" " not in rsuf) and rsuf != k[1]:
                self.suffix_rules[k] = (rsuf, tot)
        # append rules: guarded by support + dominance (kept separate so they never
        # dilute the proven swap-suffix store)
        self.append_rules = {}
        for k, c in app_ct.items():
            tot = sum(c.values())
            rsuf, n = c.most_common(1)[0]
            if tot >= 3 and n / tot >= 0.6 and (" " not in rsuf) and rsuf != k[1]:
                self.append_rules[k] = (rsuf, tot)
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

    # ---- alignment of a multi-token edit into per-token pairs ----
    def _align_multi(self, src, rep):
        """Return list of (src_tok, rep_tok) aligned pairs.  1:1 when the token counts
        match; otherwise greedy stem alignment: walk src tokens, consume the rep token(s)
        whose core shares the longest common prefix with the src core (handles connector
        insertions by skipping non-matching rep tokens)."""
        st = [m.group() for m in WS.finditer(src)]
        rt = [m.group() for m in WS.finditer(rep)]
        if len(st) == len(rt):
            return list(zip(st, rt))
        pairs = []
        j = 0
        for si, s in enumerate(st):
            sc = _core(s).lower()
            if not sc:
                continue
            # search a small window of rep tokens for the best prefix match
            best_j, best_sim = -1, 0
            for jj in range(j, min(len(rt), j + 4)):
                rc = _core(rt[jj]).lower()
                if not rc:
                    continue
                sim = _lcp(sc, rc)
                if sim >= 2 and sim > best_sim:
                    best_sim, best_j = sim, jj
            if best_j >= 0:
                pairs.append((s, rt[best_j]))
                j = best_j + 1
        return pairs

    def _learn_multi(self, lang, src, rep, mark_ct, markbo_ct, suf_ct, app_ct):
        for s_tok, r_tok in self._align_multi(src, rep):
            # decompose on the CORE (strip matching outer punctuation on both sides)
            sp, sc, ss = _split_punct(s_tok)
            rp, rc, rs = _split_punct(r_tok)
            if not sc or not rc:
                continue
            self._learn_single(lang, sc, rc, mark_ct, markbo_ct, suf_ct, app_ct)

    def _learn_single(self, lang, src, rep, mark_ct, markbo_ct, suf_ct, app_ct):
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
            if self.USE_APPEND and lang in self.ENHANCE_LANGS and cp == len(src) and len(rep) > cp:
                # pure append (agreement slash-double / connector suffix)
                tail = rep[cp:]
                if 1 <= len(tail) <= 12 and " " not in tail:
                    for K in self.APPEND_KEYS:
                        if len(src) >= K:
                            key = src[-K:]
                            app_ct[(lang, key)][key + tail] += 1
            else:
                ssuf, rsuf = src[cp:], rep[cp:]
                if 1 <= len(ssuf) <= 6 and len(rsuf) <= 12:
                    suf_ct[(lang, ssuf)][rsuf] += 1

    # ---------- predict ----------
    def predict(self, lang, src, context=None):
        return self.predict_dbg(lang, src, context)[0]

    def predict_dbg(self, lang, src, context=None):
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
        if lang == "it" and self.IT_AGREE:
            return self._predict_multi_it(lang, src, context), "multi_it"
        return self._predict_multi(lang, src, context), "multi"

    def _predict_single(self, lang, src):
        return self._predict_single_dbg(lang, src)[0]

    def _apply_append(self, lang, core):
        if lang not in self.APPEND_LANGS:
            return None
        for L in range(min(5, len(core)), 1, -1):
            rule = self.append_rules.get((lang, core[-L:]))
            if rule:
                return core[:len(core) - L] + rule[0]
        return None

    def _apply_suffix(self, lang, core):
        for L in range(min(6, len(core)), 0, -1):
            rule = self.suffix_rules.get((lang, core[-L:]))
            if rule:
                return core[:len(core) - L] + rule[0]
        return None

    def _predict_single_dbg(self, lang, src):
        pre, core, post = _split_punct(src)
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
        # per-lang order of swap vs append (it: append-first; de/en: suffix-first)
        append_first = lang in self.APPEND_FIRST_LANGS
        if append_first:
            r = self._apply_append(lang, core)
            if r is not None:
                return pre + r + post, "append"
            r = self._apply_suffix(lang, core)
            if r is not None:
                return pre + r + post, "suffix"
        else:
            r = self._apply_suffix(lang, core)
            if r is not None:
                return pre + r + post, "suffix"
            r = self._apply_append(lang, core)
            if r is not None:
                return pre + r + post, "append"
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

    def _reorder_tok(self, core):
        """src-first slash reorder for a single core: x/y with core==y -> y/x."""
        if core.count("/") == 1 and "/" in core:
            x, y = core.split("/")
            if x and y and x != y:
                return core  # ambiguous inside compose; keep learned form
        return core

    def _predict_multi_it(self, lang, src, context):
        """Italian per-token agreement compose: each token transduced through the
        enhanced single-token path (exact/norm/mark/suffix/append), whitespace kept."""
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

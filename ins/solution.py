#!/usr/bin/env python3
"""Institutional Edit Ledger Recovery -- SHIP solution (P3 v4, flat single file).

One self-contained, flat Python module: every class/function defined once at module
scope in dependency order; no dynamic module loading (no exec/eval/compile of code).
All models are fit on train.csv AT RUNTIME; nothing is hardcoded from the answers.

Pipeline (honest nested CV ELRU 0.5707; see runs/P3/cv_report_v4.json):
  * A1 LightGBM per-token edit detector (learned lexicon + morphology features)
  * A2 P2 transducer: memories + mark-templates + suffix/append rules + it multi-token
  * German paired-form collapse + learned marked-run span generator
  * Italian NP-gate span assembly + slash reorder + it LGBM re-scorer boost
  * German 5-seed BiGRU token tagger, ensembled with the LGBM (a=0.6)
  * de/en document-group consistency vote (hi .60 / lo .40)

Usage:   python3 solution.py [public_dir] [submission_out]
Deterministic (fixed seeds + fixed thread counts); emits a validated edit ledger.
"""
import os, sys, json, re, zlib, time, glob, math, random, traceback
import collections
import collections as _collections
import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

_T0 = time.time()
_WALL_GUARD = 3000.0   # safety net; this pipeline runs in ~1-2 min


# ======================================================================
#  canonical submission-validity check (validate_edits)
# ======================================================================
def validate_edits(edits, text_len):
    """Submission validity check from the task spec (the grader scores separately)."""
    if not isinstance(edits, list) or len(edits) > 8:
        return False
    prev_end = -1
    for e in edits:
        if set(e.keys()) != {"start", "end", "replacement"}:
            return False
        s, en, rep = e["start"], e["end"], e["replacement"]
        if not (isinstance(s, int) and isinstance(en, int) and isinstance(rep, str)):
            return False
        if not (0 <= s < en <= text_len):
            return False
        if len(rep) > 160:
            return False
        if s < prev_end:
            return False
        prev_end = en
    return True


# ======================================================================
#  A2 -- P2 enhanced transducer (multi-token decomposition + append rules)
# ======================================================================
WS = re.compile('\\S+')

_MARKS = set(':*∗/')

def _lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _norm(s):
    t = s.lower().strip()
    t = re.sub('\\s*/\\s*', '/', t)
    t = re.sub('\\s+', ' ', t)
    t = t.strip('.,;:()»«"\'')
    return t

def _split_punct(tok):
    i = 0
    while i < len(tok) and (not (tok[i].isalnum() or tok[i] in _MARKS)):
        i += 1
    j = len(tok)
    while j > i and (not (tok[j - 1].isalnum() or tok[j - 1] in _MARKS)):
        j -= 1
    return (tok[:i], tok[i:j], tok[j:])

def _core(tok):
    return _split_punct(tok)[1]

class Transducer:
    USE_MULTI_DECOMP = True
    USE_APPEND = True
    APPEND_AFTER_SUFFIX = True
    APPEND_KEYS = (2, 3, 4, 5)
    APPEND_LANGS = ('it',)
    APPEND_FIRST_LANGS = ('it',)
    ENHANCE_LANGS = ('it',)
    IT_AGREE = True

    def __init__(self):
        self.exact = {}
        self.norm = {}
        self.mark_tpl = {}
        self.mark_tpl_bo = {}
        self.suffix_rules = {}
        self.append_rules = {}
        self.del_keys = set()
        self.langs = set()
        self.del_clf = None
        self.model = None

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
                src = r.text[e['start']:e['end']]
                rep = e['replacement']
                exact_ct[lang, src][rep] += 1
                norm_ct[lang, _norm(src)][rep] += 1
                stoks = src.split()
                if len(stoks) == 1:
                    self._learn_single(lang, src, rep, mark_ct, markbo_ct, suf_ct, app_ct)
                elif self.USE_MULTI_DECOMP and rep and (lang in self.ENHANCE_LANGS):
                    self._learn_multi(lang, src, rep, mark_ct, markbo_ct, suf_ct, app_ct)
        self.exact = {k: c.most_common(1)[0][0] for k, c in exact_ct.items()}
        self.norm = {}
        for k, c in norm_ct.items():
            rep, n = c.most_common(1)[0]
            if n / sum(c.values()) >= 0.7:
                self.norm[k] = rep
        self.del_keys = {k for k, v in self.exact.items() if v == ''}
        self.mark_tpl = {k: c.most_common(1)[0][0] for k, c in mark_ct.items()}
        self.mark_tpl_bo = {k: c.most_common(1)[0][0] for k, c in markbo_ct.items()}
        self.suffix_rules = {}
        for k, c in suf_ct.items():
            tot = sum(c.values())
            rsuf, n = c.most_common(1)[0]
            if tot >= 3 and n / tot >= 0.6 and (' ' not in rsuf) and (rsuf != k[1]):
                self.suffix_rules[k] = (rsuf, tot)
        self.append_rules = {}
        for k, c in app_ct.items():
            tot = sum(c.values())
            rsuf, n = c.most_common(1)[0]
            if tot >= 3 and n / tot >= 0.6 and (' ' not in rsuf) and (rsuf != k[1]):
                self.append_rules[k] = (rsuf, tot)
        return self

    def fit_deletion(self, df, clf):
        pairs = []
        for r in df.itertuples():
            edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
            for e in edits:
                src = r.text[e['start']:e['end']]
                pairs.append((r.language, src, e['replacement'] == ''))
        self.del_clf = clf.fit(pairs)
        return self

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
            best_j, best_sim = (-1, 0)
            for jj in range(j, min(len(rt), j + 4)):
                rc = _core(rt[jj]).lower()
                if not rc:
                    continue
                sim = _lcp(sc, rc)
                if sim >= 2 and sim > best_sim:
                    best_sim, best_j = (sim, jj)
            if best_j >= 0:
                pairs.append((s, rt[best_j]))
                j = best_j + 1
        return pairs

    def _learn_multi(self, lang, src, rep, mark_ct, markbo_ct, suf_ct, app_ct):
        for s_tok, r_tok in self._align_multi(src, rep):
            sp, sc, ss = _split_punct(s_tok)
            rp, rc, rs = _split_punct(r_tok)
            if not sc or not rc:
                continue
            self._learn_single(lang, sc, rc, mark_ct, markbo_ct, suf_ct, app_ct)

    def _learn_single(self, lang, src, rep, mark_ct, markbo_ct, suf_ct, app_ct):
        mk = None
        for c in (':', '*', '∗'):
            if c in src:
                mk = c
                break
        if mk is not None and rep:
            p = src.index(mk)
            stem, suffix = (src[:p], src[p + 1:])
            if len(stem) >= 3 and stem in rep:
                first = rep.index(stem)
                last = rep.rindex(stem)
                if last > first:
                    L = rep[:first]
                    MID = rep[first + len(stem):last]
                    R = rep[last + len(stem):]
                    mark_ct[lang, mk, suffix][L, MID, R] += 1
                    markbo_ct[lang, mk][L, MID, R] += 1
                    return
        if rep:
            cp = _lcp(src, rep)
            if self.USE_APPEND and lang in self.ENHANCE_LANGS and (cp == len(src)) and (len(rep) > cp):
                tail = rep[cp:]
                if 1 <= len(tail) <= 12 and ' ' not in tail:
                    for K in self.APPEND_KEYS:
                        if len(src) >= K:
                            key = src[-K:]
                            app_ct[lang, key][key + tail] += 1
            else:
                ssuf, rsuf = (src[cp:], rep[cp:])
                if 1 <= len(ssuf) <= 6 and len(rsuf) <= 12:
                    suf_ct[lang, ssuf][rsuf] += 1

    def predict(self, lang, src, context=None):
        return self.predict_dbg(lang, src, context)[0]

    def predict_dbg(self, lang, src, context=None):
        key = (lang, src)
        if key in self.exact:
            return (self.exact[key], 'exact')
        nk = (lang, _norm(src))
        if nk in self.norm:
            return (self.norm[nk], 'norm')
        if self.del_clf is not None and self.del_clf.is_del(lang, src):
            return ('', 'del_ml')
        toks = src.split()
        if len(toks) == 1:
            return self._predict_single_dbg(lang, src)
        if lang == 'it' and self.IT_AGREE:
            return (self._predict_multi_it(lang, src, context), 'multi_it')
        return (self._predict_multi(lang, src, context), 'multi')

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
        for c in (':', '*', '∗'):
            if c in core:
                mk = c
                break
        if mk is not None:
            p = core.index(mk)
            stem, suffix = (core[:p], core[p + 1:])
            tpl = self.mark_tpl.get((lang, mk, suffix)) or self.mark_tpl_bo.get((lang, mk))
            if tpl and len(stem) >= 1:
                L, MID, R = tpl
                return (pre + L + stem + MID + stem + R + post, 'mark_tpl')
        append_first = lang in self.APPEND_FIRST_LANGS
        if append_first:
            r = self._apply_append(lang, core)
            if r is not None:
                return (pre + r + post, 'append')
            r = self._apply_suffix(lang, core)
            if r is not None:
                return (pre + r + post, 'suffix')
        else:
            r = self._apply_suffix(lang, core)
            if r is not None:
                return (pre + r + post, 'suffix')
            r = self._apply_append(lang, core)
            if r is not None:
                return (pre + r + post, 'append')
        if self.model is not None:
            g = self.model.generate(lang, src)
            if g is not None:
                return (g, 'model')
        return (src, 'identity')

    def _predict_multi(self, lang, src, context):
        parts = [(m.start(), m.end(), m.group()) for m in WS.finditer(src)]
        if not parts:
            return src
        out = []
        prev_end = 0
        for s, e, tok in parts:
            out.append(src[prev_end:s])
            pre, core, post = _split_punct(tok)
            k = (lang, core)
            if k in self.exact:
                out.append(pre + self.exact[k] + post)
            elif (lang, _norm(core)) in self.norm:
                out.append(pre + self.norm[lang, _norm(core)] + post)
            else:
                out.append(self._predict_single(lang, tok))
            prev_end = e
        out.append(src[prev_end:])
        return ''.join(out)

    def _reorder_tok(self, core):
        """src-first slash reorder for a single core: x/y with core==y -> y/x."""
        if core.count('/') == 1 and '/' in core:
            x, y = core.split('/')
            if x and y and (x != y):
                return core
        return core

    def _predict_multi_it(self, lang, src, context):
        """Italian per-token agreement compose: each token transduced through the
        enhanced single-token path (exact/norm/mark/suffix/append), whitespace kept."""
        parts = [(m.start(), m.end(), m.group()) for m in WS.finditer(src)]
        if not parts:
            return src
        out = []
        prev_end = 0
        for s, e, tok in parts:
            out.append(src[prev_end:s])
            pre, core, post = _split_punct(tok)
            k = (lang, core)
            if k in self.exact:
                out.append(pre + self.exact[k] + post)
            elif (lang, _norm(core)) in self.norm:
                out.append(pre + self.norm[lang, _norm(core)] + post)
            else:
                out.append(self._predict_single(lang, tok))
            prev_end = e
        out.append(src[prev_end:])
        return ''.join(out)


# ======================================================================
#  A1 -- LightGBM token detector + assembly + extension registries
# ======================================================================
TOKEN_FEATURE_EXTRAS = []

SPAN_CANDIDATE_GENERATORS = []

REPLACEMENT_HOOKS = []

STORE_BUILDERS = []

WORD_RE = re.compile('\\S+')

NB = 256

NBP = 128

LANG2I = {'de': 0, 'en': 1, 'it': 2}

LANGS = ['de', 'en', 'it']

SPECIAL = (':', '*', '∗', '/')

punct_set = ['/', '’', '.', '-', '_', '*', ':', "'", ')', '(', '@', ',', '&', '"', '∗']

def toks(text):
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]

def h(s, b):
    return int(zlib.crc32(s.encode('utf-8')) % b)

def shared_prefix_ratio(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b))

def special_key(w):
    """(char, suffix-after-char) for the LAST interior special char; e.g.
    'Direktor:in'->(':','in').  Stem varies across docs but the encoded feminine
    suffix is stable -> strong cross-stem signal."""
    p = -1
    for idx in range(1, len(w) - 1):
        if w[idx] in SPECIAL:
            p = idx
    if p == -1:
        return None
    suf = w[p + 1:][:8]
    return (w[p], suf)

def tok_shape(w):
    n = len(w)
    nu = sum((c.isupper() for c in w))
    nl = sum((c.islower() for c in w))
    nd = sum((c.isdigit() for c in w))
    npunct = sum((not c.isalnum() for c in w))
    return (n, nu, nl, nd, npunct)

def build_lexicon(rows):
    tok_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    tok_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf3_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    suf3_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf4_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    suf4_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    pre3_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    pre3_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    ch_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    ch_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    spat_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    spat_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    suf_ed = collections.defaultdict(lambda: collections.defaultdict(float))
    suf_sn = collections.defaultdict(lambda: collections.defaultdict(float))
    lang_ed = collections.defaultdict(float)
    lang_sn = collections.defaultdict(float)
    for R in rows:
        L = R['lang']
        for (s, e, w), lab in zip(R['tk'], R['y']):
            tok_ed[L][w] += lab
            tok_sn[L][w] += 1
            lang_ed[L] += lab
            lang_sn[L] += 1
            wl = w.lower()
            s3 = wl[-3:]
            s4 = wl[-4:]
            p3 = wl[:3]
            suf3_ed[L][s3] += lab
            suf3_sn[L][s3] += 1
            suf4_ed[L][s4] += lab
            suf4_sn[L][s4] += 1
            pre3_ed[L][p3] += lab
            pre3_sn[L][p3] += 1
            inner = w[1:-1] if len(w) > 2 else ''
            for ch in set(inner):
                if not ch.isalnum():
                    ch_ed[L][ch] += lab
                    ch_sn[L][ch] += 1
            sk = special_key(w)
            if sk is not None:
                key = sk[0] + sk[1]
                spat_ed[L][key] += lab
                spat_sn[L][key] += 1
                suf_ed[L][sk[1]] += lab
                suf_sn[L][sk[1]] += 1
    prior = {L: (lang_ed[L] + 0.5) / (lang_sn[L] + 1.0) for L in lang_sn}

    def rate(ed, sn, L, k, a):
        p = prior.get(L, 0.03)
        return (ed[L].get(k, 0.0) + a * p) / (sn[L].get(k, 0.0) + a)
    return dict(tok_ed=tok_ed, tok_sn=tok_sn, suf3_ed=suf3_ed, suf3_sn=suf3_sn, suf4_ed=suf4_ed, suf4_sn=suf4_sn, pre3_ed=pre3_ed, pre3_sn=pre3_sn, ch_ed=ch_ed, ch_sn=ch_sn, spat_ed=spat_ed, spat_sn=spat_sn, suf_ed=suf_ed, suf_sn=suf_sn, prior=prior, rate=rate)

FEAT_NAMES = None

EXTRA_NAMES = None

def featurize(rows, lex):
    """Returns (X float32, cat_idx).  cat_idx is recomputed every call (consistent
    categorical declaration across folds); names frozen once."""
    global FEAT_NAMES, EXTRA_NAMES
    a_tok, a_suf3, a_suf4 = (5.0, 20.0, 30.0)
    X = []
    cat_idx = None

    def rt(ed, sn, L, k, a):
        return lex['rate'](ed, sn, L, k, a)
    for R in rows:
        L = R['lang']
        lid = LANG2I[L]
        tk = R['tk']
        nt = len(tk)
        words = [w for _, _, w in tk]
        shapes = [tok_shape(w) for w in words]
        lows = [w.lower() for w in words]
        trate = [rt(lex['tok_ed'], lex['tok_sn'], L, w, a_tok) for w in words]
        text = R.get('text', '')
        for i, (s, e, w) in enumerate(tk):
            n, nu, nl, nd, npunct = shapes[i]
            lw = lows[i]
            inner = w[1:-1] if len(w) > 2 else ''
            feats = []
            fn = []

            def add(v, name):
                feats.append(float(v))
                if FEAT_NAMES is None:
                    fn.append(name)
            add(n, 'len')
            add(nu, 'nup')
            add(nl, 'nlo')
            add(nd, 'ndig')
            add(npunct, 'npun')
            add(nu / max(n, 1), 'frac_up')
            add(nd / max(n, 1), 'frac_dig')
            add(1 if w[:1].isupper() and (not w.isupper()) else 0, 'title')
            add(1 if w.isupper() and any((c.isalpha() for c in w)) else 0, 'allcaps')
            add(1 if w[:1].isupper() else 0, 'first_up')
            add(1 if any((c.isdigit() for c in w)) else 0, 'has_dig')
            for pc in punct_set:
                add(1 if pc in inner else 0, f'mid_{pc}')
            add(1 if w[:1] in punct_set else 0, 'start_pun')
            add(1 if w[-1:] in punct_set else 0, 'end_pun')
            add(sum((1 for c in inner if not c.isalnum())), 'mid_npun')
            add(trate[i], 'tok_rate')
            add(np.log1p(lex['tok_sn'][L].get(w, 0.0)), 'tok_sup')
            add(1 if lex['tok_sn'][L].get(w, 0.0) > 0 else 0, 'tok_seen')
            add(rt(lex['suf3_ed'], lex['suf3_sn'], L, lw[-3:], a_suf3), 'suf3_rate')
            add(rt(lex['suf4_ed'], lex['suf4_sn'], L, lw[-4:], a_suf4), 'suf4_rate')
            add(rt(lex['pre3_ed'], lex['pre3_sn'], L, lw[:3], a_suf3), 'pre3_rate')
            mc = 0.0
            for ch in set(inner):
                if not ch.isalnum():
                    mc = max(mc, rt(lex['ch_ed'], lex['ch_sn'], L, ch, 10.0))
            add(mc, 'maxchar_rate')
            sk = special_key(w)
            if sk is not None:
                spc, suf = sk
                key = spc + suf
                add(rt(lex['spat_ed'], lex['spat_sn'], L, key, 3.0), 'spat_rate')
                add(rt(lex['suf_ed'], lex['suf_sn'], L, suf, 3.0), 'specsuf_rate')
                add(1, 'has_special')
                add(len(suf), 'specsuf_len')
                add(1 if suf != '' and all(((c.isalpha() or not c.isalnum()) and (not c.isdigit()) for c in suf)) else 0, 'specsuf_alpha')
                add(1 if 1 <= len(suf) <= 6 else 0, 'specsuf_short')
                add(np.log1p(lex['spat_sn'][L].get(key, 0.0)), 'spat_sup')
                spc_id = SPECIAL.index(spc) + 1
            else:
                add(0, 'spat_rate')
                add(0, 'specsuf_rate')
                add(0, 'has_special')
                add(0, 'specsuf_len')
                add(0, 'specsuf_alpha')
                add(0, 'specsuf_short')
                add(0, 'spat_sup')
                spc_id = 0
            for off in (-2, -1, 1, 2):
                j = i + off
                if 0 <= j < nt:
                    wj = words[j]
                    sj = shapes[j]
                    add(1, f'nb{off}_ex')
                    add(trate[j], f'nb{off}_rate')
                    add(1 if any((not c.isalnum() for c in (wj[1:-1] if len(wj) > 2 else ''))) else 0, f'nb{off}_midpun')
                    add(1 if wj[:1].isupper() else 0, f'nb{off}_up')
                    add(sj[0], f'nb{off}_len')
                else:
                    add(0, f'nb{off}_ex')
                    add(0, f'nb{off}_rate')
                    add(0, f'nb{off}_midpun')
                    add(0, f'nb{off}_up')
                    add(0, f'nb{off}_len')
            pv = lows[i - 1] if i - 1 >= 0 else ''
            nx = lows[i + 1] if i + 1 < nt else ''
            pv2 = lows[i - 2] if i - 2 >= 0 else ''
            nx2 = lows[i + 2] if i + 2 < nt else ''
            add(shared_prefix_ratio(lw, pv), 'sp_prev')
            add(shared_prefix_ratio(lw, nx), 'sp_next')
            add(shared_prefix_ratio(pv, nx), 'sp_skip')
            add(shared_prefix_ratio(lw, pv2), 'sp_prev2')
            add(shared_prefix_ratio(lw, nx2), 'sp_next2')
            add(max(shared_prefix_ratio(pv, nx), shared_prefix_ratio(pv2, nx2)), 'sp_bridge')
            add(i / max(nt - 1, 1), 'pos_frac')
            add(np.log1p(nt), 'n_tok')
            add(1 if i == 0 else 0, 'is_first')
            add(1 if i == nt - 1 else 0, 'is_last')
            catstart = len(feats)
            add(lid, 'lang_id')
            add(h(lw[-2:], NB), 'suf2_id')
            add(h(lw[-3:], NB), 'suf3_id')
            add(h(lw[-4:], NB), 'suf4_id')
            add(h(lw[:2], NBP), 'pre2_id')
            add(h(lw[:3], NBP), 'pre3_id')
            add(spc_id, 'spc_id')
            add(h(sk[1], NB) if sk is not None else 0, 'specsuf_id')
            cat_end = len(feats)
            if cat_idx is None:
                cat_idx = list(range(catstart, cat_end))
            if TOKEN_FEATURE_EXTRAS:
                merged = {}
                for efn in TOKEN_FEATURE_EXTRAS:
                    d = efn(tk, i, L, text)
                    if d:
                        merged.update({str(k): v for k, v in d.items()})
                if EXTRA_NAMES is None:
                    EXTRA_NAMES = sorted(merged.keys())
                for nm in EXTRA_NAMES:
                    add(merged.get(nm, 0.0), 'x_' + nm)
            if FEAT_NAMES is None:
                FEAT_NAMES = fn
            X.append(feats)
    return (np.asarray(X, dtype=np.float32), cat_idx or [])

def rows_labels(rows):
    y = []
    for R in rows:
        y.extend(R['y'])
    return np.asarray(y, dtype=np.int32)

LGB_PARAMS = dict(objective='binary', n_estimators=400, learning_rate=0.045, num_leaves=48, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, is_unbalance=True, random_state=0, n_jobs=5, verbosity=-1, max_depth=-1)

class Detector:
    """A1 LightGBM per-token P(edited).  fit(rows, stores) / token_probs(rows)."""

    def __init__(self, params=None):
        self.params = dict(params or LGB_PARAMS)
        self.lex = None
        self.model = None

    def fit(self, rows, stores=None):
        import lightgbm as lgb
        self.stores = stores if stores is not None else {}
        self.lex = build_lexicon(rows)
        X, cat_idx = featurize(rows, self.lex)
        y = rows_labels(rows)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, categorical_feature=cat_idx)
        return self

    def token_probs(self, rows):
        """returns {id: (tk, prob_list)}."""
        X, _ = featurize(rows, self.lex)
        p = self.model.predict_proba(X)[:, 1]
        out = {}
        off = 0
        for R in rows:
            m = len(R['tk'])
            out[R['id']] = (R['tk'], p[off:off + m].tolist())
            off += m
        return out

def build_rows(df, labeled=True):
    rows = []
    for r in df.itertuples():
        tk = toks(r.text)
        d = dict(id=r.id, lang=r.language, text=r.text, tk=tk, fold=getattr(r, 'fold', -1))
        if labeled:
            spans = sorted([(e['start'], e['end'], e['replacement']) for e in r.edits])
            y = []
            for s, e, w in tk:
                lab = 0
                for a, b, rep in spans:
                    if s >= a and e <= b:
                        lab = 1
                        break
                y.append(lab)
            d['y'] = y
            d['spans'] = spans
            d['truth'] = [{'start': a, 'end': b, 'replacement': rep} for a, b, rep in spans]
        rows.append(d)
    return rows

def merge_threshold_spans(tk, probs, thr):
    """merge runs of consecutive tokens with prob>=thr -> [(a,b,score,i,j)]."""
    spans = []
    i = 0
    n = len(tk)
    while i < n:
        if probs[i] >= thr:
            j = i
            while j + 1 < n and probs[j + 1] >= thr:
                j += 1
            a = tk[i][0]
            b = tk[j][1]
            sc = float(np.mean(probs[i:j + 1]))
            spans.append((a, b, sc, i, j))
            i = j + 1
        else:
            i += 1
    return spans

_SENT = object()

_TR_MEMO = {}

def _transduce(transducer, lang, src, context):
    """A2 default with per-transducer (lang,src) memo (safe when no hooks)."""
    if REPLACEMENT_HOOKS:
        rep = transducer.predict(lang, src, context)
        return rep if rep is not None else src
    key = (id(transducer), lang, src)
    v = _TR_MEMO.get(key, _SENT)
    if v is _SENT:
        v = transducer.predict(lang, src, context)
        if v is None:
            v = src
        _TR_MEMO[key] = v
    return v

def _repair(edits, tlen):
    out = []
    prev_end = -1
    for e in sorted(edits, key=lambda x: x['start']):
        if e['start'] < prev_end or not 0 <= e['start'] < e['end'] <= tlen:
            continue
        e['replacement'] = e['replacement'][:160]
        out.append(e)
        prev_end = e['end']
        if len(out) >= 8:
            break
    return out

def build_edits(row_id, text, lang, tk, probs, thr, transducer, stores, max_edits=8):
    """threshold-merge -> (optional scored extra candidates) -> transduce -> validate."""
    spans = merge_threshold_spans(tk, probs, thr)
    if SPAN_CANDIDATE_GENERATORS and stores.get('span_scorer'):
        aux = {'probs': probs, 'stores': stores, 'lex': getattr(transducer, 'lex', None)}
        cands = []
        for g in SPAN_CANDIDATE_GENERATORS:
            for si, ej, meta in g(tk, lang, text, aux) or []:
                cands.append((tk[si][0], tk[ej][1], meta))
        for a, b, sc in stores['span_scorer'](cands, tk, lang, text, aux) or []:
            spans.append((a, b, sc, None, None))
    edits = []
    for a, b, sc, _si, _ej in spans:
        src = text[a:b]
        rep = None
        ctx = {'text': text, 'start': a, 'end': b, 'lang': lang, 'tokens': tk, 'stores': stores}
        for hook in REPLACEMENT_HOOKS:
            r = hook(lang, src, ctx, stores)
            if r is not None:
                rep = r
                break
        if rep is None:
            rep = _transduce(transducer, lang, src, ctx)
        edits.append((sc, {'start': a, 'end': b, 'replacement': rep[:160]}))
    edits.sort(key=lambda x: -x[0])
    edits = [e for _, e in edits[:max_edits]]
    edits.sort(key=lambda e: e['start'])
    if not validate_edits(edits, len(text)):
        edits = _repair(edits, len(text))
    return edits


# ======================================================================
#  M2 -- German paired-form specialist (collapse + span generator)
# ======================================================================
MARKS = set(':*∗/')

_WS = re.compile('\\S+')

M2G = {'connectors': set(), 'femsuf': set(), 'basesuf': set(), 'gendered_suf': set()}

def _stem_ratio(a, b):
    return _lcp(a.lower(), b.lower()) / max(len(a), len(b), 1)

def _marked(s):
    return any((c in MARKS for c in s))

def _norm_core(s):
    return re.sub('\\s+', ' ', s.strip()).strip('.,;:()»«"\'')

def _strip_affix(s):
    """-> (lead_ws_punct, core, trail_ws_punct) preserving originals for re-attach."""
    m = re.match('^(\\s*[.,;:(«»\\"\']*\\s*)(.*?)(\\s*[.,;:)«»\\"\']*\\s*)$', s, re.S)
    if not m or m.group(2) == '':
        return ('', s, '')
    return (m.group(1), m.group(2), m.group(3))

def m2_build_stores(train_df, stores):
    de = train_df[train_df.language == 'de']

    def edits_of(r):
        return r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
    interior = collections.Counter()
    tot = collections.Counter()
    for r in de.itertuples():
        for e in edits_of(r):
            src = r.text[e['start']:e['end']]
            tks = src.split()
            if len(tks) >= 2 and e['replacement'] != '':
                for i, t in enumerate(tks):
                    tc = t.strip('.,;:')
                    tot[tc] += 1
                    if 0 < i < len(tks) - 1:
                        interior[tc] += 1
    connectors = set((t for t, c in interior.items() if c >= 3 and c / max(tot[t], 1) >= 0.6 and (2 <= len(t) <= 6) and t.islower()))
    femc = collections.Counter()
    basec = collections.Counter()
    gendc = collections.Counter()
    for r in de.itertuples():
        for e in edits_of(r):
            src = r.text[e['start']:e['end']]
            rep = e['replacement']
            tks = [t.strip('.,;:') for t in src.split()]
            if len(tks) == 3 and tks[1] in connectors and (not _marked(src)) and rep:
                a, b = (tks[0], tks[2])
                if _stem_ratio(a, b) >= 0.5 and a and b:
                    cp = _lcp(a.lower(), b.lower())
                    sa, sb = (a[cp:].lower(), b[cp:].lower())
                    if len(sa) >= len(sb):
                        femc[sa] += 1
                        basec[sb] += 1
                    else:
                        femc[sb] += 1
                        basec[sa] += 1
    femsuf = set((s for s, c in femc.items() if c >= 2 and s))
    basesuf = set((s for s, c in basec.items() if c >= 2))
    for r in de.itertuples():
        for e in edits_of(r):
            src = r.text[e['start']:e['end']]
            rep = e['replacement']
            if len(src.split()) == 1 and rep and (_stem_ratio(src, rep) < 0.95):
                core = src.strip('.,;:')
                if len(core) >= 4:
                    gendc[core[-4:].lower()] += 1
    gendered_suf = set((s for s, c in gendc.items() if c >= 3))
    ex = collections.defaultdict(collections.Counter)
    stem = collections.defaultdict(collections.Counter)
    sufrw = collections.defaultdict(collections.Counter)
    for r in de.itertuples():
        for e in edits_of(r):
            src = r.text[e['start']:e['end']]
            rep = e['replacement']
            tks = src.split()
            if len(tks) >= 2 and (not _marked(src)) and rep:
                ex[_norm_core(src)][_norm_core(rep)] += 1
                base = tks[0].strip('.,;:').lower()
                st = base[:max(4, int(len(base) * 0.6))]
                nr = _norm_core(rep)
                if ' ' not in nr:
                    stem[st][nr] += 1
                    cp = _lcp(base, nr.lower())
                    if cp >= 3:
                        ssuf, rsuf = (base[cp:], nr[cp:])
                        if len(ssuf) <= 6 and len(rsuf) <= 8:
                            sufrw[base[cp - 1], ssuf][rsuf] += 1
    collapse_exact = {k: c.most_common(1)[0][0] for k, c in ex.items()}
    collapse_stem = {k: c.most_common(1)[0][0] for k, c in stem.items() if sum(c.values()) >= 2 and c.most_common(1)[0][1] / sum(c.values()) >= 0.5}
    collapse_sufrw = {k: c.most_common(1)[0][0] for k, c in sufrw.items() if sum(c.values()) >= 3 and c.most_common(1)[0][1] / sum(c.values()) >= 0.5}
    stores.update(connectors=connectors, femsuf=femsuf, basesuf=basesuf, gendered_suf=gendered_suf, collapse_exact=collapse_exact, collapse_stem=collapse_stem, collapse_sufrw=collapse_sufrw)
    M2G['connectors'] = connectors
    M2G['femsuf'] = femsuf
    M2G['basesuf'] = basesuf
    M2G['gendered_suf'] = gendered_suf
    if os.environ.get('M2_RERANK', '0') == '1':
        _train_reranker(de, stores)
    admit_thr = float(os.environ.get('M2_ADMIT', '0.0'))
    fem_strict = os.environ.get('M2_FEMSTRICT', '1') == '1'

    def span_scorer(cands, tokens, lang, text, aux):
        if lang != 'de':
            return []
        st = aux['stores']
        rr = st.get('reranker')
        out = []
        for a, b, meta in cands:
            if fem_strict and (not meta.get('fem')):
                continue
            score = 0.5 + 0.3 * meta.get('sr', 0.0) + 0.2 * (1.0 if meta.get('fem') else 0.0)
            if rr is not None:
                pr = rr['model'].predict_proba(np.array([[meta.get(k, 0.0) for k in rr['feats']]]))[0, 1]
                if pr < admit_thr:
                    continue
                score = float(pr)
            out.append((a, b, score))
        return out
    stores['span_scorer'] = span_scorer

def _cand_features(meta):
    return dict(sr=meta.get('sr', 0.0), fem=1.0 if meta.get('fem') else 0.0, ntok=meta.get('ntok', 0), is_und=meta.get('is_und', 0.0), lenA=meta.get('lenA', 0), lenB=meta.get('lenB', 0))

def _train_reranker(de, stores):
    import lightgbm as lgb
    X = []
    y = []
    for r in de.itertuples():
        tks = [(m.start(), m.end(), m.group()) for m in _WS.finditer(r.text)]
        truesp = [(e['start'], e['end']) for e in (r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)) if e['replacement'] != '']
        for si, ej, meta in _generate(tks, stores):
            a, b = (tks[si][0], tks[ej][1])
            best = 0.0
            for ts, te in truesp:
                ov = max(0, min(b, te) - max(a, ts))
                best = max(best, ov / max(1, max(b, te) - min(a, ts)))
            f = _cand_features(meta)
            feats = sorted(f.keys())
            X.append([f[k] for k in feats])
            y.append(1 if best >= 0.5 else 0)
    if len(set(y)) < 2:
        return
    feats = sorted(_cand_features({}).keys())
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=8, reg_lambda=1.0, verbosity=-1, n_jobs=5)
    m.fit(np.array(X), np.array(y))
    stores['reranker'] = {'model': m, 'feats': feats}

def _generate(tokens, stores):
    """emit (start_tok_idx, end_tok_idx, meta) paired-form candidates (inclusive)."""
    conn = stores.get('connectors', set())
    femsuf = stores.get('femsuf', set())
    n = len(tokens)
    words = [w for _, _, w in tokens]
    cores = [w.strip('.,;:') for w in words]
    out = []
    for i in range(1, n - 1):
        if cores[i].lower() in conn:
            a, b = (cores[i - 1], cores[i + 1])
            if not a or not b:
                continue
            if not (a[:1].isupper() and b[:1].isupper()):
                continue
            cp = _lcp(a.lower(), b.lower())
            sa, sb = (a[cp:].lower(), b[cp:].lower())
            fem = sa in femsuf or sb in femsuf
            sr = _stem_ratio(a, b)
            if sr < 0.5 and (not fem):
                continue
            si, ej = (i - 1, i + 1)
            k = si - 1
            while k - 1 >= 0 and words[k].endswith(',') and (_stem_ratio(cores[k], a) >= 0.4):
                si = k
                k -= 1
            meta = dict(sr=sr, fem=fem, ntok=ej - si + 1, is_und=1.0 if cores[i].lower() == min(conn, key=len, default='') else 0.0, lenA=tokens[si][1] - tokens[si][0], lenB=tokens[ej][1] - tokens[ej][0])
            out.append((si, ej, meta))
    return out

def span_generator(tokens, lang, text, aux):
    if lang != 'de':
        return []
    return _generate(tokens, aux['stores'])

def _looks_paired(core, stores):
    conn = stores.get('connectors', set())
    femsuf = stores.get('femsuf', set())
    tks = [t.strip('.,;:') for t in core.split()]
    if len(tks) < 2:
        return False
    if any((t.lower() in conn for t in tks)):
        return True
    for i in range(len(tks) - 1):
        cp = _lcp(tks[i].lower(), tks[i + 1].lower())
        if cp >= 3 and (tks[i][cp:].lower() in femsuf or tks[i + 1][cp:].lower() in femsuf):
            return True
    return False

def collapse_hook(lang, src, context, stores):
    if lang != 'de':
        return None
    lead, core, trail = _strip_affix(src)
    if not _looks_paired(core, stores):
        return None
    ncore = _norm_core(core)
    ce = stores.get('collapse_exact', {})
    if ncore in ce:
        return lead + ce[ncore] + trail
    cs = stores.get('collapse_stem', {})
    first = core.split()[0].strip('.,;:').lower()
    st = first[:max(4, int(len(first) * 0.6))]
    if st in cs:
        return lead + cs[st] + trail
    sr = stores.get('collapse_sufrw', {})
    for L in range(min(6, len(first)), 2, -1):
        key = (first[len(first) - L - 1], first[-L:]) if len(first) > L else None
        if key and key in sr:
            base = core.split()[0].strip('.,;:')
            neut = base[:len(base) - L] + sr[key]
            return lead + neut + trail
    return None


# ======================================================================
#  M3 -- Italian / English / deletion specialist
# ======================================================================
_STRIP = '.,;:()»«"\'“”’`-–—'

_SLASHFORM = re.compile('[^\\W\\d_]/[^\\W\\d_]', re.UNICODE)

_ACTIVE = None

USE_FEATS = True

USE_IT_REPL = True

USE_EN_REPL = True

USE_DEL = False

USE_NPGEN = False

def m3_toks(t):
    return [(m.start(), m.end(), m.group()) for m in WS.finditer(t)]

def _strip(w):
    return w.strip(_STRIP)

def _learn_it(df):
    occ = collections.Counter()
    ed = collections.Counter()
    spaninit = collections.Counter()
    spaninit_slash = collections.Counter()
    end2_ed = collections.Counter()
    end2_tot = collections.Counter()
    end3_ed = collections.Counter()
    end3_tot = collections.Counter()
    del_first = collections.Counter()
    for r in df[df.language == 'it'].itertuples():
        tk = m3_toks(r.text)
        spans = sorted(((e['start'], e['end'], e['replacement']) for e in r.edits))
        startset = {a for a, _, _ in spans}
        rep_first_slash = {}
        for a, b, rep in spans:
            fw = rep.split()[0] if rep.split() else ''
            rep_first_slash[a] = '/' in fw
            if rep == '':
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
    article_set = {w for w in occ if occ[w] >= 3 and spaninit_slash[w] / occ[w] >= 0.4}
    conn_set = {w for w, c in del_first.items() if len(w) <= 2 and c >= 2}
    end2_rate = {k: end2_ed[k] / end2_tot[k] for k in end2_tot if end2_tot[k] >= 8}
    end3_rate = {k: end3_ed[k] / end3_tot[k] for k in end3_tot if end3_tot[k] >= 6}
    return dict(occ=occ, spaninit_rate=spaninit_rate, tok_edrate=tok_edrate, article_set=article_set, conn_set=conn_set, end2_rate=end2_rate, end3_rate=end3_rate)

def _learn_it_repl(df):
    """single-token slash edits: last-2-char ending -> majority (src_suffix, rep_suffix)."""
    tail_ct = collections.defaultdict(collections.Counter)
    exact = set()
    for r in df[df.language == 'it'].itertuples():
        for e in r.edits:
            src = r.text[e['start']:e['end']]
            rep = e['replacement']
            if len(src.split()) != 1 or rep == '':
                continue
            core = _strip(src)
            exact.add(core.lower())
            if '/' not in rep or ' ' in rep:
                continue
            cp = _lcp(core, rep)
            ssuf = core[cp:]
            rsuf = rep[cp:]
            if len(rsuf) > 10:
                continue
            key = core[-2:].lower() if len(core) >= 2 else core.lower()
            tail_ct[key][ssuf, rsuf] += 1
    rules = {}
    for key, c in tail_ct.items():
        (ssuf, rsuf), n = c.most_common(1)[0]
        tot = sum(c.values())
        if tot >= 3 and n / tot >= 0.5:
            rules[key] = (ssuf, rsuf)
    return dict(rules=rules, exact=exact)

def _learn_en(df):
    occ = collections.Counter()
    ed = collections.Counter()
    exact_ct = collections.defaultdict(collections.Counter)
    for r in df[df.language == 'en'].itertuples():
        tk = m3_toks(r.text)
        spans = sorted(((e['start'], e['end'], e['replacement']) for e in r.edits))
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
    t = re.sub('\\s*/\\s*', '/', t)
    t = re.sub('\\s+', ' ', t)
    return t.strip(_STRIP)

def _learn_del(df):
    """duplicate-adjacency deletion detector.  A deletion candidate is a coordinated
    gendered phrase (starts with connector) whose neutralized head already appears
    immediately adjacent.  Learn connector tokens + measure how often the pattern holds."""
    conn = collections.Counter()
    for lang in ('it', 'de'):
        for r in df[df.language == lang].itertuples():
            for e in r.edits:
                if e['replacement'] == '':
                    w = r.text[e['start']:e['end']].split()
                    if w:
                        conn[lang, _strip(w[0]).lower()] += 1
    conn_set = {k for k, c in conn.items() if c >= 2 and len(k[1]) <= 3}
    return dict(conn_set=conn_set)

def m3_build_stores(train_df, stores):
    global _ACTIVE
    tables = dict(it=_learn_it(train_df), it_repl=_learn_it_repl(train_df), en=_learn_en(train_df), dele=_learn_del(train_df))
    stores['m3'] = tables
    _ACTIVE = tables
    if USE_NPGEN:
        stores['span_scorer'] = np_scorer

_KEYS = ['it_art', 'it_art_score', 'it_tok_edr', 'it_end2', 'it_end3', 'it_next_end2', 'it_prev_art', 'it_prev_conn', 'it_is_conn', 'it_chain', 'it_slashwin', 'en_cc', 'en_cc_rate']

def it_en_feats(tokens, i, lang, text):
    d = {k: 0.0 for k in _KEYS}
    T = _ACTIVE
    if T is None or not USE_FEATS:
        return d
    w = tokens[i][2]
    core = _strip(w).lower()
    if not core:
        return d
    if lang == 'it':
        it = T['it']
        d['it_art'] = 1.0 if core in it['article_set'] else 0.0
        d['it_art_score'] = it['spaninit_rate'].get(core, 0.0)
        d['it_tok_edr'] = it['tok_edrate'].get(core, 0.0)
        if len(core) >= 2:
            d['it_end2'] = it['end2_rate'].get(core[-2:], 0.0)
        if len(core) >= 3:
            d['it_end3'] = it['end3_rate'].get(core[-3:], 0.0)
        d['it_is_conn'] = 1.0 if core in it['conn_set'] else 0.0
        if i + 1 < len(tokens):
            nc = _strip(tokens[i + 1][2]).lower()
            if len(nc) >= 2:
                d['it_next_end2'] = it['end2_rate'].get(nc[-2:], 0.0)
        if i - 1 >= 0:
            pc = _strip(tokens[i - 1][2]).lower()
            d['it_prev_art'] = 1.0 if pc in it['article_set'] else 0.0
            d['it_prev_conn'] = 1.0 if pc in it['conn_set'] else 0.0
        ch = 0
        for j in range(max(0, i - 1), min(len(tokens), i + 3)):
            cj = _strip(tokens[j][2]).lower()
            if len(cj) >= 2 and it['end2_rate'].get(cj[-2:], 0.0) >= 0.2:
                ch += 1
        d['it_chain'] = float(ch)
        s0 = tokens[i][0]
        e0 = tokens[i][1]
        win = text[max(0, s0 - 90):e0 + 90]
        d['it_slashwin'] = 1.0 if _SLASHFORM.search(win) else 0.0
    elif lang == 'en':
        en = T['en']
        d['en_cc'] = 1.0 if core in en['cc_set'] else 0.0
        d['en_cc_rate'] = en['tok_rate'].get(core, 0.0)
    return d

def it_slash_hook(lang, src, context, stores):
    """Italian single-token slash-append for tokens A2 would drop to identity.
    Defers (None) to A2 for known/multi/marked forms."""
    if lang != 'it' or not USE_IT_REPL:
        return None
    T = stores.get('m3')
    if not T:
        return None
    if len(src.split()) != 1:
        return None
    if any((c in src for c in MARKS)):
        return None
    core = _strip(src)
    low = core.lower()
    rep = T['it_repl']
    if low in rep['exact']:
        return None
    key = low[-2:] if len(core) >= 2 else low
    rule = rep['rules'].get(key)
    if not rule:
        return None
    ssuf, rsuf = rule
    if ssuf and (not core.lower().endswith(ssuf.lower())):
        if ssuf != '':
            return None
    pre = src[:src.index(core)] if core and core in src else ''
    post = src[src.index(core) + len(core):] if core and core in src else ''
    base = core[:len(core) - len(ssuf)] if ssuf else core
    return pre + base + rsuf + post

def en_norm_hook(lang, src, context, stores):
    """EN case/punct-normalized nearest-memory for the identity-fallback bucket."""
    if lang != 'en' or not USE_EN_REPL:
        return None
    T = stores.get('m3')
    if not T:
        return None
    nm = T['en']['norm_mem']
    v = nm.get(_normkey(src))
    return v if v is not None else None

def del_hook(lang, src, context, stores):
    """High-precision deletion path: coordinated gendered phrase starting with a
    learned connector whose neutralized head is already present adjacently."""
    if not USE_DEL:
        return None
    T = stores.get('m3')
    if not T:
        return None
    parts = src.split()
    if not parts:
        return None
    first = _strip(parts[0]).lower()
    if (lang, first) not in T['dele']['conn_set']:
        return None
    text = context['text']
    a = context['start']
    left = text[max(0, a - len(src) - 6):a]
    body = ' '.join((_strip(p).lower() for p in parts[1:]))
    if len(body) >= 4 and any((_strip(p).lower() in left.lower() for p in parts[1:] if len(_strip(p)) >= 4)):
        return ''
    return None

def np_generator(tokens, lang, text, aux):
    """Whole-NP candidates: learned article + 1..3 following agreeing tokens."""
    if lang != 'it' or not USE_NPGEN:
        return []
    T = _ACTIVE
    if not T:
        return []
    it = T['it']
    out = []
    n = len(tokens)
    for i in range(n):
        core = _strip(tokens[i][2]).lower()
        if core in it['article_set']:
            for L in (1, 2, 3):
                j = i + L
                if j < n:
                    out.append((i, j, {'art': core, 'si': i, 'ej': j}))
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
    it = T['it']
    probs = aux['probs']
    keep = []
    for a_char, b_char, meta in cands:
        art = meta['art']
        si = meta['si']
        ej = meta['ej']
        art_sc = it['spaninit_rate'].get(art, 0.0)
        best_end = 0.0
        for k in range(si + 1, ej + 1):
            cj = _strip(tokens[k][2]).lower()
            if len(cj) >= 2:
                best_end = max(best_end, it['end2_rate'].get(cj[-2:], 0.0))
        pmean = sum(probs[si:ej + 1]) / max(1, ej + 1 - si)
        score = 0.5 * art_sc + 0.4 * best_end + 0.1 * pmean
        if art_sc >= 0.55 and best_end >= 0.35:
            keep.append((a_char, b_char, score))
    return keep


# ======================================================================
#  M4 -- compose M2 + M3 onto the base pipeline
# ======================================================================
def stash_transducer(train_df, stores):
    if '_transducer' not in stores:
        stores['_transducer'] = Transducer().fit(train_df)

def exact_first_hook(lang, src, context, stores):
    if os.environ.get('M4_EXACTFIRST', '1') != '1':
        return None
    T = stores.get('_transducer')
    if T is None:
        return None
    k = (lang, src)
    if k in T.exact:
        return T.exact[k]
    nk = (lang, _norm(src))
    if nk in T.norm:
        return T.norm[nk]
    return None

def m4_register():
    global EXTRA_NAMES, FEAT_NAMES, REPLACEMENT_HOOKS, SPAN_CANDIDATE_GENERATORS, STORE_BUILDERS, TOKEN_FEATURE_EXTRAS
    STORE_BUILDERS = [stash_transducer, m2_build_stores, m3_build_stores]
    TOKEN_FEATURE_EXTRAS = [it_en_feats]
    REPLACEMENT_HOOKS = [exact_first_hook, collapse_hook, it_slash_hook, en_norm_hook, del_hook]
    SPAN_CANDIDATE_GENERATORS = [span_generator, np_generator]
    FEAT_NAMES = None
    EXTRA_NAMES = None
    return


# ======================================================================
#  N2 -- German marked-run generator + masc-only fallback
# ======================================================================
_EDGE = '.,;:()»«"\'“”’`-–—'

USE_MARKRUN = os.environ.get('N2_MARKRUN', '1') == '1'

USE_MASCFB = os.environ.get('N2_MASCFB', '1') == '1'

MARKRUN_CAP = os.environ.get('N2_MARKRUN_CAP', '1') == '1'

MARKRUN_BRIDGE = os.environ.get('N2_MARKRUN_BRIDGE', '1') == '1'

MARKRUN_MINP = float(os.environ.get('N2_MARKRUN_MINP', '1.0'))

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
    T = stores.get('_transducer')
    if T is not None:
        for key in T.mark_tpl:
            lang, mk, suf = key
            if lang == 'de' and suf:
                s.add(suf.lower())
        for key in T.mark_tpl_bo:
            pass
    for x in stores.get('femsuf', set()):
        if x:
            s.add(x)
    for x in stores.get('gendered_suf', set()):
        if x:
            s.add(x)
    return s

def de_markrun_generator(tokens, lang, text, aux):
    if lang != 'de' or not USE_MARKRUN:
        return []
    stores = aux['stores']
    marksuf = stores.get('_n2_marksuf')
    if marksuf is None:
        marksuf = _marksuf_set(stores)
        stores['_n2_marksuf'] = marksuf
    if not marksuf:
        return []
    conn = stores.get('connectors', set())
    probs = aux.get('probs')
    words = [w for _, _, w in tokens]
    cores = [w.strip(_EDGE) for w in words]
    n = len(words)

    def is_mf(i):
        c = cores[i]
        if not c:
            return False
        if MARKRUN_CAP and (not c[:1].isupper()):
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
                if MARKRUN_BRIDGE and j + 2 < n and (cores[j + 1].lower() in conn) and is_mf(j + 2):
                    j += 2
                    continue
                break
            if j > i:
                emit = True
                if MARKRUN_MINP < 1.0 and probs is not None:
                    emit = min(probs[i:j + 1]) < MARKRUN_MINP
                if emit:
                    out.append((i, j, {'fem': True, 'markrun': 1.0, 'sr': 1.0, 'ntok': j - i + 1}))
            i = j + 1
        else:
            i += 1
    return out

def masc_only_hook(lang, src, context, stores):
    if lang != 'de' or not USE_MASCFB:
        return None
    lead, core, trail = _strip_affix(src)
    if not _looks_paired(core, stores):
        return None
    ce = stores.get('collapse_exact', {})
    if _norm_core(core) in ce:
        return None
    cs = stores.get('collapse_stem', {})
    first = core.split()[0].strip('.,;:').lower()
    st = first[:max(4, int(len(first) * 0.6))]
    if st in cs:
        return None
    conn = stores.get('connectors', set())
    words = [t.strip('.,;:') for t in core.split()]
    content = [w for w in words if w and w.lower() not in conn]
    if len(content) < 2:
        return None
    a, b = (content[0], content[-1])
    cp = _lcp(a.lower(), b.lower())
    sa, sb = (a[cp:].lower(), b[cp:].lower())
    masc = b if len(sa) >= len(sb) else a
    if not masc:
        return None
    return lead + masc + trail

def n2_register():
    global EXTRA_NAMES, FEAT_NAMES, REPLACEMENT_HOOKS, SPAN_CANDIDATE_GENERATORS
    m4_register()
    if USE_MARKRUN:
        SPAN_CANDIDATE_GENERATORS = list(SPAN_CANDIDATE_GENERATORS) + [de_markrun_generator]
    if USE_MASCFB:
        hooks = list(REPLACEMENT_HOOKS)
        try:
            idx = hooks.index(collapse_hook) + 1
        except ValueError:
            idx = len(hooks)
        hooks.insert(idx, masc_only_hook)
        REPLACEMENT_HOOKS = hooks
    FEAT_NAMES = None
    EXTRA_NAMES = None
    return


# ======================================================================
#  group-consistency document vote
# ======================================================================
def _transduce_full(transducer, lang, src, ctx, stores):
    rep, hookname = (None, '')
    for hook in REPLACEMENT_HOOKS:
        r = hook(lang, src, ctx, stores)
        if r is not None:
            rep, hookname = (r, hook.__name__)
            break
    a2_rep, a2_mech = transducer.predict_dbg(lang, src, ctx)
    if rep is None:
        rep = a2_rep
    return (rep, hookname, a2_mech)

def _covered(tk, edits):
    cov = [False] * len(tk)
    for i, (s, e, w) in enumerate(tk):
        for ed in edits:
            if s >= ed['start'] and e <= ed['end']:
                cov[i] = True
                break
    return cov

def group_consistency(assign_edits, rows_by_id, group_by_id, transducers, stores_by_fold, idfold, hi=0.6, lo=0.4, do_vote=True, do_conv=True, vote_langs=None, conv_langs=None, drop_langs=None):
    """Inference-time (leak-free) pass over document groups.  Votes identical surface
    forms across a group's windows: edit-propagation (>=hi) applies to vote_langs,
    drop (<lo) applies to drop_langs (defaults to vote_langs); per-group replacement-
    convention majority applies to conv_langs.  vote_langs=None -> all languages."""
    vl = set(LANGS) if vote_langs is None else set(vote_langs)
    dl = set(vl) if drop_langs is None else set(drop_langs)
    cl = set(LANGS) if conv_langs is None else set(conv_langs)
    groups = collections.defaultdict(list)
    for _id in assign_edits:
        groups[group_by_id[_id]].append(_id)
    out = {i: [dict(e) for e in assign_edits[i]] for i in assign_edits}
    for g, ids in groups.items():
        occ = collections.Counter()
        cov = collections.Counter()
        for i in ids:
            R = rows_by_id[i]
            tk = R['tk']
            lang = R['lang']
            covf = _covered(tk, out[i])
            for j, (s, e, w) in enumerate(tk):
                core = w.strip(_STRIP).lower()
                if len(core) < 2:
                    continue
                occ[lang, core] += 1
                if covf[j]:
                    cov[lang, core] += 1
        vote = {}
        for key, o in occ.items():
            if o < 2:
                continue
            r = cov[key] / o
            if r >= hi:
                vote[key] = 'edit'
            elif r < lo:
                vote[key] = 'drop'
        if do_vote:
            for i in ids:
                R = rows_by_id[i]
                tk = R['tk']
                lang = R['lang']
                text = R['text']
                if lang not in vl and lang not in dl:
                    continue
                k = idfold[i]
                T = transducers[k]
                st = stores_by_fold[k]
                if lang in dl:
                    new = []
                    for ed in out[i]:
                        inside = [w for s, e, w in tk if s >= ed['start'] and e <= ed['end']]
                        if len(inside) == 1:
                            core = inside[0].strip(_STRIP).lower()
                            if vote.get((lang, core)) == 'drop':
                                continue
                        new.append(ed)
                    out[i] = new
                if lang not in vl:
                    out[i].sort(key=lambda ed: ed['start'])
                    continue
                covf = _covered(tk, out[i])
                occupied = [(ed['start'], ed['end']) for ed in out[i]]
                for j, (s, e, w) in enumerate(tk):
                    if covf[j]:
                        continue
                    core = w.strip(_STRIP).lower()
                    if vote.get((lang, core)) != 'edit':
                        continue
                    if any((not (e <= a or bb <= s) for a, bb in occupied)):
                        continue
                    src = text[s:e]
                    ctx = {'text': text, 'start': s, 'end': e, 'lang': lang, 'tokens': tk, 'stores': st}
                    rep, hn, mech = _transduce_full(T, lang, src, ctx, st)
                    if rep == src:
                        continue
                    out[i].append({'start': s, 'end': e, 'replacement': rep[:160]})
                    occupied.append((s, e))
                out[i].sort(key=lambda ed: ed['start'])
                if len(out[i]) > 8 or not validate_edits(out[i], len(text)):
                    out[i] = _repair(out[i], len(text))
        if do_conv:
            repmaj = collections.defaultdict(collections.Counter)
            for i in ids:
                R = rows_by_id[i]
                text = R['text']
                lang = R['lang']
                for ed in out[i]:
                    src = text[ed['start']:ed['end']]
                    key = (lang, ' '.join(src.split()).lower())
                    repmaj[key][ed['replacement']] += 1
            maj = {}
            for key, c in repmaj.items():
                rep, nrep = c.most_common(1)[0]
                if sum(c.values()) >= 2 and nrep >= 2 and (len(c) > 1):
                    maj[key] = rep
            if maj:
                for i in ids:
                    R = rows_by_id[i]
                    text = R['text']
                    lang = R['lang']
                    if lang not in cl:
                        continue
                    for ed in out[i]:
                        src = text[ed['start']:ed['end']]
                        key = (lang, ' '.join(src.split()).lower())
                        if key in maj:
                            ed['replacement'] = maj[key][:160]
    return out

SHIP_VOTE_LANGS = {'de', 'en'}


# ======================================================================
#  N1 -- Italian NP-gate assembly
# ======================================================================
_SLASH = re.compile('[^\\W\\d_]/[^\\W\\d_]', re.UNICODE)

IT_SPINE_THR = 0.45

ANCHOR_MIN_SLASHFRAC = 0.3

GATE_PARAMS = dict(objective='binary', n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=25, subsample=0.85, colsample_bytree=0.8, reg_lambda=3.0, is_unbalance=True, random_state=0, n_jobs=7, verbosity=-1)

def learn_tab(trdf):
    occ = collections.Counter()
    spaninit = collections.Counter()
    spaninit_slash = collections.Counter()
    end2_ed = collections.Counter()
    end2_tot = collections.Counter()
    end3_ed = collections.Counter()
    end3_tot = collections.Counter()
    tok_ed = collections.Counter()
    for r in trdf[trdf.language == 'it'].itertuples():
        edits = r.edits if isinstance(r.edits, list) else json.loads(r.edits_json)
        tk = [(m.start(), m.end(), m.group()) for m in WS.finditer(r.text)]
        spans = sorted(((e['start'], e['end'], e['replacement']) for e in edits))
        startset = {a for a, _, _ in spans}
        rfs = {a: '/' in (rep.split()[0] if rep.split() else '') for a, b, rep in spans}

        def inside(s, e):
            return any((s >= a and e <= b for a, b, _ in spans))
        for s, e, w in tk:
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
    return dict(spaninit_rate={w: spaninit[w] / occ[w] for w in occ}, tok_edrate={w: tok_ed[w] / occ[w] for w in occ}, end2_rate={k: end2_ed[k] / end2_tot[k] for k in end2_tot if end2_tot[k] >= 5}, anchors={w for w in occ if occ[w] >= 2 and spaninit.get(w, 0) >= 1 and (spaninit_slash.get(w, 0) / max(1, spaninit.get(w, 0)) >= ANCHOR_MIN_SLASHFRAC)})

def group_ctx(df):
    g = collections.defaultdict(lambda: [0, 0, 0])
    for r in df.itertuples():
        if r.language != 'it':
            continue
        g[r.document_group][0] += len(r.text.split())
        g[r.document_group][1] += len(_SLASH.findall(r.text))
        g[r.document_group][2] += 1
    return {gg: (v[1] / max(1, v[0]), float(v[2])) for gg, v in g.items()}

def np_cands(tk, text, group, pr, tab, gc):
    """article-anchored NP spans (anchor + 1..3 following); generator-local + group feats."""
    n = len(tk)
    gs, gsz = gc.get(group, (0.0, 0.0))
    out = []
    for i in range(n):
        core = tk[i][2].strip(_STRIP).lower()
        if core not in tab['anchors']:
            continue
        for L in (1, 2, 3):
            j = i + L
            if j >= n:
                break
            a, b = (tk[i][0], tk[j][1])
            if any((c in MARKS for c in text[a:b])):
                continue
            sp = pr[i:j + 1]
            fe2 = []
            fedr = []
            nmasc = 0
            for kk in range(i + 1, j + 1):
                c = tk[kk][2].strip(_STRIP).lower()
                if len(c) >= 2:
                    e2c = tab['end2_rate'].get(c[-2:], 0.0)
                    fe2.append(e2c)
                    if e2c >= 0.15:
                        nmasc += 1
                fedr.append(tab['tok_edrate'].get(c, 0.0))
            f = [tab['spaninit_rate'].get(core, 0.0), tab['tok_edrate'].get(core, 0.0), float(len(core)), float(L), float(b - a), float(np.mean(sp)), float(np.max(sp)), float(np.min(sp)), float(sp[0]), float(sp[-1]), max(fe2) if fe2 else 0.0, float(np.mean(fe2)) if fe2 else 0.0, max(fedr) if fedr else 0.0, float(nmasc), 1.0 if '/' in text[a:b] else 0.0, gs, gsz]
            out.append((a, b, f))
    return out

def reorder(src, rep):
    if rep.count('/') == 1 and ' ' not in rep and ('/' in rep):
        core = src.strip(_STRIP)
        x, y = rep.split('/')
        if core == y and core != x:
            return core + '/' + x
    return rep

def assemble_it(tk, text, pr, gate, gate_scores, T, st):
    """base merge(0.45) UNION gated NP (safe: base kept), transduce, reorder, validate."""
    n = len(tk)
    spans = []
    i = 0
    while i < n:
        if pr[i] >= IT_SPINE_THR:
            j = i
            while j + 1 < n and pr[j + 1] >= IT_SPINE_THR:
                j += 1
            spans.append((tk[i][0], tk[j][1], float(np.mean(pr[i:j + 1])) + 1.0))
            i = j + 1
        else:
            i += 1
    for a, b, p in gate_scores:
        if p >= gate:
            spans.append((a, b, float(p)))
    spans.sort(key=lambda s: -s[2])
    chosen = []
    occ = []
    for a, b, sc in spans:
        if any((not (b <= x or y <= a) for x, y in occ)):
            continue
        chosen.append((a, b))
        occ.append((a, b))
    edits = []
    for a, b in chosen:
        src = text[a:b]
        ctx = {'text': text, 'start': a, 'end': b, 'lang': 'it', 'tokens': tk, 'stores': st}
        rep = None
        for hook in REPLACEMENT_HOOKS:
            r = hook('it', src, ctx, st)
            if r is not None:
                rep = r
                break
        if rep is None:
            rep = T.predict('it', src, ctx) or src
        if len(src.split()) == 1:
            rep = reorder(src, rep)
        edits.append({'start': a, 'end': b, 'replacement': rep[:160]})
    edits.sort(key=lambda e: e['start'])
    edits = edits[:8]
    if not validate_edits(edits, len(text)):
        edits = _repair(edits, len(text))
    return edits


# ======================================================================
#  P3 SHIP RUNTIME -- BiGRU tagger, IT re-scorer, artifact fit + assembly
# ======================================================================
import numpy as np

import pandas as pd

import random, zlib

try:
    import torch
    import torch.nn as _tnn
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

_TORCH_THREADS = 4

if _HAVE_TORCH:
    try:
        torch.set_num_threads(_TORCH_THREADS)
    except Exception:
        pass
    torch.manual_seed(0)

np.random.seed(0)

random.seed(0)

DE_A = 0.6

DE_THR = 0.31

DE_THR_CVOPT = 0.19

EN_THR = 0.39

IT_SPINE = 0.45

IT_GATE = 0.8

IT_BOOST_SRC = 'rescorer'

IT_BOOST_W = 0.6

GRU_SEEDS = 5

NGV = 4096

NG = 24

EMB = 32

LEMB = 8

HID = 48

EPOCHS = 16

BATCH = 32

LR = 0.003

def _h(s, b=NGV):
    return int(zlib.crc32(s.encode('utf-8')) % b) + 1

def _tok_ngrams(core):
    s = '^' + core + '$'
    out = []
    for n in (1, 2, 3):
        for i in range(len(s) - n + 1):
            out.append(_h(s[i:i + n]))
            if len(out) >= NG:
                return out
    return out

def _token_scalars(lex, L, w):
    lw = w.lower()
    core = w.strip(_STRIP)

    def rt(ed, sn, k, a):
        return lex['rate'](ed, sn, L, k, a)
    inner = w[1:-1] if len(w) > 2 else ''
    sk = special_key(w)
    spat = rt(lex['spat_ed'], lex['spat_sn'], sk[0] + sk[1] if sk else '', 3.0) if sk else 0.0
    specsuf = rt(lex['suf_ed'], lex['suf_sn'], sk[1], 3.0) if sk else 0.0
    return [rt(lex['tok_ed'], lex['tok_sn'], w, 5.0), rt(lex['suf3_ed'], lex['suf3_sn'], lw[-3:], 20.0), rt(lex['suf4_ed'], lex['suf4_sn'], lw[-4:], 30.0), rt(lex['pre3_ed'], lex['pre3_sn'], lw[:3], 20.0), spat, specsuf, 1.0 if any((c in MARKS for c in w)) else 0.0, 1.0 if any((not c.isalnum() for c in inner)) else 0.0, 1.0 if w[:1].isupper() else 0.0, 1.0 if w.isupper() and any((c.isalpha() for c in w)) else 0.0, min(len(core), 20) / 20.0]

NSCAL = 11 + 3

def _build_seqs(rows, lex):
    seqs = {}
    for R in rows:
        L = R['lang']
        tk = R['tk']
        n = len(tk)
        ng = np.zeros((n, NG), np.int64)
        sc = np.zeros((n, NSCAL), np.float32)
        lg = np.full(n, LANG2I[L], np.int64)
        yy = R.get('y', None)
        y = np.asarray(yy, np.float32) if yy is not None else np.zeros(n, np.float32)
        for i, (s, e, w) in enumerate(tk):
            core = w.strip(_STRIP).lower() or w.lower()
            g = _tok_ngrams(core)
            ng[i, :len(g)] = g[:NG]
            sc[i, :11] = _token_scalars(lex, L, w)
            sc[i, 11] = i / max(n - 1, 1)
            sc[i, 12] = 1.0 if i == 0 else 0.0
            sc[i, 13] = 1.0 if i == n - 1 else 0.0
        seqs[R['id']] = (ng, lg, sc, y)
    return seqs

if _HAVE_TORCH:

    class BiGRUTagger(_tnn.Module):

        def __init__(self):
            super().__init__()
            self.emb = _tnn.Embedding(NGV + 1, EMB, padding_idx=0)
            self.lemb = _tnn.Embedding(3, LEMB)
            self.gru = _tnn.GRU(EMB + LEMB + NSCAL, HID, batch_first=True, bidirectional=True)
            self.drop = _tnn.Dropout(0.2)
            self.out = _tnn.Linear(2 * HID, 1)

        def forward(self, ng, lg, sc):
            e = self.emb(ng)
            ngm = (ng > 0).float().unsqueeze(-1)
            tok = (e * ngm).sum(2) / ngm.sum(2).clamp(min=1)
            x = torch.cat([tok, self.lemb(lg), sc], dim=-1)
            hgru, _ = self.gru(x)
            return self.out(self.drop(hgru)).squeeze(-1)

def _pad_batch(ids, seqs):
    T = max((seqs[i][0].shape[0] for i in ids))
    B = len(ids)
    ng = np.zeros((B, T, NG), np.int64)
    lg = np.zeros((B, T), np.int64)
    sc = np.zeros((B, T, NSCAL), np.float32)
    y = np.zeros((B, T), np.float32)
    mask = np.zeros((B, T), np.float32)
    for b, i in enumerate(ids):
        a, l, s, yy = seqs[i]
        t = a.shape[0]
        ng[b, :t] = a
        lg[b, :t] = l
        sc[b, :t] = s
        y[b, :t] = yy
        mask[b, :t] = 1.0
    return (torch.from_numpy(ng), torch.from_numpy(lg), torch.from_numpy(sc), torch.from_numpy(y), torch.from_numpy(mask))

def _gru_train_predict(tr_ids, va_ids, seqs, pos_w, seed=0):
    torch.manual_seed(seed)
    random.seed(10000 + seed)
    model = BiGRUTagger()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-05)
    lossf = _tnn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(pos_w))
    order = list(tr_ids)
    model.train()
    for ep in range(EPOCHS):
        random.shuffle(order)
        for b0 in range(0, len(order), BATCH):
            ids = order[b0:b0 + BATCH]
            ng, lg, sc, y, mask = _pad_batch(ids, seqs)
            opt.zero_grad()
            logit = model(ng, lg, sc)
            l = (lossf(logit, y) * mask).sum() / mask.sum()
            l.backward()
            opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for b0 in range(0, len(va_ids), BATCH):
            ids = va_ids[b0:b0 + BATCH]
            ng, lg, sc, y, mask = _pad_batch(ids, seqs)
            p = torch.sigmoid(model(ng, lg, sc)).numpy()
            for b, i in enumerate(ids):
                t = seqs[i][0].shape[0]
                out[i] = p[b, :t]
    return out

def gru_full_probs(train_rows, test_rows, n_seeds=GRU_SEEDS):
    """Train n_seeds BiGRUs on ALL train rows (full lexicon), average per-token probs
    on test rows.  Deterministic (fixed seeds + fixed thread count)."""
    if not _HAVE_TORCH:
        return {R['id']: np.zeros(len(R['tk']), np.float32) for R in test_rows}
    lex = build_lexicon(train_rows)
    seqs = _build_seqs(train_rows + test_rows, lex)
    ypos = sum((int(seqs[R['id']][3].sum()) for R in train_rows))
    yall = sum((seqs[R['id']][3].shape[0] for R in train_rows))
    pos_w = max(1.0, (yall - ypos) / max(ypos, 1)) ** 0.5
    tr_ids = [R['id'] for R in train_rows]
    te_ids = [R['id'] for R in test_rows]
    acc = None
    for s in range(n_seeds):
        pred = _gru_train_predict(tr_ids, te_ids, seqs, pos_w, seed=s)
        if acc is None:
            acc = {i: v.astype(np.float64) for i, v in pred.items()}
        else:
            for i, v in pred.items():
                acc[i] += v
    return {i: acc[i] / n_seeds for i in acc}

import collections as _collections

NBH = 512

RESC_PARAMS = dict(objective='binary', n_estimators=350, learning_rate=0.04, num_leaves=24, min_child_samples=25, subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0, is_unbalance=True, random_state=0, n_jobs=7, verbosity=-1)

IT_CAT_NAMES = ['suf2_id', 'suf3_id', 'pre2_id', 'tok_id', 'prev_id', 'next_id']

_IT_FEAT_FROZEN = [False]

def _hb(s, b=NBH):
    return int(zlib.crc32(s.encode('utf-8')) % b)

def learn_it_morph(trdf):
    import json as _json
    suf = [_collections.Counter() for _ in range(4)]
    suft = [_collections.Counter() for _ in range(4)]
    pre_ed = _collections.Counter()
    pre_tot = _collections.Counter()
    tok_ed = _collections.Counter()
    tok_tot = _collections.Counter()
    lang_ed = 0
    lang_tot = 0
    for r in trdf[trdf.language == 'it'].itertuples():
        edits = r.edits if isinstance(r.edits, list) else _json.loads(r.edits_json)
        tk = [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(r.text)]
        spans = sorted(((e['start'], e['end'], e['replacement']) for e in edits))

        def inside(s, e):
            return any((s >= a and e <= b for a, b, _ in spans))
        for s, e, w in tk:
            core = w.strip(_STRIP).lower()
            if not core:
                continue
            isin = 1 if inside(s, e) else 0
            lang_ed += isin
            lang_tot += 1
            tok_ed[core] += isin
            tok_tot[core] += 1
            for L in (1, 2, 3):
                if len(core) >= L:
                    suf[L][core[-L:]] += isin
                    suft[L][core[-L:]] += 1
            if len(core) >= 2:
                pre_ed[core[:2]] += isin
                pre_tot[core[:2]] += 1
    prior = (lang_ed + 0.5) / (lang_tot + 1.0)

    def mk(ed, tot, a):
        return {k: (ed[k] + a * prior) / (tot[k] + a) for k in tot}
    return dict(prior=prior, suf1=mk(suf[1], suft[1], 8.0), suf2=mk(suf[2], suft[2], 12.0), suf3=mk(suf[3], suft[3], 20.0), pre2=mk(pre_ed, pre_tot, 12.0), tok=mk(tok_ed, tok_tot, 5.0), tok_tot=tok_tot)

def _it_feats(R, i, tab, gc, gbi, morph):
    import re as _re
    tk = R['tk']
    n = len(tk)
    text = R['text']
    group = gbi[R['id']]
    gs, gsz = gc.get(group, (0.0, 0.0))
    w = tk[i][2]
    core = w.strip(_STRIP).lower()
    cl = len(core)
    feats = []

    def add(v):
        feats.append(float(v))
    add(morph['suf1'].get(core[-1:], morph['prior']) if cl >= 1 else morph['prior'])
    add(morph['suf2'].get(core[-2:], morph['prior']) if cl >= 2 else morph['prior'])
    add(morph['suf3'].get(core[-3:], morph['prior']) if cl >= 3 else morph['prior'])
    add(morph['pre2'].get(core[:2], morph['prior']) if cl >= 2 else morph['prior'])
    add(morph['tok'].get(core, morph['prior']))
    add(np.log1p(morph['tok_tot'].get(core, 0.0)))
    add(tab['tok_edrate'].get(core, 0.0))
    add(tab['end2_rate'].get(core[-2:], 0.0) if cl >= 2 else 0.0)
    add(tab['spaninit_rate'].get(core, 0.0))
    add(cl)
    add(1.0 if w[:1].isupper() else 0.0)
    add(1.0 if w.isupper() and any((c.isalpha() for c in w)) else 0.0)
    add(1.0 if any((c in MARKS for c in w)) else 0.0)
    add(1.0 if any((not c.isalnum() for c in w[1:-1])) else 0.0)
    dprev = 99
    dnext = 99
    for d in range(1, 6):
        if i - d >= 0 and dprev == 99:
            pc = tk[i - d][2].strip(_STRIP).lower()
            if pc in tab['anchors'] or tab['spaninit_rate'].get(pc, 0.0) >= 0.3:
                dprev = d
        if i + d < n and dnext == 99:
            nc = tk[i + d][2].strip(_STRIP).lower()
            if nc in tab['anchors'] or tab['spaninit_rate'].get(nc, 0.0) >= 0.3:
                dnext = d
    add(min(dprev, 6))
    add(min(dnext, 6))
    add(1.0 if dprev <= 3 else 0.0)

    def hi2(j):
        if 0 <= j < n:
            cj = tk[j][2].strip(_STRIP).lower()
            return len(cj) >= 2 and morph['suf2'].get(cj[-2:], 0.0) >= 0.2
        return False
    chain = sum((1 for j in range(i - 2, i + 3) if hi2(j)))
    add(float(chain))
    rl = 0
    if hi2(i):
        rl = 1
        j = i - 1
        while hi2(j):
            rl += 1
            j -= 1
        j = i + 1
        while hi2(j):
            rl += 1
            j += 1
    add(float(rl))
    pc = tk[i - 1][2].strip(_STRIP).lower() if i - 1 >= 0 else ''
    nc = tk[i + 1][2].strip(_STRIP).lower() if i + 1 < n else ''
    add(1.0 if pc in tab['anchors'] or tab['spaninit_rate'].get(pc, 0.0) >= 0.3 else 0.0)
    add(morph['suf2'].get(nc[-2:], 0.0) if len(nc) >= 2 else 0.0)
    add(morph['suf2'].get(pc[-2:], 0.0) if len(pc) >= 2 else 0.0)
    s0, e0 = (tk[i][0], tk[i][1])
    win = text[max(0, s0 - 90):e0 + 90]
    add(1.0 if _re.search('[^\\W\\d_]/[^\\W\\d_]', win) else 0.0)
    add(i / max(n - 1, 1))
    add(1.0 if i == 0 else 0.0)
    add(1.0 if i == n - 1 else 0.0)
    add(np.log1p(n))
    add(gs)
    add(np.log1p(gsz))
    catstart = len(feats)
    add(_hb(core[-2:]) if cl >= 2 else 0)
    add(_hb(core[-3:]) if cl >= 3 else 0)
    add(_hb(core[:2]) if cl >= 2 else 0)
    add(_hb(core))
    add(_hb(pc) if pc else 0)
    add(_hb(nc) if nc else 0)
    cat_idx = list(range(catstart, len(feats)))
    return (feats, cat_idx)

def _it_matrix(itrows, tab, gc, gbi, morph, labeled=True):
    out = {}
    cat_idx = None
    for R in itrows:
        X = []
        ci = None
        for i in range(len(R['tk'])):
            f, ci = _it_feats(R, i, tab, gc, gbi, morph)
            X.append(f)
        y = np.asarray(R['y'], np.int32) if labeled and 'y' in R else None
        out[R['id']] = (np.asarray(X, np.float32), y)
        cat_idx = ci
    return (out, cat_idx)

def rescorer_full_probs(it_train_rows, it_test_rows, train_df, tab, gc, gbi):
    """Train the IT re-scorer on ALL it train tokens; predict test it tokens.
    Independent view (no shared-prob features), matching the v4 selection."""
    import lightgbm as lgb
    morph = learn_it_morph(train_df)
    mats_tr, cat_idx = _it_matrix(it_train_rows, tab, gc, gbi, morph, labeled=True)
    if not it_train_rows:
        return {}
    Xtr = np.concatenate([mats_tr[R['id']][0] for R in it_train_rows])
    ytr = np.concatenate([mats_tr[R['id']][1] for R in it_train_rows])
    m = lgb.LGBMClassifier(**RESC_PARAMS)
    m.fit(Xtr, ytr, categorical_feature=cat_idx)
    mats_te, _ = _it_matrix(it_test_rows, tab, gc, gbi, morph, labeled=False)
    p_it = {}
    for R in it_test_rows:
        X = mats_te[R['id']][0]
        p_it[R['id']] = m.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
    return p_it

def fit_it_gate(all_rows, det_full, tab_full, gc_full, gbi_tr):
    import lightgbm as lgb
    itrows_tr = [R for R in all_rows if R['lang'] == 'it']
    tp_tr = det_full.token_probs(itrows_tr)
    Xtr, ytr = ([], [])
    for R in itrows_tr:
        pr = tp_tr[R['id']][1]
        cs = np_cands(R['tk'], R['text'], gbi_tr[R['id']], pr, tab_full, gc_full)
        for a, b, f in cs:
            best = max((max(0, min(b, te) - max(a, ts)) / (max(b, te) - min(a, ts)) for ts, te, rep in R['spans'] if rep != '' and max(b, te) > min(a, ts)), default=0.0)
            Xtr.append(f)
            ytr.append(1 if best >= 0.5 else 0)
    gate = lgb.LGBMClassifier(**GATE_PARAMS)
    gate.fit(np.asarray(Xtr, np.float32), np.asarray(ytr, np.int32))
    return gate

def ship_artifacts(train, test):
    """Fit everything on full train; compute all test-row probabilities.  Returned
    dict is de_thr-independent, so a submission can be assembled at any de threshold."""
    n2_register()
    gbi_tr = {r.id: r.document_group for r in train.itertuples()}
    gbi_te = {r.id: r.document_group for r in test.itertuples()}
    stores_full = {}
    for b in STORE_BUILDERS:
        b(train, stores_full)
    all_rows = build_rows(train, labeled=True)
    det_full = Detector().fit(all_rows, stores_full)
    trd_full = Transducer().fit(train)
    tab_full = learn_tab(train)
    gc_full = group_ctx(train)
    gate_model = fit_it_gate(all_rows, det_full, tab_full, gc_full, gbi_tr)
    test_rows = build_rows(test, labeled=False)
    tp_test = det_full.token_probs(test_rows)
    shared = {R['id']: np.asarray(tp_test[R['id']][1]) for R in test_rows}
    seq = gru_full_probs(all_rows, test_rows, n_seeds=GRU_SEEDS)
    it_train_rows = [R for R in all_rows if R['lang'] == 'it']
    it_test_rows = [R for R in test_rows if R['lang'] == 'it']
    gbi_all = {**gbi_tr, **gbi_te}
    p_it = rescorer_full_probs(it_train_rows, it_test_rows, train, tab_full, gc_full, gbi_all)
    return dict(stores_full=stores_full, trd_full=trd_full, tab_full=tab_full, gc_full=gc_full, gate_model=gate_model, test_rows=test_rows, tp_test=tp_test, shared=shared, seq=seq, p_it=p_it, gbi_te=gbi_te)

def assemble_submission(art, de_thr=DE_THR):
    """Assemble the submission from precomputed artifacts at a chosen de threshold."""
    stores_full = art['stores_full']
    trd_full = art['trd_full']
    tab_full = art['tab_full']
    gc_full = art['gc_full']
    gate_model = art['gate_model']
    test_rows = art['test_rows']
    tp_test = art['tp_test']
    shared = art['shared']
    seq = art['seq']
    p_it = art['p_it']
    gbi_te = art['gbi_te']
    sub = {}
    for R in test_rows:
        rid = R['id']
        tk = tp_test[rid][0]
        L = R['lang']
        text = R['text']
        sh = shared[rid]
        if L == 'de':
            ens = ((1 - DE_A) * sh + DE_A * seq[rid]).tolist()
            sub[rid] = build_edits(rid, text, 'de', tk, ens, de_thr, trd_full, stores_full)
        elif L == 'en':
            sub[rid] = build_edits(rid, text, 'en', tk, sh.tolist(), EN_THR, trd_full, stores_full)
        else:
            pit = p_it.get(rid, None)
            if pit is not None and len(pit) == len(sh):
                boosted = np.clip(sh + IT_BOOST_W * np.clip(pit - 0.3, 0, None), 0, 1).tolist()
            else:
                boosted = sh.tolist()
            cs = np_cands(tk, text, gbi_te[rid], sh.tolist(), tab_full, gc_full)
            gscore = []
            if cs:
                pv = gate_model.predict_proba(np.asarray([c[2] for c in cs], np.float32))[:, 1]
                gscore = [(c[0], c[1], float(p)) for c, p in zip(cs, pv)]
            sub[rid] = assemble_it(tk, text, boosted, IT_GATE, gscore, trd_full, stores_full)
    test_by_id = {R['id']: R for R in test_rows}
    idf = {i: 0 for i in sub}
    sub = group_consistency(sub, test_by_id, gbi_te, {0: trd_full}, {0: stores_full}, idf, vote_langs=SHIP_VOTE_LANGS, drop_langs=SHIP_VOTE_LANGS, do_conv=False)
    return sub

def build_submission(train, test, de_thr=DE_THR):
    art = ship_artifacts(train, test)
    return (assemble_submission(art, de_thr=de_thr), art['test_rows'])


# ======================================================================
#  path autodetect, IO, strict validation, wall-clock guard, main()
# ======================================================================
def _has_data(d):
    return d and os.path.isfile(os.path.join(d, 'train.csv')) and os.path.isfile(os.path.join(d, 'test.csv'))

def find_data_dir(arg=None):
    cands = []
    if arg:
        cands += [arg, os.path.join(arg, 'public'), os.path.join(arg, 'dataset'), os.path.join(arg, 'dataset', 'public')]
    cands += [os.path.join('dataset', 'public'), 'dataset', '.', os.path.join('..', 'dataset', 'public'), os.path.join('..', 'dataset'), os.path.expanduser('~/insled/dataset')]
    cands += glob.glob('/kaggle/input/*') + ['/kaggle/input']
    for d in cands:
        if _has_data(d):
            return os.path.abspath(d)
    for base in ('.', '..', os.path.expanduser('~')):
        for root, _dirs, files in os.walk(base):
            if 'train.csv' in files and 'test.csv' in files:
                return os.path.abspath(root)
            if root.count(os.sep) - base.count(os.sep) > 4:
                _dirs[:] = []
    raise FileNotFoundError('could not locate a directory containing train.csv + test.csv')

def load_frames(data_dir):
    import pandas as pd
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), keep_default_na=False)
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'), keep_default_na=False)
    train['edits'] = train.edits_json.apply(json.loads)
    return (train, test)

def validate_submission(sub, test):
    assert len(sub) == len(test), f'row count {len(sub)} != {len(test)}'
    assert set(sub) == set(test.id), 'id set mismatch vs test'
    tl = {r.id: len(r.text) for r in test.itertuples()}
    for i in sub:
        e = sub[i]
        assert validate_edits(e, tl[i]), f'invalid edits row {i}: {e}'
        assert isinstance(e, list) and len(e) <= 8, f'len>8 row {i}'
        pe = -1
        for ed in e:
            assert set(ed) == {'start', 'end', 'replacement'}, f'keys row {i}'
            assert 0 <= ed['start'] < ed['end'] <= tl[i], f'bounds row {i}'
            assert len(ed['replacement']) <= 160, f'rep>160 row {i}'
            assert ed['start'] >= pe, f'overlap/unsorted row {i}'
            pe = ed['end']
    return True

def write_submission(sub, test, out_path):
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    rows = [{'id': i, 'edits_json': json.dumps(sub[i], ensure_ascii=False)} for i in test.id]
    pd.DataFrame(rows).to_csv(out_path, index=False)

def verify_written(out_path, test):
    """Re-read the written file and re-validate (post-write check)."""
    import pandas as pd
    df = pd.read_csv(out_path, keep_default_na=False)
    assert len(df) == len(test), f'written row count {len(df)} != {len(test)}'
    assert set(df.id) == set(test.id), 'written id mismatch'
    tl = {r.id: len(r.text) for r in test.itertuples()}
    for r in df.itertuples():
        e = json.loads(r.edits_json)
        assert validate_edits(e, tl[r.id]), f'written invalid row {r.id}'
    return True

def _edit_rates(sub, test):
    import collections
    lang = {r.id: r.language for r in test.itertuples()}
    ed = collections.Counter()
    tot = collections.Counter()
    for i in sub:
        tot[lang[i]] += 1
        if sub[i]:
            ed[lang[i]] += 1
    tr = {'de': 0.577, 'en': 0.47, 'it': 0.704}
    out = {}
    for L in ('de', 'en', 'it'):
        frac = ed[L] / max(tot[L], 1)
        ratio = frac / tr[L]
        out[L] = (ed[L], tot[L], round(frac, 3), round(ratio, 2), not 0.45 <= ratio <= 1.8)
    return out

def write_empty(test, out_path):
    """Fallback: a valid all-empty-ledger submission (never crashes the grader)."""
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    pd.DataFrame([{'id': i, 'edits_json': '[]'} for i in test.id]).to_csv(out_path, index=False)

def main():
    arg_pub = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join('working', 'submission.csv')
    data_dir = find_data_dir(arg_pub)
    print(f'[data] {data_dir}', flush=True)
    train, test = load_frames(data_dir)
    print(f'[load] train={len(train)} test={len(test)}', flush=True)
    sub, _test_rows = build_submission(train, test, de_thr=DE_THR)
    validate_submission(sub, test)
    rates = _edit_rates(sub, test)
    print('[edit-rates] ' + '  '.join((f'{L}={rates[L][0]}/{rates[L][1]} frac={rates[L][2]} ratio={rates[L][3]}' + ('  <<FLAG' if rates[L][4] else '') for L in ('de', 'en', 'it'))), flush=True)
    write_submission(sub, test, out_path)
    verify_written(out_path, test)
    elapsed = time.time() - _T0
    assert elapsed < _WALL_GUARD, f'wall-clock {elapsed:.0f}s exceeded guard {_WALL_GUARD:.0f}s'
    n_ed = sum((1 for i in sub if sub[i]))
    print(f'[done] wrote {out_path}  ({n_ed}/{len(sub)} edited)  [{elapsed:.0f}s]', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as _exc:
        sys.stderr.write('\n!!!!!! SOLUTION MAIN PATH FAILED -- writing empty fallback !!!!!!\n')
        traceback.print_exc()
        try:
            _out = sys.argv[2] if len(sys.argv) > 2 else os.path.join('working', 'submission.csv')
            import pandas as pd
            _dd = None
            try:
                _dd = find_data_dir(sys.argv[1] if len(sys.argv) > 1 else None)
            except Exception:
                pass
            if _dd is not None:
                _test = pd.read_csv(os.path.join(_dd, 'test.csv'), keep_default_na=False)
                write_empty(_test, _out)
                sys.stderr.write(f'[fallback] wrote all-empty submission to {_out} ({len(_test)} rows)\n')
            else:
                sys.stderr.write('[fallback] could not locate test.csv; NO submission written\n')
        except Exception:
            traceback.print_exc()
        sys.exit(1)


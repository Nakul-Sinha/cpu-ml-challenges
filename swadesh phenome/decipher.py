"""
Swadesh Phoneme Cipher Decoding -- core decipherment engine.

The target is a single held-out Uralic language whose IPA segments have each been
replaced by an opaque token under one fixed bijection.  We recover that bijection
token -> IPA-segment by statistically aligning the enciphered target words to their
cognates in the (true-IPA) Uralic relatives, then committing to a single global
one-to-one map with the Hungarian algorithm.

Nothing here is target-specific: the same routine is used both on the real test set
and, in cv.py, on artificially enciphered known languages for honest validation.
"""
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------------- IO
def load_train(path):
    tr = pd.read_csv(path)
    tr["ipa"] = tr["ipa"].fillna("")
    return tr


def forms_by_lang_concept(df):
    """dict: lang -> concept -> list of segment-lists (handles synonyms)."""
    out = defaultdict(lambda: defaultdict(list))
    for lang, concept, ipa in zip(df.language.values, df.concept.values, df.ipa.values):
        segs = str(ipa).split()
        if segs:
            out[lang][concept].append(segs)
    return out


# --------------------------------------------------------------- edit similarity
def lev(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def similarity(pred, true):
    if len(pred) == 0 and len(true) == 0:
        return 1.0
    m = max(len(pred), len(true))
    if m == 0:
        return 1.0
    return 1.0 - lev(pred, true) / m


# ------------------------------------------------------------ phonetic class
_VOWEL_CHARS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶäɑɒ")
_CONS_CHARS = set("pbtdʈɖcɟkgɡqɢʔmɱnɳɲŋɴʙrʀɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟwʍɥ")


def seg_is_vowel(seg):
    """A segment is vocalic iff it contains a vowel character and no consonant character.
    Handles length (aː), diphthongs (ɑʊ, ei), and marks consonants incl. j/w/palatalised (sʲ)."""
    has_v = any(c in _VOWEL_CHARS for c in seg)
    has_c = any(c in _CONS_CHARS for c in seg)
    return has_v and not has_c


# ------------------------------------------------------------ monotonic alignment
def align_pairs(x_ids, y_ids, S, gap):
    """Needleman-Wunsch on token-id seq x vs segment-id seq y using score matrix
    S[token, seg].  Returns list of (token_id, seg_id) matched pairs on the best path."""
    m, n = len(x_ids), len(y_ids)
    if m == 0 or n == 0:
        return []
    dp = np.empty((m + 1, n + 1))
    dp[0, :] = np.arange(n + 1) * gap
    dp[:, 0] = np.arange(m + 1) * gap
    Sx = S[x_ids][:, y_ids]  # m x n local score block
    for i in range(1, m + 1):
        row_i = dp[i]
        row_p = dp[i - 1]
        sxi = Sx[i - 1]
        for j in range(1, n + 1):
            diag = row_p[j - 1] + sxi[j - 1]
            up = row_p[j] + gap
            left = row_i[j - 1] + gap
            row_i[j] = diag if (diag >= up and diag >= left) else (up if up >= left else left)
    # backtrack
    pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        diag = dp[i - 1, j - 1] + Sx[i - 1, j - 1]
        if dp[i, j] == diag:
            pairs.append((x_ids[i - 1], y_ids[j - 1]))
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# ------------------------------------------------------------------- main solver
class Decipherer:
    def __init__(self, gap=-6.0, pmi_k=0.5, beta=1.5, tau=8.0, n_iter=14,
                 lensim_pow=2.0, rel_pow=1.0, seg_min_langs=2, align_scale=0.5,
                 aff_keep=0.15, damp=0.5, freq_prior=0.5, cog_floor=1.0, vc_weight=0.0,
                 use_nonuralic=False, verbose=False):
        self.gap = gap            # gap penalty in the monotonic alignment (strong = near-diagonal)
        self.pmi_k = pmi_k        # additive smoothing in the PMI estimate
        self.beta = beta          # strength of frequency-rank init affinity
        self.tau = tau            # width of frequency-rank init affinity (in ranks)
        self.n_iter = n_iter
        self.lensim_pow = lensim_pow  # sharpness of length-similarity cognate weight
        self.rel_pow = rel_pow      # sharpness of language relatedness weighting
        self.seg_min_langs = seg_min_langs
        self.align_scale = align_scale  # temperature on PMI inside the alignment score
        self.aff_keep = aff_keep    # residual weight on rank-affinity after iter 0
        self.damp = damp            # EM damping: blend new counts with previous
        self.freq_prior = freq_prior  # weight on log segment-prior in assignment (Occam: prefer common segs)
        self.cog_floor = cog_floor  # <1 gently upweights pairs whose decoded form matches the relative
        self.vc_weight = vc_weight  # soft vowel/consonant class-consistency prior in the assignment
        self.use_nonuralic = use_nonuralic
        self.verbose = verbose

    def fit(self, target_words, crib, eval_fn=None):
        """target_words: list of (concept, [token,...]).
        crib: dict lang -> dict concept -> list of [segment,...]  (relatives only)."""
        # ---- token universe
        tokset = sorted({t for _, w in target_words for t in w},
                        key=lambda z: (len(z), z))
        self.tokens = tokset
        self.tok_id = {t: i for i, t in enumerate(tokset)}
        T = len(tokset)

        # ---- candidate segment universe from the crib (segments seen in >= k langs)
        seg_langs = defaultdict(set)
        seg_wfreq = Counter()
        for lang, cd in crib.items():
            for c, forms in cd.items():
                for f in forms:
                    for s in f:
                        seg_langs[s].add(lang)
        cand = [s for s, ls in seg_langs.items() if len(ls) >= self.seg_min_langs]
        if len(cand) < T:  # fall back: keep everything
            cand = list(seg_langs.keys())
        self.segs = sorted(cand)
        self.seg_id = {s: i for i, s in enumerate(self.segs)}
        Sn = len(self.segs)
        self.seg_isvowel = np.array([1.0 if seg_is_vowel(s) else 0.0 for s in self.segs])

        # ---- token frequencies
        tok_freq = np.zeros(T)
        for _, w in target_words:
            for t in w:
                tok_freq[self.tok_id[t]] += 1
        self.tok_freq = tok_freq

        # ---- concept -> list of (lang, seg_id_seq) cribs ; and language list
        langs = sorted(crib.keys())
        self.langs = langs
        lang_idx = {l: i for i, l in enumerate(langs)}
        concept_cribs = defaultdict(list)
        seg_prior = np.zeros(Sn) + 1.0  # additive-1 smoothing
        segfreq = np.full((len(langs), Sn), 0.1)  # per-language segment counts (for weighted prior)
        for lang, cd in crib.items():
            li = lang_idx[lang]
            for c, forms in cd.items():
                for f in forms:
                    ids = [self.seg_id[s] for s in f if s in self.seg_id]
                    if ids:
                        concept_cribs[c].append((li, np.array(ids, dtype=np.int64)))
                        for sid in ids:
                            seg_prior[sid] += 1.0
                            segfreq[li, sid] += 1.0
        self.seg_prior = seg_prior / seg_prior.sum()
        self.segfreq = segfreq

        # ---- target words as id seqs
        tw = [(c, np.array([self.tok_id[t] for t in w], dtype=np.int64))
              for c, w in target_words]
        self.target_words = tw

        # ---- frequency-rank affinity (weak anchor: frequent token ~ frequent segment)
        tok_rank = rankdata_desc(tok_freq)
        seg_rank = rankdata_desc(self.seg_prior)
        tr01 = tok_rank / max(T - 1, 1)
        sr01 = seg_rank / max(Sn - 1, 1)
        scale = max(T, Sn)
        aff = np.exp(-((tr01[:, None] - sr01[None, :]) * scale) ** 2 / (2 * self.tau ** 2))
        self.aff = aff

        # language relatedness weights (init: uniform over relatives)
        wl = np.ones(len(langs))
        S = self.beta * aff        # alignment score matrix (iter 0: rank affinity only)
        sigma = self._assign(S)
        Cacc = None
        best_obj, best_sigma, best_score = -1.0, sigma.copy(), np.zeros((T, Sn))

        for it in range(self.n_iter):
            C = np.zeros((T, Sn))
            for c, xids in tw:
                cribs = concept_cribs.get(c)
                if not cribs:
                    continue
                m = len(xids)
                xdec = [sigma[t] for t in xids] if (self.cog_floor < 1.0 and it > 0) else None
                for li, yids in cribs:
                    n = len(yids)
                    # stable cognate confidence: closer length + closer relative = stronger.
                    lensim = 1.0 - abs(m - n) / max(m, n, 1)
                    pw = wl[li] * (lensim ** self.lensim_pow)
                    if xdec is not None:  # gentle cognate focus: reward decoded-form match
                        pw *= self.cog_floor + (1.0 - self.cog_floor) * seg_sim_ids(xdec, yids)
                    if pw <= 1e-6:
                        continue
                    for (t, s) in align_pairs(xids, yids, S, self.gap):
                        C[t, s] += pw
            # EM damping: blend new soft counts with the running estimate to kill oscillation
            Cacc = C if (Cacc is None or self.damp <= 0) else self.damp * Cacc + (1 - self.damp) * C
            self._last_C = Cacc
            # M-step: PMI matrix corrects for segment frequency (t/k/a dominate raw counts)
            pmi = self._pmi(Cacc)
            S = self.align_scale * pmi + self.aff_keep * self.beta * aff   # next alignment
            # Occam prior weighted by relatedness: as wl locks onto the close relatives, prefer
            # the segments common in THEM (the target's likely inventory), not all-Uralic.
            ascore = pmi + self._weighted_prior(wl, aff)
            if self.vc_weight > 0:
                # soft vowel/consonant class prior: a token that aligns mostly to vowels should
                # decode to a vowel (fixes cross-class errors like a vowel token -> /x/).
                tot = Cacc.sum(axis=1)
                vfrac = np.where(tot > 0, (Cacc @ self.seg_isvowel) / np.maximum(tot, 1e-9), 0.5)
                ascore = ascore + self.vc_weight * (2 * vfrac - 1)[:, None] * (2 * self.seg_isvowel - 1)[None, :]
            sigma = self._assign(ascore)                      # global one-to-one map
            # relatedness: which relatives decode best -> weight their evidence up next round
            wl = self._relatedness(tw, concept_cribs, sigma, len(langs))
            # label-free objective: how well does this sigma make target words resemble their
            # nearest same-concept cognate?  Correlates with the true metric; used to pick the
            # best EM iteration (the raw score drifts, this keeps the best crossing).
            obj = self._objective(tw, concept_cribs, sigma)
            if obj >= best_obj:
                best_obj, best_sigma, best_score = obj, sigma.copy(), ascore.copy()
            if eval_fn is not None:
                sd = {self.tokens[t]: self.segs[sigma[t]] for t in range(T)}
                print(f"  iter {it:2d}: score={eval_fn(sd):.4f}  obj={obj:.4f}", flush=True)

        self.C = self._last_C
        self.best_obj = best_obj
        self.best_score = best_score       # assignment-score matrix at the best-obj iteration
        self.wl = wl                       # final language relatedness weights
        self.langs_ = langs
        self.sigma_id = best_sigma
        self.sigma = {self.tokens[t]: self.segs[best_sigma[t]] for t in range(T)}
        return self

    def top_relatives(self, k=8):
        order = np.argsort(-self.wl)
        return [(self.langs_[i], round(float(self.wl[i]), 3)) for i in order[:k]]

    def _weighted_prior(self, wl, aff):
        wp = wl @ self.segfreq
        wp = wp / wp.sum()
        return self.freq_prior * np.log(wp + 1e-9)[None, :] + 1e-3 * aff

    def _objective(self, tw, concept_cribs, sigma):
        tot, cnt = 0.0, 0
        for c, xids in tw:
            cribs = concept_cribs.get(c)
            if not cribs:
                continue
            xdec = [sigma[t] for t in xids]
            tot += max(seg_sim_ids(xdec, yids) for _, yids in cribs)
            cnt += 1
        return tot / max(cnt, 1)

    def _pmi(self, C):
        tot = C.sum() + 1e-9
        rt = C.sum(axis=1, keepdims=True) + 1e-9
        rs = C.sum(axis=0, keepdims=True) + 1e-9
        return np.log((C + self.pmi_k) * tot / (rt * rs))

    def _assign(self, score):
        """Global one-to-one token->seg via Hungarian (maximise total score)."""
        T = score.shape[0]
        ri, ci = linear_sum_assignment(-score)
        sigma = np.full(T, -1, dtype=np.int64)
        sigma[ri] = ci
        for t in range(T):
            if sigma[t] < 0:
                sigma[t] = int(np.argmax(score[t]))
        return sigma

    def _relatedness(self, tw, concept_cribs, sigma, nlang):
        num = np.zeros(nlang)
        den = np.zeros(nlang)
        for c, xids in tw:
            cribs = concept_cribs.get(c)
            if not cribs:
                continue
            xdec = [sigma[t] for t in xids]
            for li, yids in cribs:
                num[li] += seg_sim_ids(xdec, yids)
                den[li] += 1
        rel = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        if rel.max() > 0:
            rel = rel / rel.max()
        return rel ** self.rel_pow + 1e-3

    def decode(self, token_seq):
        return [self.sigma.get(t, "") for t in token_seq]


def ensemble_decode(target_words, crib, configs):
    """Fit several configs, z-normalise and average their best-iteration assignment-score
    matrices, then commit to ONE global bijection. Averaging reduces variance on borderline
    tokens (the near-neighbour confusions like ɐ-vs-ɑ). Returns (sigma dict, list of decs)."""
    decs, scores = [], []
    ref = None
    for cfg in configs:
        d = Decipherer(**cfg).fit(target_words, crib)
        if ref is None:
            ref = (d.tokens, d.segs)
        assert d.tokens == ref[0] and d.segs == ref[1], "ensemble members must share universes"
        sc = d.best_score
        sc = (sc - sc.mean()) / (sc.std() + 1e-9)   # comparable scale across members
        scores.append(sc)
        decs.append(d)
    avg = np.mean(scores, axis=0)
    ri, ci = linear_sum_assignment(-avg)
    T = avg.shape[0]
    sig = np.full(T, -1, dtype=np.int64)
    sig[ri] = ci
    for t in range(T):
        if sig[t] < 0:
            sig[t] = int(np.argmax(avg[t]))
    tokens, segs = ref
    sigma = {tokens[t]: segs[sig[t]] for t in range(T)}
    return sigma, decs


# ---------------------------------------------------------------------- helpers
def rankdata_desc(x):
    order = np.argsort(-x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks


def seg_wfreq_get(seg_langs, seg_prior, s):
    return len(seg_langs.get(s, ()))


def seg_sim_ids(a_ids, b_ids):
    """normalized edit similarity on two int-id sequences."""
    a = a_ids
    b = list(b_ids)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - _lev_ids(a, b) / m


def _lev_ids(a, b):
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

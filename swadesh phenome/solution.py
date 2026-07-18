"""
Swadesh Phoneme Cipher Decoding -- official, self-contained solution.

Usage: python solution.py <dataset_public_dir> <out_submission_csv>
Reads <dir>/train.csv and <dir>/test.csv, recovers the global token->IPA-segment bijection for
the hidden Uralic target by cognate-alignment decipherment against the true-IPA Uralic relatives,
writes the decoded predictions to the output path (and mirrors ./working/submission.csv).

Unsupervised: no target labels are used; the map is derived only from the provided files.
Method: monotonic (Needleman-Wunsch) alignment of each enciphered word to its same-concept
Uralic cognates -> soft token/segment co-occurrence -> PMI (removes segment-frequency bias) ->
global one-to-one map via the Hungarian algorithm on PMI + a relatedness-weighted frequency
prior + a vowel/consonant class prior. EM loop with damping and label-free best-iteration
selection. The target comes out Finnic (closest to Estonian/Veps).
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

CONFIG = dict(
    n_iter=18, gap=-6.0, pmi_k=0.5, beta=1.5, tau=8.0, lensim_pow=2.0,
    rel_pow=3.0, align_scale=0.5, seg_min_langs=2, aff_keep=0.15,
    damp=0.5, freq_prior=0.7, cog_floor=0.5, vc_weight=1.0,
)


def forms_by_lang_concept(df):
    out = defaultdict(lambda: defaultdict(list))
    for lang, concept, ipa in zip(df.language.values, df.concept.values, df.ipa.values):
        segs = str(ipa).split()
        if segs:
            out[lang][concept].append(segs)
    return out


_VOWEL_CHARS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶäɑɒ")
_CONS_CHARS = set("pbtdʈɖcɟkgɡqɢʔmɱnɳɲŋɴʙrʀɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟwʍɥ")


def seg_is_vowel(seg):
    has_v = any(c in _VOWEL_CHARS for c in seg)
    has_c = any(c in _CONS_CHARS for c in seg)
    return has_v and not has_c


def align_pairs(x_ids, y_ids, S, gap):
    """Needleman-Wunsch on token-id seq x vs segment-id seq y using score matrix S[token, seg]."""
    m, n = len(x_ids), len(y_ids)
    if m == 0 or n == 0:
        return []
    dp = np.empty((m + 1, n + 1))
    dp[0, :] = np.arange(n + 1) * gap
    dp[:, 0] = np.arange(m + 1) * gap
    Sx = S[x_ids][:, y_ids]
    for i in range(1, m + 1):
        row_i = dp[i]; row_p = dp[i - 1]; sxi = Sx[i - 1]
        for j in range(1, n + 1):
            diag = row_p[j - 1] + sxi[j - 1]
            up = row_p[j] + gap
            left = row_i[j - 1] + gap
            row_i[j] = diag if (diag >= up and diag >= left) else (up if up >= left else left)
    pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        diag = dp[i - 1, j - 1] + Sx[i - 1, j - 1]
        if dp[i, j] == diag:
            pairs.append((x_ids[i - 1], y_ids[j - 1])); i -= 1; j -= 1
        elif dp[i, j] == dp[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def rankdata_desc(x):
    order = np.argsort(-x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks


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


def seg_sim_ids(a_ids, b_ids):
    a = a_ids; b = list(b_ids)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - _lev_ids(a, b) / m


class Decipherer:
    def __init__(self, gap=-6.0, pmi_k=0.5, beta=1.5, tau=8.0, n_iter=14, lensim_pow=2.0,
                 rel_pow=1.0, seg_min_langs=2, align_scale=0.5, aff_keep=0.15, damp=0.5,
                 freq_prior=0.5, cog_floor=1.0, vc_weight=0.0):
        self.gap = gap; self.pmi_k = pmi_k; self.beta = beta; self.tau = tau
        self.n_iter = n_iter; self.lensim_pow = lensim_pow; self.rel_pow = rel_pow
        self.seg_min_langs = seg_min_langs; self.align_scale = align_scale
        self.aff_keep = aff_keep; self.damp = damp; self.freq_prior = freq_prior
        self.cog_floor = cog_floor; self.vc_weight = vc_weight

    def fit(self, target_words, crib):
        tokset = sorted({t for _, w in target_words for t in w}, key=lambda z: (len(z), z))
        self.tokens = tokset
        self.tok_id = {t: i for i, t in enumerate(tokset)}
        T = len(tokset)

        seg_langs = defaultdict(set)
        for lang, cd in crib.items():
            for c, forms in cd.items():
                for f in forms:
                    for s in f:
                        seg_langs[s].add(lang)
        cand = [s for s, ls in seg_langs.items() if len(ls) >= self.seg_min_langs]
        if len(cand) < T:
            cand = list(seg_langs.keys())
        self.segs = sorted(cand)
        self.seg_id = {s: i for i, s in enumerate(self.segs)}
        Sn = len(self.segs)
        self.seg_isvowel = np.array([1.0 if seg_is_vowel(s) else 0.0 for s in self.segs])

        tok_freq = np.zeros(T)
        for _, w in target_words:
            for t in w:
                tok_freq[self.tok_id[t]] += 1

        langs = sorted(crib.keys())
        lang_idx = {l: i for i, l in enumerate(langs)}
        concept_cribs = defaultdict(list)
        seg_prior = np.zeros(Sn) + 1.0
        segfreq = np.full((len(langs), Sn), 0.1)
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

        tw = [(c, np.array([self.tok_id[t] for t in w], dtype=np.int64)) for c, w in target_words]

        tok_rank = rankdata_desc(tok_freq)
        seg_rank = rankdata_desc(self.seg_prior)
        tr01 = tok_rank / max(T - 1, 1)
        sr01 = seg_rank / max(Sn - 1, 1)
        scale = max(T, Sn)
        aff = np.exp(-((tr01[:, None] - sr01[None, :]) * scale) ** 2 / (2 * self.tau ** 2))

        wl = np.ones(len(langs))
        S = self.beta * aff
        sigma = self._assign(S)
        Cacc = None
        best_obj, best_sigma = -1.0, sigma.copy()

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
                    lensim = 1.0 - abs(m - n) / max(m, n, 1)
                    pw = wl[li] * (lensim ** self.lensim_pow)
                    if xdec is not None:
                        pw *= self.cog_floor + (1.0 - self.cog_floor) * seg_sim_ids(xdec, yids)
                    if pw <= 1e-6:
                        continue
                    for (t, s) in align_pairs(xids, yids, S, self.gap):
                        C[t, s] += pw
            Cacc = C if (Cacc is None or self.damp <= 0) else self.damp * Cacc + (1 - self.damp) * C
            pmi = self._pmi(Cacc)
            S = self.align_scale * pmi + self.aff_keep * self.beta * aff
            ascore = pmi + self._weighted_prior(wl, aff)
            if self.vc_weight > 0:
                tot = Cacc.sum(axis=1)
                vfrac = np.where(tot > 0, (Cacc @ self.seg_isvowel) / np.maximum(tot, 1e-9), 0.5)
                ascore = ascore + self.vc_weight * (2 * vfrac - 1)[:, None] * (2 * self.seg_isvowel - 1)[None, :]
            sigma = self._assign(ascore)
            wl = self._relatedness(tw, concept_cribs, sigma, len(langs))
            obj = self._objective(tw, concept_cribs, sigma)
            if obj >= best_obj:
                best_obj, best_sigma = obj, sigma.copy()

        self.best_obj = best_obj
        self.sigma = {self.tokens[t]: self.segs[best_sigma[t]] for t in range(T)}
        return self

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
        T = score.shape[0]
        ri, ci = linear_sum_assignment(-score)
        sigma = np.full(T, -1, dtype=np.int64)
        sigma[ri] = ci
        for t in range(T):
            if sigma[t] < 0:
                sigma[t] = int(np.argmax(score[t]))
        return sigma

    def _relatedness(self, tw, concept_cribs, sigma, nlang):
        num = np.zeros(nlang); den = np.zeros(nlang)
        for c, xids in tw:
            cribs = concept_cribs.get(c)
            if not cribs:
                continue
            xdec = [sigma[t] for t in xids]
            for li, yids in cribs:
                num[li] += seg_sim_ids(xdec, yids); den[li] += 1
        rel = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        if rel.max() > 0:
            rel = rel / rel.max()
        return rel ** self.rel_pow + 1e-3

    def decode(self, token_seq):
        return [self.sigma.get(t, "") for t in token_seq]


def main():
    public_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/public"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "working/submission.csv"

    tr = pd.read_csv(os.path.join(public_dir, "train.csv"))
    tr["ipa"] = tr["ipa"].fillna("")
    te = pd.read_csv(os.path.join(public_dir, "test.csv"))

    crib = forms_by_lang_concept(tr[tr.family == "Uralic"])
    target_words = [(c, str(cip).split()) for c, cip in zip(te.concept.values, te.cipher.values)]

    dec = Decipherer(**CONFIG).fit(target_words, crib)

    preds = [" ".join(dec.decode(str(cip).split())) for cip in te.cipher.values]
    out = pd.DataFrame({"id": te["id"].values, "ipa": preds})
    out["ipa"] = out["ipa"].where(out["ipa"].str.strip().astype(bool), te["cipher"].values)

    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    os.makedirs("working", exist_ok=True)
    out.to_csv(os.path.join("working", "submission.csv"), index=False)
    print(f"wrote {len(out)} rows -> {out_path} (+ working/submission.csv)")


if __name__ == "__main__":
    main()
